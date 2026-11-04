"""Source-dataset-stratified record-level bootstrap for frozen-score evaluation.

The released Step 1 artifacts contain record-level model outputs but no
patient-, study-, slide-, video-, or volume-level identifiers.  This module
therefore does *not* claim a patient-level confidence interval.  It draws
records with replacement within each source dataset, creates a fresh 50:50
calibration/evaluation partition for each bootstrap replicate, and reruns the
complete frozen-score decision procedure:

    router LEC calibration -> routing -> clinic-specific LEC calibration
    -> held-out system evaluation.

The 500 historical random splits are deliberately not bootstrap units.  They
are used only once to provide the reference point estimate reported by the
existing analysis.  Each outer bootstrap replicate is one independently drawn
record-level calibration/evaluation experiment.

The module reports only the prespecified primary quantities:
  * proposed-system end-to-end mFDR and AI-managed coverage;
  * specialist-only mFDR and coverage as the paired reference; and
  * paired differences (proposed minus specialist-only) for both quantities.

Usage
-----
python -m pipeline.metrics.record_bootstrap --beta 0.01 --alpha 0.2
python -m pipeline.metrics.plot_record_bootstrap \
    --bootstrap-json result/metrics/record_bootstrap_beta_0.01_alpha_0.20.json
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.metrics.compute import _lec_calibrate_np
from pipeline.metrics.config import NUM_DISEASES


DEFAULT_BETA = 0.01
DEFAULT_ALPHA = 0.20
DEFAULT_REPLICATES = 1000
DEFAULT_REFERENCE_SPLITS = 500
DEFAULT_SEED = 20260904
DEFAULT_SPLIT_RATIO = 0.50
DEFAULT_LAMBDA_1 = 0.50
DEFAULT_LAMBDA_2 = 0.50


@dataclass(frozen=True)
class FrozenScores:
    """All fixed model outputs needed to replay the two-stage workflow."""

    router_uncertainty: np.ndarray
    router_correctness: np.ndarray
    predicted_disease: np.ndarray
    true_disease: np.ndarray
    specialist_uncertainty: np.ndarray
    system_correctness: np.ndarray
    source_names: tuple[str, ...]
    source_indices: dict[str, np.ndarray]

    @property
    def n_records(self) -> int:
        return int(self.router_uncertainty.size)


def _step1_dir(input_root: Path) -> Path:
    """Accept either ``result/json`` or the ``result/json/step1`` directory."""
    input_root = Path(input_root)
    return input_root if (input_root / "phase_1.json").is_file() else input_root / "step1"


def _load_frozen_scores(input_root: Path, lambda_1: float,
                        lambda_2: float) -> FrozenScores:
    """Load and validate the immutable Step 1 router and specialist scores."""
    step1 = _step1_dir(input_root)
    phase1_path = step1 / "phase_1.json"
    if not phase1_path.is_file():
        raise FileNotFoundError(f"Step 1 router file not found: {phase1_path}")

    with open(phase1_path, encoding="utf-8") as f:
        phase1 = json.load(f)

    # Keep each source's JSON order.  It reproduces the historical Stage 2
    # reference split generation when combined with random.Random(epoch).
    source_rows: dict[str, list[int]] = defaultdict(list)
    u1, router_correct, pred, truth, u2, spec_correct = [], [], [], [], [], []

    phase2_cache: dict[str, dict[str, Any]] = {}
    for row_idx, (sample_id, p1) in enumerate(phase1.items()):
        required_p1 = {
            "uncertainty", "correctness", "predicted_disease",
            "true_disease", "source_dataset",
        }
        missing = required_p1.difference(p1)
        if missing:
            raise ValueError(f"{sample_id} is missing phase_1 fields: {sorted(missing)}")

        source = p1["source_dataset"]
        if source not in phase2_cache:
            phase2_path = step1 / f"phase_2_{source}.json"
            if not phase2_path.is_file():
                raise FileNotFoundError(f"Step 1 specialist file not found: {phase2_path}")
            with open(phase2_path, encoding="utf-8") as f:
                phase2_cache[source] = json.load(f)

        p2 = phase2_cache[source].get(sample_id)
        if p2 is None:
            raise ValueError(f"{sample_id} is absent from phase_2_{source}.json")
        if not {"uncertainty", "correctness"}.issubset(p2):
            raise ValueError(f"{sample_id} is missing phase_2 uncertainty/correctness")

        pred_k = int(p1["predicted_disease"])
        true_k = int(p1["true_disease"])
        if not 0 <= pred_k < NUM_DISEASES:
            raise ValueError(f"{sample_id} has invalid predicted_disease={pred_k}")

        # Stage 3 calibrates this value, whereas Stage 5 defines end-to-end
        # correctness from the router label and the frozen specialist result.
        route_ok = int(p1["correctness"])
        implied_route_ok = int(pred_k == true_k)
        if route_ok != implied_route_ok:
            raise ValueError(
                f"{sample_id} has inconsistent router correctness: "
                f"stored={route_ok}, predicted_disease==true_disease={implied_route_ok}"
            )

        u1.append(float(p1["uncertainty"]))
        router_correct.append(route_ok)
        pred.append(pred_k)
        truth.append(true_k)
        u2.append(float(p2["uncertainty"]))
        spec_correct.append(int(p2["correctness"]))
        source_rows[source].append(row_idx)

    source_names = tuple(sorted(source_rows))
    source_indices = {
        source: np.asarray(source_rows[source], dtype=np.int32)
        for source in source_names
    }
    u1_arr = np.asarray(u1, dtype=np.float64)
    u2_arr = np.asarray(u2, dtype=np.float64)
    router_correct_arr = np.asarray(router_correct, dtype=np.int8)
    return FrozenScores(
        router_uncertainty=u1_arr,
        router_correctness=router_correct_arr,
        predicted_disease=np.asarray(pred, dtype=np.int8),
        true_disease=np.asarray(truth, dtype=np.int8),
        specialist_uncertainty=lambda_1 * u1_arr + lambda_2 * u2_arr,
        system_correctness=(router_correct_arr & np.asarray(spec_correct, dtype=np.int8)),
        source_names=source_names,
        source_indices=source_indices,
    )


def _ratio(num: int, den: int) -> float | None:
    return float(num / den) if den else None


def _calibrate_specialists(data: FrozenScores, calib_idx: np.ndarray,
                           alpha: float, routed_only: bool,
                           router_threshold: float | None = None) -> tuple[np.ndarray, int]:
    """Calibrate one LEC threshold per predicted clinic; ``nan`` means infeasible."""
    thresholds = np.full(NUM_DISEASES, np.nan, dtype=np.float64)
    eligible = np.ones(calib_idx.size, dtype=bool)
    if routed_only:
        if router_threshold is None:
            return thresholds, NUM_DISEASES
        eligible &= data.router_uncertainty[calib_idx] <= router_threshold

    infeasible = 0
    for clinic in range(NUM_DISEASES):
        mask = eligible & (data.predicted_disease[calib_idx] == clinic)
        if not np.any(mask):
            infeasible += 1
            continue
        idx = calib_idx[mask]
        threshold = _lec_calibrate_np(
            data.specialist_uncertainty[idx], data.system_correctness[idx], alpha
        )
        if threshold is None:
            infeasible += 1
        else:
            thresholds[clinic] = threshold
    return thresholds, infeasible


def _evaluate_acceptance(data: FrozenScores, test_idx: np.ndarray,
                         thresholds: np.ndarray,
                         router_threshold: float | None = None) -> tuple[int, int, int]:
    """Return accepted cases, accepted errors, and held-out denominator."""
    if router_threshold is None:
        routed = np.ones(test_idx.size, dtype=bool)
    else:
        routed = data.router_uncertainty[test_idx] <= router_threshold

    clinic_threshold = thresholds[data.predicted_disease[test_idx]]
    accepted = routed & np.isfinite(clinic_threshold) & (
        data.specialist_uncertainty[test_idx] <= clinic_threshold
    )
    selected = int(accepted.sum())
    errors = int((accepted & (data.system_correctness[test_idx] == 0)).sum())
    return selected, errors, int(test_idx.size)


def _evaluate_split(data: FrozenScores, calib_idx: np.ndarray, test_idx: np.ndarray,
                    beta: float, alpha: float) -> dict[str, Any]:
    """Replay proposed LEC and specialist-only rejection for one split."""
    router_threshold = _lec_calibrate_np(
        data.router_uncertainty[calib_idx], data.router_correctness[calib_idx], beta
    )

    proposed_thresholds, proposed_infeasible = _calibrate_specialists(
        data, calib_idx, alpha, routed_only=True, router_threshold=router_threshold
    )
    if router_threshold is None:
        p_selected = p_errors = 0
        total = int(test_idx.size)
    else:
        p_selected, p_errors, total = _evaluate_acceptance(
            data, test_idx, proposed_thresholds, router_threshold
        )

    specialist_thresholds, specialist_infeasible = _calibrate_specialists(
        data, calib_idx, alpha, routed_only=False
    )
    s_selected, s_errors, specialist_total = _evaluate_acceptance(
        data, test_idx, specialist_thresholds
    )
    if specialist_total != total:
        raise RuntimeError("proposed and specialist-only held-out denominators differ")

    return {
        "total": total,
        "proposed_selected": p_selected,
        "proposed_errors": p_errors,
        "specialist_selected": s_selected,
        "specialist_errors": s_errors,
        "router_threshold": router_threshold,
        "router_feasible": router_threshold is not None,
        "proposed_specialist_infeasible": proposed_infeasible,
        "specialist_only_infeasible": specialist_infeasible,
    }


def _historical_split(data: FrozenScores, epoch: int,
                      split_ratio: float) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce Stage 2's source-stratified split for one historical epoch."""
    rng = random.Random(epoch)
    calib_parts, test_parts = [], []
    for source in data.source_names:
        ids = data.source_indices[source].tolist()
        rng.shuffle(ids)
        n_calib = int(len(ids) * split_ratio)
        calib_parts.append(np.asarray(ids[:n_calib], dtype=np.int32))
        test_parts.append(np.asarray(ids[n_calib:], dtype=np.int32))
    return np.concatenate(calib_parts), np.concatenate(test_parts)


def _bootstrap_split(data: FrozenScores, replicate: int, seed: int,
                     split_ratio: float) -> tuple[np.ndarray, np.ndarray]:
    """Draw one source-dataset-stratified record bootstrap split.

    Records are first sampled with replacement within source dataset.  The
    resulting *occurrences* are then shuffled and allocated to calibration or
    evaluation, preserving a disjoint partition of bootstrap occurrences.
    Duplicate original records can appear in both roles, as in a standard
    nonparametric bootstrap, but are never treated as distinct patients.
    """
    rng = np.random.default_rng(np.random.SeedSequence([seed, replicate]))
    calib_parts, test_parts = [], []
    for source in data.source_names:
        source_idx = data.source_indices[source]
        sampled = source_idx[rng.integers(0, source_idx.size, size=source_idx.size)]
        rng.shuffle(sampled)
        n_calib = int(sampled.size * split_ratio)
        calib_parts.append(sampled[:n_calib])
        test_parts.append(sampled[n_calib:])
    return np.concatenate(calib_parts), np.concatenate(test_parts)


def _pooled_reference(data: FrozenScores, beta: float, alpha: float,
                      split_ratio: float, n_splits: int) -> dict[str, Any]:
    """Compute the existing 500-split pooled reference without resampling splits."""
    totals = defaultdict(int)
    n_router_infeasible = 0
    prop_spec_infeasible = 0
    spec_only_infeasible = 0
    for epoch in range(n_splits):
        calib_idx, test_idx = _historical_split(data, epoch, split_ratio)
        result = _evaluate_split(data, calib_idx, test_idx, beta, alpha)
        for key in ("total", "proposed_selected", "proposed_errors",
                    "specialist_selected", "specialist_errors"):
            totals[key] += int(result[key])
        n_router_infeasible += int(not result["router_feasible"])
        prop_spec_infeasible += int(result["proposed_specialist_infeasible"])
        spec_only_infeasible += int(result["specialist_only_infeasible"])

    proposed_mfdr = _ratio(totals["proposed_errors"], totals["proposed_selected"])
    specialist_mfdr = _ratio(totals["specialist_errors"], totals["specialist_selected"])
    proposed_cov = _ratio(totals["proposed_selected"], totals["total"])
    specialist_cov = _ratio(totals["specialist_selected"], totals["total"])
    return {
        "n_splits": n_splits,
        "proposed": {
            "mFDR_sys": proposed_mfdr,
            "Cov_sys": proposed_cov,
            "selected": totals["proposed_selected"],
            "errors": totals["proposed_errors"],
        },
        "specialist_only": {
            "mFDR_sys": specialist_mfdr,
            "Cov_sys": specialist_cov,
            "selected": totals["specialist_selected"],
            "errors": totals["specialist_errors"],
        },
        "paired_difference": {
            "delta_mFDR_sys": (
                proposed_mfdr - specialist_mfdr
                if proposed_mfdr is not None and specialist_mfdr is not None else None
            ),
            "delta_Cov_sys": (
                proposed_cov - specialist_cov
                if proposed_cov is not None and specialist_cov is not None else None
            ),
        },
        "calibration_diagnostics": {
            "router_infeasible_splits": n_router_infeasible,
            "proposed_specialist_infeasible_cells": prop_spec_infeasible,
            "specialist_only_infeasible_cells": spec_only_infeasible,
        },
    }


def _summary(values: list[float | None], reference: float | None) -> dict[str, Any]:
    finite = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if finite.size == 0:
        return {
            "reference_pooled": reference,
            "bootstrap_mean": None,
            "bootstrap_median": None,
            "bootstrap_sd": None,
            "record_bootstrap_ci_low": None,
            "record_bootstrap_ci_high": None,
            "n_valid_replicates": 0,
        }
    return {
        "reference_pooled": reference,
        "bootstrap_mean": float(np.mean(finite)),
        "bootstrap_median": float(np.median(finite)),
        "bootstrap_sd": float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0,
        "record_bootstrap_ci_low": float(np.percentile(finite, 2.5)),
        "record_bootstrap_ci_high": float(np.percentile(finite, 97.5)),
        "n_valid_replicates": int(finite.size),
    }


def _format(value: float | None, digits: int = 4) -> str:
    return "NA" if value is None else f"{value:.{digits}f}"


def _write_markdown(result: dict[str, Any], path: Path) -> None:
    meta, summary = result["meta"], result["summary"]
    p_m = summary["proposed"]["mFDR_sys"]
    p_c = summary["proposed"]["Cov_sys"]
    d_m = summary["paired_difference"]["delta_mFDR_sys"]
    d_c = summary["paired_difference"]["delta_Cov_sys"]

    lines = [
        "# Source-dataset-stratified record-level bootstrap",
        "",
        f"Frozen-score two-stage evaluation at β = {meta['beta']:g}, α = {meta['alpha']:g}. "
        f"B = {meta['n_replicates']:,} outer bootstrap replicates.",
        "",
        "| Quantity | 500-split pooled reference | Record-bootstrap 95% percentile interval |",
        "|---|---:|---:|",
        f"| Proposed system mFDR$_{{sys}}$ | {_format(p_m['reference_pooled'])} | "
        f"[{_format(p_m['record_bootstrap_ci_low'])}, {_format(p_m['record_bootstrap_ci_high'])}] |",
        f"| Proposed AI-managed coverage | {_format(p_c['reference_pooled'])} | "
        f"[{_format(p_c['record_bootstrap_ci_low'])}, {_format(p_c['record_bootstrap_ci_high'])}] |",
        f"| ΔmFDR$_{{sys}}$ (LEC − specialist-only) | {_format(d_m['reference_pooled'])} | "
        f"[{_format(d_m['record_bootstrap_ci_low'])}, {_format(d_m['record_bootstrap_ci_high'])}] |",
        f"| ΔCoverage$_{{sys}}$ (LEC − specialist-only) | {_format(d_c['reference_pooled'])} | "
        f"[{_format(d_c['record_bootstrap_ci_low'])}, {_format(d_c['record_bootstrap_ci_high'])}] |",
        "",
        "## Scope and interpretation",
        "",
        "Records were sampled with replacement within each source dataset, then split "
        "into calibration and held-out evaluation occurrences before the router and "
        "clinic-specific LEC thresholds were recalibrated. The historical 500 random "
        "splits were not bootstrap units.",
        "",
        "No patient-, study-, slide-, video-, or volume-level identifiers are present in "
        "the released Step 1 artifacts. These are therefore source-dataset-stratified "
        "record-level bootstrap intervals conditional on the observed record composition "
        "and frozen model scores, not patient-level population confidence intervals.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_outputs(result: dict[str, Any], replicates: list[dict[str, Any]],
                   output_dir: Path, stem: str) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{stem}.json"
    csv_path = output_dir / f"{stem}_replicates.csv"
    markdown_path = output_dir / f"{stem}.md"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    columns = [
        "replicate", "proposed_mFDR_sys", "proposed_Cov_sys",
        "specialist_only_mFDR_sys", "specialist_only_Cov_sys",
        "delta_mFDR_sys", "delta_Cov_sys", "proposed_selected",
        "proposed_errors", "specialist_only_selected", "specialist_only_errors",
        "n_test", "router_feasible", "router_threshold",
        "proposed_specialist_infeasible", "specialist_only_infeasible",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(replicates)

    _write_markdown(result, markdown_path)
    return json_path, csv_path, markdown_path


def run_record_bootstrap(*, input_root: Path, output_dir: Path, beta: float,
                         alpha: float, n_replicates: int, seed: int,
                         split_ratio: float, lambda_1: float,
                         lambda_2: float, reference_splits: int) -> tuple[dict[str, Any], tuple[Path, Path, Path]]:
    """Run the prespecified source-dataset-stratified record bootstrap."""
    if not 0 < beta < 1:
        raise ValueError("beta must lie strictly between 0 and 1")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between 0 and 1")
    if not 0 < split_ratio < 1:
        raise ValueError("split_ratio must lie strictly between 0 and 1")
    if n_replicates < 2:
        raise ValueError("n_replicates must be at least 2")
    if reference_splits < 1:
        raise ValueError("reference_splits must be at least 1")

    print("Loading immutable Step 1 frozen scores...")
    data = _load_frozen_scores(input_root, lambda_1, lambda_2)
    print(f"  {data.n_records:,} records in {len(data.source_names)} source datasets")

    print(f"Computing {reference_splits}-split pooled reference (no resampling)...")
    t0 = time.time()
    reference = _pooled_reference(data, beta, alpha, split_ratio, reference_splits)
    print(f"  done in {time.time() - t0:.1f}s; "
          f"proposed mFDR={reference['proposed']['mFDR_sys']:.4f}, "
          f"coverage={reference['proposed']['Cov_sys']:.4f}")

    print(f"Running {n_replicates:,} source-stratified record bootstrap replicates...")
    replicates: list[dict[str, Any]] = []
    t0 = time.time()
    progress_every = max(1, n_replicates // 10)
    for replicate in range(n_replicates):
        calib_idx, test_idx = _bootstrap_split(data, replicate, seed, split_ratio)
        out = _evaluate_split(data, calib_idx, test_idx, beta, alpha)
        p_m = _ratio(out["proposed_errors"], out["proposed_selected"])
        p_c = _ratio(out["proposed_selected"], out["total"])
        s_m = _ratio(out["specialist_errors"], out["specialist_selected"])
        s_c = _ratio(out["specialist_selected"], out["total"])
        replicates.append({
            "replicate": replicate,
            "proposed_mFDR_sys": p_m,
            "proposed_Cov_sys": p_c,
            "specialist_only_mFDR_sys": s_m,
            "specialist_only_Cov_sys": s_c,
            "delta_mFDR_sys": p_m - s_m if p_m is not None and s_m is not None else None,
            "delta_Cov_sys": p_c - s_c if p_c is not None and s_c is not None else None,
            "proposed_selected": out["proposed_selected"],
            "proposed_errors": out["proposed_errors"],
            "specialist_only_selected": out["specialist_selected"],
            "specialist_only_errors": out["specialist_errors"],
            "n_test": out["total"],
            "router_feasible": int(out["router_feasible"]),
            "router_threshold": out["router_threshold"],
            "proposed_specialist_infeasible": out["proposed_specialist_infeasible"],
            "specialist_only_infeasible": out["specialist_only_infeasible"],
        })
        if (replicate + 1) % progress_every == 0 or replicate + 1 == n_replicates:
            print(f"  {replicate + 1:>5,}/{n_replicates:,} "
                  f"({time.time() - t0:.1f}s elapsed)")

    values = lambda name: [row[name] for row in replicates]
    result = {
        "meta": {
            "analysis": "source-dataset-stratified record-level bootstrap",
            "bootstrap_unit": "record",
            "inference_scope": (
                "Record-level uncertainty conditional on the observed source-dataset "
                "composition and frozen model scores; not patient-, study-, slide-, "
                "video-, or volume-level inference."
            ),
            "beta": beta,
            "alpha": alpha,
            "n_replicates": n_replicates,
            "seed": seed,
            "split_ratio": split_ratio,
            "reference_splits": reference_splits,
            "lambda_1": lambda_1,
            "lambda_2": lambda_2,
            "ci_method": "two-sided 2.5th/97.5th percentile interval",
            "split_note": (
                "The 500 historical random splits are used only for the pooled "
                "reference estimate and are never resampled as bootstrap units."
            ),
        },
        "data_profile": {
            "n_records": data.n_records,
            "n_source_datasets": len(data.source_names),
            "records_per_source_dataset": {
                source: int(data.source_indices[source].size)
                for source in data.source_names
            },
            "underlying_unit_ids_available": False,
        },
        "reference": reference,
        "summary": {
            "proposed": {
                "mFDR_sys": _summary(values("proposed_mFDR_sys"), reference["proposed"]["mFDR_sys"]),
                "Cov_sys": _summary(values("proposed_Cov_sys"), reference["proposed"]["Cov_sys"]),
            },
            "specialist_only": {
                "mFDR_sys": _summary(values("specialist_only_mFDR_sys"), reference["specialist_only"]["mFDR_sys"]),
                "Cov_sys": _summary(values("specialist_only_Cov_sys"), reference["specialist_only"]["Cov_sys"]),
            },
            "paired_difference": {
                "delta_mFDR_sys": _summary(values("delta_mFDR_sys"), reference["paired_difference"]["delta_mFDR_sys"]),
                "delta_Cov_sys": _summary(values("delta_Cov_sys"), reference["paired_difference"]["delta_Cov_sys"]),
            },
        },
        "replicate_diagnostics": {
            "router_infeasible_replicates": int(sum(not row["router_feasible"] for row in replicates)),
            "mean_proposed_specialist_infeasible_clinics": float(np.mean(values("proposed_specialist_infeasible"))),
            "mean_specialist_only_infeasible_clinics": float(np.mean(values("specialist_only_infeasible"))),
        },
    }

    stem = f"record_bootstrap_beta_{beta:.2f}_alpha_{alpha:.2f}"
    paths = _write_outputs(result, replicates, output_dir, stem)
    print(f"Wrote {paths[0]}")
    print(f"Wrote {paths[1]}")
    print(f"Wrote {paths[2]}")
    return result, paths


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Source-dataset-stratified record-level bootstrap for frozen scores."
    )
    parser.add_argument("--input-root", default="result/json",
                        help="Root containing step1/, or the step1 directory itself.")
    parser.add_argument("--output-dir", default="result/metrics")
    parser.add_argument("--beta", type=float, default=DEFAULT_BETA)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--n-replicates", type=int, default=DEFAULT_REPLICATES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--split-ratio", type=float, default=DEFAULT_SPLIT_RATIO)
    parser.add_argument("--lambda-1", type=float, default=DEFAULT_LAMBDA_1)
    parser.add_argument("--lambda-2", type=float, default=DEFAULT_LAMBDA_2)
    parser.add_argument("--reference-splits", type=int, default=DEFAULT_REFERENCE_SPLITS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_record_bootstrap(
        input_root=Path(args.input_root), output_dir=Path(args.output_dir),
        beta=args.beta, alpha=args.alpha, n_replicates=args.n_replicates,
        seed=args.seed, split_ratio=args.split_ratio, lambda_1=args.lambda_1,
        lambda_2=args.lambda_2, reference_splits=args.reference_splits,
    )


if __name__ == "__main__":
    main()
