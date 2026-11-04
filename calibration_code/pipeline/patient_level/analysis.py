"""Patient-disjoint three-way calibration, evaluation and cluster bootstrap."""
from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .data import PatientScores
from .metadata import DISEASE_NAMES, NUM_DISEASES


@dataclass(frozen=True)
class PatientExperimentConfig:
    """Parameters for one patient-level frozen-score experiment."""

    input_dir: Path
    output_dir: Path
    beta: float = 0.01
    alpha: float = 0.20
    n_splits: int = 500
    n_bootstrap: int = 1000
    router_calib_ratio: float = 0.40
    specialist_calib_ratio: float = 0.30
    test_ratio: float = 0.30
    seed: int = 20260919
    aggregation: str = "mean"
    # Legacy fused-score weights; specialist calibration now uses u_i^k only.
    lambda_1: float = 0.00
    lambda_2: float = 1.00
    require_real_ids: bool = False
    run_sensitivity: bool = True
    run_full_suite: bool = False

    def validate(self) -> None:
        if not 0 < self.beta < 1 or not 0 < self.alpha < 1:
            raise ValueError("beta and alpha must lie strictly between zero and one")
        ratios = (
            self.router_calib_ratio,
            self.specialist_calib_ratio,
            self.test_ratio,
        )
        if any(ratio <= 0 for ratio in ratios) or not np.isclose(sum(ratios), 1.0):
            raise ValueError("router/specialist/test ratios must be positive and sum to one")
        if self.n_splits < 1 or self.n_bootstrap < 2:
            raise ValueError("n_splits must be positive and n_bootstrap at least two")


def _ratio(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def _lec_calibrate(
    uncertainties: np.ndarray,
    correctness: np.ndarray,
    alpha: float,
) -> float | None:
    """Return max feasible LEC threshold while treating score ties together."""
    if uncertainties.size == 0:
        return None
    order = np.argsort(uncertainties, kind="stable")
    scores = uncertainties[order]
    errors = np.cumsum(correctness[order] == 0)
    last_in_tie = np.r_[scores[1:] != scores[:-1], True]
    n_accepted = np.arange(1, scores.size + 1)
    feasible = last_in_tie & (errors - alpha * n_accepted <= -1)
    if not np.any(feasible):
        return None
    return float(scores[np.flatnonzero(feasible)[-1]])


def _partition_sizes(n_items: int, ratios: tuple[float, float, float]) -> tuple[int, int, int]:
    """Allocate a source stratum by largest remainder, with non-empty parts."""
    raw = np.asarray(ratios, dtype=float) * n_items
    sizes = np.floor(raw).astype(int)
    for index in np.argsort(-(raw - sizes))[: n_items - int(sizes.sum())]:
        sizes[index] += 1
    if n_items >= 3:
        for index in np.flatnonzero(sizes == 0):
            donor = int(np.argmax(sizes))
            sizes[donor] -= 1
            sizes[index] += 1
    return int(sizes[0]), int(sizes[1]), int(sizes[2])


def patient_disjoint_split(
    data: PatientScores,
    *,
    epoch: int,
    seed: int,
    ratios: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split patient units within every source dataset into C_R, C_S and T."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, epoch]))
    router_parts, specialist_parts, test_parts = [], [], []
    for source in sorted(data.source_indices):
        indices = data.source_indices[source].copy()
        rng.shuffle(indices)
        n_router, n_specialist, _ = _partition_sizes(indices.size, ratios)
        router_parts.append(indices[:n_router])
        specialist_parts.append(indices[n_router:n_router + n_specialist])
        test_parts.append(indices[n_router + n_specialist:])
    router = np.concatenate(router_parts)
    specialist = np.concatenate(specialist_parts)
    test = np.concatenate(test_parts)
    if set(router) & set(specialist) or set(router) & set(test) or set(specialist) & set(test):
        raise RuntimeError("Patient leakage detected across C_R, C_S and T")
    return router, specialist, test


def _bootstrap_partitions(
    data: PatientScores,
    *,
    replicate: int,
    seed: int,
    ratios: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create disjoint source strata, then resample patients within each role.

    Splitting precedes resampling so the same original patient can never occur
    in calibration and test. Resampling within role/source strata keeps all
    patient-level observations paired for the proposed and comparator methods.
    """
    router, specialist, test = patient_disjoint_split(
        data, epoch=replicate, seed=seed, ratios=ratios
    )
    rng = np.random.default_rng(np.random.SeedSequence([seed, replicate, 991]))

    def resample(partition: np.ndarray) -> np.ndarray:
        parts = []
        sources = np.asarray(data.sources, dtype=object)
        for source in sorted(data.source_indices):
            stratum = partition[sources[partition] == source]
            draw = stratum[rng.integers(0, stratum.size, size=stratum.size)]
            parts.append(draw)
        return np.concatenate(parts)

    return resample(router), resample(specialist), resample(test)


def _calibrate_specialists(
    data: PatientScores,
    specialist_idx: np.ndarray,
    *,
    alpha: float,
    router_threshold: float | None,
    routed_only: bool,
    specialist_only: bool = False,
) -> tuple[np.ndarray, int]:
    """Calibrate routed specialist streams.

    ``specialist_only`` denotes the routed specialist-only ablation: router
    assignments are retained but router abstention is disabled.  It therefore
    uses exactly the same predicted pathways, routed specialist scores and
    end-to-end correctness as Proposed LEC.
    """
    thresholds = np.full(NUM_DISEASES, np.nan, dtype=float)
    eligible = np.ones(specialist_idx.size, dtype=bool)
    if routed_only:
        if router_threshold is None:
            return thresholds, NUM_DISEASES
        eligible &= data.router_uncertainty[specialist_idx] <= router_threshold
    infeasible = 0
    specialist_score = data.specialist_score
    specialist_correctness = data.system_correctness
    pathway = data.predicted_disease
    for clinic in range(NUM_DISEASES):
        mask = eligible & (pathway[specialist_idx] == clinic)
        idx = specialist_idx[mask]
        threshold = _lec_calibrate(
            specialist_score[idx], specialist_correctness[idx], alpha
        )
        if threshold is None:
            infeasible += 1
        else:
            thresholds[clinic] = threshold
    return thresholds, infeasible


def _acceptance_counts(
    data: PatientScores,
    test_idx: np.ndarray,
    thresholds: np.ndarray,
    *,
    router_threshold: float | None,
    specialist_only: bool = False,
) -> dict[str, Any]:
    routed = np.ones(test_idx.size, dtype=bool)
    if router_threshold is not None:
        routed &= data.router_uncertainty[test_idx] <= router_threshold
    pathway = data.predicted_disease
    specialist_score = data.specialist_score
    correctness = data.system_correctness
    clinic_thresholds = thresholds[pathway[test_idx]]
    accepted = routed & np.isfinite(clinic_thresholds) & (
        specialist_score[test_idx] <= clinic_thresholds
    )
    errors = accepted & (correctness[test_idx] == 0)
    accepted_wrong_route = accepted & (data.router_correctness[test_idx] == 0)
    accepted_diagnostic_error = accepted & (
        (data.router_correctness[test_idx] == 1)
        & (data.specialist_correctness[test_idx] == 0)
    )
    per_clinic = {}
    for clinic in range(NUM_DISEASES):
        clinic_mask = pathway[test_idx] == clinic
        per_clinic[str(clinic)] = {
            "total": int(clinic_mask.sum()),
            "routed": int((routed & clinic_mask).sum()),
            "selected": int((accepted & clinic_mask).sum()),
            "errors": int((errors & clinic_mask).sum()),
            "accepted_wrong_route": int((accepted_wrong_route & clinic_mask).sum()),
            "accepted_diagnostic_error": int((accepted_diagnostic_error & clinic_mask).sum()),
            "wrong_route_applicable": True,
        }
    return {
        "total": int(test_idx.size),
        "routed": int(routed.sum()),
        "selected": int(accepted.sum()),
        "errors": int(errors.sum()),
        "accepted_wrong_route": int(accepted_wrong_route.sum()),
        "accepted_diagnostic_error": int(accepted_diagnostic_error.sum()),
        "wrong_route_applicable": True,
        "per_clinic": per_clinic,
    }


def evaluate_split(
    data: PatientScores,
    router_calib_idx: np.ndarray,
    specialist_calib_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    beta: float,
    alpha: float,
) -> dict[str, Any]:
    router_threshold = _lec_calibrate(
        data.router_uncertainty[router_calib_idx],
        data.router_correctness[router_calib_idx],
        beta,
    )
    proposed_thresholds, proposed_infeasible = _calibrate_specialists(
        data,
        specialist_calib_idx,
        alpha=alpha,
        router_threshold=router_threshold,
        routed_only=True,
    )
    specialist_thresholds, specialist_infeasible = _calibrate_specialists(
        data,
        specialist_calib_idx,
        alpha=alpha,
        router_threshold=None,
        routed_only=False,
        specialist_only=True,
    )
    if router_threshold is None:
        proposed = {
            "total": int(test_idx.size),
            "routed": 0,
            "selected": 0,
            "errors": 0,
            "accepted_wrong_route": 0,
            "accepted_diagnostic_error": 0,
            "per_clinic": {
                str(clinic): {
                    "total": int(np.sum(data.predicted_disease[test_idx] == clinic)),
                    "routed": 0,
                    "selected": 0,
                    "errors": 0,
                    "accepted_wrong_route": 0,
                    "accepted_diagnostic_error": 0,
                }
                for clinic in range(NUM_DISEASES)
            },
        }
    else:
        proposed = _acceptance_counts(
            data, test_idx, proposed_thresholds, router_threshold=router_threshold
        )
    specialist_only = _acceptance_counts(
        data, test_idx, specialist_thresholds, router_threshold=None,
        specialist_only=True,
    )
    return {
        "router_threshold": router_threshold,
        "router_feasible": router_threshold is not None,
        "proposed_specialist_infeasible": proposed_infeasible,
        "specialist_only_infeasible": specialist_infeasible,
        "specialist_only_thresholds": specialist_thresholds,
        "proposed": proposed,
        "specialist_only": specialist_only,
    }


def _empty_totals() -> dict[str, Any]:
    return {
        "total": 0,
        "routed": 0,
        "selected": 0,
        "errors": 0,
        "accepted_wrong_route": 0,
        "accepted_diagnostic_error": 0,
        "wrong_route_applicable": True,
        "per_clinic": {
            str(clinic): {
                "total": 0, "routed": 0, "selected": 0, "errors": 0,
                "accepted_wrong_route": 0,
                "accepted_diagnostic_error": 0,
                "wrong_route_applicable": True,
            }
            for clinic in range(NUM_DISEASES)
        },
    }


def _add_counts(target: dict[str, Any], source: dict[str, Any]) -> None:
    target["wrong_route_applicable"] &= source.get("wrong_route_applicable", True)
    for key in (
        "total", "routed", "selected", "errors",
        "accepted_wrong_route", "accepted_diagnostic_error",
    ):
        target[key] += int(source[key])
    for clinic in range(NUM_DISEASES):
        target["per_clinic"][str(clinic)]["wrong_route_applicable"] &= (
            source["per_clinic"][str(clinic)].get("wrong_route_applicable", True)
        )
        for key in (
            "total", "routed", "selected", "errors",
            "accepted_wrong_route", "accepted_diagnostic_error",
        ):
            target["per_clinic"][str(clinic)][key] += int(
                source["per_clinic"][str(clinic)][key]
            )


def _metrics_from_counts(counts: dict[str, Any]) -> dict[str, Any]:
    result = {
        "mFDR_sys": _ratio(counts["errors"], counts["selected"]),
        "coverage": _ratio(counts["selected"], counts["total"]),
        "conditional_specialist_coverage": _ratio(
            counts["selected"], counts.get("routed", counts["total"])
        ),
        "referral": _ratio(counts["total"] - counts["selected"], counts["total"]),
        "selected": counts["selected"],
        "errors": counts["errors"],
        "test_patient_decisions": counts["total"],
        "routed_decisions": counts.get("routed", counts["total"]),
    }
    if "accepted_wrong_route" in counts:
        result.update({
            "accepted_wrong_route_rate": (
                _ratio(counts["accepted_wrong_route"], counts["selected"])
                if counts.get("wrong_route_applicable", True) else None
            ),
            "accepted_wrong_route": counts["accepted_wrong_route"],
            "accepted_diagnostic_error": counts["accepted_diagnostic_error"],
        })
    return result


def _pathway_control_summary(
    counts: dict[str, Any], alpha: float
) -> dict[str, Any]:
    """Summarize the pathway-level safety profile of one method."""
    risks = []
    for clinic in range(NUM_DISEASES):
        block = counts["per_clinic"][str(clinic)]
        if block["selected"]:
            risks.append(block["errors"] / block["selected"])
    return {
        "mFDR_macro": float(np.mean(risks)) if risks else None,
        "mFDR_worst": float(np.max(risks)) if risks else None,
        "K_meet": int(sum(risk <= alpha for risk in risks)),
        "K_estimable": len(risks),
    }


def pooled_reference(
    data: PatientScores,
    *,
    beta: float,
    alpha: float,
    reporting_alpha: float | None = None,
    n_splits: int,
    seed: int,
    ratios: tuple[float, float, float],
    split_function: Callable[..., tuple[np.ndarray, np.ndarray, np.ndarray]] | None = None,
) -> dict[str, Any]:
    splitter = split_function or patient_disjoint_split
    control_alpha = alpha if reporting_alpha is None else reporting_alpha
    totals = {"proposed": _empty_totals(), "specialist_only": _empty_totals()}
    split_rows = []
    router_infeasible = 0
    proposed_infeasible = 0
    specialist_infeasible = 0
    specialist_feasible_by_pathway = np.zeros(NUM_DISEASES, dtype=int)
    specialist_zero_acceptance_by_pathway = np.zeros(NUM_DISEASES, dtype=int)
    for epoch in range(n_splits):
        router, specialist, test = splitter(
            data, epoch=epoch, seed=seed, ratios=ratios
        )
        result = evaluate_split(
            data, router, specialist, test, beta=beta, alpha=alpha
        )
        for method in ("proposed", "specialist_only"):
            _add_counts(totals[method], result[method])
        router_infeasible += int(not result["router_feasible"])
        proposed_infeasible += int(result["proposed_specialist_infeasible"])
        specialist_infeasible += int(result["specialist_only_infeasible"])
        specialist_feasible_by_pathway += np.isfinite(
            result["specialist_only_thresholds"]
        ).astype(int)
        for clinic in range(NUM_DISEASES):
            specialist_zero_acceptance_by_pathway[clinic] += int(
                result["specialist_only"]["per_clinic"][str(clinic)]["selected"] == 0
            )
        p_metrics = _metrics_from_counts(result["proposed"])
        s_metrics = _metrics_from_counts(result["specialist_only"])
        split_rows.append({
            "split": epoch,
            "proposed_mFDR_sys": p_metrics["mFDR_sys"],
            "proposed_coverage": p_metrics["coverage"],
            "specialist_only_mFDR_sys": s_metrics["mFDR_sys"],
            "specialist_only_coverage": s_metrics["coverage"],
            "proposed_accepted_wrong_route_rate": p_metrics["accepted_wrong_route_rate"],
            "specialist_only_accepted_wrong_route_rate": s_metrics["accepted_wrong_route_rate"],
            "delta_mFDR_sys": (
                p_metrics["mFDR_sys"] - s_metrics["mFDR_sys"]
                if p_metrics["mFDR_sys"] is not None and s_metrics["mFDR_sys"] is not None
                else None
            ),
            "delta_coverage": (
                p_metrics["coverage"] - s_metrics["coverage"]
                if p_metrics["coverage"] is not None and s_metrics["coverage"] is not None
                else None
            ),
            "delta_accepted_wrong_route_rate": (
                p_metrics["accepted_wrong_route_rate"]
                - s_metrics["accepted_wrong_route_rate"]
                if p_metrics["accepted_wrong_route_rate"] is not None
                and s_metrics["accepted_wrong_route_rate"] is not None else None
            ),
        })

    result: dict[str, Any] = {
        "n_splits": n_splits,
        "proposed": _metrics_from_counts(totals["proposed"]),
        "specialist_only": _metrics_from_counts(totals["specialist_only"]),
        "per_disease": [],
        "diagnostics": {
            "router_infeasible_splits": router_infeasible,
            "proposed_specialist_infeasible_cells": proposed_infeasible,
            "specialist_only_infeasible_cells": specialist_infeasible,
            "specialist_only_feasible_splits_by_pathway": (
                specialist_feasible_by_pathway.tolist()
            ),
            "specialist_only_infeasible_frequency_by_pathway": (
                (1.0 - specialist_feasible_by_pathway / n_splits).tolist()
            ),
            "specialist_only_zero_acceptance_frequency_by_pathway": (
                (specialist_zero_acceptance_by_pathway / n_splits).tolist()
            ),
            "patient_overlap_across_roles": 0,
        },
    }
    for method in ("proposed", "specialist_only"):
        result[method].update(
            _pathway_control_summary(totals[method], control_alpha)
        )
    result["paired_difference"] = {
        "delta_mFDR_sys": (
            result["proposed"]["mFDR_sys"] - result["specialist_only"]["mFDR_sys"]
            if result["proposed"]["mFDR_sys"] is not None
            and result["specialist_only"]["mFDR_sys"] is not None else None
        ),
        "delta_coverage": (
            result["proposed"]["coverage"] - result["specialist_only"]["coverage"]
            if result["proposed"]["coverage"] is not None
            and result["specialist_only"]["coverage"] is not None else None
        ),
        "delta_mFDR_worst": (
            result["proposed"]["mFDR_worst"]
            - result["specialist_only"]["mFDR_worst"]
            if result["proposed"]["mFDR_worst"] is not None
            and result["specialist_only"]["mFDR_worst"] is not None else None
        ),
        "delta_K_meet": (
            result["proposed"]["K_meet"] - result["specialist_only"]["K_meet"]
        ),
        "delta_accepted_wrong_route_rate": (
            result["proposed"]["accepted_wrong_route_rate"]
            - result["specialist_only"]["accepted_wrong_route_rate"]
            if result["proposed"]["accepted_wrong_route_rate"] is not None
            and result["specialist_only"]["accepted_wrong_route_rate"] is not None
            else None
        ),
    }
    for clinic in range(NUM_DISEASES):
        proposed_counts = totals["proposed"]["per_clinic"][str(clinic)]
        specialist_counts = totals["specialist_only"]["per_clinic"][str(clinic)]
        result["per_disease"].append({
            "disease_id": clinic,
            "disease_name": DISEASE_NAMES[clinic],
            "proposed": _metrics_from_counts({**proposed_counts, "per_clinic": {}}),
            "specialist_only": _metrics_from_counts({**specialist_counts, "per_clinic": {}}),
        })
    return {"summary": result, "split_rows": split_rows}


def _bootstrap_summary(values: list[float | None], reference: float | None) -> dict[str, Any]:
    finite = np.asarray([
        value for value in values if value is not None and np.isfinite(value)
    ], dtype=float)
    if finite.size == 0:
        return {
            "reference_pooled": reference,
            "bootstrap_mean": None,
            "ci_low": None,
            "ci_high": None,
            "n_valid": 0,
        }
    return {
        "reference_pooled": reference,
        "bootstrap_mean": float(np.mean(finite)),
        "bootstrap_median": float(np.median(finite)),
        "bootstrap_sd": float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0,
        "ci_low": float(np.percentile(finite, 2.5)),
        "ci_high": float(np.percentile(finite, 97.5)),
        "n_valid": int(finite.size),
    }


def patient_cluster_bootstrap(
    data: PatientScores,
    reference: dict[str, Any],
    *,
    beta: float,
    alpha: float,
    reporting_alpha: float | None = None,
    n_replicates: int,
    seed: int,
    ratios: tuple[float, float, float],
    bootstrap_function: Callable[..., tuple[np.ndarray, np.ndarray, np.ndarray]] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    bootstrapper = bootstrap_function or _bootstrap_partitions
    control_alpha = alpha if reporting_alpha is None else reporting_alpha
    rows: list[dict[str, Any]] = []
    per_disease_values = {
        clinic: {
            "mFDR_sys": [], "coverage": [],
            "conditional_specialist_coverage": [],
            "specialist_only_mFDR_sys": [],
            "specialist_only_coverage": [],
            "delta_mFDR_sys": [],
        }
        for clinic in range(NUM_DISEASES)
    }
    progress_every = max(1, n_replicates // 10)
    started = time.time()
    for replicate in range(n_replicates):
        router, specialist, test = bootstrapper(
            data, replicate=replicate, seed=seed, ratios=ratios
        )
        result = evaluate_split(
            data, router, specialist, test, beta=beta, alpha=alpha
        )
        proposed = _metrics_from_counts(result["proposed"])
        specialist_only = _metrics_from_counts(result["specialist_only"])
        proposed.update(
            _pathway_control_summary(result["proposed"], control_alpha)
        )
        specialist_only.update(
            _pathway_control_summary(result["specialist_only"], control_alpha)
        )
        row = {
            "replicate": replicate,
            "proposed_mFDR_sys": proposed["mFDR_sys"],
            "proposed_coverage": proposed["coverage"],
            "proposed_conditional_specialist_coverage": (
                proposed["conditional_specialist_coverage"]
            ),
            "specialist_only_mFDR_sys": specialist_only["mFDR_sys"],
            "specialist_only_coverage": specialist_only["coverage"],
            "proposed_mFDR_worst": proposed["mFDR_worst"],
            "specialist_only_mFDR_worst": specialist_only["mFDR_worst"],
            "proposed_K_meet": proposed["K_meet"],
            "specialist_only_K_meet": specialist_only["K_meet"],
            "proposed_accepted_wrong_route_rate": proposed["accepted_wrong_route_rate"],
            "specialist_only_accepted_wrong_route_rate": specialist_only["accepted_wrong_route_rate"],
            "delta_mFDR_sys": (
                proposed["mFDR_sys"] - specialist_only["mFDR_sys"]
                if proposed["mFDR_sys"] is not None
                and specialist_only["mFDR_sys"] is not None else None
            ),
            "delta_coverage": (
                proposed["coverage"] - specialist_only["coverage"]
                if proposed["coverage"] is not None
                and specialist_only["coverage"] is not None else None
            ),
            "delta_mFDR_worst": (
                proposed["mFDR_worst"] - specialist_only["mFDR_worst"]
                if proposed["mFDR_worst"] is not None
                and specialist_only["mFDR_worst"] is not None else None
            ),
            "delta_K_meet": proposed["K_meet"] - specialist_only["K_meet"],
            "delta_accepted_wrong_route_rate": (
                proposed["accepted_wrong_route_rate"]
                - specialist_only["accepted_wrong_route_rate"]
                if proposed["accepted_wrong_route_rate"] is not None
                and specialist_only["accepted_wrong_route_rate"] is not None
                else None
            ),
            "n_test_patients": result["proposed"]["total"],
            "router_feasible": int(result["router_feasible"]),
            "proposed_infeasible_clinics": result["proposed_specialist_infeasible"],
            "specialist_only_infeasible_clinics": result["specialist_only_infeasible"],
        }
        rows.append(row)
        for clinic in range(NUM_DISEASES):
            counts = result["proposed"]["per_clinic"][str(clinic)]
            metrics = _metrics_from_counts({**counts, "per_clinic": {}})
            per_disease_values[clinic]["mFDR_sys"].append(metrics["mFDR_sys"])
            per_disease_values[clinic]["coverage"].append(metrics["coverage"])
            per_disease_values[clinic]["conditional_specialist_coverage"].append(
                metrics["conditional_specialist_coverage"]
            )
            specialist_counts = result["specialist_only"]["per_clinic"][str(clinic)]
            specialist_metrics = _metrics_from_counts({
                **specialist_counts, "per_clinic": {}
            })
            per_disease_values[clinic]["specialist_only_mFDR_sys"].append(
                specialist_metrics["mFDR_sys"]
            )
            per_disease_values[clinic]["specialist_only_coverage"].append(
                specialist_metrics["coverage"]
            )
            per_disease_values[clinic]["delta_mFDR_sys"].append(
                metrics["mFDR_sys"] - specialist_metrics["mFDR_sys"]
                if metrics["mFDR_sys"] is not None
                and specialist_metrics["mFDR_sys"] is not None else None
            )
        if (replicate + 1) % progress_every == 0:
            print(
                f"  bootstrap {replicate + 1:,}/{n_replicates:,} "
                f"({time.time() - started:.1f}s)"
            )

    values = lambda key: [row[key] for row in rows]
    summary = {
        "proposed": {
            "mFDR_sys": _bootstrap_summary(
                values("proposed_mFDR_sys"), reference["proposed"]["mFDR_sys"]
            ),
            "coverage": _bootstrap_summary(
                values("proposed_coverage"), reference["proposed"]["coverage"]
            ),
            "conditional_specialist_coverage": _bootstrap_summary(
                values("proposed_conditional_specialist_coverage"),
                reference["proposed"]["conditional_specialist_coverage"],
            ),
            "mFDR_worst": _bootstrap_summary(
                values("proposed_mFDR_worst"), reference["proposed"]["mFDR_worst"]
            ),
            "K_meet": _bootstrap_summary(
                values("proposed_K_meet"), reference["proposed"]["K_meet"]
            ),
            "accepted_wrong_route_rate": _bootstrap_summary(
                values("proposed_accepted_wrong_route_rate"),
                reference["proposed"]["accepted_wrong_route_rate"],
            ),
        },
        "specialist_only": {
            "mFDR_sys": _bootstrap_summary(
                values("specialist_only_mFDR_sys"),
                reference["specialist_only"]["mFDR_sys"],
            ),
            "coverage": _bootstrap_summary(
                values("specialist_only_coverage"),
                reference["specialist_only"]["coverage"],
            ),
            "mFDR_worst": _bootstrap_summary(
                values("specialist_only_mFDR_worst"),
                reference["specialist_only"]["mFDR_worst"],
            ),
            "K_meet": _bootstrap_summary(
                values("specialist_only_K_meet"),
                reference["specialist_only"]["K_meet"],
            ),
            "accepted_wrong_route_rate": _bootstrap_summary(
                values("specialist_only_accepted_wrong_route_rate"),
                reference["specialist_only"]["accepted_wrong_route_rate"],
            ),
        },
        "paired_difference": {
            "delta_mFDR_sys": _bootstrap_summary(
                values("delta_mFDR_sys"), reference["paired_difference"]["delta_mFDR_sys"]
            ),
            "delta_coverage": _bootstrap_summary(
                values("delta_coverage"), reference["paired_difference"]["delta_coverage"]
            ),
            "delta_mFDR_worst": _bootstrap_summary(
                values("delta_mFDR_worst"),
                reference["paired_difference"]["delta_mFDR_worst"],
            ),
            "delta_K_meet": _bootstrap_summary(
                values("delta_K_meet"),
                reference["paired_difference"]["delta_K_meet"],
            ),
            "delta_accepted_wrong_route_rate": _bootstrap_summary(
                values("delta_accepted_wrong_route_rate"),
                reference["paired_difference"]["delta_accepted_wrong_route_rate"],
            ),
        },
        "per_disease": [],
        "diagnostics": {
            "router_infeasible_replicates": int(sum(not row["router_feasible"] for row in rows)),
            "mean_proposed_infeasible_clinics": float(np.mean([
                row["proposed_infeasible_clinics"] for row in rows
            ])),
            "mean_specialist_only_infeasible_clinics": float(np.mean([
                row["specialist_only_infeasible_clinics"] for row in rows
            ])),
            "patient_overlap_across_roles": 0,
        },
    }
    for clinic in range(NUM_DISEASES):
        reference_row = reference["per_disease"][clinic]["proposed"]
        specialist_reference = reference["per_disease"][clinic]["specialist_only"]
        summary["per_disease"].append({
            "disease_id": clinic,
            "disease_name": DISEASE_NAMES[clinic],
            "mFDR_sys": _bootstrap_summary(
                per_disease_values[clinic]["mFDR_sys"], reference_row["mFDR_sys"]
            ),
            "coverage": _bootstrap_summary(
                per_disease_values[clinic]["coverage"], reference_row["coverage"]
            ),
            "conditional_specialist_coverage": _bootstrap_summary(
                per_disease_values[clinic]["conditional_specialist_coverage"],
                reference_row["conditional_specialist_coverage"],
            ),
            "specialist_only_mFDR_sys": _bootstrap_summary(
                per_disease_values[clinic]["specialist_only_mFDR_sys"],
                specialist_reference["mFDR_sys"],
            ),
            "specialist_only_coverage": _bootstrap_summary(
                per_disease_values[clinic]["specialist_only_coverage"],
                specialist_reference["coverage"],
            ),
            "delta_mFDR_sys": _bootstrap_summary(
                per_disease_values[clinic]["delta_mFDR_sys"],
                (
                    reference_row["mFDR_sys"] - specialist_reference["mFDR_sys"]
                    if reference_row["mFDR_sys"] is not None
                    and specialist_reference["mFDR_sys"] is not None else None
                ),
            ),
        })
    return summary, rows


def run_patient_experiment(config: PatientExperimentConfig) -> tuple[dict[str, Any], Path]:
    """Backward-compatible entry point; execution lives in ``runner``."""
    from .runner import run_patient_experiment as run

    return run(config)
