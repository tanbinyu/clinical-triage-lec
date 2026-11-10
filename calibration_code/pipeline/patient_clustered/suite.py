"""Patient-clustered counterparts of the legacy comparison experiments.

Every split is made on dataset-provided patient identifiers.  All genuine
dataset-defined decisions belonging to a sampled patient are then carried into
the same role.  Consequently, image- and lesion-scoped labels are preserved
without allowing correlated records to cross calibration/test boundaries.
"""
from __future__ import annotations

import csv
import json
import shutil
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

from pipeline.patient_level.analysis import _bootstrap_summary
from pipeline.patient_level.suite import (
    ALPHA_GRID,
    BETA_GRID,
    CALIBRATION_FRACTIONS,
    METHODS,
    _add_counts,
    _counts,
    _distribution,
    _empty_counts,
    _pooled_method_summary,
    _split_methods,
    alpha_baseline_grid,
    beta_grid,
    primary_suite,
)

from .analysis import (
    ClusteredExperimentConfig,
    clustered_bootstrap_partitions,
    clustered_decision_split,
    patient_balanced_bootstrap_partitions,
    patient_balanced_split,
    _expand_all,
    _expand_one,
    _patient_partitions,
)
from .data import ClusteredDecisionScores
from pipeline.patient_level.metadata import DISEASE_NAMES, NUM_DISEASES


def _metric(counts: dict[str, Any], name: str) -> float | None:
    if name == "mFDR_sys":
        return counts["errors"] / counts["selected"] if counts["selected"] else None
    if name == "coverage":
        return counts["selected"] / counts["total"] if counts["total"] else None
    raise KeyError(name)


def baseline_cluster_bootstrap(
    data: ClusteredDecisionScores,
    reference: dict[str, Any],
    config: ClusteredExperimentConfig,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Paired source-stratified patient-cluster bootstrap for all baselines."""
    ratios = (
        config.router_calib_ratio,
        config.specialist_calib_ratio,
        config.test_ratio,
    )
    values = {
        method: {"mFDR_sys": [], "coverage": []}
        for method in METHODS
    }
    rows: list[dict[str, Any]] = []
    for replicate in range(config.n_bootstrap):
        bootstrap_function = (
            patient_balanced_bootstrap_partitions
            if config.patient_balanced_primary
            else clustered_bootstrap_partitions
        )
        router, specialist, test = bootstrap_function(
            data,
            replicate=replicate,
            seed=config.seed + 100_000,
            ratios=ratios,
            max_patients_per_source=config.max_patients_per_source,
            development_split_seed=config.development_split_seed,
            development_ratios=(
                config.development_train_ratio,
                config.development_validation_ratio,
                config.development_downstream_ratio,
            ),
        )
        state = _split_methods(
            data,
            router,
            specialist,
            test,
            beta=config.beta,
            alpha=config.effective_calibration_alpha,
        )
        row: dict[str, Any] = {
            "replicate": replicate,
            "test_decisions": int(test.size),
        }
        for method in METHODS:
            counts = _counts(
                data, test, state["accepted"][method], method=method,
                routed=(state["routed_test"] if method == "proposed" else None),
            )
            for metric in ("mFDR_sys", "coverage"):
                value = _metric(counts, metric)
                values[method][metric].append(value)
                row[f"{method}_{metric}"] = value
        rows.append(row)

    summary: dict[str, Any] = {}
    for method in METHODS:
        summary[method] = {}
        for metric in ("mFDR_sys", "coverage"):
            summary[method][metric] = _bootstrap_summary(
                values[method][metric],
                reference["methods"][method][metric],
            )
    return summary, rows


def _subsample_patient_partition(
    data: ClusteredDecisionScores,
    patient_indices: np.ndarray,
    fraction: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Retain a source-stratified fraction of patient clusters."""
    sources = np.asarray(data.patient_sources, dtype=object)
    parts = []
    for source in sorted(data.source_patient_indices):
        block = patient_indices[sources[patient_indices] == source]
        n_keep = max(1, int(round(block.size * fraction)))
        parts.append(rng.permutation(block)[:n_keep])
    return np.concatenate(parts)


def patient_clustered_calibration_size_grid(
    data: ClusteredDecisionScores,
    config: ClusteredExperimentConfig,
    *,
    primary_split_cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] | None = None,
) -> list[dict[str, Any]]:
    """Vary calibration evidence by subsampling patients, never records."""
    ratios = (
        config.router_calib_ratio,
        config.specialist_calib_ratio,
        config.test_ratio,
    )
    rows: list[dict[str, Any]] = []
    for fraction in CALIBRATION_FRACTIONS:
        totals = _empty_counts()
        router_total = router_selected = router_errors = 0
        router_thresholds: list[float | None] = []
        router_patient_sizes: list[int] = []
        router_decision_sizes: list[int] = []
        specialist_patient_sizes: list[int] = []
        specialist_decision_sizes: list[int] = []
        clinic_thresholds = {k: [] for k in range(NUM_DISEASES)}
        clinic_patient_sizes = {k: [] for k in range(NUM_DISEASES)}
        clinic_decision_sizes = {k: [] for k in range(NUM_DISEASES)}

        for epoch in range(config.n_splits):
            router_patients, specialist_patients, test_patients = _patient_partitions(
                data, epoch=epoch, seed=config.seed, ratios=ratios,
                max_patients_per_source=config.max_patients_per_source,
                development_split_seed=config.development_split_seed,
                development_ratios=(
                    config.development_train_ratio,
                    config.development_validation_ratio,
                    config.development_downstream_ratio,
                ),
            )
            rng = np.random.default_rng(np.random.SeedSequence([
                config.seed, epoch, int(fraction * 1000), 1771,
            ]))
            router_sub_patients = _subsample_patient_partition(
                data, router_patients, fraction, rng
            )
            specialist_sub_patients = _subsample_patient_partition(
                data, specialist_patients, fraction, rng
            )
            if config.patient_balanced_primary:
                router = _expand_one(data, router_sub_patients, rng=np.random.default_rng(
                    np.random.SeedSequence([config.seed, epoch, 410])
                ))
                specialist = _expand_one(
                    data, specialist_sub_patients, rng=np.random.default_rng(
                        np.random.SeedSequence([config.seed, epoch, 411])
                    )
                )
                test = (
                    primary_split_cache[epoch][2]
                    if primary_split_cache is not None
                    else _expand_one(data, test_patients, rng=np.random.default_rng(
                        np.random.SeedSequence([config.seed, epoch, 412])
                    ))
                )
            else:
                router = _expand_all(data, router_sub_patients)
                specialist = _expand_all(data, specialist_sub_patients)
                test = _expand_all(data, test_patients)

            state = _split_methods(
                data, router, specialist, test,
                beta=config.beta, alpha=config.effective_calibration_alpha,
            )
            _add_counts(
                totals,
                _counts(
                    data, test, state["accepted"]["proposed"], method="proposed",
                    routed=state["routed_test"],
                ),
            )
            routed = state["routed_test"]
            router_total += int(test.size)
            router_selected += int(routed.sum())
            router_errors += int((
                routed & (data.router_correctness[test] == 0)
            ).sum())
            router_thresholds.append(state["router_threshold"])
            router_patient_sizes.append(int(router_sub_patients.size))
            router_decision_sizes.append(int(router.size))
            specialist_patient_sizes.append(int(specialist_sub_patients.size))
            specialist_decision_sizes.append(int(specialist.size))

            for k in range(NUM_DISEASES):
                threshold = state["proposed_thresholds"][k]
                clinic_thresholds[k].append(
                    None if not np.isfinite(threshold) else float(threshold)
                )
                mask = state["routed_calib"] & (
                    data.predicted_disease[specialist] == k
                )
                clinic_records = specialist[mask]
                clinic_decision_sizes[k].append(int(clinic_records.size))
                clinic_patient_sizes[k].append(int(np.unique(
                    data.decision_patient_index[clinic_records]
                ).size))

        system = _pooled_method_summary(totals, config.alpha)
        clinics = []
        for k in range(NUM_DISEASES):
            clinics.append({
                "disease_id": k,
                "disease_name": DISEASE_NAMES[k],
                "effective_patients": _distribution(clinic_patient_sizes[k]),
                "effective_decisions": _distribution(clinic_decision_sizes[k]),
                "threshold": _distribution(clinic_thresholds[k]),
                "feasible_splits": int(sum(
                    value is not None for value in clinic_thresholds[k]
                )),
                "infeasible_frequency": float(np.mean([
                    value is None for value in clinic_thresholds[k]
                ])),
            })
        rows.append({
            "fraction": fraction,
            "router": {
                "calibration_patients": _distribution(router_patient_sizes),
                "calibration_decisions": _distribution(router_decision_sizes),
                "threshold": _distribution(router_thresholds),
                "mFDR": router_errors / router_selected if router_selected else None,
                "coverage": router_selected / router_total,
                "feasible_splits": int(sum(value is not None for value in router_thresholds)),
                "infeasible_frequency": float(np.mean([
                    value is None for value in router_thresholds
                ])),
            },
            "specialist_calibration_patients": _distribution(specialist_patient_sizes),
            "specialist_calibration_decisions": _distribution(specialist_decision_sizes),
            "system": system,
            "clinics": clinics,
        })
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _rename_decision_counts(value: Any) -> None:
    """Remove patient-only field names inherited from the generic suite engine."""
    if isinstance(value, dict):
        if "test_patient_decisions" in value:
            value["test_decisions"] = value.pop("test_patient_decisions")
        for child in value.values():
            _rename_decision_counts(child)
    elif isinstance(value, list):
        for child in value:
            _rename_decision_counts(child)


def run_patient_clustered_suite(
    data: ClusteredDecisionScores,
    audit: dict[str, Any],
    config: ClusteredExperimentConfig,
    *,
    primary_result: dict[str, Any],
    patient_balanced: dict[str, Any] | None,
    strict_patient: dict[str, Any] | None,
) -> tuple[dict[str, Any], Path]:
    """Compute and render the six-figure, three-table manuscript set."""
    ratios = (
        config.router_calib_ratio,
        config.specialist_calib_ratio,
        config.test_ratio,
    )
    primary_splitter = partial(
        patient_balanced_split
        if config.patient_balanced_primary else clustered_decision_split,
        max_patients_per_source=config.max_patients_per_source,
        development_split_seed=config.development_split_seed,
        development_ratios=(
            config.development_train_ratio,
            config.development_validation_ratio,
            config.development_downstream_ratio,
        ),
    )
    # The primary, alpha-grid and beta-grid analyses use identical partitions
    # and (for patient-balanced runs) identical sampled decisions for a given
    # epoch. Cache them once: this changes no random seed or result, while
    # avoiding hundreds of millions of repeated Python-level patient lookups.
    primary_split_cache: dict[
        int, tuple[np.ndarray, np.ndarray, np.ndarray]
    ] = {}

    def cached_primary_splitter(
        source: ClusteredDecisionScores,
        *,
        epoch: int,
        seed: int,
        ratios: tuple[float, float, float],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if epoch not in primary_split_cache:
            primary_split_cache[epoch] = primary_splitter(
                source, epoch=epoch, seed=seed, ratios=ratios
            )
        return primary_split_cache[epoch]

    print("Running patient-clustered baseline comparison...")
    primary = primary_suite(
        data,
        config,
        ratios,
        split_function=cached_primary_splitter,
        calibration_alpha=config.effective_calibration_alpha,
    )
    print("Running patient-clustered specialist-target grid...")
    alpha_rows = alpha_baseline_grid(
        data,
        config,
        ratios,
        split_function=cached_primary_splitter,
        calibration_alpha_scale=config.calibration_alpha_scale,
    )
    print("Running patient-clustered router-target grid...")
    beta_rows = beta_grid(
        data,
        config,
        ratios,
        split_function=cached_primary_splitter,
        calibration_alpha=config.effective_calibration_alpha,
    )
    print("Running patient-clustered calibration-evidence grid...")
    calibration_rows = patient_clustered_calibration_size_grid(
        data, config, primary_split_cache=primary_split_cache
    )
    print("Running paired patient-cluster bootstrap for all baseline methods...")
    baseline_bootstrap, bootstrap_rows = baseline_cluster_bootstrap(
        data, primary, config
    )
    _rename_decision_counts(primary)
    _rename_decision_counts(alpha_rows)
    _rename_decision_counts(beta_rows)
    _rename_decision_counts(calibration_rows)

    # Both implementations use the same patient-disjoint splits and LEC
    # calibration.  Refuse to publish a suite assembled from mismatched runs.
    for method in ("proposed", "specialist_only"):
        for suite_metric, primary_metric in (
            ("mFDR_sys", "mFDR_sys"), ("coverage", "coverage")
        ):
            left = primary["methods"][method][suite_metric]
            right = primary_result["reference"][method][primary_metric]
            if not np.isclose(left, right, rtol=0, atol=1e-12):
                raise RuntimeError(
                    f"Primary/suite mismatch for {method} {suite_metric}: "
                    f"{left} != {right}"
                )

    suite = {
        "meta": {
            "analysis": "patient-balanced patient-clustered experiment suite",
            "unit_of_analysis": (
                "one uniformly sampled genuine decision per patient per split"
                if config.patient_balanced_primary
                else "all dataset-defined diagnostic decisions"
            ),
            "primary_analysis_unit": config.primary_analysis_unit,
            "clustering_unit": "dataset-provided patient_id",
            "split_design": "source-stratified patient-disjoint C_R/C_S/T",
            "bootstrap_design": "source-stratified patient-cluster bootstrap",
            "beta": config.beta,
            "alpha": config.alpha,
            "calibration_alpha": config.effective_calibration_alpha,
            "calibration_alpha_scale": config.calibration_alpha_scale,
            "router_score": "router_uncertainty",
            "routed_specialist_score": "root phase_2_*.json: uncertainty",
            "specialist_only_score": "root phase_2_*.json: uncertainty",
            "proposed_pathway_definition": "router predicted pathway",
            "specialist_only_pathway_definition": "router predicted pathway",
            "input_cohort_scope": config.input_cohort_scope,
            "calibration_policy_status": config.calibration_policy_status,
            "max_patients_per_source": config.max_patients_per_source,
            "source_sampling": (
                "fixed score- and outcome-blind patient cap applied within each "
                "source before repeated C_R/C_S/T assignment"
                if config.max_patients_per_source is not None else
                "all available patients within each source"
            ),
            "n_splits": config.n_splits,
            "n_bootstrap": config.n_bootstrap,
            "alpha_grid": ALPHA_GRID,
            "beta_grid": BETA_GRID,
            "calibration_fractions": CALIBRATION_FRACTIONS,
        },
        "input_audit": audit,
        "primary": primary,
        "alpha_grid": alpha_rows,
        "beta_grid": beta_rows,
        "calibration_size_grid": calibration_rows,
        "baseline_bootstrap": baseline_bootstrap,
    }
    metrics_dir = Path(config.output_dir) / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    path = metrics_dir / "patient_clustered_paper_suite.json"
    path.write_text(json.dumps(suite, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_csv(
        metrics_dir / "patient_clustered_baseline_bootstrap_replicates.csv",
        bootstrap_rows,
    )

    from .paper_reporting import (
        write_paper2_table_set,
        write_paper_outputs,
        write_supplementary_outputs,
    )

    write_paper_outputs(
        suite,
        primary_result,
        patient_balanced=patient_balanced,
        strict_patient=strict_patient,
        figure_dir=Path(config.output_dir) / "figures" / "paper",
        table_dir=Path(config.output_dir) / "tables" / "paper",
    )
    write_supplementary_outputs(
        suite,
        primary_result,
        patient_balanced=patient_balanced,
        strict_patient=strict_patient,
        figure_dir=Path(config.output_dir) / "figures" / "supplementary",
        table_dir=Path(config.output_dir) / "tables" / "supplementary",
    )
    write_paper2_table_set(
        Path(config.output_dir) / "tables" / "paper",
        Path(config.output_dir) / "tables" / "supplementary",
        Path(config.output_dir) / "tables" / "paper2" / "newtab",
    )
    if config.manuscript_dir is not None:
        source_figures = Path(config.output_dir) / "figures" / "paper"
        source_tables = Path(config.output_dir) / "tables" / "paper"
        target_figures = Path(config.manuscript_dir) / "figures" / "patient_clustered"
        target_tables = Path(config.manuscript_dir) / "tables" / "patient_clustered"
        target_figures.mkdir(parents=True, exist_ok=True)
        target_tables.mkdir(parents=True, exist_ok=True)
        for source in source_figures.glob("*.pdf"):
            shutil.copy2(source, target_figures / source.name)
        for pattern in ("*.tex", "README.md"):
            for source in source_tables.glob(pattern):
                shutil.copy2(source, target_tables / source.name)
    return suite, path
