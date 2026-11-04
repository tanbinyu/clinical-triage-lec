"""Patient-level primary and full-suite execution and artifact writing."""
from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .analysis import PatientExperimentConfig, patient_cluster_bootstrap, pooled_reference
from .data import load_patient_scores
from .suite import (
    ALPHA_GRID, BETA_GRID, CALIBRATION_FRACTIONS, METHODS,
    alpha_baseline_grid, beta_grid, calibration_size_grid,
    primary_suite, utility_grid,
)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_patient_experiment(config: PatientExperimentConfig) -> tuple[dict[str, Any], Path]:
    """Run reference splits, patient bootstrap, reports and figures."""
    config.validate()
    ratios = (
        config.router_calib_ratio,
        config.specialist_calib_ratio,
        config.test_ratio,
    )
    output_dir = Path(config.output_dir)
    metrics_dir = output_dir / "metrics"
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    for directory in (metrics_dir, figures_dir, tables_dir):
        directory.mkdir(parents=True, exist_ok=True)

    print(f"Loading and aggregating patient data ({config.aggregation})...")
    data, audit = load_patient_scores(
        config.input_dir,
        aggregation=config.aggregation,
        lambda_1=config.lambda_1,
        lambda_2=config.lambda_2,
        require_real_ids=config.require_real_ids,
    )
    print(f"  {data.n_records:,} records -> {data.n_patients:,} patient decisions")
    print(f"Running {config.n_splits:,} patient-disjoint reference splits...")
    reference_out = pooled_reference(
        data,
        beta=config.beta,
        alpha=config.alpha,
        n_splits=config.n_splits,
        seed=config.seed,
        ratios=ratios,
    )
    reference = reference_out["summary"]
    print(f"Running {config.n_bootstrap:,} split-respecting patient bootstraps...")
    bootstrap, bootstrap_rows = patient_cluster_bootstrap(
        data,
        reference,
        beta=config.beta,
        alpha=config.alpha,
        n_replicates=config.n_bootstrap,
        seed=config.seed + 100_000,
        ratios=ratios,
    )

    sensitivity = []
    if config.run_sensitivity:
        for aggregation in ("mean", "median", "max"):
            if aggregation == config.aggregation:
                sensitivity_reference = reference
            else:
                sensitivity_data, _ = load_patient_scores(
                    config.input_dir,
                    aggregation=aggregation,
                    lambda_1=config.lambda_1,
                    lambda_2=config.lambda_2,
                    require_real_ids=config.require_real_ids,
                )
                sensitivity_reference = pooled_reference(
                    sensitivity_data,
                    beta=config.beta,
                    alpha=config.alpha,
                    n_splits=config.n_splits,
                    seed=config.seed,
                    ratios=ratios,
                )["summary"]
            sensitivity.append({
                "aggregation": aggregation,
                "proposed_mFDR_sys": sensitivity_reference["proposed"]["mFDR_sys"],
                "proposed_coverage": sensitivity_reference["proposed"]["coverage"],
                "specialist_only_mFDR_sys": sensitivity_reference["specialist_only"]["mFDR_sys"],
                "specialist_only_coverage": sensitivity_reference["specialist_only"]["coverage"],
                "delta_mFDR_sys": sensitivity_reference["paired_difference"]["delta_mFDR_sys"],
                "delta_coverage": sensitivity_reference["paired_difference"]["delta_coverage"],
            })

    serialisable_config = asdict(config)
    serialisable_config["input_dir"] = str(Path(config.input_dir).resolve())
    serialisable_config["output_dir"] = str(output_dir.resolve())
    result = {
        "meta": {
            "analysis": "patient-level frozen-score evaluation",
            "unit_of_analysis": "patient",
            "split_design": "source-dataset-stratified patient-disjoint C_R/C_S/T",
            "bootstrap_design": (
                "source-stratified split-respecting patient bootstrap; patients are "
                "resampled within mutually exclusive C_R/C_S/T strata"
            ),
            "ci_method": "two-sided 2.5th/97.5th percentile interval",
            "aggregation_note": (
                "Predictions are aggregated by majority vote; uncertainty is aggregated "
                f"by {config.aggregation}. Ties prefer the lower mean-uncertainty label."
            ),
            "config": serialisable_config,
        },
        "input_audit": audit,
        "reference": reference,
        "bootstrap": bootstrap,
        "aggregation_sensitivity": sensitivity,
    }
    result_path = metrics_dir / "patient_level_results.json"
    with open(result_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    with open(metrics_dir / "input_audit.json", "w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, ensure_ascii=False)
    _write_csv(metrics_dir / "reference_split_metrics.csv", reference_out["split_rows"])
    _write_csv(metrics_dir / "patient_bootstrap_replicates.csv", bootstrap_rows)
    _write_csv(metrics_dir / "aggregation_sensitivity.csv", sensitivity)

    from .reporting import write_all_reports

    write_all_reports(result, figures_dir=figures_dir, tables_dir=tables_dir)
    if config.run_full_suite:
        run_full_suite(config, primary_result=result)
    print(f"Patient-level results written to {output_dir.resolve()}")
    return result, result_path


def _write_flat_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for row in rows for key in row})
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def run_full_suite(
    config: PatientExperimentConfig,
    *,
    primary_result: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], Path]:
    ratios = (
        config.router_calib_ratio,
        config.specialist_calib_ratio,
        config.test_ratio,
    )
    data, audit = load_patient_scores(
        config.input_dir,
        aggregation=config.aggregation,
        lambda_1=config.lambda_1,
        lambda_2=config.lambda_2,
        require_real_ids=config.require_real_ids,
    )
    print("Running patient-level primary baselines, contamination and diagnostics...")
    primary = primary_suite(data, config, ratios)
    print("Running patient-level alpha and baseline grids...")
    alpha_grid = alpha_baseline_grid(data, config, ratios)
    print("Running patient-level beta grid...")
    beta_sensitivity = beta_grid(data, config, ratios)
    print("Running patient-level calibration-size grid...")
    calibration_grid = calibration_size_grid(data, config, ratios)

    if primary_result is None:
        primary_path = Path(config.output_dir) / "metrics" / "patient_level_results.json"
        primary_result = json.loads(primary_path.read_text(encoding="utf-8"))
    result = {
        "meta": {
            "unit": "patient",
            "n_splits": config.n_splits,
            "alpha": config.alpha,
            "beta": config.beta,
            "alpha_grid": ALPHA_GRID,
            "beta_grid": BETA_GRID,
            "calibration_fractions": CALIBRATION_FRACTIONS,
            "aggregation": config.aggregation,
            "split_design": "source-stratified patient-disjoint C_R/C_S/T",
        },
        "input_audit": audit,
        "primary": primary,
        "alpha_grid": alpha_grid,
        "beta_grid": beta_sensitivity,
        "calibration_size_grid": calibration_grid,
        "utility": utility_grid(primary),
    }
    metrics_dir = Path(config.output_dir) / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    path = metrics_dir / "patient_level_full_suite.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    _write_flat_csv(metrics_dir / "patient_level_baselines.csv", [
        {"method": method, **{
            key: value for key, value in primary["methods"][method].items()
            if key not in {"per_disease", "split_mFDR", "split_coverage"}
        }}
        for method in METHODS
    ])
    _write_flat_csv(metrics_dir / "patient_level_contamination.csv", primary["contamination"])
    _write_flat_csv(metrics_dir / "patient_level_calibration_feasibility.csv", primary["calibration"])
    _write_flat_csv(metrics_dir / "patient_level_exchangeability.csv", primary["exchangeability"])
    _write_flat_csv(metrics_dir / "patient_level_alpha_grid.csv", [
        {
            "alpha": row["alpha"], "method": method,
            "mFDR_sys": row["methods"][method]["mFDR_sys"],
            "coverage": row["methods"][method]["coverage"],
            "mFDR_worst": row["methods"][method]["mFDR_worst"],
            "K_meet": row["methods"][method]["K_meet"],
        }
        for row in alpha_grid for method in METHODS
    ])
    _write_flat_csv(metrics_dir / "patient_level_beta_grid.csv", [{
        "beta": row["beta"],
        "router_mFDR": row["router_mFDR"],
        "router_coverage": row["router_coverage"],
        "proposed_mFDR_sys": row["proposed"]["mFDR_sys"],
        "proposed_coverage": row["proposed"]["coverage"],
        "specialist_only_mFDR_sys": row["specialist_only"]["mFDR_sys"],
        "specialist_only_coverage": row["specialist_only"]["coverage"],
        "rho_route_mean": row["rho_route_mean"],
        "rho_acc_mean": row["rho_acc_mean"],
    } for row in beta_sensitivity])

    from .suite_reporting import write_full_suite_outputs

    write_full_suite_outputs(
        result,
        primary_result,
        figure_dir=Path(config.output_dir) / "figures" / "paper",
        table_dir=Path(config.output_dir) / "tables" / "paper",
    )
    print(f"Full patient-level suite written to {path}")
    return result, path
