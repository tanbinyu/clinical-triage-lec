"""Execute the mixed-scope primary analysis and strict patient-level validation."""
from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from pipeline.patient_level.analysis import patient_cluster_bootstrap, pooled_reference

from .analysis import (
    ClusteredExperimentConfig,
    clustered_bootstrap,
    clustered_reference,
    eligibility_allocation_audit,
)
from .data import aggregate_strict_patient_subset, load_clustered_decisions


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for row in rows for key in row})
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _config_dict(config: ClusteredExperimentConfig) -> dict[str, Any]:
    values = asdict(config)
    for name, value in list(values.items()):
        if isinstance(value, Path):
            values[name] = str(value.resolve())
    return values


def _rename_primary_decision_counts(value: Any) -> None:
    """Remove patient-only terminology inherited from the shared metric engine."""
    if isinstance(value, dict):
        if "test_patient_decisions" in value:
            value["test_decisions"] = value.pop("test_patient_decisions")
        for child in value.values():
            _rename_primary_decision_counts(child)
    elif isinstance(value, list):
        for child in value:
            _rename_primary_decision_counts(child)


def run_clustered_experiment(
    config: ClusteredExperimentConfig,
) -> tuple[dict[str, Any], Path]:
    config.validate()
    if not (
        config.input_cohort_scope == "test_only"
        and config.calibration_policy_status == "prespecified_confirmatory"
    ):
        print(
            "WARNING: this run is exploratory, not confirmatory/publication-ready "
            f"(input_cohort_scope={config.input_cohort_scope}, "
            f"calibration_policy_status={config.calibration_policy_status})."
        )
    output_dir = Path(config.output_dir)
    metrics_dir = output_dir / "metrics"
    figures_dir = output_dir / "figures" / "primary"
    tables_dir = output_dir / "tables" / "primary"
    for directory in (metrics_dir, figures_dir, tables_dir):
        directory.mkdir(parents=True, exist_ok=True)

    print("Loading immutable mixed-scope decision scores...")
    data, audit = load_clustered_decisions(
        config.input_dir,
        lambda_1=config.lambda_1,
        lambda_2=config.lambda_2,
        require_real_ids=config.require_real_ids,
        require_specialist_only=config.require_specialist_only,
    )
    print(f"  {data.n_decisions:,} decisions in {data.n_patients:,} patient clusters")
    eligibility = eligibility_allocation_audit(data, config)
    totals = eligibility["totals"]
    print(
        "  fixed downstream eligibility: "
        f"{totals['eligible_downstream_patients']:,} patients; "
        f"{totals['included_after_source_cap']:,} after source cap; "
        "C_R/C_S/T="
        f"{totals['router_calibration_patients']:,}/"
        f"{totals['specialist_calibration_patients']:,}/"
        f"{totals['held_out_test_patients']:,}"
    )
    primary_label = (
        "one-decision-per-patient"
        if config.patient_balanced_primary else "all-decision"
    )
    print(
        f"Running {config.n_splits:,} patient-disjoint {primary_label} splits "
        f"(reporting alpha={config.alpha:g}, calibration alpha="
        f"{config.effective_calibration_alpha:g})..."
    )
    reference_output = clustered_reference(data, config)
    reference = reference_output["summary"]
    print(f"Running {config.n_bootstrap:,} source-stratified patient-cluster bootstraps...")
    bootstrap, bootstrap_rows = clustered_bootstrap(data, reference, config)
    _rename_primary_decision_counts(reference)
    for row in bootstrap_rows:
        row["n_test_decisions"] = row.pop("n_test_patients")

    result: dict[str, Any] = {
        "meta": {
            "analysis": "patient-balanced patient-clustered evaluation",
            "unit_of_analysis": (
                "one uniformly sampled genuine decision per patient per split"
                if config.patient_balanced_primary
                else "all dataset-defined diagnostic decisions"
            ),
            "primary_analysis_unit": config.primary_analysis_unit,
            "clustering_unit": "dataset-provided patient_id",
            "split_design": "source-stratified patient-disjoint C_R/C_S/T",
            "bootstrap_design": (
                "source-stratified split-respecting patient-cluster bootstrap; "
                "a sampled patient carries its single presampled decision"
                if config.patient_balanced_primary else
                "source-stratified split-respecting patient-cluster bootstrap; "
                "a sampled patient carries all dataset-defined decisions"
            ),
            "interpretation": (
                "Every patient contributes equal weight to the primary estimands. "
                "For sources with record-scoped labels, the estimand is a uniformly "
                "sampled genuine patient decision, not an invented patient diagnosis."
                if config.patient_balanced_primary else
                "mFDR and coverage are decision-level estimands; patient IDs prevent "
                "leakage and define uncertainty clusters"
            ),
            "reporting_alpha": config.alpha,
            "calibration_alpha": config.effective_calibration_alpha,
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
            "config": _config_dict(config),
        },
        "input_audit": audit,
        "eligibility_and_allocation": eligibility,
        "reference": reference,
        "bootstrap": bootstrap,
    }
    result_path = metrics_dir / "patient_clustered_results.json"
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    (metrics_dir / "input_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (metrics_dir / "eligibility_allocation_audit.json").write_text(
        json.dumps(eligibility, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _write_csv(metrics_dir / "reference_split_metrics.csv", reference_output["split_rows"])
    _write_csv(metrics_dir / "patient_cluster_bootstrap_replicates.csv", bootstrap_rows)
    if len(reference_output["split_rows"]) != config.n_splits:
        raise RuntimeError(
            f"Reference split output has {len(reference_output['split_rows'])} rows; "
            f"expected {config.n_splits}"
        )
    if len(bootstrap_rows) != config.n_bootstrap:
        raise RuntimeError(
            f"Bootstrap output has {len(bootstrap_rows)} rows; "
            f"expected {config.n_bootstrap}"
        )

    secondary_result = None
    secondary_metrics_path = None
    if config.run_all_decisions_sensitivity:
        secondary_is_balanced = not config.patient_balanced_primary
        secondary_slug = (
            "patient_balanced" if secondary_is_balanced else "all_decisions"
        )
        secondary_description = (
            "one uniformly sampled decision per patient per split"
            if secondary_is_balanced else
            "all dataset-defined decisions with patient-clustered inference"
        )
        print(f"Running secondary analysis: {secondary_description}...")
        secondary_reference_output = clustered_reference(
            data, config, patient_balanced=secondary_is_balanced
        )
        secondary_reference = secondary_reference_output["summary"]
        secondary_bootstrap, secondary_rows = clustered_bootstrap(
            data, secondary_reference, config, patient_balanced=secondary_is_balanced
        )
        _rename_primary_decision_counts(secondary_reference)
        for row in secondary_rows:
            if "n_test_patients" in row:
                row["n_test_decisions"] = row.pop("n_test_patients")
        secondary_result = {
            "meta": {
                "analysis": f"{secondary_slug} sensitivity analysis",
                "unit_of_analysis": secondary_description,
                "purpose": "sensitivity to unequal numbers of records per patient",
                "config": _config_dict(config),
            },
            "input_audit": audit,
            "reference": secondary_reference,
            "bootstrap": secondary_bootstrap,
        }
        secondary_metrics_path = metrics_dir / f"{secondary_slug}_sensitivity.json"
        secondary_metrics_path.write_text(
            json.dumps(secondary_result, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        _write_csv(
            metrics_dir / f"{secondary_slug}_reference_splits.csv",
            secondary_reference_output["split_rows"],
        )
        _write_csv(
            metrics_dir / f"{secondary_slug}_bootstrap_replicates.csv",
            secondary_rows,
        )

    strict_result = None
    if config.run_strict_patient_subset:
        print("Running strict patient/volume-scope validation subset...")
        strict_data, strict_audit = aggregate_strict_patient_subset(
            data, aggregation=config.patient_aggregation
        )
        ratios = (
            config.router_calib_ratio,
            config.specialist_calib_ratio,
            config.test_ratio,
        )
        strict_reference_output = pooled_reference(
            strict_data,
            beta=config.beta,
            alpha=config.effective_calibration_alpha,
            reporting_alpha=config.alpha,
            n_splits=config.n_splits,
            seed=config.seed,
            ratios=ratios,
        )
        strict_reference = strict_reference_output["summary"]
        strict_bootstrap, strict_rows = patient_cluster_bootstrap(
            strict_data,
            strict_reference,
            beta=config.beta,
            alpha=config.effective_calibration_alpha,
            reporting_alpha=config.alpha,
            n_replicates=config.n_bootstrap,
            seed=config.seed + 200_000,
            ratios=ratios,
        )
        strict_result = {
            "meta": {
                "analysis": "strict one-patient/volume-one-result validation",
                "unit_of_analysis": "patient or unique volume",
                "aggregation": config.patient_aggregation,
                "config": _config_dict(config),
            },
            "input_audit": strict_audit,
            "reference": strict_reference,
            "bootstrap": strict_bootstrap,
        }
        (metrics_dir / "strict_patient_subset.json").write_text(
            json.dumps(strict_result, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        _write_csv(
            metrics_dir / "strict_patient_reference_splits.csv",
            strict_reference_output["split_rows"],
        )
        _write_csv(metrics_dir / "strict_patient_bootstrap_replicates.csv", strict_rows)

    from .reporting import write_reports

    write_reports(
        result,
        patient_balanced=secondary_result,
        strict_patient=strict_result,
        figures_dir=figures_dir,
        tables_dir=tables_dir,
    )
    paper_suite_path = None
    if config.run_paper_suite:
        from .suite import run_patient_clustered_suite

        _, paper_suite_path = run_patient_clustered_suite(
            data,
            audit,
            config,
            primary_result=result,
            patient_balanced=secondary_result,
            strict_patient=strict_result,
        )
    manifest = {
        "authoritative_input": str(Path(config.input_dir).resolve()),
        "primary_metrics": str(result_path.resolve()),
        "primary_analysis_unit": config.primary_analysis_unit,
        "reporting_alpha": config.alpha,
        "calibration_alpha": config.effective_calibration_alpha,
        "router_score": "router_uncertainty",
        "routed_specialist_score": "root phase_2_*.json: uncertainty",
        "specialist_only_score": "root phase_2_*.json: uncertainty",
        "proposed_pathway_definition": "router predicted pathway",
        "specialist_only_pathway_definition": "router predicted pathway",
        "input_cohort_scope": config.input_cohort_scope,
        "calibration_policy_status": config.calibration_policy_status,
        "publication_ready": (
            config.input_cohort_scope == "test_only"
            and config.calibration_policy_status == "prespecified_confirmatory"
        ),
        "secondary_sensitivity_metrics": (
            str(secondary_metrics_path.resolve()) if secondary_metrics_path else None
        ),
        "strict_patient_metrics": (
            str((metrics_dir / "strict_patient_subset.json").resolve())
            if strict_result else None
        ),
        "paper_suite_metrics": (
            str(paper_suite_path.resolve()) if paper_suite_path else None
        ),
        "paper_figures": (
            str((output_dir / "figures" / "paper").resolve())
            if paper_suite_path else None
        ),
        "paper_tables": (
            str((output_dir / "tables" / "paper").resolve())
            if paper_suite_path else None
        ),
        "supplementary_figures": (
            str((output_dir / "figures" / "supplementary").resolve())
            if paper_suite_path else None
        ),
        "supplementary_tables": (
            str((output_dir / "tables" / "supplementary").resolve())
            if paper_suite_path else None
        ),
        "paper2_tables": (
            str((output_dir / "tables" / "paper2" / "newtab").resolve())
            if paper_suite_path else None
        ),
        "paper2_deliverables": None,
        "paper2_artifact_counts": None,
        "paper2_installed_to": None,
        "manuscript_figures": (
            str((Path(config.manuscript_dir) / "figures" / "patient_clustered").resolve())
            if paper_suite_path and config.manuscript_dir else None
        ),
        "manuscript_tables": (
            str((Path(config.manuscript_dir) / "tables" / "patient_clustered").resolve())
            if paper_suite_path and config.manuscript_dir else None
        ),
        "input_files": audit["files"],
        "eligibility_and_allocation": eligibility,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    from .provenance import build_run_provenance, write_run_provenance

    provenance = build_run_provenance(
        config=config,
        input_audit=audit,
        eligibility_audit=eligibility,
        output_dir=output_dir,
        require_results=True,
    )
    write_run_provenance(output_dir / "run_provenance.json", provenance)
    print(f"Patient-clustered results written to {output_dir.resolve()}")
    return result, result_path
