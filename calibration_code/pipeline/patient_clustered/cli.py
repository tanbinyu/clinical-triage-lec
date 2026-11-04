"""Command-line entry point for the patient-clustered decision workflow."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .analysis import ClusteredExperimentConfig
from .runner import run_clustered_experiment


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _config_from_json(path: Path) -> ClusteredExperimentConfig:
    with open(path, encoding="utf-8") as handle:
        values = json.load(handle)

    def resolved(name: str, default: str) -> Path:
        candidate = Path(values.get(name, default))
        return candidate.resolve() if candidate.is_absolute() else (PROJECT_ROOT / candidate).resolve()

    def optional_resolved(name: str) -> Path | None:
        raw = values.get(name)
        if not raw:
            return None
        candidate = Path(raw)
        return candidate.resolve() if candidate.is_absolute() else (PROJECT_ROOT / candidate).resolve()

    return ClusteredExperimentConfig(
        input_dir=resolved("input_dir", "../data/inference"),
        output_dir=resolved(
            "output_dir", "result/patient_clustered/beta-0.01_alpha-0.20"
        ),
        beta=float(values.get("beta", 0.01)),
        alpha=float(values.get("alpha", 0.20)),
        calibration_alpha=(
            None if values.get("calibration_alpha") is None
            else float(values["calibration_alpha"])
        ),
        primary_analysis_unit=str(
            values.get("primary_analysis_unit", "patient_balanced")
        ),
        input_cohort_scope=str(values.get("input_cohort_scope", "unverified")),
        calibration_policy_status=str(
            values.get("calibration_policy_status", "exploratory_post_hoc")
        ),
        n_splits=int(values.get("n_splits", 500)),
        n_bootstrap=int(values.get("n_bootstrap", 1000)),
        router_calib_ratio=float(values.get("router_calib_ratio", 0.40)),
        specialist_calib_ratio=float(values.get("specialist_calib_ratio", 0.30)),
        test_ratio=float(values.get("test_ratio", 0.30)),
        max_patients_per_source=(
            None if values.get("max_patients_per_source") is None
            else int(values["max_patients_per_source"])
        ),
        seed=int(values.get("seed", 20260929)),
        patient_aggregation=str(values.get("patient_aggregation", "mean")),
        lambda_1=float(values.get("lambda_1", 0.00)),
        lambda_2=float(values.get("lambda_2", 1.00)),
        require_real_ids=bool(values.get("require_real_ids", True)),
        require_specialist_only=bool(
            values.get("require_specialist_only", False)
        ),
        run_all_decisions_sensitivity=bool(
            values.get(
                "run_all_decisions_sensitivity",
                values.get("run_patient_balanced_sensitivity", True),
            )
        ),
        run_strict_patient_subset=bool(values.get("run_strict_patient_subset", True)),
        run_paper_suite=bool(values.get("run_paper_suite", True)),
        manuscript_dir=optional_resolved("manuscript_dir"),
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Patient-disjoint decision-level LEC analysis with patient-cluster bootstrap."
        )
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "patient_clustered_beta_001.json"),
    )
    parser.add_argument("--input-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--n-splits", type=int)
    parser.add_argument("--n-bootstrap", type=int)
    parser.add_argument(
        "--max-patients-per-source",
        type=int,
        help=(
            "optional score- and outcome-blind upper bound on unique patients "
            "retained from each source before role assignment"
        ),
    )
    parser.add_argument(
        "--no-secondary-sensitivity", action="store_true",
        help="skip the complementary all-decisions/patient-balanced analysis",
    )
    parser.add_argument(
        "--no-patient-balanced", action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--no-strict-patient-subset", action="store_true")
    parser.add_argument("--no-paper-suite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = _config_from_json(Path(args.config))
    values = dict(config.__dict__)
    if args.input_dir:
        values["input_dir"] = Path(args.input_dir)
    if args.output_dir:
        values["output_dir"] = Path(args.output_dir)
    if args.n_splits is not None:
        values["n_splits"] = args.n_splits
    if args.n_bootstrap is not None:
        values["n_bootstrap"] = args.n_bootstrap
    if args.max_patients_per_source is not None:
        values["max_patients_per_source"] = args.max_patients_per_source
    if args.no_secondary_sensitivity or args.no_patient_balanced:
        values["run_all_decisions_sensitivity"] = False
    if args.no_strict_patient_subset:
        values["run_strict_patient_subset"] = False
    if args.no_paper_suite:
        values["run_paper_suite"] = False
    run_clustered_experiment(ClusteredExperimentConfig(**values))


if __name__ == "__main__":
    main()
