"""Command-line entry point for the independent patient-level workflow."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .analysis import PatientExperimentConfig
from .runner import run_patient_experiment


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _config_from_json(path: Path) -> PatientExperimentConfig:
    with open(path, encoding="utf-8") as handle:
        values = json.load(handle)
    base = PROJECT_ROOT

    def resolved(name: str, default: str) -> Path:
        candidate = Path(values.get(name, default))
        return candidate.resolve() if candidate.is_absolute() else (base / candidate).resolve()

    return PatientExperimentConfig(
        input_dir=resolved("input_dir", "../data/inference"),
        output_dir=resolved("output_dir", "result/patient_level/beta-0.01_alpha-0.20"),
        beta=float(values.get("beta", 0.01)),
        alpha=float(values.get("alpha", 0.20)),
        n_splits=int(values.get("n_splits", 500)),
        n_bootstrap=int(values.get("n_bootstrap", 1000)),
        router_calib_ratio=float(values.get("router_calib_ratio", 0.30)),
        specialist_calib_ratio=float(values.get("specialist_calib_ratio", 0.30)),
        test_ratio=float(values.get("test_ratio", 0.40)),
        seed=int(values.get("seed", 20260919)),
        aggregation=str(values.get("aggregation", "mean")),
        lambda_1=float(values.get("lambda_1", 0.00)),
        lambda_2=float(values.get("lambda_2", 1.00)),
        require_real_ids=bool(values.get("require_real_ids", False)),
        run_sensitivity=bool(values.get("run_sensitivity", True)),
        run_full_suite=bool(values.get("run_full_suite", False)),
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Patient-level LEC experiment with patient-disjoint calibration/test splits."
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "patient_level_beta_001.json"),
    )
    parser.add_argument("--input-dir", help="Override Step-1 patient JSON directory.")
    parser.add_argument("--output-dir", help="Override patient-level result directory.")
    parser.add_argument("--n-splits", type=int)
    parser.add_argument("--n-bootstrap", type=int)
    parser.add_argument("--aggregation", choices=["mean", "median", "max"])
    parser.add_argument("--require-real-ids", action="store_true")
    parser.add_argument("--no-sensitivity", action="store_true")
    parser.add_argument(
        "--suite-only",
        action="store_true",
        help="Reuse the existing primary result and generate the full legacy-matched suite.",
    )
    parser.add_argument("--no-full-suite", action="store_true")
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
    if args.aggregation:
        values["aggregation"] = args.aggregation
    if args.require_real_ids:
        values["require_real_ids"] = True
    if args.no_sensitivity:
        values["run_sensitivity"] = False
    if args.no_full_suite:
        values["run_full_suite"] = False
    resolved_config = PatientExperimentConfig(**values)
    if args.suite_only:
        primary_path = resolved_config.output_dir / "metrics" / "patient_level_results.json"
        if not primary_path.exists():
            raise FileNotFoundError(
                f"Suite-only mode requires the existing primary result: {primary_path}"
            )
        with open(primary_path, encoding="utf-8") as handle:
            primary_result = json.load(handle)
        from .artifacts import validate_primary_config
        from .runner import run_full_suite

        validate_primary_config(primary_result, resolved_config)
        run_full_suite(resolved_config, primary_result=primary_result)
    else:
        run_patient_experiment(resolved_config)


if __name__ == "__main__":
    main()
