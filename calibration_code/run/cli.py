"""Single, explicit command-line entry point for experiments and reporting.

Examples
--------
python run/cli.py preflight --config configs/default.json
python run/cli.py experiment --config configs/default.json
python run/cli.py metrics --beta 0.01 --data-root result/json --input-root result/json
python run/cli.py figures --metrics-json result/metrics/metrics_beta_0.01.json \
    --figure-dir result/metrics/figures/new
python run/cli.py patient-render --run-dir result/patient_level/beta-0.01_alpha-0.20
python run/cli.py clustered-run --config configs/patient_clustered_beta_001.json
python run/cli.py clustered-paper --config configs/patient_clustered_beta_001.json
python run/cli.py clustered-render --run-dir result/patient_clustered/beta-0.01_alpha-0.20
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.contracts import load_and_validate_phase1, load_experiment_config
from pipeline.experiment.runner import run_experiment


def _path(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _config(path):
    return load_experiment_config(_path(path), PROJECT_ROOT)


def _add_config_argument(parser):
    parser.add_argument("--config", default="configs/default.json",
                        help="JSON experiment configuration relative to Code/")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Nature experiment command line")
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser("preflight", help="validate Step 1 inputs")
    _add_config_argument(preflight)

    experiment = subparsers.add_parser("experiment", help="run selected Step 2-6 stages")
    _add_config_argument(experiment)

    metrics = subparsers.add_parser("metrics", help="compute a complete metrics bundle")
    metrics.add_argument("--beta", type=float, default=0.1)
    metrics.add_argument("--alpha", type=float, default=0.2)
    metrics.add_argument("--data-root", default="result/json",
                         help="Root containing derived step2-step6 JSON")
    metrics.add_argument("--input-root", default="result/json",
                         help="Root containing immutable step1 JSON")
    metrics.add_argument("--output-dir", default="result/metrics")
    metrics.add_argument("--stem", default="metrics")

    alpha_grid = subparsers.add_parser("alpha-grid", help="compute baseline alpha grid")
    alpha_grid.add_argument("--beta", type=float, default=0.1)
    alpha_grid.add_argument("--data-root", default="result/json")
    alpha_grid.add_argument("--input-root", default="result/json")
    alpha_grid.add_argument("--output-json", default="result/metrics/baseline_alpha_grid.json")
    alpha_grid.add_argument("--output-csv", default="result/metrics/baseline_alpha_grid.csv")

    figures = subparsers.add_parser("figures", help="render the seven metrics figures")
    figures.add_argument("--metrics-json", default="result/metrics/metrics.json")
    figures.add_argument("--figure-dir", default="result/metrics/figures")

    bootstrap = subparsers.add_parser(
        "record-bootstrap",
        help="run source-dataset-stratified record-level bootstrap",
    )
    bootstrap.add_argument("--input-root", default="result/json")
    bootstrap.add_argument("--output-dir", default="result/metrics")
    bootstrap.add_argument("--beta", type=float, default=0.01)
    bootstrap.add_argument("--alpha", type=float, default=0.20)
    bootstrap.add_argument("--n-replicates", type=int, default=1000)
    bootstrap.add_argument("--seed", type=int, default=20260904)
    bootstrap.add_argument("--split-ratio", type=float, default=0.50)
    bootstrap.add_argument("--lambda-1", type=float, default=0.50)
    bootstrap.add_argument("--lambda-2", type=float, default=0.50)
    bootstrap.add_argument("--reference-splits", type=int, default=500)

    bootstrap_figure = subparsers.add_parser(
        "record-bootstrap-figure",
        help="render the record-level bootstrap interval figure",
    )
    bootstrap_figure.add_argument(
        "--bootstrap-json",
        default="result/metrics/record_bootstrap_beta_0.01_alpha_0.20.json",
    )
    bootstrap_figure.add_argument("--figure-dir", default="result/metrics/figures")

    subparsers.add_parser("tables-beta001", help="generate compatibility LaTeX tables for beta=0.01")

    for name, description in (
        ("patient-run", "run the patient-level primary experiment and optional full suite"),
        ("patient-suite", "run only the full patient-level suite using saved primary metrics"),
    ):
        patient = subparsers.add_parser(name, help=description)
        patient.add_argument("--config", default="configs/patient_level_beta_001.json")
        patient.add_argument("--input-dir")
        patient.add_argument("--output-dir")
        patient.add_argument("--n-splits", type=int)
        patient.add_argument("--n-bootstrap", type=int)
        patient.add_argument("--aggregation", choices=["mean", "median", "max"])
        patient.add_argument("--require-real-ids", action="store_true")
        patient.add_argument("--no-sensitivity", action="store_true")
        patient.add_argument("--no-full-suite", action="store_true")

    patient_render = subparsers.add_parser(
        "patient-render", help="render organized patient-level reports from saved metrics"
    )
    patient_render.add_argument(
        "--run-dir", default="result/patient_level/beta-0.01_alpha-0.20"
    )
    patient_render.add_argument("--primary-only", action="store_true")

    clustered = subparsers.add_parser(
        "clustered-run",
        help="run patient-disjoint mixed-scope decision analysis",
    )
    clustered.add_argument("--config", default="configs/patient_clustered_beta_001.json")
    clustered.add_argument("--input-dir")
    clustered.add_argument("--output-dir")
    clustered.add_argument("--n-splits", type=int)
    clustered.add_argument("--n-bootstrap", type=int)
    clustered.add_argument("--max-patients-per-source", type=int)
    clustered.add_argument("--no-patient-balanced", action="store_true")
    clustered.add_argument("--no-strict-patient-subset", action="store_true")

    clustered_paper = subparsers.add_parser(
        "clustered-paper",
        help="run the patient-clustered experiment and install the 11-figure/6-table Paper2 set",
    )
    clustered_paper.add_argument(
        "--config", default="configs/patient_clustered_beta_001.json"
    )
    clustered_paper.add_argument("--input-dir")
    clustered_paper.add_argument("--output-dir")
    clustered_paper.add_argument("--n-splits", type=int)
    clustered_paper.add_argument("--n-bootstrap", type=int)
    clustered_paper.add_argument("--max-patients-per-source", type=int)
    clustered_paper.add_argument("--no-patient-balanced", action="store_true")
    clustered_paper.add_argument("--no-strict-patient-subset", action="store_true")
    clustered_paper.add_argument(
        "--paper2-dir", default="../Paper2.0/Clinical_Triage__New_Version_"
    )
    clustered_paper.add_argument(
        "--no-install", action="store_true",
        help="build deliverables but do not copy them into Paper2.0",
    )

    clustered_render = subparsers.add_parser(
        "clustered-render",
        help="render the Paper2 11-figure/6-table set from saved patient-clustered metrics",
    )
    clustered_render.add_argument(
        "--run-dir", default="result/patient_clustered/beta-0.01_alpha-0.20"
    )
    clustered_render.add_argument(
        "--paper2-dir", default="../Paper2.0/Clinical_Triage__New_Version_"
    )
    clustered_render.add_argument("--install-paper2", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if args.command == "preflight":
        config = _config(args.config)
        phase1 = load_and_validate_phase1(config)
        print(f"[Preflight] OK: {len(phase1):,} phase_1 samples")
        print(f"[Preflight] Derived outputs: {config.output_dir}")
        return

    if args.command == "experiment":
        run_experiment(_config(args.config))
        return

    if args.command == "metrics":
        from pipeline.metrics.api import compute_and_write_metrics

        _, json_path, markdown_path = compute_and_write_metrics(
            beta=args.beta,
            alpha=args.alpha,
            data_root=_path(args.data_root),
            input_root=_path(args.input_root),
            output_dir=_path(args.output_dir),
            stem=args.stem,
        )
        print(f"JSON saved to {json_path}")
        print(f"Markdown saved to {markdown_path}")
        return

    if args.command == "alpha-grid":
        from pipeline.metrics.baseline_alpha_grid import main as run_alpha_grid

        run_alpha_grid(
            beta=args.beta,
            data_root=_path(args.data_root),
            input_root=_path(args.input_root),
            output_json=_path(args.output_json),
            output_csv=_path(args.output_csv),
        )
        return

    if args.command == "figures":
        import os
        from pipeline.reporting.figures import render_metrics_figures

        cache_dir = PROJECT_ROOT / "result" / ".cache" / "matplotlib"
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(cache_dir)
        render_metrics_figures(_path(args.metrics_json), _path(args.figure_dir))
        return

    if args.command == "record-bootstrap":
        from pipeline.metrics.record_bootstrap import run_record_bootstrap

        _, paths = run_record_bootstrap(
            input_root=_path(args.input_root), output_dir=_path(args.output_dir),
            beta=args.beta, alpha=args.alpha, n_replicates=args.n_replicates,
            seed=args.seed, split_ratio=args.split_ratio,
            lambda_1=args.lambda_1, lambda_2=args.lambda_2,
            reference_splits=args.reference_splits,
        )
        print("Bootstrap JSON saved to " + str(paths[0]))
        return

    if args.command == "record-bootstrap-figure":
        import os
        from pipeline.reporting.plot_record_bootstrap import render

        cache_dir = PROJECT_ROOT / "result" / ".cache" / "matplotlib"
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(cache_dir)
        paths = render(_path(args.bootstrap_json), _path(args.figure_dir))
        print("Bootstrap figure saved to " + ", ".join(str(path) for path in paths))
        return

    if args.command == "tables-beta001":
        from pipeline.reporting.tables import generate_beta001_tables

        generate_beta001_tables(PROJECT_ROOT)
        return

    if args.command in {"patient-run", "patient-suite"}:
        from pipeline.patient_level.artifacts import render_saved_patient_reports
        from pipeline.patient_level.cli import _config_from_json, main as patient_main

        patient_args = ["--config", str(_path(args.config))]
        for name in ("input_dir", "output_dir", "n_splits", "n_bootstrap", "aggregation"):
            value = getattr(args, name)
            if value is not None:
                if name in {"input_dir", "output_dir"}:
                    value = _path(value)
                patient_args.extend(("--" + name.replace("_", "-"), str(value)))
        for name in ("require_real_ids", "no_sensitivity", "no_full_suite"):
            if getattr(args, name):
                patient_args.append("--" + name.replace("_", "-"))
        if args.command == "patient-suite":
            patient_args.append("--suite-only")
        patient_main(patient_args)
        config = _config_from_json(_path(args.config))
        run_dir = _path(args.output_dir) if args.output_dir else config.output_dir
        include_suite = args.command == "patient-suite" or (
            config.run_full_suite and not args.no_full_suite
        )
        paths = render_saved_patient_reports(run_dir, include_full_suite=include_suite)
        print(f"Organized patient-level reports written to {paths.root}")
        return

    if args.command == "patient-render":
        from pipeline.patient_level.artifacts import render_saved_patient_reports

        paths = render_saved_patient_reports(
            _path(args.run_dir), include_full_suite=not args.primary_only
        )
        print(f"Organized patient-level reports written to {paths.root}")
        return

    if args.command in {"clustered-run", "clustered-paper"}:
        from pipeline.patient_clustered.cli import (
            _config_from_json as clustered_config_from_json,
            main as clustered_main,
        )

        clustered_args = ["--config", str(_path(args.config))]
        for name in (
            "input_dir", "output_dir", "n_splits", "n_bootstrap",
            "max_patients_per_source",
        ):
            value = getattr(args, name)
            if value is not None:
                if name in {"input_dir", "output_dir"}:
                    value = _path(value)
                clustered_args.extend(("--" + name.replace("_", "-"), str(value)))
        if args.no_patient_balanced:
            clustered_args.append("--no-patient-balanced")
        if args.no_strict_patient_subset:
            clustered_args.append("--no-strict-patient-subset")
        clustered_main(clustered_args)
        if args.command == "clustered-paper":
            from pipeline.patient_clustered.artifacts import render_saved_paper2_artifacts

            config = clustered_config_from_json(_path(args.config))
            run_dir = _path(args.output_dir) if args.output_dir else config.output_dir
            artifact_root = render_saved_paper2_artifacts(
                run_dir,
                paper2_root=_path(args.paper2_dir),
                install=not args.no_install,
            )
            print(f"Paper2.0 11-figure/6-table set written to {artifact_root}")
        return

    if args.command == "clustered-render":
        from pipeline.patient_clustered.artifacts import render_saved_paper2_artifacts

        artifact_root = render_saved_paper2_artifacts(
            _path(args.run_dir),
            paper2_root=_path(args.paper2_dir),
            install=args.install_paper2,
        )
        print(f"Paper2.0 11-figure/6-table set written to {artifact_root}")
        return


if __name__ == "__main__":
    main()
