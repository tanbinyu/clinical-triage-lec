from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import DATASET_ORDER, load_config
from .datasets import build_manifest, summarize_manifest
from .doctor import doctor_report
from .evaluation import evaluate_outputs
from .output import validate_json_file, validate_result_set
from .pipeline import run_pipeline
from .provenance import write_reproducibility_snapshot
from .splits import (
    build_or_load_split, split_statistics, write_downstream_partition,
)
from .workflow import load_prepared_manifests, prepare_manifests


def _add_datasets(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", action="append", choices=DATASET_ORDER)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "NatureHelp: raw datasets -> train/val/downstream -> adapters -> "
            "11 JSON files"
        )
    )
    parser.add_argument(
        "--config", type=Path,
        default=Path(__file__).resolve().parents[1] / "configs" / "default.toml",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest = subparsers.add_parser("manifest", help="scan raw data without writing state")
    _add_datasets(manifest)

    prepare = subparsers.add_parser("prepare", help="freeze manifests and patient-level splits")
    _add_datasets(prepare)
    prepare.add_argument("--rebuild-split", action="store_true")

    split = subparsers.add_parser("split", help="show persisted split statistics")
    _add_datasets(split)
    split.add_argument("--rebuild-split", action="store_true")

    partition = subparsers.add_parser(
        "partition", help="materialize CR/CS/T patient roles inside the downstream pool"
    )
    _add_datasets(partition)
    partition.add_argument("--repetition", type=int, default=0)
    partition.add_argument(
        "--reference", action="store_true",
        help="use the fixed reference seed instead of the repeated-analysis seed",
    )

    for command in ("doctor", "preflight"):
        child = subparsers.add_parser(command, help="check dependencies, data and weights")
        _add_datasets(child)
        child.add_argument("--require-adapters", action="store_true")

    provenance = subparsers.add_parser(
        "provenance", help="write dependency, seed and checkpoint inventory"
    )
    _add_datasets(provenance)
    provenance.add_argument("--hash-checkpoints", action="store_true")
    provenance.add_argument("--output", type=Path)

    run = subparsers.add_parser("run", help="run real-model inference and publish JSON")
    _add_datasets(run)
    run.add_argument(
        "--scope", choices=("all", "train", "val", "downstream"), default="all"
    )
    run.add_argument("--phase", choices=("all", "1", "2"), default="all")
    run.add_argument("--skip-raw-verification", action="store_true")
    run.add_argument("--no-resume", action="store_true")

    validate = subparsers.add_parser("validate-output", help="validate final JSON schema")
    _add_datasets(validate)
    subparsers.add_parser("evaluate", help="report phase 1, phase 2 and cascade accuracy")
    return parser


def _load_each(config, names: list[str]):
    manifests = {}
    errors = []
    for name in names:
        try:
            manifests.update(build_manifest(config, [name]))
        except Exception as exc:
            errors.append({"dataset": name, "error": str(exc)})
    return manifests, errors


def _validate_outputs(config, names: list[str]) -> list[str]:
    errors: list[str] = []
    if set(names) == set(DATASET_ORDER):
        try:
            validate_result_set(
                config.output_dir,
                {name: config.datasets[name].disease_id for name in DATASET_ORDER},
                task_contracts={
                    name: (
                        config.datasets[name].task_type,
                        config.datasets[name].num_classes,
                    )
                    for name in DATASET_ORDER
                },
            )
            return []
        except Exception as exc:
            return [str(exc)]
    phase1 = config.output_dir / "phase_1.json"
    try:
        validate_json_file(phase1, 1)
    except Exception as exc:
        errors.append(str(exc))
    for dataset in names:
        path = config.output_dir / f"phase_2_{dataset}.json"
        try:
            validate_json_file(path, 2)
        except Exception as exc:
            errors.append(str(exc))
    return errors


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    args = _parser().parse_args(argv)
    config = load_config(args.config)
    names = getattr(args, "dataset", None) or list(DATASET_ORDER)

    if args.command in {"doctor", "preflight"}:
        report = doctor_report(config, names, require_adapters=args.require_adapters)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["ok"] else 2

    if args.command == "provenance":
        path = args.output or (config.work_dir / "reproducibility_snapshot.json")
        write_reproducibility_snapshot(
            config, names, path, hash_checkpoints=args.hash_checkpoints
        )
        print(path)
        return 0

    if args.command == "manifest":
        manifests, errors = _load_each(config, names)
        print(json.dumps({
            "datasets": summarize_manifest(manifests),
            "dataset_errors": errors,
        }, ensure_ascii=False, indent=2))
        return 2 if errors else 0

    if args.command == "prepare":
        try:
            manifests = prepare_manifests(config, names, rebuild_splits=args.rebuild_split)
        except Exception as exc:
            print(f"ERROR {exc}", file=sys.stderr)
            return 2
        report = []
        for dataset, records in manifests.items():
            assignments = build_or_load_split(config, dataset, records)
            report.append({
                "dataset": dataset,
                "manifest": str(config.work_dir / "manifests" / f"{dataset}.jsonl"),
                "split": str(config.work_dir / "splits" / f"{dataset}.json"),
                "statistics": split_statistics(records, assignments),
            })
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    if args.command == "split":
        try:
            manifests = load_prepared_manifests(config, names, verify_raw=True)
            report = []
            for dataset, records in manifests.items():
                assignments = build_or_load_split(
                    config, dataset, records, rebuild=args.rebuild_split
                )
                report.append({
                    "dataset": dataset,
                    "statistics": split_statistics(records, assignments),
                })
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0
        except Exception as exc:
            print(f"ERROR {exc}", file=sys.stderr)
            return 2

    if args.command == "partition":
        try:
            manifests = load_prepared_manifests(config, names, verify_raw=True)
            path = write_downstream_partition(
                config,
                manifests,
                repetition=args.repetition,
                reference=args.reference,
            )
            payload = json.loads(path.read_text(encoding="utf-8"))
            print(json.dumps({"path": str(path), **payload}, ensure_ascii=False, indent=2))
            return 0
        except Exception as exc:
            print(f"ERROR {exc}", file=sys.stderr)
            return 2

    if args.command == "validate-output":
        errors = _validate_outputs(config, names)
        print(json.dumps({"ok": not errors, "errors": errors}, ensure_ascii=False, indent=2))
        return 2 if errors else 0

    if args.command == "evaluate":
        try:
            print(json.dumps(evaluate_outputs(config), ensure_ascii=False, indent=2))
            return 0
        except Exception as exc:
            print(f"ERROR {exc}", file=sys.stderr)
            return 2

    try:
        manifests = load_prepared_manifests(
            config, names, verify_raw=not args.skip_raw_verification
        )
        phases = (1, 2) if args.phase == "all" else (int(args.phase),)
        run_pipeline(
            config, manifests, scope=args.scope, phases=phases,
            resume=not args.no_resume,
        )
    except Exception as exc:
        print(f"ERROR {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
