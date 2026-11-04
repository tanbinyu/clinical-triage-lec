"""One command for archive preparation, manifests, training and final inference."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from naturehelp_inference.config import DATASET_ORDER, load_config
from naturehelp_inference.doctor import dependency_report
from naturehelp_inference.evaluation import evaluate_outputs
from naturehelp_inference.models import verify_weight_layout
from naturehelp_inference.pipeline import run_pipeline
from naturehelp_inference.provenance import write_reproducibility_snapshot
from naturehelp_inference.workflow import load_prepared_manifests, prepare_manifests
from prepare_datasets import prepare_breakhis, prepare_imagechd


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Run the complete NatureHelp real-data pipeline")
    parser.add_argument(
        "--config", type=Path,
        default=Path(__file__).resolve().parent / "configs" / "default.toml",
    )
    parser.add_argument("--dataset", action="append", choices=DATASET_ORDER)
    parser.add_argument(
        "--scope", choices=("all", "train", "val", "downstream"), default="all"
    )
    parser.add_argument("--skip-archive-preparation", action="store_true")
    parser.add_argument("--skip-manifest-preparation", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--resume-training", action="store_true")
    parser.add_argument("--rebuild-split", action="store_true")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--no-inference-resume", action="store_true")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    args = parser.parse_args()

    config = load_config(args.config)
    names = args.dataset or list(DATASET_ORDER)
    dependencies = dependency_report(include_inference=True)
    if not dependencies["ok"]:
        raise RuntimeError(
            "运行环境缺少依赖: " + ", ".join(dependencies["missing_modules"])
        )
    base_weight_errors = verify_weight_layout(config, names, require_adapters=False)
    if base_weight_errors:
        raise RuntimeError("基础模型权重预检失败：\n" + "\n".join(base_weight_errors))
    if not args.skip_archive_preparation:
        if "BreakHis" in names:
            prepare_breakhis(config.datasets["BreakHis"].root)
        if "ImageCHD" in names:
            prepare_imagechd(config.datasets["ImageCHD"].root)

    if not args.skip_manifest_preparation:
        prepare_manifests(config, names, rebuild_splits=args.rebuild_split)
    else:
        load_prepared_manifests(config, names, verify_raw=True)

    if not args.skip_training:
        script = Path(__file__).resolve().parent / "train_adapter.py"
        for dataset in names:
            command = [
                sys.executable, str(script), dataset,
                "--config", str(args.config),
                "--epochs", str(args.epochs),
                "--patience", str(args.patience),
            ]
            if args.resume_training:
                command.append("--resume")
            if args.rebuild_cache:
                command.append("--rebuild-cache")
            subprocess.run(command, check=True)

    weight_errors = verify_weight_layout(config, names, require_adapters=True)
    if weight_errors:
        raise RuntimeError("权重预检失败：\n" + "\n".join(weight_errors))
    manifests = load_prepared_manifests(config, names, verify_raw=True)
    write_reproducibility_snapshot(
        config,
        names,
        config.work_dir / "reproducibility_snapshot.json",
        hash_checkpoints=False,
    )
    if set(names) == set(DATASET_ORDER):
        phases = (1, 2)
    else:
        print("只选择了部分数据集：跳过全局 phase_1，仅生成所选 phase_2。", flush=True)
        phases = (2,)
    run_pipeline(
        config, manifests, scope=args.scope, phases=phases,
        resume=not args.no_inference_resume,
    )
    if set(names) == set(DATASET_ORDER):
        import json
        print(json.dumps(evaluate_outputs(config), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

