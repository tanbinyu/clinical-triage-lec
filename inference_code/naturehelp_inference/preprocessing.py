from __future__ import annotations

import sys
from pathlib import Path

from .config import PipelineConfig


def prepare_acdc_for_cinema(config: PipelineConfig) -> None:
    """Run CineMA's official ACDC preprocessing, idempotently.

    This performs the official 1x1x10 mm resampling, LV-centred 192x192 crop and
    intensity normalization.  The original dataset is never modified.
    """
    spec = config.datasets["ACDC"]
    config_paths = sorted(spec.root.glob("**/patient[0-9][0-9][0-9]/Info.cfg"))
    if not config_paths:
        raise RuntimeError(f"ACDC 未找到 Info.cfg: {spec.root}")
    complete = [
        path for path in config_paths
        if (spec.prepared_root / path.parent.name / f"{path.parent.name}_sax_ed.nii.gz").exists()
        and (spec.prepared_root / path.parent.name / f"{path.parent.name}_sax_es.nii.gz").exists()
    ]
    if len(complete) == len(config_paths):
        return

    source = config.weights_root / "CineMA" / "source"
    if not (source / "cinema" / "data" / "acdc" / "preprocess.py").exists():
        raise RuntimeError(f"缺少 CineMA 官方预处理源码: {source}")
    sys.path.insert(0, str(source))
    try:
        from cinema.data.acdc.preprocess import preprocess_pid
    except ImportError as exc:
        raise RuntimeError(
            "ACDC 官方预处理需要 pandas、SimpleITK 和 tqdm；请安装 requirements.txt"
        ) from exc
    spec.prepared_root.mkdir(parents=True, exist_ok=True)
    for index, config_path in enumerate(config_paths, start=1):
        patient = config_path.parent.name
        ed = spec.prepared_root / patient / f"{patient}_sax_ed.nii.gz"
        es = spec.prepared_root / patient / f"{patient}_sax_es.nii.gz"
        if not (ed.exists() and es.exists()):
            preprocess_pid(config_path, spec.prepared_root)
        print(f"[ACDC preprocess] {index}/{len(config_paths)} {patient}", flush=True)


def prepare_model_inputs(config: PipelineConfig, names: list[str]) -> None:
    if "ACDC" in names:
        prepare_acdc_for_cinema(config)

