from __future__ import annotations

from typing import Iterable

from .config import PipelineConfig
from .datasets import build_manifest
from .manifest import freeze_manifests, manifest_digest, read_manifest
from .records import SampleRecord
from .preprocessing import prepare_model_inputs
from .splits import build_or_load_split


def prepare_manifests(
    config: PipelineConfig,
    names: Iterable[str],
    rebuild_splits: bool = False,
) -> dict[str, list[SampleRecord]]:
    selected = list(names)
    prepare_model_inputs(config, selected)
    built = build_manifest(config, selected)
    frozen = freeze_manifests(config.work_dir / "manifests", built)
    for dataset, records in frozen.items():
        build_or_load_split(config, dataset, records, rebuild=rebuild_splits)
    return frozen


def load_prepared_manifests(
    config: PipelineConfig,
    names: Iterable[str],
    verify_raw: bool = True,
) -> dict[str, list[SampleRecord]]:
    selected = list(names)
    prepared: dict[str, list[SampleRecord]] = {}
    for dataset in selected:
        path = config.work_dir / "manifests" / f"{dataset}.jsonl"
        if not path.exists():
            raise RuntimeError(
                f"{dataset}: 尚未准备 manifest；先运行 prepare 命令: {path}"
            )
        prepared[dataset] = read_manifest(path)

    if verify_raw:
        current = build_manifest(config, selected)
        for dataset in selected:
            if manifest_digest(current[dataset]) != manifest_digest(prepared[dataset]):
                raise RuntimeError(
                    f"{dataset}: 原始数据与已冻结 manifest 不一致；"
                    "请检查数据变化后重新运行 prepare --rebuild-split"
                )
    for dataset, records in prepared.items():
        build_or_load_split(config, dataset, records)
    return prepared

