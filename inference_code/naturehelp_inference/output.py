from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any


PHASE1_FIELDS = (
    "uncertainty", "correctness", "predicted_disease", "true_disease",
    "source_dataset", "patient_id",
)
PHASE2_FIELDS = (
    "uncertainty", "correctness", "predicted", "true", "patient_id", "label_scope",
)
PHASE2_ROUTED_FIELDS = PHASE2_FIELDS + ("inference_model",)


def validate_payload(payload: dict[str, dict[str, Any]], phase: int) -> None:
    expected = PHASE1_FIELDS if phase == 1 else PHASE2_ROUTED_FIELDS
    for sample_id, item in payload.items():
        actual = tuple(item.keys())
        valid_layouts = (expected,)
        if actual not in valid_layouts:
            raise ValueError(
                f"{sample_id}: JSON 字段不匹配；"
                f"期望 {valid_layouts}，实际 {actual}"
            )
        if not isinstance(item["patient_id"], str) or not item["patient_id"]:
            raise ValueError(f"{sample_id}: patient_id 必须为非空字符串")
        uncertainty = item["uncertainty"]
        if not isinstance(uncertainty, (int, float)) or not math.isfinite(uncertainty):
            raise ValueError(f"{sample_id}: uncertainty 必须为有限数值")
        if uncertainty < 0:
            raise ValueError(f"{sample_id}: uncertainty 不能为负数")
        if item["correctness"] not in (0, 1):
            raise ValueError(f"{sample_id}: correctness 必须为 0 或 1")
        if phase == 1:
            if uncertainty > math.log(10) + 1e-5:
                raise ValueError(
                    f"{sample_id}: 10 类 Shannon entropy 超过 log(10)"
                )
            if item["correctness"] != int(item["predicted_disease"] == item["true_disease"]):
                raise ValueError(f"{sample_id}: phase 1 correctness 与预测/真值不一致")
            if not isinstance(item["source_dataset"], str) or not item["source_dataset"]:
                raise ValueError(f"{sample_id}: source_dataset 必须为非空字符串")
        else:
            # A routed phase-2 result can be wrong even when integer label values
            # happen to coincide: labels from a wrong specialist are not in the
            # source task's semantic space.  Cross-file validation below checks
            # the full route-aware correctness equation.
            if item["correctness"] == 1 and item["predicted"] != item["true"]:
                raise ValueError(f"{sample_id}: correctness=1 但预测与真值不一致")
            if not isinstance(item["label_scope"], str) or not item["label_scope"]:
                raise ValueError(f"{sample_id}: label_scope 必须为非空字符串")
            if "inference_model" in item and (
                not isinstance(item["inference_model"], str)
                or not item["inference_model"]
            ):
                raise ValueError(f"{sample_id}: inference_model 必须为非空字符串")


def atomic_write_json(path: Path, payload: dict[str, dict[str, Any]], phase: int) -> None:
    validate_payload(payload, phase)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def atomic_write_document(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def validate_json_file(path: Path, phase: int) -> None:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: JSON 顶层必须为对象")
    validate_payload(payload, phase)


def validate_result_set(
    directory: Path,
    disease_ids: dict[str, int],
    require_exact_files: bool = True,
    task_contracts: dict[str, tuple[str, int]] | None = None,
) -> None:
    expected_names = {"phase_1.json"} | {
        f"phase_2_{dataset}.json" for dataset in disease_ids
    }
    if require_exact_files:
        actual_names = {path.name for path in directory.glob("phase_*.json")}
        if actual_names != expected_names:
            raise ValueError(
                f"输出文件集合不匹配；missing={sorted(expected_names - actual_names)}, "
                f"extra={sorted(actual_names - expected_names)}"
            )
    phase1_path = directory / "phase_1.json"
    validate_json_file(phase1_path, 1)
    phase1 = json.loads(phase1_path.read_text(encoding="utf-8"))
    for dataset, disease_id in disease_ids.items():
        path = directory / f"phase_2_{dataset}.json"
        validate_json_file(path, 2)
        phase2 = json.loads(path.read_text(encoding="utf-8"))
        route_ids = {
            sample_id for sample_id, item in phase1.items()
            if item["source_dataset"] == dataset
        }
        if route_ids != set(phase2):
            raise ValueError(
                f"{dataset}: phase 1/2 样本集合不一致；"
                f"phase1_only={len(route_ids - set(phase2))}, "
                f"phase2_only={len(set(phase2) - route_ids)}"
            )
        for sample_id in route_ids:
            route_item = phase1[sample_id]
            specialist_item = phase2[sample_id]
            if route_item["patient_id"] != specialist_item["patient_id"]:
                raise ValueError(f"{sample_id}: phase 1/2 patient_id 不一致")
            if route_item["true_disease"] != disease_id:
                raise ValueError(f"{sample_id}: true_disease 与数据集映射不一致")
            expected_correctness = int(
                route_item["predicted_disease"] == route_item["true_disease"]
                and specialist_item["predicted"] == specialist_item["true"]
            )
            if specialist_item["correctness"] != expected_correctness:
                raise ValueError(
                    f"{sample_id}: phase 2 correctness 未按路由与专科预测联合计算"
                )
            if "inference_model" not in specialist_item:
                raise ValueError(f"{sample_id}: phase 2 缺少 inference_model")
            target_by_id = {value: key for key, value in disease_ids.items()}
            target = target_by_id.get(route_item["predicted_disease"])
            if target is None or not specialist_item["inference_model"].endswith(
                f"/{target}_adapter"
            ):
                raise ValueError(
                    f"{sample_id}: inference_model 与 phase 1 路由目标不一致"
                )
            if task_contracts is not None:
                task_type, num_classes = task_contracts[target]
                maximum = (
                    num_classes * math.log(2)
                    if task_type == "multilabel"
                    else math.log(num_classes)
                )
                if specialist_item["uncertainty"] > maximum + 1e-5:
                    raise ValueError(
                        f"{sample_id}: {target} uncertainty 超过论文熵定义上界 "
                        f"{maximum:.6f}"
                    )


def publish_staged_outputs(
    staging_dir: Path,
    output_dir: Path,
    names: set[str],
    archive_unexpected: bool = False,
) -> list[Path]:
    """Publish a validated set as one recoverable transaction.

    Existing files are moved to a timestamped archive.  If any replacement fails,
    every already-published file is moved back and the previous files are restored.
    """
    for name in names:
        source = staging_dir / name
        if not source.is_file():
            raise FileNotFoundError(f"staging 缺少输出: {source}")
        validate_json_file(source, 1 if name == "phase_1.json" else 2)

    output_dir.mkdir(parents=True, exist_ok=True)
    existing = [output_dir / name for name in sorted(names) if (output_dir / name).exists()]
    if archive_unexpected:
        existing.extend(
            path for path in output_dir.glob("phase_*.json")
            if path.name not in names
        )
    archive_dir = output_dir / "_previous" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    archived: list[Path] = []
    published: list[Path] = []
    try:
        if existing:
            archive_dir.mkdir(parents=True, exist_ok=False)
            for path in existing:
                destination = archive_dir / path.name
                shutil.move(str(path), destination)
                archived.append(destination)
        for name in sorted(names):
            destination = output_dir / name
            os.replace(staging_dir / name, destination)
            published.append(destination)
    except Exception:
        staging_dir.mkdir(parents=True, exist_ok=True)
        for path in reversed(published):
            if path.exists():
                shutil.move(str(path), staging_dir / path.name)
        for path in reversed(archived):
            if path.exists():
                shutil.move(str(path), output_dir / path.name)
        raise
    return archived

