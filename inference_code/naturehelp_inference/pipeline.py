from __future__ import annotations

import hashlib
import json
import os
import random
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4

from .config import DATASET_ORDER, PipelineConfig
from .media import MediaReader
from .manifest import manifest_digest
from .models import (
    CineMAExtractor,
    UniMedRouter,
    create_adapter_head,
    create_extractor,
    resolve_device,
)
from .output import (
    atomic_write_document, atomic_write_json, publish_staged_outputs, validate_result_set,
)
from .progress import ProgressStore
from .records import SampleRecord
from .splits import build_or_load_split, records_for_split, split_digest


def _configure_determinism(config: PipelineConfig) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(config.split_seed)
    try:
        import numpy as np
        np.random.seed(config.split_seed)
    except ImportError:
        pass
    torch = __import__("torch")
    torch.manual_seed(config.split_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.split_seed)
    if config.deterministic_algorithms:
        torch.use_deterministic_algorithms(True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True


def _chunks(items: list[Any], size: int):
    for start in range(0, len(items), size):
        yield items[start:start + size]


def _mean_by_lengths(tensor: Any, lengths: list[int]):
    result = []
    offset = 0
    for length in lengths:
        result.append(tensor[offset:offset + length].mean(dim=0, keepdim=True))
        offset += length
    return result


def _entropy(probabilities: Any) -> float:
    values = probabilities.detach().float().clamp_min(1e-12)
    return round(float(-(values * values.log()).sum().cpu().item()), 6)


def _binary_entropy(probabilities: Any) -> float:
    values = probabilities.detach().float().clamp(1e-7, 1.0 - 1e-7)
    entropy = -(values * values.log() + (1.0 - values) * (1.0 - values).log()).sum()
    return round(float(entropy.cpu().item()), 6)


def _sample_id(record: SampleRecord) -> str:
    if record.sample_id is None:
        raise RuntimeError(f"{record.dataset}/{record.source_id}: manifest 缺少稳定 sample_id")
    return record.sample_id


def _path_token(path: Path) -> str:
    if not path.exists():
        return f"{path}|missing"
    stat = path.stat()
    return f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}"


def _progress_identity(
    config: PipelineConfig,
    phase: int,
    records: list[SampleRecord],
    dataset: str | None = None,
    routes: dict[str, dict[str, Any]] | None = None,
) -> str:
    model_files = (
        [
            config.router_checkpoint,
            config.router_text_encoder / "config.json",
            config.router_source / "src" / "open_clip" / "model.py",
        ]
        if phase == 1 else []
    )
    if phase == 2 and dataset is not None:
        if routes is None:
            raise ValueError("phase 2 progress identity 需要 phase 1 routes")
        target_ids = {
            int(routes[_sample_id(record)]["predicted_disease"])
            for record in records
        }
        target_names = [DATASET_ORDER[index] for index in sorted(target_ids)]
        for target in target_names:
            model_files.append(config.adapters_dir / f"{target}.pt")
            backbone = config.datasets[target].backbone
            root = config.weights_root / backbone
            candidates = {
                "UNI": [root / "pytorch_model.bin", root / "model.safetensors", root / "uni.bin", root / "UNI.pth"],
                "DINOv2-large": [root / "model.safetensors", root / "config.json"],
                "RAD-DINO": [root / "model.safetensors", root / "config.json"],
                "USFM": [
                    root / "USFM_latest.pth",
                    root / "source" / "usdsgen" / "modules" / "backbone" / "vision_transformer.py",
                ],
                "CineMA": [
                    root / "pretrained" / "cinema.safetensors",
                    root / "pretrained" / "config.yaml",
                    root / "source" / "cinema" / "mae" / "mae.py",
                ],
            }
            model_files.extend(path for path in candidates[backbone] if path.exists())
        route_digest = hashlib.sha256(json.dumps(
            {
                _sample_id(record): routes[_sample_id(record)]["predicted_disease"]
                for record in records
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")).hexdigest()
    else:
        route_digest = None
    payload = {
        "phase": phase,
        "dataset": dataset,
        "preprocessing_version": config.preprocessing_version,
        "manifest_digest": manifest_digest(records),
        "models": [_path_token(path) for path in model_files],
        "route_digest": route_digest,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _selected_manifests(
    config: PipelineConfig,
    manifests: dict[str, list[SampleRecord]],
    scope: str,
) -> dict[str, list[SampleRecord]]:
    result: dict[str, list[SampleRecord]] = {}
    for name, records in manifests.items():
        if scope == "all":
            selected = records
        else:
            assignments = build_or_load_split(config, name, records)
            selected = records_for_split(records, assignments, scope)
        if not selected:
            raise RuntimeError(f"{name}: scope={scope} 后没有样本")
        result[name] = selected
    return result


def run_phase1(
    config: PipelineConfig,
    manifests: dict[str, list[SampleRecord]],
    output_path: Path | None = None,
    progress_path: Path | None = None,
    progress_identity: str | None = None,
) -> dict[str, dict[str, Any]]:
    device = resolve_device(config.device)
    router = UniMedRouter(config, device)
    progress_context = (
        ProgressStore(progress_path, progress_identity or "", phase=1)
        if progress_path is not None else nullcontext(None)
    )
    with progress_context as progress, MediaReader(config.volume_slices) as reader:
        payload = progress.load() if progress is not None else {}
        expected_ids = {_sample_id(record) for records in manifests.values() for record in records}
        unexpected = set(payload) - expected_ids
        if unexpected:
            raise RuntimeError(f"phase 1 进度包含当前运行之外的样本: {sorted(unexpected)[:5]}")
        for dataset, records in manifests.items():
            true_disease = config.datasets[dataset].disease_id
            pending = [record for record in records if _sample_id(record) not in payload]
            for base_index in range(0, len(pending), config.batch_size):
                chunk = pending[base_index:base_index + config.batch_size]
                images: list[Any] = []
                lengths: list[int] = []
                for record in chunk:
                    record_images = reader.images_for_record(record, phase2=False)
                    images.extend(record_images)
                    lengths.append(len(record_images))
                probabilities: list[Any] = []
                for image_chunk in _chunks(images, config.batch_size):
                    probabilities.append(router.predict(image_chunk))
                torch = __import__("torch")
                per_image = torch.cat(probabilities, dim=0)
                per_record = _mean_by_lengths(per_image, lengths)
                completed: dict[str, dict[str, Any]] = {}
                for record, probs in zip(chunk, per_record, strict=True):
                    predicted = int(probs.argmax(dim=-1).item())
                    sample_id = _sample_id(record)
                    completed[sample_id] = {
                        "uncertainty": _entropy(probs[0]),
                        "correctness": int(predicted == true_disease),
                        "predicted_disease": predicted,
                        "true_disease": true_disease,
                        "source_dataset": dataset,
                        "patient_id": record.patient_id,
                    }
                payload.update(completed)
                if progress is not None:
                    progress.put_many(completed)
            print(f"[phase 1] {dataset}: {len(records)} samples", flush=True)
    ordered_records = [record for records in manifests.values() for record in records]
    payload = {_sample_id(record): payload[_sample_id(record)] for record in ordered_records}
    target = output_path or config.output_dir / "phase_1.json"
    atomic_write_json(target, payload, phase=1)
    return payload


def _encode_records(
    extractor: Any,
    records: list[SampleRecord],
    reader: MediaReader,
    batch_size: int,
):
    torch = __import__("torch")
    if isinstance(extractor, CineMAExtractor):
        return torch.cat([
            extractor.encode_volume(reader.acdc_tensor(record)) for record in records
        ], dim=0)
    images: list[Any] = []
    lengths: list[int] = []
    for record in records:
        record_images = reader.images_for_record(record, phase2=True)
        images.extend(record_images)
        lengths.append(len(record_images))
    encoded = torch.cat([
        extractor.encode_images(image_chunk) for image_chunk in _chunks(images, batch_size)
    ], dim=0)
    return torch.cat(_mean_by_lengths(encoded, lengths), dim=0)


def run_phase2_dataset(
    config: PipelineConfig,
    dataset: str,
    records: list[SampleRecord],
    routes: dict[str, dict[str, Any]],
    output_path: Path | None = None,
    provenance_manifests: dict[str, list[SampleRecord]] | None = None,
    progress_path: Path | None = None,
    progress_identity: str | None = None,
) -> dict[str, dict[str, Any]]:
    torch = __import__("torch")
    device = resolve_device(config.device)
    route_by_id = {config.datasets[name].disease_id: name for name in DATASET_ORDER}
    expected_ids = {_sample_id(record) for record in records}
    missing_routes = expected_ids - set(routes)
    if missing_routes:
        raise RuntimeError(
            f"{dataset}: phase 1 缺少 {len(missing_routes)} 个样本，"
            f"例如 {sorted(missing_routes)[:5]}"
        )
    for record in records:
        item = routes[_sample_id(record)]
        if item["source_dataset"] != dataset or item["patient_id"] != record.patient_id:
            raise RuntimeError(f"{_sample_id(record)}: phase 1 与 manifest 不一致")

    progress_context = (
        ProgressStore(progress_path, progress_identity or "", phase=2)
        if progress_path is not None else nullcontext(None)
    )
    with progress_context as progress, MediaReader(config.volume_slices) as reader:
        payload = progress.load() if progress is not None else {}
        unexpected = set(payload) - expected_ids
        if unexpected:
            raise RuntimeError(f"{dataset} 进度包含当前运行之外的样本: {sorted(unexpected)[:5]}")
        pending = [record for record in records if _sample_id(record) not in payload]
        already_completed = len(records) - len(pending)
        grouped: dict[str, list[SampleRecord]] = {}
        for record in pending:
            predicted_id = int(routes[_sample_id(record)]["predicted_disease"])
            if predicted_id not in route_by_id:
                raise RuntimeError(
                    f"{_sample_id(record)}: predicted_disease={predicted_id} 越界"
                )
            grouped.setdefault(route_by_id[predicted_id], []).append(record)

        completed_count = already_completed
        for target_dataset in DATASET_ORDER:
            target_records = grouped.get(target_dataset, [])
            if not target_records:
                continue
            target_spec = config.datasets[target_dataset]
            extractor = create_extractor(config, target_spec, device)
            adapter = create_adapter_head(
                config, target_spec, extractor.feature_dim, device
            )
            metadata = getattr(adapter, "naturehelp_metadata", {})
            if provenance_manifests is not None and target_dataset in provenance_manifests:
                target_manifest = provenance_manifests[target_dataset]
                target_assignments = build_or_load_split(
                    config, target_dataset, target_manifest
                )
                expected_provenance = {
                    "manifest_digest": manifest_digest(target_manifest),
                    "split_digest": split_digest(target_assignments),
                }
                for key, expected in expected_provenance.items():
                    if metadata.get(key) != expected:
                        raise RuntimeError(
                            f"{target_dataset}: adapter {key}={metadata.get(key)!r}，"
                            f"当前数据要求 {expected!r}"
                        )
            configured_thresholds = metadata.get("multilabel_thresholds")
            if configured_thresholds is not None:
                if (
                    not isinstance(configured_thresholds, list)
                    or len(configured_thresholds) != target_spec.num_classes
                ):
                    raise RuntimeError(
                        f"{target_dataset}: adapter 中 multilabel_thresholds 长度错误"
                    )
                thresholds = torch.tensor(configured_thresholds, device=device)
            else:
                thresholds = torch.full(
                    (target_spec.num_classes,),
                    config.multilabel_threshold,
                    device=device,
                )

            for base_index in range(0, len(target_records), config.batch_size):
                chunk = target_records[base_index:base_index + config.batch_size]
                if isinstance(extractor, CineMAExtractor) and any(
                    record.locator.kind != "acdc_cine" for record in chunk
                ):
                    offender = next(
                        record for record in chunk if record.locator.kind != "acdc_cine"
                    )
                    raise RuntimeError(
                        f"{_sample_id(offender)} 被路由到 ACDC/CineMA，但原始输入不是 "
                        "ACDC cine MRI。CineMA 不接受伪造的 2D->cine 转换；"
                        "请核查路由器或在论文中定义可验证的跨模态拒绝策略。"
                    )
                features = _encode_records(
                    extractor, chunk, reader, config.batch_size
                )
                with torch.inference_mode():
                    logits = adapter(features)
                completed: dict[str, dict[str, Any]] = {}
                if target_spec.task_type == "multilabel":
                    probabilities = logits.sigmoid()
                    for record, probs in zip(chunk, probabilities, strict=True):
                        predicted_tuple = tuple(
                            int(index)
                            for index in torch.where(probs >= thresholds)[0].tolist()
                        )
                        truth = (
                            list(record.target)
                            if isinstance(record.target, tuple) else record.target
                        )
                        predicted: int | list[int] = list(predicted_tuple)
                        route_correct = target_dataset == dataset
                        completed[_sample_id(record)] = {
                            "uncertainty": _binary_entropy(probs),
                            "correctness": int(route_correct and predicted == truth),
                            "predicted": predicted,
                            "true": truth,
                            "patient_id": record.patient_id,
                            "label_scope": record.label_scope,
                            "inference_model": (
                                f"{target_spec.backbone}/{target_dataset}_adapter"
                            ),
                        }
                else:
                    probabilities = logits.softmax(dim=-1)
                    for record, probs in zip(chunk, probabilities, strict=True):
                        predicted = int(probs.argmax(dim=-1).item())
                        truth = (
                            list(record.target)
                            if isinstance(record.target, tuple) else record.target
                        )
                        route_correct = target_dataset == dataset
                        completed[_sample_id(record)] = {
                            "uncertainty": _entropy(probs),
                            "correctness": int(route_correct and predicted == truth),
                            "predicted": predicted,
                            "true": truth,
                            "patient_id": record.patient_id,
                            "label_scope": record.label_scope,
                            "inference_model": (
                                f"{target_spec.backbone}/{target_dataset}_adapter"
                            ),
                        }
                payload.update(completed)
                if progress is not None:
                    progress.put_many(completed)
                completed_count += len(chunk)
                print(
                    f"[phase 2] {dataset} -> {target_dataset}: "
                    f"{completed_count}/{len(records)}",
                    flush=True,
                )
            del adapter, extractor
            if device.type == "cuda":
                torch.cuda.empty_cache()
        if not pending:
            print(f"[phase 2] {dataset}: resumed {len(records)}/{len(records)}", flush=True)
    payload = {_sample_id(record): payload[_sample_id(record)] for record in records}
    target = output_path or config.output_dir / f"phase_2_{dataset}.json"
    atomic_write_json(target, payload, phase=2)
    return payload


def run_pipeline(
    config: PipelineConfig,
    manifests: dict[str, list[SampleRecord]],
    scope: str = "all",
    phases: Iterable[int] = (1, 2),
    resume: bool = True,
) -> None:
    _configure_determinism(config)
    selected = _selected_manifests(config, manifests, scope)
    phase_set = set(phases)
    if 1 in phase_set and set(manifests) != set(DATASET_ORDER):
        raise ValueError("phase_1.json 是全局路由结果，生成它时必须包含全部 10 个数据集")
    staging_dir = config.work_dir / "runs" / f"staging_{uuid4().hex}"
    staging_dir.mkdir(parents=True, exist_ok=False)
    generated: set[str] = set()
    phase1_payload: dict[str, dict[str, Any]] | None = None
    if 1 in phase_set:
        phase1_records = [record for records in selected.values() for record in records]
        identity = _progress_identity(config, 1, phase1_records)
        progress_path = (
            config.work_dir / "runs" / "progress" / f"phase_1_{identity[:20]}.sqlite"
            if resume else None
        )
        phase1_payload = run_phase1(
            config, selected, staging_dir / "phase_1.json", progress_path, identity
        )
        generated.add("phase_1.json")
    if 2 in phase_set:
        if phase1_payload is None:
            phase1_path = config.output_dir / "phase_1.json"
            if not phase1_path.is_file():
                raise RuntimeError(
                    "phase 2 必须读取 phase 1 路由结果；先生成 phase_1.json，"
                    "或在同一次运行中指定 --phase all"
                )
            phase1_payload = json.loads(phase1_path.read_text(encoding="utf-8"))
        for dataset, records in selected.items():
            name = f"phase_2_{dataset}.json"
            identity = _progress_identity(
                config, 2, records, dataset, routes=phase1_payload
            )
            progress_path = (
                config.work_dir / "runs" / "progress" /
                f"phase_2_{dataset}_{identity[:20]}.sqlite"
                if resume else None
            )
            run_phase2_dataset(
                config, dataset, records, phase1_payload, staging_dir / name,
                provenance_manifests=manifests,
                progress_path=progress_path,
                progress_identity=identity,
            )
            generated.add(name)
    full_run = phase_set == {1, 2} and set(manifests) == set(DATASET_ORDER)
    if full_run:
        validate_result_set(
            staging_dir,
            {name: config.datasets[name].disease_id for name in DATASET_ORDER},
            require_exact_files=True,
            task_contracts={
                name: (
                    config.datasets[name].task_type,
                    config.datasets[name].num_classes,
                )
                for name in DATASET_ORDER
            },
        )
    archived = publish_staged_outputs(
        staging_dir, config.output_dir, generated, archive_unexpected=full_run
    )
    for path in archived:
        print(f"archived previous output: {path}", flush=True)
    atomic_write_document(config.work_dir / "runs" / "latest_run.json", {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": scope,
        "phases": sorted(phase_set),
        "resume_enabled": resume,
        "seeds": {
            "fixed_patient_split": config.split_seed,
            "adapter_training": config.training_seed,
            "repeated_downstream_partition": config.analysis_seed,
        },
        "checkpoint_ids": config.checkpoint_ids,
        "output_dir": str(config.output_dir),
        "generated_files": sorted(generated),
        "datasets": {
            dataset: {
                "samples": len(selected[dataset]),
                "patients": len({record.patient_id for record in selected[dataset]}),
                "full_manifest_digest": manifest_digest(manifests[dataset]),
            }
            for dataset in selected
        },
    })

