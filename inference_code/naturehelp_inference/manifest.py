from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Iterable

from .records import MediaLocator, SampleRecord, assign_sample_ids


MANIFEST_VERSION = 2


def record_to_dict(record: SampleRecord) -> dict[str, object]:
    return {
        "version": MANIFEST_VERSION,
        "sample_id": record.sample_id,
        "dataset": record.dataset,
        "patient_id": record.patient_id,
        "target": list(record.target) if isinstance(record.target, tuple) else record.target,
        "label_scope": record.label_scope,
        "source_id": record.source_id,
        "media_fingerprint": record.media_fingerprint,
        "locator": {
            "kind": record.locator.kind,
            "path": str(record.locator.path),
            "member": record.locator.member,
            "extra": record.locator.extra,
        },
    }


def record_from_dict(item: dict[str, object]) -> SampleRecord:
    if item.get("version") != MANIFEST_VERSION:
        raise ValueError(f"不支持的 manifest 版本: {item.get('version')!r}")
    locator = item["locator"]
    if not isinstance(locator, dict):
        raise ValueError("manifest locator 必须是对象")
    target = item["target"]
    if isinstance(target, list):
        target = tuple(int(value) for value in target)
    return SampleRecord(
        dataset=str(item["dataset"]),
        patient_id=str(item["patient_id"]),
        target=int(target) if not isinstance(target, tuple) else target,
        label_scope=str(item["label_scope"]),
        locator=MediaLocator(
            kind=str(locator["kind"]),  # type: ignore[arg-type]
            path=Path(str(locator["path"])),
            member=None if locator.get("member") is None else str(locator["member"]),
            extra=dict(locator.get("extra", {})),
        ),
        source_id=str(item["source_id"]),
        sample_id=None if item.get("sample_id") is None else str(item["sample_id"]),
        media_fingerprint=(
            None if item.get("media_fingerprint") is None else str(item["media_fingerprint"])
        ),
    )


def manifest_digest(records: Iterable[SampleRecord]) -> str:
    digest = hashlib.sha256()
    for record in records:
        encoded = json.dumps(
            record_to_dict(record), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        digest.update(encoded)
        digest.update(b"\n")
    return digest.hexdigest()


def write_manifest(path: Path, records: list[SampleRecord]) -> str:
    assigned = assign_sample_ids(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            for record in assigned:
                json.dump(record_to_dict(record), handle, ensure_ascii=False, sort_keys=True)
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
    return manifest_digest(assigned)


def read_manifest(path: Path) -> list[SampleRecord]:
    records: list[SampleRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(record_from_dict(json.loads(line)))
            except Exception as exc:
                raise ValueError(f"manifest 格式错误 {path}:{line_number}: {exc}") from exc
    if not records:
        raise ValueError(f"manifest 为空: {path}")
    ids = [record.sample_id for record in records]
    if any(value is None for value in ids) or len(ids) != len(set(ids)):
        raise ValueError(f"manifest sample_id 缺失或重复: {path}")
    return records


def freeze_manifests(
    manifest_dir: Path, manifests: dict[str, list[SampleRecord]]
) -> dict[str, list[SampleRecord]]:
    """Persist canonical JSONL files and return records carrying stable IDs."""
    result: dict[str, list[SampleRecord]] = {}
    for dataset, records in manifests.items():
        path = manifest_dir / f"{dataset}.jsonl"
        write_manifest(path, records)
        result[dataset] = read_manifest(path)
    return result

