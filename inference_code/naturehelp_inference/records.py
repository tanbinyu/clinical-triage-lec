from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

Target = int | tuple[int, ...]
LocatorKind = Literal[
    "file", "zip", "tar", "dicom", "nifti", "acdc_cine", "volume_dir"
]


@dataclass(frozen=True)
class MediaLocator:
    kind: LocatorKind
    path: Path
    member: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def stable_key(self) -> str:
        return f"{self.kind}|{self.path.as_posix()}|{self.member or ''}"


@dataclass(frozen=True)
class SampleRecord:
    dataset: str
    patient_id: str
    target: Target
    label_scope: str
    locator: MediaLocator
    source_id: str
    sample_id: str | None = None
    media_fingerprint: str | None = None

    def __post_init__(self) -> None:
        if not self.patient_id.strip():
            raise ValueError(f"{self.dataset}/{self.source_id}: patient_id 为空")
        if not self.source_id.strip():
            raise ValueError(f"{self.dataset}: source_id 为空")
        if self.sample_id is not None and not self.sample_id.startswith(f"sample_{self.dataset}_"):
            raise ValueError(f"{self.dataset}/{self.source_id}: 非法 sample_id {self.sample_id!r}")
        if isinstance(self.target, tuple):
            if tuple(sorted(set(self.target))) != self.target:
                raise ValueError(f"{self.dataset}/{self.source_id}: 多标签必须去重并升序")
        elif not isinstance(self.target, int):
            raise TypeError(f"{self.dataset}/{self.source_id}: 非法标签 {self.target!r}")


def stable_sort(records: list[SampleRecord]) -> list[SampleRecord]:
    return sorted(records, key=lambda r: (r.patient_id, r.source_id, r.locator.stable_key))


def assign_sample_ids(records: list[SampleRecord]) -> list[SampleRecord]:
    """Assign IDs once on the complete, deterministically sorted manifest."""
    ordered = stable_sort(records)
    return [
        replace(record, sample_id=f"sample_{record.dataset}_{index:05d}")
        for index, record in enumerate(ordered)
    ]


def attach_media_fingerprints(records: list[SampleRecord]) -> list[SampleRecord]:
    """Bind manifests to file/container size and nanosecond modification time."""
    stat_cache: dict[Path, str] = {}

    def fingerprint(path: Path) -> str:
        if path not in stat_cache:
            resolved = path.resolve()
            stat = resolved.stat()
            stat_cache[path] = f"{resolved.as_posix()}|{stat.st_size}|{stat.st_mtime_ns}"
        return stat_cache[path]

    result: list[SampleRecord] = []
    for record in records:
        parts = [fingerprint(record.locator.path), record.locator.member or ""]
        for key, value in sorted(record.locator.extra.items()):
            candidate = Path(value) if isinstance(value, str) else None
            if candidate is not None and candidate.exists():
                parts.append(f"{key}={fingerprint(candidate)}")
            else:
                parts.append(f"{key}={value!r}")
        value = hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()
        result.append(replace(record, media_fingerprint=value))
    return result

