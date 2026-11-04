from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Iterable

from .config import PipelineConfig


DEPENDENCY_DISTRIBUTIONS = (
    "torch", "torchvision", "transformers", "timm", "safetensors", "Pillow",
    "numpy", "scipy", "openpyxl", "nibabel", "pydicom", "omegaconf",
    "hydra-core", "einops", "monai", "ftfy", "regex", "pandas",
    "SimpleITK", "tqdm",
)


def dependency_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in DEPENDENCY_DISTRIBUTIONS:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(source: Path) -> str | None:
    if not (source / ".git").exists():
        return None
    try:
        return subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"],
            text=True,
            encoding="utf-8",
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def checkpoint_files(config: PipelineConfig) -> dict[str, list[Path]]:
    root = config.weights_root
    uni = root / "UNI"
    uni_candidates = [
        uni / "pytorch_model.bin", uni / "model.safetensors",
        uni / "uni.bin", uni / "UNI.pth",
    ]
    return {
        "UniMed-CLIP": [config.router_checkpoint],
        "UNI": [path for path in uni_candidates if path.exists()],
        "DINOv2-large": [root / "DINOv2-large" / "model.safetensors"],
        "RAD-DINO": [root / "RAD-DINO" / "model.safetensors"],
        "USFM": [root / "USFM" / "USFM_latest.pth"],
        "CineMA": [root / "CineMA" / "pretrained" / "cinema.safetensors"],
    }


def reproducibility_snapshot(
    config: PipelineConfig,
    datasets: Iterable[str],
    hash_checkpoints: bool = False,
) -> dict[str, object]:
    files = checkpoint_files(config)
    checkpoints: dict[str, object] = {}
    for model, paths in files.items():
        entries = []
        for path in paths:
            entry: dict[str, object] = {
                "path": str(path),
                "exists": path.is_file(),
            }
            if path.is_file():
                stat = path.stat()
                entry["bytes"] = stat.st_size
                if hash_checkpoints:
                    entry["sha256"] = _sha256(path)
            entries.append(entry)
        source = config.weights_root / model / "source"
        checkpoints[model] = {
            "checkpoint_id": config.checkpoint_ids.get(model),
            "source_commit": _git_commit(source),
            "files": entries,
        }

    adapters: dict[str, object] = {}
    for dataset in datasets:
        path = config.adapters_dir / f"{dataset}.pt"
        entry: dict[str, object] = {"path": str(path), "exists": path.is_file()}
        if path.is_file():
            stat = path.stat()
            entry["bytes"] = stat.st_size
            if hash_checkpoints:
                entry["sha256"] = _sha256(path)
        adapters[dataset] = entry

    return {
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
            "platform": platform.platform(),
        },
        "dependencies": dependency_versions(),
        "seeds": {
            "fixed_patient_split": config.split_seed,
            "adapter_training": config.training_seed,
            "repeated_downstream_partition": config.analysis_seed,
        },
        "deterministic_algorithms": config.deterministic_algorithms,
        "preprocessing_version": config.preprocessing_version,
        "checkpoints": checkpoints,
        "adapters": adapters,
    }


def write_reproducibility_snapshot(
    config: PipelineConfig,
    datasets: Iterable[str],
    path: Path,
    hash_checkpoints: bool = False,
) -> Path:
    from .output import atomic_write_document

    payload = reproducibility_snapshot(
        config, datasets, hash_checkpoints=hash_checkpoints
    )
    atomic_write_document(path, payload)
    return path
