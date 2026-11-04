from __future__ import annotations

import importlib.util
import importlib.metadata
import platform
from typing import Iterable

from .config import PipelineConfig
from .datasets import build_manifest, summarize_manifest
from .models import verify_weight_layout
from .provenance import DEPENDENCY_DISTRIBUTIONS


COMMON_DEPENDENCIES = ("numpy", "PIL", "openpyxl")
INFERENCE_DEPENDENCIES = (
    "torch", "torchvision", "transformers", "timm", "safetensors", "scipy",
    "nibabel", "pydicom", "omegaconf", "einops", "monai", "ftfy", "regex",
    "pandas", "SimpleITK", "tqdm", "hydra",
)


def dependency_report(include_inference: bool = True) -> dict[str, object]:
    modules = COMMON_DEPENDENCIES + (INFERENCE_DEPENDENCIES if include_inference else ())
    missing = [name for name in modules if importlib.util.find_spec(name) is None]
    versions: dict[str, str | None] = {}
    for name in DEPENDENCY_DISTRIBUTIONS:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "python": platform.python_version(),
        "versions": versions,
        "required_modules": list(modules),
        "missing_modules": missing,
        "ok": not missing,
    }


def doctor_report(
    config: PipelineConfig,
    names: Iterable[str],
    require_adapters: bool = False,
) -> dict[str, object]:
    selected = list(names)
    manifests: dict[str, list] = {}
    dataset_errors: list[dict[str, str]] = []
    for name in selected:
        try:
            manifests.update(build_manifest(config, [name]))
        except Exception as exc:
            dataset_errors.append({"dataset": name, "error": str(exc)})
    weight_errors = verify_weight_layout(
        config, selected, require_adapters=require_adapters
    )
    dependencies = dependency_report(include_inference=True)
    return {
        "ok": not dataset_errors and not weight_errors and dependencies["ok"],
        "dependencies": dependencies,
        "datasets": summarize_manifest(manifests),
        "dataset_errors": dataset_errors,
        "weight_errors": weight_errors,
        "checkpoint_ids": config.checkpoint_ids,
    }

