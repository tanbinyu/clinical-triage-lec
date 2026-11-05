from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DISEASE_TO_DATASETS: dict[int, list[str]] = {
    0: ["LungHist700"],
    1: ["BreakHis"],
    2: ["COVID_CT"],
    3: ["LDFC"],
    4: ["ImageCHD"],
    5: ["ChestX_ray14"],
    6: ["CBIS-DDSM"],
    7: ["BUS-BRA"],
    8: ["PU2756"],
    9: ["ACDC"],
}

DATASET_ORDER = tuple(v[0] for _, v in sorted(DISEASE_TO_DATASETS.items()))

ROUTER_PROMPTS: dict[str, str] = {
    "LungHist700": "a histopathology image of lung tissue",
    "BreakHis": "a histopathology image of breast tissue",
    "COVID_CT": "an axial lung CT image showing pneumonia or COVID-19",
    "LDFC": "an axial lung CT image showing a pulmonary nodule",
    "ImageCHD": "a cardiac CT image showing congenital heart anatomy",
    "ChestX_ray14": "a frontal chest X-ray image",
    "CBIS-DDSM": "a mammography image showing a breast lesion",
    "BUS-BRA": "a breast ultrasound image showing a breast lesion",
    "PU2756": "a lung ultrasound image showing pulmonary disease",
    "ACDC": "a cardiac cine MRI image",
}


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    disease_id: int
    root: Path
    prepared_root: Path
    backbone: str
    task_type: str
    num_classes: int
    label_scope: str
    class_names: tuple[str, ...]
    expected_samples: int | None = None
    expected_patients: int | None = None


@dataclass(frozen=True)
class PipelineConfig:
    output_dir: Path
    work_dir: Path
    weights_root: Path
    adapters_dir: Path
    router_source: Path
    router_checkpoint: Path
    router_text_encoder: Path
    device: str
    batch_size: int
    split_seed: int
    analysis_seed: int
    training_seed: int
    train_fraction: float
    val_fraction: float
    downstream_fraction: float
    cr_fraction: float
    cs_fraction: float
    evaluation_fraction: float
    max_downstream_patients_per_source: int
    deterministic_algorithms: bool
    multilabel_threshold: float
    volume_slices: int
    adapter_bottleneck: int
    strict_data_validation: bool
    preprocessing_version: str
    checkpoint_ids: dict[str, str]
    datasets: dict[str, DatasetConfig]


_DEFAULT_CLASSES: dict[str, tuple[str, ...]] = {
    "LungHist700": ("adenocarcinoma", "squamous_cell_carcinoma", "normal"),
    "BreakHis": ("benign", "malignant"),
    "COVID_CT": ("non_COVID", "COVID"),
    "LDFC": ("benign", "malignant"),
    "ImageCHD": (
        "ASD", "AVSD", "VSD", "TOF", "PDA", "TGA", "CA", "PuA",
        "PAS", "DORV", "CAT", "DAA", "APVC", "AAH", "IAA", "DSVC",
    ),
    "ChestX_ray14": (
        "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
        "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
        "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia",
    ),
    "CBIS-DDSM": ("benign", "malignant"),
    "BUS-BRA": ("benign", "malignant"),
    "PU2756": ("benign", "malignant"),
    "ACDC": ("NOR", "MINF", "DCM", "HCM", "RV"),
}

_DEFAULT_BACKBONES = {
    "LungHist700": "UNI",
    "BreakHis": "UNI",
    "COVID_CT": "DINOv2-large",
    "LDFC": "DINOv2-large",
    "ImageCHD": "DINOv2-large",
    "ChestX_ray14": "RAD-DINO",
    "CBIS-DDSM": "RAD-DINO",
    "BUS-BRA": "USFM",
    "PU2756": "USFM",
    "ACDC": "CineMA",
}

_DEFAULT_TASKS = {
    "ImageCHD": "multilabel",
    "ChestX_ray14": "multilabel",
}

_DEFAULT_SCOPES = {
    "LungHist700": "image",
    "BreakHis": "patient",
    "COVID_CT": "patient",
    "LDFC": "nodule",
    "ImageCHD": "volume",
    "ChestX_ray14": "image",
    "CBIS-DDSM": "lesion",
    "BUS-BRA": "patient",
    "PU2756": "patient",
    "ACDC": "patient",
}

# Frozen cohort counts reported by the completed paper.  Strict validation makes a
# release/mirror mismatch explicit instead of silently changing the study cohort.
_EXPECTED_COUNTS: dict[str, tuple[int | None, int | None]] = {
    "LungHist700": (691, 45),
    "BreakHis": (7909, 82),
    "COVID_CT": (812, 271),
    "LDFC": (308, 86),
    "ImageCHD": (110, 110),
    "ChestX_ray14": (112120, 30805),
    "CBIS-DDSM": (3103, 1566),
    "BUS-BRA": (1875, 1064),
    "PU2756": (2756, 2756),
    "ACDC": (150, 150),
}


def _resolve(base: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def load_config(path: str | Path) -> PipelineConfig:
    config_path = Path(path).resolve()
    with config_path.open("rb") as handle:
        raw: dict[str, Any] = tomllib.load(handle)
    base = config_path.parent
    paths = raw["paths"]
    runtime = raw.get("runtime", {})
    work_dir = _resolve(base, paths.get("work_dir", "../work"))

    roots = raw.get("dataset_roots", {})
    datasets: dict[str, DatasetConfig] = {}
    for disease_id, names in DISEASE_TO_DATASETS.items():
        name = names[0]
        if name not in roots:
            raise ValueError(f"配置缺少 [dataset_roots].{name}")
        classes = _DEFAULT_CLASSES[name]
        expected_samples, expected_patients = _EXPECTED_COUNTS[name]
        datasets[name] = DatasetConfig(
            name=name,
            disease_id=disease_id,
            root=_resolve(base, roots[name]),
            prepared_root=work_dir / "prepared" / name,
            backbone=_DEFAULT_BACKBONES[name],
            task_type=_DEFAULT_TASKS.get(name, "multiclass"),
            num_classes=len(classes),
            label_scope=_DEFAULT_SCOPES[name],
            class_names=classes,
            expected_samples=expected_samples,
            expected_patients=expected_patients,
        )

    train_fraction = float(runtime.get("train_fraction", 0.35))
    val_fraction = float(runtime.get("val_fraction", 0.05))
    downstream_fraction = float(runtime.get("downstream_fraction", 0.60))
    fractions = (train_fraction, val_fraction, downstream_fraction)
    if any(value <= 0.0 for value in fractions) or abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError(
            "runtime.train_fraction/val_fraction/downstream_fraction "
            "必须均大于 0 且总和为 1"
        )
    downstream_role_fractions = (
        float(runtime.get("cr_fraction", 0.30)),
        float(runtime.get("cs_fraction", 0.30)),
        float(runtime.get("evaluation_fraction", 0.40)),
    )
    if any(value <= 0.0 for value in downstream_role_fractions) or abs(
        sum(downstream_role_fractions) - 1.0
    ) > 1e-9:
        raise ValueError(
            "runtime.cr_fraction/cs_fraction/evaluation_fraction "
            "必须均大于 0 且总和为 1"
        )

    return PipelineConfig(
        output_dir=_resolve(base, paths["output_dir"]),
        work_dir=work_dir,
        weights_root=_resolve(base, paths["weights_root"]),
        adapters_dir=_resolve(base, paths["adapters_dir"]),
        router_source=_resolve(base, paths["router_source"]),
        router_checkpoint=_resolve(base, paths["router_checkpoint"]),
        router_text_encoder=_resolve(base, paths["router_text_encoder"]),
        device=str(runtime.get("device", "auto")),
        batch_size=int(runtime.get("batch_size", 16)),
        split_seed=int(runtime.get("split_seed", 20261019)),
        analysis_seed=int(runtime.get("analysis_seed", 20260929)),
        training_seed=int(runtime.get("training_seed", 20261019)),
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        downstream_fraction=downstream_fraction,
        cr_fraction=downstream_role_fractions[0],
        cs_fraction=downstream_role_fractions[1],
        evaluation_fraction=downstream_role_fractions[2],
        max_downstream_patients_per_source=int(
            runtime.get("max_downstream_patients_per_source", 1850)
        ),
        deterministic_algorithms=bool(runtime.get("deterministic_algorithms", True)),
        multilabel_threshold=float(runtime.get("multilabel_threshold", 0.5)),
        volume_slices=int(runtime.get("volume_slices", 8)),
        adapter_bottleneck=int(runtime.get("adapter_bottleneck", 256)),
        strict_data_validation=bool(runtime.get("strict_data_validation", True)),
        preprocessing_version=str(runtime.get("preprocessing_version", "v1")),
        checkpoint_ids={
            str(key): str(value) for key, value in raw.get("checkpoint_ids", {}).items()
        },
        datasets=datasets,
    )

