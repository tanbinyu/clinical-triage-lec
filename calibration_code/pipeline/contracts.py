"""Explicit experiment parameters and JSON input validation."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from pipeline.core.config import DEFAULTS


@dataclass(frozen=True)
class ExperimentConfig:
    """Parameters needed to reproduce one Step 2–6 experiment."""

    phase1: Path
    phase2_dir: Path
    output_dir: Path
    num_splits: int = DEFAULTS["num_splits"]
    split_ratio: float = DEFAULTS["split_ratio"]
    beta: float = DEFAULTS["beta"]
    alpha_A: float = DEFAULTS["alpha_A"]
    alpha_B: float = DEFAULTS["alpha_B"]
    lambda_1: float = DEFAULTS["lambda_1"]
    lambda_2: float = DEFAULTS["lambda_2"]
    steps: tuple[int, ...] = (2, 3, 4, 5, 6)

    @classmethod
    def from_mapping(cls, values, base_dir):
        """Load a config mapping; relative paths are relative to ``base_dir``."""
        base_dir = Path(base_dir)

        def path_value(name, default):
            value = Path(values.get(name, default))
            return value if value.is_absolute() else base_dir / value

        return cls(
            phase1=path_value("phase1", "result/json/step1/phase_1.json"),
            phase2_dir=path_value("phase2_dir", "result/json/step1"),
            output_dir=path_value("output_dir", "result/json"),
            num_splits=int(values.get("num_splits", DEFAULTS["num_splits"])),
            split_ratio=float(values.get("split_ratio", DEFAULTS["split_ratio"])),
            beta=float(values.get("beta", DEFAULTS["beta"])),
            alpha_A=float(values.get("alpha_A", DEFAULTS["alpha_A"])),
            alpha_B=float(values.get("alpha_B", DEFAULTS["alpha_B"])),
            lambda_1=float(values.get("lambda_1", DEFAULTS["lambda_1"])),
            lambda_2=float(values.get("lambda_2", DEFAULTS["lambda_2"])),
            steps=tuple(int(step) for step in values.get("steps", (2, 3, 4, 5, 6))),
        )


def load_experiment_config(path, base_dir):
    """Read a JSON configuration file without adding third-party dependencies."""
    with open(path, encoding="utf-8") as f:
        values = json.load(f)
    return ExperimentConfig.from_mapping(values, base_dir)


def load_and_validate_phase1(config):
    """Validate the fields and phase-2 file presence required by the workflow."""
    if not config.phase1.is_file():
        raise FileNotFoundError(f"phase_1 input not found: {config.phase1}")
    if not config.phase2_dir.is_dir():
        raise FileNotFoundError(f"phase_2 directory not found: {config.phase2_dir}")
    if not config.steps or any(step not in (2, 3, 4, 5, 6) for step in config.steps):
        raise ValueError("steps must be a non-empty subset of 2, 3, 4, 5, 6")

    with open(config.phase1) as f:
        phase1 = json.load(f)

    required = {
        "uncertainty",
        "correctness",
        "predicted_disease",
        "true_disease",
        "source_dataset",
    }
    missing = [
        sample_id
        for sample_id, sample in phase1.items()
        if not required.issubset(sample)
    ]
    if missing:
        raise ValueError(
            "phase_1 samples are missing required fields; "
            f"first sample: {missing[0]}"
        )

    if 5 in config.steps:
        missing_phase2 = sorted({
            sample["source_dataset"]
            for sample in phase1.values()
            if not (config.phase2_dir / f"phase_2_{sample['source_dataset']}.json").is_file()
        })
        if missing_phase2:
            raise FileNotFoundError(
                "phase_2 files not found for: " + ", ".join(missing_phase2)
            )

    return phase1
