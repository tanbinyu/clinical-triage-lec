from __future__ import annotations

import json
from pathlib import Path

from .config import DATASET_ORDER, PipelineConfig
from .output import validate_result_set


def evaluate_outputs(config: PipelineConfig) -> dict[str, object]:
    disease_ids = {name: config.datasets[name].disease_id for name in DATASET_ORDER}
    validate_result_set(
        config.output_dir,
        disease_ids,
        task_contracts={
            name: (
                config.datasets[name].task_type,
                config.datasets[name].num_classes,
            )
            for name in DATASET_ORDER
        },
    )
    phase1 = json.loads((config.output_dir / "phase_1.json").read_text(encoding="utf-8"))
    dataset_rows: list[dict[str, object]] = []
    total_route_correct = 0
    total_cascade_correct = 0
    total_samples = 0
    for dataset in DATASET_ORDER:
        phase2 = json.loads(
            (config.output_dir / f"phase_2_{dataset}.json").read_text(encoding="utf-8")
        )
        sample_ids = list(phase2)
        count = len(sample_ids)
        route_correct = sum(int(phase1[sample_id]["correctness"]) for sample_id in sample_ids)
        routed_correct = sum(int(phase2[sample_id]["correctness"]) for sample_id in sample_ids)
        dataset_rows.append({
            "dataset": dataset,
            "samples": count,
            "phase1_routing_accuracy": route_correct / count,
            "phase2_routed_accuracy": routed_correct / count,
            "specialist_accuracy_given_correct_route": (
                routed_correct / route_correct if route_correct else None
            ),
            "cascade_accuracy": routed_correct / count,
        })
        total_route_correct += route_correct
        total_cascade_correct += routed_correct
        total_samples += count
    return {
        "samples": total_samples,
        "phase1_routing_accuracy": total_route_correct / total_samples,
        "cascade_accuracy": total_cascade_correct / total_samples,
        "datasets": dataset_rows,
    }

