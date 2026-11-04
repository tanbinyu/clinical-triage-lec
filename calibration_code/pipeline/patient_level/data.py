"""Load, validate and aggregate record-level outputs to patient decisions."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .metadata import DATASET_TO_DISEASE, NUM_DISEASES


@dataclass(frozen=True)
class PatientScores:
    """One row per patient after deterministic record aggregation."""

    patient_ids: tuple[str, ...]
    sources: tuple[str, ...]
    patient_id_types: tuple[str, ...]
    record_counts: np.ndarray
    router_uncertainty: np.ndarray
    router_correctness: np.ndarray
    predicted_disease: np.ndarray
    true_disease: np.ndarray
    specialist_uncertainty: np.ndarray
    specialist_correctness: np.ndarray
    system_correctness: np.ndarray
    source_indices: dict[str, np.ndarray]
    aggregation: str
    lambda_1: float
    lambda_2: float
    specialist_only_uncertainty: np.ndarray | None = None
    specialist_only_correctness: np.ndarray | None = None

    @property
    def n_patients(self) -> int:
        return len(self.patient_ids)

    @property
    def n_records(self) -> int:
        return int(self.record_counts.sum())

    @property
    def fused_uncertainty(self) -> np.ndarray:
        return (
            self.lambda_1 * self.router_uncertainty
            + self.lambda_2 * self.specialist_uncertainty
        )

    @property
    def specialist_score(self) -> np.ndarray:
        """Uncertainty used by the specialist acceptance rule.

        The manuscript defines the downstream rule as ``u_i^k <= lambda_k``
        and explicitly calibrates the router and specialist on their own score
        scales.  Keeping this named property prevents router uncertainty from
        leaking into the specialist-only comparator through the legacy fused
        score.
        """
        return self.specialist_uncertainty

    @property
    def specialist_only_score(self) -> np.ndarray:
        return (
            self.specialist_uncertainty
            if self.specialist_only_uncertainty is None
            else self.specialist_only_uncertainty
        )

    @property
    def specialist_only_outcome(self) -> np.ndarray:
        return (
            self.specialist_correctness
            if self.specialist_only_correctness is None
            else self.specialist_only_correctness
        )


def _aggregate_numeric(values: Iterable[float], rule: str) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    if rule == "mean":
        return float(np.mean(array))
    if rule == "median":
        return float(np.median(array))
    if rule == "max":
        return float(np.max(array))
    raise ValueError(f"Unsupported aggregation rule: {rule}")


def _normalise_label(value: Any) -> int | tuple[int, ...]:
    if isinstance(value, list):
        return tuple(sorted(int(item) for item in value))
    return int(value)


def _majority_vote(labels: list[Any], uncertainties: list[float]) -> Any:
    """Majority vote with uncertainty-aware, deterministic tie breaking."""
    normalised = [_normalise_label(label) for label in labels]
    counts = Counter(normalised)
    max_count = max(counts.values())
    candidates = [label for label, count in counts.items() if count == max_count]
    if len(candidates) == 1:
        winner = candidates[0]
    else:
        mean_uncertainty = {
            candidate: float(np.mean([
                uncertainty
                for label, uncertainty in zip(normalised, uncertainties)
                if label == candidate
            ]))
            for candidate in candidates
        }
        winner = min(candidates, key=lambda label: (mean_uncertainty[label], label))
    return list(winner) if isinstance(winner, tuple) else winner


def _labels_equal(left: Any, right: Any) -> bool:
    return _normalise_label(left) == _normalise_label(right)


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def load_patient_scores(
    input_dir: Path,
    *,
    aggregation: str = "mean",
    lambda_1: float = 0.0,
    lambda_2: float = 1.0,
    require_real_ids: bool = False,
) -> tuple[PatientScores, dict[str, Any]]:
    """Validate Step-1 files and aggregate every patient to one decision.

    The input may contain multiple records per patient. Router and specialist
    labels are aggregated by majority vote; uncertainty is aggregated using
    ``aggregation``. The returned arrays therefore have one element per patient.
    """
    input_dir = Path(input_dir)
    phase1_path = input_dir / "phase_1.json"
    if not phase1_path.is_file():
        raise FileNotFoundError(f"Missing Phase-1 file: {phase1_path}")
    if aggregation not in {"mean", "median", "max"}:
        raise ValueError("aggregation must be one of: mean, median, max")
    if not np.isclose(lambda_1 + lambda_2, 1.0):
        raise ValueError("lambda_1 and lambda_2 must sum to one")

    phase1 = _load_json(phase1_path)
    phase2_cache: dict[str, dict[str, Any]] = {}
    grouped: dict[tuple[str, str], list[tuple[str, dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    patient_sources: dict[str, set[str]] = defaultdict(set)

    required_p1 = {
        "uncertainty", "correctness", "predicted_disease", "true_disease",
        "source_dataset", "patient_id",
    }
    required_p2 = {"uncertainty", "correctness", "predicted", "true", "patient_id"}

    for sample_id, p1 in phase1.items():
        missing = required_p1.difference(p1)
        if missing:
            raise ValueError(f"{sample_id} is missing Phase-1 fields: {sorted(missing)}")
        source = str(p1["source_dataset"])
        if source not in DATASET_TO_DISEASE:
            raise ValueError(f"Unsupported source_dataset={source!r} for {sample_id}")
        expected_disease = DATASET_TO_DISEASE[source]
        if int(p1["true_disease"]) != expected_disease:
            raise ValueError(
                f"{sample_id} has true_disease={p1['true_disease']} but "
                f"{source} maps to {expected_disease}"
            )
        if source not in phase2_cache:
            phase2_path = input_dir / f"phase_2_{source}.json"
            if not phase2_path.is_file():
                raise FileNotFoundError(f"Missing Phase-2 file: {phase2_path}")
            phase2_cache[source] = _load_json(phase2_path)
        p2 = phase2_cache[source].get(sample_id)
        if p2 is None:
            raise ValueError(f"{sample_id} is missing from phase_2_{source}.json")
        missing = required_p2.difference(p2)
        if missing:
            raise ValueError(f"{sample_id} is missing Phase-2 fields: {sorted(missing)}")
        patient_id = str(p1["patient_id"])
        if not patient_id:
            raise ValueError(f"{sample_id} has an empty patient_id")
        if patient_id != str(p2["patient_id"]):
            raise ValueError(f"Patient-ID mismatch across phases for {sample_id}")
        implied_router_correct = int(int(p1["predicted_disease"]) == int(p1["true_disease"]))
        if int(p1["correctness"]) != implied_router_correct:
            raise ValueError(f"Inconsistent Phase-1 correctness for {sample_id}")
        implied_specialist_correct = int(_labels_equal(p2["predicted"], p2["true"]))
        if int(p2["correctness"]) != implied_specialist_correct:
            raise ValueError(f"Inconsistent Phase-2 correctness for {sample_id}")
        patient_sources[patient_id].add(source)
        grouped[(source, patient_id)].append((sample_id, p1, p2))

    cross_source = sorted(pid for pid, sources in patient_sources.items() if len(sources) > 1)
    if cross_source:
        raise ValueError(f"patient_id appears in multiple sources: {cross_source[0]}")

    expected_phase2_ids: dict[str, set[str]] = defaultdict(set)
    for sample_id, p1 in phase1.items():
        expected_phase2_ids[str(p1["source_dataset"])].add(sample_id)
    for source, phase2 in phase2_cache.items():
        extra = set(phase2).difference(expected_phase2_ids[source])
        if extra:
            raise ValueError(f"phase_2_{source}.json contains unmatched sample: {sorted(extra)[0]}")

    patients: list[dict[str, Any]] = []
    for (source, patient_id), rows in sorted(grouped.items()):
        true_diseases = {int(p1["true_disease"]) for _, p1, _ in rows}
        if len(true_diseases) != 1:
            raise ValueError(f"Patient {patient_id} has multiple true diseases")
        true_labels = {_normalise_label(p2["true"]) for _, _, p2 in rows}
        if len(true_labels) != 1:
            raise ValueError(f"Patient {patient_id} has ambiguous Phase-2 ground truth")

        u1_values = [float(p1["uncertainty"]) for _, p1, _ in rows]
        u2_values = [float(p2["uncertainty"]) for _, _, p2 in rows]
        if not np.all(np.isfinite(u1_values + u2_values)):
            raise ValueError(f"Patient {patient_id} contains non-finite uncertainty")
        route_pred = int(_majority_vote(
            [p1["predicted_disease"] for _, p1, _ in rows], u1_values
        ))
        specialist_pred = _majority_vote(
            [p2["predicted"] for _, _, p2 in rows], u2_values
        )
        specialist_true = rows[0][2]["true"]
        true_disease = next(iter(true_diseases))
        route_correct = int(route_pred == true_disease)
        specialist_correct = int(_labels_equal(specialist_pred, specialist_true))
        id_types = {
            str(record["patient_id_type"])
            for _, p1, p2 in rows
            for record in (p1, p2)
            if "patient_id_type" in record
        }
        if id_types and id_types != {"dataset_provided"}:
            raise ValueError(
                f"Patient {patient_id} has an unsupported patient_id_type: "
                + ", ".join(sorted(id_types))
            )
        patients.append({
            "patient_id": patient_id,
            "source": source,
            "patient_id_type": "dataset_provided",
            "record_count": len(rows),
            "router_uncertainty": _aggregate_numeric(u1_values, aggregation),
            "router_correctness": route_correct,
            "predicted_disease": route_pred,
            "true_disease": true_disease,
            "specialist_uncertainty": _aggregate_numeric(u2_values, aggregation),
            "specialist_correctness": specialist_correct,
            "system_correctness": int(route_correct and specialist_correct),
        })

    id_type_counts = Counter(patient["patient_id_type"] for patient in patients)
    if require_real_ids and any(key != "dataset_provided" for key in id_type_counts):
        raise ValueError(
            "require_real_ids=True, but patient_id_type contains: "
            + ", ".join(f"{key}={value}" for key, value in sorted(id_type_counts.items()))
        )

    source_rows: dict[str, list[int]] = defaultdict(list)
    for index, patient in enumerate(patients):
        source_rows[patient["source"]].append(index)
    source_indices = {
        source: np.asarray(indices, dtype=np.int32)
        for source, indices in sorted(source_rows.items())
    }
    scores = PatientScores(
        patient_ids=tuple(patient["patient_id"] for patient in patients),
        sources=tuple(patient["source"] for patient in patients),
        patient_id_types=tuple(patient["patient_id_type"] for patient in patients),
        record_counts=np.asarray([patient["record_count"] for patient in patients], dtype=np.int32),
        router_uncertainty=np.asarray([patient["router_uncertainty"] for patient in patients]),
        router_correctness=np.asarray([patient["router_correctness"] for patient in patients], dtype=np.int8),
        predicted_disease=np.asarray([patient["predicted_disease"] for patient in patients], dtype=np.int8),
        true_disease=np.asarray([patient["true_disease"] for patient in patients], dtype=np.int8),
        specialist_uncertainty=np.asarray([patient["specialist_uncertainty"] for patient in patients]),
        specialist_correctness=np.asarray([patient["specialist_correctness"] for patient in patients], dtype=np.int8),
        system_correctness=np.asarray([patient["system_correctness"] for patient in patients], dtype=np.int8),
        source_indices=source_indices,
        aggregation=aggregation,
        lambda_1=float(lambda_1),
        lambda_2=float(lambda_2),
    )
    audit = {
        "input_dir": str(input_dir.resolve()),
        "n_records": scores.n_records,
        "n_patients": scores.n_patients,
        "n_sources": len(scores.source_indices),
        "aggregation": aggregation,
        "patient_id_type_counts": dict(sorted(id_type_counts.items())),
        "sources": {
            source: {
                "disease_id": DATASET_TO_DISEASE[source],
                "records": int(scores.record_counts[indices].sum()),
                "patients": int(indices.size),
                "multi_record_patients": int(np.sum(scores.record_counts[indices] > 1)),
                "max_records_per_patient": int(np.max(scores.record_counts[indices])),
            }
            for source, indices in scores.source_indices.items()
        },
        "validation": {
            "phase1_phase2_sample_match": True,
            "phase1_phase2_patient_id_match": True,
            "patient_disjoint_across_sources": True,
            "within_patient_ground_truth_consistent": True,
            "stored_correctness_consistent": True,
        },
    }
    if scores.true_disease.min() < 0 or scores.true_disease.max() >= NUM_DISEASES:
        raise ValueError("true_disease lies outside the configured disease range")
    return scores, audit
