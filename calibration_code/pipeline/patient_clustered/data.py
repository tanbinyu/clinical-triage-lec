"""Validate frozen scores without collapsing genuine within-patient labels."""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from pipeline.patient_level.data import PatientScores
from pipeline.patient_level.metadata import (
    DATASET_TO_DISEASE,
    DISEASE_TO_SPECIALIST_MODEL,
    NUM_DISEASES,
)


ALLOWED_LABEL_SCOPES = frozenset({"patient", "volume", "lesion", "nodule", "image"})
STRICT_PATIENT_SCOPES = frozenset({"patient", "volume"})


def _normalise_label(value: Any) -> int | tuple[int, ...]:
    if isinstance(value, list):
        return tuple(sorted(int(item) for item in value))
    return int(value)


def _labels_equal(left: Any, right: Any) -> bool:
    return _normalise_label(left) == _normalise_label(right)


def _aggregate_numeric(values: Iterable[float], rule: str) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    if rule == "mean":
        return float(np.mean(array))
    if rule == "median":
        return float(np.median(array))
    if rule == "max":
        return float(np.max(array))
    raise ValueError(f"Unsupported aggregation rule: {rule}")


def _majority_vote(labels: list[Any], uncertainties: list[float]) -> Any:
    normalised = [_normalise_label(label) for label in labels]
    counts = Counter(normalised)
    maximum = max(counts.values())
    candidates = [label for label, count in counts.items() if count == maximum]
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ClusteredDecisionScores:
    """One row per genuine inference decision plus explicit patient clusters."""

    sample_ids: tuple[str, ...]
    patient_ids: tuple[str, ...]
    sources: tuple[str, ...]
    label_scopes: tuple[str, ...]
    specialist_predicted: tuple[Any, ...]
    specialist_true: tuple[Any, ...]
    router_uncertainty: np.ndarray
    router_correctness: np.ndarray
    predicted_disease: np.ndarray
    true_disease: np.ndarray
    specialist_uncertainty: np.ndarray
    specialist_correctness: np.ndarray
    system_correctness: np.ndarray
    specialist_only_uncertainty: np.ndarray
    specialist_only_correctness: np.ndarray
    specialist_only_predicted: tuple[Any, ...]
    specialist_only_true: tuple[Any, ...]
    decision_patient_index: np.ndarray
    cluster_patient_ids: tuple[str, ...]
    patient_sources: tuple[str, ...]
    patient_record_indices: tuple[np.ndarray, ...]
    source_patient_indices: dict[str, np.ndarray]
    lambda_1: float
    lambda_2: float

    @property
    def n_decisions(self) -> int:
        return len(self.sample_ids)

    @property
    def n_records(self) -> int:
        return self.n_decisions

    @property
    def n_patients(self) -> int:
        return len(self.cluster_patient_ids)

    @property
    def fused_uncertainty(self) -> np.ndarray:
        return (
            self.lambda_1 * self.router_uncertainty
            + self.lambda_2 * self.specialist_uncertainty
        )

    @property
    def specialist_score(self) -> np.ndarray:
        """Score used for downstream specialist calibration and acceptance.

        This is deliberately independent of router uncertainty, matching the
        paper's stage-specific rule ``u_i^k <= lambda_k``.  ``fused_uncertainty``
        is retained only for backwards compatibility with archived workflows.
        """
        return self.specialist_uncertainty

    @property
    def specialist_only_score(self) -> np.ndarray:
        return self.specialist_only_uncertainty

    @property
    def specialist_only_outcome(self) -> np.ndarray:
        return self.specialist_only_correctness


def load_clustered_decisions(
    input_dir: Path,
    *,
    lambda_1: float = 0.0,
    lambda_2: float = 1.0,
    require_real_ids: bool = True,
    require_specialist_only: bool = False,
) -> tuple[ClusteredDecisionScores, dict[str, Any]]:
    """Load routed decisions and validate patient, score and correctness contracts.

    The current manuscript comparator is *routed* specialist-only: it uses the
    same root-level routed specialist outputs as Proposed LEC and merely turns
    off router abstention.  Consequently ``require_specialist_only=False``
    deliberately ignores an adjacent ``specialist-only`` directory even when
    it exists.  The optional source-specialist loader is retained only for
    archived analyses that explicitly opt in.
    """
    input_dir = Path(input_dir)
    phase1_path = input_dir / "phase_1.json"
    if not phase1_path.is_file():
        raise FileNotFoundError(f"Missing Phase-1 file: {phase1_path}")
    if not np.isclose(lambda_1 + lambda_2, 1.0):
        raise ValueError("lambda_1 and lambda_2 must sum to one")

    with open(phase1_path, encoding="utf-8") as handle:
        phase1 = json.load(handle)
    phase2_cache: dict[str, dict[str, Any]] = {}
    phase2_paths: dict[str, Path] = {}
    specialist_only_cache: dict[str, dict[str, Any]] = {}
    specialist_only_paths: dict[str, Path] = {}
    specialist_only_dir = input_dir / "specialist-only"
    required_p1 = {
        "uncertainty", "correctness", "predicted_disease", "true_disease",
        "source_dataset", "patient_id",
    }
    required_p2 = {
        "uncertainty", "correctness", "predicted", "true", "patient_id",
        "label_scope",
    }

    rows: list[dict[str, Any]] = []
    patient_sources: dict[str, set[str]] = defaultdict(set)
    grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
    id_types = Counter()
    expected_phase2_ids: dict[str, set[str]] = defaultdict(set)

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
            path = input_dir / f"phase_2_{source}.json"
            if not path.is_file():
                raise FileNotFoundError(f"Missing Phase-2 file: {path}")
            with open(path, encoding="utf-8") as handle:
                phase2_cache[source] = json.load(handle)
            phase2_paths[source] = path
        p2 = phase2_cache[source].get(sample_id)
        if p2 is None:
            raise ValueError(f"{sample_id} is missing from phase_2_{source}.json")
        if source not in specialist_only_cache:
            if require_specialist_only:
                path = specialist_only_dir / f"phase_2_{source}.json"
                if not path.is_file():
                    raise FileNotFoundError(f"Missing specialist-only file: {path}")
                with open(path, encoding="utf-8") as handle:
                    specialist_only_cache[source] = json.load(handle)
                specialist_only_paths[source] = path
            else:
                # Routed specialist-only shares the routed Phase-2 scores.
                specialist_only_cache[source] = phase2_cache[source]
        specialist_only = specialist_only_cache[source].get(sample_id)
        if specialist_only is None:
            raise ValueError(f"{sample_id} is missing from comparator Phase-2 data")
        missing = required_p2.difference(p2)
        if missing:
            raise ValueError(f"{sample_id} is missing Phase-2 fields: {sorted(missing)}")
        missing = required_p2.difference(specialist_only)
        if missing:
            raise ValueError(
                f"{sample_id} is missing specialist-only fields: {sorted(missing)}"
            )

        patient_id = str(p1["patient_id"])
        if not patient_id:
            raise ValueError(f"{sample_id} has an empty patient_id")
        if patient_id != str(p2["patient_id"]):
            raise ValueError(f"Patient-ID mismatch across phases for {sample_id}")
        if patient_id != str(specialist_only["patient_id"]):
            raise ValueError(
                f"Patient-ID mismatch in comparator output for {sample_id}"
            )
        scope = str(p2["label_scope"])
        if scope not in ALLOWED_LABEL_SCOPES:
            raise ValueError(f"Unsupported label_scope={scope!r} for {sample_id}")

        implied_router = int(int(p1["predicted_disease"]) == int(p1["true_disease"]))
        implied_specialist = int(_labels_equal(p2["predicted"], p2["true"]))
        implied_specialist_only = int(_labels_equal(
            specialist_only["predicted"], specialist_only["true"]
        ))
        if int(p1["correctness"]) != implied_router:
            raise ValueError(f"Inconsistent Phase-1 correctness for {sample_id}")
        if int(p2["correctness"]) != implied_specialist:
            raise ValueError(f"Inconsistent Phase-2 correctness for {sample_id}")
        if int(specialist_only["correctness"]) != implied_specialist_only:
            raise ValueError(
                f"Inconsistent specialist-only correctness for {sample_id}"
            )
        expected_routed_model = DISEASE_TO_SPECIALIST_MODEL[int(p1["predicted_disease"])]
        if p2.get("inference_model") not in (None, expected_routed_model):
            raise ValueError(
                f"{sample_id} routed model {p2.get('inference_model')!r} does not "
                f"match predicted pathway model {expected_routed_model!r}"
            )
        if require_specialist_only:
            expected_specialist_only_model = DISEASE_TO_SPECIALIST_MODEL[expected_disease]
            if specialist_only.get("inference_model") not in (
                None, expected_specialist_only_model
            ):
                raise ValueError(
                    f"{sample_id} specialist-only model "
                    f"{specialist_only.get('inference_model')!r} does not match "
                    f"source pathway model {expected_specialist_only_model!r}"
                )
        uncertainties = [
            float(p1["uncertainty"]),
            float(p2["uncertainty"]),
            float(specialist_only["uncertainty"]),
        ]
        if not np.all(np.isfinite(uncertainties)):
            raise ValueError(f"Non-finite uncertainty for {sample_id}")

        types = {
            str(record["patient_id_type"])
            for record in (p1, p2)
            if "patient_id_type" in record
        }
        if types and types != {"dataset_provided"}:
            raise ValueError(
                f"{sample_id} has unsupported patient_id_type: {sorted(types)}"
            )
        id_types["dataset_provided"] += 1
        row_index = len(rows)
        rows.append({
            "sample_id": str(sample_id),
            "patient_id": patient_id,
            "source": source,
            "label_scope": scope,
            "router_uncertainty": uncertainties[0],
            "router_correctness": implied_router,
            "predicted_disease": int(p1["predicted_disease"]),
            "true_disease": int(p1["true_disease"]),
            "specialist_uncertainty": uncertainties[1],
            "specialist_correctness": implied_specialist,
            "specialist_predicted": p2["predicted"],
            "specialist_true": p2["true"],
            "system_correctness": int(implied_router and implied_specialist),
            "specialist_only_uncertainty": uncertainties[2],
            "specialist_only_correctness": implied_specialist_only,
            "specialist_only_predicted": specialist_only["predicted"],
            "specialist_only_true": specialist_only["true"],
        })
        patient_sources[patient_id].add(source)
        grouped[(source, patient_id)].append(row_index)
        expected_phase2_ids[source].add(str(sample_id))

    cross_source = sorted(pid for pid, sources in patient_sources.items() if len(sources) > 1)
    if cross_source:
        raise ValueError(f"patient_id appears in multiple sources: {cross_source[0]}")
    for source, phase2 in phase2_cache.items():
        extra = set(phase2).difference(expected_phase2_ids[source])
        if extra:
            raise ValueError(f"phase_2_{source}.json contains unmatched sample: {sorted(extra)[0]}")
    for source, phase2 in specialist_only_cache.items():
        extra = set(phase2).difference(expected_phase2_ids[source])
        if extra:
            raise ValueError(
                f"specialist-only/phase_2_{source}.json contains unmatched sample: "
                f"{sorted(extra)[0]}"
            )
    if require_real_ids and set(id_types) != {"dataset_provided"}:
        raise ValueError("Real dataset-provided patient IDs are required")

    patient_keys = list(grouped)
    key_to_index = {key: index for index, key in enumerate(patient_keys)}
    decision_patient_index = np.asarray(
        [key_to_index[(row["source"], row["patient_id"])] for row in rows],
        dtype=np.int32,
    )
    patient_record_indices = tuple(
        np.asarray(grouped[key], dtype=np.int32) for key in patient_keys
    )
    source_patients: dict[str, list[int]] = defaultdict(list)
    for index, (source, _) in enumerate(patient_keys):
        source_patients[source].append(index)
    source_patient_indices = {
        source: np.asarray(indices, dtype=np.int32)
        for source, indices in sorted(source_patients.items())
    }

    scores = ClusteredDecisionScores(
        sample_ids=tuple(row["sample_id"] for row in rows),
        patient_ids=tuple(row["patient_id"] for row in rows),
        sources=tuple(row["source"] for row in rows),
        label_scopes=tuple(row["label_scope"] for row in rows),
        specialist_predicted=tuple(row["specialist_predicted"] for row in rows),
        specialist_true=tuple(row["specialist_true"] for row in rows),
        router_uncertainty=np.asarray([row["router_uncertainty"] for row in rows]),
        router_correctness=np.asarray([row["router_correctness"] for row in rows], dtype=np.int8),
        predicted_disease=np.asarray([row["predicted_disease"] for row in rows], dtype=np.int8),
        true_disease=np.asarray([row["true_disease"] for row in rows], dtype=np.int8),
        specialist_uncertainty=np.asarray([row["specialist_uncertainty"] for row in rows]),
        specialist_correctness=np.asarray([row["specialist_correctness"] for row in rows], dtype=np.int8),
        system_correctness=np.asarray([row["system_correctness"] for row in rows], dtype=np.int8),
        specialist_only_uncertainty=np.asarray([
            row["specialist_only_uncertainty"] for row in rows
        ]),
        specialist_only_correctness=np.asarray([
            row["specialist_only_correctness"] for row in rows
        ], dtype=np.int8),
        specialist_only_predicted=tuple(
            row["specialist_only_predicted"] for row in rows
        ),
        specialist_only_true=tuple(row["specialist_only_true"] for row in rows),
        decision_patient_index=decision_patient_index,
        cluster_patient_ids=tuple(patient_id for _, patient_id in patient_keys),
        patient_sources=tuple(source for source, _ in patient_keys),
        patient_record_indices=patient_record_indices,
        source_patient_indices=source_patient_indices,
        lambda_1=float(lambda_1),
        lambda_2=float(lambda_2),
    )
    if scores.true_disease.min() < 0 or scores.true_disease.max() >= NUM_DISEASES:
        raise ValueError("true_disease lies outside the configured disease range")

    source_audit = {}
    scope_array = np.asarray(scores.label_scopes, dtype=object)
    for source, patient_indices in scores.source_patient_indices.items():
        decision_indices = np.concatenate([
            scores.patient_record_indices[index] for index in patient_indices
        ])
        heterogeneous = 0
        multi_record = 0
        max_records = 0
        class_counts: Counter[str] = Counter()
        empty_label_decisions = 0
        for decision_index in decision_indices:
            truth = scores.specialist_true[int(decision_index)]
            if isinstance(truth, (list, tuple)):
                if not truth:
                    empty_label_decisions += 1
                class_counts.update(str(int(label)) for label in truth)
            else:
                class_counts[str(int(truth))] += 1
        for patient_index in patient_indices:
            indices = scores.patient_record_indices[patient_index]
            multi_record += int(indices.size > 1)
            max_records = max(max_records, int(indices.size))
            truths = {_normalise_label(scores.specialist_true[index]) for index in indices}
            heterogeneous += int(len(truths) > 1)
        source_audit[source] = {
            "disease_id": DATASET_TO_DISEASE[source],
            "decisions": int(decision_indices.size),
            "patients": int(patient_indices.size),
            "multi_record_patients": multi_record,
            "heterogeneous_label_patients": heterogeneous,
            "max_records_per_patient": max_records,
            "label_scope_counts": dict(sorted(Counter(scope_array[decision_indices]).items())),
            "class_counts": dict(sorted(class_counts.items(), key=lambda item: int(item[0]))),
            "empty_label_decisions": empty_label_decisions,
        }

    files = [
        phase1_path,
        *[phase2_paths[source] for source in sorted(phase2_paths)],
        *[specialist_only_paths[source] for source in sorted(specialist_only_paths)],
    ]
    audit = {
        "input_dir": str(input_dir.resolve()),
        "n_decisions": scores.n_decisions,
        "n_records": scores.n_records,
        "n_patients": scores.n_patients,
        "n_sources": len(scores.source_patient_indices),
        "decision_scope_counts": dict(sorted(Counter(scores.label_scopes).items())),
        "patient_id_type_counts": {"dataset_provided": scores.n_patients},
        "sources": source_audit,
        "files": [
            {
                "name": str(path.relative_to(input_dir)),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in files
        ],
        "validation": {
            "phase1_phase2_sample_match": True,
            "phase1_phase2_patient_id_match": True,
            "patient_disjoint_across_sources": True,
            "stored_correctness_consistent": True,
            "routed_model_matches_predicted_pathway": True,
            "specialist_only_model_matches_source_pathway": bool(specialist_only_paths),
            "routed_specialist_only_uses_root_phase2": not require_specialist_only,
            "label_scope_present": True,
            "within_patient_label_consistency_required": False,
        },
    }
    return scores, audit


def aggregate_strict_patient_subset(
    data: ClusteredDecisionScores,
    *,
    aggregation: str = "mean",
) -> tuple[PatientScores, dict[str, Any]]:
    """Aggregate only sources whose truth is explicitly patient- or volume-scoped."""
    if aggregation not in {"mean", "median", "max"}:
        raise ValueError("aggregation must be one of: mean, median, max")
    scope_array = np.asarray(data.label_scopes, dtype=object)
    eligible_sources = []
    excluded_sources = []
    for source, patient_indices in data.source_patient_indices.items():
        decision_indices = np.concatenate([
            data.patient_record_indices[index] for index in patient_indices
        ])
        if set(scope_array[decision_indices]).issubset(STRICT_PATIENT_SCOPES):
            eligible_sources.append(source)
        else:
            excluded_sources.append(source)

    patients = []
    for patient_index, (patient_id, source) in enumerate(
        zip(data.cluster_patient_ids, data.patient_sources)
    ):
        if source not in eligible_sources:
            continue
        indices = data.patient_record_indices[patient_index]
        truths = {_normalise_label(data.specialist_true[index]) for index in indices}
        if len(truths) != 1:
            raise ValueError(
                f"Strict patient/volume source {source} has heterogeneous truth for {patient_id}"
            )
        u1 = data.router_uncertainty[indices].tolist()
        u2 = data.specialist_uncertainty[indices].tolist()
        u2_specialist_only = data.specialist_only_uncertainty[indices].tolist()
        route_pred = int(_majority_vote(
            data.predicted_disease[indices].tolist(), u1
        ))
        specialist_pred = _majority_vote(
            [data.specialist_predicted[index] for index in indices], u2
        )
        specialist_true = data.specialist_true[int(indices[0])]
        specialist_only_pred = _majority_vote(
            [data.specialist_only_predicted[index] for index in indices],
            u2_specialist_only,
        )
        specialist_only_true = data.specialist_only_true[int(indices[0])]
        true_disease = int(data.true_disease[int(indices[0])])
        route_correct = int(route_pred == true_disease)
        specialist_correct = int(_labels_equal(specialist_pred, specialist_true))
        patients.append({
            "patient_id": patient_id,
            "source": source,
            "record_count": int(indices.size),
            "router_uncertainty": _aggregate_numeric(u1, aggregation),
            "router_correctness": route_correct,
            "predicted_disease": route_pred,
            "true_disease": true_disease,
            "specialist_uncertainty": _aggregate_numeric(u2, aggregation),
            "specialist_correctness": specialist_correct,
            "system_correctness": int(route_correct and specialist_correct),
            "specialist_only_uncertainty": _aggregate_numeric(
                u2_specialist_only, aggregation
            ),
            "specialist_only_correctness": int(_labels_equal(
                specialist_only_pred, specialist_only_true
            )),
        })

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
        patient_id_types=tuple("dataset_provided" for _ in patients),
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
        lambda_1=data.lambda_1,
        lambda_2=data.lambda_2,
        specialist_only_uncertainty=np.asarray([
            patient["specialist_only_uncertainty"] for patient in patients
        ]),
        specialist_only_correctness=np.asarray([
            patient["specialist_only_correctness"] for patient in patients
        ], dtype=np.int8),
    )
    audit = {
        "unit_of_analysis": "patient_or_volume",
        "eligible_label_scopes": sorted(STRICT_PATIENT_SCOPES),
        "eligible_sources": sorted(eligible_sources),
        "excluded_sources": sorted(excluded_sources),
        "n_patients": scores.n_patients,
        "n_underlying_records": scores.n_records,
        "aggregation": aggregation,
        "sources": {
            source: {
                "patients": int(indices.size),
                "underlying_records": int(scores.record_counts[indices].sum()),
            }
            for source, indices in source_indices.items()
        },
    }
    return scores, audit
