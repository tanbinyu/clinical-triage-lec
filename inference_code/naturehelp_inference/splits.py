from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

from .config import DatasetConfig, PipelineConfig
from .manifest import manifest_digest
from .records import SampleRecord


SPLIT_VERSION = 4
SPLIT_NAMES = ("train", "val", "downstream")
DOWNSTREAM_PARTITION_VERSION = 1
DOWNSTREAM_ROLES = ("CR", "CS", "T")


def _labels(record: SampleRecord) -> tuple[int, ...]:
    return record.target if isinstance(record.target, tuple) else (record.target,)


def _stable_tiebreak(seed: int, dataset: str, patient: str) -> int:
    token = f"{seed}\0{dataset}\0{patient}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(token).digest()[:8], "big")


def _patient_groups(records: list[SampleRecord]) -> dict[str, list[SampleRecord]]:
    groups: dict[str, list[SampleRecord]] = defaultdict(list)
    for record in records:
        groups[record.patient_id].append(record)
    return dict(groups)


def _patient_vector(
    group: list[SampleRecord], num_classes: int, include_empty: bool = False
) -> list[int]:
    vector = [0] * (num_classes + int(include_empty))
    for record in group:
        labels = _labels(record)
        if include_empty and not labels:
            vector[-1] += 1
        for label in labels:
            vector[label] += 1
    return vector


def _largest_remainder(count: int, fractions: tuple[float, ...]) -> list[int]:
    raw = [count * value for value in fractions]
    result = [int(value) for value in raw]
    remainder = count - sum(result)
    # Decimal fractions such as 0.35/0.05/0.60 can produce binary floating-point
    # near-ties.  Round the residual, then give a true tie to the larger target
    # partition.  This reproduces the reference patient counts in Table A5.
    order = sorted(
        range(len(fractions)),
        key=lambda index: (
            -round(raw[index] - result[index], 12), -fractions[index], index
        ),
    )
    for index in order[:remainder]:
        result[index] += 1
    if count >= len(fractions):
        for empty_index in [index for index, value in enumerate(result) if value == 0]:
            donor = max(range(len(fractions)), key=lambda index: result[index])
            result[donor] -= 1
            result[empty_index] += 1
    return result


def _capacities(count: int, fractions: tuple[float, float, float]) -> list[int]:
    return _largest_remainder(count, fractions)


def _fixed_split_capacities(
    count: int, fractions: tuple[float, float, float]
) -> list[int]:
    if all(
        abs(actual - expected) < 1e-12
        for actual, expected in zip(fractions, (0.35, 0.05, 0.60), strict=True)
    ):
        final = _largest_remainder(count, (0.35, 0.05, 0.24, 0.18, 0.18))
        return [final[0], final[1], sum(final[2:])]
    return _capacities(count, fractions)


def _consistent_multiclass_split(
    groups: dict[str, list[SampleRecord]],
    spec: DatasetConfig,
    seed: int,
    fractions: tuple[float, float, float],
) -> dict[str, str] | None:
    if spec.task_type != "multiclass":
        return None
    strata: dict[int, list[str]] = defaultdict(list)
    for patient, group in groups.items():
        targets = {record.target for record in group}
        if len(targets) != 1:
            return None
        target = next(iter(targets))
        if not isinstance(target, int):
            return None
        strata[target].append(patient)
    assignments: dict[str, str] = {}
    for target in sorted(strata):
        patients = sorted(
            strata[target],
            key=lambda patient: (
                -len(groups[patient]), _stable_tiebreak(seed, spec.name, patient)
            ),
        )
        counts = _fixed_split_capacities(len(patients), fractions)
        patient_used = [0, 0, 0]
        sample_used = [0, 0, 0]
        sample_total = sum(len(groups[patient]) for patient in patients)
        sample_targets = [sample_total * fraction for fraction in fractions]
        for patient in patients:
            group_size = len(groups[patient])
            candidates: list[tuple[float, int]] = []
            for split_index in range(3):
                if patient_used[split_index] >= counts[split_index]:
                    continue
                current = sample_used[split_index]
                target_samples = sample_targets[split_index]
                sample_delta = (
                    (current + group_size - target_samples) ** 2
                    - (current - target_samples) ** 2
                ) / max(sample_total**2, 1)
                target_patients = counts[split_index]
                patient_delta = (
                    (patient_used[split_index] + 1 - target_patients) ** 2
                    - (patient_used[split_index] - target_patients) ** 2
                ) / max(len(patients) ** 2, 1)
                candidates.append((3.0 * sample_delta + patient_delta, split_index))
            _, chosen = min(candidates, key=lambda item: (item[0], item[1]))
            assignments[patient] = SPLIT_NAMES[chosen]
            patient_used[chosen] += 1
            sample_used[chosen] += group_size
        split_index_by_name = {name: index for index, name in enumerate(SPLIT_NAMES)}
        for _ in range(100):
            best_delta = -1e-12
            best_swap: tuple[str, str, int, int] | None = None
            candidates_by_split: list[list[str]] = []
            for split_index in range(3):
                ranked = sorted(
                    (patient for patient in patients if assignments[patient] == SPLIT_NAMES[split_index]),
                    key=lambda patient: (len(groups[patient]), patient),
                )
                candidates_by_split.append(list(dict.fromkeys(ranked[:20] + ranked[-20:])))
            for first_split in range(3):
                for second_split in range(first_split + 1, 3):
                    for first in candidates_by_split[first_split]:
                        first_size = len(groups[first])
                        for second in candidates_by_split[second_split]:
                            second_size = len(groups[second])
                            delta = 0.0
                            for split_index, change in (
                                (first_split, second_size - first_size),
                                (second_split, first_size - second_size),
                            ):
                                current = sample_used[split_index]
                                target_samples = sample_targets[split_index]
                                delta += (
                                    (current + change - target_samples) ** 2
                                    - (current - target_samples) ** 2
                                ) / max(sample_total**2, 1)
                            if delta < best_delta:
                                best_delta = delta
                                best_swap = (first, second, first_split, second_split)
            if best_swap is None:
                break
            first, second, first_split, second_split = best_swap
            first_size = len(groups[first])
            second_size = len(groups[second])
            assignments[first] = SPLIT_NAMES[second_split]
            assignments[second] = SPLIT_NAMES[first_split]
            sample_used[first_split] += second_size - first_size
            sample_used[second_split] += first_size - second_size
    return assignments


def _stable_rank_split(
    groups: dict[str, list[SampleRecord]],
    spec: DatasetConfig,
    seed: int,
    fractions: tuple[float, float, float],
) -> dict[str, str]:
    """Exact patient-ratio split; large cohorts are statistically label-stable."""
    patients = sorted(
        groups, key=lambda patient: _stable_tiebreak(seed, spec.name, patient)
    )
    counts = _fixed_split_capacities(len(patients), fractions)
    assignments: dict[str, str] = {}
    offset = 0
    for split, count in zip(SPLIT_NAMES, counts, strict=True):
        for patient in patients[offset:offset + count]:
            assignments[patient] = split
        offset += count
    return assignments


def make_patient_stratified_split(
    records: list[SampleRecord],
    spec: DatasetConfig,
    seed: int,
    fractions: tuple[float, float, float],
) -> dict[str, str]:
    """Greedy iterative stratification with patients as indivisible groups.

    The objective jointly balances patient counts, sample counts and per-class
    positive counts.  Stable hashes only break ties, so repeated runs are identical.
    """
    groups = _patient_groups(records)
    patient_count = len(groups)
    if patient_count < len(SPLIT_NAMES):
        raise ValueError(f"{spec.name}: 患者数不足以建立 train/val/downstream")

    if patient_count >= 1000:
        large = _stable_rank_split(groups, spec, seed, fractions)
        validate_split(records, large)
        return large

    split_dimensions = spec.num_classes + int(spec.task_type == "multilabel")
    vectors = {
        patient: _patient_vector(
            group, spec.num_classes, include_empty=spec.task_type == "multilabel"
        )
        for patient, group in groups.items()
    }
    total_labels = [sum(vector[index] for vector in vectors.values()) for index in range(split_dimensions)]
    label_patient_frequency = [
        sum(vector[index] > 0 for vector in vectors.values()) for index in range(split_dimensions)
    ]
    ordered = sorted(
        groups,
        key=lambda patient: (
            -sum(
                (1.0 / max(label_patient_frequency[index], 1))
                for index, value in enumerate(vectors[patient]) if value
            ),
            -len(groups[patient]),
            _stable_tiebreak(seed, spec.name, patient),
        ),
    )

    patient_capacities = _fixed_split_capacities(patient_count, fractions)
    patient_targets = [float(value) for value in patient_capacities]
    sample_total = len(records)
    sample_targets = [sample_total * value for value in fractions]
    label_targets = [
        [count * fraction for count in total_labels] for fraction in fractions
    ]
    patient_used = [0, 0, 0]
    sample_used = [0, 0, 0]
    label_used = [[0] * split_dimensions for _ in SPLIT_NAMES]
    assignments: dict[str, str] = {}

    for position, patient in enumerate(ordered):
        group_size = len(groups[patient])
        vector = vectors[patient]
        remaining = len(ordered) - position
        candidates: list[tuple[float, int]] = []
        for split_index in range(3):
            if patient_used[split_index] >= patient_capacities[split_index]:
                continue
            empty_other = sum(
                patient_used[index] == 0 for index in range(3) if index != split_index
            )
            if remaining <= empty_other and patient_used[split_index] > 0:
                continue
            patient_cost = (
                (patient_used[split_index] + 1 - patient_targets[split_index]) ** 2
                - (patient_used[split_index] - patient_targets[split_index]) ** 2
            ) / max(patient_count**2, 1)
            sample_cost = (
                (sample_used[split_index] + group_size - sample_targets[split_index]) ** 2
                - (sample_used[split_index] - sample_targets[split_index]) ** 2
            ) / max(sample_total**2, 1)
            label_cost = 0.0
            for class_index, increment in enumerate(vector):
                if not increment:
                    continue
                target = label_targets[split_index][class_index]
                current = label_used[split_index][class_index]
                label_cost += (
                    (current + increment - target) ** 2 - (current - target) ** 2
                ) / max(total_labels[class_index] ** 2, 1)
            cost = patient_cost + 3.0 * sample_cost + 3.0 * label_cost
            candidates.append((cost, split_index))
        _, chosen = min(candidates, key=lambda item: (item[0], item[1]))
        assignments[patient] = SPLIT_NAMES[chosen]
        patient_used[chosen] += 1
        sample_used[chosen] += group_size
        for class_index, increment in enumerate(vector):
            label_used[chosen][class_index] += increment

    validate_split(records, assignments)
    return assignments


def validate_split(records: list[SampleRecord], assignments: dict[str, str]) -> None:
    patients = {record.patient_id for record in records}
    if set(assignments) != patients:
        missing = sorted(patients - set(assignments))[:10]
        extra = sorted(set(assignments) - patients)[:10]
        raise ValueError(f"split 患者集合不匹配；missing={missing}, extra={extra}")
    unknown = sorted(set(assignments.values()) - set(SPLIT_NAMES))
    if unknown:
        raise ValueError(f"split 含未知分区: {unknown}")
    empty = [name for name in SPLIT_NAMES if name not in assignments.values()]
    if empty:
        raise ValueError(f"split 存在空分区: {empty}")


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
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


def split_digest(assignments: dict[str, str]) -> str:
    encoded = json.dumps(assignments, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_or_load_split(
    config: PipelineConfig,
    dataset: str,
    records: list[SampleRecord],
    rebuild: bool = False,
) -> dict[str, str]:
    path = config.work_dir / "splits" / f"{dataset}.json"
    current_manifest_digest = manifest_digest(records)
    if path.exists() and not rebuild:
        payload = json.loads(path.read_text(encoding="utf-8"))
        expected = {
            "version": SPLIT_VERSION,
            "dataset": dataset,
            "seed": config.split_seed,
            "manifest_digest": current_manifest_digest,
        }
        mismatches = [key for key, value in expected.items() if payload.get(key) != value]
        if mismatches:
            raise RuntimeError(
                f"{dataset}: 已保存 split 与当前配置/manifest 不一致 ({', '.join(mismatches)})；"
                "确认数据变化后使用 --rebuild-split"
            )
        assignments = {str(k): str(v) for k, v in payload["patients"].items()}
        validate_split(records, assignments)
        return assignments

    spec = config.datasets[dataset]
    fractions = (
        config.train_fraction, config.val_fraction, config.downstream_fraction,
    )
    assignments = make_patient_stratified_split(records, spec, config.split_seed, fractions)
    payload: dict[str, object] = {
        "version": SPLIT_VERSION,
        "dataset": dataset,
        "seed": config.split_seed,
        "fractions": dict(zip(SPLIT_NAMES, fractions, strict=True)),
        "manifest_digest": current_manifest_digest,
        "split_digest": split_digest(assignments),
        "patients": assignments,
        "statistics": split_statistics(records, assignments),
    }
    _atomic_json(path, payload)
    return assignments


def records_for_split(
    records: list[SampleRecord], assignments: dict[str, str], split: str
) -> list[SampleRecord]:
    if split == "all":
        return records
    if split in {"patient_test", "test"}:
        split = "downstream"
    if split not in SPLIT_NAMES:
        raise ValueError(f"未知 split: {split}")
    return [record for record in records if assignments[record.patient_id] == split]


def split_statistics(
    records: list[SampleRecord], assignments: dict[str, str]
) -> dict[str, object]:
    result: dict[str, object] = {}
    for split in SPLIT_NAMES:
        selected = records_for_split(records, assignments, split)
        labels: Counter[str] = Counter()
        for record in selected:
            values = _labels(record)
            if isinstance(record.target, tuple) and not values:
                labels["normal/none"] += 1
            for label in values:
                labels[str(label)] += 1
        result[split] = {
            "samples": len(selected),
            "patients": len({record.patient_id for record in selected}),
            "labels": dict(sorted(labels.items())),
        }
    return result


def _decision_tiebreak(
    seed: int, dataset: str, patient: str, sample_id: str
) -> int:
    token = f"{seed}\0{dataset}\0{patient}\0{sample_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(token).digest()[:8], "big")


def make_downstream_partition(
    config: PipelineConfig,
    manifests: dict[str, list[SampleRecord]],
    repetition: int = 0,
    reference: bool = False,
) -> dict[str, object]:
    """Create the paper's second-stage patient partition without outcome leakage.

    First, only patients assigned to the fixed 60% downstream pool are eligible.
    At most 2,000 patients are selected independently in each source.  Selected
    patients are then assigned to CR/CS/T with largest-remainder 40/30/30 counts.
    Finally, one genuine source-defined decision is selected per patient for the
    primary analysis.  Hash ranking is equivalent to seeded sampling without
    replacement and is invariant to filesystem enumeration order.
    """
    if repetition < 0:
        raise ValueError("repetition 必须为非负整数")
    base_seed = config.split_seed if reference else config.analysis_seed + repetition
    role_fractions = (
        config.cr_fraction, config.cs_fraction, config.evaluation_fraction,
    )
    datasets: dict[str, object] = {}
    global_role_counts: Counter[str] = Counter()

    for dataset in sorted(manifests, key=lambda name: config.datasets[name].disease_id):
        records = manifests[dataset]
        split_assignments = build_or_load_split(config, dataset, records)
        groups = _patient_groups(records)
        eligible = [
            patient for patient in groups
            if split_assignments[patient] == "downstream"
        ]
        ranked = sorted(
            eligible,
            key=lambda patient: _stable_tiebreak(base_seed, dataset, patient),
        )
        selected = (
            ranked if reference
            else ranked[:config.max_downstream_patients_per_source]
        )
        if reference:
            final_counts = _largest_remainder(
                len(groups), (0.35, 0.05, 0.24, 0.18, 0.18)
            )
            role_counts = final_counts[2:]
            if sum(role_counts) != len(selected):
                raise RuntimeError(
                    f"{dataset}: reference downstream count 与固定 split 不一致"
                )
        else:
            role_counts = _capacities(len(selected), role_fractions)
        roles: dict[str, str] = {}
        offset = 0
        for role, count in zip(DOWNSTREAM_ROLES, role_counts, strict=True):
            for patient in selected[offset:offset + count]:
                roles[patient] = role
            global_role_counts[role] += count
            offset += count

        primary_decisions: dict[str, str] = {}
        for patient in selected:
            candidates = groups[patient]
            chosen = min(
                candidates,
                key=lambda record: _decision_tiebreak(
                    base_seed,
                    dataset,
                    patient,
                    record.sample_id or record.source_id,
                ),
            )
            if chosen.sample_id is None:
                raise RuntimeError(
                    f"{dataset}/{chosen.source_id}: manifest 缺少 sample_id"
                )
            primary_decisions[patient] = chosen.sample_id

        datasets[dataset] = {
            "eligible_downstream_patients": len(eligible),
            "selected_patients": len(selected),
            "role_counts": dict(zip(DOWNSTREAM_ROLES, role_counts, strict=True)),
            "patient_roles": roles,
            "primary_decisions": primary_decisions,
            "split_digest": split_digest(split_assignments),
            "manifest_digest": manifest_digest(records),
        }

    return {
        "version": DOWNSTREAM_PARTITION_VERSION,
        "kind": "reference" if reference else "repeated",
        "repetition": repetition,
        "seed": base_seed,
        "patient_cap_per_source": config.max_downstream_patients_per_source,
        "fractions": dict(zip(DOWNSTREAM_ROLES, role_fractions, strict=True)),
        "global_role_counts": dict(global_role_counts),
        "datasets": datasets,
    }


def write_downstream_partition(
    config: PipelineConfig,
    manifests: dict[str, list[SampleRecord]],
    repetition: int = 0,
    reference: bool = False,
) -> Path:
    payload = make_downstream_partition(
        config, manifests, repetition=repetition, reference=reference
    )
    suffix = "reference" if reference else f"repeat_{repetition:03d}"
    path = config.work_dir / "downstream_partitions" / f"{suffix}.json"
    _atomic_json(path, payload)
    return path

