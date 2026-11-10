"""Patient-disjoint splitting and cluster resampling for decision-level scores."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np

from pipeline.patient_level.analysis import (
    _partition_sizes,
    patient_cluster_bootstrap,
    pooled_reference,
)
from pipeline.patient_level.metadata import DATASET_TO_DISEASE

from .data import ClusteredDecisionScores


@dataclass(frozen=True)
class ClusteredExperimentConfig:
    """Parameters for the mixed-scope patient-clustered frozen-score analysis.

    ``alpha`` is the clinical reporting target. ``calibration_alpha`` is the
    (optionally more conservative) level supplied to LEC.  Keeping these two
    quantities explicit prevents a safety margin from being hidden in plots or
    captions.
    """

    input_dir: Path
    output_dir: Path
    beta: float = 0.01
    alpha: float = 0.20
    calibration_alpha: float | None = None
    primary_analysis_unit: str = "patient_balanced"
    input_cohort_scope: str = "unverified"
    calibration_policy_status: str = "exploratory_post_hoc"
    n_splits: int = 500
    n_bootstrap: int = 1000
    router_calib_ratio: float = 0.30
    specialist_calib_ratio: float = 0.30
    test_ratio: float = 0.40
    # Optional source-balancing rule applied before role assignment.  A fixed,
    # score- and outcome-blind patient subset is selected within every source,
    # then reshuffled into C_R/C_S/T for each repetition.
    max_patients_per_source: int | None = None
    seed: int = 20260929
    development_split_seed: int = 20261019
    development_train_ratio: float = 0.35
    development_validation_ratio: float = 0.05
    development_downstream_ratio: float = 0.60
    patient_aggregation: str = "mean"
    # Retained for archived fused-score compatibility.  The current manuscript
    # workflow uses specialist_uncertainty directly at the specialist stage.
    lambda_1: float = 0.00
    lambda_2: float = 1.00
    require_real_ids: bool = True
    require_specialist_only: bool = False
    run_all_decisions_sensitivity: bool = True
    run_strict_patient_subset: bool = True
    run_paper_suite: bool = True
    manuscript_dir: Path | None = None
    config_file: Path | None = None

    @property
    def effective_calibration_alpha(self) -> float:
        return self.alpha if self.calibration_alpha is None else self.calibration_alpha

    @property
    def calibration_alpha_scale(self) -> float:
        return self.effective_calibration_alpha / self.alpha

    @property
    def patient_balanced_primary(self) -> bool:
        return self.primary_analysis_unit == "patient_balanced"

    def validate(self) -> None:
        if not 0 < self.beta < 1 or not 0 < self.alpha < 1:
            raise ValueError("beta and alpha must lie strictly between zero and one")
        if not 0 < self.effective_calibration_alpha <= self.alpha:
            raise ValueError(
                "calibration_alpha must be positive and no greater than reporting alpha"
            )
        if self.primary_analysis_unit not in {"patient_balanced", "all_decisions"}:
            raise ValueError(
                "primary_analysis_unit must be patient_balanced or all_decisions"
            )
        if self.input_cohort_scope not in {
            "test_only", "fixed_downstream_pool", "all_available_records", "unverified"
        }:
            raise ValueError(
                "input_cohort_scope must be test_only, fixed_downstream_pool, "
                "all_available_records or unverified"
            )
        if self.calibration_policy_status not in {
            "prespecified_confirmatory", "exploratory_post_hoc"
        }:
            raise ValueError(
                "calibration_policy_status must be prespecified_confirmatory or "
                "exploratory_post_hoc"
            )
        ratios = (
            self.router_calib_ratio,
            self.specialist_calib_ratio,
            self.test_ratio,
        )
        if any(value <= 0 for value in ratios) or not np.isclose(sum(ratios), 1.0):
            raise ValueError("router/specialist/test ratios must be positive and sum to one")
        if self.n_splits < 1 or self.n_bootstrap < 2:
            raise ValueError("n_splits must be positive and n_bootstrap at least two")
        development_ratios = (
            self.development_train_ratio,
            self.development_validation_ratio,
            self.development_downstream_ratio,
        )
        if any(value <= 0 for value in development_ratios) or not np.isclose(
            sum(development_ratios), 1.0
        ):
            raise ValueError(
                "development train/validation/downstream ratios must be positive "
                "and sum to one"
            )
        if (
            self.max_patients_per_source is not None
            and self.max_patients_per_source < 3
        ):
            raise ValueError("max_patients_per_source must be at least three")
        if self.patient_aggregation not in {"mean", "median", "max"}:
            raise ValueError("patient_aggregation must be mean, median or max")
        if not np.isclose(self.lambda_1 + self.lambda_2, 1.0):
            raise ValueError("lambda_1 and lambda_2 must sum to one")


def _source_partition_sizes(
    source_counts: list[int],
    ratios: tuple[float, float, float],
) -> list[tuple[int, int, int]]:
    """Allocate source strata while matching aggregate role quotas exactly.

    Largest-remainder allocation is first applied independently within each
    source. A deterministic minimum-cost rebalancing then moves the smallest
    possible number of patients between roles so the column totals equal the
    largest-remainder allocation of the full source-balanced cohort. This
    preserves source stratification and reproduces the manuscript's
    1,659/1,659/2,211 allocation for 5,529 included patients.
    """
    if not source_counts:
        return []
    raw = np.asarray(source_counts, dtype=float)[:, None] * np.asarray(
        ratios, dtype=float
    )[None, :]
    sizes = np.asarray(
        [_partition_sizes(count, ratios) for count in source_counts], dtype=int
    )
    target = np.asarray(_partition_sizes(sum(source_counts), ratios), dtype=int)
    current = sizes.sum(axis=0)

    while not np.array_equal(current, target):
        donors = np.flatnonzero(current > target)
        receivers = np.flatnonzero(current < target)
        candidates: list[tuple[float, int, int, int]] = []
        for source_index in range(len(source_counts)):
            for donor in donors:
                if sizes[source_index, donor] <= 1:
                    continue
                for receiver in receivers:
                    before = (
                        (sizes[source_index, donor] - raw[source_index, donor]) ** 2
                        + (sizes[source_index, receiver] - raw[source_index, receiver]) ** 2
                    )
                    after = (
                        (sizes[source_index, donor] - 1 - raw[source_index, donor]) ** 2
                        + (sizes[source_index, receiver] + 1 - raw[source_index, receiver]) ** 2
                    )
                    candidates.append(
                        (
                            round(float(after - before), 12),
                            source_index,
                            int(donor),
                            int(receiver),
                        )
                    )
        if not candidates:
            raise RuntimeError("Unable to reconcile source-stratified role totals")
        _, source_index, donor, receiver = min(candidates)
        sizes[source_index, donor] -= 1
        sizes[source_index, receiver] += 1
        current[donor] -= 1
        current[receiver] += 1

    return [tuple(int(value) for value in row) for row in sizes]


def _stable_patient_rank(seed: int, source: str, patient_id: str) -> int:
    token = f"{seed}\0{source}\0{patient_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(token).digest()[:8], "big")


def _ordered_sources(data: ClusteredDecisionScores) -> list[str]:
    """Return the clinical pathway order, independent of dictionary ordering."""
    return sorted(
        data.source_patient_indices,
        key=lambda source: (DATASET_TO_DISEASE[source], source),
    )


def _largest_remainder(count: int, fractions: tuple[float, ...]) -> list[int]:
    raw = [count * value for value in fractions]
    result = [int(value) for value in raw]
    remainder = count - sum(result)
    order = sorted(
        range(len(fractions)),
        key=lambda index: (
            -round(raw[index] - result[index], 12), -fractions[index], index
        ),
    )
    for index in order[:remainder]:
        result[index] += 1
    return result


def _fixed_development_counts(
    count: int,
    *,
    train_ratio: float,
    validation_ratio: float,
    downstream_ratio: float,
    downstream_role_ratios: tuple[float, float, float],
) -> tuple[int, int, int]:
    """Return exact train/validation/downstream counts for a source.

    The five-way largest-remainder allocation makes the one-time 35/5/60
    development split arithmetically consistent with the downstream 30/30/40
    role allocation.  This is the rule used to obtain the manuscript's source
    counts (including 639 BUS-BRA and 1,653 PU2756 downstream patients).
    """
    router_ratio, specialist_ratio, test_ratio = downstream_role_ratios
    five_way = _largest_remainder(
        count,
        (
            train_ratio,
            validation_ratio,
            downstream_ratio * test_ratio,
            downstream_ratio * router_ratio,
            downstream_ratio * specialist_ratio,
        ),
    )
    return five_way[0], five_way[1], sum(five_way[2:])


def _fixed_downstream_patients(
    data: ClusteredDecisionScores,
    source: str,
    *,
    development_split_seed: int,
    development_ratios: tuple[float, float, float],
    downstream_role_ratios: tuple[float, float, float],
) -> np.ndarray:
    patients = data.source_patient_indices[source]
    ranked = sorted(
        (int(index) for index in patients),
        key=lambda index: _stable_patient_rank(
            development_split_seed, source, data.cluster_patient_ids[index]
        ),
    )
    train_count, validation_count, downstream_count = _fixed_development_counts(
        len(ranked),
        train_ratio=development_ratios[0],
        validation_ratio=development_ratios[1],
        downstream_ratio=development_ratios[2],
        downstream_role_ratios=downstream_role_ratios,
    )
    start = train_count + validation_count
    selected = ranked[start:start + downstream_count]
    if len(selected) != downstream_count:
        raise RuntimeError(f"{source}: fixed downstream allocation is incomplete")
    return np.asarray(selected, dtype=np.int32)


def _patient_partitions(
    data: ClusteredDecisionScores,
    *,
    epoch: int,
    seed: int,
    ratios: tuple[float, float, float],
    max_patients_per_source: int | None = None,
    development_split_seed: int = 20261019,
    development_ratios: tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(np.random.SeedSequence([seed, epoch]))
    selected_by_source: list[np.ndarray] = []
    for source in _ordered_sources(data):
        patients = _fixed_downstream_patients(
            data,
            source,
            development_split_seed=development_split_seed,
            development_ratios=development_ratios,
            downstream_role_ratios=ratios,
        )
        if max_patients_per_source is not None:
            patients = patients[:max_patients_per_source]
        patients = patients.copy()
        rng.shuffle(patients)
        selected_by_source.append(patients)

    allocations = _source_partition_sizes(
        [int(patients.size) for patients in selected_by_source], ratios
    )
    router_parts, specialist_parts, test_parts = [], [], []
    for patients, (n_router, n_specialist, _) in zip(
        selected_by_source, allocations, strict=True
    ):
        router_parts.append(patients[:n_router])
        specialist_parts.append(patients[n_router:n_router + n_specialist])
        test_parts.append(patients[n_router + n_specialist:])
    router = np.concatenate(router_parts)
    specialist = np.concatenate(specialist_parts)
    test = np.concatenate(test_parts)
    if set(router) & set(specialist) or set(router) & set(test) or set(specialist) & set(test):
        raise RuntimeError("Patient leakage detected across C_R, C_S and T")
    return router, specialist, test


def _expand_all(
    data: ClusteredDecisionScores, patient_indices: np.ndarray
) -> np.ndarray:
    return np.concatenate([
        data.patient_record_indices[int(patient)] for patient in patient_indices
    ]).astype(np.int32, copy=False)


def _expand_one(
    data: ClusteredDecisionScores,
    patient_indices: np.ndarray,
    *,
    rng: np.random.Generator,
) -> np.ndarray:
    patients = np.asarray(patient_indices, dtype=np.int32)
    records = [data.patient_record_indices[int(patient)] for patient in patients]
    sizes = np.fromiter((item.size for item in records), dtype=np.int64)
    # Passing an array of patient-specific upper bounds is exactly equivalent
    # to the former scalar RNG calls for the same generator state, but avoids
    # one Python-to-NumPy call per patient.
    offsets = rng.integers(0, sizes)
    return np.fromiter(
        (item[int(offset)] for item, offset in zip(records, offsets)),
        dtype=np.int32,
        count=patients.size,
    )


def clustered_decision_split(
    data: ClusteredDecisionScores,
    *,
    epoch: int,
    seed: int,
    ratios: tuple[float, float, float],
    max_patients_per_source: int | None = None,
    development_split_seed: int = 20261019,
    development_ratios: tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assign patients to roles and include every genuine decision in that role."""
    partitions = _patient_partitions(
        data, epoch=epoch, seed=seed, ratios=ratios,
        max_patients_per_source=max_patients_per_source,
        development_split_seed=development_split_seed,
        development_ratios=development_ratios,
    )
    return tuple(_expand_all(data, patients) for patients in partitions)  # type: ignore[return-value]


def patient_balanced_split(
    data: ClusteredDecisionScores,
    *,
    epoch: int,
    seed: int,
    ratios: tuple[float, float, float],
    max_patients_per_source: int | None = None,
    development_split_seed: int = 20261019,
    development_ratios: tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select one decision uniformly per patient after patient-disjoint assignment."""
    partitions = _patient_partitions(
        data, epoch=epoch, seed=seed, ratios=ratios,
        max_patients_per_source=max_patients_per_source,
        development_split_seed=development_split_seed,
        development_ratios=development_ratios,
    )
    outputs = []
    for role, patients in enumerate(partitions):
        rng = np.random.default_rng(np.random.SeedSequence([seed, epoch, 410 + role]))
        outputs.append(_expand_one(data, patients, rng=rng))
    return tuple(outputs)  # type: ignore[return-value]


def _cluster_bootstrap(
    data: ClusteredDecisionScores,
    *,
    replicate: int,
    seed: int,
    ratios: tuple[float, float, float],
    one_per_patient: bool,
    max_patients_per_source: int | None = None,
    development_split_seed: int = 20261019,
    development_ratios: tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    partitions = _patient_partitions(
        data, epoch=replicate, seed=seed, ratios=ratios,
        max_patients_per_source=max_patients_per_source,
        development_split_seed=development_split_seed,
        development_ratios=development_ratios,
    )
    patient_sources = np.asarray(data.patient_sources, dtype=object)
    outputs = []
    for role, partition in enumerate(partitions):
        rng = np.random.default_rng(np.random.SeedSequence([seed, replicate, 991, role]))
        drawn_parts = []
        for source in _ordered_sources(data):
            stratum = partition[patient_sources[partition] == source]
            drawn_parts.append(stratum[rng.integers(0, stratum.size, size=stratum.size)])
        if one_per_patient:
            # Select one genuine decision once for every original patient, then
            # resample those patient-level observations.  If a patient is drawn
            # twice, both bootstrap copies must carry the same selected decision;
            # drawing a fresh record for every copy would no longer be a cluster
            # bootstrap.
            selected_records = _expand_one(data, partition, rng=rng)
            selected_by_patient = dict(zip(
                (int(patient) for patient in partition),
                (int(record) for record in selected_records),
            ))
            drawn = np.concatenate(drawn_parts)
            outputs.append(np.asarray(
                [selected_by_patient[int(patient)] for patient in drawn],
                dtype=np.int32,
            ))
        else:
            drawn = np.concatenate(drawn_parts)
            outputs.append(_expand_all(data, drawn))
    return tuple(outputs)  # type: ignore[return-value]


def clustered_bootstrap_partitions(
    data: ClusteredDecisionScores,
    *,
    replicate: int,
    seed: int,
    ratios: tuple[float, float, float],
    max_patients_per_source: int | None = None,
    development_split_seed: int = 20261019,
    development_ratios: tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return _cluster_bootstrap(
        data, replicate=replicate, seed=seed, ratios=ratios,
        one_per_patient=False,
        max_patients_per_source=max_patients_per_source,
        development_split_seed=development_split_seed,
        development_ratios=development_ratios,
    )


def patient_balanced_bootstrap_partitions(
    data: ClusteredDecisionScores,
    *,
    replicate: int,
    seed: int,
    ratios: tuple[float, float, float],
    max_patients_per_source: int | None = None,
    development_split_seed: int = 20261019,
    development_ratios: tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return _cluster_bootstrap(
        data, replicate=replicate, seed=seed, ratios=ratios,
        one_per_patient=True,
        max_patients_per_source=max_patients_per_source,
        development_split_seed=development_split_seed,
        development_ratios=development_ratios,
    )


def eligibility_allocation_audit(
    data: ClusteredDecisionScores,
    config: ClusteredExperimentConfig,
) -> dict:
    """Return a de-identified, run-specific eligibility/allocation audit."""
    role_ratios = (
        config.router_calib_ratio,
        config.specialist_calib_ratio,
        config.test_ratio,
    )
    development_ratios = (
        config.development_train_ratio,
        config.development_validation_ratio,
        config.development_downstream_ratio,
    )
    source_rows = []
    fixed_membership = hashlib.sha256()
    included_membership = hashlib.sha256()
    capped_counts = []
    for source in _ordered_sources(data):
        source_patients = data.source_patient_indices[source]
        train_count, validation_count, downstream_count = _fixed_development_counts(
            int(source_patients.size),
            train_ratio=development_ratios[0],
            validation_ratio=development_ratios[1],
            downstream_ratio=development_ratios[2],
            downstream_role_ratios=role_ratios,
        )
        downstream = _fixed_downstream_patients(
            data,
            source,
            development_split_seed=config.development_split_seed,
            development_ratios=development_ratios,
            downstream_role_ratios=role_ratios,
        )
        for patient_index in sorted(
            (int(index) for index in downstream),
            key=lambda index: data.cluster_patient_ids[index],
        ):
            fixed_membership.update(source.encode("utf-8"))
            fixed_membership.update(b"\0")
            fixed_membership.update(data.cluster_patient_ids[patient_index].encode("utf-8"))
            fixed_membership.update(b"\n")
        capped = min(
            downstream_count,
            config.max_patients_per_source
            if config.max_patients_per_source is not None else downstream_count,
        )
        for patient_index in downstream[:capped]:
            included_membership.update(source.encode("utf-8"))
            included_membership.update(b"\0")
            included_membership.update(
                data.cluster_patient_ids[int(patient_index)].encode("utf-8")
            )
            included_membership.update(b"\n")
        capped_counts.append(capped)
        source_rows.append({
            "source": source,
            "all_export_patients": int(source_patients.size),
            "adapter_train_patients": train_count,
            "adapter_validation_patients": validation_count,
            "eligible_downstream_patients": downstream_count,
            "included_after_source_cap": capped,
        })

    allocations = _source_partition_sizes(capped_counts, role_ratios)
    for row, (router, specialist, test) in zip(source_rows, allocations, strict=True):
        row["router_calibration_patients"] = router
        row["specialist_calibration_patients"] = specialist
        row["held_out_test_patients"] = test

    partitions = _patient_partitions(
        data,
        epoch=0,
        seed=config.seed,
        ratios=role_ratios,
        max_patients_per_source=config.max_patients_per_source,
        development_split_seed=config.development_split_seed,
        development_ratios=development_ratios,
    )
    role_membership = hashlib.sha256()
    for role, partition in zip(("CR", "CS", "T"), partitions, strict=True):
        for patient_index in sorted(
            (int(index) for index in partition),
            key=lambda index: (
                data.patient_sources[index], data.cluster_patient_ids[index]
            ),
        ):
            role_membership.update(role.encode("ascii"))
            role_membership.update(b"\0")
            role_membership.update(data.patient_sources[patient_index].encode("utf-8"))
            role_membership.update(b"\0")
            role_membership.update(data.cluster_patient_ids[patient_index].encode("utf-8"))
            role_membership.update(b"\n")

    totals = {
        "all_export_patients": sum(row["all_export_patients"] for row in source_rows),
        "adapter_train_patients": sum(row["adapter_train_patients"] for row in source_rows),
        "adapter_validation_patients": sum(
            row["adapter_validation_patients"] for row in source_rows
        ),
        "eligible_downstream_patients": sum(
            row["eligible_downstream_patients"] for row in source_rows
        ),
        "included_after_source_cap": sum(
            row["included_after_source_cap"] for row in source_rows
        ),
        "router_calibration_patients": sum(
            row["router_calibration_patients"] for row in source_rows
        ),
        "specialist_calibration_patients": sum(
            row["specialist_calibration_patients"] for row in source_rows
        ),
        "held_out_test_patients": sum(
            row["held_out_test_patients"] for row in source_rows
        ),
    }
    return {
        "eligibility_algorithm": "stable_sha256_patient_rank_v1",
        "allocation_algorithm": "source_stratified_largest_remainder_v1",
        "development_split_seed": config.development_split_seed,
        "development_ratios": {
            "adapter_train": config.development_train_ratio,
            "adapter_validation": config.development_validation_ratio,
            "downstream": config.development_downstream_ratio,
        },
        "downstream_role_seed": config.seed,
        "downstream_role_ratios": {
            "CR": config.router_calib_ratio,
            "CS": config.specialist_calib_ratio,
            "T": config.test_ratio,
        },
        "max_patients_per_source": config.max_patients_per_source,
        "fixed_downstream_membership_sha256": fixed_membership.hexdigest(),
        "included_after_cap_membership_sha256": included_membership.hexdigest(),
        "repetition_0_role_membership_sha256": role_membership.hexdigest(),
        "sources": source_rows,
        "totals": totals,
    }


def clustered_reference(
    data: ClusteredDecisionScores,
    config: ClusteredExperimentConfig,
    *,
    patient_balanced: bool | None = None,
) -> dict:
    ratios = (
        config.router_calib_ratio,
        config.specialist_calib_ratio,
        config.test_ratio,
    )
    if patient_balanced is None:
        patient_balanced = config.patient_balanced_primary
    return pooled_reference(
        data,
        beta=config.beta,
        alpha=config.effective_calibration_alpha,
        reporting_alpha=config.alpha,
        n_splits=config.n_splits,
        seed=config.seed,
        ratios=ratios,
        split_function=partial(
            patient_balanced_split if patient_balanced else clustered_decision_split,
            max_patients_per_source=config.max_patients_per_source,
            development_split_seed=config.development_split_seed,
            development_ratios=(
                config.development_train_ratio,
                config.development_validation_ratio,
                config.development_downstream_ratio,
            ),
        ),
    )


def clustered_bootstrap(
    data: ClusteredDecisionScores,
    reference: dict,
    config: ClusteredExperimentConfig,
    *,
    patient_balanced: bool | None = None,
) -> tuple[dict, list[dict]]:
    ratios = (
        config.router_calib_ratio,
        config.specialist_calib_ratio,
        config.test_ratio,
    )
    if patient_balanced is None:
        patient_balanced = config.patient_balanced_primary
    return patient_cluster_bootstrap(
        data,
        reference,
        beta=config.beta,
        alpha=config.effective_calibration_alpha,
        reporting_alpha=config.alpha,
        n_replicates=config.n_bootstrap,
        seed=config.seed + 100_000,
        ratios=ratios,
        bootstrap_function=partial(
            patient_balanced_bootstrap_partitions
            if patient_balanced else clustered_bootstrap_partitions,
            max_patients_per_source=config.max_patients_per_source,
            development_split_seed=config.development_split_seed,
            development_ratios=(
                config.development_train_ratio,
                config.development_validation_ratio,
                config.development_downstream_ratio,
            ),
        ),
    )
