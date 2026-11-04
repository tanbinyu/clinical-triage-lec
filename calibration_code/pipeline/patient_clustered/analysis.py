"""Patient-disjoint splitting and cluster resampling for decision-level scores."""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np

from pipeline.patient_level.analysis import (
    _partition_sizes,
    patient_cluster_bootstrap,
    pooled_reference,
)

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
    router_calib_ratio: float = 0.40
    specialist_calib_ratio: float = 0.30
    test_ratio: float = 0.30
    # Optional source-balancing rule applied before role assignment.  A fresh,
    # score- and outcome-blind patient sample is drawn within every source and
    # split, then divided into C_R/C_S/T using the configured ratios.
    max_patients_per_source: int | None = None
    seed: int = 20260929
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
            "test_only", "all_available_records", "unverified"
        }:
            raise ValueError(
                "input_cohort_scope must be test_only, all_available_records or unverified"
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
        if (
            self.max_patients_per_source is not None
            and self.max_patients_per_source < 3
        ):
            raise ValueError("max_patients_per_source must be at least three")
        if self.patient_aggregation not in {"mean", "median", "max"}:
            raise ValueError("patient_aggregation must be mean, median or max")
        if not np.isclose(self.lambda_1 + self.lambda_2, 1.0):
            raise ValueError("lambda_1 and lambda_2 must sum to one")


def _patient_partitions(
    data: ClusteredDecisionScores,
    *,
    epoch: int,
    seed: int,
    ratios: tuple[float, float, float],
    max_patients_per_source: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(np.random.SeedSequence([seed, epoch]))
    router_parts, specialist_parts, test_parts = [], [], []
    for source in sorted(data.source_patient_indices):
        patients = data.source_patient_indices[source].copy()
        rng.shuffle(patients)
        if max_patients_per_source is not None:
            patients = patients[:max_patients_per_source]
        n_router, n_specialist, _ = _partition_sizes(patients.size, ratios)
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assign patients to roles and include every genuine decision in that role."""
    partitions = _patient_partitions(
        data, epoch=epoch, seed=seed, ratios=ratios,
        max_patients_per_source=max_patients_per_source,
    )
    return tuple(_expand_all(data, patients) for patients in partitions)  # type: ignore[return-value]


def patient_balanced_split(
    data: ClusteredDecisionScores,
    *,
    epoch: int,
    seed: int,
    ratios: tuple[float, float, float],
    max_patients_per_source: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select one decision uniformly per patient after patient-disjoint assignment."""
    partitions = _patient_partitions(
        data, epoch=epoch, seed=seed, ratios=ratios,
        max_patients_per_source=max_patients_per_source,
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    partitions = _patient_partitions(
        data, epoch=replicate, seed=seed, ratios=ratios,
        max_patients_per_source=max_patients_per_source,
    )
    patient_sources = np.asarray(data.patient_sources, dtype=object)
    outputs = []
    for role, partition in enumerate(partitions):
        rng = np.random.default_rng(np.random.SeedSequence([seed, replicate, 991, role]))
        drawn_parts = []
        for source in sorted(data.source_patient_indices):
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return _cluster_bootstrap(
        data, replicate=replicate, seed=seed, ratios=ratios,
        one_per_patient=False,
        max_patients_per_source=max_patients_per_source,
    )


def patient_balanced_bootstrap_partitions(
    data: ClusteredDecisionScores,
    *,
    replicate: int,
    seed: int,
    ratios: tuple[float, float, float],
    max_patients_per_source: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return _cluster_bootstrap(
        data, replicate=replicate, seed=seed, ratios=ratios,
        one_per_patient=True,
        max_patients_per_source=max_patients_per_source,
    )


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
        ),
    )
