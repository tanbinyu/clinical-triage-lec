"""Patient-level baseline and sensitivity calculations for the full suite.

The functions in this module intentionally mirror the record-level experiments
while using one aggregated row per patient and mutually exclusive router
calibration, specialist calibration and test patients.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .analysis import (
    PatientExperimentConfig,
    _lec_calibrate,
    _metrics_from_counts,
    patient_disjoint_split,
)
from .data import PatientScores
from .metadata import DISEASE_NAMES, NUM_DISEASES


ALPHA_GRID = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.40, 0.80]
BETA_GRID = [0.005, 0.01, 0.02, 0.025, 0.03, 0.04, 0.05, 0.10, 0.15, 0.20]
CALIBRATION_FRACTIONS = [0.25, 0.50, 0.75, 1.00]
METHODS = (
    "full_automation",
    "router_only",
    "specialist_only",
    "shared_threshold",
    "oracle_clean",
    "proposed",
)
METHOD_LABELS = {
    "full_automation": "Full automation",
    "router_only": "Router-only rejection",
    "specialist_only": "Routed specialist-only",
    "shared_threshold": "Shared threshold",
    "oracle_clean": "Oracle-clean calibration",
    "proposed": "Proposed LEC",
}


def _empty_counts() -> dict[str, Any]:
    return {
        "total": 0, "routed": 0,
        "selected": 0,
        "errors": 0,
        "per_clinic": {
            str(k): {"total": 0, "routed": 0, "selected": 0, "errors": 0}
            for k in range(NUM_DISEASES)
        },
    }


def _add_counts(target: dict[str, Any], source: dict[str, Any]) -> None:
    for key in ("total", "routed", "selected", "errors"):
        target[key] += int(source[key])
    for k in range(NUM_DISEASES):
        for key in ("total", "routed", "selected", "errors"):
            target["per_clinic"][str(k)][key] += int(
                source["per_clinic"][str(k)][key]
            )


def _counts(
    data: PatientScores,
    test_idx: np.ndarray,
    accepted: np.ndarray,
    *,
    method: str = "proposed",
    routed: np.ndarray | None = None,
) -> dict[str, Any]:
    correctness = data.system_correctness
    pathway = data.predicted_disease
    errors = accepted & (correctness[test_idx] == 0)
    routed_mask = (
        np.ones(test_idx.size, dtype=bool) if routed is None else routed
    )
    result = {
        "total": int(test_idx.size),
        "routed": int(routed_mask.sum()),
        "selected": int(accepted.sum()),
        "errors": int(errors.sum()),
        "per_clinic": {},
    }
    for k in range(NUM_DISEASES):
        clinic = pathway[test_idx] == k
        result["per_clinic"][str(k)] = {
            "total": int(clinic.sum()),
            "routed": int((routed_mask & clinic).sum()),
            "selected": int((accepted & clinic).sum()),
            "errors": int((errors & clinic).sum()),
        }
    return result


def _clinic_thresholds(
    data: PatientScores,
    calib_idx: np.ndarray,
    eligible: np.ndarray,
    alpha: float,
    specialist_only: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    thresholds = np.full(NUM_DISEASES, np.nan, dtype=float)
    sizes = np.zeros(NUM_DISEASES, dtype=int)
    pathway = data.predicted_disease
    score = data.specialist_score
    correctness = data.system_correctness
    for k in range(NUM_DISEASES):
        mask = eligible & (pathway[calib_idx] == k)
        idx = calib_idx[mask]
        sizes[k] = idx.size
        threshold = _lec_calibrate(
            score[idx], correctness[idx], alpha
        )
        if threshold is not None:
            thresholds[k] = threshold
    return thresholds, sizes


def _accept_by_thresholds(
    data: PatientScores,
    test_idx: np.ndarray,
    thresholds: np.ndarray,
    routed: np.ndarray,
    specialist_only: bool = False,
) -> np.ndarray:
    pathway = data.predicted_disease
    score = data.specialist_score
    clinic_threshold = thresholds[pathway[test_idx]]
    return routed & np.isfinite(clinic_threshold) & (
        score[test_idx] <= clinic_threshold
    )


def _split_methods(
    data: PatientScores,
    router_idx: np.ndarray,
    specialist_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    beta: float,
    alpha: float,
) -> dict[str, Any]:
    router_threshold = _lec_calibrate(
        data.router_uncertainty[router_idx], data.router_correctness[router_idx], beta
    )
    if router_threshold is None:
        routed_calib = np.zeros(specialist_idx.size, dtype=bool)
        routed_test = np.zeros(test_idx.size, dtype=bool)
    else:
        routed_calib = data.router_uncertainty[specialist_idx] <= router_threshold
        routed_test = data.router_uncertainty[test_idx] <= router_threshold

    all_calib = np.ones(specialist_idx.size, dtype=bool)
    proposed_thresholds, proposed_sizes = _clinic_thresholds(
        data, specialist_idx, routed_calib, alpha
    )
    specialist_thresholds, specialist_sizes = _clinic_thresholds(
        data, specialist_idx, all_calib, alpha, specialist_only=True
    )
    clean_thresholds, _ = _clinic_thresholds(
        data,
        specialist_idx,
        routed_calib & (data.router_correctness[specialist_idx] == 1),
        alpha,
    )
    pooled_idx = specialist_idx[routed_calib]
    shared_threshold = _lec_calibrate(
        data.specialist_score[pooled_idx], data.system_correctness[pooled_idx], alpha
    )
    shared_accepted = np.zeros(test_idx.size, dtype=bool)
    if shared_threshold is not None:
        shared_accepted = routed_test & (
            data.specialist_score[test_idx] <= shared_threshold
        )

    accepted = {
        "full_automation": np.ones(test_idx.size, dtype=bool),
        "router_only": routed_test.copy(),
        "specialist_only": _accept_by_thresholds(
            data, test_idx, specialist_thresholds, np.ones(test_idx.size, dtype=bool),
            specialist_only=True,
        ),
        "shared_threshold": shared_accepted,
        "oracle_clean": _accept_by_thresholds(
            data, test_idx, clean_thresholds, routed_test
        ),
        "proposed": _accept_by_thresholds(
            data, test_idx, proposed_thresholds, routed_test
        ),
    }
    return {
        "router_threshold": router_threshold,
        "routed_calib": routed_calib,
        "routed_test": routed_test,
        "proposed_thresholds": proposed_thresholds,
        "proposed_calibration_sizes": proposed_sizes,
        "specialist_thresholds": specialist_thresholds,
        "specialist_only_calibration_sizes": specialist_sizes,
        "clean_thresholds": clean_thresholds,
        "shared_threshold": shared_threshold,
        "accepted": accepted,
    }


def _pooled_method_summary(counts: dict[str, Any], alpha: float) -> dict[str, Any]:
    metric = _metrics_from_counts(counts)
    per_clinic = []
    for k in range(NUM_DISEASES):
        block = counts["per_clinic"][str(k)]
        risk = block["errors"] / block["selected"] if block["selected"] else None
        coverage = block["selected"] / block["total"] if block["total"] else None
        conditional_coverage = (
            block["selected"] / block["routed"] if block["routed"] else None
        )
        per_clinic.append({
            "disease_id": k,
            "disease_name": DISEASE_NAMES[k],
            "mFDR": risk,
            "coverage": coverage,
            "conditional_specialist_coverage": conditional_coverage,
            "referral": 1 - coverage if coverage is not None else None,
            **block,
        })
    estimable = [row["mFDR"] for row in per_clinic if row["mFDR"] is not None]
    metric.update({
        "mFDR_macro": float(np.mean(estimable)) if estimable else None,
        "mFDR_worst": float(np.max(estimable)) if estimable else None,
        "K_meet": int(sum(value <= alpha for value in estimable)),
        "K_estimable": len(estimable),
        "per_disease": per_clinic,
    })
    return metric


def _split_metric(counts: dict[str, Any]) -> tuple[float | None, float]:
    risk = counts["errors"] / counts["selected"] if counts["selected"] else None
    coverage = counts["selected"] / counts["total"] if counts["total"] else 0.0
    return risk, coverage


def _distribution(values: list[float | None]) -> dict[str, Any]:
    finite = np.asarray([value for value in values if value is not None], dtype=float)
    if finite.size == 0:
        return {
            "mean": None, "sd": None, "median": None, "min": None,
            "max": None, "p2.5": None, "p97.5": None, "n": 0,
        }
    return {
        "mean": float(np.mean(finite)),
        "sd": float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0,
        "median": float(np.median(finite)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "p2.5": float(np.percentile(finite, 2.5)),
        "p97.5": float(np.percentile(finite, 97.5)),
        "n": int(finite.size),
    }


def _ks_statistic(left: np.ndarray, right: np.ndarray) -> float | None:
    if left.size == 0 or right.size == 0:
        return None
    left = np.sort(left)
    right = np.sort(right)
    grid = np.sort(np.concatenate([left, right]))
    cdf_left = np.searchsorted(left, grid, side="right") / left.size
    cdf_right = np.searchsorted(right, grid, side="right") / right.size
    return float(np.max(np.abs(cdf_left - cdf_right)))


def primary_suite(
    data: PatientScores,
    config: PatientExperimentConfig,
    ratios: tuple[float, float, float],
    *,
    split_function: Callable[..., tuple[np.ndarray, np.ndarray, np.ndarray]] | None = None,
    calibration_alpha: float | None = None,
) -> dict[str, Any]:
    splitter = split_function or patient_disjoint_split
    fitted_alpha = config.alpha if calibration_alpha is None else calibration_alpha
    method_counts = {method: _empty_counts() for method in METHODS}
    method_split_values = {
        method: {"mFDR": [], "coverage": []} for method in METHODS
    }
    router_totals = {"total": 0, "selected": 0, "errors": 0}
    router_split_values = {"mFDR": [], "coverage": []}
    router_thresholds: list[float | None] = []
    router_calibration_sizes: list[int] = []
    specialist_thresholds = {k: [] for k in range(NUM_DISEASES)}
    specialist_only_thresholds = {k: [] for k in range(NUM_DISEASES)}
    specialist_only_zero_acceptance = np.zeros(NUM_DISEASES, dtype=int)
    calibration_sizes = {k: [] for k in range(NUM_DISEASES)}
    proposed_clinic_fdp = {k: [] for k in range(NUM_DISEASES)}
    proposed_conditional_coverage: list[float] = []
    contamination = {
        k: defaultdict(int) for k in range(NUM_DISEASES)
    }
    method_contamination = {
        method: {k: defaultdict(int) for k in range(NUM_DISEASES)}
        for method in METHODS
    }
    exchangeability = {
        k: {
            "base_cal_errors": 0, "base_cal_total": 0,
            "base_test_errors": 0, "base_test_total": 0,
            "fdp_cal_errors": 0, "fdp_cal_selected": 0,
            "fdp_test_errors": 0, "fdp_test_selected": 0,
            "ks": [], "ks_reference_95": [],
        }
        for k in range(NUM_DISEASES)
    }

    for epoch in range(config.n_splits):
        router_idx, specialist_idx, test_idx = splitter(
            data, epoch=epoch, seed=config.seed, ratios=ratios
        )
        state = _split_methods(
            data, router_idx, specialist_idx, test_idx,
            beta=config.beta, alpha=fitted_alpha,
        )
        router_thresholds.append(state["router_threshold"])
        router_calibration_sizes.append(int(router_idx.size))
        routed_test = state["routed_test"]
        split_router_selected = int(routed_test.sum())
        split_router_errors = int((
            routed_test & (data.router_correctness[test_idx] == 0)
        ).sum())
        router_totals["total"] += int(test_idx.size)
        router_totals["selected"] += split_router_selected
        router_totals["errors"] += split_router_errors
        router_split_values["mFDR"].append(
            split_router_errors / split_router_selected
            if split_router_selected else None
        )
        router_split_values["coverage"].append(
            split_router_selected / test_idx.size
        )

        split_counts: dict[str, dict[str, Any]] = {}
        for method in METHODS:
            counts = _counts(
                data, test_idx, state["accepted"][method], method=method,
                routed=(state["routed_test"] if method == "proposed" else None),
            )
            split_counts[method] = counts
            _add_counts(method_counts[method], counts)
            risk, coverage = _split_metric(counts)
            method_split_values[method]["mFDR"].append(risk)
            method_split_values[method]["coverage"].append(coverage)
            accepted_method = state["accepted"][method]
            pathway = data.predicted_disease
            for k in range(NUM_DISEASES):
                accepted_clinic = accepted_method & (
                    pathway[test_idx] == k
                )
                wrong_route = accepted_clinic & (
                    data.router_correctness[test_idx] == 0
                )
                diagnostic_error = accepted_clinic & (
                    (data.router_correctness[test_idx] == 1)
                    & (data.specialist_correctness[test_idx] == 0)
                )
                block = method_contamination[method][k]
                block["accepted"] += int(accepted_clinic.sum())
                block["accepted_wrong_route"] += int(wrong_route.sum())
                block["accepted_diagnostic_error"] += int(diagnostic_error.sum())

        proposed_conditional_coverage.append(
            split_counts["proposed"]["selected"] / split_router_selected
            if split_router_selected else 0.0
        )
        for k in range(NUM_DISEASES):
            clinic_counts = split_counts["proposed"]["per_clinic"][str(k)]
            proposed_clinic_fdp[k].append(
                clinic_counts["errors"] / clinic_counts["selected"]
                if clinic_counts["selected"] else None
            )

        proposed_accepted = state["accepted"]["proposed"]
        for k in range(NUM_DISEASES):
            threshold = state["proposed_thresholds"][k]
            specialist_thresholds[k].append(
                None if not np.isfinite(threshold) else float(threshold)
            )
            specialist_only_threshold = state["specialist_thresholds"][k]
            specialist_only_thresholds[k].append(
                None if not np.isfinite(specialist_only_threshold)
                else float(specialist_only_threshold)
            )
            specialist_only_zero_acceptance[k] += int(
                split_counts["specialist_only"]["per_clinic"][str(k)]["selected"] == 0
            )
            calibration_sizes[k].append(int(state["proposed_calibration_sizes"][k]))
            cal_clinic = state["routed_calib"] & (
                data.predicted_disease[specialist_idx] == k
            )
            test_clinic = routed_test & (data.predicted_disease[test_idx] == k)
            accepted_clinic = proposed_accepted & (
                data.predicted_disease[test_idx] == k
            )
            wrong_routed = test_clinic & (data.router_correctness[test_idx] == 0)
            accepted_wrong = accepted_clinic & (data.router_correctness[test_idx] == 0)
            accepted_diag_error = accepted_clinic & (
                (data.router_correctness[test_idx] == 1)
                & (data.specialist_correctness[test_idx] == 0)
            )
            block = contamination[k]
            block["routed"] += int(test_clinic.sum())
            block["wrong_routed"] += int(wrong_routed.sum())
            block["accepted"] += int(accepted_clinic.sum())
            block["accepted_wrong"] += int(accepted_wrong.sum())
            block["accepted_diag_error"] += int(accepted_diag_error.sum())

            ex = exchangeability[k]
            ex["base_cal_total"] += int(cal_clinic.sum())
            ex["base_cal_errors"] += int((
                cal_clinic & (data.system_correctness[specialist_idx] == 0)
            ).sum())
            ex["base_test_total"] += int(test_clinic.sum())
            ex["base_test_errors"] += int((
                test_clinic & (data.system_correctness[test_idx] == 0)
            ).sum())
            if np.isfinite(threshold):
                cal_acc = cal_clinic & (data.specialist_score[specialist_idx] <= threshold)
                ex["fdp_cal_selected"] += int(cal_acc.sum())
                ex["fdp_cal_errors"] += int((
                    cal_acc & (data.system_correctness[specialist_idx] == 0)
                ).sum())
                ex["fdp_test_selected"] += int(accepted_clinic.sum())
                ex["fdp_test_errors"] += int((
                    accepted_clinic & (data.system_correctness[test_idx] == 0)
                ).sum())
            cal_scores = data.specialist_score[specialist_idx[cal_clinic]]
            test_scores = data.specialist_score[test_idx[test_clinic]]
            ks = _ks_statistic(cal_scores, test_scores)
            if ks is not None:
                ex["ks"].append(ks)
                ex["ks_reference_95"].append(
                    1.36 * np.sqrt((cal_scores.size + test_scores.size)
                                   / (cal_scores.size * test_scores.size))
                )

    methods = {}
    for method in METHODS:
        methods[method] = _pooled_method_summary(method_counts[method], config.alpha)
        methods[method]["split_mFDR"] = _distribution(
            method_split_values[method]["mFDR"]
        )
        methods[method]["split_coverage"] = _distribution(
            method_split_values[method]["coverage"]
        )
        accepted_total = 0
        wrong_route_total = 0
        for k, row in enumerate(methods[method]["per_disease"]):
            block = method_contamination[method][k]
            accepted_total += block["accepted"]
            wrong_route_total += block["accepted_wrong_route"]
            row["accepted_wrong_route_rate"] = (
                block["accepted_wrong_route"] / block["accepted"]
                if block["accepted"] else None
            )
            row["accepted_wrong_route"] = block["accepted_wrong_route"]
            row["accepted_diagnostic_error"] = block["accepted_diagnostic_error"]
        methods[method]["accepted_wrong_route_rate"] = (
            wrong_route_total / accepted_total if accepted_total else None
        )
        methods[method]["accepted_wrong_route"] = wrong_route_total
    methods["proposed"]["conditional_specialist_coverage"] = (
        method_counts["proposed"]["selected"] / router_totals["selected"]
        if router_totals["selected"] else None
    )
    methods["proposed"]["split_conditional_specialist_coverage"] = _distribution(
        proposed_conditional_coverage
    )
    for k, row in enumerate(methods["proposed"]["per_disease"]):
        row["split_FDP"] = _distribution(proposed_clinic_fdp[k])

    router = {
        "mFDR": (router_totals["errors"] / router_totals["selected"]
                  if router_totals["selected"] else None),
        "coverage": router_totals["selected"] / router_totals["total"],
        "referral": 1 - router_totals["selected"] / router_totals["total"],
        "threshold": _distribution(router_thresholds),
        "effective_size": _distribution(router_calibration_sizes),
        "split_mFDR": _distribution(router_split_values["mFDR"]),
        "split_coverage": _distribution(router_split_values["coverage"]),
        "infeasible_splits": int(sum(value is None for value in router_thresholds)),
        **router_totals,
    }
    calibration = []
    for k in range(NUM_DISEASES):
        values = np.asarray(calibration_sizes[k], dtype=float)
        calibration.append({
            "disease_id": k,
            "disease_name": DISEASE_NAMES[k],
            "effective_size": {
                "mean": float(np.mean(values)),
                "sd": float(np.std(values, ddof=1)),
                "median": float(np.median(values)),
                "min": int(np.min(values)),
                "p2.5": float(np.percentile(values, 2.5)),
                "p97.5": float(np.percentile(values, 97.5)),
            },
            "threshold": _distribution(specialist_thresholds[k]),
            "feasible_splits": int(sum(value is not None for value in specialist_thresholds[k])),
            "infeasible_frequency": float(np.mean([
                value is None for value in specialist_thresholds[k]
            ])),
        })

    contamination_rows = []
    exchangeability_rows = []
    for k in range(NUM_DISEASES):
        block = contamination[k]
        accepted_errors = block["accepted_wrong"] + block["accepted_diag_error"]
        contamination_rows.append({
            "disease_id": k,
            "disease_name": DISEASE_NAMES[k],
            "rho_route": block["wrong_routed"] / block["routed"] if block["routed"] else None,
            "rho_acc": block["accepted_wrong"] / block["accepted"] if block["accepted"] else None,
            "wrong_route_referral": (
                (block["wrong_routed"] - block["accepted_wrong"]) / block["wrong_routed"]
                if block["wrong_routed"] else None
            ),
            "route_error_share": block["accepted_wrong"] / accepted_errors if accepted_errors else None,
            "diagnostic_error_share": block["accepted_diag_error"] / accepted_errors if accepted_errors else None,
            **dict(block),
        })
        ex = exchangeability[k]
        exchangeability_rows.append({
            "disease_id": k,
            "disease_name": DISEASE_NAMES[k],
            "base_error_cal": ex["base_cal_errors"] / ex["base_cal_total"] if ex["base_cal_total"] else None,
            "base_error_test": ex["base_test_errors"] / ex["base_test_total"] if ex["base_test_total"] else None,
            "FDP_cal": ex["fdp_cal_errors"] / ex["fdp_cal_selected"] if ex["fdp_cal_selected"] else None,
            "FDP_test": ex["fdp_test_errors"] / ex["fdp_test_selected"] if ex["fdp_test_selected"] else None,
            "KS_mean": float(np.mean(ex["ks"])) if ex["ks"] else None,
            "KS_p95": float(np.percentile(ex["ks"], 95)) if ex["ks"] else None,
            "KS_reference_95_mean": float(np.mean(ex["ks_reference_95"])) if ex["ks_reference_95"] else None,
            "mean_effective_calibration_size": calibration[k]["effective_size"]["mean"],
        })
    return {
        "router": router,
        "methods": methods,
        "calibration": calibration,
        "contamination": contamination_rows,
        "exchangeability": exchangeability_rows,
        "diagnostics": {
            "specialist_only_feasible_splits_by_pathway": [
                int(sum(value is not None for value in specialist_only_thresholds[k]))
                for k in range(NUM_DISEASES)
            ],
            "specialist_only_infeasible_frequency_by_pathway": [
                float(np.mean([
                    value is None for value in specialist_only_thresholds[k]
                ])) for k in range(NUM_DISEASES)
            ],
            "specialist_only_zero_acceptance_frequency_by_pathway": (
                specialist_only_zero_acceptance / config.n_splits
            ).tolist(),
        },
    }


def alpha_baseline_grid(
    data: PatientScores,
    config: PatientExperimentConfig,
    ratios: tuple[float, float, float],
    *,
    split_function: Callable[..., tuple[np.ndarray, np.ndarray, np.ndarray]] | None = None,
    calibration_alpha_scale: float = 1.0,
) -> list[dict[str, Any]]:
    splitter = split_function or patient_disjoint_split
    totals = {
        alpha: {method: _empty_counts() for method in METHODS}
        for alpha in ALPHA_GRID
    }
    split_values = {
        alpha: {
            method: {"mFDR": [], "coverage": []}
            for method in METHODS
        }
        for alpha in ALPHA_GRID
    }
    contamination = {
        alpha: {k: defaultdict(int) for k in range(NUM_DISEASES)}
        for alpha in ALPHA_GRID
    }
    calibration_fdp = {
        alpha: {k: defaultdict(int) for k in range(NUM_DISEASES)}
        for alpha in ALPHA_GRID
    }
    for epoch in range(config.n_splits):
        router, specialist, test = splitter(
            data, epoch=epoch, seed=config.seed, ratios=ratios
        )
        for alpha in ALPHA_GRID:
            state = _split_methods(
                data, router, specialist, test,
                beta=config.beta, alpha=alpha * calibration_alpha_scale,
            )
            for method in METHODS:
                counts = _counts(
                    data, test, state["accepted"][method], method=method,
                    routed=(state["routed_test"] if method == "proposed" else None),
                )
                _add_counts(totals[alpha][method], counts)
                risk, coverage = _split_metric(counts)
                split_values[alpha][method]["mFDR"].append(risk)
                split_values[alpha][method]["coverage"].append(coverage)
            proposed = state["accepted"]["proposed"]
            for k in range(NUM_DISEASES):
                routed = state["routed_test"] & (data.predicted_disease[test] == k)
                accepted = proposed & (data.predicted_disease[test] == k)
                threshold = state["proposed_thresholds"][k]
                cal_routed = state["routed_calib"] & (
                    data.predicted_disease[specialist] == k
                )
                cal_accepted = np.zeros(specialist.size, dtype=bool)
                if np.isfinite(threshold):
                    cal_accepted = cal_routed & (
                        data.specialist_score[specialist] <= threshold
                    )
                cal_block = calibration_fdp[alpha][k]
                cal_block["selected"] += int(cal_accepted.sum())
                cal_block["errors"] += int((
                    cal_accepted & (data.system_correctness[specialist] == 0)
                ).sum())
                wrong = routed & (data.router_correctness[test] == 0)
                accepted_wrong = accepted & (data.router_correctness[test] == 0)
                diag_error = accepted & (data.router_correctness[test] == 1) & (
                    data.specialist_correctness[test] == 0
                )
                block = contamination[alpha][k]
                block["routed"] += int(routed.sum())
                block["wrong"] += int(wrong.sum())
                block["accepted"] += int(accepted.sum())
                block["accepted_wrong"] += int(accepted_wrong.sum())
                block["diag_error"] += int(diag_error.sum())
    rows = []
    for alpha in ALPHA_GRID:
        method_rows = {
            method: _pooled_method_summary(totals[alpha][method], alpha)
            for method in METHODS
        }
        for method in METHODS:
            method_rows[method]["split_mFDR"] = _distribution(
                split_values[alpha][method]["mFDR"]
            )
            method_rows[method]["split_coverage"] = _distribution(
                split_values[alpha][method]["coverage"]
            )
        contamination_rows = []
        exchange_rows = []
        for k in range(NUM_DISEASES):
            block = contamination[alpha][k]
            contamination_rows.append({
                "disease_id": k,
                "rho_route": block["wrong"] / block["routed"] if block["routed"] else None,
                "rho_acc": block["accepted_wrong"] / block["accepted"] if block["accepted"] else None,
                "wrong_route_referral": (
                    (block["wrong"] - block["accepted_wrong"]) / block["wrong"]
                    if block["wrong"] else None
                ),
                "route_errors": block["accepted_wrong"],
                "diagnostic_errors": block["diag_error"],
            })
            cal_block = calibration_fdp[alpha][k]
            test_row = method_rows["proposed"]["per_disease"][k]
            fdp_cal = (
                cal_block["errors"] / cal_block["selected"]
                if cal_block["selected"] else None
            )
            fdp_test = test_row["mFDR"]
            exchange_rows.append({
                "disease_id": k,
                "FDP_cal": fdp_cal,
                "FDP_test": fdp_test,
                "excess_FDP": (
                    fdp_test - fdp_cal
                    if fdp_test is not None and fdp_cal is not None else None
                ),
            })
        rows.append({
            "alpha": alpha,
            "calibration_alpha": alpha * calibration_alpha_scale,
            "methods": method_rows,
            "contamination": contamination_rows,
            "exchangeability": exchange_rows,
        })
    return rows


def beta_grid(
    data: PatientScores,
    config: PatientExperimentConfig,
    ratios: tuple[float, float, float],
    *,
    split_function: Callable[..., tuple[np.ndarray, np.ndarray, np.ndarray]] | None = None,
    calibration_alpha: float | None = None,
) -> list[dict[str, Any]]:
    splitter = split_function or patient_disjoint_split
    fitted_alpha = config.alpha if calibration_alpha is None else calibration_alpha
    rows = []
    for beta in BETA_GRID:
        proposed_totals = _empty_counts()
        specialist_totals = _empty_counts()
        full_totals = _empty_counts()
        router_total = router_selected = router_errors = 0
        contamination = {k: defaultdict(int) for k in range(NUM_DISEASES)}
        for epoch in range(config.n_splits):
            router, specialist, test = splitter(
                data, epoch=epoch, seed=config.seed, ratios=ratios
            )
            state = _split_methods(
                data, router, specialist, test,
                beta=beta, alpha=fitted_alpha,
            )
            for method, target in (
                ("proposed", proposed_totals),
                ("specialist_only", specialist_totals),
                ("full_automation", full_totals),
            ):
                _add_counts(
                    target,
                    _counts(
                        data, test, state["accepted"][method], method=method,
                        routed=(state["routed_test"] if method == "proposed" else None),
                    ),
                )
            routed = state["routed_test"]
            router_total += test.size
            router_selected += int(routed.sum())
            router_errors += int((routed & (data.router_correctness[test] == 0)).sum())
            accepted = state["accepted"]["proposed"]
            for k in range(NUM_DISEASES):
                clinic_route = routed & (data.predicted_disease[test] == k)
                clinic_accept = accepted & (data.predicted_disease[test] == k)
                block = contamination[k]
                block["routed"] += int(clinic_route.sum())
                block["wrong_routed"] += int((
                    clinic_route & (data.router_correctness[test] == 0)
                ).sum())
                block["accepted"] += int(clinic_accept.sum())
                block["accepted_wrong"] += int((
                    clinic_accept & (data.router_correctness[test] == 0)
                ).sum())
        proposed = _pooled_method_summary(proposed_totals, config.alpha)
        specialist_only = _pooled_method_summary(specialist_totals, config.alpha)
        full = _pooled_method_summary(full_totals, config.alpha)
        clinic_rho_route = []
        clinic_rho_acc = []
        for k in range(NUM_DISEASES):
            block = contamination[k]
            clinic_rho_route.append(
                block["wrong_routed"] / block["routed"] if block["routed"] else np.nan
            )
            clinic_rho_acc.append(
                block["accepted_wrong"] / block["accepted"] if block["accepted"] else np.nan
            )
        rows.append({
            "beta": beta,
            "router_mFDR": router_errors / router_selected if router_selected else None,
            "router_coverage": router_selected / router_total,
            "proposed": proposed,
            "specialist_only": specialist_only,
            "full_automation": full,
            "router_referral": 1 - router_selected / router_total,
            "specialist_referral": router_selected / router_total - proposed["coverage"],
            "rho_route_mean": float(np.nanmean(clinic_rho_route)),
            "rho_route_min": float(np.nanmin(clinic_rho_route)),
            "rho_route_max": float(np.nanmax(clinic_rho_route)),
            "rho_acc_mean": float(np.nanmean(clinic_rho_acc)),
        })
    return rows


def _subsample_by_source(
    data: PatientScores,
    indices: np.ndarray,
    fraction: float,
    rng: np.random.Generator,
) -> np.ndarray:
    sources = np.asarray(data.sources, dtype=object)
    parts = []
    for source in sorted(data.source_indices):
        block = indices[sources[indices] == source]
        n_keep = max(1, int(round(block.size * fraction)))
        parts.append(rng.permutation(block)[:n_keep])
    return np.concatenate(parts)


def calibration_size_grid(
    data: PatientScores,
    config: PatientExperimentConfig,
    ratios: tuple[float, float, float],
) -> list[dict[str, Any]]:
    rows = []
    for fraction in CALIBRATION_FRACTIONS:
        totals = _empty_counts()
        router_total = router_selected = router_errors = 0
        router_thresholds = []
        router_sub_sizes: list[int] = []
        specialist_sub_sizes: list[int] = []
        clinic_thresholds = {k: [] for k in range(NUM_DISEASES)}
        clinic_sizes = {k: [] for k in range(NUM_DISEASES)}
        for epoch in range(config.n_splits):
            router, specialist, test = patient_disjoint_split(
                data, epoch=epoch, seed=config.seed, ratios=ratios
            )
            rng = np.random.default_rng(np.random.SeedSequence([
                config.seed, epoch, int(fraction * 1000), 771,
            ]))
            router_sub = _subsample_by_source(data, router, fraction, rng)
            specialist_sub = _subsample_by_source(data, specialist, fraction, rng)
            router_sub_sizes.append(int(router_sub.size))
            specialist_sub_sizes.append(int(specialist_sub.size))
            state = _split_methods(
                data, router_sub, specialist_sub, test,
                beta=config.beta, alpha=config.alpha,
            )
            _add_counts(
                totals,
                _counts(
                    data, test, state["accepted"]["proposed"], method="proposed",
                    routed=state["routed_test"],
                ),
            )
            routed = state["routed_test"]
            router_total += test.size
            router_selected += int(routed.sum())
            router_errors += int((routed & (data.router_correctness[test] == 0)).sum())
            router_thresholds.append(state["router_threshold"])
            for k in range(NUM_DISEASES):
                threshold = state["proposed_thresholds"][k]
                clinic_thresholds[k].append(
                    None if not np.isfinite(threshold) else float(threshold)
                )
                clinic_sizes[k].append(int(state["proposed_calibration_sizes"][k]))
        system = _pooled_method_summary(totals, config.alpha)
        clinic_summary = []
        for k in range(NUM_DISEASES):
            finite = [x for x in clinic_thresholds[k] if x is not None]
            clinic_summary.append({
                "disease_id": k,
                "disease_name": DISEASE_NAMES[k],
                "mean_calibration_size": float(np.mean(clinic_sizes[k])),
                "threshold_sd": float(np.std(finite, ddof=1)) if len(finite) > 1 else None,
                "infeasible_frequency": float(np.mean([
                    value is None for value in clinic_thresholds[k]
                ])),
            })
        router_finite = [value for value in router_thresholds if value is not None]
        rows.append({
            "fraction": fraction,
            "mean_router_calibration_patients": float(np.mean(router_sub_sizes)),
            "mean_specialist_calibration_patients": float(np.mean(specialist_sub_sizes)),
            "router_mFDR": router_errors / router_selected if router_selected else None,
            "router_coverage": router_selected / router_total,
            "router_threshold_sd": (
                float(np.std(router_finite, ddof=1)) if len(router_finite) > 1 else None
            ),
            "router_infeasible_frequency": float(np.mean([
                value is None for value in router_thresholds
            ])),
            "system": system,
            "clinics": clinic_summary,
        })
    return rows


def utility_grid(primary: dict[str, Any]) -> dict[str, float]:
    proposed = primary["methods"]["proposed"]
    total = proposed["test_patient_decisions"]
    correct = proposed["selected"] - proposed["errors"]
    errors = proposed["errors"]
    referred = total - proposed["selected"]
    values = {}
    for c_error in (1, 5, 10, 20, 50):
        for c_referral in (0.1, 0.5, 1.0, 2.0):
            values[f"{c_error}|{c_referral}"] = (
                correct - c_error * errors - c_referral * referred
            ) / total
    return values


def run_full_suite(
    config: PatientExperimentConfig, *, primary_result: dict[str, Any] | None = None
) -> tuple[dict[str, Any], Path]:
    """Backward-compatible entry point; execution lives in ``runner``."""
    from .runner import run_full_suite as run

    return run(config, primary_result=primary_result)
