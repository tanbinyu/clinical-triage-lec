"""Baseline comparison across a grid of specialist risk levels (Methods 2.13).

At the alpha = 0.2 operating point alone the six systems are nearly
indistinguishable (proposed 0.1685 vs specialist-only 0.1687, and full
automation already sits below the target), so a single operating point cannot
support the necessity of clinic-specific routed-stream calibration. This module
sweeps alpha with beta held fixed and reports, for every system, the
system-level risk/coverage/referral, the macro, micro and worst-specialist
risks, the number of specialists meeting the target, the number of evaluable
specialists, and specialist infeasibility.

All six systems are evaluated through the same evaluator, on the same splits
and the same held-out test samples, so they differ only in which stages may
reject and in how the specialist thresholds are calibrated. "Accept
everything" is expressed as an infinite threshold rather than a separate code
path, which keeps per-clinic metrics available for the unfiltered systems too.

Usage:
    cd calibration_code
    python3 -m pipeline.metrics.baseline_alpha_grid
"""
import csv
import json
import os
import sys
import time

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pipeline.metrics.config import NUM_DISEASES, BETA, DISEASE_NAMES
from pipeline.metrics.compute import _lec_calibrate_np
from pipeline.metrics.compute_extra import (
    precompute_all_with_rc,
    _evaluate_specialist_system,
    _lambda_r_array,
)
from pipeline.metrics.data_loader import DataLoader

ALPHA_GRID = [0.05, 0.10, 0.15, 0.20]

OUT_JSON = os.path.join(PROJECT_ROOT, "result", "metrics", "baseline_alpha_grid.json")
OUT_CSV = os.path.join(PROJECT_ROOT, "result", "metrics", "baseline_alpha_grid.csv")

# Display order and provenance of the guarantee each system can claim. The
# scope strings follow Methods 2.13; an empirical risk below alpha is not by
# itself evidence of a calibration guarantee.
SYSTEMS = [
    ("full_automation", "Full automation", "None", False),
    ("router_only", "Router-only rejection", "Router only", False),
    ("specialist_only", "Specialist-only rejection",
     "Specialist and system, on unfiltered routed streams", True),
    ("shared_threshold", "Shared threshold",
     "Pooled system only, not clinic-specific", True),
    ("oracle_clean", "Oracle-clean calibration",
     "None on contaminated operational stream", True),
    ("proposed", "Proposed system (LEC)",
     "Router + clinic-specific specialists + system", True),
]


# ════════════════════════════════════════════════════════════════════
#  Per-system threshold matrices
# ════════════════════════════════════════════════════════════════════

def _accept_all_matrix(n_splits):
    """Thresholds that accept every technically processable case."""
    return np.full((NUM_DISEASES, n_splits), np.inf)


def _calibrate_per_clinic(calib, n_splits, alpha, base_mask, per_clinic_extra=None):
    """Run LEC once per clinic and split on the masked calibration stream."""
    t_hats = np.full((NUM_DISEASES, n_splits), np.nan)
    for k in range(NUM_DISEASES):
        mk = base_mask & (calib["pred_k"] == k)
        if per_clinic_extra is not None:
            mk = mk & per_clinic_extra
        for b in range(n_splits):
            mask = mk & (calib["split"] == b)
            if not mask.any():
                continue
            th = _lec_calibrate_np(calib["unc"][mask], calib["corr"][mask], alpha)
            if th is not None:
                t_hats[k, b] = th
    return t_hats


def _calibrate_shared(calib, n_splits, alpha, base_mask):
    """Pool every clinic's routed calibration stream and run LEC once."""
    shared = np.full(n_splits, np.nan)
    for b in range(n_splits):
        mask = base_mask & (calib["split"] == b)
        if not mask.any():
            continue
        th = _lec_calibrate_np(calib["unc"][mask], calib["corr"][mask], alpha)
        if th is not None:
            shared[b] = th
    return np.tile(shared, (NUM_DISEASES, 1)), shared


# ════════════════════════════════════════════════════════════════════
#  Derived summaries
# ════════════════════════════════════════════════════════════════════

def _summarise(result, alpha, n_splits, extra=None):
    """Add K_meet, evaluable-specialist count and infeasibility summaries."""
    spec = result["specialist"]

    # Systems that never reject carry an infinite threshold, which is not
    # representable in JSON and is not a calibrated quantity anyway.
    for rec in spec.values():
        for field in ("threshold_mean", "threshold_std"):
            v = rec.get(field)
            if v is not None and not np.isfinite(v):
                rec[field] = None

    estimable = [k for k in range(NUM_DISEASES)
                 if spec[str(k)]["mFDR_k"] is not None]
    meeting = [k for k in estimable if spec[str(k)]["mFDR_k"] <= alpha]

    inf_freqs = [spec[str(k)]["infeasible_freq"] for k in range(NUM_DISEASES)]

    out = {
        "alpha": alpha,
        "beta": BETA,
        "n_splits": n_splits,
        "mFDR_sys": result["mFDR_sys"]["pooled"],
        "mFDR_sys_split_mean": result["mFDR_sys"]["mean"],
        "mFDR_sys_split_sd": result["mFDR_sys"]["std"],
        "n_estimable_splits": result["mFDR_sys"]["n_valid"],
        "Cov_sys": result["Cov_sys"]["pooled"],
        "Cov_sys_split_mean": result["Cov_sys"]["mean"],
        "Cov_sys_split_sd": result["Cov_sys"]["std"],
        "Ref_sys": result["Ref_sys"]["pooled"],
        "mFDR_spec_macro": result["mfdr_spec"]["macro"],
        "mFDR_spec_micro": result["mfdr_spec"]["micro"],
        "mFDR_spec_worst": result["mfdr_spec"]["worst"],
        "K_meet": len(meeting),
        "n_evaluable_specialists": len(estimable),
        "specialist_infeasibility_mean": round(float(np.mean(inf_freqs)), 4),
        "specialist_infeasibility_max": round(float(np.max(inf_freqs)), 4),
        "n_clinics_ever_infeasible": int(sum(f > 0 for f in inf_freqs)),
        "empirical_mFDR_sys_le_alpha": (result["mFDR_sys"]["pooled"] is not None
                                        and result["mFDR_sys"]["pooled"] <= alpha),
        "N_AI_correct": result["N_AI_correct"],
        "N_AI_error": result["N_AI_error"],
        "N_human": result["N_human"],
        "clinics_not_meeting": [DISEASE_NAMES[k] for k in estimable
                                if k not in meeting],
        "specialist": spec,
    }
    if extra:
        out.update(extra)
    return out


# ════════════════════════════════════════════════════════════════════
#  Main sweep
# ════════════════════════════════════════════════════════════════════

def compute_grid(dl, alphas=ALPHA_GRID):
    n_splits = len(dl.splits_phase1["splits"])

    print("Precomputing sample arrays...")
    t0 = time.time()
    pre = precompute_all_with_rc(dl)
    print(f"  done in {time.time() - t0:.1f}s")

    calib, test = pre["calib"], pre["test"]
    lr = _lambda_r_array(dl)

    # Router acceptance on both streams, using the beta-calibrated threshold.
    c_valid = ~np.isnan(calib["unc"])
    c_sr = calib["u1"] <= lr[calib["split"]]
    routed_cal = c_valid & c_sr

    s_r = test["u1"] <= lr[test["split"]]
    test_routed = {k: test[k][s_r] for k in test}
    # Coverage denominators always refer to every held-out test case, so that
    # router-stage referrals are charged to the systems that make them.
    total_n = np.bincount(test["split"], minlength=n_splits).astype(np.float64)

    accept_all = _accept_all_matrix(n_splits)

    grid = {key: [] for key, _, _, _ in SYSTEMS}

    for alpha in alphas:
        print(f"\nalpha = {alpha}")

        # --- 1. Full automation: no rejection at either stage -------------
        t0 = time.time()
        # An infinite threshold has no finite mean, which numpy flags while
        # summarising thresholds; the value is discarded in _summarise.
        with np.errstate(invalid="ignore"):
            r = _evaluate_specialist_system(accept_all, test, n_splits, alpha,
                                            total_n)
        grid["full_automation"].append(
            _summarise(r, alpha, n_splits,
                       {"router_threshold_applied": False,
                        "alpha_dependent": False}))
        print(f"  full automation            {time.time() - t0:5.1f}s"
              f"  mFDR={r['mFDR_sys']['pooled']:.4f}  Cov={r['Cov_sys']['pooled']:.4f}")

        # --- 2. Router-only: router threshold, specialists accept all -----
        t0 = time.time()
        with np.errstate(invalid="ignore"):
            r = _evaluate_specialist_system(accept_all, test_routed, n_splits,
                                            alpha, total_n)
        grid["router_only"].append(
            _summarise(r, alpha, n_splits,
                       {"router_threshold_applied": True,
                        "alpha_dependent": False}))
        print(f"  router-only rejection      {time.time() - t0:5.1f}s"
              f"  mFDR={r['mFDR_sys']['pooled']:.4f}  Cov={r['Cov_sys']['pooled']:.4f}")

        # --- 3. Specialist-only: no router threshold, per-clinic LEC ------
        t0 = time.time()
        th = _calibrate_per_clinic(calib, n_splits, alpha, c_valid)
        r = _evaluate_specialist_system(th, test, n_splits, alpha, total_n)
        grid["specialist_only"].append(
            _summarise(r, alpha, n_splits,
                       {"router_threshold_applied": False,
                        "alpha_dependent": True}))
        print(f"  specialist-only rejection  {time.time() - t0:5.1f}s"
              f"  mFDR={r['mFDR_sys']['pooled']:.4f}  Cov={r['Cov_sys']['pooled']:.4f}")

        # --- 4. Shared threshold: one pooled LEC for all clinics ----------
        t0 = time.time()
        th, shared = _calibrate_shared(calib, n_splits, alpha, routed_cal)
        r = _evaluate_specialist_system(th, test_routed, n_splits, alpha, total_n)
        sv = shared[~np.isnan(shared)]
        grid["shared_threshold"].append(
            _summarise(r, alpha, n_splits,
                       {"router_threshold_applied": True,
                        "alpha_dependent": True,
                        "shared_threshold_mean": (round(float(sv.mean()), 6)
                                                  if sv.size else None),
                        "shared_threshold_sd": (round(float(sv.std()), 6)
                                                if sv.size else None),
                        "shared_infeasible_freq": round(
                            float(np.isnan(shared).sum()) / n_splits, 4)}))
        print(f"  shared threshold           {time.time() - t0:5.1f}s"
              f"  mFDR={r['mFDR_sys']['pooled']:.4f}  Cov={r['Cov_sys']['pooled']:.4f}")

        # --- 5. Oracle-clean: drop wrong-routed cases when calibrating ----
        t0 = time.time()
        th = _calibrate_per_clinic(calib, n_splits, alpha, routed_cal,
                                   per_clinic_extra=(calib["rc"] == 1))
        r = _evaluate_specialist_system(th, test_routed, n_splits, alpha, total_n)
        grid["oracle_clean"].append(
            _summarise(r, alpha, n_splits,
                       {"router_threshold_applied": True,
                        "alpha_dependent": True}))
        print(f"  oracle-clean calibration   {time.time() - t0:5.1f}s"
              f"  mFDR={r['mFDR_sys']['pooled']:.4f}  Cov={r['Cov_sys']['pooled']:.4f}")

        # --- 6. Proposed: per-clinic LEC on the full routed stream --------
        t0 = time.time()
        th = _calibrate_per_clinic(calib, n_splits, alpha, routed_cal)
        r = _evaluate_specialist_system(th, test_routed, n_splits, alpha, total_n)
        grid["proposed"].append(
            _summarise(r, alpha, n_splits,
                       {"router_threshold_applied": True,
                        "alpha_dependent": True}))
        print(f"  proposed system (LEC)      {time.time() - t0:5.1f}s"
              f"  mFDR={r['mFDR_sys']['pooled']:.4f}  Cov={r['Cov_sys']['pooled']:.4f}")

    return grid


# ════════════════════════════════════════════════════════════════════
#  Output
# ════════════════════════════════════════════════════════════════════

CSV_FIELDS = [
    "system", "guarantee_scope", "alpha", "beta", "alpha_dependent",
    "mFDR_sys", "mFDR_sys_split_mean", "mFDR_sys_split_sd",
    "Cov_sys", "Cov_sys_split_mean", "Cov_sys_split_sd", "Ref_sys",
    "mFDR_spec_macro", "mFDR_spec_micro", "mFDR_spec_worst",
    "K_meet", "n_evaluable_specialists",
    "specialist_infeasibility_mean", "specialist_infeasibility_max",
    "n_clinics_ever_infeasible", "empirical_mFDR_sys_le_alpha",
    "N_AI_correct", "N_AI_error", "N_human",
    "n_estimable_splits", "clinics_not_meeting",
]


def write_outputs(grid, beta=BETA, output_json=OUT_JSON, output_csv=OUT_CSV):
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    meta = {
        "beta": beta,
        "alpha_grid": ALPHA_GRID,
        "systems": [{"key": k, "label": lab, "guarantee_scope": g,
                     "alpha_dependent": dep}
                    for k, lab, g, dep in SYSTEMS],
        "note": ("Split-wise mean and SD are reported instead of bootstrap "
                 "confidence intervals: repeated splits reuse patients, so "
                 "pooled decisions are not independent observations."),
        "calibrator": ("Specialist thresholds are selected by the exhaustive LEC "
                       "scan (_lec_calibrate_np), which returns the largest "
                       "feasible threshold. The router threshold is read from "
                       "step3, where the binary search of pipeline.lec is used; "
                       "at beta = 0.1 the two agree on all 500 splits, so the "
                       "router stage is unaffected here. The binary search does "
                       "return a smaller threshold in about 13% of specialist "
                       "calibrations, which is why the proposed system differs "
                       "slightly from metrics.json."),
    }
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "grid": grid}, f, indent=1)
    print(f"\nwrote {output_json} ({os.path.getsize(output_json) / 1024:.0f} KB)")

    scope = {k: (lab, g) for k, lab, g, _ in SYSTEMS}
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for key, _, _, _ in SYSTEMS:
            for row in grid[key]:
                out = dict(row)
                out["system"] = scope[key][0]
                out["guarantee_scope"] = scope[key][1]
                out["clinics_not_meeting"] = "|".join(row["clinics_not_meeting"])
                w.writerow(out)
    print(f"wrote {output_csv}")


def print_table(grid, beta=BETA):
    print("\n" + "=" * 112)
    print("Baseline comparison across the specialist risk grid "
          f"(beta = {beta}, B = 500 splits)")
    print("=" * 112)
    print(f"{'System':<28}{'alpha':>6}{'mFDR_sys':>10}{'Cov_sys':>9}"
          f"{'Ref_sys':>9}{'macro':>8}{'worst':>8}{'K_meet':>9}"
          f"{'max infeas':>12}{'<=a':>6}")
    print("-" * 112)
    for key, label, _, dep in SYSTEMS:
        for row in grid[key]:
            # mFDR and coverage are fixed for the unfiltered systems, but
            # K_meet still moves because the target itself moves.
            k_meet = f"{row['K_meet']}/{row['n_evaluable_specialists']}"
            print(f"{label:<28}{row['alpha']:>6}"
                  f"{row['mFDR_sys']:>10.4f}{row['Cov_sys']:>9.4f}"
                  f"{row['Ref_sys']:>9.4f}{row['mFDR_spec_macro']:>8.4f}"
                  f"{row['mFDR_spec_worst']:>8.4f}{k_meet:>9}"
                  f"{row['specialist_infeasibility_max']:>12.4f}"
                  f"{'yes' if row['empirical_mFDR_sys_le_alpha'] else 'NO':>6}")
        if not dep:
            print(f"{'':<28}  (risk and coverage are alpha-independent; "
                  "only the target moves)")
        print("-" * 112)


def main(beta=BETA, data_root=None, input_root=None,
         output_json=OUT_JSON, output_csv=OUT_CSV):
    print(f"Loading data (beta={beta})...")
    dl = DataLoader(beta=beta, alpha=0.2,
                    data_root=data_root, input_root=input_root)
    grid = compute_grid(dl)
    print_table(grid, beta=beta)
    write_outputs(grid, beta=beta, output_json=output_json, output_csv=output_csv)


if __name__ == "__main__":
    main()
