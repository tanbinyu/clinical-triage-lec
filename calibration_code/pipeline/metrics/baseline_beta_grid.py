"""Baseline comparison across a grid of router risk levels (Methods 2.12).

The alpha sweep showed that the proposed system and specialist-only rejection
are indistinguishable at every alpha (they differ by at most 2.5e-5), because
at beta = 0.1 the router refers only 0.014% of cases and the routed calibration
stream is therefore almost the unfiltered stream. That leaves the contribution
of the first stage untested. This module holds alpha fixed and sweeps beta, so
the gap between the proposed system and specialist-only rejection at each beta
measures what the routing stage actually adds.

Specialist-only rejection and full automation apply no router threshold, so
they are fixed points of this sweep and serve as the no-routing-stage
reference.

The router threshold is recomputed here with the same exhaustive LEC scan used
for the specialist thresholds rather than read from step3. The binary search in
pipeline.lec returns a smaller router threshold on 62/500 splits at beta = 0.01
and 18/500 at beta = 0.025; a stricter router would reject more and so would
overstate the very contribution this sweep is meant to measure.

Usage:
    cd calibration_code
    python3 -m pipeline.metrics.baseline_beta_grid
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

from pipeline.metrics.config import NUM_DISEASES, DISEASE_NAMES
from pipeline.metrics.compute import _lec_calibrate_np
from pipeline.metrics.compute_extra import (
    precompute_all_with_rc,
    _evaluate_specialist_system,
)
from pipeline.metrics.data_loader import DataLoader
from pipeline.metrics.baseline_alpha_grid import (
    SYSTEMS, _accept_all_matrix, _calibrate_per_clinic, _calibrate_shared,
    _summarise,
)

BETA_GRID = [0.01, 0.025, 0.05, 0.10, 0.15, 0.20]
ALPHA_FIXED = 0.20

OUT_JSON = os.path.join(PROJECT_ROOT, "result", "metrics", "baseline_beta_grid.json")
OUT_CSV = os.path.join(PROJECT_ROOT, "result", "metrics", "baseline_beta_grid.csv")

# Which systems actually consult the router threshold.
BETA_DEPENDENT = {"router_only", "shared_threshold", "oracle_clean", "proposed"}


def router_thresholds(dl, beta):
    """Recalibrate the router on each split's calibration set at level beta."""
    splits = dl.splits_phase1["splits"]
    out = np.full(len(splits), np.nan)
    for b, sp in enumerate(splits):
        ids = sp["calib"]
        u = np.fromiter((dl.phase1_lookup[s][0] for s in ids), np.float64, len(ids))
        a = np.fromiter((dl.phase1[s]["correctness"] for s in ids), np.int64, len(ids))
        th = _lec_calibrate_np(u, a, beta)
        if th is not None:
            out[b] = th
    return out


def router_metrics(test, lr, n_splits, beta):
    """Router-stage risk, coverage and referral on the held-out test stream.

    The router error indicator is 1 - routing_correctness, so this measures
    accepted clinic assignments that send a patient to the wrong clinic.
    """
    sr = test["u1"] <= lr[test["split"]]
    err = (test["rc"] == 0)
    acc = np.bincount(test["split"], weights=sr.astype(np.float64),
                      minlength=n_splits)
    acc_err = np.bincount(test["split"], weights=(sr & err).astype(np.float64),
                          minlength=n_splits)
    total = np.bincount(test["split"], minlength=n_splits).astype(np.float64)

    per_split = np.divide(acc_err, acc, out=np.full(n_splits, np.nan),
                          where=acc > 0)
    with np.errstate(invalid="ignore"):
        return {
            "mFDR_R": round(float(acc_err.sum() / acc.sum()), 6) if acc.sum() else None,
            "mFDR_R_split_mean": round(float(np.nanmean(per_split)), 6),
            "mFDR_R_split_sd": round(float(np.nanstd(per_split)), 6),
            "Cov_R": round(float(acc.sum() / total.sum()), 6),
            "Ref_R": round(float(1 - acc.sum() / total.sum()), 6),
            "router_infeasible_freq": round(float(np.isnan(lr).sum()) / n_splits, 4),
            "lambda_R_mean": (round(float(np.nanmean(lr)), 6)
                              if np.isfinite(lr).any() else None),
            "lambda_R_sd": (round(float(np.nanstd(lr)), 6)
                            if np.isfinite(lr).any() else None),
            "beta": beta,
        }


def stream_diagnostics(calib, test, routed_cal, lr, n_splits):
    """Per-clinic routed calibration size and routed contamination."""
    sr = test["u1"] <= lr[test["split"]]
    t_pred, t_rc = test["pred_k"][sr], test["rc"][sr]
    c_pred = calib["pred_k"][routed_cal]

    per_clinic, sizes, rhos = {}, [], []
    for k in range(NUM_DISEASES):
        n_cal = int((c_pred == k).sum())
        mk = t_pred == k
        n_routed = int(mk.sum())
        n_wrong = int((t_rc[mk] == 0).sum()) if n_routed else 0
        rho = round(n_wrong / n_routed, 6) if n_routed else None
        per_clinic[str(k)] = {
            "disease_name": DISEASE_NAMES[k],
            "n_routed_calibration_total": n_cal,
            "mean_routed_calibration_size": round(n_cal / n_splits, 2),
            "n_routed_test_total": n_routed,
            "rho_route": rho,
        }
        sizes.append(n_cal / n_splits)
        if rho is not None:
            rhos.append(rho)
    return per_clinic, {
        "mean_specialist_routed_calibration_size": round(float(np.mean(sizes)), 2),
        "min_specialist_routed_calibration_size": round(float(np.min(sizes)), 2),
        "mean_rho_route": round(float(np.mean(rhos)), 6) if rhos else None,
    }


def compute_grid(dl, betas=BETA_GRID, alpha=ALPHA_FIXED):
    n_splits = len(dl.splits_phase1["splits"])

    print("Precomputing sample arrays...")
    t0 = time.time()
    pre = precompute_all_with_rc(dl)
    print(f"  done in {time.time() - t0:.1f}s")
    calib, test = pre["calib"], pre["test"]
    c_valid = ~np.isnan(calib["unc"])
    total_n = np.bincount(test["split"], minlength=n_splits).astype(np.float64)
    accept_all = _accept_all_matrix(n_splits)

    grid = {key: [] for key, _, _, _ in SYSTEMS}

    for beta in betas:
        print(f"\nbeta = {beta}")
        t0 = time.time()
        lr = router_thresholds(dl, beta)
        rm = router_metrics(test, lr, n_splits, beta)
        print(f"  router recalibrated in {time.time() - t0:.1f}s"
              f"  lambda_R={rm['lambda_R_mean']:.4f}"
              f"  mFDR_R={rm['mFDR_R']:.4f}  Cov_R={rm['Cov_R']:.4f}")

        c_sr = calib["u1"] <= lr[calib["split"]]
        routed_cal = c_valid & c_sr
        s_r = test["u1"] <= lr[test["split"]]
        test_routed = {k: test[k][s_r] for k in test}
        per_clinic, agg = stream_diagnostics(calib, test, routed_cal, lr, n_splits)

        def record(key, result, extra=None):
            base = {**rm, **agg,
                    "beta_dependent": key in BETA_DEPENDENT,
                    "router_stream": per_clinic}
            if extra:
                base.update(extra)
            grid[key].append(_summarise(result, alpha, n_splits, base))

        # Systems that ignore the router threshold are fixed points of this
        # sweep; they are still recomputed so every row shares one code path.
        with np.errstate(invalid="ignore"):
            record("full_automation",
                   _evaluate_specialist_system(accept_all, test, n_splits,
                                               alpha, total_n))
            record("router_only",
                   _evaluate_specialist_system(accept_all, test_routed, n_splits,
                                               alpha, total_n))

        th = _calibrate_per_clinic(calib, n_splits, alpha, c_valid)
        record("specialist_only",
               _evaluate_specialist_system(th, test, n_splits, alpha, total_n))

        th, shared = _calibrate_shared(calib, n_splits, alpha, routed_cal)
        sv = shared[~np.isnan(shared)]
        record("shared_threshold",
               _evaluate_specialist_system(th, test_routed, n_splits, alpha, total_n),
               {"shared_threshold_mean": (round(float(sv.mean()), 6)
                                          if sv.size else None),
                "shared_infeasible_freq": round(
                    float(np.isnan(shared).sum()) / n_splits, 4)})

        th = _calibrate_per_clinic(calib, n_splits, alpha, routed_cal,
                                   per_clinic_extra=(calib["rc"] == 1))
        record("oracle_clean",
               _evaluate_specialist_system(th, test_routed, n_splits, alpha, total_n))

        th = _calibrate_per_clinic(calib, n_splits, alpha, routed_cal)
        record("proposed",
               _evaluate_specialist_system(th, test_routed, n_splits, alpha, total_n))

        for key, label, _, _ in SYSTEMS:
            r = grid[key][-1]
            print(f"  {label:<28} mFDR={r['mFDR_sys']:.4f}"
                  f"  Cov={r['Cov_sys']:.4f}  K_meet={r['K_meet']}/10")

    return grid


CSV_FIELDS = [
    "system", "guarantee_scope", "beta", "alpha", "beta_dependent",
    "lambda_R_mean", "lambda_R_sd", "router_infeasible_freq",
    "mFDR_R", "Cov_R", "Ref_R",
    "mFDR_sys", "mFDR_sys_split_sd", "Cov_sys", "Ref_sys",
    "mFDR_spec_macro", "mFDR_spec_worst", "K_meet", "n_evaluable_specialists",
    "specialist_infeasibility_max", "empirical_mFDR_sys_le_alpha",
    "mean_specialist_routed_calibration_size",
    "min_specialist_routed_calibration_size", "mean_rho_route",
    "N_AI_correct", "N_AI_error", "N_human",
]


def write_outputs(grid):
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    meta = {
        "alpha_fixed": ALPHA_FIXED,
        "beta_grid": BETA_GRID,
        "systems": [{"key": k, "label": lab, "guarantee_scope": g,
                     "beta_dependent": k in BETA_DEPENDENT}
                    for k, lab, g, _ in SYSTEMS],
        "calibrator": ("Both the router and the specialist thresholds are "
                       "selected by the exhaustive LEC scan, so the sweep "
                       "isolates the effect of beta. Reading the router "
                       "threshold from step3 instead would apply a stricter "
                       "threshold on 62/500 splits at beta = 0.01 and 18/500 "
                       "at beta = 0.025, overstating the routing stage."),
        "reference_system": ("Specialist-only rejection applies no router "
                             "threshold and is therefore the no-routing-stage "
                             "reference; its gap to the proposed system at each "
                             "beta is the contribution of the routing stage."),
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "grid": grid}, f, indent=1)
    print(f"\nwrote {OUT_JSON} ({os.path.getsize(OUT_JSON) / 1024:.0f} KB)")

    scope = {k: (lab, g) for k, lab, g, _ in SYSTEMS}
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for key, _, _, _ in SYSTEMS:
            for row in grid[key]:
                out = dict(row)
                out["system"], out["guarantee_scope"] = scope[key]
                w.writerow(out)
    print(f"wrote {OUT_CSV}")


def print_table(grid):
    print("\n" + "=" * 104)
    print(f"Baseline comparison across the router risk grid (alpha = {ALPHA_FIXED})")
    print("=" * 104)
    print(f"{'System':<28}{'beta':>6}{'mFDR_R':>9}{'Cov_R':>8}"
          f"{'mFDR_sys':>10}{'Cov_sys':>9}{'worst':>8}{'K_meet':>9}"
          f"{'cal size':>10}")
    print("-" * 104)
    for key, label, _, _ in SYSTEMS:
        for r in grid[key]:
            k_meet = f"{r['K_meet']}/10"
            print(f"{label:<28}{r['beta']:>6}{r['mFDR_R']:>9.4f}{r['Cov_R']:>8.4f}"
                  f"{r['mFDR_sys']:>10.4f}{r['Cov_sys']:>9.4f}"
                  f"{r['mFDR_spec_worst']:>8.4f}{k_meet:>9}"
                  f"{r['mean_specialist_routed_calibration_size']:>10.1f}")
        if key not in BETA_DEPENDENT:
            print(f"{'':<28}  (no router threshold; fixed point of this sweep)")
        print("-" * 104)

    print("\nRouting-stage contribution (proposed minus specialist-only):")
    print(f"  {'beta':>6}{'d mFDR_sys':>13}{'d Cov_sys':>12}{'d worst':>10}"
          f"{'d K_meet':>10}")
    for p, s in zip(grid["proposed"], grid["specialist_only"]):
        print(f"  {p['beta']:>6}{p['mFDR_sys'] - s['mFDR_sys']:>13.6f}"
              f"{p['Cov_sys'] - s['Cov_sys']:>12.6f}"
              f"{p['mFDR_spec_worst'] - s['mFDR_spec_worst']:>10.6f}"
              f"{p['K_meet'] - s['K_meet']:>10}")


def main():
    print(f"Loading data (alpha fixed at {ALPHA_FIXED})...")
    dl = DataLoader(beta=0.1, alpha=ALPHA_FIXED)
    grid = compute_grid(dl)
    print_table(grid)
    write_outputs(grid)


if __name__ == "__main__":
    main()
