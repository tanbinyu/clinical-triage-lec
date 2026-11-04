"""Empirical check of routed-stream exchangeability (Methods 2.8, Eq. 28).

The specialist risk guarantee rests on Eq. (28): after the router threshold is
fixed, the routed specialist calibration stream and the routed deployment stream
must be exchangeable in the uncertainty scores and correctness indicators that
the calibration rule consumes. The paper states this condition but never tests
it against data. This module tests it.

The check is layered from weakest to most directly consequential:

  1. Marginal of A_k. If the calibration stream is easier than the deployment
     stream, the calibrated threshold is too permissive.
  2. Marginal of u_k, via a two-sample Kolmogorov-Smirnov statistic and
     quantile shifts. Detects drift in the score distribution itself.
  3. FDP_k at the calibrated threshold, on the calibration stream versus the
     held-out stream. This is the quantity that decides whether the guarantee
     transfers: LEC forces FDP_cal <= alpha by construction, so the excess
     FDP_test - FDP_cal is the calibration optimism as the guarantee feels it.

Layer 3 is the result; layers 1 and 2 explain it.

Everything is computed per split and then pooled by aggregating numerators and
denominators, per Methods 2.15. Splits in which a clinic admits no feasible
threshold, or accepts no case, are excluded and counted.

Usage:
    cd calibration_code
    python3 -m pipeline.metrics.exchangeability_check
"""
import argparse
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
from pipeline.metrics.compute_extra import precompute_all_with_rc
from pipeline.metrics.data_loader import DataLoader

ALPHA_GRID = [0.05, 0.10, 0.15, 0.20]
ALPHA_MAIN = 0.20
DEFAULT_BETA = 0.10

OUT_JSON = os.path.join(PROJECT_ROOT, "result", "metrics",
                        "exchangeability_check.json")
OUT_CSV = os.path.join(PROJECT_ROOT, "result", "metrics",
                       "exchangeability_check.csv")


def ks_2samp(a, b):
    """Two-sample Kolmogorov-Smirnov statistic.

    Hand-rolled because scipy is not installed. Walks the merged support once
    and takes the largest gap between the two empirical CDFs.
    """
    if a.size == 0 or b.size == 0:
        return None
    a = np.sort(a)
    b = np.sort(b)
    allv = np.concatenate([a, b])
    # searchsorted with "right" gives the ECDF value at each observed point.
    cdf_a = np.searchsorted(a, allv, side="right") / a.size
    cdf_b = np.searchsorted(b, allv, side="right") / b.size
    return float(np.max(np.abs(cdf_a - cdf_b)))


def ks_null_reference(n, m):
    """Asymptotic null mean and 95th percentile of the two-sample KS statistic.

    The statistic is not scale-free in sample size: under exchangeability it
    still averages about 0.87 * sqrt((n+m)/nm). Without this reference a clinic
    with a small routed stream looks drifted purely because it is small, which
    matters here because the smallest clinic has 84 cases per split against
    1300 for the largest.
    """
    if n == 0 or m == 0:
        return None, None
    scale = np.sqrt((n + m) / (n * m))
    return 0.8687 * scale, 1.3581 * scale


def _pooled(num, den):
    return float(num / den) if den > 0 else None


def check_clinic(k, calib, test, routed_cal, routed_test, n_splits, alphas):
    """Compare the routed calibration and held-out streams for one clinic."""
    cal_k = routed_cal & (calib["pred_k"] == k)
    test_k = routed_test & (test["pred_k"] == k)

    # ---- layer 1 and 2: stream-level, independent of alpha ----------------
    n_cal_tot = n_test_tot = 0
    err_cal_tot = err_test_tot = 0
    ks_per_split, dq = [], {25: [], 50: [], 75: []}

    for b in range(n_splits):
        cm = cal_k & (calib["split"] == b)
        tm = test_k & (test["split"] == b)
        uc, ac = calib["unc"][cm], calib["corr"][cm]
        ut, at = test["unc"][tm], test["corr"][tm]
        if uc.size == 0 or ut.size == 0:
            continue
        n_cal_tot += uc.size
        n_test_tot += ut.size
        err_cal_tot += int((ac == 0).sum())
        err_test_tot += int((at == 0).sum())
        ks = ks_2samp(uc, ut)
        if ks is not None:
            ks_per_split.append(ks)
        for q in dq:
            dq[q].append(float(np.percentile(ut, q) - np.percentile(uc, q)))

    rec = {
        "disease_name": DISEASE_NAMES[k],
        "n_calibration_total": n_cal_tot,
        "n_test_total": n_test_tot,
        "mean_calibration_size": round(n_cal_tot / n_splits, 1),
        "mean_test_size": round(n_test_tot / n_splits, 1),
        "base_err_cal": round(_pooled(err_cal_tot, n_cal_tot), 6),
        "base_err_test": round(_pooled(err_test_tot, n_test_tot), 6),
        "ks_mean": round(float(np.mean(ks_per_split)), 6),
        "ks_sd": round(float(np.std(ks_per_split)), 6),
        "ks_max": round(float(np.max(ks_per_split)), 6),
        "n_splits_compared": len(ks_per_split),
    }
    rec["delta_base_err"] = round(rec["base_err_test"] - rec["base_err_cal"], 6)
    for q in dq:
        rec[f"delta_u_p{q}"] = round(float(np.mean(dq[q])), 6)

    # Per-split sizes, so the null reference matches the scale at which the KS
    # statistic was actually computed.
    n_per = n_cal_tot / max(len(ks_per_split), 1)
    m_per = n_test_tot / max(len(ks_per_split), 1)
    null_mean, null_p95 = ks_null_reference(n_per, m_per)
    rec["ks_null_mean"] = round(null_mean, 6)
    rec["ks_null_p95"] = round(null_p95, 6)
    rec["ks_ratio_to_null"] = round(rec["ks_mean"] / null_mean, 4)

    # ---- layer 3: FDP at the calibrated threshold, per alpha --------------
    rec["by_alpha"] = {}
    for alpha in alphas:
        sel_c = err_c = sel_t = err_t = 0
        n_feasible = n_evaluable = 0
        excess = []
        for b in range(n_splits):
            cm = cal_k & (calib["split"] == b)
            uc, ac = calib["unc"][cm], calib["corr"][cm]
            if uc.size == 0:
                continue
            th = _lec_calibrate_np(uc, ac, alpha)
            if th is None:
                continue
            n_feasible += 1
            tm = test_k & (test["split"] == b)
            ut, at = test["unc"][tm], test["corr"][tm]

            kc = uc <= th
            kt = ut <= th
            sel_c += int(kc.sum())
            err_c += int((ac[kc] == 0).sum())
            sel_t += int(kt.sum())
            err_t += int((at[kt] == 0).sum())
            if kc.sum() > 0 and kt.sum() > 0:
                n_evaluable += 1
                excess.append((at[kt] == 0).mean() - (ac[kc] == 0).mean())

        f_cal = _pooled(err_c, sel_c)
        f_test = _pooled(err_t, sel_t)
        rec["by_alpha"][f"{alpha:g}"] = {
            "alpha": alpha,
            "FDP_cal": round(f_cal, 6) if f_cal is not None else None,
            "FDP_test": round(f_test, 6) if f_test is not None else None,
            "delta_FDP": (round(f_test - f_cal, 6)
                          if None not in (f_cal, f_test) else None),
            "excess_split_mean": (round(float(np.mean(excess)), 6)
                                  if excess else None),
            "excess_split_sd": (round(float(np.std(excess)), 6)
                                if excess else None),
            "n_feasible_splits": n_feasible,
            "n_evaluable_splits": n_evaluable,
            "FDP_cal_le_alpha": bool(f_cal is not None and f_cal <= alpha),
            "FDP_test_le_alpha": bool(f_test is not None and f_test <= alpha),
        }
    return rec


def run(dl, beta, alphas=ALPHA_GRID):
    n_splits = len(dl.splits_phase1["splits"])
    print("Precomputing sample arrays...")
    t0 = time.time()
    pre = precompute_all_with_rc(dl)
    print(f"  done in {time.time() - t0:.1f}s")
    calib, test = pre["calib"], pre["test"]

    lr = np.array([t if t is not None else np.nan
                   for t in dl.lec_phase1_t_hats], dtype=np.float64)
    routed_cal = (~np.isnan(calib["unc"])) & (calib["u1"] <= lr[calib["split"]])
    routed_test = test["u1"] <= lr[test["split"]]
    print(f"  router threshold from step3 at beta={beta}: "
          f"mean={np.nanmean(lr):.4f}")

    out = {}
    for k in range(NUM_DISEASES):
        t0 = time.time()
        out[str(k)] = check_clinic(k, calib, test, routed_cal, routed_test,
                                   n_splits, alphas)
        r = out[str(k)]
        m = r["by_alpha"][f"{ALPHA_MAIN:g}"]
        print(f"  D{k} {r['disease_name']:<26} "
              f"dBase={r['delta_base_err']:+.4f}  KS={r['ks_mean']:.4f}  "
              f"FDP_cal={m['FDP_cal']:.4f} FDP_test={m['FDP_test']:.4f}  "
              f"dFDP={m['delta_FDP']:+.4f}  ({time.time() - t0:.0f}s)")
    return out


CSV_FIELDS = [
    "k", "disease_name", "mean_calibration_size", "mean_test_size",
    "base_err_cal", "base_err_test", "delta_base_err",
    "ks_mean", "ks_sd", "ks_max", "ks_null_mean", "ks_null_p95",
    "ks_ratio_to_null", "delta_u_p25", "delta_u_p50", "delta_u_p75",
    "alpha", "FDP_cal", "FDP_test", "delta_FDP",
    "excess_split_mean", "excess_split_sd",
    "n_feasible_splits", "n_evaluable_splits",
    "FDP_cal_le_alpha", "FDP_test_le_alpha",
]


def write_outputs(res, alphas, beta):
    meta = {
        "alpha_grid": alphas,
        "alpha_main": ALPHA_MAIN,
        "beta": beta,
        "condition": ("Methods Eq. (28): after the router threshold is fixed, "
                      "the routed specialist calibration stream and the routed "
                      "deployment stream must share the conditional law of "
                      "(u_k, A_k)."),
        "layers": {
            "1": "Marginal of A_k, as the routed base error rate.",
            "2": "Marginal of u_k, as a two-sample KS statistic and quantile "
                 "shifts, computed per split and averaged. Reported against "
                 "the asymptotic null mean 0.8687 * sqrt((n+m)/nm), because "
                 "the statistic grows as the routed stream shrinks and the "
                 "clinics differ by a factor of 15 in stream size.",
            "3": "FDP_k at the calibrated threshold on each stream. LEC forces "
                 "FDP_cal <= alpha by construction, so delta_FDP = FDP_test - "
                 "FDP_cal is the calibration optimism that the guarantee sees.",
        },
        "estimator": ("Counts are aggregated across splits before taking "
                      "ratios. Splits where a clinic has no feasible threshold "
                      "or accepts nothing are excluded and counted."),
        "caveat": ("The routed calibration stream is drawn from the same "
                   "calibration partition used to fit the router threshold, so "
                   "the sufficient condition stated in Methods 2.8 (C_S "
                   "independent of C_R) does not hold by construction. This "
                   "check therefore tests Eq. (28) directly rather than that "
                   "sufficient condition."),
    }
    with open(OUT_JSON, "w") as f:
        json.dump({"meta": meta, "clinics": res}, f, indent=1)
    print(f"\nwrote {OUT_JSON} ({os.path.getsize(OUT_JSON) / 1024:.0f} KB)")

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for k in range(NUM_DISEASES):
            rec = res[str(k)]
            for a in alphas:
                row = {kk: vv for kk, vv in rec.items() if kk != "by_alpha"}
                row.update(rec["by_alpha"][f"{a:g}"])
                row["k"] = k
                w.writerow(row)
    print(f"wrote {OUT_CSV}")


def print_table(res, alphas):
    print("\n" + "=" * 96)
    print("Routed-stream exchangeability check (Methods Eq. 28)")
    print("=" * 96)
    print("Layers 1-2: stream comparison, independent of alpha")
    print(f"{'clinic':<28}{'n_cal':>7}{'base_cal':>10}{'base_test':>11}"
          f"{'delta':>9}{'KS_obs':>8}{'KS_null':>9}{'obs/null':>10}")
    print("-" * 96)
    for k in range(NUM_DISEASES):
        r = res[str(k)]
        print(f"D{k} {r['disease_name']:<24}{r['mean_calibration_size']:>7.0f}"
              f"{r['base_err_cal']:>10.4f}"
              f"{r['base_err_test']:>11.4f}{r['delta_base_err']:>+9.4f}"
              f"{r['ks_mean']:>8.4f}{r['ks_null_mean']:>9.4f}"
              f"{r['ks_ratio_to_null']:>10.2f}")

    print("\nLayer 3: does the guarantee transfer to held-out data?")
    for a in alphas:
        key = f"{a:g}"
        print(f"\n  alpha = {a}")
        print(f"  {'clinic':<28}{'FDP_cal':>10}{'FDP_test':>10}"
              f"{'delta':>9}{'test<=a':>9}{'feas':>7}")
        n_ok = 0
        for k in range(NUM_DISEASES):
            m = res[str(k)]["by_alpha"][key]
            n_ok += m["FDP_test_le_alpha"]
            print(f"  D{k} {res[str(k)]['disease_name']:<24}"
                  f"{m['FDP_cal']:>10.4f}{m['FDP_test']:>10.4f}"
                  f"{m['delta_FDP']:>+9.4f}"
                  f"{('yes' if m['FDP_test_le_alpha'] else 'NO'):>9}"
                  f"{m['n_feasible_splits']:>7}")
        deltas = [res[str(k)]["by_alpha"][key]["delta_FDP"]
                  for k in range(NUM_DISEASES)]
        print(f"  {'':<28}mean delta = {np.mean(deltas):+.5f}, "
              f"max = {np.max(deltas):+.5f}, "
              f"held-out target met by {n_ok}/{NUM_DISEASES}")


def main():
    parser = argparse.ArgumentParser(
        description="Empirically check routed-stream exchangeability.")
    parser.add_argument("--beta", type=float, default=DEFAULT_BETA,
                        help=f"Router risk level (default: {DEFAULT_BETA:g}).")
    args = parser.parse_args()
    if not 0 < args.beta < 1:
        parser.error("--beta must lie strictly between 0 and 1.")

    print(f"Loading data (alpha={ALPHA_MAIN}, beta={args.beta})...")
    dl = DataLoader(beta=args.beta, alpha=ALPHA_MAIN)
    res = run(dl, beta=args.beta)
    print_table(res, ALPHA_GRID)
    write_outputs(res, ALPHA_GRID, beta=args.beta)


if __name__ == "__main__":
    main()
