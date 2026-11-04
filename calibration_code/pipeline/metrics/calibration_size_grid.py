"""Calibration-size sensitivity, per clinic (Methods 2.14).

The existing calibration-size analysis reports system risk and coverage only.
That misses the mechanism. The exchangeability check showed the guarantee fails
where the routed calibration stream is small: clinic 9 carries 84 cases per
split and at alpha = 0.05 its calibration FDP is 0.003 against a held-out 0.145,
feasible in only 66 of 500 splits. Calibration size is the lever, so it has to
be measured per clinic, together with the two feasibility notions that only
separate once streams start running out.

Design. The held-out test set is held fixed and the calibration set is
subsampled, so differences across ratios come from calibration size alone and
the largest ratio reproduces the existing results exactly as an anchor. The
calibration list is grouped by source dataset, so subsampling shuffles within
each split first; taking a prefix would draw only 3 of the 11 datasets.

Per Methods 2.14 the three feasibility notions are kept apart:

    f_R^infeas      router calibration infeasible
    f_{k|R}^infeas  specialist k infeasible among router-feasible splits
    f_k^unavail     clinic k unusable for any reason

They coincide at the full calibration size because the router never fails there.
They are expected to separate as the ratio drops, which is the point.

Usage:
    cd calibration_code
    python3 -m pipeline.metrics.calibration_size_grid
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
from pipeline.metrics.compute_extra import precompute_all_with_rc
from pipeline.metrics.data_loader import DataLoader

RATIOS = [0.10, 0.20, 0.30, 0.40, 0.50]
ALPHAS = [0.05, 0.20]
BETA = 0.10
SEED = 20260701

OUT_JSON = os.path.join(PROJECT_ROOT, "result", "metrics",
                        "calibration_size_grid.json")
OUT_CSV = os.path.join(PROJECT_ROOT, "result", "metrics",
                       "calibration_size_grid.csv")


def block_offsets(split_ids, n_splits):
    """Start index and length of each split's block in a flat array."""
    counts = np.bincount(split_ids, minlength=n_splits)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    return starts, counts


def subsample_mask(calib, test, ratio, n_splits, seed=SEED):
    """Boolean mask selecting a calibration subsample of the given ratio.

    The ratio is relative to the whole pool available to a split, so the
    largest ratio keeps the entire existing calibration set.
    """
    c_start, c_len = block_offsets(calib["split"], n_splits)
    _, t_len = block_offsets(test["split"], n_splits)
    mask = np.zeros(calib["split"].shape, dtype=bool)
    rng = np.random.default_rng(seed)
    sizes = []
    for b in range(n_splits):
        pool = c_len[b] + t_len[b]
        n_take = min(int(round(ratio * pool)), c_len[b])
        # Shuffle within the block: the list is ordered by source dataset, so a
        # prefix would be a biased sample of modalities rather than of patients.
        idx = rng.permutation(c_len[b])[:n_take] + c_start[b]
        mask[idx] = True
        sizes.append(n_take)
    return mask, int(np.mean(sizes))


def _dist(v):
    """Summary of a per-split size distribution, as required by Methods 2.14."""
    v = np.asarray(v, dtype=np.float64)
    if v.size == 0:
        return {}
    return {
        "mean": round(float(v.mean()), 2),
        "sd": round(float(v.std()), 2),
        "median": round(float(np.median(v)), 1),
        "min": int(v.min()),
        "p2.5": round(float(np.percentile(v, 2.5)), 1),
        "p97.5": round(float(np.percentile(v, 97.5)), 1),
    }


def router_stage(calib, test, sub, alpha_unused, n_splits):
    """Recalibrate the router on the subsample and evaluate it on the test set."""
    lr = np.full(n_splits, np.nan)
    for b in range(n_splits):
        m = sub & (calib["split"] == b)
        if not m.any():
            continue
        th = _lec_calibrate_np(calib["u1"][m], (calib["rc"][m] == 1).astype(
            np.int64), BETA)
        if th is not None:
            lr[b] = th

    sr = test["u1"] <= lr[test["split"]]
    acc = np.bincount(test["split"], weights=sr.astype(np.float64),
                      minlength=n_splits)
    acc_err = np.bincount(test["split"],
                          weights=(sr & (test["rc"] == 0)).astype(np.float64),
                          minlength=n_splits)
    total = float(test["split"].size)
    return lr, sr, {
        "lambda_R_mean": (round(float(np.nanmean(lr)), 6)
                          if np.isfinite(lr).any() else None),
        "lambda_R_sd": (round(float(np.nanstd(lr)), 6)
                        if np.isfinite(lr).any() else None),
        "mFDR_R": round(float(acc_err.sum() / acc.sum()), 6) if acc.sum() else None,
        "Cov_R": round(float(acc.sum() / total), 6),
        "f_R_infeas": round(float(np.isnan(lr).sum()) / n_splits, 4),
        "n_router_feasible_splits": int(np.isfinite(lr).sum()),
    }


def clinic_stage(k, calib, test, sub, lr, sr, alpha, n_splits):
    """Calibrate clinic k on the subsampled routed stream and evaluate it."""
    router_ok = np.isfinite(lr)
    with np.errstate(invalid="ignore"):
        cal_routed = sub & (calib["u1"] <= lr[calib["split"]]) \
            & (~np.isnan(calib["unc"])) & (calib["pred_k"] == k)
    test_routed = sr & (test["pred_k"] == k)

    th = np.full(n_splits, np.nan)
    cal_sizes = []
    empty_stream = np.zeros(n_splits, dtype=bool)
    lec_infeasible = np.zeros(n_splits, dtype=bool)
    sel_c = err_c = 0

    for b in range(n_splits):
        if not router_ok[b]:
            continue
        m = cal_routed & (calib["split"] == b)
        n = int(m.sum())
        cal_sizes.append(n)
        if n == 0:
            empty_stream[b] = True
            continue
        u, a = calib["unc"][m], calib["corr"][m]
        t = _lec_calibrate_np(u, a, alpha)
        if t is None:
            lec_infeasible[b] = True
            continue
        th[b] = t
        keep = u <= t
        sel_c += int(keep.sum())
        err_c += int((a[keep] == 0).sum())

    with np.errstate(invalid="ignore"):
        acc = test_routed & (test["unc"] <= th[test["split"]])
    n_routed = int(test_routed.sum())
    n_acc = int(acc.sum())
    n_err = int((acc & (test["corr"] == 0)).sum())

    n_router_ok = int(router_ok.sum())
    unavail = (~router_ok) | empty_stream | lec_infeasible
    f_cal = (err_c / sel_c) if sel_c else None
    f_test = (n_err / n_acc) if n_acc else None

    return {
        "disease_name": DISEASE_NAMES[k],
        "routed_calibration_size": _dist(cal_sizes),
        "threshold_mean": (round(float(np.nanmean(th)), 6)
                           if np.isfinite(th).any() else None),
        "threshold_sd": (round(float(np.nanstd(th)), 6)
                         if np.isfinite(th).any() else None),
        "n_feasible_splits": int(np.isfinite(th).sum()),
        "n_empty_stream_splits": int(empty_stream.sum()),
        "n_lec_infeasible_splits": int(lec_infeasible.sum()),
        "f_cond_infeas": (round(float((empty_stream | lec_infeasible).sum())
                                / n_router_ok, 4) if n_router_ok else None),
        "f_unavail": round(float(unavail.sum()) / n_splits, 4),
        "total_routed": n_routed,
        "total_accepted": n_acc,
        "total_accepted_errors": n_err,
        "Cov_k": round(n_acc / n_routed, 6) if n_routed else None,
        "mFDR_k": round(f_test, 6) if f_test is not None else None,
        "FDP_cal": round(f_cal, 6) if f_cal is not None else None,
        # Ties this analysis back to the exchangeability check: the excess is
        # what the guarantee actually loses at this calibration size.
        "excess_FDP": (round(f_test - f_cal, 6)
                       if None not in (f_cal, f_test) else None),
        "mFDR_k_le_alpha": bool(f_test is not None and f_test <= alpha),
    }


def system_stage(clinics, test, n_splits, alpha):
    n_acc = sum(c["total_accepted"] for c in clinics.values())
    n_err = sum(c["total_accepted_errors"] for c in clinics.values())
    total = int(test["split"].size)
    est = [c["mFDR_k"] for c in clinics.values() if c["mFDR_k"] is not None]
    return {
        "mFDR_sys": round(n_err / n_acc, 6) if n_acc else None,
        "Cov_sys": round(n_acc / total, 6),
        "Ref_sys": round(1 - n_acc / total, 6),
        "mFDR_spec_macro": round(float(np.mean(est)), 6) if est else None,
        "mFDR_spec_worst": round(float(np.max(est)), 6) if est else None,
        "K_meet": sum(c["mFDR_k_le_alpha"] for c in clinics.values()),
        "n_evaluable_specialists": len(est),
        "N_AI_correct": n_acc - n_err,
        "N_AI_error": n_err,
        "N_human": total - n_acc,
    }


def compute_grid(dl, ratios=RATIOS, alphas=ALPHAS):
    n_splits = len(dl.splits_phase1["splits"])
    print("Precomputing sample arrays...")
    t0 = time.time()
    pre = precompute_all_with_rc(dl)
    print(f"  done in {time.time() - t0:.1f}s")
    calib, test = pre["calib"], pre["test"]

    grid = {}
    for ratio in ratios:
        sub, mean_size = subsample_mask(calib, test, ratio, n_splits)
        lr, sr, rstat = router_stage(calib, test, sub, None, n_splits)
        print(f"\nratio = {ratio:.2f}  (mean calibration size {mean_size})"
              f"  lambda_R={rstat['lambda_R_mean']:.4f}"
              f"  mFDR_R={rstat['mFDR_R']:.4f}  Cov_R={rstat['Cov_R']:.4f}"
              f"  f_R_infeas={rstat['f_R_infeas']}")
        for alpha in alphas:
            t0 = time.time()
            clinics = {str(k): clinic_stage(k, calib, test, sub, lr, sr, alpha,
                                           n_splits)
                       for k in range(NUM_DISEASES)}
            sys_stat = system_stage(clinics, test, n_splits, alpha)
            grid[f"{ratio:g}|{alpha:g}"] = {
                "ratio": ratio, "alpha": alpha, "beta": BETA,
                "mean_calibration_size": mean_size,
                "router": rstat, "system": sys_stat, "per_disease": clinics,
            }
            ex = [c["excess_FDP"] for c in clinics.values()
                  if c["excess_FDP"] is not None]
            print(f"    alpha={alpha:<5} mFDR_sys={sys_stat['mFDR_sys']:.4f}"
                  f"  Cov_sys={sys_stat['Cov_sys']:.4f}"
                  f"  K_meet={sys_stat['K_meet']}/{sys_stat['n_evaluable_specialists']}"
                  f"  mean excess={np.mean(ex):+.4f}"
                  f"  max excess={np.max(ex):+.4f}  ({time.time() - t0:.0f}s)")
    return grid


CSV_FIELDS = [
    "ratio", "alpha", "beta", "mean_calibration_size", "k", "disease_name",
    "lambda_R_mean", "mFDR_R", "Cov_R", "f_R_infeas",
    "mFDR_sys", "Cov_sys", "K_meet", "n_evaluable_specialists",
    "cal_size_mean", "cal_size_sd", "cal_size_median", "cal_size_min",
    "cal_size_p2.5", "cal_size_p97.5",
    "threshold_mean", "threshold_sd", "n_feasible_splits",
    "n_empty_stream_splits", "n_lec_infeasible_splits",
    "f_cond_infeas", "f_unavail",
    "total_routed", "total_accepted", "total_accepted_errors",
    "Cov_k", "mFDR_k", "FDP_cal", "excess_FDP", "mFDR_k_le_alpha",
]


def write_outputs(grid):
    meta = {
        "ratios": RATIOS, "alphas": ALPHAS, "beta": BETA, "seed": SEED,
        "design": ("The held-out test set is fixed and the calibration set is "
                   "subsampled, so differences across ratios are attributable "
                   "to calibration size alone. Ratio 0.5 keeps the entire "
                   "existing calibration set and reproduces the alpha grid."),
        "subsampling": ("Sampling shuffles within each split before taking a "
                        "subset. The calibration list is grouped by source "
                        "dataset, so a prefix of 20% would contain only 3 of "
                        "the 11 datasets."),
        "feasibility": ("f_R_infeas, f_cond_infeas and f_unavail follow "
                        "Methods 2.14. f_cond_infeas is conditional on router "
                        "feasibility and counts both an empty routed stream "
                        "and an empty LEC feasible set, which are reported "
                        "separately as n_empty_stream_splits and "
                        "n_lec_infeasible_splits."),
        "excess_FDP": ("Held-out mFDR_k minus the calibration-stream FDP at "
                       "the same threshold. LEC forces the latter below alpha "
                       "by construction, so the excess is what the guarantee "
                       "loses at this calibration size."),
    }
    with open(OUT_JSON, "w") as f:
        json.dump({"meta": meta, "grid": grid}, f, indent=1)
    print(f"\nwrote {OUT_JSON} ({os.path.getsize(OUT_JSON) / 1024:.0f} KB)")

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for key in grid:
            blk = grid[key]
            for k in range(NUM_DISEASES):
                c = blk["per_disease"][str(k)]
                row = {kk: vv for kk, vv in c.items()
                       if kk != "routed_calibration_size"}
                for stat, v in c["routed_calibration_size"].items():
                    row[f"cal_size_{stat}"] = v
                row.update({kk: vv for kk, vv in blk["router"].items()})
                row.update({kk: vv for kk, vv in blk["system"].items()})
                row.update(ratio=blk["ratio"], alpha=blk["alpha"],
                           beta=blk["beta"], k=k,
                           mean_calibration_size=blk["mean_calibration_size"])
                w.writerow(row)
    print(f"wrote {OUT_CSV}")


def print_table(grid):
    for alpha in ALPHAS:
        print("\n" + "=" * 104)
        print(f"Calibration-size sensitivity at alpha = {alpha}, beta = {BETA}")
        print("=" * 104)
        print(f"{'ratio':>7}{'cal n':>8}{'mFDR_R':>9}{'Cov_R':>8}"
              f"{'mFDR_sys':>10}{'Cov_sys':>9}{'worst':>8}{'K_meet':>9}"
              f"{'mean exc':>10}{'max exc':>9}")
        for ratio in RATIOS:
            b = grid[f"{ratio:g}|{alpha:g}"]
            s, r = b["system"], b["router"]
            ex = [c["excess_FDP"] for c in b["per_disease"].values()
                  if c["excess_FDP"] is not None]
            k_meet = f"{s['K_meet']}/{s['n_evaluable_specialists']}"
            print(f"{ratio:>7.2f}{b['mean_calibration_size']:>8}"
                  f"{r['mFDR_R']:>9.4f}{r['Cov_R']:>8.4f}"
                  f"{s['mFDR_sys']:>10.4f}{s['Cov_sys']:>9.4f}"
                  f"{s['mFDR_spec_worst']:>8.4f}{k_meet:>9}"
                  f"{np.mean(ex):>+10.4f}{np.max(ex):>+9.4f}")

        print(f"\n  Smallest clinic (D9) at alpha = {alpha}")
        print(f"  {'ratio':>7}{'cal mean':>10}{'cal min':>9}{'feasible':>10}"
              f"{'empty':>7}{'infeas':>8}{'f_cond':>9}{'f_unavail':>11}"
              f"{'mFDR_k':>9}{'excess':>9}")
        for ratio in RATIOS:
            c = grid[f"{ratio:g}|{alpha:g}"]["per_disease"]["9"]
            d = c["routed_calibration_size"]
            mk = c["mFDR_k"] if c["mFDR_k"] is not None else float("nan")
            ex = c["excess_FDP"] if c["excess_FDP"] is not None else float("nan")
            print(f"  {ratio:>7.2f}{d['mean']:>10.1f}{d['min']:>9}"
                  f"{c['n_feasible_splits']:>10}{c['n_empty_stream_splits']:>7}"
                  f"{c['n_lec_infeasible_splits']:>8}{c['f_cond_infeas']:>9.4f}"
                  f"{c['f_unavail']:>11.4f}{mk:>9.4f}{ex:>+9.4f}")


def main():
    print(f"Loading data (beta={BETA})...")
    dl = DataLoader(beta=BETA, alpha=0.2)
    grid = compute_grid(dl)
    print_table(grid)
    write_outputs(grid)


if __name__ == "__main__":
    main()
