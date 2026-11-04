"""Routing contamination and error decomposition across the risk grid (2.11).

The paper defines six contamination and error-source quantities, but only the
routed contamination rate is swept, and only over beta. The remaining five exist
at a single operating point. That is the wrong axis: the routed contamination
rate is fixed once the router threshold is fixed and does not depend on alpha at
all, whereas every quantity measured after specialist rejection does.

The claim at stake is the one in Methods 2.11, that specialist rejection
prevents routing errors from progressing to AI-managed outcomes. How strongly it
holds is governed by alpha, so a single alpha cannot support it.

Definitions follow Methods 2.11 on the held-out routed stream of clinic k:

    rho_route   wrong-routed fraction of the routed stream, before rejection
    rho_acc     wrong-routed fraction among cases the specialist accepts
    Ref_wrong   fraction of wrong-routed cases the specialist rejects, Eq. (54)
    N_route_err accepted cases that were wrongly routed
    N_diag_err  accepted cases routed correctly but misdiagnosed
    q_route_err N_route_err / N_err, Eq. (57)
    q_diag_err  N_diag_err / N_err, Eq. (58)

Note that this module writes Ref_wrong, following Eq. (54). The field is named
Rej_wrong in metrics.json and in Table 4, which predates the current Methods.

Thresholds come from the same routine used for the alpha grid, so the numbers
here line up with the proposed system in baseline_alpha_grid.json.

Usage:
    cd calibration_code
    python3 -m pipeline.metrics.contamination_alpha_grid
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
from pipeline.metrics.compute_extra import precompute_all_with_rc
from pipeline.metrics.data_loader import DataLoader
from pipeline.metrics.baseline_alpha_grid import _calibrate_per_clinic

ALPHA_GRID = [0.05, 0.10, 0.15, 0.20]
BETA_MAIN = 0.10

OUT_JSON = os.path.join(PROJECT_ROOT, "result", "metrics",
                        "contamination_alpha_grid.json")
OUT_CSV = os.path.join(PROJECT_ROOT, "result", "metrics",
                       "contamination_alpha_grid.csv")


def _ratio(num, den):
    return round(float(num / den), 6) if den > 0 else None


def _split_summary(num, den):
    """Mean and SD of a per-split ratio, over splits where it is estimable."""
    with np.errstate(invalid="ignore", divide="ignore"):
        v = np.divide(num, den, out=np.full(den.shape, np.nan), where=den > 0)
    ok = ~np.isnan(v)
    if not ok.any():
        return None, None, 0
    return (round(float(v[ok].mean()), 6), round(float(v[ok].std()), 6),
            int(ok.sum()))


def clinic_contamination(k, test, routed_test, th_k, n_splits):
    """Contamination and error decomposition for one clinic at one alpha.

    th_k is the per-split calibrated threshold; NaN marks an infeasible split,
    where the specialist accepts nothing.
    """
    m = routed_test & (test["pred_k"] == k)
    split = test["split"][m]
    unc = test["unc"][m]
    corr = test["corr"][m]
    rc = test["rc"][m]

    thr = th_k[split]
    # NaN threshold accepts nothing, which is the operational meaning of an
    # infeasible calibration; the comparison already yields False for NaN.
    with np.errstate(invalid="ignore"):
        acc = unc <= thr

    wrong = rc == 0
    bc = lambda mask: np.bincount(split[mask], minlength=n_splits).astype(
        np.float64)
    n_routed = bc(np.ones(split.shape, bool))
    n_wrong = bc(wrong)
    n_acc = bc(acc)
    n_acc_wrong = bc(acc & wrong)
    n_wrong_rejected = bc(wrong & ~acc)
    n_acc_diag_err = bc(acc & ~wrong & (corr == 0))

    N_route_err = int(n_acc_wrong.sum())
    N_diag_err = int(n_acc_diag_err.sum())
    N_err = N_route_err + N_diag_err

    rec = {
        "disease_name": DISEASE_NAMES[k],
        "total_routed": int(n_routed.sum()),
        "total_wrong_routed": int(n_wrong.sum()),
        "total_accepted": int(n_acc.sum()),
        "N_route_err": N_route_err,
        "N_diag_err": N_diag_err,
        "N_err": N_err,
        "rho_route": _ratio(n_wrong.sum(), n_routed.sum()),
        "rho_acc": _ratio(n_acc_wrong.sum(), n_acc.sum()),
        "Ref_wrong": _ratio(n_wrong_rejected.sum(), n_wrong.sum()),
        "q_route_err": _ratio(N_route_err, N_err),
        "q_diag_err": _ratio(N_diag_err, N_err),
        "n_feasible_splits": int(np.isfinite(th_k).sum()),
    }
    for name, num, den in (("rho_route", n_wrong, n_routed),
                           ("rho_acc", n_acc_wrong, n_acc),
                           ("Ref_wrong", n_wrong_rejected, n_wrong)):
        mean, sd, n_ok = _split_summary(num, den)
        rec[f"{name}_split_mean"] = mean
        rec[f"{name}_split_sd"] = sd
        rec[f"{name}_n_estimable_splits"] = n_ok

    # How much of the routed contamination the specialist stage removes. The
    # ratio is more comparable across clinics than the difference, because the
    # routed rates themselves span an order of magnitude.
    if rec["rho_route"] and rec["rho_acc"] is not None:
        rec["rho_acc_over_route"] = round(rec["rho_acc"] / rec["rho_route"], 6)
        rec["rho_reduction"] = round(rec["rho_route"] - rec["rho_acc"], 6)
    else:
        rec["rho_acc_over_route"] = None
        rec["rho_reduction"] = None
    return rec


def compute_grid(dl, alphas=ALPHA_GRID):
    n_splits = len(dl.splits_phase1["splits"])
    print("Precomputing sample arrays...")
    t0 = time.time()
    pre = precompute_all_with_rc(dl)
    print(f"  done in {time.time() - t0:.1f}s")
    calib, test = pre["calib"], pre["test"]

    lr = np.array([t if t is not None else np.nan
                   for t in dl.lec_phase1_t_hats], dtype=np.float64)
    c_valid = ~np.isnan(calib["unc"])
    routed_cal = c_valid & (calib["u1"] <= lr[calib["split"]])
    routed_test = test["u1"] <= lr[test["split"]]
    print(f"  router threshold at beta={BETA_MAIN}: mean={np.nanmean(lr):.4f}")

    grid = {}
    for alpha in alphas:
        t0 = time.time()
        th = _calibrate_per_clinic(calib, n_splits, alpha, routed_cal)
        rows = {str(k): clinic_contamination(k, test, routed_test, th[k],
                                            n_splits)
                for k in range(NUM_DISEASES)}
        grid[f"{alpha:g}"] = {"alpha": alpha, "per_disease": rows}
        red = [r["rho_reduction"] for r in rows.values()
               if r["rho_reduction"] is not None]
        rw = [r["Ref_wrong"] for r in rows.values() if r["Ref_wrong"] is not None]
        print(f"  alpha={alpha:<5} mean rho reduction={np.mean(red):+.4f}  "
              f"mean Ref_wrong={np.mean(rw):.4f}  ({time.time() - t0:.0f}s)")
    return grid


CSV_FIELDS = [
    "k", "disease_name", "alpha",
    "total_routed", "total_wrong_routed", "total_accepted",
    "rho_route", "rho_route_split_mean", "rho_route_split_sd",
    "rho_acc", "rho_acc_split_mean", "rho_acc_split_sd",
    "rho_reduction", "rho_acc_over_route",
    "Ref_wrong", "Ref_wrong_split_mean", "Ref_wrong_split_sd",
    "N_route_err", "N_diag_err", "N_err", "q_route_err", "q_diag_err",
    "n_feasible_splits", "rho_acc_n_estimable_splits",
]


def write_outputs(grid, alphas):
    meta = {
        "alpha_grid": alphas,
        "beta": BETA_MAIN,
        "naming": ("Ref_wrong follows Eq. (54) of the current Methods. The same "
                   "quantity appears as Rej_wrong in metrics.json and Table 4, "
                   "which predate it."),
        "alpha_dependence": ("rho_route is fixed by the router threshold and is "
                             "identical across the grid; it is repeated so each "
                             "row is self-contained. Every quantity measured "
                             "after specialist rejection varies with alpha."),
        "estimator": ("Counts are aggregated over held-out decisions from all "
                      "splits before taking ratios. Split-wise mean and SD are "
                      "computed over splits where the denominator is non-zero, "
                      "whose number is reported. Infeasible splits accept "
                      "nothing rather than being skipped."),
        "definitions": {
            "rho_route": "Wrong-routed fraction of the routed stream.",
            "rho_acc": "Wrong-routed fraction among accepted cases.",
            "Ref_wrong": "Fraction of wrong-routed cases the specialist rejects.",
            "q_route_err": "Share of accepted errors caused by routing.",
            "q_diag_err": "Share of accepted errors caused by misdiagnosis.",
        },
    }
    with open(OUT_JSON, "w") as f:
        json.dump({"meta": meta, "grid": grid}, f, indent=1)
    print(f"\nwrote {OUT_JSON} ({os.path.getsize(OUT_JSON) / 1024:.0f} KB)")

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for a in alphas:
            blk = grid[f"{a:g}"]
            for k in range(NUM_DISEASES):
                row = dict(blk["per_disease"][str(k)])
                row["k"] = k
                row["alpha"] = a
                w.writerow(row)
    print(f"wrote {OUT_CSV}")


def print_table(grid, alphas):
    print("\n" + "=" * 100)
    print(f"Routing contamination across the specialist risk grid "
          f"(beta = {BETA_MAIN})")
    print("=" * 100)
    for a in alphas:
        rows = grid[f"{a:g}"]["per_disease"]
        print(f"\n  alpha = {a}")
        print(f"  {'clinic':<28}{'rho_route':>11}{'rho_acc':>10}"
              f"{'reduction':>11}{'acc/route':>11}{'Ref_wrong':>11}"
              f"{'q_route':>9}{'N_err':>10}")
        for k in range(NUM_DISEASES):
            r = rows[str(k)]
            print(f"  D{k} {r['disease_name']:<24}{r['rho_route']:>11.4f}"
                  f"{r['rho_acc']:>10.4f}{r['rho_reduction']:>+11.4f}"
                  f"{r['rho_acc_over_route']:>11.3f}{r['Ref_wrong']:>11.4f}"
                  f"{r['q_route_err']:>9.4f}{r['N_err']:>10,}")
        red = [rows[str(k)]["rho_reduction"] for k in range(NUM_DISEASES)]
        rw = [rows[str(k)]["Ref_wrong"] for k in range(NUM_DISEASES)]
        ratio = [rows[str(k)]["rho_acc_over_route"] for k in range(NUM_DISEASES)]
        print(f"  {'':<28}{'mean':>11}{'':>10}{np.mean(red):>+11.4f}"
              f"{np.mean(ratio):>11.3f}{np.mean(rw):>11.4f}")


def main():
    print(f"Loading data (beta={BETA_MAIN})...")
    dl = DataLoader(beta=BETA_MAIN, alpha=0.2)
    grid = compute_grid(dl)
    print_table(grid, ALPHA_GRID)
    write_outputs(grid, ALPHA_GRID)


if __name__ == "__main__":
    main()
