"""Per-split raw counts for every method and risk level.

Every table and figure in this project is a ratio of counts taken over held-out
decisions. Storing those counts per split, rather than only the finished
aggregates, means the tables and figures can be rebuilt without recalibrating
anything, and it makes the identity checks in the sanity report possible at the
level where they are meaningful.

Two tables are written, with the field sets specified in the experiment
instructions:

  system-level     one row per (method, alpha, beta, split)
  specialist-level one row per (method, alpha, beta, split, clinic)

Configurations cover the alpha grid at the default router level and the beta
grid at the default specialist level, which together are the source of the
baseline comparison and both sensitivity analyses. Cells shared by the two
sweeps are emitted once.

Infeasibility is recorded rather than smoothed over: an infeasible decision
point accepts nothing and its cases become referrals, so counts stay additive
and no threshold is invented for a split that had none.

Usage:
    cd calibration_code
    python3 -m pipeline.metrics.per_split_counts
"""
import csv
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
from pipeline.metrics.baseline_alpha_grid import (
    SYSTEMS, _accept_all_matrix, _calibrate_per_clinic, _calibrate_shared,
)

ALPHA_GRID = [0.05, 0.10, 0.15, 0.20]
BETA_GRID = [0.01, 0.025, 0.05, 0.10, 0.15, 0.20]
ALPHA_DEFAULT = 0.20
BETA_DEFAULT = 0.10
CAL_RATIO = 0.50

OUT_DIR = os.path.join(PROJECT_ROOT, "result", "metrics", "per_split")
SYS_CSV = os.path.join(OUT_DIR, "system_per_split.csv")
SPEC_CSV = os.path.join(OUT_DIR, "specialist_per_split.csv")

SYS_FIELDS = [
    "split_id", "method", "alpha", "beta", "calibration_ratio",
    "n_test",
    "router_feasible", "router_threshold",
    "n_router_accepted", "n_router_accepted_errors", "n_router_referrals",
    "n_system_accepted", "n_system_accepted_errors",
    "n_specialist_stage_referrals",
    "n_ai_correct", "n_ai_error", "n_human",
]

SPEC_FIELDS = [
    "split_id", "method", "clinic_id", "alpha", "beta",
    "router_feasible", "specialist_feasible",
    "n_routed_calibration", "n_wrong_routed_calibration",
    "specialist_threshold",
    "n_routed_test", "n_wrong_routed_test",
    "n_specialist_accepted", "n_specialist_accepted_errors",
    "n_accepted_routing_errors", "n_accepted_diagnostic_errors",
    "n_wrong_routed_referred", "n_technically_incompatible",
    "base_error_numerator", "base_error_denominator",
]


def configurations():
    """(method, alpha, beta) cells to emit, with shared cells emitted once."""
    seen, out = set(), []
    for alpha in ALPHA_GRID:
        for key, _, _, _ in SYSTEMS:
            out.append((key, alpha, BETA_DEFAULT))
            seen.add((key, alpha, BETA_DEFAULT))
    for beta in BETA_GRID:
        for key, _, _, _ in SYSTEMS:
            cell = (key, ALPHA_DEFAULT, beta)
            if cell not in seen:
                out.append(cell)
                seen.add(cell)
    return out


def router_thresholds(dl, beta, cache):
    if beta in cache:
        return cache[beta]
    splits = dl.splits_phase1["splits"]
    out = np.full(len(splits), np.nan)
    for b, sp in enumerate(splits):
        ids = sp["calib"]
        u = np.fromiter((dl.phase1_lookup[s][0] for s in ids), np.float64, len(ids))
        a = np.fromiter((dl.phase1[s]["correctness"] for s in ids), np.int64, len(ids))
        th = _lec_calibrate_np(u, a, beta)
        if th is not None:
            out[b] = th
    cache[beta] = out
    return out


def specialist_thresholds(method, calib, n_splits, alpha, routed_cal, c_valid):
    """Per-clinic thresholds for one method. +inf means accept everything."""
    if method in ("full_automation", "router_only"):
        return _accept_all_matrix(n_splits), None
    if method == "specialist_only":
        return _calibrate_per_clinic(calib, n_splits, alpha, c_valid), None
    if method == "shared_threshold":
        return _calibrate_shared(calib, n_splits, alpha, routed_cal)
    if method == "oracle_clean":
        return _calibrate_per_clinic(calib, n_splits, alpha, routed_cal,
                                     per_clinic_extra=(calib["rc"] == 1)), None
    return _calibrate_per_clinic(calib, n_splits, alpha, routed_cal), None


def emit_rows(method, alpha, beta, dl, pre, lr, n_splits):
    """Per-split system and specialist rows for one configuration."""
    calib, test = pre["calib"], pre["test"]
    c_valid = ~np.isnan(calib["unc"])
    with np.errstate(invalid="ignore"):
        routed_cal = c_valid & (calib["u1"] <= lr[calib["split"]])
    th, _ = specialist_thresholds(method, calib, n_splits, alpha, routed_cal,
                                  c_valid)

    # Router-only and the proposed family gate on the router; the two systems
    # that ignore it see the unfiltered stream.
    uses_router = method not in ("full_automation", "specialist_only")
    with np.errstate(invalid="ignore"):
        s_r = (test["u1"] <= lr[test["split"]]) if uses_router \
            else np.ones(test["split"].shape, dtype=bool)
        acc = s_r & (test["unc"] <= th[test["pred_k"], test["split"]])

    router_ok = np.isfinite(lr)
    err = test["corr"] == 0
    wrong = test["rc"] == 0
    split = test["split"]

    def bc(mask, sel=None):
        s = split if sel is None else split[sel]
        m = mask if sel is None else mask[sel]
        return np.bincount(s[m], minlength=n_splits)

    n_test = np.bincount(split, minlength=n_splits)
    n_r_acc = bc(s_r)
    n_r_err = bc(s_r & wrong)
    n_acc = bc(acc)
    n_acc_err = bc(acc & err)

    sys_rows = []
    for b in range(n_splits):
        rt = lr[b]
        sys_rows.append({
            "split_id": b, "method": method, "alpha": alpha, "beta": beta,
            "calibration_ratio": CAL_RATIO,
            "n_test": int(n_test[b]),
            "router_feasible": int(bool(router_ok[b])),
            "router_threshold": (round(float(rt), 6) if np.isfinite(rt)
                                 else None) if uses_router else None,
            "n_router_accepted": int(n_r_acc[b]),
            "n_router_accepted_errors": int(n_r_err[b]),
            "n_router_referrals": int(n_test[b] - n_r_acc[b]),
            "n_system_accepted": int(n_acc[b]),
            "n_system_accepted_errors": int(n_acc_err[b]),
            "n_specialist_stage_referrals": int(n_r_acc[b] - n_acc[b]),
            "n_ai_correct": int(n_acc[b] - n_acc_err[b]),
            "n_ai_error": int(n_acc_err[b]),
            "n_human": int(n_test[b] - n_acc[b]),
        })

    spec_rows = []
    for k in range(NUM_DISEASES):
        kc = calib["pred_k"] == k
        kt = test["pred_k"] == k
        cal_k = routed_cal & kc if uses_router else c_valid & kc
        n_cal = np.bincount(calib["split"][cal_k], minlength=n_splits)
        n_cal_wrong = np.bincount(
            calib["split"][cal_k & (calib["rc"] == 0)], minlength=n_splits)

        routed_k = s_r & kt
        n_rt = bc(routed_k)
        n_rt_wrong = bc(routed_k & wrong)
        n_a = bc(acc & kt)
        n_a_err = bc(acc & kt & err)
        n_a_route = bc(acc & kt & wrong)
        n_a_diag = bc(acc & kt & ~wrong & err)
        n_ref_wrong = bc(routed_k & wrong & ~acc)
        n_base_err = bc(routed_k & err)

        for b in range(n_splits):
            t = th[k, b]
            feasible = bool(np.isfinite(t))
            spec_rows.append({
                "split_id": b, "method": method, "clinic_id": k,
                "alpha": alpha, "beta": beta,
                "router_feasible": int(bool(router_ok[b])),
                "specialist_feasible": int(feasible or np.isinf(t)),
                "n_routed_calibration": int(n_cal[b]),
                "n_wrong_routed_calibration": int(n_cal_wrong[b]),
                "specialist_threshold": round(float(t), 6) if feasible else None,
                "n_routed_test": int(n_rt[b]),
                "n_wrong_routed_test": int(n_rt_wrong[b]),
                "n_specialist_accepted": int(n_a[b]),
                "n_specialist_accepted_errors": int(n_a_err[b]),
                "n_accepted_routing_errors": int(n_a_route[b]),
                "n_accepted_diagnostic_errors": int(n_a_diag[b]),
                "n_wrong_routed_referred": int(n_ref_wrong[b]),
                # No technically incompatible inputs exist in this dataset; the
                # column is kept so the schema matches the specification.
                "n_technically_incompatible": 0,
                "base_error_numerator": int(n_base_err[b]),
                "base_error_denominator": int(n_rt[b]),
            })
    return sys_rows, spec_rows


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Loading data (default alpha={ALPHA_DEFAULT}, beta={BETA_DEFAULT})...")
    dl = DataLoader(beta=BETA_DEFAULT, alpha=ALPHA_DEFAULT)
    n_splits = len(dl.splits_phase1["splits"])
    print("Precomputing sample arrays...")
    t0 = time.time()
    pre = precompute_all_with_rc(dl)
    print(f"  done in {time.time() - t0:.1f}s")

    configs = configurations()
    print(f"{len(configs)} configurations x {n_splits} splits")
    lr_cache = {}
    n_sys = n_spec = 0

    with open(SYS_CSV, "w", newline="") as fs, open(SPEC_CSV, "w", newline="") as fk:
        ws = csv.DictWriter(fs, fieldnames=SYS_FIELDS)
        wk = csv.DictWriter(fk, fieldnames=SPEC_FIELDS)
        ws.writeheader()
        wk.writeheader()
        for i, (method, alpha, beta) in enumerate(configs, 1):
            t0 = time.time()
            lr = router_thresholds(dl, beta, lr_cache)
            s, k = emit_rows(method, alpha, beta, dl, pre, lr, n_splits)
            ws.writerows(s)
            wk.writerows(k)
            n_sys += len(s)
            n_spec += len(k)
            print(f"  [{i:>2}/{len(configs)}] {method:<18} "
                  f"alpha={alpha:<5} beta={beta:<6} "
                  f"({time.time() - t0:.1f}s)")

    for path, n in ((SYS_CSV, n_sys), (SPEC_CSV, n_spec)):
        print(f"\nwrote {path}")
        print(f"  {n:,} rows, {os.path.getsize(path) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
