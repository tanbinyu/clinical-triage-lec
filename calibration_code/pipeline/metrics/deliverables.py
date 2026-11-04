"""Assemble the machine-readable deliverables (instructions section 20).

Fourteen files are produced from the artefacts already computed, so nothing is
recalibrated here. Files 02 and 03 are slices of the per-split counts; 04 to 13
are flattened views of the grid JSONs; 01 is derived from the split definitions;
14 runs the identity checks of section 19 against the per-split counts, which is
the only level at which they are meaningful.

Usage:
    cd calibration_code
    python3 -m pipeline.metrics.deliverables
"""
import csv
import json
import os
import sys
from collections import defaultdict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pipeline.metrics.config import NUM_DISEASES, DISEASE_NAMES

METRICS = os.path.join(PROJECT_ROOT, "result", "metrics")
OUT = os.path.join(METRICS, "deliverables")
PER_SPLIT = os.path.join(METRICS, "per_split")
SPLITS_JSON = os.path.join(PROJECT_ROOT, "result", "json", "step2",
                           "splits_phase_1.json")

ALPHA_DEFAULT, BETA_DEFAULT = 0.20, 0.10


def _write(name, fields, rows):
    path = os.path.join(OUT, name)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"  {name:<46} {len(rows):>8,} rows  "
          f"{os.path.getsize(path) / 1024:>8.0f} KB")
    return len(rows)


def _load(name):
    with open(os.path.join(METRICS, name)) as f:
        return json.load(f)


# ---------------------------------------------------------------- 01
def split_manifest():
    with open(SPLITS_JSON) as f:
        splits = json.load(f)["splits"]
    rows = []
    for b, sp in enumerate(splits):
        n_c, n_t = len(sp["calib"]), len(sp["test"])
        rows.append({
            "split_id": b,
            "epoch": sp.get("epoch"),
            "n_calibration": n_c,
            "n_test": n_t,
            "n_total": n_c + n_t,
            "calibration_ratio": round(n_c / (n_c + n_t), 6),
            "disjoint": int(not (set(sp["calib"]) & set(sp["test"]))),
        })
    return _write("01_split_manifest.csv",
                  ["split_id", "epoch", "n_calibration", "n_test", "n_total",
                   "calibration_ratio", "disjoint"], rows)


# ---------------------------------------------------------------- 02, 03
def default_per_split():
    n = 0
    for src, dst in (("system_per_split.csv", "02_default_system_per_split.csv"),
                     ("specialist_per_split.csv",
                      "03_default_specialist_per_split.csv")):
        with open(os.path.join(PER_SPLIT, src)) as f:
            rd = csv.DictReader(f)
            fields = rd.fieldnames
            rows = [r for r in rd
                    if r["method"] == "proposed"
                    and float(r["alpha"]) == ALPHA_DEFAULT
                    and float(r["beta"]) == BETA_DEFAULT]
        n += _write(dst, fields, rows)
    return n


# ---------------------------------------------------------------- 04, 05, 08, 09
def _grid_views(sys_name, spec_name, alpha_filter=None):
    """Flatten the alpha grid. Files 04/05 carry the default operating point,
    files 08/09 the whole grid, so the two pairs are not copies."""
    data = _load("baseline_alpha_grid.json")
    scope = {s["key"]: s for s in data["meta"]["systems"]}
    sys_rows, spec_rows = [], []
    for key, rows in data["grid"].items():
        for r in rows:
            if alpha_filter is not None and abs(r["alpha"] - alpha_filter) > 1e-9:
                continue
            base = {"system": scope[key]["label"],
                    "system_key": key,
                    "guarantee_scope": scope[key]["guarantee_scope"]}
            sys_rows.append({**base, **{k: v for k, v in r.items()
                                        if k != "specialist"}})
            for c in range(NUM_DISEASES):
                s = r["specialist"][str(c)]
                spec_rows.append({**base, "clinic_id": c,
                                  "alpha": r["alpha"], "beta": r["beta"],
                                  **{k: v for k, v in s.items()
                                     if k not in ("ci_low", "ci_high")}})
    sys_fields = list(dict.fromkeys(k for r in sys_rows for k in r))
    spec_fields = list(dict.fromkeys(k for r in spec_rows for k in r))
    return (_write(sys_name, sys_fields, sys_rows)
            + _write(spec_name, spec_fields, spec_rows))


# ---------------------------------------------------------------- 06, 07
def beta_views():
    data = _load("baseline_beta_grid.json")
    scope = {s["key"]: s for s in data["meta"]["systems"]}
    sys_rows, spec_rows = [], []
    for key, rows in data["grid"].items():
        for r in rows:
            base = {"system": scope[key]["label"], "system_key": key,
                    "guarantee_scope": scope[key]["guarantee_scope"]}
            sys_rows.append({**base, **{k: v for k, v in r.items()
                                        if k not in ("specialist",
                                                     "router_stream")}})
            for c in range(NUM_DISEASES):
                s = r["specialist"][str(c)]
                stream = r["router_stream"][str(c)]
                spec_rows.append({**base, "clinic_id": c,
                                  "alpha": r["alpha"], "beta": r["beta"],
                                  **{k: v for k, v in s.items()
                                     if k not in ("ci_low", "ci_high")},
                                  **{k: v for k, v in stream.items()
                                     if k != "disease_name"}})
    sf = list(dict.fromkeys(k for r in sys_rows for k in r))
    kf = list(dict.fromkeys(k for r in spec_rows for k in r))
    return (_write("06_beta_sensitivity_system.csv", sf, sys_rows)
            + _write("07_beta_sensitivity_specialist.csv", kf, spec_rows))


# ---------------------------------------------------------------- 10, 11
def calibration_size_views():
    data = _load("calibration_size_grid.json")
    sys_rows, spec_rows = [], []
    for blk in data["grid"].values():
        head = {"ratio": blk["ratio"], "alpha": blk["alpha"],
                "beta": blk["beta"],
                "mean_calibration_size": blk["mean_calibration_size"]}
        sys_rows.append({**head, **blk["router"], **blk["system"]})
        for c in range(NUM_DISEASES):
            s = blk["per_disease"][str(c)]
            row = {**head, "clinic_id": c,
                   **{k: v for k, v in s.items()
                      if k != "routed_calibration_size"}}
            for stat, v in s["routed_calibration_size"].items():
                row[f"cal_size_{stat}"] = v
            spec_rows.append(row)
    sf = list(dict.fromkeys(k for r in sys_rows for k in r))
    kf = list(dict.fromkeys(k for r in spec_rows for k in r))
    return (_write("10_calibration_size_system.csv", sf, sys_rows)
            + _write("11_calibration_size_specialist.csv", kf, spec_rows))


# ---------------------------------------------------------------- 12
def contamination_view():
    data = _load("contamination_alpha_grid.json")
    rows = []
    for blk in data["grid"].values():
        for c in range(NUM_DISEASES):
            rows.append({"clinic_id": c, "alpha": blk["alpha"],
                         "beta": data["meta"]["beta"],
                         **blk["per_disease"][str(c)]})
    fields = list(dict.fromkeys(k for r in rows for k in r))
    return _write("12_contamination_and_error_decomposition.csv", fields, rows)


# ---------------------------------------------------------------- 13
def threshold_feasibility_view():
    """Thresholds, feasibility and effective calibration size in one place."""
    cal = _load("calibration_size_grid.json")["grid"]
    exch = _load("exchangeability_check.json")["clinics"]
    rows = []
    for key, blk in cal.items():
        r = blk["router"]
        rows.append({
            "decision_point": "router", "clinic_id": None,
            "disease_name": None, "ratio": blk["ratio"],
            "alpha": blk["alpha"], "beta": blk["beta"],
            "threshold_mean": r["lambda_R_mean"],
            "threshold_sd": r["lambda_R_sd"],
            "n_feasible_splits": r["n_router_feasible_splits"],
            "f_R_infeas": r["f_R_infeas"],
        })
        for c in range(NUM_DISEASES):
            s = blk["per_disease"][str(c)]
            d = s["routed_calibration_size"]
            rows.append({
                "decision_point": f"specialist_{c}", "clinic_id": c,
                "disease_name": DISEASE_NAMES[c], "ratio": blk["ratio"],
                "alpha": blk["alpha"], "beta": blk["beta"],
                "threshold_mean": s["threshold_mean"],
                "threshold_sd": s["threshold_sd"],
                "n_feasible_splits": s["n_feasible_splits"],
                "f_R_infeas": r["f_R_infeas"],
                "f_cond_infeas": s["f_cond_infeas"],
                "f_unavail": s["f_unavail"],
                "n_empty_stream_splits": s["n_empty_stream_splits"],
                "n_lec_infeasible_splits": s["n_lec_infeasible_splits"],
                "cal_size_mean": d["mean"], "cal_size_sd": d["sd"],
                "cal_size_median": d["median"], "cal_size_min": d["min"],
                "cal_size_p2.5": d["p2.5"], "cal_size_p97.5": d["p97.5"],
                # Held-out KS against its null is alpha- and ratio-independent,
                # so it is attached once as a stream-quality reference.
                "ks_observed": exch[str(c)]["ks_mean"],
                "ks_null_mean": exch[str(c)]["ks_null_mean"],
            })
    fields = list(dict.fromkeys(k for r in rows for k in r))
    return _write("13_threshold_and_feasibility.csv", fields, rows)


# ---------------------------------------------------------------- 14
def sanity_report():
    """Identity checks of section 19, evaluated on the per-split counts."""
    lines, failures = [], 0

    def check(name, ok, detail=""):
        nonlocal failures
        if not ok:
            failures += 1
        lines.append(f"[{'PASS' if ok else 'FAIL'}] {name}"
                     + (f"\n       {detail}" if detail else ""))

    # --- per-split structural identities --------------------------------
    with open(os.path.join(PER_SPLIT, "system_per_split.csv")) as f:
        srows = list(csv.DictReader(f))
    bad = defaultdict(int)
    for r in srows:
        g = lambda k: int(r[k])
        if g("n_ai_correct") + g("n_ai_error") + g("n_human") != g("n_test"):
            bad["N_AI_correct + N_AI_error + N_human = N_test"] += 1
        if g("n_router_accepted") + g("n_router_referrals") != g("n_test"):
            bad["router accepted + router referrals = N_test"] += 1
        if g("n_system_accepted") + g("n_specialist_stage_referrals") \
                != g("n_router_accepted"):
            bad["system accepted + specialist referrals = router accepted"] += 1
        if g("n_system_accepted") > g("n_router_accepted"):
            bad["system accepted <= router accepted"] += 1
        if g("n_system_accepted_errors") > g("n_system_accepted"):
            bad["accepted errors <= accepted cases"] += 1
        if g("n_router_accepted_errors") > g("n_router_accepted"):
            bad["router accepted errors <= router accepted"] += 1
    for name in ("N_AI_correct + N_AI_error + N_human = N_test",
                 "router accepted + router referrals = N_test",
                 "system accepted + specialist referrals = router accepted",
                 "system accepted <= router accepted",
                 "accepted errors <= accepted cases",
                 "router accepted errors <= router accepted"):
        check(f"{name}  ({len(srows):,} system rows)", bad[name] == 0,
              f"{bad[name]} violations" if bad[name] else "")

    with open(os.path.join(PER_SPLIT, "specialist_per_split.csv")) as f:
        krows = list(csv.DictReader(f))
    kbad = defaultdict(int)
    thresh_on_infeasible = 0
    for r in krows:
        g = lambda k: int(r[k])
        if g("n_accepted_routing_errors") + g("n_accepted_diagnostic_errors") \
                != g("n_specialist_accepted_errors"):
            kbad["N_err = N_route_err + N_diag_err"] += 1
        if g("n_specialist_accepted") > g("n_routed_test"):
            kbad["specialist accepted <= routed test"] += 1
        if g("n_specialist_accepted_errors") > g("n_specialist_accepted"):
            kbad["accepted errors <= accepted cases (clinic)"] += 1
        if g("n_wrong_routed_test") > g("n_routed_test"):
            kbad["wrong routed <= routed"] += 1
        if r["specialist_feasible"] == "0" and r["specialist_threshold"]:
            thresh_on_infeasible += 1
    for name in ("N_err = N_route_err + N_diag_err",
                 "specialist accepted <= routed test",
                 "accepted errors <= accepted cases (clinic)",
                 "wrong routed <= routed"):
        check(f"{name}  ({len(krows):,} clinic rows)", kbad[name] == 0,
              f"{kbad[name]} violations" if kbad[name] else "")
    check("threshold recorded only on feasible splits",
          thresh_on_infeasible == 0,
          f"{thresh_on_infeasible} infeasible rows carry a threshold"
          if thresh_on_infeasible else "")

    # --- calibration and test sets are disjoint -------------------------
    with open(SPLITS_JSON) as f:
        splits = json.load(f)["splits"]
    overlaps = sum(1 for sp in splits
                   if set(sp["calib"]) & set(sp["test"]))
    check(f"no test sample used in calibration  ({len(splits)} splits)",
          overlaps == 0, f"{overlaps} splits overlap" if overlaps else "")

    # --- pooled identities on the aggregates ----------------------------
    agg = _load("baseline_alpha_grid.json")["grid"]
    micro_bad = ref_bad = 0
    for rows in agg.values():
        for r in rows:
            if abs(r["mFDR_spec_micro"] - r["mFDR_sys"]) > 1e-9:
                micro_bad += 1
            if abs(r["Ref_sys"] - (1 - r["Cov_sys"])) > 1e-9:
                ref_bad += 1
    n_cells = sum(len(v) for v in agg.values())
    check(f"micro specialist mFDR = system mFDR  ({n_cells} cells)",
          micro_bad == 0, f"{micro_bad} violations" if micro_bad else "")
    check(f"Ref_sys = 1 - Cov_sys  ({n_cells} cells)", ref_bad == 0,
          f"{ref_bad} violations" if ref_bad else "")

    con = _load("contamination_alpha_grid.json")["grid"]
    q_bad = ref_w_bad = 0
    n_con = 0
    for blk in con.values():
        for c in range(NUM_DISEASES):
            s = blk["per_disease"][str(c)]
            n_con += 1
            if abs(s["q_route_err"] + s["q_diag_err"] - 1) > 1e-6:
                q_bad += 1
            if abs(s["Ref_wrong"]
                   - (1 - s["N_route_err"] / s["total_wrong_routed"])) > 1e-5:
                ref_w_bad += 1
    check(f"q_route_err + q_diag_err = 1  ({n_con} cells)", q_bad == 0,
          f"{q_bad} violations" if q_bad else "")
    check(f"Ref_wrong = 1 - N_route_err / n_wrong_routed  ({n_con} cells)",
          ref_w_bad == 0, f"{ref_w_bad} violations" if ref_w_bad else "")

    # --- monotonicity expectations of section 16 -------------------------
    prop = sorted((r for r in agg["proposed"]), key=lambda r: r["alpha"])
    mono_th = mono_cov = 0
    for c in range(NUM_DISEASES):
        th = [r["specialist"][str(c)]["threshold_mean"] for r in prop]
        sel = [r["specialist"][str(c)]["total_selected"] for r in prop]
        mono_th += any(th[i + 1] < th[i] - 1e-9 for i in range(len(th) - 1))
        mono_cov += any(sel[i + 1] < sel[i] for i in range(len(sel) - 1))
    check("specialist threshold non-decreasing in alpha", mono_th == 0,
          f"{mono_th} clinics violate" if mono_th else "")
    check("specialist accepted count non-decreasing in alpha", mono_cov == 0,
          f"{mono_cov} clinics violate" if mono_cov else "")
    cov = [r["Cov_sys"] for r in prop]
    check("system coverage non-decreasing in alpha",
          all(cov[i + 1] >= cov[i] - 1e-9 for i in range(len(cov) - 1)))

    # --- reconstruction from the raw counts -----------------------------
    tot = defaultdict(lambda: defaultdict(int))
    for r in srows:
        key = (r["method"], float(r["alpha"]), float(r["beta"]))
        for fld in ("n_test", "n_system_accepted", "n_system_accepted_errors"):
            tot[key][fld] += int(r[fld])
    recon_bad = 0
    for key, rows in agg.items():
        for r in rows:
            d = tot[(key, r["alpha"], BETA_DEFAULT)]
            # The grids store ratios rounded to six decimals, so the
            # reconstruction is rounded the same way before comparing; a raw
            # tolerance of 1e-9 would flag every cell on rounding alone.
            if round(d["n_system_accepted_errors"] / d["n_system_accepted"],
                     6) != r["mFDR_sys"]:
                recon_bad += 1
            if round(d["n_system_accepted"] / d["n_test"], 6) != r["Cov_sys"]:
                recon_bad += 1
    check(f"aggregates reproducible from per-split counts  ({n_cells} cells)",
          recon_bad == 0, f"{recon_bad} mismatches" if recon_bad else "")

    header = [
        "Sanity check report",
        "=" * 72,
        "Identities from section 19 of the experiment instructions, evaluated",
        "on the per-split counts in result/metrics/per_split/ and on the",
        "aggregate grids. Monotonicity expectations follow section 16.",
        "",
    ]
    footer = ["", "=" * 72,
              f"{len(lines) - failures} passed, {failures} failed"]
    path = os.path.join(OUT, "14_sanity_check_report.txt")
    with open(path, "w") as f:
        f.write("\n".join(header + lines + footer) + "\n")
    print(f"  {'14_sanity_check_report.txt':<46} "
          f"{len(lines):>8} checks  {os.path.getsize(path) / 1024:>8.0f} KB")
    return failures


def main():
    os.makedirs(OUT, exist_ok=True)
    print(f"Writing deliverables to {OUT}\n")
    split_manifest()
    default_per_split()
    _grid_views("04_baseline_system_results.csv",
                "05_baseline_specialist_results.csv",
                alpha_filter=ALPHA_DEFAULT)
    beta_views()
    _grid_views("08_alpha_sensitivity_system.csv",
                "09_alpha_sensitivity_specialist.csv")
    calibration_size_views()
    contamination_view()
    threshold_feasibility_view()
    failures = sanity_report()

    files = sorted(os.listdir(OUT))
    print(f"\n{len(files)} files written")
    if failures:
        print(f"WARNING: {failures} sanity checks failed; see "
              f"14_sanity_check_report.txt")
    else:
        print("All sanity checks passed.")


if __name__ == "__main__":
    main()
