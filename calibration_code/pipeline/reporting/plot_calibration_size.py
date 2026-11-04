"""Figure: how large the routed calibration stream has to be (Methods 2.14).

Reads result/metrics/calibration_size_grid.json.

The organising variable is the per-clinic routed calibration size, not the
nominal calibration ratio, because that is the quantity the guarantee depends
on and it varies fifteen-fold across clinics at any fixed ratio. Pooling the
fifty clinic-by-ratio cells therefore turns the sweep into one relationship.

Panel (b) marks the hard feasibility floor: LEC needs E - alpha*N <= -1 with
E >= 0, so no threshold can be feasible unless the stream carries at least
1/alpha cases. That single line explains why the smallest clinic is infeasible
in every split at the strictest level and the tightest ratio.

Usage:
    cd calibration_code
    python3 -m pipeline.reporting.plot_calibration_size
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pipeline.metrics.config import NUM_DISEASES
from pipeline.reporting.plot_metrics import (
    C, INK, _setup_style, _savefig, _darken, _panel, _legend, _soft_ygrid,
)

DATA_JSON = os.path.join(PROJECT_ROOT, "result", "metrics",
                         "calibration_size_grid.json")
FIG_DIR = os.path.join(PROJECT_ROOT, "result", "metrics", "figures")

FOCUS = 9
A_STYLE = {0.05: dict(color=C["risk"], marker="o"),
           0.2: dict(color=C["coverage"], marker="^")}


def _cells(grid, alpha, field, size_field="mean"):
    """Every (clinic, ratio) cell for one alpha, as (size, value, clinic)."""
    out = []
    for key, blk in grid.items():
        if abs(blk["alpha"] - alpha) > 1e-9:
            continue
        for k in range(NUM_DISEASES):
            c = blk["per_disease"][str(k)]
            v = c[field]
            if v is None:
                continue
            out.append((c["routed_calibration_size"][size_field], v, k))
    return out


def plot_size(data):
    grid = data["grid"]
    meta = data["meta"]
    ratios = meta["ratios"]
    alphas = meta["alphas"]

    fig, axes = plt.subplots(2, 2, figsize=(7.6, 6.6), constrained_layout=True)
    (ax_a, ax_b), (ax_c, ax_d) = axes

    # ── (a) held-out excess against stream size ─────────────────────────
    for alpha in alphas:
        st = A_STYLE[alpha]
        cells = _cells(grid, alpha, "excess_FDP")
        xs = np.array([c[0] for c in cells])
        ys = np.array([c[1] for c in cells])
        ax_a.scatter(xs, ys, s=[52 if c[2] == FOCUS else 24 for c in cells],
                     marker=st["marker"], color=st["color"],
                     edgecolor=_darken(st["color"], 0.3), linewidth=0.5,
                     alpha=0.85, zorder=5, label=f"$\\alpha$ = {alpha:g}")
        # Median trend over size octaves, to show the shape without fitting a
        # model the data does not justify.
        edges = np.array([10, 25, 50, 100, 200, 400])
        mids, meds = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            sel = (xs >= lo) & (xs < hi)
            if sel.sum() >= 2:
                mids.append(np.sqrt(lo * hi))
                meds.append(np.median(ys[sel]))
        ax_a.plot(mids, meds, "-", color=_darken(st["color"], 0.2),
                  linewidth=1.3, alpha=0.9, zorder=4)
    ax_a.axhline(0, color=C["gray"], linewidth=0.8, alpha=0.7)
    ax_a.set_xscale("log")
    ax_a.set_xlabel("Routed calibration size of the clinic (cases per split)")
    ax_a.set_ylabel("Held-out excess FDP$_{test}$ $-$ FDP$_{cal}$")
    ax_a.set_ylim(-0.03, 0.19)
    ax_a.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax_a.set_xticks([10, 25, 50, 100, 200, 400])
    _soft_ygrid(ax_a)
    _legend(ax_a, loc="upper right", fontsize=6.8)
    _panel(ax_a, "a", "Guarantee transfer vs stream size")

    # ── (b) feasibility against stream size, with the 1/alpha floor ─────
    for alpha in alphas:
        st = A_STYLE[alpha]
        cells = _cells(grid, alpha, "f_cond_infeas")
        ax_b.scatter([c[0] for c in cells], [c[1] for c in cells],
                     s=[52 if c[2] == FOCUS else 24 for c in cells],
                     marker=st["marker"], color=st["color"],
                     edgecolor=_darken(st["color"], 0.3), linewidth=0.5,
                     alpha=0.85, zorder=5, label=f"$\\alpha$ = {alpha:g}")
        floor = 1.0 / alpha
        ax_b.axvline(floor, color=st["color"], linestyle=":", linewidth=1.1,
                     alpha=0.8, zorder=3)
        ax_b.text(floor * 1.06, 0.94 if alpha == 0.05 else 0.80,
                  f"$1/\\alpha$ = {floor:.0f}", fontsize=6.5,
                  color=_darken(st["color"], 0.2), rotation=90, va="top")
    ax_b.set_xscale("log")
    ax_b.set_xlabel("Routed calibration size of the clinic (cases per split)")
    ax_b.set_ylabel("$\\hat{f}_{k|R}^{infeas}$ (conditional infeasibility)")
    ax_b.set_ylim(-0.05, 1.05)
    ax_b.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax_b.set_xticks([10, 25, 50, 100, 200, 400])
    _soft_ygrid(ax_b)
    _legend(ax_b, loc="center right", fontsize=6.8)
    _panel(ax_b, "b", "Feasibility floor")

    # ── (c) the smallest clinic, ratio by ratio ─────────────────────────
    xr = np.arange(len(ratios))
    d9 = [grid[f"{r:g}|0.2"]["per_disease"][str(FOCUS)] for r in ratios]
    sizes = [c["routed_calibration_size"]["mean"] for c in d9]
    ax_c.bar(xr, sizes, width=0.55, color=C["light"],
             edgecolor=_darken(C["gray"], 0.1), linewidth=0.6, zorder=3,
             label=f"D{FOCUS} mean routed calibration size")
    ax_c.errorbar(xr, sizes,
                  yerr=[np.array(sizes) - np.array(
                      [c["routed_calibration_size"]["min"] for c in d9]),
                      np.array([c["routed_calibration_size"]["p97.5"]
                                for c in d9]) - np.array(sizes)],
                  fmt="none", ecolor=_darken(C["gray"], 0.2), elinewidth=0.8,
                  capsize=2.5, zorder=5)
    for alpha in alphas:
        ax_c.axhline(1.0 / alpha, color=A_STYLE[alpha]["color"], linestyle=":",
                     linewidth=1.1, alpha=0.85, zorder=4,
                     label=f"$1/\\alpha$ floor at $\\alpha$ = {alpha:g}")
    ax_c.set_xticks(xr)
    ax_c.set_xticklabels([f"{r:.0%}" for r in ratios])
    ax_c.set_xlabel("Calibration ratio")
    ax_c.set_ylabel(f"D{FOCUS} routed calibration size")
    ax_c.set_ylim(0, 105)
    _soft_ygrid(ax_c)
    _legend(ax_c, loc="upper left", fontsize=6.3)
    _panel(ax_c, "c", "The binding clinic")

    # ── (d) what the system pays ────────────────────────────────────────
    for alpha in alphas:
        st = A_STYLE[alpha]
        km = [grid[f"{r:g}|{alpha:g}"]["system"]["K_meet"] for r in ratios]
        nev = [grid[f"{r:g}|{alpha:g}"]["system"]["n_evaluable_specialists"]
               for r in ratios]
        ax_d.plot(xr, km, marker=st["marker"], color=st["color"], linewidth=1.6,
                  markersize=6, markeredgecolor=_darken(st["color"], 0.3),
                  markeredgewidth=0.6, zorder=5,
                  label=f"$K_{{meet}}$ at $\\alpha$ = {alpha:g}")
        # Where a clinic stops being estimable at all, K_meet is out of fewer
        # than ten and the count alone would be misleading.
        for i, (a_, b_) in enumerate(zip(km, nev)):
            if b_ < NUM_DISEASES:
                ax_d.annotate(f"{a_}/{b_}", xy=(xr[i], a_), xytext=(0, -13),
                              textcoords="offset points", fontsize=6.5,
                              ha="center", color=_darken(st["color"], 0.2))
    ax_d.set_xticks(xr)
    ax_d.set_xticklabels([f"{r:.0%}" for r in ratios])
    ax_d.set_xlabel("Calibration ratio")
    ax_d.set_ylabel("Clinics meeting the target ($K_{meet}$)")
    ax_d.set_ylim(-0.6, 10.8)
    ax_d.set_yticks(range(0, 11, 2))
    _soft_ygrid(ax_d)
    _legend(ax_d, loc="center right", fontsize=6.8)
    _panel(ax_d, "d", "Clinics under control")

    fig.suptitle("Calibration-Size Sensitivity per Clinic "
                 f"($\\beta$ = {meta['beta']})",
                 fontsize=11.5, fontweight="bold", color=INK["text"])
    _savefig(fig, "fig12_calibration_size")
    print("  Fig 12: calibration-size sensitivity")


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    _setup_style()
    with open(DATA_JSON) as f:
        data = json.load(f)
    print("Generating figure...")
    plot_size(data)
    print(f"\nSaved to {FIG_DIR}")


if __name__ == "__main__":
    main()
