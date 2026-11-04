"""Router risk sweep: does the first stage contribute? (Methods 2.12)

Reads result/metrics/baseline_beta_grid.json. Specialist-only rejection applies
no router threshold, so the vertical gap between it and the proposed system at
each beta is exactly what the routing stage adds. Panel (d) plots that gap
directly, because at the default beta = 0.1 it is too small to see on the same
axis as the curves themselves.

Usage:
    cd calibration_code
    python3 -m pipeline.reporting.plot_router_sweep
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
    C, INK, BL_STYLE, _setup_style, _savefig, _darken, _panel, _legend,
)

GRID_JSON = os.path.join(PROJECT_ROOT, "result", "metrics",
                         "baseline_beta_grid.json")
FIG_DIR = os.path.join(PROJECT_ROOT, "result", "metrics", "figures")

STYLE_KEY = {
    "full_automation": "Full Automation",
    "router_only": "Router-Only Rejection",
    "specialist_only": "Specialist-Only Rejection",
    "shared_threshold": "Shared Threshold",
    "oracle_clean": "Oracle-Clean Calibration",
    "proposed": "Proposed System (LEC)",
}
CURVE_ORDER = ["router_only", "shared_threshold", "oracle_clean", "proposed"]
FIXED_ORDER = ["full_automation", "specialist_only"]
# The two unfiltered-router systems are horizontal references; interleaved
# dashes keep both visible where they nearly coincide.
FIXED_DASH = {"full_automation": (0, (7, 3)), "specialist_only": (0, (2, 4))}


def _st(k):
    return BL_STYLE[STYLE_KEY[k]]


def _v(rows, f):
    return [r[f] for r in rows]


def _curve(ax, x, y, key, label=True):
    st = _st(key)
    hero = key == "proposed"
    ax.plot(x, y, marker=st["marker"], color=st["color"], linestyle="-",
            linewidth=1.7 if hero else 1.1,
            markersize=6.5 if hero else 4.5,
            markeredgecolor=_darken(st["color"], 0.25), markeredgewidth=0.6,
            label=st["short"] if label else None, zorder=5 if hero else 3)


def _fixed(ax, val, key, label=True):
    st = _st(key)
    ax.axhline(val, color=st["color"], linestyle=FIXED_DASH[key],
               linewidth=1.2, alpha=0.9, zorder=2,
               label=(st["short"] + " (no router)") if label else None)


def plot_sweep(data):
    grid, meta = data["grid"], data["meta"]
    betas = meta["beta_grid"]
    alpha = meta["alpha_fixed"]
    # The grid is discrete and its two smallest levels are 0.015 apart, which
    # collide as tick labels on a linear axis; evenly spaced positions keep
    # every level legible.
    x = np.arange(len(betas))
    xlim = (-0.25, len(betas) - 0.75)

    def _xaxis(ax):
        ax.set_xticks(x)
        ax.set_xticklabels([f"{b:g}" for b in betas])
        ax.set_xlim(*xlim)
        ax.set_xlabel("Router risk level $\\beta$")

    fig, axes = plt.subplots(2, 2, figsize=(7.6, 6.4), constrained_layout=True)
    (ax_a, ax_b), (ax_c, ax_d) = axes

    # ── (a) system risk ─────────────────────────────────────────────────
    for key in CURVE_ORDER:
        _curve(ax_a, x, _v(grid[key], "mFDR_sys"), key)
    for key in FIXED_ORDER:
        _fixed(ax_a, grid[key][0]["mFDR_sys"], key)
    ax_a.axhline(alpha, color=C["gray"], linestyle="--", linewidth=0.8, alpha=0.6)
    ax_a.text(xlim[1], alpha + 0.002, f"$\\alpha$ = {alpha}", fontsize=7,
              color=C["gray"], va="bottom", ha="right")
    ax_a.set_ylabel("System risk (mFDR$_{sys}$)")
    ax_a.set_ylim(0.13, 0.215)
    _xaxis(ax_a)
    _legend(ax_a, loc="lower right", fontsize=6.2, ncol=2)
    _panel(ax_a, "a", "System risk")

    # ── (b) coverage ────────────────────────────────────────────────────
    for key in CURVE_ORDER:
        _curve(ax_b, x, _v(grid[key], "Cov_sys"), key)
    for key in FIXED_ORDER:
        _fixed(ax_b, grid[key][0]["Cov_sys"], key)
    ax_b.set_ylabel("System coverage (Cov$_{sys}$)")
    ax_b.set_ylim(0.82, 1.015)
    _xaxis(ax_b)
    ax_b.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    _legend(ax_b, loc="lower right", fontsize=6.2, ncol=2)
    _panel(ax_b, "b", "AI-managed coverage")

    # ── (c) router stage itself ─────────────────────────────────────────
    rr = grid["proposed"]
    mfdr_r = _v(rr, "mFDR_R")
    # Once the empirical router error stops tracking beta the constraint is
    # slack and the threshold no longer moves.
    plateau = [i for i, m in enumerate(mfdr_r) if abs(m - mfdr_r[-1]) < 1e-4]
    if plateau:
        ax_c.axvspan(min(plateau) - 0.15, xlim[1], color=C["gray"],
                     alpha=0.06, zorder=0)
    ax_c.plot(x, betas, ":", color=C["gray"], linewidth=1.0,
              label="y = $\\beta$ (target)", zorder=2)
    ax_c.plot(x, mfdr_r, "o-", color=C["risk"], linewidth=1.7,
              markersize=6, markeredgecolor=_darken(C["risk"], 0.25),
              markeredgewidth=0.6, label="mFDR$_R$", zorder=5)
    ax_c.set_ylabel("Router accepted-case error (mFDR$_R$)")
    ax_c.set_ylim(0, 0.088)
    _xaxis(ax_c)
    if plateau:
        ax_c.text((min(plateau) + len(betas) - 1) / 2, 0.085,
                  "constraint slack", fontsize=6.5,
                  color=INK["muted"], ha="center", va="top", fontstyle="italic")
    ax2 = ax_c.twinx()
    ax2.plot(x, _v(rr, "Cov_R"), "^-", color=C["coverage"], linewidth=1.3,
             markersize=5, markeredgecolor=_darken(C["coverage"], 0.25),
             markeredgewidth=0.6, label="Cov$_R$", zorder=4)
    ax2.set_ylabel("Router coverage (Cov$_R$)", color=C["coverage"])
    ax2.tick_params(axis="y", labelcolor=C["coverage"], labelsize=8.5)
    ax2.set_ylim(0.84, 1.02)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_linewidth(0.7)
    ax2.spines["top"].set_visible(False)
    h1, l1 = ax_c.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax_c.legend(h1 + h2, l1 + l2, loc="lower right", frameon=False,
                fontsize=6.5, handlelength=1.5, handletextpad=0.5,
                borderaxespad=0.3)
    _panel(ax_c, "c", "Routing stage")

    # ── (d) what the routing stage adds ─────────────────────────────────
    prop, spec = grid["proposed"], grid["specialist_only"]
    d_risk = [p["mFDR_sys"] - s["mFDR_sys"] for p, s in zip(prop, spec)]
    d_cov = [p["Cov_sys"] - s["Cov_sys"] for p, s in zip(prop, spec)]
    ax_d.axhline(0, color=C["gray"], linewidth=0.8, alpha=0.7)
    ax_d.plot(x, d_risk, "o-", color=C["risk"], linewidth=1.7, markersize=6,
              markeredgecolor=_darken(C["risk"], 0.25), markeredgewidth=0.6,
              label="$\\Delta$ risk (gain)", zorder=5)
    ax_d.plot(x, d_cov, "^-", color=C["coverage"], linewidth=1.3,
              markersize=5, markeredgecolor=_darken(C["coverage"], 0.25),
              markeredgewidth=0.6, label="$\\Delta$ coverage (cost)", zorder=4)
    # Only the two strict levels move at all; labelling the flat tail would
    # just crowd the zero line. The second label is pushed further up so the
    # coverage curve does not run through it.
    for i, dy in ((0, 6), (1, 12)):
        ax_d.annotate(f"{d_risk[i]:+.3f}", xy=(x[i], d_risk[i]),
                      xytext=(8, dy), textcoords="offset points", fontsize=6.5,
                      color=_darken(C["risk"], 0.3))
    ax_d.set_ylabel("Proposed $-$ specialist-only")
    ax_d.set_ylim(-0.125, 0.024)
    _xaxis(ax_d)
    ax_d.text(len(betas) - 1.6, -0.055,
              "routing stage inert\nfor $\\beta \\geq 0.05$",
              fontsize=6.8, color=INK["muted"], ha="center", va="center",
              fontstyle="italic")
    _legend(ax_d, loc="lower left", fontsize=6.5)
    _panel(ax_d, "d", "Contribution of the routing stage")

    fig.suptitle("Baseline Comparison across the Router Risk Level "
                 f"($\\alpha$ = {alpha})",
                 fontsize=11.5, fontweight="bold", color=INK["text"])
    _savefig(fig, "fig9_router_risk_sweep")
    print("  Fig 9: router risk sweep")


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    _setup_style()
    with open(GRID_JSON) as f:
        data = json.load(f)
    print("Generating figure...")
    plot_sweep(data)
    print(f"\nSaved to {FIG_DIR}")


if __name__ == "__main__":
    main()
