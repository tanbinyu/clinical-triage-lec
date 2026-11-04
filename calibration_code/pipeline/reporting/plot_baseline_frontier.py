"""Risk-coverage frontier across the specialist risk grid (Methods 2.13).

Reads result/metrics/baseline_alpha_grid.json and draws what a single operating
point cannot show: how each system moves through the risk-coverage plane as
alpha varies, and where clinic-specific control breaks down even while pooled
system risk stays on target.

Two pairs of systems nearly coincide and are drawn so that both stay visible:
specialist-only tracks the proposed system because the router rejects only
0.014% of cases at beta = 0.1, and full automation sits on top of router-only
for the same reason.

Usage:
    cd calibration_code
    python3 -m pipeline.reporting.plot_baseline_frontier
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
                         "baseline_alpha_grid.json")
FIG_DIR = os.path.join(PROJECT_ROOT, "result", "metrics", "figures")

# Reuse the marker/colour vocabulary of Fig 4 so systems stay recognisable.
STYLE_KEY = {
    "full_automation": "Full Automation",
    "router_only": "Router-Only Rejection",
    "specialist_only": "Specialist-Only Rejection",
    "shared_threshold": "Shared Threshold",
    "oracle_clean": "Oracle-Clean Calibration",
    "proposed": "Proposed System (LEC)",
}

# Draw order for the alpha-dependent curves. The proposed system goes down as a
# solid heavy line and specialist-only follows as a dashed overlay, because the
# two coincide to within 2.5e-5 and a single opaque line would hide one of them.
CURVE_ORDER = ["shared_threshold", "oracle_clean", "proposed", "specialist_only"]
DASH_OVERLAY = {"specialist_only": (0, (4, 3))}

# Full automation and router-only differ by 1.2e-4 in pooled risk, so their
# reference lines are interleaved dashes rather than two opaque overlapping
# lines that would show only whichever was drawn last.
FIXED_DASH = {"full_automation": (0, (7, 3)), "router_only": (0, (2, 5))}


def _st(key):
    return BL_STYLE[STYLE_KEY[key]]


def _vals(rows, field):
    return [r[field] for r in rows]


def _draw_curve(ax, x, y, key, label=True):
    st = _st(key)
    hero = key == "proposed"
    ax.plot(x, y, marker=st["marker"], color=st["color"],
            linestyle=DASH_OVERLAY.get(key, "-"),
            linewidth=1.7 if hero else 1.1,
            markersize=6.5 if hero else 4.5,
            markeredgecolor=_darken(st["color"], 0.25), markeredgewidth=0.6,
            label=st["short"] if label else None,
            zorder=6 if key in DASH_OVERLAY else (5 if hero else 3))


def plot_frontier(data):
    grid, meta = data["grid"], data["meta"]
    alphas = meta["alpha_grid"]
    systems = [(s["key"], s["label"], s["alpha_dependent"])
               for s in meta["systems"]]
    fixed_keys = [k for k, _, dep in systems if not dep]

    fig, axes = plt.subplots(2, 2, figsize=(7.6, 6.4), constrained_layout=True)
    (ax_a, ax_b), (ax_c, ax_d) = axes

    # ── (a) risk-coverage frontier ──────────────────────────────────────
    for key in CURVE_ORDER:
        _draw_curve(ax_a, _vals(grid[key], "Cov_sys"),
                    _vals(grid[key], "mFDR_sys"), key)
    for key in fixed_keys:
        r, st = grid[key][0], _st(key)
        ax_a.scatter(r["Cov_sys"], r["mFDR_sys"], marker=st["marker"],
                     color=st["color"], s=95,
                     edgecolors=_darken(st["color"], 0.25), linewidth=0.6,
                     zorder=4)

    for r in grid["proposed"]:
        ax_a.annotate(f"$\\alpha$={r['alpha']:g}",
                      xy=(r["Cov_sys"], r["mFDR_sys"]),
                      xytext=(6, -8), textcoords="offset points", fontsize=6.5,
                      color=_darken(_st("proposed")["color"], 0.3))

    # The two unfiltered systems land within 1.4e-4 of each other, so a shared
    # leader line is clearer than two overlapping markers.
    ax_a.annotate("Full Auto. / Router-Only\n(no specialist rejection)",
                  xy=(grid["full_automation"][0]["Cov_sys"],
                      grid["full_automation"][0]["mFDR_sys"]),
                  xytext=(-24, -30), textcoords="offset points", fontsize=6.5,
                  color=INK["title"], ha="right", linespacing=1.35,
                  arrowprops=dict(arrowstyle="-", color=INK["hairline"],
                                  linewidth=0.5, shrinkB=4))
    ax_a.set_xlabel("System coverage (Cov$_{sys}$)")
    ax_a.set_ylabel("System risk (mFDR$_{sys}$)")
    ax_a.set_xlim(0.36, 1.06)
    ax_a.set_ylim(0, 0.235)
    ax_a.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    _legend(ax_a, loc="upper left", fontsize=6.5)
    _panel(ax_a, "a", "Risk-coverage frontier")

    # ── (b) empirical risk against target ───────────────────────────────
    ax_b.plot(alphas, alphas, ":", color=C["gray"], linewidth=1.0,
              label="y = $\\alpha$ (target)", zorder=2)
    for key in CURVE_ORDER:
        _draw_curve(ax_b, alphas, _vals(grid[key], "mFDR_sys"), key)
    for key in fixed_keys:
        st = _st(key)
        ax_b.axhline(grid[key][0]["mFDR_sys"], color=st["color"],
                     linestyle=FIXED_DASH[key], linewidth=1.2, alpha=0.9,
                     label=st["short"], zorder=2)
    ax_b.set_xlabel("Specialist risk level $\\alpha$")
    ax_b.set_ylabel("System risk (mFDR$_{sys}$)")
    ax_b.set_xlim(0.035, 0.215)
    ax_b.set_ylim(0, 0.235)
    ax_b.set_xticks(alphas)
    _legend(ax_b, loc="upper left", fontsize=6.2, ncol=2)
    _panel(ax_b, "b", "Pooled risk meets its target")

    # ── (c) clinics meeting the target ──────────────────────────────────
    # Categorical positions: the alpha grid is uneven in print width, and six
    # bars per group need even spacing to stay inside their own group.
    xpos = np.arange(len(alphas))
    width = 0.13
    offs = (np.arange(len(systems)) - (len(systems) - 1) / 2) * width
    for i, (key, _, _) in enumerate(systems):
        st = _st(key)
        ax_c.bar(xpos + offs[i], _vals(grid[key], "K_meet"), width=width,
                 color=st["color"], edgecolor=_darken(st["color"], 0.25),
                 linewidth=0.4, label=st["short"], zorder=3)
    ax_c.axhline(NUM_DISEASES, color=C["gray"], linestyle="--",
                 linewidth=0.6, alpha=0.6)
    ax_c.text(xpos[0] - 0.45, NUM_DISEASES + 0.15, f"all {NUM_DISEASES} clinics",
              fontsize=6.5, color=C["gray"], va="bottom")
    ax_c.set_xlabel("Specialist risk level $\\alpha$")
    ax_c.set_ylabel("Clinics meeting target ($K_{\\mathrm{meet}}$)")
    ax_c.set_xticks(xpos)
    ax_c.set_xticklabels([f"{a:g}" for a in alphas])
    ax_c.set_xlim(-0.6, len(alphas) - 0.4)
    ax_c.set_ylim(0, NUM_DISEASES + 2.6)
    ax_c.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    _legend(ax_c, loc="upper left", ncol=3, fontsize=6.0)
    _panel(ax_c, "c", "Clinic-specific control")

    # ── (d) worst-specialist risk ───────────────────────────────────────
    ax_d.plot(alphas, alphas, ":", color=C["gray"], linewidth=1.0,
              label="y = $\\alpha$ (target)", zorder=2)
    for key in CURVE_ORDER:
        _draw_curve(ax_d, alphas, _vals(grid[key], "mFDR_spec_worst"), key)
    for key in fixed_keys:
        st = _st(key)
        ax_d.axhline(grid[key][0]["mFDR_spec_worst"], color=st["color"],
                     linestyle=FIXED_DASH[key], linewidth=1.2, alpha=0.9,
                     zorder=2)
    ax_d.set_xlabel("Specialist risk level $\\alpha$")
    ax_d.set_ylabel("Worst-specialist risk")
    ax_d.set_xlim(0.035, 0.215)
    ax_d.set_ylim(0, 0.42)
    ax_d.set_xticks(alphas)
    _legend(ax_d, loc="upper left", fontsize=6.2, ncol=2)
    _panel(ax_d, "d", "Worst clinic exceeds the target")

    fig.suptitle("Baseline Comparison across the Specialist Risk Level "
                 f"($\\beta$ = {meta['beta']})",
                 fontsize=11.5, fontweight="bold", color=INK["text"])
    _savefig(fig, "fig8_risk_coverage_frontier")
    print("  Fig 8: risk-coverage frontier")


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    _setup_style()
    with open(GRID_JSON) as f:
        data = json.load(f)
    print("Generating figure...")
    plot_frontier(data)
    print(f"\nSaved to {FIG_DIR}")


if __name__ == "__main__":
    main()
