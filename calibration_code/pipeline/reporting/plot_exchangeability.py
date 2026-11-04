"""Figure for the routed-stream exchangeability check (Methods Eq. 28).

Reads result/metrics/exchangeability_check.json.

Panel (b) plots the observed KS statistic against its asymptotic null mean
rather than on its own, because the statistic grows as the routed stream
shrinks: the smallest clinic carries 84 cases per split against 1300 for the
largest, so a bare KS bar chart would show an apparent outlier that is nothing
but sample size.

Panel (d) is the consequential one. LEC forces the calibration-stream FDP below
alpha by construction, so the held-out excess is what decides whether the
target is actually met in deployment.

Usage:
    cd calibration_code
    python3 -m pipeline.reporting.plot_exchangeability
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pipeline.metrics.config import NUM_DISEASES
from pipeline.reporting.plot_metrics import (
    C, INK, _setup_style, _savefig, _darken, _panel, _legend,
)

DATA_JSON = os.path.join(PROJECT_ROOT, "result", "metrics",
                         "exchangeability_check.json")
FIG_DIR = os.path.join(PROJECT_ROOT, "result", "metrics", "figures")

ALPHA_MAIN = "0.2"
# D9 carries the smallest routed stream and the largest held-out excess, so it
# is traced through every panel.
FOCUS = 9

# Both scatter panels put several clinics within a label-width of each other, so
# the offsets are set per clinic instead of using one direction for all. In
# panel (c) D2 and D4 differ by 1e-4 on both axes, which is below the width of a
# marker, so they carry one shared label.
OFF_A = {0: (5, -9), 1: (5, -3), 2: (-16, 5), 3: (6, -2), 4: (5, -9),
         5: (-15, 4), 6: (6, -8), 7: (6, -3), 8: (6, -3), 9: (6, -3)}

# In panel (c) five clinics fall inside a region about 4 pt across, so they get
# one grouped label with a leader rather than five overlapping ones. Their exact
# values are in exchangeability_check.csv.
OFF_C = {0: (6, -3), 1: (6, -3), 5: (-16, 4), 6: (6, -8), 9: (-22, -3)}
CLUSTER_C = (2, 3, 4, 7, 8)


def _clinics(data):
    return [data["clinics"][str(k)] for k in range(NUM_DISEASES)]


def _identity(ax, lo, hi):
    ax.plot([lo, hi], [lo, hi], "-", color=C["gray"], linewidth=0.8,
            alpha=0.6, zorder=1, label="identity")


def plot_check(data):
    cl = _clinics(data)
    alphas = data["meta"]["alpha_grid"]

    fig, axes = plt.subplots(2, 2, figsize=(7.6, 6.6), constrained_layout=True)
    (ax_a, ax_b), (ax_c, ax_d) = axes

    # ── (a) layer 1: marginal correctness ───────────────────────────────
    xa = [r["base_err_cal"] for r in cl]
    ya = [r["base_err_test"] for r in cl]
    _identity(ax_a, 0.06, 0.38)
    # All points are estimates from the proposed LEC workflow; blue therefore
    # carries the primary result throughout the figure.  Vermillion is reserved
    # below for actual held-out target failures.
    ax_a.scatter(xa, ya, s=42, color=C["proposed"],
                 edgecolor=_darken(C["proposed"], 0.3), linewidth=0.6,
                 zorder=5)
    for k, (x0, y0) in enumerate(zip(xa, ya)):
        ax_a.annotate(f"D{k}", xy=(x0, y0), xytext=OFF_A[k],
                      textcoords="offset points", fontsize=6.5,
                      color=INK["muted"])
    worst = max(abs(r["delta_base_err"]) for r in cl)
    ax_a.text(0.06, 0.365,
              f"all ten clinics agree to within {worst:.4f}\n"
              f"across a range of {min(xa):.2f}$-${max(xa):.2f}",
              fontsize=6.8, color=INK["muted"], va="top", fontstyle="italic")
    ax_a.set_xlabel("Routed base error, calibration stream")
    ax_a.set_ylabel("Routed base error, held-out stream")
    ax_a.set_xlim(0.055, 0.385)
    ax_a.set_ylim(0.055, 0.385)
    ax_a.set_aspect("equal", adjustable="box")
    _panel(ax_a, "a", "Layer 1: correctness marginal")

    # ── (b) layer 2: score distribution against its null ────────────────
    order = np.argsort([r["mean_calibration_size"] for r in cl])
    xb = np.arange(NUM_DISEASES)
    obs = [cl[i]["ks_mean"] for i in order]
    null = [cl[i]["ks_null_mean"] for i in order]
    p95 = [cl[i]["ks_null_p95"] for i in order]
    ax_b.fill_between(xb, null, p95, color=C["gray"], alpha=0.13, zorder=1,
                      label="null mean to 95th pct")
    ax_b.plot(xb, null, "--", color=C["gray"], linewidth=1.0, zorder=2,
              label="null mean under Eq. (28)")
    ax_b.plot(xb, obs, "o-", color=C["proposed"], linewidth=1.4, markersize=5.5,
              markeredgecolor=_darken(C["proposed"], 0.3), markeredgewidth=0.6,
              zorder=5, label="observed")
    ax_b.set_xticks(xb)
    ax_b.set_xticklabels([f"D{i}\n{cl[i]['mean_calibration_size']:.0f}"
                          for i in order], fontsize=6.5)
    ax_b.set_xlabel("Clinic, ordered by routed calibration size")
    ax_b.set_ylabel("Two-sample KS statistic")
    ax_b.set_ylim(0, max(p95) * 1.18)
    ratios = [r["ks_ratio_to_null"] for r in cl]
    ax_b.text(0.5, max(p95) * 1.12,
              f"observed / null = {min(ratios):.2f}$-${max(ratios):.2f}; "
              "no clinic exceeds its null band",
              fontsize=6.8, color=INK["muted"], va="top", fontstyle="italic")
    _legend(ax_b, loc="lower left", fontsize=6.5)
    _panel(ax_b, "b", "Layer 2: score distribution")

    # ── (c) layer 3 at the operating point ──────────────────────────────
    a_main = float(ALPHA_MAIN)
    xc = [r["by_alpha"][ALPHA_MAIN]["FDP_cal"] for r in cl]
    yc = [r["by_alpha"][ALPHA_MAIN]["FDP_test"] for r in cl]
    _identity(ax_c, 0.07, 0.215)
    ax_c.axhline(a_main, color=C["gray"], linestyle=":", linewidth=0.9,
                 alpha=0.7)
    ax_c.axvline(a_main, color=C["gray"], linestyle=":", linewidth=0.9,
                 alpha=0.7)
    ax_c.text(0.072, a_main + 0.002, f"$\\alpha$ = {a_main}", fontsize=6.8,
              color=C["gray"], va="bottom")
    # The smallest stream is secondary context, not a failure at this operating
    # point; use the fixed secondary purple rather than the risk colour.
    cols = [C["proposed"] if k != FOCUS else C["route_err"]
            for k in range(NUM_DISEASES)]
    ax_c.scatter(xc, yc, s=[42 if k != FOCUS else 70
                            for k in range(NUM_DISEASES)],
                 color=cols, edgecolor=[_darken(c, 0.3) for c in cols],
                 linewidth=0.6, zorder=5)
    for k, (x0, y0) in enumerate(zip(xc, yc)):
        if k not in OFF_C:
            continue
        ax_c.annotate(f"D{k}", xy=(x0, y0), xytext=OFF_C[k],
                      textcoords="offset points", fontsize=6.5,
                      color=INK["muted"] if k != FOCUS else C["route_err"])
    cx = float(np.mean([xc[k] for k in CLUSTER_C]))
    cy = float(np.mean([yc[k] for k in CLUSTER_C]))
    ax_c.annotate("D2$-$D4, D7, D8", xy=(cx, cy), xytext=(0.150, 0.207),
                  fontsize=6.5, color=INK["muted"], ha="left", va="center",
                  arrowprops=dict(arrowstyle="-", linewidth=0.45,
                                  color=INK["hairline"],
                                  shrinkA=1, shrinkB=5))
    ax_c.set_xlabel("FDP at $\\hat{\\lambda}_k$, calibration stream")
    ax_c.set_ylabel("FDP at $\\hat{\\lambda}_k$, held-out stream")
    ax_c.set_xlim(0.07, 0.215)
    ax_c.set_ylim(0.07, 0.215)
    ax_c.set_aspect("equal", adjustable="box")
    _panel(ax_c, "c", f"Layer 3 at $\\alpha$ = {a_main}")

    # ── (d) layer 3 across the risk grid ────────────────────────────────
    xd = np.arange(len(alphas))
    ax_d.axhline(0, color=C["gray"], linewidth=0.8, alpha=0.7)
    risk_legend_drawn = False
    for k in range(NUM_DISEASES):
        d = [cl[k]["by_alpha"][f"{a:g}"]["delta_FDP"] for a in alphas]
        focus = k == FOCUS
        ax_d.plot(xd, d, "-o" if focus else "-",
                  color=C["route_err"] if focus else C["light"],
                  linewidth=1.8 if focus else 0.85,
                  markersize=5 if focus else 0,
                  markeredgecolor=_darken(C["route_err"], 0.3),
                  markeredgewidth=0.6,
                  alpha=1.0 if focus else 0.9,
                  zorder=6 if focus else 3,
                  label=f"D{FOCUS} (smallest stream)" if focus else None)
        # Red is used exclusively where the held-out stream breaches its
        # prespecified risk target; this applies to any clinic, not only D9.
        failed = [i for i, a in enumerate(alphas)
                  if not cl[k]["by_alpha"][f"{a:g}"]["FDP_test_le_alpha"]]
        if failed:
            ax_d.scatter([xd[i] for i in failed], [d[i] for i in failed],
                         s=34, color=C["risk"],
                         edgecolor=_darken(C["risk"], 0.3), linewidth=0.5,
                         zorder=8,
                         label=("held-out target exceeded"
                                if not risk_legend_drawn else None))
            risk_legend_drawn = True
    mean_d = [np.mean([cl[k]["by_alpha"][f"{a:g}"]["delta_FDP"]
                       for k in range(NUM_DISEASES)]) for a in alphas]
    ax_d.plot(xd, mean_d, "s--", color=C["proposed"], linewidth=1.5, markersize=5,
              markeredgecolor=_darken(C["proposed"], 0.3), markeredgewidth=0.6,
              zorder=7, label="mean over clinics")
    # How many clinics actually meet the target on held-out data.
    for i, a in enumerate(alphas):
        n_ok = sum(cl[k]["by_alpha"][f"{a:g}"]["FDP_test_le_alpha"]
                   for k in range(NUM_DISEASES))
        ax_d.annotate(f"{n_ok}/10", xy=(xd[i], 0), xytext=(0, -14),
                      textcoords="offset points", fontsize=6.5, ha="center",
                      color=_darken(C["coverage"], 0.2) if n_ok == 10
                      else _darken(C["risk"], 0.2))
    ax_d.text(0.06, 0.128,
              "held-out FDP $\\leq \\alpha$ (count below axis)",
              fontsize=6.5, color=INK["muted"], fontstyle="italic")
    ax_d.set_xticks(xd)
    ax_d.set_xticklabels([f"{a:g}" for a in alphas])
    ax_d.set_xlim(-0.2, len(alphas) - 0.8)
    ax_d.set_ylim(-0.022, 0.15)
    ax_d.set_xlabel("Specialist risk level $\\alpha$")
    ax_d.set_ylabel("Held-out excess FDP$_{test}$ $-$ FDP$_{cal}$")
    _legend(ax_d, loc="upper right", fontsize=6.5)
    _panel(ax_d, "d", "Layer 3 across the risk grid")

    fig.suptitle("Empirical Check of Routed-Stream Exchangeability "
                 "(Methods Eq. 28)"
                 f"   ($\\beta$ = {data['meta']['beta']})",
                 fontsize=11.5, fontweight="bold", color=INK["text"])
    _savefig(fig, "fig10_exchangeability")
    print("  Fig 10: routed-stream exchangeability check")


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    _setup_style()
    with open(DATA_JSON) as f:
        data = json.load(f)
    print("Generating figure...")
    plot_check(data)
    print(f"\nSaved to {FIG_DIR}")


if __name__ == "__main__":
    main()
