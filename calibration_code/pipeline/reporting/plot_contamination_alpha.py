"""Figure: routing contamination across the specialist risk grid (2.11).

Reads result/metrics/contamination_alpha_grid.json.

The claim under test is that specialist rejection prevents routing errors from
reaching AI-managed outcomes. Panel (a) shows how much contamination the stage
actually removes, panel (b) how often it catches a wrong-routed case, panel (c)
what the surviving errors are made of, and panel (d) normalises the residual so
clinics with routed rates an order of magnitude apart can be compared.

Aggregate curves pool counts across clinics rather than averaging clinic ratios,
matching the estimator used everywhere else.

Usage:
    cd calibration_code
    python3 -m pipeline.reporting.plot_contamination_alpha
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
    C, INK, _setup_style, _savefig, _darken, _panel, _legend,
)

DATA_JSON = os.path.join(PROJECT_ROOT, "result", "metrics",
                         "contamination_alpha_grid.json")
FIG_DIR = os.path.join(PROJECT_ROOT, "result", "metrics", "figures")

# D5 is the clinic whose specialist uncertainty barely separates wrong-routed
# from correctly routed cases, so it is the counterexample to the claim; D7 is
# the clearest positive case. Both are traced by name.
WEAK, STRONG = 5, 7
FAINT = dict(color=C["gray"], linewidth=0.8, alpha=0.4, zorder=2)


def _blocks(data):
    grid = data["grid"]
    alphas = data["meta"]["alpha_grid"]
    return alphas, [grid[f"{a:g}"]["per_disease"] for a in alphas]


def _clinic_series(blocks, k, field):
    return [b[str(k)][field] for b in blocks]


def _pooled(blocks, num_field, den_field):
    out = []
    for b in blocks:
        num = sum(b[str(k)][num_field] for k in range(NUM_DISEASES))
        den = sum(b[str(k)][den_field] for k in range(NUM_DISEASES))
        out.append(num / den if den else np.nan)
    return out


def _named(ax, x, y, k, color, label):
    ax.plot(x, y, "o-", color=color, linewidth=1.6, markersize=5,
            markeredgecolor=_darken(color, 0.3), markeredgewidth=0.6,
            zorder=6, label=label)


def plot_contamination(data):
    alphas, blocks = _blocks(data)
    x = np.arange(len(alphas))

    def _xaxis(ax):
        ax.set_xticks(x)
        ax.set_xticklabels([f"{a:g}" for a in alphas])
        ax.set_xlim(-0.2, len(alphas) - 0.8)
        ax.set_xlabel("Specialist risk level $\\alpha$")

    fig, axes = plt.subplots(2, 2, figsize=(7.6, 6.6), constrained_layout=True)
    (ax_a, ax_b), (ax_c, ax_d) = axes

    # ── (a) how much contamination the stage removes ────────────────────
    # Pooled only. Individual clinics span 0.012 to 0.154 in routed rate, so
    # overlaying them here would either clip or flatten the pooled curves; the
    # per-clinic story is panel (d), normalised so the clinics are comparable.
    p_route = _pooled(blocks, "total_wrong_routed", "total_routed")
    p_acc = _pooled(blocks, "N_route_err", "total_accepted")
    ax_a.fill_between(x, p_acc, p_route, color=C["route_err"], alpha=0.16,
                      zorder=3, label="removed by specialist rejection")
    ax_a.plot(x, p_route, "s--", color=C["rho_route"], linewidth=1.5,
              markersize=5, markeredgecolor=_darken(C["rho_route"], 0.3),
              markeredgewidth=0.6, zorder=5,
              label="$\\hat{\\rho}^{route}$ pooled (before rejection)")
    ax_a.plot(x, p_acc, "o-", color=C["rho_acc"], linewidth=1.7, markersize=5.5,
              markeredgecolor=_darken(C["rho_acc"], 0.35), markeredgewidth=0.6,
              zorder=5, label="$\\hat{\\rho}^{acc}$ pooled (after rejection)")
    for i in (0, len(x) - 1):
        ax_a.annotate(f"{p_route[i] - p_acc[i]:+.3f}",
                      xy=(x[i], (p_route[i] + p_acc[i]) / 2), xytext=(7, -2),
                      textcoords="offset points", fontsize=6.5,
                      color=_darken(C["route_err"], 0.35))
    ax_a.set_ylabel("Wrong-routed fraction of the stream")
    ax_a.set_ylim(0, 0.068)
    _xaxis(ax_a)
    _legend(ax_a, loc="lower left", fontsize=6.3)
    _panel(ax_a, "a", "Contamination removed")

    # ── (b) how often a wrong-routed case is caught ─────────────────────
    for k in range(NUM_DISEASES):
        if k in (WEAK, STRONG):
            continue
        ax_b.plot(x, _clinic_series(blocks, k, "Ref_wrong"), "-", **FAINT)
    _named(ax_b, x, _clinic_series(blocks, STRONG, "Ref_wrong"), STRONG,
           C["controlled"], f"D{STRONG} (strongest)")
    _named(ax_b, x, _clinic_series(blocks, WEAK, "Ref_wrong"), WEAK,
           C["risk"], f"D{WEAK} (weakest)")
    ax_b.set_ylabel("$\\widehat{\\mathrm{Ref}}^{wrong}$: wrong-routed rejected")
    ax_b.set_ylim(-0.04, 1.06)
    ax_b.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    _xaxis(ax_b)
    _legend(ax_b, loc="lower left", fontsize=6.5)
    _panel(ax_b, "b", "Wrong-routed cases caught")

    # ── (c) what the surviving accepted errors are made of ──────────────
    q_route = _pooled(blocks, "N_route_err", "N_err")
    q_diag = [1 - v for v in q_route]
    ax_c.stackplot(x, q_route, q_diag,
                   colors=[C["route_err"], C["diag_err"]], alpha=0.75,
                   edgecolor="none",
                   labels=["$\\hat{q}^{route\\text{-}err}$ routing",
                           "$\\hat{q}^{diag\\text{-}err}$ misdiagnosis"])
    ax_c.plot(x, q_route, "-", color=_darken(C["route_err"], 0.3),
              linewidth=1.2, zorder=5)
    for i, v in enumerate(q_route):
        ax_c.annotate(f"{v:.2f}", xy=(x[i], v), xytext=(0, 5),
                      textcoords="offset points", fontsize=6.5, ha="center",
                      color=_darken(C["route_err"], 0.4))
    ax_c.set_ylabel("Share of pooled accepted errors")
    ax_c.set_ylim(0, 1)
    ax_c.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    _xaxis(ax_c)
    _legend(ax_c, loc="center right", fontsize=6.5)
    _panel(ax_c, "c", "Source of surviving errors")

    # ── (d) residual contamination, normalised ──────────────────────────
    ax_d.axhline(1.0, color=C["gray"], linestyle="--", linewidth=0.9,
                 alpha=0.7)
    ax_d.text(-0.12, 1.03, "no reduction", fontsize=6.5, color=INK["muted"],
              va="bottom", fontstyle="italic")
    for k in range(NUM_DISEASES):
        if k in (WEAK, STRONG):
            continue
        ax_d.plot(x, _clinic_series(blocks, k, "rho_acc_over_route"), "-",
                  **FAINT)
    _named(ax_d, x, _clinic_series(blocks, STRONG, "rho_acc_over_route"),
           STRONG, C["controlled"], f"D{STRONG} (strongest)")
    _named(ax_d, x, _clinic_series(blocks, WEAK, "rho_acc_over_route"), WEAK,
           C["risk"], f"D{WEAK} (weakest)")
    ax_d.set_ylabel("$\\hat{\\rho}^{acc} / \\hat{\\rho}^{route}$ (residual share)")
    ax_d.set_ylim(-0.05, 1.22)
    _xaxis(ax_d)
    _legend(ax_d, loc="center left", fontsize=6.5)
    _panel(ax_d, "d", "Residual contamination share")

    fig.suptitle("Routing Contamination across the Specialist Risk Level "
                 f"($\\beta$ = {data['meta']['beta']})",
                 fontsize=11.5, fontweight="bold", color=INK["text"])
    _savefig(fig, "fig11_contamination_alpha")
    print("  Fig 11: contamination across the risk grid")


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    _setup_style()
    with open(DATA_JSON) as f:
        data = json.load(f)
    print("Generating figure...")
    plot_contamination(data)
    print(f"\nSaved to {FIG_DIR}")


if __name__ == "__main__":
    main()
