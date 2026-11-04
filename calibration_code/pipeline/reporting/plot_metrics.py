"""Publication-quality figures for the Nature paper.

The plotting layer uses a restrained, colour-blind-safe visual language:
blue identifies the proposed system, orange flags risk, and greys carry all
supporting information.  Dense per-disease spaghetti plots are replaced by
summaries, muted individual trajectories, or compact heatmaps.  This module
only renders metrics; it never changes a result JSON or its calculations.

Usage:
    cd calibration_code
    python -m pipeline.reporting.plot_metrics
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap, to_hex
from matplotlib.patches import FancyArrowPatch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pipeline.metrics.config import NUM_DISEASES

METRICS_JSON = os.path.join(PROJECT_ROOT, "result", "metrics", "metrics.json")
FIG_DIR = os.path.join(PROJECT_ROOT, "result", "metrics", "figures")

# ════════════════════════════════════════════════════════════════════
#  Disease name mappings
# ════════════════════════════════════════════════════════════════════

SHORT_NAMES = {
    "pathology_lung_cancer": "Lung Cancer\n(Pathology)",
    "pathology_breast_cancer": "Breast Cancer\n(Pathology)",
    "ct_pneumonia": "Pneumonia\n(CT)",
    "ct_lung_nodule": "Lung Nodule\n(CT)",
    "ct_congenital_heart": "Congenital Heart\n(CT)",
    "xray_lung": "Lung Disease\n(X-ray)",
    "mammography_breast_lesion_benign_malignant":
        "Breast Lesion\n(X-ray)",
    "us_breast_cancer": "Breast Cancer\n(US)",
    "us_lung_disease": "Lung Disease\n(Lung US)",
    "mri_cardiomyopathy": "Cardiomyopathy\n(MRI)",
}

SHORT_NAMES_ONELINE = {
    "pathology_lung_cancer": "D0  Lung Ca. (Path)",
    "pathology_breast_cancer": "D1  Breast Ca. (Path)",
    "ct_pneumonia": "D2  Pneumonia (CT)",
    "ct_lung_nodule": "D3  Lung Nod. (CT)",
    "ct_congenital_heart": "D4  CHD (CT)",
    "xray_lung": "D5  Lung (X-ray)",
    "mammography_breast_lesion_benign_malignant":
        "D6  Breast Lesion (X-ray)",
    "us_breast_cancer": "D7  Breast Ca. (US)",
    "us_lung_disease": "D8  Lung Disease (Lung US)",
    "mri_cardiomyopathy": "D9  CMP (MRI)",
}

# ════════════════════════════════════════════════════════════════════
#  Colour constants — every hex in this module lives in this block
# ════════════════════════════════════════════════════════════════════

C = {
    # Okabe-Ito-derived palette: each hue has one consistent semantic role.
    "risk":       "#D55E00",   # risk / target exceedance (vermillion)
    "risk2":      "#E69F00",   # secondary risk component (orange)
    "coverage":   "#0072B2",   # AI-managed coverage (blue)
    "controlled": "#009E73",   # target attained (bluish green)
    "unctl":      "#D55E00",   # target not attained
    "route_err":  "#CC79A7",   # routing-specific component (purple)
    "diag_err":   "#0072B2",   # diagnostic component (blue)
    "rho_route":  "#6A51A3",   # pre-rejection contamination (violet)
    "rho_acc":    "#7A7A7A",   # post-rejection contamination (neutral)
    "proposed":   "#0072B2",   # proposed LEC: the sole hero colour
    "gray":       "#4D4D4D",   # charcoal
    "light":      "#D9D9D9",   # support grey
    "gold":       "#8C8C8C",   # operating-point rule (neutral, not a data hue)
}

# Neutral ink ladder. Kept separate from C{} so that the 13 semantic keys in
# C{} stay untouched, while still centralising every hex in one block.
INK = {
    "text":     "#1A1A1A",   # densest ink: panel letters, heat-map values
    "title":    "#333333",   # panel titles, axis labels, in-panel numbers
    "axis":     "#4A4A4A",   # spines and tick marks
    "muted":    "#666666",   # subtitles and secondary notes
    "hairline": "#999999",   # annotation leader lines
    "grid":     "#EEEEEE",   # whisper-light horizontal rules
}

# Baseline styling
BL_STYLE = {
    "Full Automation":            {"color": "#8C8C8C", "marker": "X",  "short": "Full automation"},
    "Router-Only Rejection":      {"color": "#767676", "marker": "D",  "short": "Router-only"},
    "Specialist-Only Rejection":  {"color": "#595959", "marker": "v",  "short": "Specialist-only"},
    "Shared Threshold":           {"color": "#A6A6A6", "marker": "s",  "short": "Shared threshold"},
    "Oracle-Clean Calibration":   {"color": "#6A51A3", "marker": "P",  "short": "Oracle-clean"},
    "Proposed System (LEC)":      {"color": "#0072B2", "marker": "*",  "short": "LEC (proposed)"},
}

# Colour-blind-safe diverging utility scale with a neutral midpoint.
UTILITY_CMAP = LinearSegmentedColormap.from_list(
    "utility",
    ["#CA562C", "#F4A582", "#F7F7F7", "#A6CEE3", "#2166AC"],
    N=256,
)

# Per-disease palette for multi-clinic panels
# Individual disease trajectories are deliberately neutral; showing ten hues
# creates a legend-matching task rather than an interpretable scientific plot.
DISEASE_COLORS = ["#B8B8B8"] * NUM_DISEASES

# Leader-line offsets for the baseline scatter. Three systems sit at
# (Cov 1.000, mFDR ~0.195) and two more at (~0.953, ~0.169), so the offsets
# fan the two-line labels apart vertically.
BL_ANNOT = {
    "Full Automation":           {"dx": -16, "dy": 34,  "ha": "right"},
    "Router-Only Rejection":     {"dx": 18,  "dy": 4,   "ha": "left"},
    "Shared Threshold":          {"dx": 18,  "dy": -26, "ha": "left"},
    "Specialist-Only Rejection": {"dx": -16, "dy": 26,  "ha": "right"},
    "Oracle-Clean Calibration":  {"dx": 16,  "dy": 22,  "ha": "left"},
    "Proposed System (LEC)":     {"dx": -16, "dy": -26, "ha": "right"},
}

# ── Contrast layer for Fig 4 ─────────────────────────────────────────────
# At the operating point beta = 0.1 the calibrated router threshold refers
# only 0.014% of cases, so the proposed system sits exactly on specialist-
# only rejection and the routing stage is invisible in the baseline scatter;
# the contrast layer replays the router-dependent systems at beta = 0.01,
# where the constraint binds and every router-dependent system moves
# down-left. When the operating point already is beta = 0.01 the contrast
# layer mirrors to the relaxed beta = 0.1 replay, where the proposed system
# degenerates onto specialist-only rejection — the same fact told in reverse.
BETA_STRICT = 0.01
BETA_RELAXED = 0.10
BETA_GRID_JSON = os.path.join(PROJECT_ROOT, "result", "metrics",
                              "baseline_beta_grid.json")

# System keys in baseline_beta_grid.json -> display names used by BL_STYLE.
GRID_KEY_TO_NAME = {
    "full_automation": "Full Automation",
    "router_only": "Router-Only Rejection",
    "specialist_only": "Specialist-Only Rejection",
    "shared_threshold": "Shared Threshold",
    "oracle_clean": "Oracle-Clean Calibration",
    "proposed": "Proposed System (LEC)",
}

# Router-only and shared-threshold coincide to display precision at
# beta = 0.01, so they share one label anchored at the router-only point.
# The strict cluster is a tight staircase (four points within ~0.015 Cov),
# so these labels are pinned in data coordinates: router/shared sits in the
# free rectangle left of the specialist-only label, while proposed and
# oracle stack in two rows along the bottom strip, clear of the migration
# arrows and the operating-point labels.
BL_ANNOT_STRICT = {
    "Router-Only Rejection":     {"xy": (0.830, 0.1640), "ha": "left", "va": "center"},
    "Oracle-Clean Calibration":  {"xy": (0.883, 0.1435), "ha": "left", "va": "center"},
    "Proposed System (LEC)":     {"xy": (0.828, 0.1370), "ha": "left", "va": "top"},
}

# Layout when the operating point itself is the strict one (beta = 0.01):
# the six operating-point markers then cluster on the left (Cov 0.84–0.96)
# while the four relaxed beta = 0.1 ghosts sit at Cov 0.95–1.00. Operating-
# point labels are pinned in data coordinates around the left cluster;
# Full Automation stays at Cov 1.000 and keeps an offset label.
BL_ANNOT_OP_STRICT = {
    "Full Automation":           {"dx": -16, "dy": 34,  "ha": "right"},
    "Router-Only Rejection":     {"xy": (0.830, 0.1665), "ha": "left", "va": "center"},
    "Specialist-Only Rejection": {"xy": (0.900, 0.1755), "ha": "left", "va": "center"},
    "Oracle-Clean Calibration":  {"xy": (0.883, 0.1435), "ha": "left", "va": "center"},
    "Proposed System (LEC)":     {"xy": (0.828, 0.1370), "ha": "left", "va": "top"},
    # Shared threshold coincides with router-only here; it carries no label
    # of its own and is covered by the shared "Router-Only / Shared" text.
    "Shared Threshold":          None,
}

# Relaxed ghosts (beta = 0.1, open markers): router-only and shared both sit
# at Cov 1.000 next to Full Automation, and the relaxed proposed ghost sits
# exactly on the specialist-only operating point — the coincidence is the
# story, so that label says so explicitly.
BL_ANNOT_CONTRAST_RELAXED = {
    "Router-Only Rejection":     {"xy": (0.870, 0.1965), "ha": "left", "va": "bottom"},
    "Oracle-Clean Calibration":  {"xy": (0.905, 0.1850), "ha": "left", "va": "center"},
    "Proposed System (LEC)":     {"xy": (0.958, 0.1590), "ha": "center", "va": "top"},
}


# ════════════════════════════════════════════════════════════════════
#  Style setup
# ════════════════════════════════════════════════════════════════════

def _setup_style():
    """Set publication-quality matplotlib defaults."""
    plt.rcParams.update({
        # Standard editable sans serif face for Nature production.
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "mathtext.fontset": "stixsans",
        "font.size": 7,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 6.5,
        # Figure
        "figure.dpi": 300,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.08,
        # Keep all text editable in the vector submission files.
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        # Lines & markers — thin and precise beats thick and loud
        "lines.linewidth": 1.0,
        "lines.markersize": 4.0,
        "lines.markeredgewidth": 0.6,
        # Axes — no grid by default; figures opt in where values are read off
        "axes.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "axes.axisbelow": True,
        # Ink ladder: text darkest, spines lighter, ticks lightest
        "axes.labelcolor": INK["title"],
        "axes.edgecolor": INK["axis"],
        "xtick.color": INK["axis"],
        "ytick.color": INK["axis"],
        "text.color": INK["text"],
        # Ticks
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.direction": "out",
        "ytick.direction": "out",
    })


# ════════════════════════════════════════════════════════════════════
#  Helpers
# ════════════════════════════════════════════════════════════════════

def _load_data(metrics_json=None):
    with open(metrics_json or METRICS_JSON) as f:
        return json.load(f)

def _short(name):
    return SHORT_NAMES.get(name, name)

def _short1(name):
    return SHORT_NAMES_ONELINE.get(name, name)

def _val(v):
    if isinstance(v, dict):
        return v.get("pooled")
    return v

def _savefig(fig, name):
    # SVG travels alongside PDF because journal production asks for an editable
    # vector form, and PNG for drafts.
    for ext in ("png", "pdf", "svg"):
        fig.savefig(os.path.join(FIG_DIR, f"{name}.{ext}"))
    plt.close(fig)


def _darken(hex_color, ratio=0.20):
    """Return a hex color darkened by `ratio` (0-1) for marker edges."""
    c = hex_color.lstrip("#")
    r, g, b = int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
    r = max(0, int(r * (1 - ratio)))
    g = max(0, int(g * (1 - ratio)))
    b = max(0, int(b * (1 - ratio)))
    return f"#{r:02X}{g:02X}{b:02X}"


def _cell_text_color(bg_hex):
    """Return black or white text based on bg brightness."""
    c = bg_hex.lstrip("#")
    r, g, b = int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
    return "white" if (0.299 * r + 0.587 * g + 0.114 * b) < 128 else INK["text"]


def _line(ax, x, y, color, marker="o", label=None, linewidth=1.2,
          markersize=4.5, linestyle="-", zorder=3):
    """Draw a metric line with same-hue darkened marker edges."""
    ax.plot(x, y, marker=marker, linestyle=linestyle, color=color,
            linewidth=linewidth, markersize=markersize,
            markeredgecolor=_darken(color, 0.25),
            markeredgewidth=0.6,
            label=label, zorder=zorder)


def _panel(ax, letter, title):
    """Panel letter tucked against the axes, with a left-aligned title."""
    ax.text(-0.12, 1.04, letter, transform=ax.transAxes,
            fontsize=10, fontweight="bold",
            va="bottom", ha="right", color=INK["text"])
    if title:
        ax.set_title(title, fontsize=8, fontweight="bold",
                     pad=6, loc="left", color=INK["title"])


def _legend(ax, loc="best", ncol=1, fontsize=7.5, **kw):
    """Frameless legend with uniform spacing (Cell house style)."""
    return ax.legend(loc=loc, ncol=ncol, fontsize=fontsize, frameon=False,
                     handlelength=1.5, handletextpad=0.5, columnspacing=1.0,
                     borderaxespad=0.3, **kw)


def _soft_ygrid(ax):
    """Keep a compatibility hook while following Nature's no-grid guidance."""
    ax.grid(False)


def _op_line(ax, xval, label=None, at="top"):
    """Neutral operating-point rule, optionally annotated at one end.

    Call after the y-limits are final so the label lands inside the axes.
    `at="bottom"` keeps the note clear of an upper-corner legend.
    """
    ax.axvline(xval, color=C["gold"], linewidth=0.8, alpha=0.9, zorder=1)
    if label:
        lo, hi = ax.get_ylim()
        anchor, dy, va = (lo, 8, "bottom") if at == "bottom" else (hi, -8, "top")
        ax.annotate(label, xy=(xval, anchor),
                    xytext=(0, dy), textcoords="offset points",
                    fontsize=6.3, color=INK["muted"], ha="center", va=va)


def _end_label(ax, x, y, color, fmt="{:.3f}"):
    """Print the value of a curve's right-most finite point next to it."""
    xs = np.asarray(x, dtype=float)
    ys = np.asarray(y, dtype=float)
    ok = np.isfinite(xs) & np.isfinite(ys)
    if not ok.any():
        return
    i = np.flatnonzero(ok)[-1]
    ax.annotate(fmt.format(ys[i]), xy=(xs[i], ys[i]),
                xytext=(4, 0), textcoords="offset points",
                fontsize=6.5, va="center", ha="left",
                color=_darken(color, 0.30))


def _clinic_stat(rows, container, field):
    """Collect a per-clinic field across sweep rows -> (matrix, mean, min, max).

    Missing/not-estimable entries become NaN so lines break instead of
    silently dropping to zero.
    """
    mat = np.full((NUM_DISEASES, len(rows)), np.nan)
    for j, r in enumerate(rows):
        per = r.get(container) or {}
        for k in range(NUM_DISEASES):
            rec = per.get(str(k))
            if rec is None:
                continue
            v = rec.get(field)
            if v is not None:
                mat[k, j] = v
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(mat, axis=0)
        lo = np.nanmin(mat, axis=0)
        hi = np.nanmax(mat, axis=0)
    return mat, mean, lo, hi


# ════════════════════════════════════════════════════════════════════
#  Fig 1: Per-Disease Specialist Risk — Forest Plot
# ════════════════════════════════════════════════════════════════════

def plot_disease_forest(data):
    rc = data["risk_control"]
    alpha = data["meta"]["alpha"]
    beta = data["meta"]["beta"]
    names = data["meta"]["disease_names"]

    rows = []
    for k in range(NUM_DISEASES):
        s = rc["specialist"].get(str(k))
        if s is None or s["pooled_mfdr"] is None:
            continue
        m = s["pooled_mfdr"]
        rows.append({
            "mfdr": m,
            "low": s.get("ci_low") or m,
            "high": s.get("ci_high") or m,
            "label": _short1(names[str(k)]),
        })

    # Ordering by effect size makes the safety margin immediately readable.
    rows.sort(key=lambda row: row["mfdr"], reverse=True)
    n = len(rows)
    n_controlled = sum(row["mfdr"] <= alpha for row in rows)
    y = np.arange(n)

    fig, ax = plt.subplots(figsize=(5.1, 4.15), constrained_layout=True)
    ax.axvspan(0, alpha, color=C["proposed"], alpha=0.035, zorder=0)

    for i, row in enumerate(rows):
        color = C["proposed"] if row["mfdr"] <= alpha else C["risk"]
        ax.plot([row["low"], row["high"]], [y[i], y[i]], color=color,
                linewidth=1.15, solid_capstyle="round", zorder=3)
        ax.plot(row["mfdr"], y[i], "o", color=color, markersize=4.3,
                markeredgecolor=_darken(color, 0.25), markeredgewidth=0.55,
                zorder=4)
        ax.text(row["mfdr"] + 0.006, y[i], f"{row['mfdr']:.3f}", fontsize=6.4,
                va="center", ha="left", color=INK["text"])

    ax.axvline(alpha, color=C["gray"], linestyle="--", linewidth=1.0, zorder=3)
    ax.annotate(f"$\\alpha$ = {alpha}", xy=(alpha, 1.0),
                xycoords=("data", "axes fraction"),
                xytext=(0, 4), textcoords="offset points",
                fontsize=6.5, color=INK["muted"], ha="center", va="bottom")

    ax.set_yticks(y)
    ax.set_yticklabels([row["label"] for row in rows], fontsize=6.6)
    ax.set_xlabel("mFDR (pooled)")
    ax.invert_yaxis()
    ax.set_xlim(0, 0.245)

    ax.set_title("Per-disease specialist risk", fontsize=8.3,
                 fontweight="bold", loc="left", pad=14, color=INK["text"])
    ax.text(0, 1.015,
            f"{n_controlled}/{NUM_DISEASES} diseases controlled at "
            f"$\\alpha$ = {alpha}, $\\beta$ = {beta}",
            transform=ax.transAxes, fontsize=6.3, color=INK["muted"],
            va="bottom", ha="left")

    _savefig(fig, "fig1_disease_forest")
    print("  Fig 1: disease forest plot")


# ════════════════════════════════════════════════════════════════════
#  Fig 2: β Sweep — Router Risk Sensitivity
# ════════════════════════════════════════════════════════════════════

def plot_beta_sweep(data):
    """Fig 2: sensitivity to the router risk level beta.

    Reports the full set required by Methods (Sensitivity to Routing Risk
    Level): router coverage, routed contamination, specialist coverage,
    end-to-end accepted-case error and human referral across the beta grid.
    """
    bs = [row for row in data["beta_sweep"] if row.get("status") != "no_data"]
    if not bs:
        raise ValueError("beta sweep has no complete rows; generate Step 3 results first")
    if len(bs) != len(data["beta_sweep"]):
        print("  Fig 2 warning: skipped beta-sweep rows with missing prerequisites")
    alpha = data["meta"]["alpha"]
    op_beta = data["meta"]["beta"]

    betas = [r["beta"] for r in bs]
    mfdr_r = [r["mFDR_R"] for r in bs]
    mfdr_sys = [r["mFDR_sys"] for r in bs]
    ci_lo = [r.get("ci_low_mfdr_sys", r["mFDR_sys"]) for r in bs]
    ci_hi = [r.get("ci_high_mfdr_sys", r["mFDR_sys"]) for r in bs]
    cov_r = [r["Cov_R"] for r in bs]
    cov_sys = [r["Cov_sys"] for r in bs]
    cov_spec = [r["Cov_sys"] / r["Cov_R"] if r["Cov_R"] else np.nan for r in bs]
    ref_sys = [r["Ref_sys"] for r in bs]
    ref_r = [1.0 - r["Cov_R"] for r in bs]
    ref_spec = [r["Cov_R"] - r["Cov_sys"] for r in bs]

    _, rho_rt, rho_rt_lo, rho_rt_hi = _clinic_stat(bs, "contamination", "rho_route")
    _, rho_ac, _, _ = _clinic_stat(bs, "contamination", "rho_acc")

    fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.15), constrained_layout=True)
    (ax_a, ax_b), (ax_c, ax_d) = axes
    xlim = (0, 0.215)

    # (a) risk: two data curves plus the constraint and beta identity.
    ax_a.fill_between(betas, ci_lo, ci_hi, color=C["risk"], alpha=0.12, zorder=1)
    ax_a.plot(betas, betas, ":", color=INK["muted"], linewidth=0.9,
              label="$y=\\beta$", zorder=2)
    _line(ax_a, betas, mfdr_r, C["risk2"], "s", "Router risk")
    _line(ax_a, betas, mfdr_sys, C["risk"], "o", "System risk")
    ax_a.axhline(alpha, color=INK["muted"], linestyle="--", linewidth=0.8)
    ax_a.annotate(f"target $\\alpha={alpha}$", xy=(xlim[1], alpha),
                  xytext=(-2, 3), textcoords="offset points", fontsize=6.1,
                  color=INK["muted"], va="bottom", ha="right")
    ax_a.set_ylabel("Accepted-case error rate")
    ax_a.set_ylim(0, 0.28)
    ax_a.set_xlim(*xlim)
    _op_line(ax_a, op_beta, f"Operating $\\beta={op_beta:g}$", at="bottom")
    _legend(ax_a, loc="upper left", fontsize=6.1)
    _panel(ax_a, "a", "Risk control")

    # (b) coverage: system-level curves only; conditional coverage is moved
    # to the per-disease alpha figure rather than creating a third line here.
    _line(ax_b, betas, cov_r, C["gray"], "^", "Routed")
    _line(ax_b, betas, cov_sys, C["coverage"], "o", "AI-managed")
    ax_b.set_ylabel("Coverage")
    ax_b.set_ylim(0.82, 1.012)
    ax_b.set_xlim(*xlim)
    _op_line(ax_b, op_beta)
    _legend(ax_b, loc="lower right", fontsize=6.1)
    _panel(ax_b, "b", "Coverage")

    # (c) referral is intentionally decomposed using line style as well as
    # hue, so the panel remains interpretable in greyscale.
    _line(ax_c, betas, ref_sys, C["risk"], "o", "Total referral")
    _line(ax_c, betas, ref_r, C["gray"], "s", "Router referral")
    _line(ax_c, betas, ref_spec, C["route_err"], "^", "Specialist referral")
    ax_c.set_xlabel("Router risk level $\\beta$")
    ax_c.set_ylabel("Human referral rate")
    ax_c.set_ylim(-0.006, 0.175)
    ax_c.set_xlim(*xlim)
    _op_line(ax_c, op_beta)
    _legend(ax_c, loc="upper right", fontsize=6.0)
    _panel(ax_c, "c", "Human referral")

    # (d) one summary and one uncertainty envelope replace per-disease lines.
    ax_d.fill_between(betas, rho_rt_lo, rho_rt_hi, color=C["rho_route"],
                      alpha=0.12, zorder=1, label="Disease range")
    _line(ax_d, betas, rho_rt, C["rho_route"], "o", "Before rejection")
    _line(ax_d, betas, rho_ac, C["rho_acc"], "s", "After rejection")
    ax_d.set_xlabel("Router risk level $\\beta$")
    ax_d.set_ylabel("Routed contamination")
    ax_d.set_ylim(0, 0.185)
    ax_d.set_xlim(*xlim)
    _op_line(ax_d, op_beta)
    _legend(ax_d, loc="upper left", fontsize=6.0)
    _panel(ax_d, "d", "Routing contamination")

    fig.suptitle(f"Sensitivity to router risk level $\\beta$  ($\\alpha={alpha}$)",
                 fontsize=8.7, fontweight="bold", color=INK["text"])
    _savefig(fig, "fig2_beta_sweep")
    print("  Fig 2: beta sweep (risk / coverage / referral / contamination)")


# ════════════════════════════════════════════════════════════════════
#  Fig 3: α Sweep — Specialist Risk Sensitivity
# ════════════════════════════════════════════════════════════════════

def plot_alpha_sweep(data):
    """Fig 3: sensitivity to the specialist risk level alpha.

    Reports the full set required by Methods (Sensitivity to Specialist Risk
    Level): per-specialist accepted-case error, specialist coverage,
    AI-managed rate, end-to-end accepted-case error and human referral.
    """
    asw = data["alpha_sweep"]
    names = data["meta"]["disease_names"]
    op_alpha = data["meta"]["alpha"]
    beta = data["meta"]["beta"]

    alphas = [r["alpha"] for r in asw]
    mfdr_sys = [r.get("mFDR_sys") for r in asw]
    macro = [r.get("mFDR_spec_macro") for r in asw]
    worst = [r.get("mFDR_spec_worst") for r in asw]
    cov_sys = [r.get("Cov_sys") for r in asw]
    ref_sys = [r.get("Ref_sys") for r in asw]
    n_ctrl = [r.get("n_diseases_controlled", 0) for r in asw]
    ci_lo = [r.get("ci_low_mfdr_sys") or r.get("mFDR_sys") for r in asw]
    ci_hi = [r.get("ci_high_mfdr_sys") or r.get("mFDR_sys") for r in asw]

    mfdr_k, _, _, _ = _clinic_stat(asw, "diseases", "mFDR_k")
    cov_k, _, _, _ = _clinic_stat(asw, "per_disease", "Cov_k")

    fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.15), constrained_layout=True)
    (ax_a, ax_b), (ax_c, ax_d) = axes
    alpha_ticks = [0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8]
    log_xlim = (0.009, 0.9)

    def _alpha_log_axis(ax, label=False):
        """Use a labelled log axis: valid for positive risk levels spanning 80x."""
        ax.set_xscale("log")
        ax.set_xlim(*log_xlim)
        ax.set_xticks(alpha_ticks)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda value, _: f"{value:g}"))
        ax.xaxis.set_minor_locator(mticker.NullLocator())
        if label:
            ax.set_xlabel("Specialist risk level $\\alpha$ (log scale)")

    # (a) system-level and specialist summary risk.
    ax_a.fill_between(alphas, ci_lo, ci_hi, color=C["risk"], alpha=0.12, zorder=1)
    ax_a.plot(alphas, alphas, ":", color=INK["muted"], linewidth=0.9,
              label="$y=\\alpha$", zorder=2)
    _line(ax_a, alphas, mfdr_sys, C["risk"], "o", "System")
    _line(ax_a, alphas, macro, C["gray"], "s", "Specialist mean")
    _line(ax_a, alphas, worst, C["rho_route"], "^", "Specialist worst")
    ax_a.set_ylabel("Accepted-case error rate")
    ax_a.set_ylim(0, 0.86)
    _alpha_log_axis(ax_a)
    _op_line(ax_a, op_alpha, f"Operating $\\alpha={op_alpha:g}$", at="bottom")
    _legend(ax_a, loc="upper left", fontsize=6.0)
    _panel(ax_a, "a", "Risk control")

    # (b) replaces ten coloured curves with a target-normalised heatmap.
    # Values above 1 are constraint violations; values below 1 are controlled.
    ratio = mfdr_k / np.asarray(alphas, dtype=float)[None, :]
    finite_ratio = ratio[np.isfinite(ratio)]
    vmax = max(1.05, float(np.nanmax(finite_ratio))) if finite_ratio.size else 1.05
    risk_cmap = LinearSegmentedColormap.from_list(
        "risk_ratio", ["#2166AC", "#F7F7F7", "#D55E00"], N=256)
    log_centres = np.log(np.asarray(alphas, dtype=float))
    log_edges = np.empty(len(alphas) + 1)
    log_edges[1:-1] = (log_centres[:-1] + log_centres[1:]) / 2
    log_edges[0] = log_centres[0] - (log_centres[1] - log_centres[0]) / 2
    log_edges[-1] = log_centres[-1] + (log_centres[-1] - log_centres[-2]) / 2
    im = ax_b.pcolormesh(np.exp(log_edges), np.arange(NUM_DISEASES + 1) - 0.5,
                         np.ma.masked_invalid(ratio), cmap=risk_cmap,
                         norm=TwoSlopeNorm(vmin=0, vcenter=1, vmax=vmax),
                         shading="flat")
    _alpha_log_axis(ax_b, label=True)
    ax_b.axvline(op_alpha, color=INK["text"], linewidth=0.75)
    ax_b.set_yticks(range(NUM_DISEASES))
    ax_b.set_yticklabels([f"D{k}" for k in range(NUM_DISEASES)])
    ax_b.set_ylabel("Specialist")
    ax_b.invert_yaxis()
    cbar = fig.colorbar(im, ax=ax_b, fraction=0.047, pad=0.025)
    cbar.set_label("Risk / target", fontsize=6.4)
    cbar.ax.tick_params(labelsize=5.8)
    _panel(ax_b, "b", "Per-specialist risk relative to target")

    # (c) system automation and referral, the practical utility trade-off.
    _line(ax_c, alphas, cov_sys, C["coverage"], "^", "AI-managed")
    _line(ax_c, alphas, ref_sys, C["risk"], "o", "Human referral")
    ax_c.set_ylabel("Rate")
    ax_c.set_ylim(-0.02, 1.05)
    _alpha_log_axis(ax_c, label=True)
    _op_line(ax_c, op_alpha)
    _legend(ax_c, loc="center right", fontsize=6.1)
    _panel(ax_c, "c", "Automation and referral")

    # (d) retains all diseases while making the central tendency legible.
    q25, med, q75 = (np.nanpercentile(cov_k, q, axis=0) for q in (25, 50, 75))
    for curve in cov_k:
        ax_d.plot(alphas, curve, color="#C4C4C4", linewidth=0.65,
                  alpha=0.85, zorder=1)
    ax_d.fill_between(alphas, q25, q75, color=C["coverage"], alpha=0.16,
                      label="IQR across diseases", zorder=2)
    _line(ax_d, alphas, med, C["coverage"], "o", "Median", zorder=3)
    ax_d.set_ylabel("Per-specialist coverage")
    ax_d.set_ylim(-0.03, 1.05)
    _alpha_log_axis(ax_d, label=True)
    _op_line(ax_d, op_alpha)
    _legend(ax_d, loc="lower right", fontsize=6.0)
    _panel(ax_d, "d", "Coverage across 10 specialists")

    fig.suptitle(f"Sensitivity to specialist risk level $\\alpha$  ($\\beta={beta:g}$)",
                 fontsize=8.7, fontweight="bold", color=INK["text"])
    _savefig(fig, "fig3_alpha_sweep")
    print("  Fig 3: alpha sweep (risk / per-clinic risk / automation / per-clinic coverage)")


# ════════════════════════════════════════════════════════════════════
#  Fig 4: Baseline Comparison — Scatter
# ════════════════════════════════════════════════════════════════════

def _contrast_beta_rows(contrast_beta):
    """Router-risk grid rows at the contrast beta level, keyed by display name.

    Only beta-dependent systems move; the two no-router systems are dropped
    because their contrast rows are identical to the operating-point rows.
    """
    if not os.path.exists(BETA_GRID_JSON):
        return {}
    with open(BETA_GRID_JSON) as f:
        grid = json.load(f)["grid"]
    out = {}
    for key, rows in grid.items():
        name = GRID_KEY_TO_NAME.get(key)
        row = next((r for r in rows if r.get("beta") == contrast_beta), None)
        if name and row is not None and row.get("beta_dependent"):
            out[name] = row
    return out


def _plot_baselines_legacy(data):
    bl = data["baselines"]
    alpha = data["meta"]["alpha"]
    beta = data["meta"]["beta"]
    # The contrast layer mirrors the operating point: strict routers get a
    # relaxed beta = 0.1 replay (proposed degenerates onto specialist-only),
    # relaxed routers get the strict beta = 0.01 replay (routing binds).
    op_is_strict = abs(beta - BETA_STRICT) < 1e-9
    contrast_beta = BETA_RELAXED if op_is_strict else BETA_STRICT

    systems = [
        ("Full Automation", bl["full_automation"]["mFDR_sys"],
         bl["full_automation"]["Cov_sys"]),
        ("Router-Only Rejection", _val(bl["router_only_rejection"]["mFDR_sys"]),
         _val(bl["router_only_rejection"]["Cov_sys"])),
        ("Specialist-Only Rejection",
         _val(bl["specialist_only_rejection"]["mFDR_sys"]),
         _val(bl["specialist_only_rejection"]["Cov_sys"])),
        ("Shared Threshold", _val(bl["shared_threshold"]["mFDR_sys"]),
         _val(bl["shared_threshold"]["Cov_sys"])),
        ("Oracle-Clean Calibration",
         _val(bl["oracle_clean_calibration"]["mFDR_sys"]),
         _val(bl["oracle_clean_calibration"]["Cov_sys"])),
        ("Proposed System (LEC)", data["risk_control"]["system"]["pooled"],
         data["coverage"]["Cov_sys"]["pooled"]),
    ]

    contrast = _contrast_beta_rows(contrast_beta)

    fig, ax = plt.subplots(figsize=(6.6, 4.4), constrained_layout=True)
    # The strict-router cluster sits near Cov = 0.85, so the left margin now
    # reaches below the operating-point range; the right margin still only
    # has to hold the labels of the three systems at Cov = 1.000.
    xlim = (0.826, 1.042)

    # Safe-zone shading (below α)
    ax.axhspan(0, alpha, color=C["controlled"], alpha=0.04, zorder=0)

    # α threshold
    ax.axhline(alpha, color=C["gray"], linestyle="--", linewidth=1.0, zorder=2)
    ax.text(xlim[0] + 0.004, alpha + 0.0008, f"$\\alpha$ = {alpha}",
            fontsize=8, color=C["gray"], va="bottom", ha="left")

    filled = {}
    for name, mfdr, cov in systems:
        st = BL_STYLE[name]
        is_hero = "Proposed" in name
        ax.scatter(cov, mfdr, marker=st["marker"], color=st["color"],
                   s=220 if is_hero else 80, zorder=5,
                   edgecolors=_darken(st["color"], 0.25), linewidth=0.6)
        filled[name] = (cov, mfdr)

        # Direct labelling with the underlying numbers. At the strict
        # operating point router-only and shared coincide, so they share one
        # label anchored at the router-only marker.
        cfg = (BL_ANNOT_OP_STRICT if op_is_strict else BL_ANNOT)[name]
        label = st["short"]
        if op_is_strict and name == "Router-Only Rejection":
            label = "Router-Only / Shared"
        if cfg is None:
            continue
        if "xy" in cfg:
            xytext, coords = cfg["xy"], "data"
        else:
            xytext, coords = (cfg["dx"], cfg["dy"]), "offset points"
        ax.annotate(
            f"{label}\n(mFDR={mfdr:.3f}, Cov={cov:.3f})",
            (cov, mfdr), fontsize=7,
            fontweight="bold" if is_hero else "normal",
            color=st["color"] if is_hero else INK["title"],
            xytext=xytext, textcoords=coords,
            ha=cfg["ha"], va=cfg.get("va", "center"), linespacing=1.35,
            arrowprops=dict(arrowstyle="-", color=INK["hairline"],
                            linewidth=0.5, shrinkA=2, shrinkB=2),
            bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="none",
                      alpha=0.75),
        )

    # ── contrast-router layer ─────────────────────────────────────────────
    if contrast:
        # Migration arrows run underneath the markers and labels they
        # connect, so the operating-point annotations stay readable.
        for name, row in contrast.items():
            st = BL_STYLE[name]
            is_hero = "Proposed" in name
            if name in filled:
                cov0, mfdr0 = filled[name]
                ax.add_patch(FancyArrowPatch(
                    (cov0, mfdr0), (row["Cov_sys"], row["mFDR_sys"]),
                    arrowstyle="-|>", color=st["color"],
                    linewidth=1.3 if is_hero else 0.8,
                    alpha=0.60 if is_hero else 0.26,
                    shrinkA=9, shrinkB=9,
                    mutation_scale=10 if is_hero else 7, zorder=2.5))
            ax.scatter(row["Cov_sys"], row["mFDR_sys"], marker=st["marker"],
                       facecolors="white", edgecolors=st["color"],
                       s=220 if is_hero else 80, zorder=5, linewidth=1.1)

        annot_c = BL_ANNOT_CONTRAST_RELAXED if op_is_strict else BL_ANNOT_STRICT
        for name, cfg in annot_c.items():
            if name not in contrast:
                continue
            row = contrast[name]
            st = BL_STYLE[name]
            is_hero = "Proposed" in name
            if name == "Router-Only Rejection":
                txt = (f"Router-Only / Shared, $\\beta$={contrast_beta:g}\n"
                       f"(mFDR$\\approx${row['mFDR_sys']:.3f}, "
                       f"Cov$\\approx${row['Cov_sys']:.3f})")
            elif op_is_strict and name == "Proposed System (LEC)":
                txt = (f"{st['short']}, $\\beta$={contrast_beta:g}\n"
                       f"(coincides with Spec.-Only;\n"
                       f"mFDR={row['mFDR_sys']:.3f}, "
                       f"Cov={row['Cov_sys']:.3f})")
            else:
                txt = (f"{st['short']}, $\\beta$={contrast_beta:g}\n"
                       f"(mFDR={row['mFDR_sys']:.3f}, "
                       f"Cov={row['Cov_sys']:.3f})")
            ax.annotate(
                txt, (row["Cov_sys"], row["mFDR_sys"]), fontsize=7,
                fontweight="bold" if is_hero else "normal",
                color=st["color"] if is_hero else INK["title"],
                xytext=cfg["xy"], textcoords="data",
                ha=cfg["ha"], va=cfg["va"], linespacing=1.35,
                arrowprops=dict(arrowstyle="-", color=INK["hairline"],
                                linewidth=0.5, shrinkA=2, shrinkB=2),
                bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="none",
                          alpha=0.75))

        # Marker-fill legend: operating point vs the contrast router level.
        contrast_role = ("relaxed router" if contrast_beta > beta
                         else "strict router")
        handles = [
            plt.Line2D([], [], marker="o", linestyle="none", markersize=5,
                       markerfacecolor=C["gray"], markeredgecolor=C["gray"],
                       label=f"$\\beta$ = {beta:g} (operating point)"),
            plt.Line2D([], [], marker="o", linestyle="none", markersize=5,
                       markerfacecolor="white", markeredgecolor=C["gray"],
                       markeredgewidth=0.8,
                       label=f"$\\beta$ = {contrast_beta:g} ({contrast_role})"),
        ]
        ax.legend(handles=handles, loc="lower right", frameon=False,
                  fontsize=7, handlelength=1.2, handletextpad=0.5,
                  borderaxespad=0.3)

    ax.set_xlabel("System Coverage (Cov$_{sys}$)")
    ax.set_ylabel("System Risk (mFDR$_{sys}$)")
    ax.set_title(f"Baseline Comparison ($\\alpha$ = {alpha}, "
                 f"$\\beta$ = {beta:g} vs {contrast_beta:g})",
                 fontsize=11, fontweight="bold",
                 loc="left", pad=10, color=INK["text"])
    ax.set_xlim(*xlim)
    ax.set_ylim(0.122, 0.213)
    # Right margin exists only to host the labels of the Cov = 1.000 cluster;
    # ticks stop at 100% so the axis never implies impossible coverage.
    ax.set_xticks([0.84, 0.88, 0.92, 0.96, 1.00])
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=1))

    # Directional guide, kept inside the attainable coverage range
    ax.annotate("", xy=(0.998, 0.1585), xytext=(0.968, 0.1665),
                arrowprops=dict(arrowstyle="->", color=C["controlled"],
                                linewidth=1.0, alpha=0.30))
    ax.text(0.996, 0.1567, "Better", fontsize=7, color=C["controlled"],
            alpha=0.55, ha="center", fontstyle="italic")

    _savefig(fig, "fig4_baselines")
    print(f"  Fig 4: baseline comparison (operating β={beta:g} "
          f"vs contrast β={contrast_beta:g})")


# ════════════════════════════════════════════════════════════════════
#  Fig 4 (revised): Baseline comparison — sparse, direct comparison
# ════════════════════════════════════════════════════════════════════

def plot_baselines(data):
    """Show the coverage-risk frontier without annotation clutter.

    Filled markers are the operating point.  Hollow markers show the paired
    router-risk contrast when it is available; faint connectors preserve that
    information without asking the reader to parse six arrows and six labels.
    """
    bl = data["baselines"]
    alpha = data["meta"]["alpha"]
    beta = data["meta"]["beta"]
    op_is_strict = abs(beta - BETA_STRICT) < 1e-9
    contrast_beta = BETA_RELAXED if op_is_strict else BETA_STRICT

    systems = [
        ("Full Automation", bl["full_automation"]["mFDR_sys"],
         bl["full_automation"]["Cov_sys"]),
        ("Router-Only Rejection", _val(bl["router_only_rejection"]["mFDR_sys"]),
         _val(bl["router_only_rejection"]["Cov_sys"])),
        ("Specialist-Only Rejection", _val(bl["specialist_only_rejection"]["mFDR_sys"]),
         _val(bl["specialist_only_rejection"]["Cov_sys"])),
        ("Shared Threshold", _val(bl["shared_threshold"]["mFDR_sys"]),
         _val(bl["shared_threshold"]["Cov_sys"])),
        ("Oracle-Clean Calibration", _val(bl["oracle_clean_calibration"]["mFDR_sys"]),
         _val(bl["oracle_clean_calibration"]["Cov_sys"])),
        ("Proposed System (LEC)", data["risk_control"]["system"]["pooled"],
         data["coverage"]["Cov_sys"]["pooled"]),
    ]
    contrast = _contrast_beta_rows(contrast_beta)
    all_cov = [cov for _, _, cov in systems]
    all_risk = [risk for _, risk, _ in systems]
    for row in contrast.values():
        all_cov.append(row["Cov_sys"])
        all_risk.append(row["mFDR_sys"])
    xlim = (min(all_cov) - 0.018, min(1.01, max(all_cov) + 0.012))
    ylim = (max(0.0, min(all_risk) - 0.015), max(alpha + 0.010, max(all_risk) + 0.012))

    fig, ax = plt.subplots(figsize=(6.8, 4.25), constrained_layout=True)
    ax.axhspan(0, alpha, color=C["proposed"], alpha=0.035, zorder=0)
    ax.axhline(alpha, color=INK["muted"], linestyle="--", linewidth=0.8)
    ax.annotate(f"risk target $\\alpha={alpha}$", xy=(xlim[0], alpha),
                xytext=(3, 3), textcoords="offset points", fontsize=6.1,
                color=INK["muted"], ha="left", va="bottom")

    points = {}
    for name, risk, cov in systems:
        st = BL_STYLE[name]
        hero = name == "Proposed System (LEC)"
        ax.scatter(cov, risk, marker=st["marker"], color=st["color"],
                   s=155 if hero else 48, edgecolors=_darken(st["color"], 0.25),
                   linewidth=0.55, zorder=4, label=st["short"])
        points[name] = (cov, risk)

    for name, row in contrast.items():
        if name not in points:
            continue
        cov0, risk0 = points[name]
        st = BL_STYLE[name]
        ax.plot([cov0, row["Cov_sys"]], [risk0, row["mFDR_sys"]],
                color="#C7C7C7", linestyle=(0, (2, 2)), linewidth=0.8, zorder=1)
        ax.scatter(row["Cov_sys"], row["mFDR_sys"], marker=st["marker"],
                   s=145 if name == "Proposed System (LEC)" else 48,
                   facecolors="white", edgecolors=st["color"], linewidth=0.85,
                   zorder=3)

    # Direct labels only for the three points that establish the story.
    for name, offset in (("Proposed System (LEC)", (14, -18)),
                         ("Oracle-Clean Calibration", (24, 12)),
                         ("Full Automation", (-50, 14))):
        cov, risk = points[name]
        ax.annotate(BL_STYLE[name]["short"], xy=(cov, risk), xytext=offset,
                    textcoords="offset points", fontsize=6.4,
                    fontweight="bold" if name == "Proposed System (LEC)" else "normal",
                    color=INK["text"], ha="right" if offset[0] < 0 else "left",
                    va="center", arrowprops=dict(arrowstyle="-", color=INK["hairline"],
                                                   linewidth=0.5, shrinkA=2, shrinkB=2))

    method_legend = ax.legend(loc="lower right", frameon=False, ncol=2,
                              fontsize=5.7, handlelength=1.2, handletextpad=0.35,
                              columnspacing=0.85, borderaxespad=0.25)
    if contrast:
        ax.add_artist(method_legend)
        ax.legend(handles=[
            plt.Line2D([], [], marker="o", linestyle="none", markersize=4,
                       color=INK["muted"], label=f"Filled: $\\beta={beta:g}$"),
            plt.Line2D([], [], marker="o", linestyle="none", markersize=4,
                       markerfacecolor="white", markeredgecolor=INK["muted"],
                       color="white", label=f"Hollow: $\\beta={contrast_beta:g}$"),
        ], loc="upper left", frameon=False, fontsize=5.8,
            handlelength=1.1, handletextpad=0.4)

    ax.set_xlabel("System coverage")
    ax.set_ylabel("System risk (mFDR)")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_title("Baseline comparison", fontsize=8.5, fontweight="bold",
                 loc="left", pad=15, color=INK["text"])

    _savefig(fig, "fig4_baselines")
    print(f"  Fig 4: baseline comparison (operating beta={beta:g})")


# ════════════════════════════════════════════════════════════════════
#  Fig 5: Routing Contamination Decomposition
# ════════════════════════════════════════════════════════════════════

def plot_contamination(data):
    cont = data["contamination"]
    names = data["meta"]["disease_names"]
    alpha = data["meta"]["alpha"]
    beta = data["meta"]["beta"]

    disease_labels = []
    rho_route, rho_acc, q_route, q_diag = [], [], [], []

    for k in range(NUM_DISEASES):
        c = cont.get(str(k))
        if c is None:
            continue
        disease_labels.append(_short1(names[str(k)]))
        rho_route.append(_val(c["rho_route"]) or 0)
        rho_acc.append(_val(c["rho_acc"]) or 0)
        q_route.append(c.get("q_route_err") or 0)
        q_diag.append(c.get("q_diag_err") or 0)

    # Horizontal layouts eliminate rotated labels.  Sort by pre-rejection
    # contamination, the quantity that determines where routing is fragile.
    order = np.argsort(np.asarray(rho_route))[::-1]
    labels = [disease_labels[i] for i in order]
    rho_route = np.asarray(rho_route)[order]
    rho_acc = np.asarray(rho_acc)[order]
    q_route = np.asarray(q_route)[order]
    q_diag = np.asarray(q_diag)[order]
    y = np.arange(len(labels))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.25, 4.35), sharey=True,
                                   constrained_layout=True)

    # (a) paired dumbbell plot: the distance itself encodes the effect of
    # rejection, without two near-identical bars for every disease.
    for yi, before, after in zip(y, rho_route, rho_acc):
        ax1.plot([after, before], [yi, yi], color="#C7C7C7", linewidth=1.0,
                 zorder=1, solid_capstyle="round")
    ax1.scatter(rho_route, y, color=C["rho_route"], s=22, zorder=3,
                label="Before rejection")
    ax1.scatter(rho_acc, y, color=C["rho_acc"], s=22, zorder=3,
                label="After rejection")
    ax1.set_yticks(y)
    ax1.set_yticklabels(labels, fontsize=6.3)
    ax1.invert_yaxis()
    ax1.set_xlabel("Contamination rate")
    ax1.set_xlim(left=0)
    _legend(ax1, loc="lower right", fontsize=6.0)
    _panel(ax1, "a", "Routing contamination")

    # (b) composition follows precisely the same disease ordering.
    ax2.barh(y, q_route, 0.56, color=C["route_err"],
             label="Routing", edgecolor="white", linewidth=0.35, zorder=2)
    ax2.barh(y, q_diag, 0.56, left=q_route, color=C["diag_err"],
             label="Diagnostic", edgecolor="white", linewidth=0.35, zorder=2)
    ax2.set_xlabel("Fraction of accepted errors")
    ax2.set_xlim(0, 1.02)
    ax2.tick_params(axis="y", left=False, labelleft=False)
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=2,
               fontsize=6.0, frameon=False, handlelength=1.2,
               handletextpad=0.4, columnspacing=1.0, borderaxespad=0.0)
    _panel(ax2, "b", "Error-source composition")

    fig.suptitle(f"Routing contamination and error decomposition  "
                 f"($\\alpha={alpha}$, $\\beta={beta:g}$)",
                 fontsize=8.7, fontweight="bold", color=INK["text"])
    _savefig(fig, "fig5_contamination")
    print("  Fig 5: contamination")


# ════════════════════════════════════════════════════════════════════
#  Fig 6: Calibration Size Sensitivity
# ════════════════════════════════════════════════════════════════════

def plot_calibration_size(data):
    """Fig 6: calibration-size sensitivity.

    Reports the full set required by Methods (Calibration Size and
    Feasibility): threshold variability, accepted-case error, AI-managed rate
    and per-decision-point infeasibility frequency.
    """
    csa = data["calibration_size_analysis"]
    names = data["meta"]["disease_names"]
    alpha = data["meta"]["alpha"]
    beta = data["meta"]["beta"]

    ratios = [r["split_ratio"] for r in csa]
    pct = [f"{int(r * 100)}%" for r in ratios]
    mfdr_sys = [r["mFDR_sys"] for r in csa]
    ci_lo = [r.get("ci_low_sys", r["mFDR_sys"]) for r in csa]
    ci_hi = [r.get("ci_high_sys", r["mFDR_sys"]) for r in csa]
    cov_sys = [r["Cov_sys"] for r in csa]
    mfdr_r = [r["router"]["mFDR_R"] for r in csa]
    cov_r = [r["router"]["Cov_R"] for r in csa]
    thr_std_r = [r["router"]["threshold_std"] for r in csa]
    inf_r = [r["router"]["infeasible_freq"] for r in csa]

    mfdr_k, mfdr_k_mean, _, _ = _clinic_stat(csa, "specialist", "mFDR_k")
    thr_k, _, thr_k_lo, thr_k_hi = _clinic_stat(csa, "specialist", "threshold_std")
    inf_k, _, _, _ = _clinic_stat(csa, "specialist", "infeasible_freq")

    fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.15), constrained_layout=True)
    (ax_a, ax_b), (ax_c, ax_d) = axes
    xlim = (0.07, 0.525)

    def _xaxis(ax, label=False):
        ax.set_xticks(ratios)
        ax.set_xticklabels(pct)
        ax.set_xlim(*xlim)
        if label:
            ax.set_xlabel("Calibration set ratio")

    # (a) accepted-case error ------------------------------------------------
    ax_a.fill_between(ratios, ci_lo, ci_hi, color=C["risk"], alpha=0.12, zorder=1)
    _line(ax_a, ratios, mfdr_sys, C["risk"], "o", "System")
    _line(ax_a, ratios, mfdr_r, C["risk2"], "s", "Router")
    _line(ax_a, ratios, mfdr_k_mean, C["gray"], "^", "Specialist mean")
    ax_a.axhline(alpha, color=INK["muted"], linestyle="--", linewidth=0.8)
    ax_a.set_ylabel("Accepted-case error rate")
    ax_a.set_ylim(0, 0.245)
    _legend(ax_a, loc="lower right", fontsize=6.0)
    _xaxis(ax_a)
    _panel(ax_a, "a", "Risk control")

    # (b) AI-managed rate with data-driven limits so a near-constant series
    # remains visible rather than disappearing into the upper frame.
    _line(ax_b, ratios, cov_r, C["gray"], "^", "Routed")
    _line(ax_b, ratios, cov_sys, C["coverage"], "o", "AI-managed")
    cov_all = np.asarray(cov_r + cov_sys, dtype=float)
    lo, hi = np.nanmin(cov_all), np.nanmax(cov_all)
    pad = max(0.004, (hi - lo) * 0.35)
    ax_b.set_ylabel("Coverage")
    ax_b.set_ylim(max(0, lo - pad), min(1.005, hi + pad))
    _legend(ax_b, loc="lower right", fontsize=6.0)
    _xaxis(ax_b)
    _panel(ax_b, "b", "AI-managed coverage")

    # (c) all ten disease trajectories are neutral context; the IQR and
    # median are the readable statistical summary.
    q25, med, q75 = (np.nanpercentile(thr_k, q, axis=0) for q in (25, 50, 75))
    for curve in thr_k:
        ax_c.plot(ratios, curve, color="#C4C4C4", linewidth=0.65,
                  alpha=0.85, zorder=1)
    ax_c.fill_between(ratios, q25, q75, color=C["coverage"], alpha=0.16,
                      label="Specialist IQR", zorder=2)
    _line(ax_c, ratios, med, C["coverage"], "o", "Specialist median", zorder=3)
    _line(ax_c, ratios, thr_std_r, C["risk"], "s", "Router", zorder=4)
    ax_c.set_ylabel("Threshold s.d. across splits")
    ax_c.set_ylim(bottom=0)
    _legend(ax_c, loc="upper right", fontsize=5.9)
    _xaxis(ax_c, label=True)
    _panel(ax_c, "c", "Threshold variability")

    # (d) preserve the one non-zero disease trajectory and summarise the
    # otherwise zero set in a short note rather than a ten-item legend.
    _line(ax_d, ratios, inf_r, C["gray"], "s", "Router", zorder=3)
    nonzero = [k for k in range(NUM_DISEASES) if np.nansum(inf_k[k]) > 0]
    for k in nonzero:
        _line(ax_d, ratios, inf_k[k], C["risk"], "o", f"D{k}", zorder=4)
    ax_d.set_ylabel("Infeasibility frequency")
    ax_d.set_ylim(bottom=-0.012)
    _legend(ax_d, loc="upper right", fontsize=6.0)
    ax_d.text(0.53, 0.42, "All remaining specialists: 0\nat every calibration ratio",
              transform=ax_d.transAxes, fontsize=6.1, color=INK["muted"],
              ha="center", va="center")
    _xaxis(ax_d, label=True)
    _panel(ax_d, "d", "Calibration feasibility")

    fig.suptitle(f"Sensitivity to calibration-set size  ($\\alpha={alpha}$, $\\beta={beta:g}$)",
                 fontsize=8.7, fontweight="bold", color=INK["text"])
    _savefig(fig, "fig6_calibration_size")
    print("  Fig 6: calibration size (risk / coverage / threshold s.d. / feasibility)")


# ════════════════════════════════════════════════════════════════════
#  Fig 7: Clinical Utility Heatmap
# ════════════════════════════════════════════════════════════════════

def plot_utility_heatmap(data):
    grid = data["clinical_utility"]["utility_grid"]
    alpha = data["meta"]["alpha"]
    beta = data["meta"]["beta"]
    c_errs = [1, 5, 10, 20, 50]
    c_refs = [0.1, 0.5, 1.0, 2.0]

    mat = np.zeros((len(c_errs), len(c_refs)))
    for i, ce in enumerate(c_errs):
        for j, cr in enumerate(c_refs):
            key = f"c_err={ce}_c_ref={cr}"
            mat[i, j] = grid.get(key, 0)

    vmax = max(abs(mat.max()), abs(mat.min()))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(4.55, 3.55), constrained_layout=True)
    im = ax.imshow(mat, cmap=UTILITY_CMAP, norm=norm, aspect="auto")

    ax.set_xticks(range(len(c_refs)))
    ax.set_xticklabels([str(cr) for cr in c_refs], fontsize=7)
    ax.set_yticks(range(len(c_errs)))
    ax.set_yticklabels([str(ce) for ce in c_errs], fontsize=7)
    ax.set_xlabel("Referral cost ($c_{ref}$)")
    ax.set_ylabel("Error cost ($c_{err}$)")
    ax.set_title(r"Clinical utility $\hat{U}(c_{err}, c_{ref})$",
                 fontsize=8.5, fontweight="bold", loc="left", pad=14,
                 color=INK["text"])
    ax.text(0, 1.015,
            f"Operating point: $\\alpha$ = {alpha}, $\\beta$ = {beta}",
            transform=ax.transAxes, fontsize=6.1, color=INK["muted"],
            va="bottom", ha="left")

    # Cell annotations — ink chosen from the rendered cell brightness
    for i in range(len(c_errs)):
        for j in range(len(c_refs)):
            val = mat[i, j]
            bg = to_hex(UTILITY_CMAP(norm(val)))
            ax.text(j, i, f"{val:+.2f}", ha="center", va="center",
                    fontsize=7, color=_cell_text_color(bg),
                    fontweight="bold" if abs(val) > vmax * 0.25 else "normal")

    # Cell borders via minor ticks
    ax.set_xticks(np.arange(-0.5, len(c_refs), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(c_errs), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", length=0)

    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.03)
    cbar.ax.tick_params(labelsize=6)
    cbar.outline.set_linewidth(0.7)
    cbar.set_label("Utility score", fontsize=7)

    _savefig(fig, "fig7_utility_heatmap")
    print("  Fig 7: utility heatmap")


# ════════════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════════════

def main(metrics_json=None, fig_dir=None):
    """Render the seven figures with explicit optional input/output paths."""
    global FIG_DIR
    previous_fig_dir = FIG_DIR
    if fig_dir is not None:
        FIG_DIR = os.fspath(fig_dir)
    try:
        os.makedirs(FIG_DIR, exist_ok=True)
        _setup_style()
        data = _load_data(metrics_json)

        print("Generating figures...")
        plot_disease_forest(data)
        plot_beta_sweep(data)
        plot_alpha_sweep(data)
        plot_baselines(data)
        plot_contamination(data)
        plot_calibration_size(data)
        plot_utility_heatmap(data)
        print(f"\nAll figures saved to {FIG_DIR}")
    finally:
        FIG_DIR = previous_fig_dir


if __name__ == "__main__":
    main()
