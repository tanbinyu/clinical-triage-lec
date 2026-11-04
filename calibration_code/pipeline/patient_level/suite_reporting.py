"""Full-suite patient-level figures and report orchestration."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from .metadata import DISEASE_NAMES, DISEASE_SHORT_NAMES, NUM_DISEASES
from .suite import METHOD_LABELS, METHODS
from .suite_tables import write_manifest, write_tables


BLUE = "#0072B2"
GRAY = "#4D4D4D"
ORANGE = "#D55E00"
GREEN = "#009E73"
PURPLE = "#CC79A7"
REFERENCE = "#9A9A9A"
LIGHT = "#E8E8E8"
METHOD_COLORS = {
    "full_automation": "#999999",
    "router_only": "#009E73",
    "specialist_only": "#4D4D4D",
    "shared_threshold": "#CC79A7",
    "oracle_clean": "#E69F00",
    "proposed": "#0072B2",
}
METHOD_MARKERS = {
    "full_automation": "X", "router_only": "^", "specialist_only": "s",
    "shared_threshold": "D", "oracle_clean": "v", "proposed": "o",
}


def _style() -> None:
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 8.0,
        "axes.labelsize": 8.5,
        "axes.titlesize": 9.0,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.0,
        "axes.linewidth": 0.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
    })


def _clean(axis: plt.Axes, grid_axis: str = "both") -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    if grid_axis != "none":
        axis.grid(axis=grid_axis, color=LIGHT, linewidth=0.55, zorder=0)
    axis.tick_params(width=0.7, length=3)


def _panel(axis: plt.Axes, letter: str, title: str) -> None:
    axis.set_title(f"{letter}  {title}", loc="left", fontweight="bold")


def _line(axis: plt.Axes, x, y, color, marker, label, **kwargs) -> None:
    axis.plot(
        x, y, color=color, marker=marker, markersize=3.8,
        linewidth=1.25, label=label, zorder=3, **kwargs,
    )


def _save(fig: plt.Figure, directory: Path, stem: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory / f"{stem}.pdf")
    fig.savefig(directory / f"{stem}.png", dpi=400)
    plt.close(fig)


def _finite(value: float | None) -> float:
    return np.nan if value is None else float(value)


def plot_fig1_forest(suite: dict[str, Any], primary: dict[str, Any], directory: Path) -> None:
    rows = primary["bootstrap"]["per_disease"]
    alpha = suite["meta"]["alpha"]
    ordered = sorted(rows, key=lambda row: row["mFDR_sys"]["reference_pooled"], reverse=True)
    y = np.arange(len(ordered))
    fig, axis = plt.subplots(figsize=(5.2, 4.15), constrained_layout=True)
    axis.axvspan(0, alpha, color=BLUE, alpha=0.035)
    for position, row in enumerate(ordered):
        summary = row["mFDR_sys"]
        estimate = summary["reference_pooled"]
        low, high = summary["ci_low"], summary["ci_high"]
        color = BLUE if estimate <= alpha else ORANGE
        axis.hlines(position, low, high, color=color, linewidth=1.2)
        axis.vlines([low, high], position - 0.07, position + 0.07,
                    color=color, linewidth=0.9)
        axis.plot(estimate, position, "o", color=color, markersize=4.4)
        axis.text(estimate + 0.006, position, f"{estimate:.3f}", va="center", fontsize=6.5)
    axis.axvline(alpha, color=REFERENCE, linestyle="--", linewidth=1)
    axis.set_yticks(y, [row["disease_name"] for row in ordered])
    axis.invert_yaxis()
    axis.set_xlabel("Patient-level mFDR")
    axis.set_title("Clinic-specific patient-level accepted-case risk", loc="left", fontweight="bold")
    _clean(axis, "x")
    _save(fig, directory, "fig1_disease_forest")


def plot_fig2_beta(suite: dict[str, Any], directory: Path) -> None:
    rows = suite["beta_grid"]
    beta = [row["beta"] for row in rows]
    alpha = suite["meta"]["alpha"]
    op = suite["meta"]["beta"]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2), constrained_layout=True)
    (a, b), (c, d) = axes
    _line(a, beta, [row["router_mFDR"] for row in rows], ORANGE, "s", "Router risk")
    _line(a, beta, [row["proposed"]["mFDR_sys"] for row in rows], BLUE, "o", "System risk")
    a.plot(beta, beta, ":", color=REFERENCE, linewidth=0.9, label=r"$y=\beta$")
    a.axhline(alpha, color=REFERENCE, linestyle="--", linewidth=0.9)
    a.set_ylabel("Accepted-case error rate")
    a.legend(frameon=False)
    _panel(a, "a", "Risk control")
    _line(b, beta, [row["router_coverage"] for row in rows], GRAY, "^", "Routed")
    _line(b, beta, [row["proposed"]["coverage"] for row in rows], BLUE, "o", "AI-managed")
    b.set_ylabel("Coverage")
    b.legend(frameon=False, loc="lower right")
    _panel(b, "b", "Coverage")
    _line(c, beta, [1-row["proposed"]["coverage"] for row in rows], ORANGE, "o", "Total")
    _line(c, beta, [row["router_referral"] for row in rows], GRAY, "s", "Router")
    _line(c, beta, [row["specialist_referral"] for row in rows], GREEN, "^", "Specialist")
    c.set_xlabel(r"Router risk level $\beta$")
    c.set_ylabel("Patient referral rate")
    c.legend(frameon=False)
    _panel(c, "c", "Human referral")
    d.fill_between(beta, [row["rho_route_min"] for row in rows],
                   [row["rho_route_max"] for row in rows], color=PURPLE, alpha=0.12,
                   label="Clinic range")
    _line(d, beta, [row["rho_route_mean"] for row in rows], PURPLE, "o", "Before rejection")
    _line(d, beta, [row["rho_acc_mean"] for row in rows], GREEN, "s", "After rejection")
    d.set_xlabel(r"Router risk level $\beta$")
    d.set_ylabel("Routing contamination")
    d.legend(frameon=False)
    _panel(d, "d", "Routing contamination")
    for axis in axes.flat:
        axis.axvline(op, color=REFERENCE, linewidth=0.75, alpha=0.8)
        _clean(axis)
    fig.suptitle(rf"Patient-level sensitivity to router target ($\alpha={alpha:g}$)", fontweight="bold")
    _save(fig, directory, "fig2_beta_sweep")


def plot_fig3_alpha(suite: dict[str, Any], directory: Path) -> None:
    rows = suite["alpha_grid"]
    alphas = np.asarray([row["alpha"] for row in rows])
    proposed = [row["methods"]["proposed"] for row in rows]
    op = suite["meta"]["alpha"]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2), constrained_layout=True)
    (a, b), (c, d) = axes
    _line(a, alphas, [row["mFDR_sys"] for row in proposed], BLUE, "o", "System")
    _line(a, alphas, [row["mFDR_macro"] for row in proposed], GRAY, "s", "Specialist mean")
    _line(a, alphas, [row["mFDR_worst"] for row in proposed], PURPLE, "^", "Specialist worst")
    a.plot(alphas, alphas, ":", color=REFERENCE, linewidth=0.9, label=r"$y=\alpha$")
    a.set_ylabel("Accepted-case error rate")
    a.legend(frameon=False, fontsize=6.4)
    _panel(a, "a", "Risk control")
    matrix = np.asarray([
        [row["methods"]["proposed"]["per_disease"][k]["mFDR"] / row["alpha"]
         if row["methods"]["proposed"]["per_disease"][k]["mFDR"] is not None else np.nan
         for row in rows]
        for k in range(NUM_DISEASES)
    ])
    vmax = max(1.05, float(np.nanmax(matrix)))
    cmap = LinearSegmentedColormap.from_list("patient_risk", ["#2166AC", "#F7F7F7", ORANGE])
    image = b.imshow(matrix, aspect="auto", cmap=cmap,
                     norm=TwoSlopeNorm(vmin=0, vcenter=1, vmax=vmax))
    b.set_xticks(range(len(alphas)), [f"{x:g}" for x in alphas], rotation=35)
    b.set_yticks(range(NUM_DISEASES), [f"D{k}" for k in range(NUM_DISEASES)])
    b.set_xlabel(r"Specialist target $\alpha$")
    b.set_ylabel("Specialist")
    fig.colorbar(image, ax=b, fraction=0.047, pad=0.025, label="Risk / target")
    _panel(b, "b", "Clinic risk relative to target")
    _line(c, alphas, [row["coverage"] for row in proposed], BLUE, "o", "AI-managed")
    _line(c, alphas, [row["referral"] for row in proposed], ORANGE, "s", "Human referral")
    c.set_xlabel(r"Specialist target $\alpha$")
    c.set_ylabel("Rate")
    c.legend(frameon=False)
    _panel(c, "c", "Automation and referral")
    coverage = np.asarray([
        [row["methods"]["proposed"]["per_disease"][k]["coverage"] for row in rows]
        for k in range(NUM_DISEASES)
    ], dtype=float)
    for curve in coverage:
        d.plot(alphas, curve, color="#C4C4C4", linewidth=0.65)
    q25, med, q75 = (np.nanpercentile(coverage, q, axis=0) for q in (25, 50, 75))
    d.fill_between(alphas, q25, q75, color=BLUE, alpha=0.15)
    _line(d, alphas, med, BLUE, "o", "Median")
    d.set_xlabel(r"Specialist target $\alpha$")
    d.set_ylabel("Clinic coverage")
    _panel(d, "d", "Coverage across clinics")
    for axis in (a, c, d):
        axis.set_xscale("log")
        axis.xaxis.set_major_formatter(mticker.FuncFormatter(lambda value, _: f"{value:g}"))
        axis.axvline(op, color=REFERENCE, linewidth=0.75)
        _clean(axis)
    _clean(b, "none")
    fig.suptitle(rf"Patient-level sensitivity to specialist target ($\beta={suite['meta']['beta']:g}$)", fontweight="bold")
    _save(fig, directory, "fig3_alpha_sweep")


def plot_fig4_baselines(suite: dict[str, Any], directory: Path) -> None:
    methods = suite["primary"]["methods"]
    alpha = suite["meta"]["alpha"]
    fig, axis = plt.subplots(figsize=(6.4, 4.3), constrained_layout=True)
    axis.axhspan(0, alpha, color=BLUE, alpha=0.035)
    axis.axhline(alpha, color=REFERENCE, linestyle="--", linewidth=1)
    for method in METHODS:
        row = methods[method]
        axis.scatter(row["coverage"], row["mFDR_sys"], s=100 if method == "proposed" else 55,
                     marker=METHOD_MARKERS[method], color=METHOD_COLORS[method],
                     label=METHOD_LABELS[method], zorder=3)
    axis.set_xlabel("Patient-level AI-managed coverage")
    axis.set_ylabel("Patient-level mFDR")
    axis.set_title("Risk–coverage comparison at the primary operating point", loc="left", fontweight="bold")
    axis.legend(loc="lower left", ncol=2, frameon=False, handletextpad=0.4,
                columnspacing=0.9)
    _clean(axis)
    _save(fig, directory, "fig4_baselines")


def plot_fig5_contamination(suite: dict[str, Any], directory: Path) -> None:
    rows = suite["primary"]["contamination"]
    labels = [DISEASE_SHORT_NAMES[row["disease_id"]] for row in rows]
    y = np.arange(len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 4.7), sharey=True, constrained_layout=True)
    width = 0.35
    axes[0].barh(y - width/2, [_finite(row["rho_route"]) for row in rows], width,
                 color=PURPLE, label="Before rejection")
    axes[0].barh(y + width/2, [_finite(row["rho_acc"]) for row in rows], width,
                 color=GREEN, label="Accepted")
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Wrong-routed fraction")
    axes[0].legend(frameon=False)
    _panel(axes[0], "a", "Routing contamination")
    route = np.asarray([_finite(row["route_error_share"]) for row in rows])
    diag = np.asarray([_finite(row["diagnostic_error_share"]) for row in rows])
    axes[1].barh(y, route, color=ORANGE, label="Routing-induced")
    axes[1].barh(y, diag, left=route, color=GRAY, label="Within-clinic")
    axes[1].set_xlabel("Share of accepted errors")
    axes[1].legend(frameon=False)
    _panel(axes[1], "b", "Accepted-error source")
    for axis in axes:
        _clean(axis, "x")
    _save(fig, directory, "fig5_contamination")


def plot_fig6_calibration(suite: dict[str, Any], directory: Path) -> None:
    rows = suite["calibration_size_grid"]
    x = [row["fraction"] for row in rows]
    min_clinic = min(
        rows[-1]["clinics"], key=lambda row: row["mean_calibration_size"]
    )["disease_id"]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2), constrained_layout=True)
    (a, b), (c, d) = axes
    _line(a, x, [row["router_mFDR"] for row in rows], ORANGE, "s", "Router")
    _line(a, x, [row["system"]["mFDR_sys"] for row in rows], BLUE, "o", "System")
    _line(a, x, [row["system"]["mFDR_macro"] for row in rows], GRAY, "^", "Clinic mean")
    a.axhline(suite["meta"]["alpha"], color=REFERENCE, linestyle="--", linewidth=0.9)
    a.set_ylabel("Accepted-case error rate")
    a.legend(frameon=False)
    _panel(a, "a", "Risk")
    _line(b, x, [row["router_coverage"] for row in rows], GRAY, "s", "Router")
    _line(b, x, [row["system"]["coverage"] for row in rows], BLUE, "o", "AI-managed")
    b.set_ylabel("Coverage")
    b.legend(frameon=False)
    _panel(b, "b", "Coverage")
    _line(c, x, [row["router_threshold_sd"] for row in rows], ORANGE, "s", "Router")
    clinic_median_sd = [np.nanmedian([
        _finite(clinic["threshold_sd"]) for clinic in row["clinics"]
    ]) for row in rows]
    _line(c, x, clinic_median_sd, BLUE, "o", "Median specialist")
    c.set_xlabel("Fraction of primary calibration patients retained")
    c.set_ylabel("Threshold SD")
    c.legend(frameon=False)
    _panel(c, "c", "Threshold stability")
    _line(d, x, [row["router_infeasible_frequency"] for row in rows], ORANGE, "s", "Router")
    _line(d, x, [row["clinics"][min_clinic]["infeasible_frequency"] for row in rows],
          BLUE, "o", f"D{min_clinic}: smallest stream")
    d.set_xlabel("Fraction of primary calibration patients retained")
    d.set_ylabel("Infeasible-split frequency")
    d.legend(frameon=False)
    _panel(d, "d", "Calibration feasibility")
    for axis in axes.flat:
        _clean(axis)
    _save(fig, directory, "fig6_calibration_size")
    # Retain the legacy duplicate name so every historical artifact has a
    # patient-level counterpart; the manuscript should cite only Fig. 6.
    plot_fig12_calibration_detail(suite, directory, min_clinic)


def plot_fig12_calibration_detail(suite: dict[str, Any], directory: Path, min_clinic: int) -> None:
    rows = suite["calibration_size_grid"]
    x = [row["fraction"] for row in rows]
    fig, axis = plt.subplots(figsize=(5.3, 3.6), constrained_layout=True)
    for k in range(NUM_DISEASES):
        axis.plot(x, [row["clinics"][k]["mean_calibration_size"] for row in rows],
                  color=BLUE if k == min_clinic else "#C4C4C4",
                  linewidth=1.5 if k == min_clinic else 0.7,
                  marker="o" if k == min_clinic else None,
                  label=f"D{min_clinic}: {DISEASE_SHORT_NAMES[min_clinic]}" if k == min_clinic else None)
    axis.set_xlabel("Fraction of primary calibration patients retained")
    axis.set_ylabel("Effective specialist calibration patients")
    axis.set_title("Patient-level routed calibration evidence", loc="left", fontweight="bold")
    axis.legend(frameon=False)
    _clean(axis)
    _save(fig, directory, "fig12_calibration_size")


def plot_fig7_utility(suite: dict[str, Any], directory: Path) -> None:
    c_errors = [1, 5, 10, 20, 50]
    c_refs = [0.1, 0.5, 1.0, 2.0]
    matrix = np.asarray([
        [suite["utility"][f"{error}|{referral}"] for referral in c_refs]
        for error in c_errors
    ])
    vmax = max(abs(float(matrix.min())), abs(float(matrix.max())))
    cmap = LinearSegmentedColormap.from_list("utility", [ORANGE, "#F7F7F7", BLUE])
    fig, axis = plt.subplots(figsize=(4.6, 3.6), constrained_layout=True)
    image = axis.imshow(matrix, aspect="auto", cmap=cmap,
                        norm=TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax))
    axis.set_xticks(range(len(c_refs)), c_refs)
    axis.set_yticks(range(len(c_errors)), c_errors)
    axis.set_xlabel(r"Referral cost $c_{ref}$")
    axis.set_ylabel(r"Error cost $c_{err}$")
    axis.set_title("Patient-level illustrative clinical utility", loc="left", fontweight="bold")
    for i in range(len(c_errors)):
        for j in range(len(c_refs)):
            axis.text(j, i, f"{matrix[i,j]:+.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(image, ax=axis, shrink=0.85, label="Utility score")
    _save(fig, directory, "fig7_utility_heatmap")


def plot_fig8_frontier(suite: dict[str, Any], directory: Path) -> None:
    rows = suite["alpha_grid"]
    alphas = [row["alpha"] for row in rows]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2), constrained_layout=True)
    (a, b), (c, d) = axes
    for method in METHODS:
        risk = [row["methods"][method]["mFDR_sys"] for row in rows]
        coverage = [row["methods"][method]["coverage"] for row in rows]
        a.plot(coverage, risk, marker=METHOD_MARKERS[method], markersize=3.3,
               color=METHOD_COLORS[method], linewidth=1.0, label=METHOD_LABELS[method])
    a.axhline(suite["meta"]["alpha"], color=REFERENCE, linestyle="--", linewidth=0.8)
    a.set_xlabel("Coverage")
    a.set_ylabel("mFDR")
    a.legend(frameon=False, fontsize=5.6)
    _panel(a, "a", "Risk–coverage frontiers")
    for method in ("proposed", "specialist_only", "shared_threshold", "oracle_clean"):
        _line(b, alphas, [row["methods"][method]["mFDR_sys"] for row in rows],
              METHOD_COLORS[method], METHOD_MARKERS[method], METHOD_LABELS[method])
    b.plot(alphas, alphas, ":", color=REFERENCE, linewidth=0.8)
    b.set_xlabel(r"Specialist target $\alpha$")
    b.set_ylabel("System mFDR")
    b.legend(frameon=False, fontsize=5.8)
    _panel(b, "b", "System risk against target")
    for method in ("proposed", "shared_threshold", "oracle_clean"):
        _line(c, alphas, [row["methods"][method]["K_meet"] for row in rows],
              METHOD_COLORS[method], METHOD_MARKERS[method], METHOD_LABELS[method])
    c.set_xlabel(r"Specialist target $\alpha$")
    c.set_ylabel("Clinics meeting target")
    c.set_ylim(-0.3, 10.5)
    c.legend(frameon=False, fontsize=5.8)
    _panel(c, "c", "Clinic-level control")
    for method in ("proposed", "shared_threshold", "oracle_clean"):
        _line(d, alphas, [row["methods"][method]["mFDR_worst"] for row in rows],
              METHOD_COLORS[method], METHOD_MARKERS[method], METHOD_LABELS[method])
    d.plot(alphas, alphas, ":", color=REFERENCE, linewidth=0.8)
    d.set_xlabel(r"Specialist target $\alpha$")
    d.set_ylabel("Worst-clinic mFDR")
    d.legend(frameon=False, fontsize=5.8)
    _panel(d, "d", "Worst clinic")
    for axis in axes.flat:
        _clean(axis)
    for axis in (b, c, d):
        axis.set_xscale("log")
        axis.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda value, _: f"{value:g}")
        )
    _save(fig, directory, "fig8_risk_coverage_frontier")


def plot_fig9_router(suite: dict[str, Any], directory: Path) -> None:
    rows = suite["beta_grid"]
    beta = [row["beta"] for row in rows]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2), constrained_layout=True)
    (a, b), (c, d) = axes
    _line(a, beta, [row["proposed"]["mFDR_sys"] for row in rows], BLUE, "o", "Proposed")
    _line(a, beta, [row["specialist_only"]["mFDR_sys"] for row in rows], GRAY, "s", "Specialist-only")
    _line(a, beta, [row["full_automation"]["mFDR_sys"] for row in rows], REFERENCE, "^", "Full automation")
    a.set_ylabel("System mFDR")
    a.legend(frameon=False)
    _panel(a, "a", "System risk")
    _line(b, beta, [row["proposed"]["coverage"] for row in rows], BLUE, "o", "Proposed")
    _line(b, beta, [row["specialist_only"]["coverage"] for row in rows], GRAY, "s", "Specialist-only")
    b.set_ylabel("AI-managed coverage")
    b.legend(frameon=False)
    _panel(b, "b", "Coverage")
    _line(c, beta, [row["router_mFDR"] for row in rows], ORANGE, "s", "Router risk")
    _line(c, beta, [row["router_coverage"] for row in rows], GREEN, "o", "Router coverage")
    c.plot(beta, beta, ":", color=REFERENCE, linewidth=0.8)
    c.set_xlabel(r"Router target $\beta$")
    c.set_ylabel("Rate")
    c.legend(frameon=False)
    _panel(c, "c", "Router operating point")
    delta = [row["proposed"]["mFDR_sys"] - row["specialist_only"]["mFDR_sys"] for row in rows]
    _line(d, beta, delta, ORANGE, "o", r"$\Delta$mFDR")
    d.axhline(0, color=REFERENCE, linestyle="--", linewidth=0.8)
    d2 = d.twinx()
    d2.plot(beta, [row["router_coverage"] for row in rows], color=BLUE,
            marker="s", markersize=3.5, linewidth=1.1, label="Router coverage")
    d.set_xlabel(r"Router target $\beta$")
    d.set_ylabel(r"Proposed $-$ specialist-only")
    d2.set_ylabel("Router coverage", color=BLUE)
    _panel(d, "d", "Incremental router contribution")
    for axis in axes.flat:
        _clean(axis)
    _save(fig, directory, "fig9_router_risk_sweep")


def plot_fig10_exchangeability(suite: dict[str, Any], directory: Path) -> None:
    rows = suite["primary"]["exchangeability"]
    alpha_rows = suite["alpha_grid"]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2), constrained_layout=True)
    (a, b), (c, d) = axes
    a.scatter([row["base_error_cal"] for row in rows], [row["base_error_test"] for row in rows],
              color=BLUE, s=24)
    maximum = max(max(row["base_error_cal"] for row in rows), max(row["base_error_test"] for row in rows))
    a.plot([0, maximum], [0, maximum], "--", color=REFERENCE, linewidth=0.8)
    a.set_xlabel("Specialist-calibration base error")
    a.set_ylabel("Held-out base error")
    _panel(a, "a", "Routed-stream base error")
    ordered = sorted(rows, key=lambda row: row["mean_effective_calibration_size"])
    x = np.arange(len(ordered))
    b.bar(x, [row["KS_mean"] for row in ordered], color=BLUE, label="Observed mean")
    b.plot(x, [row["KS_reference_95_mean"] for row in ordered], "--", color=REFERENCE,
           marker="o", markersize=3, label="Nominal 95% reference")
    b.set_xticks(x, [f"D{row['disease_id']}" for row in ordered])
    b.set_ylabel("Two-sample KS statistic")
    b.legend(frameon=False, fontsize=6)
    _panel(b, "b", "Uncertainty-score comparability")
    c.scatter([row["FDP_cal"] for row in rows], [row["FDP_test"] for row in rows],
              color=BLUE, s=24)
    c.plot([0, 0.3], [0, 0.3], "--", color=REFERENCE, linewidth=0.8)
    c.axhline(suite["meta"]["alpha"], color=REFERENCE, linestyle=":", linewidth=0.8)
    c.axvline(suite["meta"]["alpha"], color=REFERENCE, linestyle=":", linewidth=0.8)
    c.set_xlabel("Calibration FDP")
    c.set_ylabel("Held-out FDP")
    _panel(c, "c", "FDP at calibrated thresholds")
    alphas = [row["alpha"] for row in alpha_rows]
    matrix = np.asarray([[entry["excess_FDP"] for entry in row["exchangeability"]]
                         for row in alpha_rows], dtype=float)
    smallest = min(rows, key=lambda row: row["mean_effective_calibration_size"])["disease_id"]
    _line(d, alphas, np.nanmean(matrix, axis=1), BLUE, "o", "Mean across clinics")
    _line(d, alphas, matrix[:, smallest], ORANGE, "s", f"D{smallest}: smallest stream")
    d.axhline(0, color=REFERENCE, linestyle="--", linewidth=0.8)
    d.set_xscale("log")
    d.xaxis.set_major_formatter(mticker.FuncFormatter(lambda value, _: f"{value:g}"))
    d.set_xlabel(r"Specialist target $\alpha$")
    d.set_ylabel("Held-out FDP − calibration FDP")
    d.legend(frameon=False, fontsize=6)
    _panel(d, "d", "Calibration–test FDP excess")
    for axis in axes.flat:
        _clean(axis)
    _save(fig, directory, "fig10_exchangeability")


def plot_fig11_contamination_alpha(suite: dict[str, Any], directory: Path) -> None:
    rows = suite["alpha_grid"]
    alphas = [row["alpha"] for row in rows]
    rho_route = [np.nanmean([_finite(x["rho_route"]) for x in row["contamination"]]) for row in rows]
    rho_acc = [np.nanmean([_finite(x["rho_acc"]) for x in row["contamination"]]) for row in rows]
    referral_matrix = np.asarray([
        [_finite(x["wrong_route_referral"]) for x in row["contamination"]]
        for row in rows
    ])
    route_errors = [sum(x["route_errors"] for x in row["contamination"]) for row in rows]
    diag_errors = [sum(x["diagnostic_errors"] for x in row["contamination"]) for row in rows]
    ratio_matrix = np.asarray([
        [_finite(
            x["rho_acc"] / x["rho_route"]
            if x["rho_acc"] is not None and x["rho_route"] not in (None, 0)
            else None
        )
         for x in row["contamination"]]
        for row in rows
    ])
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2), constrained_layout=True)
    (a, b), (c, d) = axes
    _line(a, alphas, rho_route, PURPLE, "o", "Before rejection")
    _line(a, alphas, rho_acc, GREEN, "s", "Accepted")
    a.set_ylabel("Wrong-routed fraction")
    a.legend(frameon=False)
    _panel(a, "a", "Pooled contamination")
    operating_index = int(np.argmin(np.abs(np.asarray(alphas) - suite["meta"]["alpha"])))
    strongest = int(np.nanargmax(referral_matrix[operating_index, :]))
    weakest = int(np.nanargmin(referral_matrix[operating_index, :]))
    _line(b, alphas, referral_matrix[:, strongest], GREEN, "o", f"D{strongest}: strongest")
    _line(b, alphas, referral_matrix[:, weakest], ORANGE, "s", f"D{weakest}: weakest")
    b.set_ylabel("Wrong-route referral rate")
    b.legend(frameon=False)
    _panel(b, "b", "Pathway-dependent containment")
    totals = np.asarray(route_errors) + np.asarray(diag_errors)
    route_share = np.divide(
        np.asarray(route_errors), totals,
        out=np.zeros_like(totals, dtype=float), where=totals > 0,
    )
    diagnostic_share = np.divide(
        np.asarray(diag_errors), totals,
        out=np.zeros_like(totals, dtype=float), where=totals > 0,
    )
    c.stackplot(alphas, route_share, diagnostic_share,
                colors=[ORANGE, GRAY], labels=["Routing-induced", "Within-clinic"])
    c.set_xlabel(r"Specialist target $\alpha$")
    c.set_ylabel("Share of accepted errors")
    c.legend(frameon=False)
    _panel(c, "c", "Residual error source")
    for k in range(NUM_DISEASES):
        d.plot(alphas, ratio_matrix[:, k], color="#C4C4C4", linewidth=0.65)
    _line(d, alphas, np.nanmedian(ratio_matrix, axis=1), BLUE, "o", "Clinic median")
    d.axhline(1, color=REFERENCE, linestyle="--", linewidth=0.8)
    d.set_xlabel(r"Specialist target $\alpha$")
    d.set_ylabel(r"$\rho^{acc}/\rho^{route}$")
    d.legend(frameon=False)
    _panel(d, "d", "Relative contamination")
    for axis in axes.flat:
        axis.set_xscale("log")
        axis.xaxis.set_major_formatter(mticker.FuncFormatter(lambda value, _: f"{value:g}"))
        _clean(axis)
    _save(fig, directory, "fig11_contamination_alpha")


def write_full_suite_outputs(
    suite: dict[str, Any],
    primary: dict[str, Any],
    *,
    figure_dir: Path,
    table_dir: Path,
) -> None:
    _style()
    plot_fig1_forest(suite, primary, figure_dir)
    plot_fig2_beta(suite, figure_dir)
    plot_fig3_alpha(suite, figure_dir)
    plot_fig4_baselines(suite, figure_dir)
    plot_fig5_contamination(suite, figure_dir)
    plot_fig6_calibration(suite, figure_dir)
    plot_fig7_utility(suite, figure_dir)
    plot_fig8_frontier(suite, figure_dir)
    plot_fig9_router(suite, figure_dir)
    plot_fig10_exchangeability(suite, figure_dir)
    plot_fig11_contamination_alpha(suite, figure_dir)
    write_tables(suite, primary, table_dir)
    write_manifest(suite, table_dir)
