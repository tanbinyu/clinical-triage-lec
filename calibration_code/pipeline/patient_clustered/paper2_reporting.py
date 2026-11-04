"""Paper2.0-compatible figures for the patient-clustered experiment.

This module contains only presentation logic.  It reads saved metric bundles
and bootstrap replicate rows; it never recalibrates a model or edits source
JSON files.  The filenames deliberately match the LaTeX ``includegraphics``
paths in ``Paper2.0/Clinical_Triage__New_Version_``.
"""
from __future__ import annotations

import csv
import shutil
import tempfile
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from pipeline.patient_level.metadata import DISEASE_NAMES, DISEASE_SHORT_NAMES

from .paper_reporting import (
    COMPARATOR,
    LIGHT,
    PROPOSED,
    PURPLE,
    REFERENCE,
    RISK,
    SECONDARY,
    _clean,
    _finite,
    _line,
    _panel,
    _point,
    _style,
    _validate_subplot_axis_labels,
    plot_calibration_size,
    plot_exchangeability,
    plot_utility,
)


RESULT_FIGURE_PATHS = (
    Path("newfig/fig1_disease_forest.pdf"),
    Path("newfig/fig2_beta_sweep.pdf"),
    Path("newfig/fig3_alpha_sweep.pdf"),
    Path("newfig/fig5_contamination.pdf"),
    Path("newfig/fig6_calibration_size.pdf"),
    Path("newfig/fig7_utility_heatmap.pdf"),
    Path("newfig/fig10_exchangeability.pdf"),
    Path("fig9_router_risk_sweep.pdf"),
    Path("fig13_patient_cluster_bootstrap.pdf"),
)


def _save_pdf(fig: plt.Figure, path: Path) -> None:
    _validate_subplot_axis_labels(fig)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def _log_beta_axes(axes: list[plt.Axes] | np.ndarray) -> None:
    for axis in np.asarray(axes, dtype=object).flat:
        axis.set_xscale("log")
        axis.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda value, _: f"{value:g}")
        )
        _clean(axis)


def plot_pathway_forest(primary: dict[str, Any], path: Path) -> None:
    """Absolute and paired pathway risk comparison."""
    rows = primary["bootstrap"]["per_disease"]
    alpha = primary["meta"]["config"]["alpha"]
    y = np.arange(len(rows))
    fig, axes = plt.subplots(
        1, 2, figsize=(9.4, 4.8), constrained_layout=True,
        gridspec_kw={"width_ratios": (1.35, 1.0)},
    )
    axis, delta_axis = axes
    for position, row in zip(y, rows):
        _point(axis, row["mFDR_sys"], float(position) - 0.12, PROPOSED)
        _point(
            axis, row["specialist_only_mFDR_sys"],
            float(position) + 0.12, COMPARATOR,
        )
    axis.plot([], [], "o", color=PROPOSED, label="Proposed LEC")
    axis.plot([], [], "o", color=COMPARATOR, label="Routed specialist-only")
    axis.axvline(alpha, color=REFERENCE, linestyle="--", linewidth=0.9,
                 label=rf"Target $\alpha={alpha:g}$")
    axis.set_yticks(y, [DISEASE_NAMES[row["disease_id"]] for row in rows])
    axis.invert_yaxis()
    axis.set_xlabel("Accepted-decision error rate")
    axis.set_ylabel("Diagnostic pathway")
    axis.legend(
        frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.14),
        ncol=3, fontsize=6.4,
    )
    _panel(axis, "a", "Pathway-specific risk")
    _clean(axis, "x")

    for position, row in zip(y, rows):
        _point(delta_axis, row["delta_mFDR_sys"], float(position), PROPOSED)
    delta_axis.axvspan(
        delta_axis.get_xlim()[0], 0, color=PROPOSED, alpha=0.05, zorder=0,
    )
    delta_axis.axvline(0, color=REFERENCE, linestyle="--", linewidth=0.9)
    delta_axis.set_yticks(y, [f"D{row['disease_id']}" for row in rows])
    delta_axis.invert_yaxis()
    delta_axis.set_xlabel(
        "mFDR difference\n(Proposed minus comparator)"
    )
    delta_axis.set_ylabel("Diagnostic pathway")
    _panel(delta_axis, "b", "Paired risk difference")
    _clean(delta_axis, "x")
    _save_pdf(fig, path)


def plot_alpha_sweep(suite: dict[str, Any], path: Path) -> None:
    """Direct LEC-versus-specialist-only comparison across risk targets."""
    rows = suite["alpha_grid"]
    alphas = np.asarray([row["alpha"] for row in rows])
    proposed = [row["methods"]["proposed"] for row in rows]
    specialist = [row["methods"]["specialist_only"] for row in rows]
    fig, axes = plt.subplots(2, 2, figsize=(7.3, 5.3), constrained_layout=True)
    (a, b), (c, d) = axes

    _line(a, alphas, [row["mFDR_sys"] for row in proposed],
          PROPOSED, "o", "Proposed LEC")
    _line(a, alphas, [row["mFDR_sys"] for row in specialist],
          COMPARATOR, "s", "Routed specialist-only")
    a.plot(alphas, alphas, ":", color=REFERENCE, linewidth=0.9, label=r"$y=\alpha$")
    a.set_xlabel(r"Reporting target $\alpha$")
    a.set_ylabel("System mFDR")
    a.legend(frameon=False, fontsize=6.2)
    _panel(a, "a", "Pooled accepted-decision risk")

    _line(b, alphas, [row["mFDR_worst"] for row in proposed],
          PROPOSED, "o", "Proposed LEC")
    _line(b, alphas, [row["mFDR_worst"] for row in specialist],
          COMPARATOR, "s", "Routed specialist-only")
    b.plot(alphas, alphas, ":", color=REFERENCE, linewidth=0.9, label=r"$y=\alpha$")
    b.set_xlabel(r"Reporting target $\alpha$")
    b.set_ylabel("Worst-pathway mFDR")
    b.legend(frameon=False, fontsize=6.2)
    _panel(b, "b", "Tail risk")

    _line(c, alphas, [row["coverage"] for row in proposed],
          PROPOSED, "o", "Proposed LEC")
    _line(c, alphas, [row["coverage"] for row in specialist],
          COMPARATOR, "s", "Routed specialist-only")
    c.set_xlabel(r"Reporting target $\alpha$")
    c.set_ylabel("AI-managed coverage")
    c.legend(frameon=False, fontsize=6.2)
    _panel(c, "c", "Automation")

    _line(d, alphas, [row["K_meet"] for row in proposed],
          PROPOSED, "o", "Proposed LEC")
    _line(d, alphas, [row["K_meet"] for row in specialist],
          COMPARATOR, "s", "Routed specialist-only")
    d.set_xlabel(r"Reporting target $\alpha$")
    d.set_ylabel("Pathways at or below target")
    d.set_ylim(-0.3, 10.5)
    d.legend(frameon=False, fontsize=6.2)
    _panel(d, "d", "Empirical pathway attainment")

    for axis in axes.flat:
        axis.set_xscale("log")
        axis.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda value, _: f"{value:g}")
        )
        axis.axvline(suite["meta"]["alpha"], color=REFERENCE, linewidth=0.7)
        _clean(axis)
    _save_pdf(fig, path)


def plot_beta_sweep(suite: dict[str, Any], path: Path) -> None:
    """Risk, coverage, referral and contamination across router targets."""
    rows = suite["beta_grid"]
    beta = np.asarray([row["beta"] for row in rows])
    fig, axes = plt.subplots(2, 2, figsize=(7.3, 5.3), constrained_layout=True)
    (a, b), (c, d) = axes

    _line(a, beta, [row["router_mFDR"] for row in rows], RISK, "s", "Router")
    _line(a, beta, [row["proposed"]["mFDR_sys"] for row in rows],
          PROPOSED, "o", "End-to-end")
    a.plot(beta, beta, ":", color=REFERENCE, linewidth=0.8, label=r"$y=\beta$")
    a.axhline(suite["meta"]["alpha"], color=REFERENCE, linestyle="--", linewidth=0.8)
    a.set_xlabel(r"Router target $\beta$")
    a.set_ylabel("Accepted-decision error rate")
    a.legend(frameon=False)
    _panel(a, "a", "Risk")

    _line(b, beta, [row["router_coverage"] for row in rows],
          SECONDARY, "s", "Router")
    _line(b, beta, [row["proposed"]["coverage"] for row in rows],
          PROPOSED, "o", "End-to-end")
    b.set_xlabel(r"Router target $\beta$")
    b.set_ylabel("Coverage")
    b.legend(frameon=False)
    _panel(b, "b", "Automation")

    _line(c, beta, [row["proposed"]["referral"] for row in rows],
          RISK, "o", "Total")
    _line(c, beta, [row["router_referral"] for row in rows],
          COMPARATOR, "s", "Router stage")
    _line(c, beta, [row["specialist_referral"] for row in rows],
          SECONDARY, "^", "Specialist stage")
    c.set_xlabel(r"Router target $\beta$")
    c.set_ylabel("Human-referral rate")
    c.legend(frameon=False)
    _panel(c, "c", "Human referral")

    lower = np.asarray([_finite(row["rho_route_min"]) for row in rows])
    upper = np.asarray([_finite(row["rho_route_max"]) for row in rows])
    d.fill_between(beta, lower, upper, color=PURPLE, alpha=0.14,
                   label="Pathway range")
    _line(d, beta, [row["rho_route_mean"] for row in rows],
          PURPLE, "o", "Before rejection")
    _line(d, beta, [row["rho_acc_mean"] for row in rows],
          SECONDARY, "s", "Accepted")
    d.set_xlabel(r"Router target $\beta$")
    d.set_ylabel("Wrong-routed fraction")
    d.legend(frameon=False)
    _panel(d, "d", "Routing contamination")

    for axis in axes.flat:
        axis.axvline(suite["meta"]["beta"], color=REFERENCE, linewidth=0.7)
    _log_beta_axes(axes)
    _save_pdf(fig, path)


def plot_router_contribution(suite: dict[str, Any], path: Path) -> None:
    """Incremental contribution of router rejection across beta targets."""
    rows = suite["beta_grid"]
    beta = np.asarray([row["beta"] for row in rows])
    fig, axes = plt.subplots(2, 2, figsize=(7.3, 5.3), constrained_layout=True)
    (a, b), (c, d) = axes

    for key, color, marker, label in (
        ("proposed", PROPOSED, "o", "Proposed LEC"),
        ("specialist_only", COMPARATOR, "s", "Routed specialist-only"),
        ("full_automation", REFERENCE, "^", "Full automation"),
    ):
        _line(a, beta, [row[key]["mFDR_sys"] for row in rows],
              color, marker, label)
    a.axhline(suite["meta"]["alpha"], color=REFERENCE, linestyle="--", linewidth=0.8)
    a.set_xlabel(r"Router target $\beta$")
    a.set_ylabel("System mFDR")
    a.legend(frameon=False)
    _panel(a, "a", "End-to-end risk")

    _line(b, beta, [row["proposed"]["coverage"] for row in rows],
          PROPOSED, "o", "Proposed LEC")
    _line(b, beta, [row["specialist_only"]["coverage"] for row in rows],
          COMPARATOR, "s", "Routed specialist-only")
    b.set_xlabel(r"Router target $\beta$")
    b.set_ylabel("AI-managed coverage")
    b.legend(frameon=False)
    _panel(b, "b", "End-to-end coverage")

    _line(c, beta, [row["router_mFDR"] for row in rows], RISK, "s", "Router mFDR")
    c.plot(beta, beta, ":", color=REFERENCE, linewidth=0.8, label=r"$y=\beta$")
    c.set_xlabel(r"Router target $\beta$")
    c.set_ylabel("Router mFDR")
    c2 = c.twinx()
    _line(c2, beta, [row["router_coverage"] for row in rows],
          SECONDARY, "o", "Router coverage")
    c2.set_ylabel("Router coverage", color=SECONDARY)
    c.legend(frameon=False, loc="upper left")
    c2.legend(frameon=False, loc="lower right")
    _panel(c, "c", "Router operating point")

    delta = [
        row["proposed"]["mFDR_sys"] - row["specialist_only"]["mFDR_sys"]
        for row in rows
    ]
    _line(d, beta, delta, RISK, "o", r"$\Delta$mFDR")
    d.axhline(0, color=REFERENCE, linestyle="--", linewidth=0.8)
    d.set_xlabel(r"Router target $\beta$")
    d.set_ylabel("Proposed minus routed specialist-only mFDR")
    d2 = d.twinx()
    _line(d2, beta, [row["router_coverage"] for row in rows],
          SECONDARY, "s", "Router coverage")
    d2.set_ylabel("Router coverage", color=SECONDARY)
    d.legend(frameon=False, loc="upper left")
    d2.legend(frameon=False, loc="lower right")
    _panel(d, "d", "Incremental router contribution")

    for axis in axes.flat:
        axis.axvline(suite["meta"]["beta"], color=REFERENCE, linewidth=0.7)
    _log_beta_axes(axes)
    for twin in (c2, d2):
        twin.set_xscale("log")
        twin.grid(False)
    _save_pdf(fig, path)


def plot_contamination(suite: dict[str, Any], path: Path) -> None:
    """Routing-contamination mechanism and accepted-risk consequences."""
    rows = suite["primary"]["contamination"]
    labels = [DISEASE_SHORT_NAMES[row["disease_id"]] for row in rows]
    y = np.arange(len(rows))
    fig, axes = plt.subplots(1, 3, figsize=(11.6, 4.8), constrained_layout=True)
    a, b, c = axes
    width = 0.35
    a.barh(y - width / 2, [_finite(row["rho_route"]) for row in rows], width,
           color=PURPLE, label="Before specialist rejection")
    a.barh(y + width / 2, [_finite(row["rho_acc"]) for row in rows], width,
           color=SECONDARY, label="Among accepted decisions")
    a.set_yticks(y, labels)
    a.invert_yaxis()
    a.set_xlabel("Wrong-routed fraction")
    a.set_ylabel("Diagnostic pathway")
    a.legend(
        frameon=False, fontsize=6.2, loc="upper center",
        bbox_to_anchor=(0.5, -0.14), ncol=1,
    )
    _panel(a, "a", "Routing contamination")

    proposed_paths = suite["primary"]["methods"]["proposed"]["per_disease"]
    specialist_paths = suite["primary"]["methods"]["specialist_only"]["per_disease"]
    b.barh(y - width / 2, [_finite(row["mFDR"]) for row in proposed_paths],
           width, color=PROPOSED, label="Proposed LEC")
    b.barh(y + width / 2, [_finite(row["mFDR"]) for row in specialist_paths],
           width, color=COMPARATOR, label="Routed specialist-only")
    b.set_yticks(y, labels)
    b.invert_yaxis()
    b.set_xlabel("Accepted-decision error rate")
    b.set_ylabel("Diagnostic pathway")
    b.legend(
        frameon=False, fontsize=6.2, loc="upper center",
        bbox_to_anchor=(0.5, -0.14), ncol=2,
    )
    _panel(b, "b", "Risk comparator")

    route = np.asarray([_finite(row["route_error_share"]) for row in rows])
    diagnostic = np.asarray([_finite(row["diagnostic_error_share"]) for row in rows])
    c.barh(y, route, color=RISK, label="Routing-induced")
    c.barh(y, diagnostic, left=route, color=COMPARATOR, label="Within-pathway")
    c.set_yticks(y, labels)
    c.invert_yaxis()
    c.set_xlabel("Share of accepted errors")
    c.set_ylabel("Diagnostic pathway")
    c.legend(
        frameon=False, fontsize=6.2, loc="upper left",
        bbox_to_anchor=(1.01, 1.0), borderaxespad=0,
    )
    _panel(c, "c", "Error sources")
    for axis in axes:
        _clean(axis, "x")
    _save_pdf(fig, path)


def _bootstrap_rows(path: Path) -> dict[str, np.ndarray]:
    columns = (
        "proposed_mFDR_sys", "specialist_only_mFDR_sys", "delta_mFDR_sys",
        "proposed_coverage", "specialist_only_coverage", "delta_coverage",
        "proposed_mFDR_worst", "specialist_only_mFDR_worst",
        "delta_mFDR_worst", "proposed_K_meet", "specialist_only_K_meet",
        "delta_K_meet",
    )
    values = {column: [] for column in columns}
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            for column in columns:
                if row.get(column) not in (None, ""):
                    values[column].append(float(row[column]))
    if any(not values[column] for column in columns):
        missing = [column for column in columns if not values[column]]
        raise ValueError(f"Bootstrap CSV has no values for: {', '.join(missing)}")
    return {key: np.asarray(value) for key, value in values.items()}


def plot_paired_bootstrap(
    primary: dict[str, Any], bootstrap_csv: Path, path: Path
) -> None:
    """Paired patient-cluster bootstrap comparison used in the primary results."""
    values = _bootstrap_rows(bootstrap_csv)
    fig, axes = plt.subplots(2, 2, figsize=(7.3, 5.3), constrained_layout=True)
    (a, b), (c, d) = axes

    for axis, metric, letter, title in (
        (a, "mFDR_sys", "a", "Overall accepted-decision risk"),
        (b, "coverage", "b", "AI-managed coverage"),
        (c, "mFDR_worst", "c", "Worst-pathway risk"),
    ):
        for y, (method, color) in enumerate((
            ("proposed", PROPOSED), ("specialist_only", COMPARATOR)
        )):
            _point(axis, primary["bootstrap"][method][metric], y, color)
        axis.set_yticks([0, 1], ["Proposed LEC", "Routed specialist-only"])
        axis.invert_yaxis()
        axis.set_xlabel("Proportion")
        axis.set_ylabel("Method")
        _panel(axis, letter, title)
        _clean(axis, "x")
    a.axvline(primary["meta"]["config"]["alpha"], color=REFERENCE,
              linestyle="--", linewidth=0.8)
    c.axvline(primary["meta"]["config"]["alpha"], color=REFERENCE,
              linestyle="--", linewidth=0.8)

    column = "delta_mFDR_sys"
    d.hist(values[column], bins=28, color=PROPOSED, alpha=0.82,
           edgecolor="white", linewidth=0.35)
    d.axvline(0, color=REFERENCE, linestyle="--", linewidth=0.9)
    d.axvline(float(np.mean(values[column])), color="#222222", linewidth=1.0)
    d.set_xlabel("Proposed minus routed specialist-only system mFDR")
    d.set_ylabel("Bootstrap replicates")
    d.xaxis.set_major_locator(mticker.MaxNLocator(5))
    d.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    _panel(d, "d", "Paired risk difference")
    _clean(d, "x")
    _save_pdf(fig, path)


def _copy_rendered(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def render_paper2_result_figures(
    suite: dict[str, Any],
    primary: dict[str, Any],
    *,
    patient_balanced: dict[str, Any] | None,
    strict_patient: dict[str, Any] | None,
    bootstrap_csv: Path,
    figure_root: Path,
) -> list[Path]:
    """Render the nine data-derived PDFs referenced by Paper2.0."""
    _style()
    destinations = {path.name: figure_root / path for path in RESULT_FIGURE_PATHS}
    plot_pathway_forest(primary, destinations["fig1_disease_forest.pdf"])
    plot_beta_sweep(suite, destinations["fig2_beta_sweep.pdf"])
    plot_alpha_sweep(suite, destinations["fig3_alpha_sweep.pdf"])
    plot_router_contribution(suite, destinations["fig9_router_risk_sweep.pdf"])
    plot_contamination(suite, destinations["fig5_contamination.pdf"])
    plot_paired_bootstrap(
        primary, bootstrap_csv, destinations["fig13_patient_cluster_bootstrap.pdf"]
    )

    # Reuse the general patient-clustered implementations for the four figures
    # whose panel contracts already match the Paper2.0 manuscript.
    with tempfile.TemporaryDirectory() as temporary:
        temp = Path(temporary)
        plot_calibration_size(suite, temp)
        plot_exchangeability(suite, temp)
        plot_utility(primary, patient_balanced, strict_patient, temp)
        sources = {
            "fig6_calibration_size.pdf": temp / "figS1_patient_clustered_calibration_size.pdf",
            "fig10_exchangeability.pdf": temp / "figS2_patient_clustered_exchangeability.pdf",
            "fig7_utility_heatmap.pdf": temp / "figS3_patient_clustered_utility.pdf",
        }
        for name, source in sources.items():
            _copy_rendered(source, destinations[name])
    return [figure_root / path for path in RESULT_FIGURE_PATHS]
