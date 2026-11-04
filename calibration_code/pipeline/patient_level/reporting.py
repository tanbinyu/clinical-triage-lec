"""Primary patient-level figures and report orchestration."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from .primary_tables import write_summary, write_tables


PROPOSED = "#0072B2"
COMPARATOR = "#4D4D4D"
RISK = "#D55E00"
SECONDARY = "#009E73"
ACCENT = "#CC79A7"
REFERENCE = "#9A9A9A"


def _style() -> None:
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 8.5,
        "axes.labelsize": 9,
        "axes.titlesize": 9.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
    })


def _clean_axis(axis: plt.Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="x", color="#E8E8E8", linewidth=0.6, zorder=0)
    axis.tick_params(width=0.7, length=3)


def _ci(summary: dict[str, Any]) -> tuple[float, float, float]:
    return (
        float(summary["reference_pooled"]),
        float(summary["ci_low"]),
        float(summary["ci_high"]),
    )


def _interval_point(
    axis: plt.Axes,
    *,
    estimate: float,
    low: float,
    high: float,
    position: float,
    color: str,
    markersize: float = 6,
) -> None:
    """Draw a percentile interval and an independent pooled point estimate."""
    axis.hlines(position, low, high, color=color, linewidth=1.4, zorder=2)
    axis.vlines([low, high], position - 0.07, position + 0.07,
                color=color, linewidth=1.0, zorder=2)
    axis.plot(estimate, position, "o", color=color, markersize=markersize, zorder=3)


def _save(fig: plt.Figure, directory: Path, stem: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory / f"{stem}.pdf")
    fig.savefig(directory / f"{stem}.png", dpi=400)
    plt.close(fig)


def plot_primary(result: dict[str, Any], figure_dir: Path) -> None:
    summary = result["bootstrap"]
    alpha = result["meta"]["config"]["alpha"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    methods = ["Proposed LEC", "Specialist-only"]
    colors = [PROPOSED, COMPARATOR]
    keys = ["proposed", "specialist_only"]
    for panel, metric, title in (
        (0, "mFDR_sys", "Accepted-case error rate"),
        (1, "coverage", "AI-managed coverage"),
    ):
        axis = axes[panel]
        for position, (key, color) in enumerate(zip(keys, colors)):
            estimate, low, high = _ci(summary[key][metric])
            _interval_point(
                axis, estimate=estimate, low=low, high=high,
                position=position, color=color,
            )
        axis.set_yticks(range(2), methods)
        axis.invert_yaxis()
        axis.set_xlabel("Proportion")
        axis.set_title(f"{'ab'[panel]}  {title}", loc="left", fontweight="bold")
        if metric == "mFDR_sys":
            axis.axvline(alpha, color=REFERENCE, linestyle="--", linewidth=1.0,
                         label=rf"Target $\alpha={alpha:g}$")
            axis.legend(frameon=False, loc="upper right")
        _clean_axis(axis)
    fig.suptitle(
        "Patient-level performance with 95% stratified cluster-bootstrap intervals",
        y=1.02,
        fontsize=10,
    )
    fig.tight_layout()
    _save(fig, figure_dir, "fig_patient_primary")


def plot_paired_effects(result: dict[str, Any], figure_dir: Path) -> None:
    paired = result["bootstrap"]["paired_difference"]
    labels = [r"$\Delta$ mFDR", r"$\Delta$ coverage"]
    metrics = ["delta_mFDR_sys", "delta_coverage"]
    colors = [RISK, PROPOSED]
    fig, axis = plt.subplots(figsize=(4.8, 2.6))
    for position, (metric, color) in enumerate(zip(metrics, colors)):
        estimate, low, high = _ci(paired[metric])
        _interval_point(
            axis, estimate=estimate, low=low, high=high,
            position=position, color=color,
        )
    axis.axvline(0, color=REFERENCE, linestyle="--", linewidth=1.0)
    axis.set_yticks(range(2), labels)
    axis.invert_yaxis()
    axis.set_xlabel("Proposed LEC − specialist-only")
    axis.set_title("Paired patient-bootstrap differences", loc="left", fontweight="bold")
    _clean_axis(axis)
    fig.tight_layout()
    _save(fig, figure_dir, "fig_patient_paired_effects")


def plot_disease_forest(result: dict[str, Any], figure_dir: Path) -> None:
    rows = result["bootstrap"]["per_disease"]
    alpha = result["meta"]["config"]["alpha"]
    labels = [row["disease_name"] for row in rows]
    y = np.arange(len(rows))
    fig, axes = plt.subplots(
        1, 2, figsize=(8.4, 4.8), sharey=True,
        gridspec_kw={"width_ratios": [1.05, 1]},
    )
    for axis, metric, title in (
        (axes[0], "mFDR_sys", "Clinic-specific mFDR"),
        (axes[1], "coverage", "Clinic-specific coverage"),
    ):
        for position, row in zip(y, rows):
            summary = row[metric]
            if summary["reference_pooled"] is None:
                continue
            estimate, low, high = _ci(summary)
            _interval_point(
                axis, estimate=estimate, low=low, high=high,
                position=position, color=PROPOSED, markersize=4.5,
            )
        axis.set_xlabel("Proportion")
        axis.set_title(title, loc="left", fontweight="bold")
        _clean_axis(axis)
    axes[0].axvline(alpha, color=REFERENCE, linestyle="--", linewidth=1.0,
                    label=rf"Target $\alpha={alpha:g}$")
    axes[0].legend(frameon=False, loc="upper right")
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    fig.suptitle("Patient-level specialist-pathway performance", y=1.01, fontsize=10)
    fig.tight_layout()
    _save(fig, figure_dir, "fig_patient_disease_forest")


def plot_sensitivity(result: dict[str, Any], figure_dir: Path) -> None:
    rows = result.get("aggregation_sensitivity", [])
    if not rows:
        return
    labels = [row["aggregation"].capitalize() for row in rows]
    x = np.arange(len(rows))
    width = 0.34
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for axis, metric, title in (
        (axes[0], "mFDR_sys", "Accepted-case error rate"),
        (axes[1], "coverage", "AI-managed coverage"),
    ):
        proposed = [row[f"proposed_{metric}"] for row in rows]
        comparator = [row[f"specialist_only_{metric}"] for row in rows]
        axis.bar(x - width / 2, proposed, width, color=PROPOSED, label="Proposed LEC")
        axis.bar(x + width / 2, comparator, width, color=COMPARATOR,
                 label="Specialist-only")
        axis.set_xticks(x, labels)
        axis.set_ylabel("Proportion")
        axis.set_title(title, loc="left", fontweight="bold")
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.grid(axis="y", color="#E8E8E8", linewidth=0.6, zorder=0)
    axes[0].axhline(
        result["meta"]["config"]["alpha"], color=REFERENCE,
        linestyle="--", linewidth=1.0,
    )
    axes[1].legend(frameon=False, loc="best")
    fig.suptitle("Sensitivity to within-patient uncertainty aggregation", y=1.02, fontsize=10)
    fig.tight_layout()
    _save(fig, figure_dir, "fig_patient_aggregation_sensitivity")


def write_all_reports(
    result: dict[str, Any], *, figures_dir: Path, tables_dir: Path
) -> None:
    _style()
    plot_primary(result, figures_dir)
    plot_paired_effects(result, figures_dir)
    plot_disease_forest(result, figures_dir)
    plot_sensitivity(result, figures_dir)
    write_tables(result, tables_dir)
    write_summary(result, tables_dir)
