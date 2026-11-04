"""Figures and tables for patient-clustered mixed-scope analysis."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROPOSED = "#0072B2"
COMPARATOR = "#4D4D4D"
RISK = "#D55E00"
SECONDARY = "#009E73"
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


def _save(figure: plt.Figure, directory: Path, stem: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    figure.savefig(directory / f"{stem}.pdf")
    figure.savefig(directory / f"{stem}.png", dpi=400)
    plt.close(figure)


def _clean(axis: plt.Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="x", color="#E8E8E8", linewidth=0.6, zorder=0)


def _point(axis: plt.Axes, summary: dict[str, Any], y: float, color: str) -> None:
    estimate = summary["reference_pooled"]
    low = summary["ci_low"]
    high = summary["ci_high"]
    if estimate is None or low is None or high is None:
        return
    axis.hlines(y, low, high, color=color, linewidth=1.4, zorder=2)
    axis.vlines([low, high], y - 0.07, y + 0.07, color=color, linewidth=1.0)
    axis.plot(estimate, y, "o", color=color, markersize=5.5, zorder=3)


def _fmt(value: float | None, digits: int = 3) -> str:
    return "NA" if value is None else f"{value:.{digits}f}"


def _interval(summary: dict[str, Any]) -> str:
    return (
        f"{_fmt(summary['reference_pooled'])} "
        f"[{_fmt(summary['ci_low'])}, {_fmt(summary['ci_high'])}]"
    )


def _analysis_label(result: dict[str, Any]) -> str:
    meta = result.get("meta", {})
    unit = meta.get("primary_analysis_unit") or meta.get("analysis", "")
    if "patient_balanced" in unit or "patient-balanced" in unit:
        return "One decision per patient"
    if "all_decisions" in unit or "all-decision" in unit:
        return "All decisions (patient clustered)"
    return str(meta.get("unit_of_analysis", "Analysis"))


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot_primary(result: dict[str, Any], directory: Path) -> None:
    bootstrap = result["bootstrap"]
    alpha = result["meta"]["config"]["alpha"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for panel, metric, title in (
        (0, "mFDR_sys", "Accepted-decision error rate"),
        (1, "coverage", "AI-managed decision coverage"),
    ):
        axis = axes[panel]
        for y, (key, color) in enumerate((
            ("proposed", PROPOSED), ("specialist_only", COMPARATOR)
        )):
            _point(axis, bootstrap[key][metric], y, color)
        axis.set_yticks([0, 1], ["Proposed LEC", "Specialist-only"])
        axis.invert_yaxis()
        axis.set_xlabel("Proportion")
        axis.set_ylabel("Method")
        axis.set_title(f"{'ab'[panel]}  {title}", loc="left", fontweight="bold")
        if metric == "mFDR_sys":
            axis.axvline(alpha, color=REFERENCE, linestyle="--", linewidth=1.0,
                         label=rf"Target $\alpha={alpha:g}$")
            axis.legend(frameon=False, loc="upper right")
        _clean(axis)
    fig.suptitle(
        "Patient-clustered decision performance with 95% cluster-bootstrap intervals",
        y=1.02, fontsize=10,
    )
    fig.tight_layout()
    _save(fig, directory, "fig_clustered_primary")


def _plot_disease(result: dict[str, Any], directory: Path) -> None:
    rows = result["bootstrap"]["per_disease"]
    alpha = result["meta"]["config"]["alpha"]
    y = np.arange(len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.8), sharey=True)
    for axis, metric, title in (
        (axes[0], "mFDR_sys", "Clinic-specific mFDR"),
        (axes[1], "coverage", "Clinic-specific coverage"),
    ):
        for position, row in zip(y, rows):
            _point(axis, row[metric], float(position), PROPOSED)
        axis.set_xlabel("Proportion")
        axis.set_ylabel("Diagnostic pathway")
        axis.set_title(title, loc="left", fontweight="bold")
        _clean(axis)
    axes[0].axvline(alpha, color=REFERENCE, linestyle="--", linewidth=1.0,
                    label=rf"Target $\alpha={alpha:g}$")
    axes[0].legend(frameon=False)
    axes[0].set_yticks(y, [row["disease_name"] for row in rows])
    axes[0].invert_yaxis()
    fig.suptitle("Patient-clustered performance across ten decision pathways", y=1.01)
    fig.tight_layout()
    _save(fig, directory, "fig_clustered_disease_forest")


def _plot_robustness(
    primary: dict[str, Any],
    balanced: dict[str, Any] | None,
    strict: dict[str, Any] | None,
    directory: Path,
) -> None:
    analyses = [(_analysis_label(primary).replace(" per ", "\nper "), primary, PROPOSED)]
    if balanced:
        analyses.append((_analysis_label(balanced).replace(" per ", "\nper "), balanced, SECONDARY))
    if strict:
        analyses.append(("Strict patient/\nvolume subset", strict, COMPARATOR))
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2))
    for panel, metric, title in (
        (0, "mFDR_sys", "Proposed-system mFDR"),
        (1, "coverage", "Proposed-system coverage"),
    ):
        axis = axes[panel]
        for y, (_, result, color) in enumerate(analyses):
            _point(axis, result["bootstrap"]["proposed"][metric], y, color)
        axis.set_yticks(range(len(analyses)), [label for label, _, _ in analyses])
        axis.invert_yaxis()
        axis.set_xlabel("Proportion")
        axis.set_ylabel("Analysis unit")
        axis.set_title(title, loc="left", fontweight="bold")
        _clean(axis)
    axes[0].axvline(
        primary["meta"]["config"]["alpha"], color=REFERENCE,
        linestyle="--", linewidth=1.0,
    )
    fig.suptitle("Robustness across prespecified analysis units", y=1.02, fontsize=10)
    fig.tight_layout()
    _save(fig, directory, "fig_analysis_unit_robustness")


def _write_primary_tables(
    result: dict[str, Any],
    balanced: dict[str, Any] | None,
    strict: dict[str, Any] | None,
    directory: Path,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, analysis in (
        (_analysis_label(result), result),
        (_analysis_label(balanced), balanced) if balanced else ("", None),
        ("Strict patient/volume subset", strict),
    ):
        if analysis is None:
            continue
        for method, key in (("Proposed LEC", "proposed"), ("Specialist-only", "specialist_only")):
            rows.append({
                "analysis": name,
                "method": method,
                "mFDR": analysis["bootstrap"][key]["mFDR_sys"]["reference_pooled"],
                "mFDR_ci_low": analysis["bootstrap"][key]["mFDR_sys"]["ci_low"],
                "mFDR_ci_high": analysis["bootstrap"][key]["mFDR_sys"]["ci_high"],
                "coverage": analysis["bootstrap"][key]["coverage"]["reference_pooled"],
                "coverage_ci_low": analysis["bootstrap"][key]["coverage"]["ci_low"],
                "coverage_ci_high": analysis["bootstrap"][key]["coverage"]["ci_high"],
            })
    _write_rows(directory / "analysis_unit_summary.csv", rows)
    tex = [
        r"\begin{table*}[!t]", r"\centering",
        r"\caption{Primary patient-clustered decision analysis and prespecified patient-balanced validation analyses.}",
        r"\label{tab:clustered_primary}",
        r"\begin{tabular}{llcc}", r"\toprule",
        r"Analysis unit & Method & System mFDR (95\% CI) & Coverage (95\% CI) \\",
        r"\midrule",
    ]
    for name, analysis in (
        (_analysis_label(result), result),
        (_analysis_label(balanced), balanced) if balanced else ("", None),
        ("Strict patient/volume subset", strict),
    ):
        if analysis is None:
            continue
        safe_name = name.replace("/", r"/")
        for method, key in (("Proposed LEC", "proposed"), ("Specialist-only", "specialist_only")):
            tex.append(
                f"{safe_name} & {method} & "
                f"{_interval(analysis['bootstrap'][key]['mFDR_sys'])} & "
                f"{_interval(analysis['bootstrap'][key]['coverage'])} \\\\"
            )
    tex.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    (directory / "clustered_primary.tex").write_text("\n".join(tex), encoding="utf-8")


def _write_cohort_table(result: dict[str, Any], directory: Path) -> None:
    sources = result["input_audit"]["sources"]
    rows = [
        {"source": source, **values}
        for source, values in sorted(sources.items())
    ]
    flat_rows = []
    for row in rows:
        flat_rows.append({
            "source": row["source"],
            "disease_id": row["disease_id"],
            "patients": row["patients"],
            "decisions": row["decisions"],
            "multi_record_patients": row["multi_record_patients"],
            "heterogeneous_label_patients": row["heterogeneous_label_patients"],
            "label_scopes": ";".join(
                f"{key}:{value}" for key, value in row["label_scope_counts"].items()
            ),
        })
    _write_rows(directory / "cohort_by_scope.csv", flat_rows)
    tex = [
        r"\begin{table*}[!t]", r"\centering",
        r"\caption{Patient clusters, dataset-defined decisions and label scopes in the immutable evaluation input.}",
        r"\label{tab:clustered_cohort}",
        r"\begin{tabular}{lrrrrl}", r"\toprule",
        r"Source & Patients & Decisions & Multi-record & Heterogeneous truth & Label scope \\",
        r"\midrule",
    ]
    for row in flat_rows:
        source = row["source"].replace("_", r"\_")
        scopes = row["label_scopes"].replace("_", r"\_")
        tex.append(
            f"{source} & {row['patients']:,} & {row['decisions']:,} & "
            f"{row['multi_record_patients']:,} & {row['heterogeneous_label_patients']:,} & "
            f"{scopes} \\\\"
        )
    tex.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    (directory / "cohort_by_scope.tex").write_text("\n".join(tex), encoding="utf-8")


def _write_summary(
    primary: dict[str, Any],
    balanced: dict[str, Any] | None,
    strict: dict[str, Any] | None,
    directory: Path,
) -> None:
    audit = primary["input_audit"]
    lines = [
        "# Patient-clustered mixed-scope experiment",
        "",
        f"- Authoritative decisions: {audit['n_decisions']:,}",
        f"- Dataset-provided patient IDs: {audit['n_patients']:,}",
        f"- Decision scopes: {audit['decision_scope_counts']}",
        "- No within-patient truth labels were merged or overwritten.",
        "- Patients are disjoint across router calibration, specialist calibration and test.",
        "- Bootstrap sampling unit: patient cluster.",
        "",
        "The primary estimands give every patient equal weight by drawing one genuine "
        "dataset-defined decision per patient in each split. Patient IDs prevent leakage; "
        "record-scoped labels are not relabelled as patient diagnoses.",
    ]
    if balanced:
        lines.append(
            "The complementary sensitivity analysis is: " + _analysis_label(balanced) + "."
        )
    if strict:
        sources = ", ".join(strict["input_audit"]["eligible_sources"])
        lines.append(
            "The strict patient/volume validation contains only: " + sources + "."
        )
    lines.append("")
    (directory / "README.md").write_text("\n".join(lines), encoding="utf-8")


def write_reports(
    result: dict[str, Any],
    *,
    patient_balanced: dict[str, Any] | None,
    strict_patient: dict[str, Any] | None,
    figures_dir: Path,
    tables_dir: Path,
) -> None:
    _style()
    _plot_primary(result, figures_dir)
    _plot_disease(result, figures_dir)
    _plot_robustness(result, patient_balanced, strict_patient, figures_dir)
    _write_primary_tables(result, patient_balanced, strict_patient, tables_dir)
    _write_cohort_table(result, tables_dir)
    _write_summary(result, patient_balanced, strict_patient, tables_dir)
