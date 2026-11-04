"""Publication figures and tables for the patient-clustered experiment suite."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from pipeline.patient_level.metadata import DISEASE_NAMES, DISEASE_SHORT_NAMES
from pipeline.patient_level.suite import METHOD_LABELS, METHODS


PROPOSED = "#0072B2"
COMPARATOR = "#4D4D4D"
RISK = "#D55E00"
SECONDARY = "#009E73"
PURPLE = "#CC79A7"
REFERENCE = "#9A9A9A"
LIGHT = "#E8E8E8"
METHOD_COLORS = {
    "full_automation": "#9A9A9A",
    "router_only": SECONDARY,
    "specialist_only": COMPARATOR,
    "shared_threshold": PURPLE,
    "oracle_clean": "#E69F00",
    "proposed": PROPOSED,
}
METHOD_MARKERS = {
    "full_automation": "X",
    "router_only": "^",
    "specialist_only": "s",
    "shared_threshold": "D",
    "oracle_clean": "v",
    "proposed": "o",
}

SOURCE_METADATA = {
    "LungHist700": ("Pathology", "UNI", "Lung cancer"),
    "BreakHis": ("Pathology", "UNI", "Breast cancer"),
    "COVID_CT": ("CT", "DINOv2-large", "Pneumonia/COVID-19"),
    "LIDC_IDRI": ("CT", "DINOv2-large", "Lung nodule"),
    "LDFC": ("CT", "DINOv2-large", "Pulmonary nodule malignancy"),
    "ImageCHD": ("CT", "DINOv2-large", "Congenital heart disease"),
    "ChestX_ray14": ("X-ray", "RAD-DINO", "Thoracic findings"),
    "CBIS-DDSM": ("X-ray", "RAD-DINO", "Breast-lesion malignancy (mammography)"),
    "BUS-BRA": ("Ultrasound", "USFM", "Breast cancer"),
    "PU2756": ("Ultrasound", "USFM", "Pulmonary-tumour malignancy"),
    "ACDC": ("MRI", "CineMA", "Cardiac diagnosis"),
}

SOURCE_CLASS_NAMES = {
    "LungHist700": ("Adenocarcinoma", "squamous-cell carcinoma", "normal"),
    "BreakHis": ("Benign", "malignant"),
    "COVID_CT": ("Non-COVID", "COVID-19"),
    "LDFC": ("Benign", "malignant"),
    "ImageCHD": (
        "ASD", "AVSD", "VSD", "TOF", "PDA", "TGA", "CA", "PuA",
        "PAS", "DORV", "CAT", "DAA", "APVC", "AAH", "IAA", "DSVC",
    ),
    "ChestX_ray14": (
        "Atelectasis", "cardiomegaly", "effusion", "infiltration", "mass",
        "nodule", "pneumonia", "pneumothorax", "consolidation", "edema",
        "emphysema", "fibrosis", "pleural thickening", "hernia",
    ),
    "CBIS-DDSM": ("Benign", "malignant"),
    "BUS-BRA": ("Benign", "malignant"),
    "PU2756": ("Benign", "malignant"),
    "ACDC": ("NOR", "MINF", "DCM", "HCM", "RV"),
}


def _style() -> None:
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 8.2,
        "axes.labelsize": 8.7,
        "axes.titlesize": 9.2,
        "xtick.labelsize": 7.7,
        "ytick.labelsize": 7.7,
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


def _validate_subplot_axis_labels(fig: plt.Figure) -> None:
    """Fail fast when a manuscript subplot lacks either axis label."""
    missing = []
    for index, axis in enumerate(fig.axes, start=1):
        # Matplotlib colorbars and secondary twinx axes are annotations of a
        # primary subplot, not independent panels.
        if axis.get_label() == "<colorbar>" or not axis.patch.get_visible():
            continue
        absent = []
        if not axis.get_xlabel().strip():
            absent.append("x")
        if not axis.get_ylabel().strip():
            absent.append("y")
        if absent:
            missing.append(f"axis {index}: {'/'.join(absent)}")
    if missing:
        raise ValueError(
            "Every manuscript subplot must have x/y labels; missing "
            + ", ".join(missing)
        )


def _line(axis: plt.Axes, x, y, color, marker, label, **kwargs) -> None:
    axis.plot(
        x, y, color=color, marker=marker, markersize=3.8,
        linewidth=1.25, label=label, zorder=3, **kwargs,
    )


def _save(fig: plt.Figure, directory: Path, stem: str) -> None:
    _validate_subplot_axis_labels(fig)
    directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory / f"{stem}.pdf")
    fig.savefig(directory / f"{stem}.png", dpi=400)
    plt.close(fig)


def _point(axis: plt.Axes, summary: dict[str, Any], y: float, color: str) -> None:
    estimate = summary["reference_pooled"]
    low, high = summary["ci_low"], summary["ci_high"]
    if estimate is None or low is None or high is None:
        return
    axis.hlines(y, low, high, color=color, linewidth=1.35, zorder=2)
    axis.vlines([low, high], y - 0.07, y + 0.07, color=color, linewidth=0.9)
    axis.plot(estimate, y, "o", color=color, markersize=5.0, zorder=3)


def _finite(value: float | None) -> float:
    return np.nan if value is None else float(value)


def _fmt(value: float | None, digits: int = 3) -> str:
    return "NA" if value is None else f"{value:.{digits}f}"


def _interval(summary: dict[str, Any]) -> str:
    return (
        f"{_fmt(summary['reference_pooled'])} "
        f"[{_fmt(summary['ci_low'])}, {_fmt(summary['ci_high'])}]"
    )


def _latex(value: str) -> str:
    return value.replace("_", r"\_").replace("—", "--")


def _analysis_label(result: dict[str, Any], *, multiline: bool = False) -> str:
    """Return a truthful display label from metric-bundle metadata."""
    meta = result.get("meta", {})
    unit = meta.get("primary_analysis_unit") or meta.get("analysis", "")
    if "patient_balanced" in unit or "patient-balanced" in unit:
        label = "One decision per patient"
    elif "all_decisions" in unit or "all-decision" in unit:
        label = "All decisions (patient clustered)"
    elif "strict" in unit:
        label = "Strict patient/volume subset"
    else:
        label = str(meta.get("unit_of_analysis", "Analysis"))
    return label.replace(" per ", "\nper ") if multiline else label


def _calibration_note(suite: dict[str, Any]) -> str:
    report = float(suite["meta"]["alpha"])
    fitted = float(suite["meta"].get("calibration_alpha", report))
    if np.isclose(report, fitted):
        return rf"$\alpha={report:.2f}$"
    return (
        rf"reporting target $\alpha={report:.2f}$ and configured "
        rf"calibration level $\alpha_{{cal}}={fitted:.2f}$"
    )


def _target_text(value: float) -> str:
    """Format operating targets without rounding 0.005 to 0.01."""
    return f"{float(value):.3f}".rstrip("0").rstrip(".")


def _source_sampling_note(suite: dict[str, Any]) -> str:
    cap = suite["meta"].get("max_patients_per_source")
    if cap is None:
        return ""
    return " Source-balanced patient sampling was applied before calibration/test role assignment."


def plot_primary_and_robustness(
    primary: dict[str, Any],
    balanced: dict[str, Any] | None,
    strict: dict[str, Any] | None,
    directory: Path,
) -> None:
    analyses = [(_analysis_label(primary, multiline=True), primary, PROPOSED)]
    if balanced:
        analyses.append((_analysis_label(balanced, multiline=True), balanced, SECONDARY))
    if strict:
        analyses.append(("Strict patient/\nvolume subset", strict, COMPARATOR))
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.4), constrained_layout=True)
    for col, metric, title in (
        (0, "mFDR_sys", "Accepted-decision error rate"),
        (1, "coverage", "AI-managed decision coverage"),
    ):
        top = axes[0, col]
        for y, (method, color) in enumerate((
            ("proposed", PROPOSED), ("specialist_only", COMPARATOR)
        )):
            _point(top, primary["bootstrap"][method][metric], y, color)
        top.set_yticks([0, 1], ["Proposed LEC", "Routed specialist-only"])
        top.invert_yaxis()
        top.set_xlabel("Proportion")
        top.set_ylabel("Method")
        _panel(top, "ab"[col], title)
        _clean(top, "x")

        bottom = axes[1, col]
        for y, (_, result, color) in enumerate(analyses):
            _point(bottom, result["bootstrap"]["proposed"][metric], y, color)
        bottom.set_yticks(range(len(analyses)), [label for label, _, _ in analyses])
        bottom.invert_yaxis()
        bottom.set_xlabel("Proportion")
        bottom.set_ylabel("Analysis unit")
        _panel(bottom, "cd"[col], f"Robustness of {title.lower()}")
        _clean(bottom, "x")
    alpha = primary["meta"]["config"]["alpha"]
    axes[0, 0].axvline(alpha, color=REFERENCE, linestyle="--", linewidth=0.9)
    axes[1, 0].axvline(alpha, color=REFERENCE, linestyle="--", linewidth=0.9)
    _save(fig, directory, "fig1_patient_clustered_primary")


def plot_pathway_forest(primary: dict[str, Any], directory: Path) -> None:
    rows = primary["bootstrap"]["per_disease"]
    alpha = primary["meta"]["config"]["alpha"]
    y = np.arange(len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.8), sharey=True, constrained_layout=True)
    for axis, metric, title in (
        (axes[0], "mFDR_sys", "Clinic-specific mFDR"),
        (axes[1], "coverage", "Clinic-specific coverage"),
    ):
        for position, row in zip(y, rows):
            _point(axis, row[metric], float(position), PROPOSED)
        axis.set_xlabel("Proportion")
        axis.set_ylabel("Diagnostic pathway")
        axis.set_title(title, loc="left", fontweight="bold")
        _clean(axis, "x")
    axes[0].axvline(alpha, color=REFERENCE, linestyle="--", linewidth=0.9)
    axes[0].set_yticks(y, [DISEASE_NAMES[row["disease_id"]] for row in rows])
    axes[0].invert_yaxis()
    _save(fig, directory, "fig2_patient_clustered_pathways")


def plot_alpha_sensitivity(suite: dict[str, Any], directory: Path) -> None:
    rows = suite["alpha_grid"]
    alphas = np.asarray([row["alpha"] for row in rows])
    proposed = [row["methods"]["proposed"] for row in rows]
    fig, axes = plt.subplots(2, 2, figsize=(7.3, 5.3), constrained_layout=True)
    (a, b), (c, d) = axes
    _line(a, alphas, [row["mFDR_sys"] for row in proposed], PROPOSED, "o", "System")
    _line(a, alphas, [row["mFDR_macro"] for row in proposed], COMPARATOR, "s", "Clinic mean")
    _line(a, alphas, [row["mFDR_worst"] for row in proposed], PURPLE, "^", "Worst clinic")
    a.plot(alphas, alphas, ":", color=REFERENCE, linewidth=0.9, label=r"$y=\alpha$")
    a.set_xlabel(r"Reporting target $\alpha$")
    a.set_ylabel("Accepted-decision error rate")
    a.legend(frameon=False, fontsize=6.2)
    _panel(a, "a", "Risk against target")

    matrix = np.asarray([
        [row["methods"]["proposed"]["per_disease"][k]["mFDR"] / row["alpha"]
         if row["methods"]["proposed"]["per_disease"][k]["mFDR"] is not None
         else np.nan for row in rows]
        for k in range(10)
    ])
    vmax = max(1.05, float(np.nanmax(matrix)))
    cmap = LinearSegmentedColormap.from_list("risk", [PROPOSED, "#F7F7F7", RISK])
    image = b.imshow(
        matrix, aspect="auto", cmap=cmap,
        norm=TwoSlopeNorm(vmin=0, vcenter=1, vmax=vmax),
    )
    b.set_xticks(range(len(alphas)), [f"{x:g}" for x in alphas], rotation=35)
    b.set_yticks(range(10), [f"D{k}" for k in range(10)])
    b.set_xlabel(r"Reporting target $\alpha$")
    b.set_ylabel("Pathway")
    fig.colorbar(image, ax=b, fraction=0.047, pad=0.025, label="mFDR / target")
    _panel(b, "b", "Pathway risk relative to target")

    _line(c, alphas, [row["coverage"] for row in proposed], PROPOSED, "o", "AI-managed")
    _line(c, alphas, [row["referral"] for row in proposed], RISK, "s", "Human referral")
    c.set_xlabel(r"Reporting target $\alpha$")
    c.set_ylabel("Rate")
    c.legend(frameon=False)
    _panel(c, "c", "Coverage and referral")

    for method in ("proposed", "shared_threshold", "oracle_clean"):
        _line(
            d, alphas,
            [row["methods"][method]["K_meet"] for row in rows],
            METHOD_COLORS[method], METHOD_MARKERS[method], METHOD_LABELS[method],
        )
    d.set_xlabel(r"Reporting target $\alpha$")
    d.set_ylabel("Pathways at or below target")
    d.set_ylim(-0.3, 10.5)
    d.legend(frameon=False, fontsize=6.0)
    _panel(d, "d", "Empirical pathway attainment")
    for axis in (a, c, d):
        axis.set_xscale("log")
        axis.xaxis.set_major_formatter(mticker.FuncFormatter(lambda value, _: f"{value:g}"))
        axis.axvline(suite["meta"]["alpha"], color=REFERENCE, linewidth=0.7)
        _clean(axis)
    _clean(b, "none")
    _save(fig, directory, "fig3_patient_clustered_alpha_sensitivity")


def plot_baselines(suite: dict[str, Any], directory: Path) -> None:
    methods = suite["primary"]["methods"]
    boot = suite["baseline_bootstrap"]
    alpha_rows = suite["alpha_grid"]
    alphas = np.asarray([row["alpha"] for row in alpha_rows])
    fig, axes = plt.subplots(2, 2, figsize=(7.3, 5.4), constrained_layout=True)
    (a, b), (c, d) = axes
    a.axhline(suite["meta"]["alpha"], color=REFERENCE, linestyle="--", linewidth=0.9)
    for method in METHODS:
        x = boot[method]["coverage"]
        y = boot[method]["mFDR_sys"]
        # With deliberately tiny smoke-test bootstrap counts, the fixed
        # reference estimate can fall outside the empirical percentile range.
        # Matplotlib requires non-negative error lengths; clipping only the
        # displayed whisker length leaves every reported estimate unchanged.
        x_left = max(0.0, x["reference_pooled"] - x["ci_low"])
        x_right = max(0.0, x["ci_high"] - x["reference_pooled"])
        y_low = max(0.0, y["reference_pooled"] - y["ci_low"])
        y_high = max(0.0, y["ci_high"] - y["reference_pooled"])
        a.errorbar(
            x["reference_pooled"], y["reference_pooled"],
            xerr=[[x_left], [x_right]],
            yerr=[[y_low], [y_high]],
            fmt=METHOD_MARKERS[method], color=METHOD_COLORS[method],
            markersize=6 if method == "proposed" else 4.5,
            capsize=2, linewidth=0.9, label=METHOD_LABELS[method], zorder=3,
        )
    a.set_xlabel("AI-managed decision coverage")
    a.set_ylabel("System mFDR")
    a.legend(frameon=False, fontsize=5.6, ncol=2)
    _panel(a, "a", "Primary operating point with 95% CI")

    for method in METHODS:
        b.plot(
            [row["methods"][method]["coverage"] for row in alpha_rows],
            [row["methods"][method]["mFDR_sys"] for row in alpha_rows],
            marker=METHOD_MARKERS[method], markersize=3.0,
            color=METHOD_COLORS[method], linewidth=1.0,
            label=METHOD_LABELS[method],
        )
    b.set_xlabel("Coverage")
    b.set_ylabel("System mFDR")
    _panel(b, "b", "Risk–coverage frontiers")

    for method in ("proposed", "specialist_only", "shared_threshold", "oracle_clean"):
        _line(
            c, alphas, [row["methods"][method]["mFDR_sys"] for row in alpha_rows],
            METHOD_COLORS[method], METHOD_MARKERS[method], METHOD_LABELS[method],
        )
    c.plot(alphas, alphas, ":", color=REFERENCE, linewidth=0.8)
    c.set_xlabel(r"Reporting target $\alpha$")
    c.set_ylabel("System mFDR")
    c.legend(frameon=False, fontsize=5.6)
    _panel(c, "c", "System risk against target")

    for method in ("proposed", "shared_threshold", "oracle_clean"):
        _line(
            d, alphas, [row["methods"][method]["mFDR_worst"] for row in alpha_rows],
            METHOD_COLORS[method], METHOD_MARKERS[method], METHOD_LABELS[method],
        )
    d.plot(alphas, alphas, ":", color=REFERENCE, linewidth=0.8)
    d.set_xlabel(r"Reporting target $\alpha$")
    d.set_ylabel("Worst-pathway mFDR")
    d.legend(frameon=False, fontsize=5.8)
    _panel(d, "d", "Worst pathway")
    for axis in axes.flat:
        _clean(axis)
    for axis in (c, d):
        axis.set_xscale("log")
        axis.xaxis.set_major_formatter(mticker.FuncFormatter(lambda value, _: f"{value:g}"))
    _save(fig, directory, "fig4_patient_clustered_baselines")


def plot_contamination(suite: dict[str, Any], directory: Path) -> None:
    rows = suite["primary"]["contamination"]
    alpha_rows = suite["alpha_grid"]
    labels = [DISEASE_SHORT_NAMES[row["disease_id"]] for row in rows]
    y = np.arange(len(rows))
    fig, axes = plt.subplots(2, 2, figsize=(7.8, 6.5), constrained_layout=True)
    (a, b), (c, d) = axes
    width = 0.35
    a.barh(y - width / 2, [_finite(row["rho_route"]) for row in rows], width,
           color=PURPLE, label="Before specialist rejection")
    a.barh(y + width / 2, [_finite(row["rho_acc"]) for row in rows], width,
           color=SECONDARY, label="Among accepted decisions")
    a.set_yticks(y, labels)
    a.invert_yaxis()
    a.set_xlabel("Wrong-routed fraction")
    a.set_ylabel("Diagnostic pathway")
    a.legend(frameon=False, fontsize=6.0)
    _panel(a, "a", "Pathway routing contamination")

    route = np.asarray([_finite(row["route_error_share"]) for row in rows])
    diagnostic = np.asarray([_finite(row["diagnostic_error_share"]) for row in rows])
    b.barh(y, route, color=RISK, label="Routing-induced")
    b.barh(y, diagnostic, left=route, color=COMPARATOR, label="Within-clinic")
    b.set_yticks(y, labels)
    b.invert_yaxis()
    b.set_xlabel("Share of accepted errors")
    b.set_ylabel("Diagnostic pathway")
    b.legend(frameon=False, fontsize=6.0)
    _panel(b, "b", "Accepted-error source")

    alphas = np.asarray([row["alpha"] for row in alpha_rows])
    rho_route = [np.nanmean([_finite(x["rho_route"]) for x in row["contamination"]]) for row in alpha_rows]
    rho_acc = [np.nanmean([_finite(x["rho_acc"]) for x in row["contamination"]]) for row in alpha_rows]
    _line(c, alphas, rho_route, PURPLE, "o", "Before rejection")
    _line(c, alphas, rho_acc, SECONDARY, "s", "Accepted")
    c.set_xlabel(r"Reporting target $\alpha$")
    c.set_ylabel("Mean wrong-routed fraction")
    c.legend(frameon=False)
    _panel(c, "c", "Contamination across targets")

    referral = np.asarray([
        [_finite(x["wrong_route_referral"]) for x in row["contamination"]]
        for row in alpha_rows
    ])
    for k in range(10):
        d.plot(alphas, referral[:, k], color="#C4C4C4", linewidth=0.65)
    _line(d, alphas, np.nanmedian(referral, axis=1), PROPOSED, "o", "Pathway median")
    d.set_xlabel(r"Reporting target $\alpha$")
    d.set_ylabel("Wrong-route referral rate")
    d.legend(frameon=False)
    _panel(d, "d", "Routing-error containment")
    for axis in axes.flat:
        _clean(axis)
    for axis in (c, d):
        axis.set_xscale("log")
        axis.xaxis.set_major_formatter(mticker.FuncFormatter(lambda value, _: f"{value:g}"))
    _save(fig, directory, "fig5_patient_clustered_contamination")


def plot_beta_sensitivity(suite: dict[str, Any], directory: Path) -> None:
    rows = suite["beta_grid"]
    beta = np.asarray([row["beta"] for row in rows])
    fig, axes = plt.subplots(2, 2, figsize=(7.3, 5.3), constrained_layout=True)
    (a, b), (c, d) = axes
    _line(a, beta, [row["proposed"]["mFDR_sys"] for row in rows], PROPOSED, "o", "Proposed LEC")
    _line(a, beta, [row["specialist_only"]["mFDR_sys"] for row in rows], COMPARATOR, "s", "Routed specialist-only")
    _line(a, beta, [row["full_automation"]["mFDR_sys"] for row in rows], REFERENCE, "^", "Full automation")
    a.axhline(suite["meta"]["alpha"], color=REFERENCE, linestyle="--", linewidth=0.8)
    a.set_xlabel(r"Router target $\beta$")
    a.set_ylabel("System mFDR")
    a.legend(frameon=False)
    _panel(a, "a", "System risk")

    _line(b, beta, [row["proposed"]["coverage"] for row in rows], PROPOSED, "o", "Proposed LEC")
    _line(b, beta, [row["specialist_only"]["coverage"] for row in rows], COMPARATOR, "s", "Routed specialist-only")
    _line(b, beta, [row["router_coverage"] for row in rows], SECONDARY, "^", "Router coverage")
    b.set_xlabel(r"Router target $\beta$")
    b.set_ylabel("Coverage")
    b.legend(frameon=False)
    _panel(b, "b", "Automation")

    _line(c, beta, [row["router_mFDR"] for row in rows], RISK, "s", "Router mFDR")
    c.plot(beta, beta, ":", color=REFERENCE, linewidth=0.8, label=r"$y=\beta$")
    c.set_xlabel(r"Router target $\beta$")
    c.set_ylabel("Router accepted-case error")
    c.legend(frameon=False)
    _panel(c, "c", "Router risk control")

    delta_risk = [row["proposed"]["mFDR_sys"] - row["specialist_only"]["mFDR_sys"] for row in rows]
    delta_coverage = [row["proposed"]["coverage"] - row["specialist_only"]["coverage"] for row in rows]
    _line(d, beta, delta_risk, RISK, "o", r"$\Delta$mFDR")
    _line(d, beta, delta_coverage, PROPOSED, "s", r"$\Delta$coverage")
    d.axhline(0, color=REFERENCE, linestyle="--", linewidth=0.8)
    d.set_xlabel(r"Router target $\beta$")
    d.set_ylabel("Proposed minus routed specialist-only")
    d.legend(frameon=False)
    _panel(d, "d", "Incremental router contribution")
    for axis in axes.flat:
        axis.axvline(suite["meta"]["beta"], color=REFERENCE, linewidth=0.7)
        _clean(axis)
    _save(fig, directory, "fig6_patient_clustered_beta_sensitivity")


def write_tables(
    suite: dict[str, Any],
    primary: dict[str, Any],
    balanced: dict[str, Any] | None,
    strict: dict[str, Any] | None,
    directory: Path,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    audit = suite["input_audit"]

    # These names belonged to the earlier three-main-table layout.  Remove
    # them when re-rendering so the directory contains one unambiguous set.
    for stale_name in (
        "table2_patient_clustered_primary.tex",
        "table3_patient_clustered_baselines.tex",
    ):
        (directory / stale_name).unlink(missing_ok=True)

    table1 = [
        r"\begin{table*}[!t]", r"\centering",
        r"\caption{\textbf{Source cohorts, reference-label distributions and source-balanced role allocation.} Cohort totals are calculated from the frozen inference exports before source balancing. For every repeated partition, up to 2,000 unique patients are sampled without replacement from each source; smaller sources are retained in full. The sampled patients are assigned without overlap to router calibration ($C_R$), specialist calibration ($C_S$) and held-out evaluation ($T$) in proportions 0.40/0.30/0.30. Because the primary analysis selects one decision per patient after assignment, the patient counts in the last three columns are also the primary decision counts. Class distributions are decision-level reference-label counts before source balancing. Multi-label positive counts can sum to more than the number of decisions; ``none'' denotes an empty label set.}",
        r"\label{tab:patient_clustered_cohort}",
        r"\textbf{a. Cohort size and primary role allocation}\par\smallskip",
        r"\begin{adjustbox}{max width=\textwidth,center}", r"\small",
        r"\renewcommand{\arraystretch}{1.06}",
        r"\begin{tabular}{llllrrrrrr}", r"\toprule",
        r"Source & Pathway & Model & Label scope & Patients & Decisions & Included & $C_R$ & $C_S$ & $T$ \\",
        r"\midrule",
    ]

    ratios = (
        primary["meta"]["config"]["router_calib_ratio"],
        primary["meta"]["config"]["specialist_calib_ratio"],
        primary["meta"]["config"]["test_ratio"],
    )
    cap = suite["meta"].get("max_patients_per_source")

    def role_counts(n_patients: int) -> tuple[int, int, int]:
        raw = np.asarray(ratios, dtype=float) * n_patients
        sizes = np.floor(raw).astype(int)
        remainder = n_patients - int(sizes.sum())
        order = np.argsort(-(raw - sizes), kind="stable")
        for index in order[:remainder]:
            sizes[int(index)] += 1
        return tuple(int(value) for value in sizes)

    ordered_sources = sorted(
        audit["sources"].items(), key=lambda item: item[1]["disease_id"]
    )
    included_total = router_total = specialist_total = test_total = 0
    for source, row in ordered_sources:
        _, model, disease = SOURCE_METADATA[source]
        scope = next(iter(row["label_scope_counts"]))
        included = min(row["patients"], cap) if cap is not None else row["patients"]
        n_router, n_specialist, n_test = role_counts(included)
        included_total += included
        router_total += n_router
        specialist_total += n_specialist
        test_total += n_test
        table1.append(
            f"{_latex(source)} & {_latex(disease)} & {_latex(model)} & "
            f"{_latex(scope.title())} & {row['patients']:,} & {row['decisions']:,} & "
            f"{included:,} & {n_router:,} & {n_specialist:,} & {n_test:,} \\\\"
        )
    table1.extend([
        r"\midrule",
        f"Total & 10 pathways & 5 model families & Mixed & {audit['n_patients']:,} & "
        f"{audit['n_decisions']:,} & {included_total:,} & {router_total:,} & "
        f"{specialist_total:,} & {test_total:,} \\\\ ",
        r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}",
        r"\vspace{0.6em}",
        r"\textbf{b. Decision-level reference-label distribution}\par\smallskip",
        r"\footnotesize", r"\renewcommand{\arraystretch}{1.04}",
        r"\begin{tabular}{@{}lp{0.82\textwidth}@{}}", r"\toprule",
        r"Source & Reference-label counts \\", r"\midrule",
    ])
    for source, row in ordered_sources:
        class_counts = row.get("class_counts", {})
        labels = SOURCE_CLASS_NAMES[source]
        items = [
            f"{labels[index]} {int(class_counts.get(str(index), 0)):,}"
            for index in range(len(labels))
        ]
        if int(row.get("empty_label_decisions", 0)):
            items.append(f"none {int(row['empty_label_decisions']):,}")
        table1.append(f"{_latex(source)} & {_latex('; '.join(items))}. \\\\ ")
    table1.extend([
        r"\bottomrule", r"\end{tabular}",
        r"\tabnote{ImageCHD abbreviations follow the source label names. ACDC labels are NOR (normal), MINF (previous myocardial infarction), DCM (dilated cardiomyopathy), HCM (hypertrophic cardiomyopathy) and RV (abnormal right ventricle).}",
        r"\end{table*}", "",
    ])
    (directory / "table1_patient_clustered_cohort.tex").write_text(
        "\n".join(table1), encoding="utf-8"
    )

    table2 = [
        r"\begin{table*}[!t]", r"\centering",
        rf"\caption{{\textbf{{Patient-clustered system performance and analysis-unit robustness at the primary operating point.}} Results use {_calibration_note(suite)} and $\beta={_target_text(suite['meta']['beta'])}$ across $B={suite['meta']['n_splits']}$ patient-disjoint partitions.{_source_sampling_note(suite)} Intervals are 95\% source-dataset-stratified patient-cluster bootstrap confidence intervals.}}",
        r"\label{tab:patient_clustered_system}",
        r"\begin{adjustbox}{max width=\textwidth,center}", r"\small",
        r"\renewcommand{\arraystretch}{1.12}",
        r"\begin{tabular}{llccccc}", r"\toprule",
        r"Analysis & Method & System mFDR (95\% CI) & Coverage (95\% CI) & Accepted wrong-route (95\% CI) & Referral & Mean test decisions \\",
        r"\midrule",
    ]
    analyses = [
        (_analysis_label(primary), primary),
        (_analysis_label(balanced), balanced) if balanced else ("", None),
        ("Strict patient/volume subset", strict),
    ]
    for name, result in analyses:
        if result is None:
            continue
        for label, method in (("Proposed LEC", "proposed"), ("Routed specialist-only", "specialist_only")):
            m = result["bootstrap"][method]
            coverage = m["coverage"]["reference_pooled"]
            total = result["reference"][method].get(
                "test_decisions", result["reference"][method].get("test_patient_decisions")
            )
            table2.append(
                f"{_latex(name)} & {label} & {_interval(m['mFDR_sys'])} & "
                f"{_interval(m['coverage'])} & "
                f"{_interval(m['accepted_wrong_route_rate'])} & "
                f"{_fmt(1 - coverage)} & "
                f"{total / result['reference']['n_splits']:.1f} \\\\"
            )
    table2.extend([r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table*}", ""])
    (directory / "table2_patient_clustered_system.tex").write_text(
        "\n".join(table2), encoding="utf-8"
    )

    table3 = [
        r"\begin{table*}[!t]", r"\centering",
        rf"\caption{{\textbf{{Patient-clustered pathway-specific accepted-decision risk and retention at the primary operating point.}} Proposed LEC uses {_calibration_note(suite)} and $\beta={_target_text(suite['meta']['beta'])}$ across $B={suite['meta']['n_splits']}$ patient-disjoint partitions.{_source_sampling_note(suite)} Routed specialist-only uses the same router assignments and routed specialist outputs but disables router abstention. Predicted-pathway end-to-end retention divides final accepted decisions by all test decisions assigned to the predicted pathway; conditional specialist retention divides them by router-accepted decisions in that pathway. Entries are pooled estimates over the 500 partitions. Method-specific intervals are marginal 2.5th--97.5th percentile intervals from 1,000 source-dataset-stratified, split-respecting patient-cluster bootstrap replicates; only the Proposed LEC minus routed specialist-only $\Delta$mFDR intervals are paired differences.}}",
        r"\label{tab:patient_clustered_pathways}", r"\small",
        r"\renewcommand{\arraystretch}{1.12}",
        r"\textbf{a, Accepted-decision risk.}\par\smallskip",
        r"\begin{adjustbox}{max width=\textwidth,center}",
        r"\begin{tabular}{lccc}", r"\toprule",
        r"Pathway & LEC mFDR (95\% CI) & Routed specialist-only mFDR (95\% CI) & $\Delta$mFDR (95\% CI) \\",
        r"\midrule",
    ]
    pathway_rows = primary["bootstrap"]["per_disease"]
    for row in pathway_rows:
        table3.append(
            f"{_latex(DISEASE_NAMES[row['disease_id']])} & {_interval(row['mFDR_sys'])} & "
            f"{_interval(row['specialist_only_mFDR_sys'])} & "
            f"{_interval(row['delta_mFDR_sys'])} \\\\"
        )
    table3.extend([
        r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\medskip",
        r"\textbf{b, Retention.}\par\smallskip",
        r"\begin{adjustbox}{max width=\textwidth,center}",
        r"\begin{tabular}{lcc}", r"\toprule",
        r"Pathway & Predicted-pathway retention (95\% CI) & Conditional specialist retention $\mid R$ (95\% CI) \\",
        r"\midrule",
    ])
    for row in pathway_rows:
        table3.append(
            f"{_latex(DISEASE_NAMES[row['disease_id']])} & {_interval(row['coverage'])} & "
            f"{_interval(row['conditional_specialist_coverage'])} \\\\"
        )
    table3.extend([r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table*}", ""])
    (directory / "table3_patient_clustered_pathways.tex").write_text(
        "\n".join(table3), encoding="utf-8"
    )

    table4 = [
        r"\begin{table*}[!t]", r"\centering",
        rf"\caption{{\textbf{{Patient-clustered comparison of the proposed workflow and baseline strategies.}} All methods use the same patient-disjoint partitions with {_calibration_note(suite)} and $\beta={_target_text(suite['meta']['beta'])}$.{_source_sampling_note(suite)} Routed specialist-only retains every router assignment and applies pathway-specific specialist calibration to the resulting unfiltered routed streams; Proposed LEC additionally applies calibrated router abstention. Wrong-route fractions therefore remain defined for both methods. Entries are pooled estimates over the 500 partitions. Each displayed method interval is a marginal 2.5th--97.5th percentile interval from 1,000 source-dataset-stratified, split-respecting patient-cluster bootstrap replicates.}}",
        r"\label{tab:patient_clustered_baselines}",
        r"\begin{adjustbox}{max width=\textwidth,center}", r"\small",
        r"\renewcommand{\arraystretch}{1.12}",
        r"\begin{tabular}{lccccc}", r"\toprule",
        r"Method & System mFDR (95\% CI) & Coverage (95\% CI) & Worst-pathway mFDR & Pathways $\leq\alpha$ & Wrong-route fraction \\",
        r"\midrule",
    ]
    for method in METHODS:
        summary = suite["baseline_bootstrap"][method]
        values = suite["primary"]["methods"][method]
        table4.append(
            f"{METHOD_LABELS[method]} & {_interval(summary['mFDR_sys'])} & "
            f"{_interval(summary['coverage'])} & {_fmt(values['mFDR_worst'])} & "
            f"{values['K_meet']}/{values['K_estimable']} & "
            f"{_fmt(values['accepted_wrong_route_rate'])} \\\\"
        )
    table4.extend([r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table*}", ""])
    (directory / "table4_patient_clustered_baselines.tex").write_text(
        "\n".join(table4), encoding="utf-8"
    )

    manifest = [
        "# Patient-clustered manuscript set",
        "",
        "Primary estimand: dataset-defined diagnostic decisions.",
        "Patients are disjoint across router calibration, specialist calibration and test.",
        "All confidence intervals use source-stratified patient-cluster bootstrap resampling.",
        "Within-patient image/lesion labels are retained rather than collapsed.",
        "",
        "Figures: 6 unique figures (PDF and PNG).",
        "Tables: 4 main-text LaTeX tables.",
        "Together with the 2 supplementary tables, the Paper2-compatible set contains 6 tables.",
        "",
    ]
    (directory / "README.md").write_text("\n".join(manifest), encoding="utf-8")


def write_paper2_table_set(
    main_directory: Path,
    supplementary_directory: Path,
    target_directory: Path,
) -> list[Path]:
    """Write six drop-in Paper2.0 table fragments with its existing labels.

    The source tables retain descriptive patient-clustered filenames for audit
    purposes.  This compatibility layer gives the manuscript the stable
    ``tab1.tex``--``tab6.tex`` interface already used by its section files.
    """
    target_directory.mkdir(parents=True, exist_ok=True)
    mappings = (
        (main_directory / "table2_patient_clustered_system.tex", "tab1.tex",
         "tab:patient_clustered_system", "tab:system_summary"),
        (main_directory / "table3_patient_clustered_pathways.tex", "tab2.tex",
         "tab:patient_clustered_pathways", "tab:per_disease_risk"),
        (supplementary_directory / "tableS1_patient_clustered_calibration_feasibility.tex", "tab3.tex",
         "tab:patient_clustered_calibration_feasibility", "tab:calibration_feasibility"),
        (supplementary_directory / "tableS2_patient_clustered_contamination.tex", "tab4.tex",
         "tab:patient_clustered_contamination", "tab:contamination"),
        (main_directory / "table4_patient_clustered_baselines.tex", "tab5.tex",
         "tab:patient_clustered_baselines", "tab:baselines"),
        (main_directory / "table1_patient_clustered_cohort.tex", "tab6.tex",
         "tab:patient_clustered_cohort", "tab:dataset_summary"),
    )
    written: list[Path] = []
    for source, target_name, source_label, target_label in mappings:
        if not source.exists():
            raise FileNotFoundError(f"Cannot stage Paper2.0 table; missing {source}")
        content = source.read_text(encoding="utf-8")
        content = content.replace(
            rf"\label{{{source_label}}}", rf"\label{{{target_label}}}", 1
        )
        target = target_directory / target_name
        target.write_text(content, encoding="utf-8")
        written.append(target)
    readme = [
        "# Paper2.0 patient-clustered table set",
        "",
        "These six files are drop-in replacements for tables/newtab/tab1.tex--tab6.tex.",
        "Their labels match the existing Paper2.0 section references.",
        "All numerical results are derived from the patient-clustered workflow.",
        "tab6 reports source patients, decisions, reference-label distributions and exact primary C_R/C_S/T role sizes.",
        "",
    ]
    (target_directory / "README.md").write_text("\n".join(readme), encoding="utf-8")
    return written


def write_paper_outputs(
    suite: dict[str, Any],
    primary: dict[str, Any],
    *,
    patient_balanced: dict[str, Any] | None,
    strict_patient: dict[str, Any] | None,
    figure_dir: Path,
    table_dir: Path,
) -> None:
    _style()
    plot_primary_and_robustness(primary, patient_balanced, strict_patient, figure_dir)
    plot_pathway_forest(primary, figure_dir)
    plot_alpha_sensitivity(suite, figure_dir)
    plot_baselines(suite, figure_dir)
    plot_contamination(suite, figure_dir)
    plot_beta_sensitivity(suite, figure_dir)
    write_tables(suite, primary, patient_balanced, strict_patient, table_dir)


def plot_calibration_size(suite: dict[str, Any], directory: Path) -> None:
    rows = suite["calibration_size_grid"]
    x = np.asarray([row["fraction"] for row in rows])
    fig, axes = plt.subplots(2, 2, figsize=(7.3, 5.3), constrained_layout=True)
    (a, b), (c, d) = axes

    _line(a, x, [row["router"]["mFDR"] for row in rows], RISK, "s", "Router")
    _line(a, x, [row["system"]["mFDR_sys"] for row in rows], PROPOSED, "o", "System")
    _line(a, x, [row["system"]["mFDR_macro"] for row in rows], COMPARATOR, "^", "Pathway mean")
    a.axhline(suite["meta"]["alpha"], color=REFERENCE, linestyle="--", linewidth=0.8)
    a.set_xlabel("Fraction of calibration patient clusters retained")
    a.set_ylabel("Accepted-decision error rate")
    a.legend(frameon=False)
    _panel(a, "a", "Risk")

    _line(b, x, [row["router"]["coverage"] for row in rows], SECONDARY, "s", "Router")
    _line(b, x, [row["system"]["coverage"] for row in rows], PROPOSED, "o", "AI-managed")
    b.set_xlabel("Fraction of calibration patient clusters retained")
    b.set_ylabel("Coverage")
    b.legend(frameon=False)
    _panel(b, "b", "Coverage")

    router_sd = [_finite(row["router"]["threshold"]["sd"]) for row in rows]
    clinic_median_sd = [np.nanmedian([
        _finite(clinic["threshold"]["sd"]) for clinic in row["clinics"]
    ]) for row in rows]
    _line(c, x, router_sd, RISK, "s", "Router")
    _line(c, x, clinic_median_sd, PROPOSED, "o", "Median pathway")
    c.set_xlabel("Fraction of calibration patient clusters retained")
    c.set_ylabel("Threshold SD")
    c.legend(frameon=False)
    _panel(c, "c", "Threshold stability")

    _line(d, x, [row["router"]["infeasible_frequency"] for row in rows],
          RISK, "s", "Router")
    _line(d, x, [np.mean([
        clinic["infeasible_frequency"] for clinic in row["clinics"]
    ]) for row in rows], PROPOSED, "o", "Pathway mean")
    _line(d, x, [max(
        clinic["infeasible_frequency"] for clinic in row["clinics"]
    ) for row in rows], COMPARATOR, "^", "Pathway maximum")
    d.set_xlabel("Fraction of calibration patient clusters retained")
    d.set_ylabel("Infeasible-split frequency")
    d.legend(frameon=False)
    _panel(d, "d", "Calibration feasibility")
    for axis in axes.flat:
        axis.set_xticks(x)
        _clean(axis)
    _save(fig, directory, "figS1_patient_clustered_calibration_size")


def plot_exchangeability(suite: dict[str, Any], directory: Path) -> None:
    rows = suite["primary"]["exchangeability"]
    fig, axes = plt.subplots(2, 2, figsize=(7.3, 5.3), constrained_layout=True)
    (a, b), (c, d) = axes

    cal_base = np.asarray([_finite(row["base_error_cal"]) for row in rows])
    test_base = np.asarray([_finite(row["base_error_test"]) for row in rows])
    a.scatter(cal_base, test_base, color=PROPOSED, s=25, zorder=3)
    maximum = float(np.nanmax(np.r_[cal_base, test_base]))
    a.plot([0, maximum], [0, maximum], "--", color=REFERENCE, linewidth=0.8)
    a.set_xlabel("Specialist-calibration base error")
    a.set_ylabel("Held-out base error")
    _panel(a, "a", "Routed-stream base error")

    ordered = sorted(rows, key=lambda row: row["mean_effective_calibration_size"])
    positions = np.arange(len(ordered))
    b.bar(positions, [_finite(row["KS_mean"]) for row in ordered],
          color=PROPOSED, label="Observed mean")
    b.plot(positions, [_finite(row["KS_reference_95_mean"]) for row in ordered],
           "--", color=REFERENCE, marker="o", markersize=3,
           label="Nominal 95% reference")
    b.set_xticks(positions, [f"D{row['disease_id']}" for row in ordered])
    b.set_xlabel("Diagnostic pathway")
    b.set_ylabel("Two-sample KS statistic")
    b.legend(frameon=False, fontsize=6.0)
    _panel(b, "b", "Score comparability")

    cal_fdp = np.asarray([_finite(row["FDP_cal"]) for row in rows])
    test_fdp = np.asarray([_finite(row["FDP_test"]) for row in rows])
    c.scatter(cal_fdp, test_fdp, color=PROPOSED, s=25, zorder=3)
    limit = float(np.nanmax(np.r_[cal_fdp, test_fdp, suite["meta"]["alpha"]])) * 1.08
    c.plot([0, limit], [0, limit], "--", color=REFERENCE, linewidth=0.8)
    c.axhline(suite["meta"]["alpha"], color=REFERENCE, linestyle=":", linewidth=0.8)
    c.axvline(suite["meta"]["alpha"], color=REFERENCE, linestyle=":", linewidth=0.8)
    c.set_xlabel("Calibration FDP")
    c.set_ylabel("Held-out FDP")
    _panel(c, "c", "FDP at calibrated thresholds")

    effective_size = np.asarray([
        _finite(row["mean_effective_calibration_size"]) for row in rows
    ])
    absolute_gap = np.abs(test_fdp - cal_fdp)
    colors = [RISK if gap > 0.05 else PROPOSED for gap in absolute_gap]
    d.scatter(effective_size, absolute_gap, c=colors, s=28, zorder=3)
    annotation_offsets = {5: (3, 8), 8: (3, -9)}
    for row, x_value, y_value in zip(rows, effective_size, absolute_gap):
        d.annotate(
            f"D{row['disease_id']}", (x_value, y_value),
            xytext=annotation_offsets.get(row["disease_id"], (3, 3)),
            textcoords="offset points", fontsize=6.0,
        )
    d.set_xscale("log")
    d.xaxis.set_major_locator(mticker.FixedLocator([20, 50, 100, 200, 500]))
    d.xaxis.set_major_formatter(mticker.FuncFormatter(lambda value, _: f"{value:g}"))
    d.xaxis.set_minor_locator(mticker.NullLocator())
    d.set_xlabel("Mean routed calibration decisions")
    d.set_ylabel("Absolute calibration-held-out FDP gap")
    _panel(d, "d", "Finite-evidence sensitivity")
    for axis in axes.flat:
        _clean(axis)
    _save(fig, directory, "figS2_patient_clustered_exchangeability")


def _utility_matrix(result: dict[str, Any]) -> np.ndarray:
    def components(method: str) -> tuple[float, float, float]:
        metrics = result["reference"][method]
        risk = metrics["mFDR_sys"]
        coverage = metrics["coverage"]
        return coverage * (1 - risk), coverage * risk, 1 - coverage

    proposed = components("proposed")
    specialist_only = components("specialist_only")
    c_errors = (1, 5, 10, 20, 50)
    c_referrals = (0.1, 0.5, 1.0, 2.0)
    return np.asarray([
        [
            (proposed[0] - c_error * proposed[1] - c_referral * proposed[2])
            - (
                specialist_only[0]
                - c_error * specialist_only[1]
                - c_referral * specialist_only[2]
            )
            for c_referral in c_referrals
        ]
        for c_error in c_errors
    ])


def plot_utility(
    primary: dict[str, Any],
    balanced: dict[str, Any] | None,
    strict: dict[str, Any] | None,
    directory: Path,
) -> None:
    analyses = [("One decision\nper patient", primary)]
    if balanced:
        analyses.append(("All decisions\n(patient clustered)", balanced))
    if strict:
        analyses.append(("Strict patient/\nvolume subset", strict))
    matrices = [_utility_matrix(result) for _, result in analyses]
    extent = max(abs(float(matrix.min())) for matrix in matrices)
    extent = max(extent, max(abs(float(matrix.max())) for matrix in matrices))
    cmap = LinearSegmentedColormap.from_list("utility", [RISK, "#F7F7F7", PROPOSED])
    fig, axes = plt.subplots(1, len(analyses), figsize=(3.1 * len(analyses), 3.5),
                             constrained_layout=True)
    axes = np.atleast_1d(axes)
    image = None
    for index, (axis, (title, _), matrix) in enumerate(zip(axes, analyses, matrices)):
        image = axis.imshow(
            matrix, aspect="auto", cmap=cmap,
            norm=TwoSlopeNorm(vmin=-extent, vcenter=0, vmax=extent),
        )
        axis.set_xticks(range(4), [0.1, 0.5, 1.0, 2.0])
        axis.set_yticks(range(5), [1, 5, 10, 20, 50])
        axis.set_xlabel(r"Referral cost $c_{ref}$")
        axis.set_ylabel(r"Error cost $c_{err}$")
        _panel(axis, chr(ord("a") + index), title)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                axis.text(j, i, f"{matrix[i, j]:+.2f}", ha="center", va="center", fontsize=6.2)
    fig.colorbar(
        image, ax=axes.tolist(), shrink=0.82,
        label=r"$\Delta$ utility (Proposed minus comparator)",
    )
    _save(fig, directory, "figS3_patient_clustered_utility")


def write_supplementary_tables(suite: dict[str, Any], directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    full = suite["calibration_size_grid"][-1]
    table = [
        r"\begin{table*}[!t]", r"\centering",
        rf"\caption{{\textbf{{Patient-clustered calibration evidence and feasibility at the primary operating point.}} Effective calibration sizes and thresholds describe routed Proposed LEC calibration. Routed specialist-only uses the same predicted pathways and specialist scores but calibrates on the complete unfiltered routed streams. Zero-acceptance frequency is the fraction of held-out partitions in which routed specialist-only accepted no decision for that pathway. Results use {_calibration_note(suite)} and $\beta={_target_text(suite['meta']['beta'])}$ across $B={suite['meta']['n_splits']}$ patient-disjoint partitions.{_source_sampling_note(suite)}}}",
        r"\label{tab:patient_clustered_calibration_feasibility}",
        r"\begin{adjustbox}{max width=\textwidth,center}", r"\small",
        r"\renewcommand{\arraystretch}{1.12}",
        r"\begin{tabular}{llcccccccc}", r"\toprule",
        r"$k$ & Decision point & Effective patients & Effective decisions & Threshold mean (SD) & Min patients & LEC feasible & LEC infeasible & Routed specialist-only feasible & Routed specialist-only zero-acceptance \\",
        r"\midrule",
    ]
    router = full["router"]
    table.append(
        f"-- & Router & {router['calibration_patients']['mean']:.1f} & "
        f"{router['calibration_decisions']['mean']:.1f} & "
        f"{router['threshold']['mean']:.3f} ({router['threshold']['sd']:.3f}) & "
        f"{router['calibration_patients']['min']:.0f} & {router['feasible_splits']} & "
        f"{router['infeasible_frequency']:.3f} & --- & --- \\\\"
    )
    table.append(r"\midrule")
    diagnostics = suite["primary"]["diagnostics"]
    so_feasible = diagnostics["specialist_only_feasible_splits_by_pathway"]
    so_zero = diagnostics["specialist_only_zero_acceptance_frequency_by_pathway"]
    for row in full["clinics"]:
        threshold_mean = _fmt(row["threshold"]["mean"])
        threshold_sd = _fmt(row["threshold"]["sd"])
        table.append(
            f"{row['disease_id']} & {_latex(DISEASE_NAMES[row['disease_id']])} & "
            f"{row['effective_patients']['mean']:.1f} & "
            f"{row['effective_decisions']['mean']:.1f} & "
            f"{threshold_mean} ({threshold_sd}) & "
            f"{row['effective_patients']['min']:.0f} & {row['feasible_splits']} & "
            f"{row['infeasible_frequency']:.3f} & "
            f"{so_feasible[row['disease_id']]} & {so_zero[row['disease_id']]:.3f} \\\\"
        )
    table.extend([r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table*}", ""])
    (directory / "tableS1_patient_clustered_calibration_feasibility.tex").write_text(
        "\n".join(table), encoding="utf-8"
    )

    contamination = [
        r"\begin{table*}[!t]", r"\centering",
        r"\caption{\textbf{Patient-clustered routing contamination and accepted-error decomposition.} Pooled held-out counts quantify wrong-pathway routing before specialist rejection, contamination among accepted AI-managed decisions, and the contribution of routing errors to accepted errors.}",
        r"\label{tab:patient_clustered_contamination}",
        r"\begin{adjustbox}{max width=\textwidth,center}", r"\small",
        r"\renewcommand{\arraystretch}{1.12}",
        r"\begin{tabular}{rlccccrrrr}", r"\toprule",
        r"$k$ & Pathway & $\rho^{route}$ & $\rho^{acc}$ & Wrong-route referral & Route-error share & Routed & Wrong-routed & Accepted & Accepted errors \\",
        r"\midrule",
    ]
    for row in suite["primary"]["contamination"]:
        contamination.append(
            f"{row['disease_id']} & {_latex(DISEASE_NAMES[row['disease_id']])} & "
            f"{_fmt(row['rho_route'])} & {_fmt(row['rho_acc'])} & "
            f"{_fmt(row['wrong_route_referral'])} & {_fmt(row['route_error_share'])} & "
            f"{row['routed']:,} & {row['wrong_routed']:,} & {row['accepted']:,} & "
            f"{row['accepted_wrong'] + row['accepted_diag_error']:,} \\\\"
        )
    contamination.extend([r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table*}", ""])
    (directory / "tableS2_patient_clustered_contamination.tex").write_text(
        "\n".join(contamination), encoding="utf-8"
    )
    readme = [
        "# Additional patient-clustered manuscript artifacts", "",
        "Figures: calibration evidence, routed-stream comparability and utility.",
        "Tables: calibration feasibility and routing contamination.",
        "Calibration fractions are applied to patient clusters within source datasets, not to records.",
        "Exchangeability panels are empirical diagnostics and do not prove future-distribution exchangeability.",
        "",
    ]
    (directory / "README.md").write_text("\n".join(readme), encoding="utf-8")


def write_supplementary_outputs(
    suite: dict[str, Any],
    primary: dict[str, Any],
    *,
    patient_balanced: dict[str, Any] | None,
    strict_patient: dict[str, Any] | None,
    figure_dir: Path,
    table_dir: Path,
) -> None:
    _style()
    plot_calibration_size(suite, figure_dir)
    plot_exchangeability(suite, figure_dir)
    plot_utility(primary, patient_balanced, strict_patient, figure_dir)
    write_supplementary_tables(suite, table_dir)
