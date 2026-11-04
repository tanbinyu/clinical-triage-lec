"""Primary patient-level LaTeX/CSV tables and Markdown summary."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


def _fmt(value: float | None, digits: int = 3) -> str:
    return "NA" if value is None else f"{value:.{digits}f}"


def _interval(summary: dict[str, Any]) -> str:
    return f"{_fmt(summary['reference_pooled'])} [{_fmt(summary['ci_low'])}, {_fmt(summary['ci_high'])}]"


def _latex_escape(text: str) -> str:
    return text.replace("&", r"\&").replace("_", r"\_")


def write_tables(result: dict[str, Any], table_dir: Path) -> None:
    table_dir.mkdir(parents=True, exist_ok=True)
    bootstrap = result["bootstrap"]
    reference = result["reference"]
    primary_rows = []
    for label, key in (("Proposed LEC", "proposed"), ("Specialist-only", "specialist_only")):
        primary_rows.append({
            "method": label,
            "mFDR_sys": bootstrap[key]["mFDR_sys"]["reference_pooled"],
            "mFDR_ci_low": bootstrap[key]["mFDR_sys"]["ci_low"],
            "mFDR_ci_high": bootstrap[key]["mFDR_sys"]["ci_high"],
            "coverage": bootstrap[key]["coverage"]["reference_pooled"],
            "coverage_ci_low": bootstrap[key]["coverage"]["ci_low"],
            "coverage_ci_high": bootstrap[key]["coverage"]["ci_high"],
        })
    with open(table_dir / "patient_level_primary.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(primary_rows[0]))
        writer.writeheader()
        writer.writerows(primary_rows)

    primary_tex = [
        r"\begin{table}[!t]",
        r"\centering",
        r"\caption{Patient-level primary performance with 95\% stratified cluster-bootstrap confidence intervals.}",
        r"\label{tab:patient_level_primary}",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Method & System mFDR & AI-managed coverage \\",
        r"\midrule",
    ]
    for label, key in (("Proposed LEC", "proposed"), ("Specialist-only", "specialist_only")):
        primary_tex.append(
            f"{label} & {_interval(bootstrap[key]['mFDR_sys'])} & "
            f"{_interval(bootstrap[key]['coverage'])} \\\\"
        )
    primary_tex.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    (table_dir / "patient_level_primary.tex").write_text("\n".join(primary_tex), encoding="utf-8")

    per_disease_rows = []
    per_disease_tex = [
        r"\begin{table*}[!t]",
        r"\centering",
        r"\caption{Patient-level clinic-specific performance of the proposed LEC system.}",
        r"\label{tab:patient_level_disease}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Specialist pathway & mFDR (95\% CI) & Coverage (95\% CI) & Referral & Test-patient decisions \\",
        r"\midrule",
    ]
    for reference_row, bootstrap_row in zip(reference["per_disease"], bootstrap["per_disease"]):
        proposed = reference_row["proposed"]
        row = {
            "disease_id": reference_row["disease_id"],
            "disease_name": reference_row["disease_name"],
            "mFDR_sys": bootstrap_row["mFDR_sys"]["reference_pooled"],
            "mFDR_ci_low": bootstrap_row["mFDR_sys"]["ci_low"],
            "mFDR_ci_high": bootstrap_row["mFDR_sys"]["ci_high"],
            "coverage": bootstrap_row["coverage"]["reference_pooled"],
            "coverage_ci_low": bootstrap_row["coverage"]["ci_low"],
            "coverage_ci_high": bootstrap_row["coverage"]["ci_high"],
            "referral": proposed["referral"],
            "test_patient_decisions": proposed["test_patient_decisions"],
        }
        per_disease_rows.append(row)
        per_disease_tex.append(
            f"{_latex_escape(row['disease_name'])} & {_interval(bootstrap_row['mFDR_sys'])} & "
            f"{_interval(bootstrap_row['coverage'])} & {_fmt(row['referral'])} & "
            f"{row['test_patient_decisions']:,} \\\\"
        )
    per_disease_tex.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    with open(table_dir / "patient_level_per_disease.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_disease_rows[0]))
        writer.writeheader()
        writer.writerows(per_disease_rows)
    (table_dir / "patient_level_per_disease.tex").write_text(
        "\n".join(per_disease_tex), encoding="utf-8"
    )


def write_summary(result: dict[str, Any], table_dir: Path) -> None:
    meta = result["meta"]
    audit = result["input_audit"]
    bootstrap = result["bootstrap"]
    lines = [
        "# Patient-level experiment",
        "",
        f"- Records: {audit['n_records']:,}",
        f"- Patient units: {audit['n_patients']:,}",
        f"- Aggregation: {audit['aggregation']}",
        f"- Split: {meta['split_design']}",
        f"- Patient-ID types: {audit['patient_id_type_counts']}",
        "",
        "| Quantity | Pooled estimate | 95% patient-bootstrap CI |",
        "|---|---:|---:|",
    ]
    for label, section, metric in (
        ("Proposed system mFDR", "proposed", "mFDR_sys"),
        ("Proposed coverage", "proposed", "coverage"),
        ("ΔmFDR (LEC − specialist-only)", "paired_difference", "delta_mFDR_sys"),
        ("Δcoverage (LEC − specialist-only)", "paired_difference", "delta_coverage"),
    ):
        summary = bootstrap[section][metric]
        lines.append(
            f"| {label} | {_fmt(summary['reference_pooled'], 4)} | "
            f"[{_fmt(summary['ci_low'], 4)}, {_fmt(summary['ci_high'], 4)}] |"
        )
    lines.extend([
        "",
        "The 500 repeated splits describe split sensitivity and are not treated as "
        "independent cohorts. Confidence intervals are generated by the separate "
        "source-stratified, split-respecting patient bootstrap.",
        "",
    ])
    (table_dir / "patient_level_summary.md").write_text("\n".join(lines), encoding="utf-8")
