"""LaTeX tables and table manifest for the patient-level full suite."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .suite import METHOD_LABELS, METHODS


def _fmt(value: float | None, digits: int = 4) -> str:
    return "---" if value is None else f"{value:.{digits}f}"


def _escape(value: str) -> str:
    return value.replace("&", r"\&").replace("_", r"\_")


def _write_compact_tables(suite: dict[str, Any], primary: dict[str, Any], directory: Path) -> None:
    """Retained compact legacy layout; the full suite uses ``write_tables``."""
    directory.mkdir(parents=True, exist_ok=True)
    p = suite["primary"]
    proposed = p["methods"]["proposed"]
    boot = primary["bootstrap"]
    router = p["router"]
    tab1 = [
        r"\begin{table*}[t]", r"\centering",
        r"\caption{Patient-level system performance at the primary operating point. Bootstrap intervals use 1,000 source-stratified patient resamples.}",
        r"\label{tab:patient_system_summary}",
        r"\begin{tabular}{lccc}", r"\toprule",
        r"Metric & Pooled estimate & Split mean (SD) & 95\% patient-bootstrap CI \\",
        r"\midrule",
        f"Router mFDR & {_fmt(router['mFDR'])} & --- & --- \\\\",
        f"Router coverage & {_fmt(router['coverage'])} & --- & --- \\\\",
        f"System mFDR & {_fmt(proposed['mFDR_sys'])} & {_fmt(proposed['split_mFDR']['mean'])} ({_fmt(proposed['split_mFDR']['sd'])}) & [{_fmt(boot['proposed']['mFDR_sys']['ci_low'])}, {_fmt(boot['proposed']['mFDR_sys']['ci_high'])}] \\\\",
        f"AI-managed coverage & {_fmt(proposed['coverage'])} & {_fmt(proposed['split_coverage']['mean'])} ({_fmt(proposed['split_coverage']['sd'])}) & [{_fmt(boot['proposed']['coverage']['ci_low'])}, {_fmt(boot['proposed']['coverage']['ci_high'])}] \\\\",
        f"Human referral & {_fmt(proposed['referral'])} & --- & --- \\\\",
        r"\bottomrule", r"\end{tabular}", r"\end{table*}", "",
    ]
    (directory / "tab1_system_summary.tex").write_text("\n".join(tab1), encoding="utf-8")

    tab2 = [
        r"\begin{table*}[t]", r"\centering",
        r"\caption{Patient-level clinic-specific performance at the primary operating point.}",
        r"\label{tab:patient_per_disease}",
        r"\begin{tabular}{rlccccr}", r"\toprule",
        r"$k$ & Specialist pathway & mFDR & 95\% CI & Coverage & Referral & Test-patient decisions \\",
        r"\midrule",
    ]
    for row, ci in zip(proposed["per_disease"], boot["per_disease"]):
        tab2.append(
            f"{row['disease_id']} & {_escape(row['disease_name'])} & {_fmt(row['mFDR'])} & "
            f"[{_fmt(ci['mFDR_sys']['ci_low'])}, {_fmt(ci['mFDR_sys']['ci_high'])}] & "
            f"{_fmt(row['coverage'])} & {_fmt(row['referral'])} & {row['total']:,} \\\\"
        )
    tab2.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    (directory / "tab2_per_disease_risk.tex").write_text("\n".join(tab2), encoding="utf-8")

    tab3 = [
        r"\begin{table*}[t]", r"\centering",
        r"\caption{Patient-level effective specialist calibration size, threshold stability and feasibility.}",
        r"\label{tab:patient_calibration_feasibility}",
        r"\begin{tabular}{rlrrrrrr}", r"\toprule",
        r"$k$ & Pathway & Mean size & SD & Min & [2.5,97.5] & Threshold SD & Infeasible frequency \\",
        r"\midrule",
    ]
    for row in p["calibration"]:
        size = row["effective_size"]
        tab3.append(
            f"{row['disease_id']} & {_escape(row['disease_name'])} & {size['mean']:.1f} & "
            f"{size['sd']:.1f} & {size['min']} & [{size['p2.5']:.1f},{size['p97.5']:.1f}] & "
            f"{_fmt(row['threshold']['sd'])} & {_fmt(row['infeasible_frequency'], 3)} \\\\"
        )
    tab3.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    (directory / "tab3_calibration_feasibility.tex").write_text("\n".join(tab3), encoding="utf-8")

    tab4 = [
        r"\begin{table*}[t]", r"\centering",
        r"\caption{Patient-level routing contamination and accepted-error decomposition.}",
        r"\label{tab:patient_contamination}",
        r"\begin{tabular}{rlcccccrrr}", r"\toprule",
        r"$k$ & Pathway & $\rho^{route}$ & $\rho^{acc}$ & Wrong-route referral & Route-error share & Diagnostic-error share & Routed & Accepted & Errors \\",
        r"\midrule",
    ]
    for row in p["contamination"]:
        tab4.append(
            f"{row['disease_id']} & {_escape(row['disease_name'])} & {_fmt(row['rho_route'],3)} & "
            f"{_fmt(row['rho_acc'],3)} & {_fmt(row['wrong_route_referral'],3)} & "
            f"{_fmt(row['route_error_share'],3)} & {_fmt(row['diagnostic_error_share'],3)} & "
            f"{row['routed']:,} & {row['accepted']:,} & "
            f"{row['accepted_wrong'] + row['accepted_diag_error']:,} \\\\"
        )
    tab4.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    (directory / "tab4_contamination.tex").write_text("\n".join(tab4), encoding="utf-8")

    tab5 = [
        r"\begin{table*}[t]", r"\centering",
        r"\caption{Patient-level comparison with baseline workflows at the primary operating point.}",
        r"\label{tab:patient_baselines}",
        r"\begin{tabular}{lccccc}", r"\toprule",
        r"System & mFDR & Coverage & Referral & Worst-clinic mFDR & $K_{meet}$ \\",
        r"\midrule",
    ]
    for method in METHODS:
        row = p["methods"][method]
        label = METHOD_LABELS[method]
        if method == "proposed":
            label = r"\textbf{Proposed LEC}"
        tab5.append(
            f"{label} & {_fmt(row['mFDR_sys'])} & {_fmt(row['coverage'])} & "
            f"{_fmt(row['referral'])} & {_fmt(row['mFDR_worst'])} & "
            f"{row['K_meet']}/{row['K_estimable']} \\\\"
        )
    tab5.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    (directory / "tab5_baselines.tex").write_text("\n".join(tab5), encoding="utf-8")

    audit = suite["input_audit"]
    tab6 = [
        r"\begin{table*}[t]", r"\centering",
        r"\caption{Patient-level evaluation cohort and derived records by source dataset.}",
        r"\label{tab:patient_cohort}",
        r"\begin{tabular}{lrrrr}", r"\toprule",
        r"Source dataset & Pathway & Patients & Derived records & Multi-record patients \\",
        r"\midrule",
    ]
    for source, row in audit["sources"].items():
        tab6.append(
            f"{_escape(source)} & {row['disease_id']} & {row['patients']:,} & "
            f"{row['records']:,} & {row['multi_record_patients']:,} \\\\"
        )
    tab6.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    (directory / "tab6_cohort.tex").write_text("\n".join(tab6), encoding="utf-8")


def write_tables(suite: dict[str, Any], primary: dict[str, Any], directory: Path) -> None:
    """Write field-complete patient-level counterparts of manuscript Tables 1--6."""
    directory.mkdir(parents=True, exist_ok=True)
    block = suite["primary"]
    proposed = block["methods"]["proposed"]
    bootstrap = primary["bootstrap"]
    router = block["router"]

    def split_cells(distribution: dict[str, Any], *, invert: bool = False) -> tuple[str, str]:
        if distribution.get("mean") is None:
            return "---", "---"
        mean = float(distribution["mean"])
        low = float(distribution["p2.5"])
        high = float(distribution["p97.5"])
        if invert:
            mean, low, high = 1 - mean, 1 - high, 1 - low
        return (
            f"{_fmt(mean)} ({_fmt(distribution['sd'])})",
            f"[{_fmt(low)}, {_fmt(high)}]",
        )

    router_risk_split = split_cells(router["split_mFDR"])
    router_coverage_split = split_cells(router["split_coverage"])
    router_referral_split = split_cells(router["split_coverage"], invert=True)
    system_risk_split = split_cells(proposed["split_mFDR"])
    system_coverage_split = split_cells(proposed["split_coverage"])
    system_referral_split = split_cells(proposed["split_coverage"], invert=True)
    conditional_split = split_cells(proposed["split_conditional_specialist_coverage"])
    conditional_referral_split = split_cells(
        proposed["split_conditional_specialist_coverage"], invert=True
    )
    conditional_coverage = proposed["conditional_specialist_coverage"]

    tab1 = [
        r"\begin{table*}[t]", r"\centering",
        r"\caption{Patient-level system performance at the prespecified operating point ($\alpha=0.20$, $\beta=0.01$). Split percentiles summarize sensitivity across 500 patient-disjoint partitions and are not confidence intervals. Patient-bootstrap intervals use 1,000 source-stratified patient resamples. Counts pool held-out patient decisions across splits and are not unique-patient cohort counts.}",
        r"\label{tab:patient_system_summary}",
        r"\begin{adjustbox}{max width=\textwidth,center}", r"\small",
        r"\begin{tabular}{llcccc}", r"\toprule",
        r"Category & Metric & Estimate & Split mean (SD) & Split $[2.5,97.5]$ & 95\% patient-bootstrap CI \\",
        r"\midrule",
        f"Risk & Router mFDR & {_fmt(router['mFDR'])} & {router_risk_split[0]} & {router_risk_split[1]} & --- \\\\",
        f"Risk & System mFDR & {_fmt(proposed['mFDR_sys'])} & {system_risk_split[0]} & {system_risk_split[1]} & [{_fmt(bootstrap['proposed']['mFDR_sys']['ci_low'])}, {_fmt(bootstrap['proposed']['mFDR_sys']['ci_high'])}] \\\\",
        f"Risk & Specialist macro mFDR & {_fmt(proposed['mFDR_macro'])} & --- & --- & --- \\\\",
        f"Risk & Worst-clinic mFDR & {_fmt(proposed['mFDR_worst'])} & --- & --- & --- \\\\",
        r"\midrule",
        f"Coverage & Router coverage & {_fmt(router['coverage'])} & {router_coverage_split[0]} & {router_coverage_split[1]} & --- \\\\",
        f"Coverage & Specialist coverage $\\mid R$ & {_fmt(conditional_coverage)} & {conditional_split[0]} & {conditional_split[1]} & --- \\\\",
        f"Coverage & System coverage & {_fmt(proposed['coverage'])} & {system_coverage_split[0]} & {system_coverage_split[1]} & [{_fmt(bootstrap['proposed']['coverage']['ci_low'])}, {_fmt(bootstrap['proposed']['coverage']['ci_high'])}] \\\\",
        r"\midrule",
        f"Referral & Router referral & {_fmt(1-router['coverage'])} & {router_referral_split[0]} & {router_referral_split[1]} & --- \\\\",
        f"Referral & Specialist referral $\\mid R$ & {_fmt(1-conditional_coverage)} & {conditional_referral_split[0]} & {conditional_referral_split[1]} & --- \\\\",
        f"Referral & System referral & {_fmt(proposed['referral'])} & {system_referral_split[0]} & {system_referral_split[1]} & --- \\\\",
        r"\midrule",
        f"Counts & AI-correct decisions & {proposed['selected']-proposed['errors']:,} & --- & --- & --- \\\\",
        f"Counts & AI-error decisions & {proposed['errors']:,} & --- & --- & --- \\\\",
        f"Counts & Human-referred decisions & {proposed['test_patient_decisions']-proposed['selected']:,} & --- & --- & --- \\\\",
        r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table*}", "",
    ]
    (directory / "tab1_system_summary.tex").write_text("\n".join(tab1), encoding="utf-8")

    tab2 = [
        r"\begin{table*}[t]", r"\centering",
        r"\caption{Patient-level clinic-specific specialist risk and coverage at the prespecified operating point. Base error is measured on the routed stream before specialist rejection. Patient-bootstrap intervals use source-stratified patient resampling; $\sigma(\mathrm{FDP})$ summarizes the 500 patient-disjoint splits. Counts pool held-out patient decisions across splits.}",
        r"\label{tab:patient_per_disease}",
        r"\begin{adjustbox}{max width=\textwidth,center}", r"\small",
        r"\begin{tabular}{rlcccccrrr}", r"\toprule",
        r"$k$ & Specialist pathway & Base error & mFDR & 95\% CI & $\sigma(\mathrm{FDP})$ & Coverage & Routed & Accepted & Errors \\",
        r"\midrule",
    ]
    for row, ci, exchange, contamination in zip(
        proposed["per_disease"], bootstrap["per_disease"],
        block["exchangeability"], block["contamination"],
    ):
        tab2.append(
            f"{row['disease_id']} & {_escape(row['disease_name'])} & "
            f"{_fmt(exchange['base_error_test'])} & {_fmt(row['mFDR'])} & "
            f"[{_fmt(ci['mFDR_sys']['ci_low'])}, {_fmt(ci['mFDR_sys']['ci_high'])}] & "
            f"{_fmt(row['split_FDP']['sd'])} & {_fmt(row['coverage'])} & "
            f"{contamination['routed']:,} & {row['selected']:,} & {row['errors']:,} \\\\"
        )
    tab2.extend([
        r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table*}", "",
    ])
    (directory / "tab2_per_disease_risk.tex").write_text("\n".join(tab2), encoding="utf-8")

    tab3 = [
        r"\begin{table*}[t]", r"\centering",
        r"\caption{Patient-level effective calibration size, threshold stability and feasibility at the prespecified operating point. Threshold summaries use feasible splits only.}",
        r"\label{tab:patient_calibration_feasibility}",
        r"\begin{adjustbox}{max width=\textwidth,center}", r"\small",
        r"\begin{tabular}{rlrrrrrrrr}", r"\toprule",
        r"$k$ & Decision point & Mean (SD) & Median & Min & $[2.5,97.5]$ & Mean threshold & Threshold SD & Feasible splits & Infeasible frequency \\",
        r"\midrule",
    ]
    router_size = router["effective_size"]
    router_threshold = router["threshold"]
    tab3.append(
        f"--- & Router & {router_size['mean']:.1f} ({router_size['sd']:.1f}) & "
        f"{router_size['median']:.0f} & {router_size['min']:.0f} & "
        f"[{router_size['p2.5']:.0f},{router_size['p97.5']:.0f}] & "
        f"{_fmt(router_threshold['mean'])} & {_fmt(router_threshold['sd'])} & "
        f"{suite['meta']['n_splits']-router['infeasible_splits']} & "
        f"{router['infeasible_splits']/suite['meta']['n_splits']:.3f} \\\\"
    )
    tab3.append(r"\midrule")
    for row in block["calibration"]:
        size = row["effective_size"]
        tab3.append(
            f"{row['disease_id']} & {_escape(row['disease_name'])} & "
            f"{size['mean']:.1f} ({size['sd']:.1f}) & {size['median']:.0f} & "
            f"{size['min']} & [{size['p2.5']:.1f},{size['p97.5']:.1f}] & "
            f"{_fmt(row['threshold']['mean'])} & {_fmt(row['threshold']['sd'])} & "
            f"{row['feasible_splits']} & {_fmt(row['infeasible_frequency'], 3)} \\\\"
        )
    tab3.extend([
        r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table*}", "",
    ])
    (directory / "tab3_calibration_feasibility.tex").write_text("\n".join(tab3), encoding="utf-8")

    tab4 = [
        r"\begin{table*}[t]", r"\centering",
        r"\caption{Patient-level routing contamination and accepted-error decomposition at the prespecified operating point. Counts pool held-out patient decisions across splits.}",
        r"\label{tab:patient_contamination}",
        r"\begin{adjustbox}{max width=\textwidth,center}", r"\small",
        r"\begin{tabular}{rlcccccrrrrrr}", r"\toprule",
        r"$k$ & Pathway & $\rho^{route}$ & $\rho^{acc}$ & Wrong-route referral & Route-error share & Diagnostic-error share & Route errors & Diagnostic errors & Total errors & Routed & Wrong-routed & Accepted \\",
        r"\midrule",
    ]
    for row in block["contamination"]:
        total_errors = row["accepted_wrong"] + row["accepted_diag_error"]
        tab4.append(
            f"{row['disease_id']} & {_escape(row['disease_name'])} & "
            f"{_fmt(row['rho_route'],3)} & {_fmt(row['rho_acc'],3)} & "
            f"{_fmt(row['wrong_route_referral'],3)} & "
            f"{_fmt(row['route_error_share'],3)} & {_fmt(row['diagnostic_error_share'],3)} & "
            f"{row['accepted_wrong']:,} & {row['accepted_diag_error']:,} & "
            f"{total_errors:,} & {row['routed']:,} & {row['wrong_routed']:,} & "
            f"{row['accepted']:,} \\\\"
        )
    tab4.extend([
        r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table*}", "",
    ])
    (directory / "tab4_contamination.tex").write_text("\n".join(tab4), encoding="utf-8")

    tab5 = [
        r"\begin{table*}[t]", r"\centering",
        r"\caption{Patient-level comparison with baseline workflows over the specialist-risk grid at fixed router target $\beta=0.01$. Split SD summarizes 500 patient-disjoint partitions. $K_{\mathrm{meet}}$ is reported among evaluable pathways.}",
        r"\label{tab:patient_baselines}",
        r"\textbf{(a) Guarantee attributable to each construction}", r"\par\smallskip",
        r"\begin{tabular}{ll}", r"\toprule", r"System & Guarantee scope \\", r"\midrule",
        r"Full automation & None \\",
        r"Router-only rejection & Router only \\",
        r"Specialist-only rejection & Specialist and system on unfiltered streams \\",
        r"Shared threshold & Pooled system only, not pathway-specific \\",
        r"Oracle-clean calibration & None on the contaminated operational stream \\",
        r"\textbf{Proposed LEC} & \textbf{Router, pathway-specific specialists and system} \\",
        r"\bottomrule", r"\end{tabular}", r"\par\bigskip",
        r"\textbf{(b) Outcomes across the specialist-risk grid}", r"\par\smallskip",
        r"\begin{adjustbox}{max width=\textwidth,center}", r"\small",
        r"\begin{tabular}{llccccccc}", r"\toprule",
        r"$\alpha$ & System & mFDR (split SD) & Coverage & Referral & Macro mFDR & Worst mFDR & $K_{\mathrm{meet}}$ & System mFDR $\leq\alpha$ \\",
        r"\midrule",
    ]
    selected_alphas = {0.05, 0.10, 0.15, 0.20}
    for alpha_row in suite["alpha_grid"]:
        alpha = float(alpha_row["alpha"])
        if alpha not in selected_alphas:
            continue
        for method in METHODS:
            row = alpha_row["methods"][method]
            label = METHOD_LABELS[method]
            if method == "proposed":
                label = r"\textbf{Proposed LEC}"
            meets = (
                "Yes"
                if row["mFDR_sys"] is not None and row["mFDR_sys"] <= alpha
                else "No"
            )
            tab5.append(
                f"{alpha:.2f} & {label} & {_fmt(row['mFDR_sys'])} "
                f"({_fmt(row['split_mFDR']['sd'],3)}) & {_fmt(row['coverage'])} & "
                f"{_fmt(row['referral'])} & {_fmt(row['mFDR_macro'])} & "
                f"{_fmt(row['mFDR_worst'])} & {row['K_meet']}/{row['K_estimable']} & "
                f"{meets} \\\\"
            )
        tab5.append(r"\midrule")
    tab5[-1] = r"\bottomrule"
    tab5.extend([
        r"\end{tabular}", r"\end{adjustbox}", r"\end{table*}", "",
    ])
    (directory / "tab5_baselines.tex").write_text("\n".join(tab5), encoding="utf-8")

    audit = suite["input_audit"]
    tab6 = [
        r"\begin{table*}[t]", r"\centering",
        r"\caption{Patient-level evaluation cohort and derived records by source dataset.}",
        r"\label{tab:patient_cohort}",
        r"\begin{tabular}{lrrrr}", r"\toprule",
        r"Source dataset & Pathway & Patients & Derived records & Multi-record patients \\",
        r"\midrule",
    ]
    for source, row in audit["sources"].items():
        tab6.append(
            f"{_escape(source)} & {row['disease_id']} & {row['patients']:,} & "
            f"{row['records']:,} & {row['multi_record_patients']:,} \\\\"
        )
    tab6.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    (directory / "tab6_cohort.tex").write_text("\n".join(tab6), encoding="utf-8")


def write_manifest(suite: dict[str, Any], directory: Path) -> None:
    lines = [
        "# Full patient-level figure and table suite",
        "",
        "All empirical quantities use one aggregated decision per patient and source-stratified, patient-disjoint C_R/C_S/T partitions.",
        "",
        "## Figures",
        "",
        "- Fig. 1: clinic-specific patient mFDR forest plot.",
        "- Fig. 2: beta sensitivity.",
        "- Fig. 3: alpha sensitivity.",
        "- Fig. 4: baseline risk-coverage comparison.",
        "- Fig. 5: routing contamination and accepted-error source.",
        "- Fig. 6: calibration-size sensitivity.",
        "- Fig. 7: illustrative utility heatmap.",
        "- Fig. 8: baseline risk-coverage frontier over alpha.",
        "- Fig. 9: router contribution over beta.",
        "- Fig. 10: calibration-test comparability diagnostics.",
        "- Fig. 11: routing-error containment over alpha.",
        "- Fig. 12: detailed effective calibration size (duplicate view retained for traceability).",
        "",
        "## Tables",
        "",
        "- Table 1: primary system risk, coverage, referral and decision counts.",
        "- Table 2: pathway-specific risk, confidence intervals and held-out counts.",
        "- Table 3: calibration evidence, threshold stability and feasibility.",
        "- Table 4: routing contamination and accepted-error decomposition.",
        "- Table 5: guarantee scope and six-method comparison over alpha.",
        "- Table 6: source-dataset patient and derived-record audit.",
        "",
        "## Interpretation boundary",
        "",
        "Repeated-split percentiles describe split sensitivity. The primary 95% confidence intervals are the separate 1,000-replicate patient bootstrap intervals in patient_level_results.json.",
        "",
    ]
    (directory / "README.md").write_text("\n".join(lines), encoding="utf-8")
