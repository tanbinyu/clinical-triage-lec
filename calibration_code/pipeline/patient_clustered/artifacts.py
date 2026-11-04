"""Build and optionally install the complete Paper2.0 artifact set.

The deliverable contains exactly 11 figure PDFs and 6 LaTeX table fragments.
Nine figures and all numerical tables are rendered from saved patient-clustered
metrics.  The two workflow schematics are static manuscript assets and are
copied unchanged, because they do not encode an analysis unit or numerical
result.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from .paper2_reporting import RESULT_FIGURE_PATHS, render_paper2_result_figures
from .paper_reporting import (
    write_paper2_table_set,
    write_supplementary_tables,
    write_tables,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PAPER2_ROOT = (
    PROJECT_ROOT.parent / "Paper2.0" / "Clinical_Triage__New_Version_"
)
STATIC_FIGURE_PATHS = (Path("overview.pdf"), Path("method_overview.pdf"))
TABLE_PATHS = tuple(Path("newtab") / f"tab{index}.tex" for index in range(1, 7))


def _load_json(path: Path, *, optional: bool = False) -> dict[str, Any] | None:
    if optional and not path.exists():
        return None
    if not path.exists():
        raise FileNotFoundError(f"Required patient-clustered metric is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def expected_artifact_paths(root: Path) -> tuple[list[Path], list[Path]]:
    figures = [root / "figures" / path for path in STATIC_FIGURE_PATHS]
    figures.extend(root / "figures" / path for path in RESULT_FIGURE_PATHS)
    tables = [root / "tables" / path for path in TABLE_PATHS]
    return figures, tables


def validate_paper2_artifact_set(root: Path) -> tuple[list[Path], list[Path]]:
    figures, tables = expected_artifact_paths(root)
    missing = [path for path in figures + tables if not path.is_file() or path.stat().st_size == 0]
    if missing:
        details = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(f"Incomplete Paper2.0 artifact set:\n{details}")
    if len(figures) != 11 or len(tables) != 6:
        raise RuntimeError("Paper2.0 contract must contain exactly 11 figures and 6 tables")
    return figures, tables


def install_paper2_artifacts(artifact_root: Path, manuscript_root: Path) -> None:
    """Copy the validated set into the exact paths already used by LaTeX."""
    figures, tables = validate_paper2_artifact_set(artifact_root)
    for source in figures:
        relative = source.relative_to(artifact_root / "figures")
        target = manuscript_root / "figures" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    for source in tables:
        relative = source.relative_to(artifact_root / "tables")
        target = manuscript_root / "tables" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def render_saved_paper2_artifacts(
    run_dir: Path,
    *,
    paper2_root: Path | None = None,
    install: bool = False,
) -> Path:
    """Render the 17-item manuscript set from existing metric files.

    This is the fast path after styling or caption-compatible figure changes:
    no split, calibration or bootstrap calculation is repeated.
    """
    run_dir = Path(run_dir).resolve()
    paper2_root = Path(paper2_root or DEFAULT_PAPER2_ROOT).resolve()
    metrics = run_dir / "metrics"
    suite_path = metrics / "patient_clustered_paper_suite.json"
    primary_path = metrics / "patient_clustered_results.json"
    secondary_path = metrics / "all_decisions_sensitivity.json"
    if not secondary_path.exists():
        # Backward-compatible rendering of runs created before the primary unit
        # was changed to one decision per patient.
        secondary_path = metrics / "patient_balanced_sensitivity.json"
    strict_path = metrics / "strict_patient_subset.json"
    bootstrap_csv = metrics / "patient_cluster_bootstrap_replicates.csv"

    suite = _load_json(suite_path)
    primary = _load_json(primary_path)
    secondary = _load_json(secondary_path, optional=True)
    strict = _load_json(strict_path, optional=True)
    if not bootstrap_csv.exists():
        raise FileNotFoundError(f"Required paired bootstrap rows are missing: {bootstrap_csv}")
    if suite["meta"]["beta"] != primary["meta"]["config"]["beta"]:
        raise ValueError("Primary and suite beta values do not match")
    if suite["meta"]["alpha"] != primary["meta"]["config"]["alpha"]:
        raise ValueError("Primary and suite alpha values do not match")

    artifact_root = run_dir / "deliverables" / "paper2"
    figure_root = artifact_root / "figures"
    table_root = artifact_root / "tables"
    # Exact stale name from the former record-bootstrap manuscript contract.
    # It is removed only from generated/manuscript figure directories.
    (figure_root / "fig13_record_bootstrap.pdf").unlink(missing_ok=True)
    render_paper2_result_figures(
        suite,
        primary,
        patient_balanced=secondary,
        strict_patient=strict,
        bootstrap_csv=bootstrap_csv,
        figure_root=figure_root,
    )

    for relative in STATIC_FIGURE_PATHS:
        source = paper2_root / "figures" / relative
        if not source.exists():
            raise FileNotFoundError(
                f"Static workflow schematic is missing from Paper2.0: {source}"
            )
        target = figure_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    main_tables = run_dir / "tables" / "paper"
    supplementary_tables = run_dir / "tables" / "supplementary"
    write_tables(suite, primary, secondary, strict, main_tables)
    write_supplementary_tables(suite, supplementary_tables)
    write_paper2_table_set(
        main_tables, supplementary_tables, table_root / "newtab"
    )

    figures, tables = validate_paper2_artifact_set(artifact_root)
    publication_ready = (
        primary["meta"].get("input_cohort_scope") == "test_only"
        and primary["meta"].get("calibration_policy_status")
        == "prespecified_confirmatory"
    )
    installed = bool(install and publication_ready)
    manifest = {
        "contract": "Paper2.0 patient-clustered 11-figure/6-table set",
        "authoritative_input": primary["input_audit"]["input_dir"],
        "analysis_unit": primary["meta"]["unit_of_analysis"],
        "primary_analysis_unit": primary["meta"].get("primary_analysis_unit"),
        "clustering_unit": primary["meta"]["clustering_unit"],
        "beta": suite["meta"]["beta"],
        "alpha": suite["meta"]["alpha"],
        "calibration_alpha": suite["meta"].get(
            "calibration_alpha", suite["meta"]["alpha"]
        ),
        "router_score": suite["meta"].get("router_score"),
        "routed_specialist_score": suite["meta"].get(
            "routed_specialist_score", "root phase_2_*.json: uncertainty"
        ),
        "specialist_only_score": suite["meta"].get(
            "specialist_only_score",
            "root phase_2_*.json: uncertainty",
        ),
        "proposed_pathway_definition": suite["meta"].get(
            "proposed_pathway_definition", "router predicted pathway"
        ),
        "specialist_only_pathway_definition": suite["meta"].get(
            "specialist_only_pathway_definition", "router predicted pathway"
        ),
        "input_cohort_scope": primary["meta"].get(
            "input_cohort_scope", "unverified"
        ),
        "calibration_policy_status": primary["meta"].get(
            "calibration_policy_status", "unverified"
        ),
        "max_patients_per_source": primary["meta"].get(
            "max_patients_per_source"
        ),
        "source_sampling": primary["meta"].get("source_sampling"),
        "modality_grouping": (
            "five groups: pathology, CT, X-ray, ultrasound and MRI; "
            "chest radiography and mammography are grouped under X-ray"
        ),
        "publication_ready": publication_ready,
        "manuscript_install_requested": bool(install),
        "manuscript_installed": installed,
        "n_splits": suite["meta"]["n_splits"],
        "n_bootstrap": suite["meta"]["n_bootstrap"],
        "source_metrics": {
            "primary": str(primary_path),
            "suite": str(suite_path),
            "paired_bootstrap_rows": str(bootstrap_csv),
            "secondary_sensitivity": str(secondary_path) if secondary else None,
        },
        "input_files": primary["input_audit"].get("files", []),
        "counts": {
            "figures": 11,
            "data_derived_figures": 9,
            "static_method_figures": 2,
            "tables": 6,
            "total": 17,
        },
        "figures": [
            {
                "path": str(path.relative_to(artifact_root)),
                "role": "static method schematic" if path.name in {
                    item.name for item in STATIC_FIGURE_PATHS
                } else "patient-clustered result",
                "sha256": _sha256(path),
            }
            for path in figures
        ],
        "tables": [
            {
                "path": str(path.relative_to(artifact_root)),
                "role": "patient-clustered result or experiment metadata",
                "sha256": _sha256(path),
            }
            for path in tables
        ],
    }
    (artifact_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    readme = [
        "# Paper2.0 patient-clustered deliverables",
        "",
        "This directory is the single manuscript-facing output of the current workflow.",
        "It contains exactly 11 figure PDFs and 6 LaTeX table fragments.",
        "Nine figures and all numerical table values are rendered from the saved",
        "patient-clustered metric bundles. `overview.pdf` and `method_overview.pdf`",
        "are static workflow schematics copied unchanged from Paper2.0.",
        "",
        "The original JSON inputs and historical record-level outputs are never modified.",
        "See `manifest.json` for provenance, counts and SHA-256 hashes.",
        "The `publication_ready` flag is true only for a test-only cohort with a",
        "calibration policy fixed before confirmatory evaluation.",
        "",
    ]
    (artifact_root / "README.md").write_text("\n".join(readme), encoding="utf-8")
    if install and not publication_ready:
        print(
            "WARNING: manuscript installation skipped because this run is not "
            "confirmatory/publication-ready; deliverables were still generated."
        )
    if installed:
        install_paper2_artifacts(artifact_root, paper2_root)
        (paper2_root / "figures" / "fig13_record_bootstrap.pdf").unlink(missing_ok=True)
    run_manifest_path = run_dir / "manifest.json"
    if run_manifest_path.exists():
        run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
        run_manifest.pop("specialist_score", None)
        run_manifest.update(
            {
                "routed_specialist_score": "root phase_2_*.json: uncertainty",
                "specialist_only_score": (
                    "root phase_2_*.json: uncertainty"
                ),
                "proposed_pathway_definition": "router predicted pathway",
                "specialist_only_pathway_definition": "router predicted pathway",
            }
        )
        run_manifest["paper2_deliverables"] = str(artifact_root)
        run_manifest["paper2_artifact_counts"] = {"figures": 11, "tables": 6, "total": 17}
        run_manifest["paper2_installed_to"] = str(paper2_root) if installed else None
        run_manifest_path.write_text(
            json.dumps(run_manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return artifact_root
