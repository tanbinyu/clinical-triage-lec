"""Artifact paths, compatibility checks, and report-only rendering.

The numerical experiment remains in ``analysis`` and ``suite``. This module
only reads their JSON outputs and writes figures, tables, and a provenance
manifest into the organized patient-level layout.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .analysis import PatientExperimentConfig


@dataclass(frozen=True)
class PatientArtifactPaths:
    root: Path

    @property
    def primary_json(self) -> Path:
        return self.root / "metrics" / "patient_level_results.json"

    @property
    def full_suite_json(self) -> Path:
        return self.root / "metrics" / "patient_level_full_suite.json"

    @property
    def primary_figures(self) -> Path:
        return self.root / "figures" / "primary"

    @property
    def full_suite_figures(self) -> Path:
        return self.root / "figures" / "full_suite"

    @property
    def primary_tables(self) -> Path:
        return self.root / "tables" / "primary"

    @property
    def full_suite_tables(self) -> Path:
        return self.root / "tables" / "full_suite"

    @property
    def manifest(self) -> Path:
        return self.root / "manifest.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_primary_config(
    primary: dict[str, Any], config: PatientExperimentConfig
) -> None:
    """Refuse to mix a saved primary run with different suite parameters."""
    saved = primary["meta"]["config"]
    checked = (
        "alpha", "beta", "n_splits", "seed", "aggregation",
        "router_calib_ratio", "specialist_calib_ratio", "test_ratio",
        "lambda_1", "lambda_2", "require_real_ids",
    )
    mismatches = [name for name in checked if saved.get(name) != getattr(config, name)]
    if Path(saved["input_dir"]).resolve() != Path(config.input_dir).resolve():
        mismatches.append("input_dir")
    if mismatches:
        raise ValueError(
            "Existing patient-level primary result does not match suite config: "
            + ", ".join(mismatches)
            + ". Re-run the primary experiment or select its original config."
        )


def validate_report_pair(primary: dict[str, Any], suite: dict[str, Any]) -> None:
    """Check that two saved metric bundles describe the same patient run."""
    primary_config = primary["meta"]["config"]
    suite_meta = suite["meta"]
    mismatches = [
        name for name in ("alpha", "beta", "n_splits", "aggregation")
        if primary_config[name] != suite_meta[name]
    ]
    if primary["input_audit"] != suite["input_audit"]:
        mismatches.append("input_audit")
    for method in ("proposed", "specialist_only"):
        primary_metrics = primary["reference"][method]
        suite_metrics = suite["primary"]["methods"][method]
        for metric in ("mFDR_sys", "coverage", "selected", "errors", "test_patient_decisions"):
            if primary_metrics[metric] != suite_metrics[metric]:
                mismatches.append(f"{method}.{metric}")
    if mismatches:
        raise ValueError(
            "Primary and full-suite metric bundles do not match: "
            + ", ".join(mismatches)
        )


def render_saved_patient_reports(
    run_dir: Path, *, include_full_suite: bool = True
) -> PatientArtifactPaths:
    """Regenerate reports from saved metrics, without rerunning calibration."""
    paths = PatientArtifactPaths(Path(run_dir).resolve())
    primary = json.loads(paths.primary_json.read_text(encoding="utf-8"))
    suite = None
    if include_full_suite:
        suite = json.loads(paths.full_suite_json.read_text(encoding="utf-8"))
        validate_report_pair(primary, suite)

    from .reporting import write_all_reports

    write_all_reports(
        primary, figures_dir=paths.primary_figures, tables_dir=paths.primary_tables
    )
    if suite is not None:
        from .suite_reporting import write_full_suite_outputs

        write_full_suite_outputs(
            suite, primary,
            figure_dir=paths.full_suite_figures,
            table_dir=paths.full_suite_tables,
        )

    metric_paths = [paths.primary_json]
    report_dirs = [paths.primary_figures, paths.primary_tables]
    if suite is not None:
        metric_paths.append(paths.full_suite_json)
        report_dirs.extend((paths.full_suite_figures, paths.full_suite_tables))
    manifest = {
        "schema_version": 1,
        "unit_of_analysis": "patient",
        "source_metrics": {
            str(path.relative_to(paths.root)).replace("\\", "/"): _sha256(path)
            for path in metric_paths
        },
        "report_groups": {
            str(directory.relative_to(paths.root)).replace("\\", "/"): sorted(
                str(path.relative_to(paths.root)).replace("\\", "/")
                for path in directory.iterdir() if path.is_file()
            )
            for directory in report_dirs
        },
        "compatibility_note": (
            "Existing files directly under figures/ and tables/, and in their paper/ "
            "subdirectories, are retained as legacy outputs. Organized report "
            "directories are generated from the saved metrics listed above."
        ),
    }
    paths.manifest.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return paths
