"""Machine-readable provenance for the manuscript patient-clustered run."""
from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .analysis import ClusteredExperimentConfig


MANIFEST_SCHEMA_VERSION = 1
DEPENDENCIES = ("numpy", "matplotlib", "tqdm", "torch")
SUPERSEDED_PUBLIC_COMMIT = "a0452eb3d6491fdfecf3fa5f6f93d54058812ce1"
DECLARED_RELEASE = "v1.0.0"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_state(path: Path) -> dict[str, Any]:
    candidate = path.resolve()
    for root in (candidate, *candidate.parents):
        if (root / ".git").exists():
            try:
                commit = subprocess.check_output(
                    ["git", "-C", str(root), "rev-parse", "HEAD"],
                    text=True,
                    encoding="utf-8",
                    stderr=subprocess.DEVNULL,
                ).strip()
                status = subprocess.check_output(
                    ["git", "-C", str(root), "status", "--porcelain"],
                    text=True,
                    encoding="utf-8",
                    stderr=subprocess.DEVNULL,
                )
                diff = subprocess.check_output(
                    ["git", "-C", str(root), "diff", "--binary", "HEAD"],
                    stderr=subprocess.DEVNULL,
                )
                try:
                    release_commit = subprocess.check_output(
                        ["git", "-C", str(root), "rev-list", "-n", "1", DECLARED_RELEASE],
                        text=True,
                        encoding="utf-8",
                        stderr=subprocess.DEVNULL,
                    ).strip()
                except (OSError, subprocess.CalledProcessError):
                    release_commit = None
                return {
                    "commit": commit,
                    "working_tree_clean": not bool(status.strip()),
                    "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
                    "declared_release": DECLARED_RELEASE,
                    "declared_release_commit": release_commit,
                    "matches_declared_release": (
                        not bool(status.strip()) and commit == release_commit
                    ),
                }
            except (OSError, subprocess.CalledProcessError):
                return {"commit": None, "working_tree_clean": None}
    return {"commit": None, "working_tree_clean": None}


def _dependency_versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for name in DEPENDENCIES:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def _source_tree_digest(project_root: Path) -> dict[str, Any]:
    """Hash executable/configuration files, including files not yet committed."""
    included: list[Path] = []
    for path in project_root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(project_root)
        if any(part in {"result", "__pycache__", ".test-tmp"} for part in relative.parts):
            continue
        if relative.as_posix() == "reproducibility/manuscript_run_manifest.json":
            continue
        if path.suffix.lower() in {".py", ".json", ".txt", ".yml", ".yaml", ".ps1"}:
            included.append(path)
    digest = hashlib.sha256()
    for path in sorted(included, key=lambda item: item.relative_to(project_root).as_posix()):
        relative = path.relative_to(project_root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return {"sha256": digest.hexdigest(), "file_count": len(included)}


def _serializable_config(config: ClusteredExperimentConfig) -> dict[str, Any]:
    values = asdict(config)
    for name, value in list(values.items()):
        if isinstance(value, Path):
            values[name] = f"<LOCAL_{name.upper()}>"
    return values


def _file_entry(path: Path, *, root: Path | None = None) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "name": str(path.relative_to(root)) if root is not None else path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    return entry


def _csv_rows(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def build_run_provenance(
    *,
    config: ClusteredExperimentConfig,
    input_audit: dict[str, Any],
    eligibility_audit: dict[str, Any],
    output_dir: Path,
    require_results: bool,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    config_file = config.config_file.resolve() if config.config_file else None
    result_path = output_dir / "metrics" / "patient_clustered_results.json"
    bootstrap_path = output_dir / "metrics" / "patient_cluster_bootstrap_replicates.csv"
    reference_path = output_dir / "metrics" / "reference_split_metrics.csv"
    expected = {
        "primary_metrics": result_path,
        "bootstrap_replicates": bootstrap_path,
        "reference_splits": reference_path,
    }
    missing = [name for name, path in expected.items() if not path.is_file()]
    if require_results and missing:
        raise FileNotFoundError(
            "Run-result provenance is incomplete; missing " + ", ".join(missing)
        )

    result_files: dict[str, Any] = {}
    for name, path in expected.items():
        if path.is_file():
            result_files[name] = _file_entry(path, root=output_dir)
    if bootstrap_path.is_file():
        replicate_rows = _csv_rows(bootstrap_path)
        result_files["bootstrap_replicates"]["rows"] = replicate_rows
        result_files["bootstrap_replicates"]["expected_rows"] = config.n_bootstrap
        result_files["bootstrap_replicates"]["row_count_matches_config"] = (
            replicate_rows == config.n_bootstrap
        )
        if require_results and replicate_rows != config.n_bootstrap:
            raise ValueError(
                f"Bootstrap CSV contains {replicate_rows} rows; expected "
                f"{config.n_bootstrap}"
            )
    if reference_path.is_file():
        split_rows = _csv_rows(reference_path)
        result_files["reference_splits"]["rows"] = split_rows
        result_files["reference_splits"]["expected_rows"] = config.n_splits
        result_files["reference_splits"]["row_count_matches_config"] = (
            split_rows == config.n_splits
        )
        if require_results and split_rows != config.n_splits:
            raise ValueError(
                f"Reference split CSV contains {split_rows} rows; expected "
                f"{config.n_splits}"
            )

    generated_artifacts = []
    if output_dir.is_dir():
        for path in sorted(output_dir.rglob("*")):
            if not path.is_file() or path.name == "run_provenance.json":
                continue
            if any(part in {"figures", "tables", "deliverables"} for part in path.parts):
                generated_artifacts.append(_file_entry(path, root=output_dir))

    canonical_command = (
        "python run/cli.py clustered-paper "
        "--config configs/patient_clustered_source_cap_1850_beta_001.json "
        "--input-dir ../data/inference "
        "--output-dir result/patient_clustered/"
        "beta-0.01_alpha-0.20-source-cap-1850 --no-install"
    )
    config_entry = None
    if config_file is not None and config_file.is_file():
        config_entry = {
            "path": "calibration_code/configs/" + config_file.name,
            "bytes": config_file.stat().st_size,
            "sha256": sha256_file(config_file),
        }

    public_input_audit = dict(input_audit)
    public_input_audit.pop("input_dir", None)
    run_identity = hashlib.sha256()
    if config_entry is not None:
        run_identity.update(config_entry["sha256"].encode("ascii"))
    for item in sorted(public_input_audit.get("files", []), key=lambda row: row["name"]):
        run_identity.update(item["name"].encode("utf-8"))
        run_identity.update(b"\0")
        run_identity.update(item["sha256"].encode("ascii"))
        run_identity.update(b"\n")
    run_identity.update(
        eligibility_audit["included_after_cap_membership_sha256"].encode("ascii")
    )
    git_state = _git_state(Path(__file__))
    source_tree = _source_tree_digest(Path(__file__).resolve().parents[2])
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "study_run": "manuscript_primary_patient_clustered_alpha020_beta001",
        "code": {
            "repository": "https://github.com/tanbinyu/clinical-triage-lec",
            "release": DECLARED_RELEASE,
            **git_state,
            "executing_source_tree": source_tree,
            "canonical_command": canonical_command,
        },
        "run_identity_sha256": run_identity.hexdigest(),
        "configuration": {
            "file": config_entry,
            "values": _serializable_config(config),
            "seed_map": {
                "fixed_adapter_train_validation_downstream_split": (
                    config.development_split_seed
                ),
                "repeated_downstream_partition_base": config.seed,
                "reference_split_repetition_r": "seed sequence [base_seed, r]",
                "primary_bootstrap_base": config.seed + 100_000,
                "bootstrap_replicate_r": "seed sequence [bootstrap_base, r]",
                "strict_subset_bootstrap_base": config.seed + 200_000,
            },
            "superseded_public_snapshot": {
                "commit": SUPERSEDED_PUBLIC_COMMIT,
                "cap_1850_config_present": False,
                "available_primary_config": (
                    "configs/patient_clustered_source_cap_2000_beta_001.json"
                ),
                "max_patients_per_source": 2000,
                "downstream_role_ratios": {"CR": 0.40, "CS": 0.30, "T": 0.30},
                "evidence_command": (
                    "git show " + SUPERSEDED_PUBLIC_COMMIT
                    + ":calibration_code/configs/"
                    "patient_clustered_source_cap_2000_beta_001.json"
                ),
            },
        },
        "environment": {
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "dependencies": _dependency_versions(),
            "lock_files": [
                "calibration_code/requirements-lock.txt",
                "inference_code/environment.yml",
                "inference_code/requirements.txt",
            ],
        },
        "frozen_export_audit": public_input_audit,
        "eligibility_and_allocation": eligibility_audit,
        "bootstrap_chain": {
            "entry_point": "pipeline.patient_clustered.runner.run_clustered_experiment",
            "generator": "pipeline.patient_clustered.analysis.clustered_bootstrap",
            "resampler": (
                "pipeline.patient_clustered.analysis."
                "patient_balanced_bootstrap_partitions"
            ),
            "calibration_and_evaluation": (
                "pipeline.patient_level.analysis.patient_cluster_bootstrap"
            ),
            "configured_replicates": config.n_bootstrap,
            "results": result_files,
            "status": "complete" if not missing else "results_not_in_public_snapshot",
            "missing_public_result_files": missing,
        },
        "generated_artifacts": generated_artifacts,
        "access": {
            "public_code": (
                "https://github.com/tanbinyu/clinical-triage-lec/tree/v1.0.0"
            ),
            "public_numerical_source_data": (
                "submitted separately as Nature_Communications_Source_Data.xlsx"
            ),
            "restricted_inputs_and_run_bundle": (
                "not public; provide to editors/reviewers through the journal "
                "submission system or a controlled institutional transfer, subject "
                "to source-dataset and checkpoint terms"
            ),
            "reviewer_bundle_builder": (
                "calibration_code/tools/package_reviewer_bundle.py"
            ),
            "contains_patient_identifiers": False,
        },
    }


def write_run_provenance(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
