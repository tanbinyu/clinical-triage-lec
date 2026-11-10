"""Create a de-identified reviewer bundle from a verified completed run."""
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_PROJECT_FILES = (
    PROJECT_ROOT / "requirements-lock.txt",
    PROJECT_ROOT / "environment.yml",
    PROJECT_ROOT / "configs" / "patient_clustered_source_cap_1850_beta_001.json",
    PROJECT_ROOT / "reproducibility" / "README.md",
    PROJECT_ROOT / "reproducibility" / "REVIEWER_ACCESS.md",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        help=(
            "verified provenance JSON; defaults to <run-dir>/run_provenance.json "
            "or reproducibility/manuscript_run_manifest.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "reproducibility" / "Reviewer_Reproducibility_Bundle.zip",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    provenance_path = args.manifest.resolve() if args.manifest else (
        run_dir / "run_provenance.json"
    )
    if not provenance_path.is_file() and args.manifest is None:
        provenance_path = PROJECT_ROOT / "reproducibility" / "manuscript_run_manifest.json"
    if not provenance_path.is_file():
        raise FileNotFoundError(
            "Missing verified provenance JSON; complete the run or invoke "
            "tools/build_run_manifest.py --run-dir <RUN> --require-results first."
        )
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    chain = provenance.get("bootstrap_chain", {})
    if chain.get("status") != "complete":
        raise ValueError("Reviewer bundle requires a complete result provenance chain")
    for key in ("bootstrap_replicates", "reference_splits"):
        entry = chain.get("results", {}).get(key, {})
        if not entry.get("row_count_matches_config"):
            raise ValueError(f"Result validation failed for {key}")

    run_files = [provenance_path]
    for entry in chain["results"].values():
        run_files.append(run_dir / entry["name"])
    eligibility = run_dir / "metrics" / "eligibility_allocation_audit.json"
    input_audit = run_dir / "metrics" / "input_audit.json"
    run_files.extend(path for path in (eligibility, input_audit) if path.is_file())

    missing = [path for path in (*REQUIRED_PROJECT_FILES, *run_files) if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing bundle files: " + ", ".join(map(str, missing)))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in REQUIRED_PROJECT_FILES:
            archive.write(path, Path("code") / path.relative_to(PROJECT_ROOT))
        for path in sorted(set(run_files)):
            try:
                relative = path.relative_to(run_dir)
            except ValueError:
                relative = Path(path.name)
            archive.write(path, Path("run") / relative)
    print(f"Reviewer bundle written to {args.output.resolve()}")
    print("Raw frozen JSON exports and patient identifiers were not included.")


if __name__ == "__main__":
    main()
