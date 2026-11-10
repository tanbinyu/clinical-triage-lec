"""Build or verify the de-identified manuscript run provenance manifest."""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.patient_clustered.analysis import eligibility_allocation_audit
from pipeline.patient_clustered.cli import _config_from_json
from pipeline.patient_clustered.data import load_clustered_decisions
from pipeline.patient_clustered.provenance import (
    build_run_provenance,
    write_run_provenance,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT
        / "configs"
        / "patient_clustered_source_cap_1850_beta_001.json",
    )
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "reproducibility" / "manuscript_run_manifest.json",
    )
    parser.add_argument(
        "--require-results",
        action="store_true",
        help="fail unless the 500-split and 1,000-bootstrap output files exist",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = _config_from_json(args.config.resolve())
    overrides = {}
    if args.input_dir is not None:
        overrides["input_dir"] = args.input_dir.resolve()
    if args.run_dir is not None:
        overrides["output_dir"] = args.run_dir.resolve()
    if overrides:
        config = replace(config, **overrides)
    config.validate()

    data, audit = load_clustered_decisions(
        config.input_dir,
        lambda_1=config.lambda_1,
        lambda_2=config.lambda_2,
        require_real_ids=config.require_real_ids,
        require_specialist_only=config.require_specialist_only,
    )
    eligibility = eligibility_allocation_audit(data, config)
    payload = build_run_provenance(
        config=config,
        input_audit=audit,
        eligibility_audit=eligibility,
        output_dir=config.output_dir,
        require_results=args.require_results,
    )
    path = write_run_provenance(args.output.resolve(), payload)
    totals = eligibility["totals"]
    print(f"Manifest written to {path}")
    print(
        "Eligibility/allocation: "
        f"{totals['included_after_source_cap']} patients -> "
        f"{totals['router_calibration_patients']}/"
        f"{totals['specialist_calibration_patients']}/"
        f"{totals['held_out_test_patients']}"
    )
    print("Result status: " + payload["bootstrap_chain"]["status"])


if __name__ == "__main__":
    main()
