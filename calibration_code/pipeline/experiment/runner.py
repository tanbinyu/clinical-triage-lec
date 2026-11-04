"""Compatibility-preserving orchestration for Steps 2–6."""
import json

from pipeline.contracts import load_and_validate_phase1
from pipeline.core.artifacts import ArtifactPaths
from pipeline.experiment import (
    stage2_split,
    stage3_calibrate_phase1,
    stage4_split_branches,
    stage5_aggregate,
    stage6_calibrate_phase2,
)


def run_experiment(config, validate_inputs=True):
    """Run requested stages while retaining every historical JSON name/schema."""
    paths = ArtifactPaths(config.output_dir)
    paths.ensure_stage_directories()

    if validate_inputs:
        phase1 = load_and_validate_phase1(config)
    else:
        with open(config.phase1) as f:
            phase1 = json.load(f)

    if 2 in config.steps:
        print(
            f"[Step 2] Generating splits (num_splits={config.num_splits}, "
            f"split_ratio={config.split_ratio})..."
        )
        splits = stage2_split.run(phase1, config.num_splits, config.split_ratio)
        with open(paths.splits_phase1, "w") as f:
            json.dump(splits, f, indent=2)
        print("[Step 2] Done. Saved step2/splits_phase_1.json")

    if 3 in config.steps:
        print(f"[Step 3] Phase 1 LEC calibration (beta={config.beta})...")
        with open(paths.splits_phase1) as f:
            splits = json.load(f)
        lec_p1 = stage3_calibrate_phase1.run(phase1, splits, config.beta)
        with open(paths.lec_phase1(config.beta), "w") as f:
            json.dump(lec_p1, f, indent=2)
        infeasible = sum(1 for split in lec_p1["splits"] if split["t_hat"] is None)
        print(
            f"[Step 3] Done. base_error_rate={lec_p1['base_error_rate']}, "
            f"infeasible={infeasible}/{len(lec_p1['splits'])}"
        )

    if 4 in config.steps:
        print("[Step 4] Splitting into Branch A / Branch B...")
        with open(paths.splits_phase1) as f:
            splits = json.load(f)
        with open(paths.lec_phase1(config.beta)) as f:
            lec_p1 = json.load(f)
        splits_a, splits_b = stage4_split_branches.run(phase1, splits, lec_p1)
        with open(paths.splits_branch_a(config.beta), "w") as f:
            json.dump(splits_a, f, indent=2)
        with open(paths.splits_branch_b(config.beta), "w") as f:
            json.dump(splits_b, f, indent=2)
        print(
            f"[Step 4] Done. Branch A epochs={len(splits_a['splits'])}, "
            f"Branch B epochs={len(splits_b['splits'])}"
        )

    if 5 in config.steps:
        print(
            f"[Step 5] Aggregating per disease (lambda_1={config.lambda_1}, "
            f"lambda_2={config.lambda_2})..."
        )
        stage5_aggregate.run(
            phase1,
            config.phase2_dir,
            paths.step5_dir,
            config.lambda_1,
            config.lambda_2,
        )

    if 6 in config.steps:
        print(
            f"[Step 6] Phase 2 LEC calibration (alpha_A={config.alpha_A}, "
            f"alpha_B={config.alpha_B})..."
        )
        with open(paths.splits_branch_a(config.beta)) as f:
            splits_a = json.load(f)
        with open(paths.splits_branch_b(config.beta)) as f:
            splits_b = json.load(f)
        stage6_calibrate_phase2.run(
            paths.step5_dir,
            paths.step6_dir,
            splits_a,
            splits_b,
            config.alpha_A,
            config.alpha_B,
            config.beta,
        )

    print("[Pipeline] All requested steps completed.")
