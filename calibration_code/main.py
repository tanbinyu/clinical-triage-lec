"""Backward-compatible entry point for the Step 2–6 experiment workflow."""
import argparse
from pathlib import Path

from pipeline import config
from pipeline.contracts import ExperimentConfig
from pipeline.experiment.runner import run_experiment


def parse_args():
    parser = argparse.ArgumentParser(
        description="Nature Pipeline Steps 2-6: mFDR-controlled medical image classification"
    )
    parser.add_argument("--phase1", default="result/json/step1/phase_1.json",
                        help="Path to phase_1.json")
    parser.add_argument("--phase2_dir", default="result/json/step1",
                        help="Directory containing {dataset}_phase_2.json files")
    parser.add_argument("--output_dir", default="result/json",
                        help="Output root directory")
    parser.add_argument("--num_splits", type=int,
                        default=config.DEFAULTS["num_splits"],
                        help="Number of random split epochs")
    parser.add_argument("--split_ratio", type=float,
                        default=config.DEFAULTS["split_ratio"],
                        help="Calibration set ratio (per dataset)")
    parser.add_argument("--beta", type=float,
                        default=config.DEFAULTS["beta"],
                        help="Target mFDR for Phase 1 (routing)")
    parser.add_argument("--alpha_A", type=float,
                        default=config.DEFAULTS["alpha_A"],
                        help="Target mFDR for Phase 2 Branch A (AI)")
    parser.add_argument("--alpha_B", type=float,
                        default=config.DEFAULTS["alpha_B"],
                        help="Target mFDR for Phase 2 Branch B (Human)")
    parser.add_argument("--lambda_1", type=float,
                        default=config.DEFAULTS["lambda_1"],
                        help="Weight for u1 in Branch A fused score")
    parser.add_argument("--lambda_2", type=float,
                        default=config.DEFAULTS["lambda_2"],
                        help="Weight for u2 in Branch A fused score")
    parser.add_argument("--steps", nargs="+", type=int,
                        default=[2, 3, 4, 5, 6], choices=[2, 3, 4, 5, 6],
                        help="Steps to run (default: all)")
    return parser.parse_args()


def main():
    args = parse_args()
    experiment_config = ExperimentConfig(
        phase1=Path(args.phase1),
        phase2_dir=Path(args.phase2_dir),
        output_dir=Path(args.output_dir),
        num_splits=args.num_splits,
        split_ratio=args.split_ratio,
        beta=args.beta,
        alpha_A=args.alpha_A,
        alpha_B=args.alpha_B,
        lambda_1=args.lambda_1,
        lambda_2=args.lambda_2,
        steps=tuple(args.steps),
    )
    run_experiment(experiment_config)


if __name__ == "__main__":
    main()
