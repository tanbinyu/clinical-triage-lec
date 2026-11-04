"""Names and directories for the compatibility-preserving JSON artifacts."""
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ArtifactPaths:
    """Resolve current JSON names without embedding paths in stage code."""

    output_dir: Path

    @property
    def step2_dir(self):
        return self.output_dir / "step2"

    @property
    def step3_dir(self):
        return self.output_dir / "step3"

    @property
    def step4_dir(self):
        return self.output_dir / "step4"

    @property
    def step5_dir(self):
        return self.output_dir / "step5"

    @property
    def step6_dir(self):
        return self.output_dir / "step6"

    def ensure_stage_directories(self):
        for directory in (
            self.step2_dir,
            self.step3_dir,
            self.step4_dir,
            self.step5_dir,
            self.step6_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    @property
    def splits_phase1(self):
        return self.step2_dir / "splits_phase_1.json"

    def lec_phase1(self, beta):
        return self.step3_dir / f"lec_phase_1_beta_{beta}.json"

    def splits_branch_a(self, beta):
        return self.step4_dir / f"splits_branch_a_beta_{beta}.json"

    def splits_branch_b(self, beta):
        return self.step4_dir / f"splits_branch_b_beta_{beta}.json"
