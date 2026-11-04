"""Lightweight compatibility tests for the reorganised Step 2–6 workflow."""
import json
import tempfile
import unittest
from pathlib import Path

from pipeline.contracts import ExperimentConfig
from pipeline.core.artifacts import ArtifactPaths
from pipeline.experiment.runner import run_experiment
from pipeline.experiment.stage2_split import run as canonical_stage2
from pipeline.step2_split import run as legacy_stage2


class ExperimentContractTests(unittest.TestCase):
    def test_legacy_stage_import_remains_compatible(self):
        self.assertIs(legacy_stage2, canonical_stage2)

    def test_artifact_paths_keep_historical_names(self):
        paths = ArtifactPaths(Path("result/json"))
        self.assertEqual(str(paths.splits_phase1).replace("\\", "/"),
                         "result/json/step2/splits_phase_1.json")
        self.assertEqual(paths.lec_phase1(0.1).name, "lec_phase_1_beta_0.1.json")
        self.assertEqual(paths.splits_branch_a(0.01).name,
                         "splits_branch_a_beta_0.01.json")

    def test_fixture_run_preserves_stage_schema(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            phase1 = {
                "s1": {"uncertainty": 0.1, "correctness": 1,
                       "predicted_disease": 0, "true_disease": 0,
                       "source_dataset": "LungHist700"},
                "s2": {"uncertainty": 0.2, "correctness": 1,
                       "predicted_disease": 0, "true_disease": 0,
                       "source_dataset": "LungHist700"},
                "s3": {"uncertainty": 0.8, "correctness": 0,
                       "predicted_disease": 0, "true_disease": 0,
                       "source_dataset": "LungHist700"},
                "s4": {"uncertainty": 0.9, "correctness": 1,
                       "predicted_disease": 0, "true_disease": 0,
                       "source_dataset": "LungHist700"},
            }
            phase2 = {
                sid: {"uncertainty": sample["uncertainty"],
                      "correctness": sample["correctness"]}
                for sid, sample in phase1.items()
            }
            phase1_path = input_dir / "phase_1.json"
            with open(phase1_path, "w") as f:
                json.dump(phase1, f)
            with open(input_dir / "phase_2_LungHist700.json", "w") as f:
                json.dump(phase2, f)

            config = ExperimentConfig(
                phase1=phase1_path,
                phase2_dir=input_dir,
                output_dir=output_dir,
                num_splits=2,
                split_ratio=0.5,
                beta=0.5,
                alpha_A=0.5,
                alpha_B=0.5,
                lambda_1=0.5,
                lambda_2=0.5,
            )
            run_experiment(config)

            self.assertTrue((output_dir / "step2" / "splits_phase_1.json").is_file())
            self.assertTrue((output_dir / "step3" / "lec_phase_1_beta_0.5.json").is_file())
            self.assertTrue((output_dir / "step4" / "splits_branch_a_beta_0.5.json").is_file())
            self.assertTrue((output_dir / "step5" / "branch_a_disease_0.json").is_file())
            step6 = output_dir / "step6" / "lec_branch_a_disease_0_beta_0.5_alphaA_0.5.json"
            self.assertTrue(step6.is_file())
            with open(step6) as f:
                self.assertEqual(set(json.load(f)), {"alpha", "base_error_rate", "splits"})


if __name__ == "__main__":
    unittest.main()
