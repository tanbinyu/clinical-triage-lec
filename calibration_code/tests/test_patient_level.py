"""Contract tests for the additive patient-level workflow."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pipeline.patient_level.analysis import PatientExperimentConfig, patient_disjoint_split
from pipeline.patient_level.artifacts import (
    PatientArtifactPaths,
    validate_primary_config,
    validate_report_pair,
)
from pipeline.patient_level.data import load_patient_scores


class PatientLevelTests(unittest.TestCase):
    def _fixture(self, root: Path) -> None:
        phase1 = {}
        phase2 = {}
        for patient in range(9):
            for record in range(2):
                sid = f"sample_{patient}_{record}"
                predicted = 0 if patient != 8 else 1
                phase1[sid] = {
                    "uncertainty": 0.1 + patient * 0.01 + record * 0.001,
                    "correctness": int(predicted == 0),
                    "predicted_disease": predicted,
                    "true_disease": 0,
                    "source_dataset": "LungHist700",
                    "patient_id": f"p{patient}",
                    "patient_id_type": "dataset_provided",
                }
                phase2[sid] = {
                    "uncertainty": 0.2 + patient * 0.01 + record * 0.001,
                    "correctness": 1,
                    "predicted": patient % 3,
                    "true": patient % 3,
                    "patient_id": f"p{patient}",
                    "patient_id_type": "dataset_provided",
                }
        (root / "phase_1.json").write_text(json.dumps(phase1), encoding="utf-8")
        (root / "phase_2_LungHist700.json").write_text(
            json.dumps(phase2), encoding="utf-8"
        )

    def test_aggregation_produces_one_row_per_patient(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._fixture(root)
            scores, audit = load_patient_scores(root, require_real_ids=True)
            self.assertEqual(scores.n_records, 18)
            self.assertEqual(scores.n_patients, 9)
            self.assertEqual(audit["patient_id_type_counts"], {"dataset_provided": 9})
            self.assertTrue(np.all(scores.record_counts == 2))

    def test_three_way_split_is_patient_disjoint_and_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._fixture(root)
            scores, _ = load_patient_scores(root, require_real_ids=True)
            router, specialist, test = patient_disjoint_split(
                scores, epoch=0, seed=7, ratios=(0.4, 0.3, 0.3)
            )
            self.assertFalse(set(router) & set(specialist))
            self.assertFalse(set(router) & set(test))
            self.assertFalse(set(specialist) & set(test))
            self.assertEqual(set(router) | set(specialist) | set(test), set(range(9)))

    def test_provided_patient_ids_need_no_extra_type_field(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._fixture(root)
            phase1 = json.loads((root / "phase_1.json").read_text())
            phase2 = json.loads((root / "phase_2_LungHist700.json").read_text())
            for sample in phase1.values():
                sample.pop("patient_id_type")
            for sample in phase2.values():
                sample.pop("patient_id_type")
            (root / "phase_1.json").write_text(json.dumps(phase1))
            (root / "phase_2_LungHist700.json").write_text(json.dumps(phase2))
            scores, audit = load_patient_scores(root, require_real_ids=True)
            self.assertEqual(scores.n_patients, 9)
            self.assertEqual(audit["patient_id_type_counts"], {"dataset_provided": 9})

    def test_unsupported_id_type_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._fixture(root)
            phase1 = json.loads((root / "phase_1.json").read_text())
            phase2 = json.loads((root / "phase_2_LungHist700.json").read_text())
            for sample in phase1.values():
                sample["patient_id_type"] = "record_derived"
            for sample in phase2.values():
                sample["patient_id_type"] = "record_derived"
            (root / "phase_1.json").write_text(json.dumps(phase1))
            (root / "phase_2_LungHist700.json").write_text(json.dumps(phase2))
            with self.assertRaisesRegex(ValueError, "unsupported patient_id_type"):
                load_patient_scores(root, require_real_ids=True)

    def test_suite_only_rejects_different_primary_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            config = PatientExperimentConfig(
                input_dir=Path(directory), output_dir=Path(directory) / "out"
            )
            saved = {"meta": {"config": {
                "input_dir": str(config.input_dir),
                **{name: getattr(config, name) for name in (
                    "alpha", "beta", "n_splits", "seed", "aggregation",
                    "router_calib_ratio", "specialist_calib_ratio", "test_ratio",
                    "lambda_1", "lambda_2", "require_real_ids",
                )},
            }}}
            validate_primary_config(saved, config)
            changed = PatientExperimentConfig(**{**config.__dict__, "beta": 0.1})
            with self.assertRaisesRegex(ValueError, "beta"):
                validate_primary_config(saved, changed)

    def test_report_pair_rejects_mixed_runs(self):
        primary = {
            "meta": {"config": {"alpha": 0.2, "beta": 0.01,
                                  "n_splits": 500, "aggregation": "mean"}},
            "input_audit": {"n_patients": 9},
            "reference": {name: {
                "mFDR_sys": 0.1, "coverage": 0.8,
                "selected": 8, "errors": 1, "test_patient_decisions": 10,
            } for name in ("proposed", "specialist_only")},
        }
        suite = {
            "meta": {"alpha": 0.2, "beta": 0.01,
                     "n_splits": 500, "aggregation": "mean"},
            "input_audit": {"n_patients": 9},
            "primary": {"methods": {name: dict(primary["reference"][name])
                                    for name in ("proposed", "specialist_only")}},
        }
        validate_report_pair(primary, suite)
        suite["input_audit"]["n_patients"] = 10
        with self.assertRaisesRegex(ValueError, "input_audit"):
            validate_report_pair(primary, suite)
        suite["input_audit"]["n_patients"] = 9
        suite["primary"]["methods"]["proposed"]["selected"] = 7
        with self.assertRaisesRegex(ValueError, "proposed.selected"):
            validate_report_pair(primary, suite)

    def test_organized_paths_do_not_replace_legacy_paths(self):
        root = Path("patient_run")
        paths = PatientArtifactPaths(root)
        self.assertEqual(paths.primary_figures, root / "figures" / "primary")
        self.assertEqual(paths.full_suite_figures, root / "figures" / "full_suite")
        self.assertEqual(paths.primary_json.name, "patient_level_results.json")


if __name__ == "__main__":
    unittest.main()
