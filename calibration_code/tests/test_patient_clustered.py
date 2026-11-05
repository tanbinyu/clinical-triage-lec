"""Contract tests for mixed-scope patient-clustered evaluation."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from pipeline.patient_clustered.analysis import (
    ClusteredExperimentConfig,
    _expand_one,
    _patient_partitions,
    _source_partition_sizes,
    clustered_decision_split,
    patient_balanced_bootstrap_partitions,
    patient_balanced_split,
)
from pipeline.patient_clustered.data import (
    aggregate_strict_patient_subset,
    load_clustered_decisions,
)
from pipeline.patient_clustered.suite import patient_clustered_calibration_size_grid
from pipeline.patient_level.suite import _clinic_thresholds, _counts
from pipeline.patient_clustered.paper_reporting import (
    _validate_subplot_axis_labels,
    write_paper2_table_set,
)
from pipeline.patient_clustered.artifacts import (
    expected_artifact_paths,
    install_paper2_artifacts,
    validate_paper2_artifact_set,
)


class PatientClusteredTests(unittest.TestCase):
    def _fixture(self, root: Path) -> None:
        phase1 = {}
        phase2_lung = {}
        phase2_acdc = {}
        for patient in range(9):
            for record in range(2):
                sid = f"lung_{patient}_{record}"
                true = (patient + record) % 3
                phase1[sid] = {
                    "uncertainty": 0.1 + patient * 0.01,
                    "correctness": 1,
                    "predicted_disease": 0,
                    "true_disease": 0,
                    "source_dataset": "LungHist700",
                    "patient_id": f"lung_p{patient}",
                }
                phase2_lung[sid] = {
                    "uncertainty": 0.2 + record * 0.01,
                    "correctness": 1,
                    "predicted": true,
                    "true": true,
                    "patient_id": f"lung_p{patient}",
                    "label_scope": "image",
                }
        for patient in range(6):
            sid = f"acdc_{patient}"
            phase1[sid] = {
                "uncertainty": 0.12,
                "correctness": 1,
                "predicted_disease": 9,
                "true_disease": 9,
                "source_dataset": "ACDC",
                "patient_id": f"acdc_p{patient}",
            }
            phase2_acdc[sid] = {
                "uncertainty": 0.21,
                "correctness": 1,
                "predicted": patient % 5,
                "true": patient % 5,
                "patient_id": f"acdc_p{patient}",
                "label_scope": "patient",
            }
        (root / "phase_1.json").write_text(json.dumps(phase1), encoding="utf-8")
        (root / "phase_2_LungHist700.json").write_text(
            json.dumps(phase2_lung), encoding="utf-8"
        )
        (root / "phase_2_ACDC.json").write_text(
            json.dumps(phase2_acdc), encoding="utf-8"
        )

    def test_heterogeneous_image_truth_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._fixture(root)
            scores, audit = load_clustered_decisions(root)
            self.assertEqual(scores.n_decisions, 24)
            self.assertEqual(scores.n_patients, 15)
            self.assertEqual(
                audit["sources"]["LungHist700"]["heterogeneous_label_patients"], 9
            )
            self.assertEqual(audit["decision_scope_counts"], {"image": 18, "patient": 6})

    def test_all_decisions_from_a_patient_stay_in_one_role(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._fixture(root)
            scores, _ = load_clustered_decisions(root)
            roles = clustered_decision_split(
                scores, epoch=0, seed=7, ratios=(0.4, 0.3, 0.3)
            )
            patient_sets = [
                set(scores.decision_patient_index[indices].tolist()) for indices in roles
            ]
            self.assertFalse(patient_sets[0] & patient_sets[1])
            self.assertFalse(patient_sets[0] & patient_sets[2])
            self.assertFalse(patient_sets[1] & patient_sets[2])
            self.assertEqual(set.union(*patient_sets), set(range(scores.n_patients)))

    def test_patient_balanced_split_uses_one_decision_per_patient(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._fixture(root)
            scores, _ = load_clustered_decisions(root)
            roles = patient_balanced_split(
                scores, epoch=0, seed=7, ratios=(0.4, 0.3, 0.3)
            )
            self.assertEqual(sum(len(indices) for indices in roles), scores.n_patients)

    def test_source_patient_cap_is_applied_before_role_assignment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._fixture(root)
            scores, _ = load_clustered_decisions(root)
            roles = _patient_partitions(
                scores,
                epoch=0,
                seed=7,
                ratios=(0.4, 0.3, 0.3),
                max_patients_per_source=6,
            )
            selected = np.concatenate(roles)
            selected_sources = np.asarray(scores.patient_sources, dtype=object)[selected]
            self.assertEqual(selected.size, 12)
            self.assertEqual(int((selected_sources == "LungHist700").sum()), 6)
            self.assertEqual(int((selected_sources == "ACDC").sum()), 6)
            role_sets = [set(role.tolist()) for role in roles]
            self.assertFalse(role_sets[0] & role_sets[1])
            self.assertFalse(role_sets[0] & role_sets[2])
            self.assertFalse(role_sets[1] & role_sets[2])

    def test_manuscript_source_allocations_match_global_role_quotas(self):
        source_counts = [27, 49, 163, 52, 66, 1850, 940, 639, 1653, 90]
        expected = [
            (8, 8, 11),
            (14, 15, 20),
            (49, 49, 65),
            (16, 15, 21),
            (20, 20, 26),
            (555, 555, 740),
            (282, 282, 376),
            (192, 192, 255),
            (496, 496, 661),
            (27, 27, 36),
        ]
        actual = _source_partition_sizes(source_counts, (0.30, 0.30, 0.40))
        self.assertEqual(actual, expected)
        self.assertEqual(
            [sum(row[index] for row in actual) for index in range(3)],
            [1659, 1659, 2211],
        )

    def test_vectorized_one_decision_sampling_preserves_scalar_rng_sequence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._fixture(root)
            scores, _ = load_clustered_decisions(root)
            patients = np.arange(scores.n_patients, dtype=np.int32)
            scalar_rng = np.random.default_rng(123)
            expected = np.asarray([
                int(records[int(scalar_rng.integers(0, records.size))])
                for records in scores.patient_record_indices
            ], dtype=np.int32)
            actual = _expand_one(
                scores, patients, rng=np.random.default_rng(123)
            )
            np.testing.assert_array_equal(actual, expected)

    def test_patient_balanced_bootstrap_keeps_one_selected_decision_per_cluster(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._fixture(root)
            scores, _ = load_clustered_decisions(root)
            observed_duplicate = False
            for replicate in range(10):
                roles = patient_balanced_bootstrap_partitions(
                    scores,
                    replicate=replicate,
                    seed=19,
                    ratios=(0.4, 0.3, 0.3),
                )
                for indices in roles:
                    selected = {}
                    for index in indices:
                        patient = int(scores.decision_patient_index[index])
                        selected.setdefault(patient, set()).add(int(index))
                    self.assertTrue(all(len(values) == 1 for values in selected.values()))
                    observed_duplicate |= len(indices) > len(selected)
            self.assertTrue(observed_duplicate)

    def test_primary_defaults_are_patient_balanced_and_alpha_is_explicit(self):
        config = ClusteredExperimentConfig(
            input_dir=Path("input"), output_dir=Path("output"),
            alpha=0.2, calibration_alpha=0.18,
        )
        config.validate()
        self.assertTrue(config.patient_balanced_primary)
        self.assertAlmostEqual(config.effective_calibration_alpha, 0.18)
        self.assertAlmostEqual(config.calibration_alpha_scale, 0.9)

    def test_specialist_calibration_uses_specialist_uncertainty_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._fixture(root)
            scores, _ = load_clustered_decisions(
                root, lambda_1=1.0, lambda_2=0.0
            )
            lung = np.flatnonzero(np.asarray(scores.sources) == "LungHist700")
            thresholds, _ = _clinic_thresholds(
                scores, lung, np.ones(lung.size, dtype=bool), alpha=0.2
            )
            self.assertAlmostEqual(thresholds[0], 0.21)
            np.testing.assert_array_equal(
                scores.specialist_score, scores.specialist_uncertainty
            )
            self.assertFalse(np.array_equal(
                scores.specialist_score, scores.fused_uncertainty
            ))

    def test_routed_specialist_only_uses_predicted_pathway_and_system_error(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._fixture(root)
            phase1_path = root / "phase_1.json"
            phase1 = json.loads(phase1_path.read_text(encoding="utf-8"))
            phase1["lung_0_0"]["predicted_disease"] = 1
            phase1["lung_0_0"]["correctness"] = 0
            phase1_path.write_text(json.dumps(phase1), encoding="utf-8")
            scores, _ = load_clustered_decisions(root)
            all_rows = np.arange(scores.n_decisions, dtype=np.int32)
            accepted = np.ones(scores.n_decisions, dtype=bool)
            counts = _counts(
                scores, all_rows, accepted, method="specialist_only"
            )
            self.assertEqual(counts["errors"], 1)
            self.assertEqual(counts["per_clinic"]["1"]["total"], 1)
            self.assertEqual(counts["per_clinic"]["1"]["errors"], 1)

    def test_default_loader_ignores_source_specialist_subdirectory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._fixture(root)
            source_dir = root / "specialist-only"
            source_dir.mkdir()
            for source in ("LungHist700", "ACDC"):
                source_file = root / f"phase_2_{source}.json"
                source_dir.joinpath(source_file.name).write_text(
                    source_file.read_text(encoding="utf-8"), encoding="utf-8"
                )
            scores, audit = load_clustered_decisions(root)
            np.testing.assert_array_equal(
                scores.specialist_only_uncertainty, scores.specialist_uncertainty
            )
            audited_names = [row["name"] for row in audit["files"]]
            self.assertFalse(any(name.startswith("specialist-only") for name in audited_names))

    def test_manuscript_subplots_require_both_axis_labels(self):
        figure, axis = plt.subplots()
        axis.set_xlabel("X")
        with self.assertRaises(ValueError):
            _validate_subplot_axis_labels(figure)
        axis.set_ylabel("Y")
        _validate_subplot_axis_labels(figure)
        plt.close(figure)

    def test_strict_subset_excludes_image_scope_without_relabelling(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._fixture(root)
            scores, _ = load_clustered_decisions(root)
            strict, audit = aggregate_strict_patient_subset(scores)
            self.assertEqual(strict.n_patients, 6)
            self.assertEqual(audit["eligible_sources"], ["ACDC"])
            self.assertEqual(audit["excluded_sources"], ["LungHist700"])

    def test_calibration_size_grid_subsamples_patient_clusters(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._fixture(root)
            scores, _ = load_clustered_decisions(root)
            config = ClusteredExperimentConfig(
                input_dir=root,
                output_dir=root / "out",
                beta=0.2,
                alpha=0.2,
                n_splits=2,
                n_bootstrap=2,
                seed=17,
                run_paper_suite=False,
            )
            rows = patient_clustered_calibration_size_grid(scores, config)
            self.assertEqual([row["fraction"] for row in rows], [0.25, 0.5, 0.75, 1.0])
            self.assertGreaterEqual(
                rows[-1]["router"]["calibration_decisions"]["mean"],
                rows[-1]["router"]["calibration_patients"]["mean"],
            )
            self.assertEqual(len(rows[-1]["clinics"]), 10)

    def test_paper2_compatibility_set_has_six_stable_tables(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            main = root / "main"
            supplementary = root / "supplementary"
            target = root / "paper2" / "newtab"
            main.mkdir()
            supplementary.mkdir()
            sources = {
                main / "table2_patient_clustered_system.tex": "tab:patient_clustered_system",
                main / "table3_patient_clustered_pathways.tex": "tab:patient_clustered_pathways",
                supplementary / "tableS1_patient_clustered_calibration_feasibility.tex": "tab:patient_clustered_calibration_feasibility",
                supplementary / "tableS2_patient_clustered_contamination.tex": "tab:patient_clustered_contamination",
                main / "table4_patient_clustered_baselines.tex": "tab:patient_clustered_baselines",
                main / "table1_patient_clustered_cohort.tex": "tab:patient_clustered_cohort",
            }
            for path, label in sources.items():
                path.write_text(rf"\label{{{label}}}", encoding="utf-8")
            written = write_paper2_table_set(main, supplementary, target)
            self.assertEqual([path.name for path in written], [f"tab{i}.tex" for i in range(1, 7)])
            expected_labels = (
                "tab:system_summary", "tab:per_disease_risk",
                "tab:calibration_feasibility", "tab:contamination",
                "tab:baselines", "tab:dataset_summary",
            )
            for path, label in zip(written, expected_labels):
                self.assertIn(rf"\label{{{label}}}", path.read_text(encoding="utf-8"))

    def test_paper2_artifact_contract_is_eleven_figures_and_six_tables(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "artifacts"
            figures, tables = expected_artifact_paths(root)
            self.assertEqual(len(figures), 11)
            self.assertEqual(len(tables), 6)
            for path in figures + tables:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"artifact")
            checked_figures, checked_tables = validate_paper2_artifact_set(root)
            self.assertEqual(checked_figures, figures)
            self.assertEqual(checked_tables, tables)

            manuscript = Path(directory) / "manuscript"
            install_paper2_artifacts(root, manuscript)
            for source in figures:
                relative = source.relative_to(root / "figures")
                self.assertEqual(
                    (manuscript / "figures" / relative).read_bytes(), b"artifact"
                )
            for source in tables:
                relative = source.relative_to(root / "tables")
                self.assertEqual(
                    (manuscript / "tables" / relative).read_bytes(), b"artifact"
                )


if __name__ == "__main__":
    unittest.main()
