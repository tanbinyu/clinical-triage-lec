"""Regression checks for report-only rendering and legacy entry points."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pipeline.patient_level.artifacts import render_saved_patient_reports


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class PatientArtifactTests(unittest.TestCase):
    def test_report_only_render_preserves_metric_json(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metrics = root / "metrics"
            metrics.mkdir()
            primary = {
                "meta": {"config": {
                    "alpha": 0.2, "beta": 0.01, "n_splits": 5,
                    "aggregation": "mean",
                }},
                "input_audit": {"n_patients": 9},
                "reference": {name: {
                    "mFDR_sys": 0.1, "coverage": 0.8,
                    "selected": 8, "errors": 1, "test_patient_decisions": 10,
                } for name in ("proposed", "specialist_only")},
            }
            suite = {
                "meta": {
                    "alpha": 0.2, "beta": 0.01, "n_splits": 5,
                    "aggregation": "mean",
                },
                "input_audit": {"n_patients": 9},
                "primary": {"methods": {name: dict(primary["reference"][name])
                                        for name in ("proposed", "specialist_only")}},
            }
            primary_path = metrics / "patient_level_results.json"
            suite_path = metrics / "patient_level_full_suite.json"
            primary_path.write_text(json.dumps(primary), encoding="utf-8")
            suite_path.write_text(json.dumps(suite), encoding="utf-8")
            before = {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in (primary_path, suite_path)
            }

            def write_primary(_result, *, figures_dir, tables_dir):
                figures_dir.mkdir(parents=True)
                tables_dir.mkdir(parents=True)
                (figures_dir / "primary.pdf").write_bytes(b"test figure")
                (tables_dir / "primary.tex").write_text("test table")

            def write_suite(_suite, _primary, *, figure_dir, table_dir):
                figure_dir.mkdir(parents=True)
                table_dir.mkdir(parents=True)
                (figure_dir / "suite.pdf").write_bytes(b"test figure")
                (table_dir / "suite.tex").write_text("test table")

            with patch("pipeline.patient_level.reporting.write_all_reports", write_primary), \
                 patch("pipeline.patient_level.suite_reporting.write_full_suite_outputs", write_suite):
                paths = render_saved_patient_reports(root)

            manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
            self.assertEqual(len(manifest["report_groups"]), 4)
            self.assertIn("figures/full_suite/suite.pdf", manifest["report_groups"]["figures/full_suite"])
            for path in (primary_path, suite_path):
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), before[path.name])
                self.assertEqual(
                    manifest["source_metrics"]["metrics/" + path.name], before[path.name]
                )

    def test_legacy_launchers_retain_original_names(self):
        for name in (
            "check_tables_beta001.py", "gen_step6.py", "gen_tables_beta001.py",
            "plot_beta001.py", "run_alpha_grid_beta001.py",
            "run_metrics_beta001.py", "xcheck_tab2.py",
        ):
            launcher = PROJECT_ROOT / "result" / "_tools" / name
            implementation = PROJECT_ROOT / "run" / "legacy" / name
            self.assertTrue(launcher.is_file(), name)
            self.assertTrue(implementation.is_file(), name)
            self.assertIn('run_name="__main__"', launcher.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
