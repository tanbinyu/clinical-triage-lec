"""Historical plotting imports remain aliases after the reporting split."""
import importlib
import unittest


class ReportingStructureTests(unittest.TestCase):
    def test_record_level_plot_modules_keep_legacy_imports(self):
        for name in (
            "plot_metrics", "plot_baseline_frontier", "plot_calibration_size",
            "plot_contamination_alpha", "plot_exchangeability",
            "plot_record_bootstrap", "plot_router_sweep",
        ):
            with self.subTest(name=name):
                old = importlib.import_module(f"pipeline.metrics.{name}")
                current = importlib.import_module(f"pipeline.reporting.{name}")
                self.assertIs(old, current)
                self.assertTrue(callable(current.main))


if __name__ == "__main__":
    unittest.main()
