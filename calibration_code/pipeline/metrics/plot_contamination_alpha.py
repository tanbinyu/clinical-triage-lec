"""Compatibility alias for the record-level reporting implementation."""
import importlib
import sys


_TARGET = "pipeline.reporting.plot_contamination_alpha"
if __name__ == "__main__":
    importlib.import_module(_TARGET).main()
else:
    sys.modules[__name__] = importlib.import_module(_TARGET)
