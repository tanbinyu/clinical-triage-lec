"""Backward-compatible import path for LEC primitives."""
from pipeline.core.lec import lec_calibrate, lec_evaluate

__all__ = ["lec_calibrate", "lec_evaluate"]
