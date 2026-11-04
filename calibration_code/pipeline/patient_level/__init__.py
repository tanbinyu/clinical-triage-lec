"""Patient-level evaluation using immutable record-level frozen scores.

This package is deliberately separate from :mod:`pipeline.experiment` and
:mod:`pipeline.metrics` so the historical record-level workflow and artifacts
remain unchanged.
"""

from .analysis import PatientExperimentConfig
from .runner import run_patient_experiment

__all__ = ["PatientExperimentConfig", "run_patient_experiment"]
