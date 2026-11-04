# Patient-level experiment

This workflow is additive. It does not modify the historical record-level
Step 2–6 pipeline or any file under `result/json`.

## Statistical unit and frozen-score aggregation

The input is `phase_1.json` plus the matching `phase_2_{dataset}.json` files.
All records sharing `(source_dataset, patient_id)` are reduced to one patient
decision before calibration or evaluation:

- router and specialist predictions use majority vote;
- tied votes prefer the label with lower mean uncertainty, then the stable
  numeric/lexicographic label order;
- router and specialist uncertainty use the configured mean, median or maximum;
- end-to-end correctness requires both the aggregated router decision and the
  aggregated specialist decision to be correct.

The main analysis uses mean uncertainty. Median and maximum aggregation are
reported as a sensitivity analysis because the JSON files do not contain
patient-level logits from which a model-native aggregation could be derived.

## Data separation

Every repeated split is stratified by source dataset and partitions unique
patients into three mutually exclusive roles:

- `C_R` (40%): router LEC calibration;
- `C_S` (30%): clinic-specific specialist LEC calibration;
- `T` (30%): final held-out patient evaluation.

No patient can occur in more than one role. The 500 repeated splits describe
partition sensitivity; they are not treated as independent cohorts.

## Confidence intervals

The separate 1,000-replicate bootstrap first creates patient-disjoint
`C_R/C_S/T` strata within each source dataset and then samples patients with
replacement inside each role/source stratum. Proposed LEC and specialist-only
are evaluated on the same replicate, enabling paired intervals for:

- proposed end-to-end patient mFDR;
- proposed patient-level AI-managed coverage;
- `Delta mFDR` (proposed minus specialist-only);
- `Delta coverage` (proposed minus specialist-only).

## Run

The preferred entry point is shared with the record-level workflow:

```powershell
python run/cli.py patient-run `
  --config configs/patient_level_beta_001.json
```

To compute only the full suite from an existing compatible primary result:

```powershell
python run/cli.py patient-suite `
  --config configs/patient_level_beta_001.json
```

`patient-suite` rejects a saved primary result with different input path,
operating point, split design, seed, aggregation or score-combination weights.
It does not prove that the contents of an external input directory are
unchanged since the primary run.

To regenerate figures and tables without recalculating any metrics:

```powershell
python run/cli.py patient-render `
  --run-dir result/patient_level/beta-0.01_alpha-0.20
```

The older module entry point remains supported:

```powershell
python -m pipeline.patient_level `
  --config configs/patient_level_beta_001.json
```

Internally, `data.py` handles patient aggregation; `analysis.py` and
`suite.py` contain statistical calculations; `runner.py` executes runs and
writes metrics; the two reporting modules draw figures while
`primary_tables.py` and `suite_tables.py` write tables. This separation lets
figure/table changes use `patient-render` without recalibrating the system.

The checked-in configuration also replays the complete historical figure and
table suite at patient level. If the primary 500-split/1,000-bootstrap result
already exists, regenerate only that suite with:

```powershell
python -m pipeline.patient_level `
  --config configs/patient_level_beta_001.json --suite-only
```

For a quick smoke test:

```powershell
python -m pipeline.patient_level `
  --config configs/patient_level_beta_001.json `
  --n-splits 5 --n-bootstrap 10 --no-sensitivity `
  --output-dir result/patient_level/smoke
```

## Outputs

`result/patient_level/beta-0.01_alpha-0.20/` contains:

- `metrics/patient_level_results.json`: complete machine-readable result;
- `metrics/input_audit.json`: provenance, counts and validation results;
- `metrics/reference_split_metrics.csv`: split-sensitivity results;
- `metrics/patient_bootstrap_replicates.csv`: paired bootstrap replicates;
- `metrics/aggregation_sensitivity.csv`: mean/median/max sensitivity;
- `figures/primary/`: four primary-analysis figures in PDF and PNG;
- `tables/primary/`: primary and per-disease CSV/LaTeX tables plus summary;
- `figures/full_suite/`: twelve legacy-matched figures in PDF and PNG;
- `tables/full_suite/`: six field-complete LaTeX tables and a README;
- `manifest.json`: organized report paths and source-metrics SHA-256 hashes.

The prior outputs directly under `figures/` and `tables/`, and under their
`paper/` subdirectories, are retained as compatibility snapshots. The new
organized directories are rendered from the same saved metric bundles;
`metrics/patient_level_full_suite.json` contains the baseline, alpha, beta,
calibration-size, contamination and exchangeability results. Fig. 12 is a
detailed duplicate view of the calibration-size analysis, not an additional
main-text claim. Full-suite membership does not imply main-paper selection.

The project-provided inference files contain real `patient_id` values. The
loader checks that they are nonempty, match across both phases, do not cross
source datasets, and identify patients consistently. An optional
`patient_id_type` field may explicitly say `dataset_provided`; when it is
absent, the provided `patient_id` is treated as the patient identifier.
The configured run enables `require_real_ids`.
