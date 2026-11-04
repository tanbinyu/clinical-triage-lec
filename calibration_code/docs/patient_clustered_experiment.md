# Patient-clustered mixed-scope experiment

## Authoritative input

The only input is the immutable frozen-score directory:

```text
../data/inference
```

The workflow never rewrites these JSON files. The root-level ten
`phase_2_*.json` files contain routed phase-2 inference from the specialist
selected by the router's predicted pathway. The optional files under
`specialist-only/` are not used by the manuscript workflow. Every output
manifest records the SHA-256 digest of `phase_1.json` and the ten root-level
routed phase-2 files only.

## Statistical units

- A JSON record is a dataset-defined diagnostic decision and retains its own
  `true`, `predicted`, uncertainty and `label_scope`.
- `patient_id` is the grouping unit. All decisions from a patient are assigned
  together to router calibration, specialist calibration or held-out test.
- The primary mFDR and coverage are decision-level estimands.
- Proposed LEC uses the root-level routed-specialist score and system
  correctness, grouped by predicted pathway, after calibrated router
  abstention. Routed specialist-only uses the same predicted pathways, scores
  and correctness but disables router abstention and calibrates on the full
  unfiltered routed streams.
- End-to-end coverage is accepted decisions divided by all test decisions.
  Conditional specialist coverage is accepted routed decisions divided by
  router-passed decisions and is reported separately; the two denominators are
  never mixed.
- Optional source balancing is controlled by `max_patients_per_source`. When
  configured, every split draws at most that many patients from each source
  before the source-stratified 40%/30%/30% role assignment. Sampling is blind
  to labels, model scores and outcomes, and no patient can cross roles. The
  `patient_clustered_source_cap_2000_beta_001.json` is the current manuscript
  configuration. Its 2,000-patient limit prevents the largest source from
  dominating the micro-pooled estimand, while the router target `beta = 0.01`
  yields feasible router and Proposed LEC pathway calibrations across all 500
  reference splits at the full calibration allocation. Earlier 10,000-, 3,000-
  and `beta = 0.005` configurations are retained as historical sensitivity
  analyses.
- Bootstrap resampling is source-stratified and uses the patient cluster as the
  resampling unit; an included patient carries all of their decisions.
- No majority vote or other operation is used to overwrite heterogeneous image-
  or lesion-level ground truth.

Two validation analyses accompany the primary analysis:

1. one decision is sampled uniformly per patient per replicate, preventing
   patients with many records from receiving greater weight;
2. sources whose label scope is exclusively `patient` or `volume` are aggregated
   to one result per patient or volume and evaluated separately.

## Command

```powershell
python run/cli.py clustered-run `
  --config configs/patient_clustered_source_cap_2000_beta_001.json
```

For routine Paper2.0 production, use the wrapper instead:

```powershell
powershell -ExecutionPolicy Bypass -File run\patient_paper2.ps1
```

After only figure/table code changes, use `-RenderOnly` to read the saved
metric JSON/CSV files and avoid rerunning the statistical experiment.

A smoke run can use:

```powershell
python run/cli.py clustered-run `
  --config configs/patient_clustered_source_cap_2000_beta_001.json `
  --n-splits 5 --n-bootstrap 10 `
  --output-dir result/patient_clustered/smoke
```

## Outputs

`result/patient_clustered/<run>/` contains:

- `manifest.json`: authoritative input path and hashes;
- `metrics/patient_clustered_results.json`: ten-pathway primary analysis;
- `metrics/patient_balanced_sensitivity.json`: one-decision-per-patient analysis;
- `metrics/strict_patient_subset.json`: strict patient/volume-scope validation;
- `figures/primary/`: figures generated from these metric files;
- `tables/primary/`: cohort, primary and validation tables.
- `metrics/patient_clustered_paper_suite.json`: patient-clustered baseline,
  specialist-target and router-target experiments used by the manuscript set;
- `figures/paper/`: exactly six manuscript result figures, each in PDF and PNG;
- `tables/paper/`: exactly four main-text LaTeX tables plus a manifest;
- `figures/supplementary/`: calibration-size sensitivity, exchangeability
  diagnostics and decision-utility sensitivity, each in PDF and PNG;
- `tables/supplementary/`: calibration-feasibility and routing-contamination
  audit tables.
- `tables/paper2/newtab/`: six drop-in `tab1.tex`--`tab6.tex` fragments whose
  labels match the Paper2.0 manuscript.
- `deliverables/paper2/`: canonical Paper2.0 package containing exactly 11
  figure PDFs, 6 table fragments, a human-readable README and a provenance
  manifest. Two figures are static workflow schematics; the remaining nine
  figures and all numerical table values come from this patient-clustered run.

The six manuscript figures cover the primary result and analysis-unit
robustness, ten-pathway performance, specialist-target sensitivity, baseline
comparison, routing contamination, and router-target sensitivity.  The three
main tables cover the patient-clustered cohort, system-level robustness,
pathway-specific results, and all six baseline strategies.  Historical
record-level artifacts are not read when
these files are generated.

The supplementary set fills the quantitative analyses that are not duplicated
by the six main figures and four main tables.  Calibration-size subsampling is
performed on patient clusters within source datasets; exchangeability is shown
as a diagnostic rather than a proof; and utility is reported for the primary,
one-decision-per-patient and strict patient/volume analyses.

This namespace is independent of `result/metrics/` (historical record-level) and
`result/patient_level/` (the older one-label-per-patient aggregation workflow).
