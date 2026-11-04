# Source-dataset-stratified record-level bootstrap

## Scope

The immutable Step 1 artifacts contain frozen router and specialist scores for
13,908 records from 11 source datasets. They do not contain patient, study,
slide, video, volume, subject, or other grouping identifiers. The analysis is
therefore explicitly record-level and must never be described as a
patient-level confidence interval.

## Procedure

For each of 1,000 outer bootstrap replicates, records are sampled with
replacement within each source dataset, preserving each source dataset's
observed sample size. The resampled record occurrences are randomly split 50:50
into calibration and held-out evaluation sets. The frozen-score workflow is
then rerun in full: router LEC calibration at beta, routing, clinic-specific
LEC calibration at alpha, and held-out system evaluation. Specialist-only
rejection is evaluated on the same resampled calibration/evaluation occurrence
sets, making the proposed-minus-specialist-only differences paired.

The historical 500 random splits are not bootstrap units. They are evaluated
once only to recover the reference pooled point estimate. Existing percentile
summaries formed by resampling those 500 splits remain split-sensitivity
descriptions, not the intervals reported here.

## Outputs

For beta 0.01 and alpha 0.20 the default command writes:

- `result/metrics/record_bootstrap_beta_0.01_alpha_0.20.json`: metadata,
  reference estimates, percentile intervals, and calibration diagnostics.
- `result/metrics/record_bootstrap_beta_0.01_alpha_0.20_replicates.csv`:
  one row per bootstrap replicate, enabling independent audit and replotting.
- `result/metrics/record_bootstrap_beta_0.01_alpha_0.20.md`: concise report.
- `result/metrics/figures/fig13_record_bootstrap.{pdf,png,svg}`: a three-panel
  uncertainty figure for system mFDR, AI-managed coverage, and paired effects.

Run the analysis with:

```powershell
python run/cli.py record-bootstrap --beta 0.01 --alpha 0.2
python run/cli.py record-bootstrap-figure
```

## Manuscript wording

> We performed a source-dataset-stratified record-level nonparametric bootstrap
> (1,000 replicates), rerunning frozen-score calibration, routing, specialist
> calibration, and held-out evaluation in every replicate. Because the released
> artifacts lacked patient-, study-, slide-, video-, and volume-level
> identifiers, the resulting percentile intervals quantify uncertainty
> conditional on the observed record-level source-dataset composition and do
> not constitute patient-level population confidence intervals.
