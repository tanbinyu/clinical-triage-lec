# Manuscript-run reproducibility record

This directory documents the exact calibration-side run contract. It contains
no raw images, direct identifiers, or patient-level membership lists.

## Primary configuration

The machine-readable configuration is
`../configs/patient_clustered_source_cap_1850_beta_001.json`. It fixes:

- reporting and calibration target `alpha = 0.20` and router target
  `beta = 0.01`;
- a one-time, source-stratified 35%/5%/60% adapter-train,
  adapter-validation and downstream split with seed `20261019`;
- a fixed, score- and outcome-blind maximum of 1,850 downstream patients per
  source;
- repeated patient-disjoint `C_R/C_S/T = 30%/30%/40%` assignments with base
  seed `20260929`;
- 500 reference partitions and 1,000 source-stratified patient-cluster
  bootstrap replicates.

For the frozen export used by the manuscript, the eligibility audit must report
5,529 included downstream patients and exact role totals of
1,659/1,659/2,211. The corresponding source-level counts, input-file SHA-256
digests, fixed included-cohort digest, repetition-0 role digest, code state,
seeds, environment and output hashes are recorded in
`manuscript_run_manifest.json`.

## Build and verify the manifest

From `calibration_code`:

```powershell
python tools/build_run_manifest.py `
  --config configs/patient_clustered_source_cap_1850_beta_001.json `
  --input-dir <FROZEN_JSON_DIRECTORY> `
  --run-dir <COMPLETED_RUN_DIRECTORY> `
  --output <COMPLETED_RUN_DIRECTORY>/run_provenance.json `
  --require-results
```

`--require-results` is the submission check. It fails unless the completed run
contains the primary result JSON, exactly 500 reference-split rows and exactly
1,000 bootstrap rows. Omitting the flag is permitted only to audit inputs and
eligibility before a run; the resulting manifest is explicitly marked
`results_not_in_public_snapshot`.

The canonical full run is:

```powershell
python run/cli.py clustered-paper `
  --config configs/patient_clustered_source_cap_1850_beta_001.json `
  --input-dir <FROZEN_JSON_DIRECTORY> `
  --output-dir result/patient_clustered/beta-0.01_alpha-0.20-source-cap-1850 `
  --no-install
```

## Bootstrap chain and interpretation

For each of 1,000 replicates, the fixed source-capped downstream cohort is
repartitioned into source-stratified, patient-disjoint `C_R`, `C_S` and `T`
roles. Patients are then sampled with replacement within each source/role
stratum. A duplicated patient carries the same presampled genuine decision in
the patient-balanced primary analysis. Router calibration, routed-specialist
calibration and held-out evaluation are rerun for that replicate. The saved
file `metrics/patient_cluster_bootstrap_replicates.csv` has one row per
replicate; percentile limits are the 2.5th and 97.5th percentiles of these
1,000 rows. They describe this single-run repartitioning/resampling procedure
and are not confidence intervals for the pooled estimator across the separate
500-partition reference summary.

## Historical public-snapshot difference

Commit `a0452eb3d6491fdfecf3fa5f6f93d54058812ce1` did not contain the
1,850-cap configuration. Its corresponding primary source-cap file used a
2,000 cap and `C_R/C_S/T = 40%/30%/30%`. This can be inspected without trusting
the prose:

```powershell
git show a0452eb3d6491fdfecf3fa5f6f93d54058812ce1:calibration_code/configs/patient_clustered_source_cap_2000_beta_001.json
```

The current manifest records both the exact Git state and this superseded
snapshot. A dirty working tree is never represented as matching the declared
release.

## Environment

Create the calibration environment with:

```powershell
conda env create -f environment.yml
conda activate clinical-triage-lec-calibration
```

`environment.yml` pins Python and pip; `requirements-lock.txt` pins every
direct runtime dependency used by the calibration repository. The manifest
also records the versions visible to the executing interpreter.

## Editor and reviewer access

Public code is available from the repository release named in the manifest.
The frozen prediction exports and generated numerical run bundle are not
silently treated as public: they must be delivered through the journal's
reviewer-only upload area or a controlled institutional transfer, subject to
the original dataset and checkpoint terms. Use
`../tools/package_reviewer_bundle.py` after a verified run to create a
de-identified archive containing the config, hashes, environment, primary
metrics and replicate files; raw JSON exports and patient IDs are excluded.

This operational detail may be stated outside the main scientific narrative:
the manuscript needs a concise Code/Data availability statement, while the
full instructions belong here, in the Reporting Summary/checklists, and in the
private reviewer bundle supplied through the submission system.
