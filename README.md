# Risk-controlled clinical image triage

This repository contains the research software accompanying the manuscript
**Risk-Controlled Clinical Deployment of Foundation Models for Triage and
Specialist Diagnosis across Chest Disease Pathways**. It implements the two
software stages used in the study:

1. frozen foundation-model inference and patient-disjoint data preparation;
2. LEC calibration, routed evaluation, bootstrap analysis, and generation of
   the 11 figures and 6 tables used by the paper workflow.

The repository contains code and configuration only. Medical images, patient
identifiers, trained adapters, foundation-model checkpoints, frozen prediction
exports, and generated manuscript artifacts are intentionally excluded.

## Repository layout

```text
.
|-- inference_code/       Dataset preparation, adapter training and frozen inference
|-- calibration_code/     LEC calibration, evaluation, bootstrap and paper reporting
|-- data/                 Local-only data area (ignored by Git)
|   `-- inference/        phase_1.json and ten phase_2_<source>.json files
`-- weights/              Local-only checkpoints and trained adapters (ignored by Git)
```

The two stages are independent at the file boundary: `inference_code` writes
the frozen JSON contract, and `calibration_code` consumes that contract without
retraining or modifying the upstream models.

## Requirements

- Python 3.11
- A CUDA-capable PyTorch environment is recommended for adapter training and
  foundation-model inference.
- CPU execution is sufficient for calibration and most reporting tasks.

For the inference environment:

```bash
cd inference_code
conda env create -f environment.yml
conda activate naturehelp-paper
```

For calibration only:

```bash
cd calibration_code
conda env create -f environment.yml
conda activate clinical-triage-lec-calibration
```

## Configure local data and checkpoints

All paths in the committed configuration files are repository-relative. Place
or link source datasets under `data/raw/`, model checkpoints under `weights/`,
and frozen inference exports under `data/inference/`. These directories are
ignored by Git and must not contain identifiable information.

The expected calibration input is:

```text
data/inference/
|-- phase_1.json
|-- phase_2_LungHist700.json
|-- phase_2_BreakHis.json
|-- phase_2_COVID_CT.json
|-- phase_2_LDFC.json
|-- phase_2_ImageCHD.json
|-- phase_2_ChestX_ray14.json
|-- phase_2_CBIS-DDSM.json
|-- phase_2_BUS-BRA.json
|-- phase_2_PU2756.json
`-- phase_2_ACDC.json
```

Edit a copied configuration file, or pass CLI overrides, when datasets or
checkpoints are stored elsewhere. Do not commit machine-specific paths.

## Run inference

See [`inference_code/README.md`](inference_code/README.md) for dataset-specific
requirements, controlled checkpoints, validation rules and the complete
training/inference workflow. The top-level command is:

```bash
cd inference_code
python run_full_pipeline.py --scope all
```

## Reproduce calibration and paper artifacts

The primary operating point is `alpha=0.20`, `beta=0.01`, with a maximum of
1,850 eligible downstream patients per source. The resulting 5,529-patient
cohort is allocated to `C_R/C_S/T` as 30%/30%/40%
(1,659/1,659/2,211). From `calibration_code` run:

```bash
python run/cli.py clustered-paper \
  --config configs/patient_clustered_source_cap_1850_beta_001.json \
  --input-dir ../data/inference \
  --output-dir result/patient_clustered/beta-0.01_alpha-0.20-source-cap-1850 \
  --no-install
```

The command performs 500 patient-disjoint downstream partitions, 1,000
patient-cluster bootstrap replicates, sensitivity analyses, and writes a
manifested 17-item paper suite (11 figures and 6 tables) below the selected
output directory. Generated files are ignored by Git.

The run-specific eligibility/export hashes, exact 5,529-patient allocation,
seed map, bootstrap-to-result mapping and environment are documented and
machine-checked as described in
[`calibration_code/reproducibility/README.md`](calibration_code/reproducibility/README.md).
Before submission, build the manifest with `--require-results`; this rejects a
run unless it contains exactly 500 reference partitions and 1,000 bootstrap
replicates. Editor/reviewer delivery is described in
[`calibration_code/reproducibility/REVIEWER_ACCESS.md`](calibration_code/reproducibility/REVIEWER_ACCESS.md).

For a fast structural check that does not run the full experiment:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

## Reproducibility and data governance

- Adapter train/validation patients are disjoint from the downstream pool.
- Router calibration, specialist calibration and test roles are patient
  disjoint within each repetition.
- Patient identifiers in the frozen JSON contract must be de-identified study
  keys; provider identifiers and source images are not redistributed.
- Random seeds, eligibility/allocation digests, input hashes, result hashes and
  the executing code/environment are recorded by the calibration manifest.
- The calibration workflow does not consume `specialist-only/` exports for the
  primary routed-specialist comparison.

Dataset access remains governed by the original providers. Refer to the
manuscript Supplementary Methods for source-specific access conditions.

## Citation

Citation metadata are provided in [`CITATION.cff`](CITATION.cff). Please cite
both the clinical workflow paper and the original LEC paper when reusing the
calibration method.

## License

The code in this repository is released under the MIT License. Dataset terms,
foundation-model licenses and checkpoint access restrictions remain separate
and are not superseded by this license.

