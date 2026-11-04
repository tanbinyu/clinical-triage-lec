# Patient-clustered workflow module map

This package is the supported analysis for the authoritative patient-ID JSON
files under the repository-local `../data/inference` directory.

| Module | Single responsibility |
|---|---|
| `data.py` | Validate frozen phase-1/phase-2 JSON and preserve every dataset-defined decision and `patient_id` |
| `analysis.py` | Patient-disjoint splitting, patient-balanced primary sampling, LEC calibration and patient-cluster bootstrap |
| `suite.py` | Baselines and alpha/beta/calibration-evidence sensitivity calculations |
| `runner.py` | Orchestrate calculations and write numerical JSON/CSV bundles |
| `reporting.py` | Internal primary-analysis diagnostics |
| `paper_reporting.py` | Reusable patient-clustered figures and LaTeX table generators |
| `paper2_reporting.py` | Nine data-derived figures with filenames/panels matching Paper2.0 |
| `artifacts.py` | Assemble, validate, hash and optionally install the exact 11-figure/6-table Paper2.0 contract |
| `cli.py` | Package-level calculation entry point used by `run/cli.py` |

The boundaries are intentional: calculation modules do not know manuscript
filenames, and artifact packaging never recalibrates models. The two static
workflow schematics are not statistical results and are copied unchanged.

## Public commands

```powershell
# Complete calculation and Paper2.0 packaging
python run/cli.py clustered-paper --config configs/patient_clustered_beta_001.json

# Rebuild only figures/tables from existing metrics
python run/cli.py clustered-render `
  --run-dir result/patient_clustered/beta-0.01_alpha-0.20 `
  --install-paper2
```

For the shortest Windows command, use `run/patient_paper2.ps1`.

## Statistical contract

- Primary: patients are split disjointly across `C_R`, `C_S` and `T`, then one
  genuine decision is sampled uniformly for every patient in each split. This
  gives every patient equal weight without inventing a patient diagnosis for
  image-/lesion-scoped labels.
- Optional source cap: `max_patients_per_source` draws a score- and
  outcome-blind patient subsample within every source before role assignment.
  It changes the target source mixture and is therefore reported as a separate
  source-balanced analysis rather than silently replacing the uncapped cohort.
- Sensitivity: all genuine decisions are retained, with patient IDs defining
  the split and bootstrap clusters.
- Bootstrap: the decision for a patient is chosen before resampling; duplicate
  bootstrap copies of that patient therefore carry the same decision.
- Stage-specific scores: router calibration uses `router_uncertainty`; both
  Proposed LEC and Routed specialist-only use the routed specialist outputs in
  the root `phase_2_*.json` files. The comparator differs only by disabling
  router abstention and fitting specialist thresholds on all assignments to
  each predicted pathway. The optional `specialist-only/` directory is ignored.
- The proposed-versus-routed-specialist-only comparison reports overall mFDR and
  coverage together with worst-pathway mFDR, pathways meeting the target and
  routing contamination for both methods. This directly isolates the
  routing-stage abstention contribution without changing pathway assignments,
  specialist scores or held-out cases.
- `alpha` is the reporting target. `calibration_alpha` is the explicit level
  used to fit LEC and is never silently substituted for the reporting target.

The supplied config uses the same prespecified value, 0.20, for the reporting
target and specialist calibration level; no additional safety-margin value is
substituted. It marks the current all-record input as exploratory and still
produces all 17 artifacts, but automatic
installation into Paper2.0 is skipped. A confirmatory run may be installed
only after the input is test-only and the calibration policy was fixed before
examining that test cohort; set `input_cohort_scope` to `test_only` and
`calibration_policy_status` to `prespecified_confirmatory` only when those
statements are factually true.

## Output contract

```text
result/patient_clustered/<run>/
├── metrics/                 numerical source of truth
├── figures/                 analysis/intermediate reports
├── tables/                  analysis/intermediate reports
├── deliverables/paper2/     only manuscript-facing package
│   ├── figures/             exactly 11 PDF files
│   ├── tables/newtab/       exactly 6 TeX files
│   ├── manifest.json        provenance and SHA-256 hashes
│   └── README.md
└── manifest.json            run-level paths and input hashes
```

Historical record-level code and results remain in their own namespaces and
are not imported by this workflow.
