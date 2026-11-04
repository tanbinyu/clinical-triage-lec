# Patient-level artifact map

## Current mixed-scope patient-clustered manuscript set

For the authoritative `NatureHelp/data/json` input, use
`result/patient_clustered/beta-0.01_alpha-0.20/`.  Its manuscript artifacts
are deliberately separated from both historical record-level results and the
older one-label-per-patient aggregation workflow.

| Directory | Contents | Use |
|---|---|---|
| `metrics/patient_clustered_results.json` | Proposed and specialist-only primary estimates with patient-cluster bootstrap | Primary numerical source |
| `metrics/patient_clustered_paper_suite.json` | Six methods plus alpha/beta and contamination grids | Old-experiment patient-clustered replacement |
| `figures/paper/` | Exactly six unique figures, in PDF and PNG | Main-text experimental figures |
| `tables/paper/` | Exactly four LaTeX tables and README | Main-text tables |
| `figures/supplementary/` | Three non-duplicative patient-clustered sensitivity/diagnostic figures, in PDF and PNG | Supplementary figures |
| `tables/supplementary/` | Calibration-feasibility and contamination audit tables | Supplementary tables |
| `tables/paper2/newtab/` | Six drop-in fragments named `tab1.tex`--`tab6.tex` with Paper2.0 labels | Direct Paper2.0 replacement set |
| `deliverables/paper2/` | Exact 11-figure/6-table Paper2.0 contract plus hashes | Preferred manuscript-facing output |
| `manifest.json` | Authoritative input path, hashes and report locations | Provenance audit |

The same six PDFs and four main-table `.tex` files are copied to
`Latex_Paper/figures/patient_clustered/` and
`Latex_Paper/tables/patient_clustered/` for manuscript use.

## Older one-label-per-patient aggregation artifacts

`result/patient_level/<run>/` is a run directory, not a manuscript figure
directory. Its numerical sources are `metrics/patient_level_results.json`
(primary estimates and patient bootstrap) and
`metrics/patient_level_full_suite.json` (comparisons and sensitivity grids).
The original `result/json/step1`--`step6` tree is the separate historical
record-level experiment and is not part of this patient-level run.

| Directory | Contents | Use |
|---|---|---|
| `metrics/` | JSON/CSV numerical outputs | Source of truth for estimates |
| `figures/primary/` | Four risk, paired-effect, disease and aggregation figures | Primary/bootstrap diagnostics |
| `tables/primary/` | Primary and disease estimates, summary | Precise primary/bootstrap values |
| `figures/full_suite/` | Fig. 1--12 patient-level counterparts | Complete experimental figure set; Fig. 12 is a detailed duplicate of Fig. 6 |
| `tables/full_suite/` | Tables 1--6 and README | Complete experimental table set |
| `manifest.json` | Source JSON hashes and organized report listing | Check what was rendered from which metrics |

`figures/paper/`, `tables/paper/`, and files directly under `figures/` or
`tables/` are retained for old commands and references. They are not the
preferred paths for new manuscript links. The directory name `full_suite`
means all analyses were rendered; it does **not** mean all twelve figures or
six tables should appear in the main text. Main-text selection and final
captions should be recorded in the LaTeX project separately.

To recolor or revise a figure, change the relevant patient-level reporting
function and run `python run/cli.py patient-render --run-dir <run>`. This
reads the saved metrics and does not redo the 500 splits or 1,000 bootstrap
replicates. The renderer checks that primary and suite bundles agree on
operating point, split count, aggregation, input audit and key reference
estimates before combining them. It does not verify that external Step-1 JSON
contents are unchanged.
