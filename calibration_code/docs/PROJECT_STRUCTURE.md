# Project structure and compatible workflow

`result/json/step1` remains the immutable input location for the historical
record-level workflow.  The authoritative current patient-ID inputs are the
unchanged `phase_1.json` and `phase_2_{dataset}.json` files under the local
`../data/inference` directory; the patient-clustered workflow
records their hashes and never rewrites them.

```text
pipeline/core/        stable LEC, metadata, artifact paths
pipeline/experiment/  Step 2–6 algorithms and orchestration
pipeline/metrics/     record-level metrics and sensitivity grids
pipeline/patient_level/ patient-level data, calculations, execution and reports
pipeline/patient_clustered/ mixed-scope decisions with patient-disjoint splits and cluster bootstrap
pipeline/reporting/   record-level figure implementations and table entry points
run/cli.py            one supported command-line entry point
run/legacy/           historical beta=0.01 helper implementations
configs/              reproducible parameter sets
result/               generated artifacts; _tools/ contains compatibility launchers
```

## Default compatible workflow

The historical commands remain supported:

```powershell
python main.py --steps 2 3 4 5 6 --beta 0.1
python -m pipeline.metrics.run_all
```

They still write the same files under `result/json/step2` through `step6`.

The patient-level workflow is documented in `docs/patient_level_experiment.md`.
It writes only under `result/patient_level/` and does not alter historical
record-level artifacts. Both workflows use `run/cli.py` as the preferred
entry point; `main.py`, `python -m pipeline.patient_level`, and the old
`result/_tools/*.py` paths remain compatibility entry points.

## Preferred workflow

```powershell
python run/cli.py preflight --config configs/default.json
python run/cli.py experiment --config configs/default.json
python run/cli.py metrics --beta 0.1 --alpha 0.2
python run/cli.py figures

# Optional uncertainty analysis; this never overwrites the split-sensitivity
# summaries in metrics.json.
python run/cli.py record-bootstrap --beta 0.01 --alpha 0.2
python run/cli.py record-bootstrap-figure

# Patient-level results from frozen Step-1 scores:
python run/cli.py patient-run --config configs/patient_level_beta_001.json

# Figure/table-only refresh from existing patient-level metrics (no calibration):
python run/cli.py patient-render --run-dir result/patient_level/beta-0.01_alpha-0.20

# Mixed patient/image/lesion/volume truth without relabelling:
python run/cli.py clustered-run --config configs/patient_clustered_beta_001.json

# Recommended one-command Paper2.0 workflow (full calculation + 11 figures + 6 tables):
powershell -ExecutionPolicy Bypass -File run\patient_paper2.ps1

# Fast report-only refresh from saved metrics:
powershell -ExecutionPolicy Bypass -File run\patient_paper2.ps1 -RenderOnly
```

The mixed-scope workflow is documented in
`docs/patient_clustered_experiment.md`. It is the supported workflow for the
authoritative frozen JSON input when a patient can have different
image- or lesion-level labels.

Patient-level reports are grouped into `figures/primary`,
`figures/full_suite`, `tables/primary`, and `tables/full_suite`. Historical
reports directly under `figures/` and `tables/`, and the former `paper/`
subdirectories, remain untouched for compatibility. The run-level
`manifest.json` lists organized reports and SHA-256 hashes of their source
metrics. See `docs/ARTIFACT_MAP.md` for the distinction between these report
groups and manuscript selection.

The current mixed-scope patient-clustered manuscript set is grouped into
`result/patient_clustered/<run>/figures/paper` (six figures) and
`result/patient_clustered/<run>/tables/paper` (four main-text tables), with two
additional audit tables under `tables/supplementary`.  A six-table drop-in
Paper2.0 set is rendered under `tables/paper2/newtab`.  These outputs
retain every image- or lesion-scoped label, split on patient identifiers, and
use source-stratified patient-cluster bootstrap intervals.

The manuscript-facing contract is
`result/patient_clustered/<run>/deliverables/paper2`: exactly 11 figure PDFs
and 6 LaTeX table fragments. Nine figures are regenerated from the current
patient-clustered metrics; `overview.pdf` and `method_overview.pdf` are static
workflow schematics copied unchanged. `manifest.json` records provenance,
artifact roles and hashes. The `clustered-render` command rebuilds this set
without repeating calibration or bootstrap calculations.

For the patient-clustered workflow, the primary analysis gives each patient
equal weight by sampling one genuine decision per patient after patient-
disjoint assignment. The all-decisions analysis is a named sensitivity output
(`metrics/all_decisions_sensitivity.json`). Reporting target `alpha` and the
fitted `calibration_alpha` are separate config and manifest fields. An
exploratory run can always generate the 17-item package, but it is installed
into Paper2.0 only when its manifest is confirmatory/publication-ready.

### Patient-level module ownership

| Module | Responsibility |
|---|---|
| `data.py`, `metadata.py` | Validate Step-1 inputs and aggregate records by patient |
| `analysis.py` | Primary calibration, held-out evaluation and patient bootstrap calculations |
| `suite.py` | Baselines and alpha/beta/calibration-size sensitivity calculations |
| `runner.py` | Execute either analysis, write metric JSON/CSV and invoke reports |
| `reporting.py`, `suite_reporting.py` | Draw primary and full-suite figures and assemble reports |
| `primary_tables.py`, `suite_tables.py` | Generate primary and full-suite tables |
| `artifacts.py` | Organized report paths, metric-pair checks and render-only workflow |
| `cli.py` | Patient-level compatibility command-line interface |

The top-level `run/cli.py` is the preferred public entry point. Historical
imports from `analysis.py`, `suite.py`, `reporting.py`, and
`suite_reporting.py` still work. No numerical method or JSON schema changed
in this module split.

For the record-level workflow, the seven `plot_*.py` implementations now
belong to `pipeline/reporting/`; `pipeline/metrics/plot_*.py` are import and
`python -m` compatibility aliases. The computation modules remain under
`pipeline/metrics/`. This keeps the historical commands runnable while making
the calculation/rendering boundary visible in the directory tree.

For an isolated experiment, set `output_dir` in a config to a run-specific
`result/runs/<name>/json` directory. Metrics then use that derived root while
still reading immutable inputs from `result/json`:

```powershell
python run/cli.py experiment --config configs/beta_001.json
python run/cli.py metrics --beta 0.01 --alpha 0.2 `
  --data-root result/runs/beta-0.01_alpha-0.20/json `
  --input-root result/json `
  --output-dir result/runs/beta-0.01_alpha-0.20/metrics
python run/cli.py figures `
  --metrics-json result/runs/beta-0.01_alpha-0.20/metrics/metrics.json `
  --figure-dir result/runs/beta-0.01_alpha-0.20/figures
```

The beta-sweep figure requires a `lec_phase_1_beta_<beta>.json` file for every
value in `pipeline.metrics.config.BETA_SWEEP_FULL`. Use `experiment` with
`steps: [3]` and the corresponding beta values before computing that sweep.

`calibration/` and the shell scripts in `run/` remain legacy/parallel flows;
they use a different data contract (`res/`, eight diseases) and are not part
of this ten-disease pipeline.
