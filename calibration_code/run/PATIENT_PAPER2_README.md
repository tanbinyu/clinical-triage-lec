# Patient-clustered Paper2.0 one-command workflow

Run from the `Code` directory:

```powershell
powershell -ExecutionPolicy Bypass -File run\patient_paper2.ps1
```

The default input is the repository-local frozen-score directory
`../data/inference`. Only `phase_1.json` and the ten
root-level `phase_2_*.json` files are read. The command reruns the
patient-disjoint calibration/evaluation with one equally weighted decision per
patient, performs source-stratified patient-cluster bootstrap, and builds
exactly 11 figure PDFs and 6 LaTeX tables.

The ten root-level `phase_2_*.json` files are the routed phase-2 outputs: each
decision was evaluated by the specialist selected by the predicted pathway.
The optional `specialist-only/` subdirectory is ignored. Routed
specialist-only uses the same root-level Phase-2 outputs as Proposed LEC and
differs only by disabling router abstention. It calibrates each specialist on
every assignment to the router-predicted pathway, retains misroutes as
downstream errors, and therefore has defined wrong-route metrics.

The primary reporting target and the level used for specialist calibration are
both `alpha = 0.20`, the router target is `beta = 0.01`, and a score- and
outcome-blind cap of 1,850 patients per source is applied before the
source-stratified 30%/30%/40% role assignment. This gives 5,529 included
patients and exact aggregate role counts of 1,659/1,659/2,211; there is no
hidden 0.18 safety-margin substitution. The default
config remains marked exploratory because the current directory contains all
available records. The artifacts are built,
but installation into Paper2.0 is automatically skipped. Installation is
enabled only when the config truthfully declares both a `test_only` input and
a `prespecified_confirmatory` calibration policy.

To redraw the 17 artifacts from existing metrics without repeating the 500
splits or 1,000 bootstrap replicates:

```powershell
powershell -ExecutionPolicy Bypass -File run\patient_paper2.ps1 -RenderOnly
```

To build the deliverable set without changing the Paper2.0 folder, add
`-NoInstall`. Override `-InputDir`, `-OutputDir`, `-Paper2Dir`, `-Splits`, or
`-Bootstraps` when creating a separate run.

The canonical deliverable directory is:

```text
result/patient_clustered/<run>/deliverables/paper2/
├── figures/          11 PDFs (9 results + 2 static workflow schematics)
├── tables/newtab/     6 LaTeX tables
├── manifest.json      provenance, roles and SHA-256 hashes
└── README.md
```

Historical `result/json` record-level outputs are not read or overwritten.
