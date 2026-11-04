"""Plot source-dataset-stratified record-level bootstrap results.

Usage:
    python -m pipeline.reporting.plot_record_bootstrap \
        --bootstrap-json result/metrics/record_bootstrap_beta_0.01_alpha_0.20.json
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_MPL_CACHE = PROJECT_ROOT / "result" / ".cache" / "matplotlib"
_MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(_MPL_CACHE)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.reporting.plot_metrics import C, INK, _darken, _panel, _setup_style


def _point_interval(ax, y: float, row: dict, color: str, marker: str = "o") -> None:
    point = row["reference_pooled"]
    lo = row["record_bootstrap_ci_low"]
    hi = row["record_bootstrap_ci_high"]
    if None in (point, lo, hi):
        return
    # The historical 500-split pooled reference and the outer bootstrap
    # distribution are reported separately.  With finite B the reference can
    # fall just outside a percentile interval, so draw the interval directly
    # instead of incorrectly treating it as a symmetric error bar about the
    # reference point.
    ax.hlines(y, lo, hi, color=color, linewidth=1.5, zorder=4)
    ax.vlines([lo, hi], y - 0.075, y + 0.075, color=color, linewidth=1.1, zorder=4)
    ax.scatter(point, y, marker=marker, color=color,
               edgecolor=_darken(color, 0.28), linewidth=0.6, s=42, zorder=5)


def _span(rows: list[dict], include: float | None = None,
          lower_floor: float | None = None,
          upper_ceiling: float | None = None) -> tuple[float, float]:
    vals = []
    for row in rows:
        vals.extend(v for v in (row["record_bootstrap_ci_low"],
                                row["record_bootstrap_ci_high"],
                                row["reference_pooled"]) if v is not None)
    if include is not None:
        vals.append(include)
    lo, hi = min(vals), max(vals)
    pad = max((hi - lo) * 0.16, 0.008)
    lo, hi = lo - pad, hi + pad
    if lower_floor is not None:
        lo = min(lo, lower_floor)
    if upper_ceiling is not None:
        hi = max(hi, upper_ceiling)
    return lo, hi


def render(bootstrap_json: Path, figure_dir: Path) -> tuple[Path, Path, Path]:
    with open(bootstrap_json, encoding="utf-8") as f:
        data = json.load(f)
    meta, summary = data["meta"], data["summary"]

    prop_m = summary["proposed"]["mFDR_sys"]
    spec_m = summary["specialist_only"]["mFDR_sys"]
    prop_c = summary["proposed"]["Cov_sys"]
    spec_c = summary["specialist_only"]["Cov_sys"]
    delta_m = summary["paired_difference"]["delta_mFDR_sys"]
    delta_c = summary["paired_difference"]["delta_Cov_sys"]

    # The replicate-level CSV is retained separately from the compact JSON so
    # that the figure can show the paired effect distribution, not only its
    # percentile endpoints.
    replicate_csv = bootstrap_json.with_name(
        bootstrap_json.stem + "_replicates.csv"
    )
    if not replicate_csv.is_file():
        raise FileNotFoundError(
            f"Replicate CSV required for the paired-distribution panels: {replicate_csv}"
        )
    with open(replicate_csv, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    delta_m_values = np.asarray([float(r["delta_mFDR_sys"]) for r in rows])
    delta_c_values = np.asarray([float(r["delta_Cov_sys"]) for r in rows])
    n_replicates = len(rows)

    _setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(7.6, 5.15), constrained_layout=True)
    (ax_a, ax_b), (ax_c, ax_d) = axes

    # (a) Primary result: end-to-end accepted-case risk.
    _point_interval(ax_a, 1, prop_m, C["proposed"])
    _point_interval(ax_a, 0, spec_m, C["gray"], marker="s")
    ax_a.axvline(meta["alpha"], color=C["gray"], linestyle=":", linewidth=0.9,
                 alpha=0.75, zorder=1)
    ax_a.text(meta["alpha"], 1.35, f"target $\\alpha$ = {meta['alpha']:g}",
              ha="right", va="bottom", fontsize=6.3, color=INK["muted"])
    ax_a.set_ylim(-0.42, 1.48)
    ax_a.set_yticks([0, 1])
    ax_a.set_yticklabels(["Specialist-only", "Proposed LEC"])
    ax_a.set_xlabel("System mFDR")
    ax_a.set_xlim(*_span([prop_m, spec_m], include=meta["alpha"], lower_floor=0))
    ax_a.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ref_delta = prop_m["reference_pooled"] - spec_m["reference_pooled"]
    relative_reduction = -ref_delta / spec_m["reference_pooled"]
    n_lower = int(np.sum(delta_m_values < 0))
    ax_a.text(
        0.03, 0.10,
        f"LEC reduction: {100 * ref_delta:.2f} pp\n"
        f"({100 * relative_reduction:.1f}% relative; {n_lower:,}/{n_replicates:,} draws lower)",
        transform=ax_a.transAxes, ha="left", va="bottom", fontsize=6.5,
        color=C["proposed"], fontweight="bold",
    )
    _panel(ax_a, "a", "Primary advantage: lower accepted-case risk")

    # (b) Paired bootstrap distribution of the risk effect.
    ax_b.hist(delta_m_values, bins=26, color=C["proposed"], alpha=0.86,
              edgecolor="white", linewidth=0.35, zorder=3)
    ax_b.axvline(0, color=C["gray"], linestyle=":", linewidth=0.9, zorder=2)
    ax_b.axvline(delta_m["reference_pooled"], color=C["proposed"],
                 linestyle="--", linewidth=1.1, zorder=4)
    ax_b.text(
        0.97, 0.92,
        f"{n_lower:,}/{n_replicates:,} paired draws\n"
        "favour lower LEC risk",
        transform=ax_b.transAxes, ha="right", va="top", fontsize=6.5,
        color=C["proposed"], fontweight="bold",
    )
    ax_b.set_xlabel("$\\Delta$mFDR$_{sys}$ (LEC $-$ specialist-only)")
    ax_b.set_ylabel("Bootstrap replicates")
    ax_b.set_xlim(*_span([delta_m], include=0))
    ax_b.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    _panel(ax_b, "b", "Paired risk advantage")

    # (c) Coverage is the explicit operating trade-off, not a method failure.
    _point_interval(ax_c, 1, prop_c, C["proposed"])
    _point_interval(ax_c, 0, spec_c, C["gray"], marker="s")
    ax_c.set_ylim(-0.42, 1.48)
    ax_c.set_yticks([0, 1])
    ax_c.set_yticklabels(["Specialist-only", "Proposed LEC"])
    ax_c.set_xlabel("AI-managed coverage")
    ax_c.set_xlim(*_span([prop_c, spec_c], lower_floor=0, upper_ceiling=1))
    ax_c.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ref_cov_delta = prop_c["reference_pooled"] - spec_c["reference_pooled"]
    ax_c.text(
        0.03, 0.10,
        f"Expected referral trade-off: {100 * ref_cov_delta:.2f} pp",
        transform=ax_c.transAxes, ha="left", va="bottom", fontsize=6.5,
        color=C["route_err"], fontweight="bold",
    )
    _panel(ax_c, "c", "Coverage trade-off")

    # (d) Paired coverage distribution, using the same resampled records as b.
    ax_d.hist(delta_c_values, bins=26, color=C["route_err"], alpha=0.82,
              edgecolor="white", linewidth=0.35, zorder=3)
    ax_d.axvline(0, color=C["gray"], linestyle=":", linewidth=0.9, zorder=2)
    ax_d.axvline(delta_c["reference_pooled"], color=C["route_err"],
                 linestyle="--", linewidth=1.1, zorder=4)
    ax_d.text(
        0.97, 0.92,
        f"{int(np.sum(delta_c_values < 0)):,}/{n_replicates:,} draws\n"
        "have lower LEC coverage",
        transform=ax_d.transAxes, ha="right", va="top", fontsize=6.5,
        color=C["route_err"], fontweight="bold",
    )
    ax_d.set_xlabel("$\\Delta$Coverage$_{sys}$ (LEC $-$ specialist-only)")
    ax_d.set_ylabel("Bootstrap replicates")
    ax_d.set_xlim(*_span([delta_c], include=0))
    ax_d.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    _panel(ax_d, "d", "Paired coverage trade-off")

    fig.suptitle(
        "Record-Level Bootstrap: LEC Advantage and Operating Trade-off "
        f"($\\beta$ = {meta['beta']:g}, $\\alpha$ = {meta['alpha']:g}; "
        f"B = {meta['n_replicates']:,})",
        fontsize=10.5, fontweight="bold", color=INK["text"],
    )
    fig.text(
        0.5, -0.02,
        "Frozen model scores; source-dataset-stratified record-level intervals conditional on observed composition, not patient-level confidence intervals.",
        ha="center", va="top", fontsize=6.0, color=INK["muted"],
    )

    figure_dir.mkdir(parents=True, exist_ok=True)
    stem = "fig13_record_bootstrap"
    paths = tuple(figure_dir / f"{stem}.{ext}" for ext in ("pdf", "png", "svg"))
    for path in paths:
        fig.savefig(path)
    plt.close(fig)
    return paths


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render record-bootstrap figure.")
    parser.add_argument(
        "--bootstrap-json",
        default="result/metrics/record_bootstrap_beta_0.01_alpha_0.20.json",
    )
    parser.add_argument("--figure-dir", default="result/metrics/figures")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    paths = render(Path(args.bootstrap_json), Path(args.figure_dir))
    print("Saved " + ", ".join(str(path) for path in paths))


if __name__ == "__main__":
    main()
