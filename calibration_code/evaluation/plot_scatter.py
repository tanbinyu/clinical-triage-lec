import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np

from evaluation.evaluate import evaluate_lec_result
from evaluation.plot_coverage import compute_coverage
from pipeline.config import NUM_DISEASES, DISEASE_NAMES


def collect_data(output_dir, beta, alpha_A, alpha_B):
    dir_step4 = os.path.join(output_dir, "step4")
    dir_step5 = os.path.join(output_dir, "step5")
    dir_step6 = os.path.join(output_dir, "step6")

    splits_map = {}
    for branch in ["a", "b"]:
        with open(os.path.join(dir_step4, f"splits_branch_{branch}_beta_{beta}.json")) as f:
            splits_map[branch] = json.load(f)

    points = []
    for k in range(NUM_DISEASES):
        for branch, alpha, suffix in [("a", alpha_A, f"alphaA_{alpha_A}"), ("b", alpha_B, f"alphaB_{alpha_B}")]:
            lec_path = os.path.join(dir_step6,
                                    f"lec_branch_{branch}_disease_{k}_beta_{beta}_{suffix}.json")
            disease_path = os.path.join(dir_step5, f"branch_{branch}_disease_{k}.json")
            if not os.path.exists(lec_path) or not os.path.exists(disease_path):
                continue

            with open(lec_path) as f:
                lec_result = json.load(f)
            with open(disease_path) as f:
                disease_data = json.load(f)

            ev = evaluate_lec_result(lec_result, alpha)
            if ev["mean_fdp"] is None:
                continue
            mean_cov, _ = compute_coverage(lec_result, disease_data, splits_map[branch])
            if mean_cov is None:
                continue

            points.append({
                "disease": k,
                "branch": branch,
                "mean_fdp": ev["mean_fdp"],
                "coverage": mean_cov,
                "name": DISEASE_NAMES[k],
            })

    return points


def plot_scatter(points, alpha_A, alpha_B, img_dir):
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "stix",
        "font.size": 11,
    })

    fig, ax = plt.subplots(figsize=(9, 7))

    markers_a, markers_b = [], []
    for pt in points:
        if pt["branch"] == "a":
            markers_a.append(pt)
        else:
            markers_b.append(pt)

    if markers_a:
        ax.scatter([p["coverage"] for p in markers_a],
                   [p["mean_fdp"] for p in markers_a],
                   c="steelblue", marker="o", s=80, edgecolors="k", linewidths=0.5,
                   label="Branch A", zorder=5)
    if markers_b:
        ax.scatter([p["coverage"] for p in markers_b],
                   [p["mean_fdp"] for p in markers_b],
                   c="coral", marker="s", s=80, edgecolors="k", linewidths=0.5,
                   label="Branch B", zorder=5)

    for pt in points:
        offset_x = 0.012
        offset_y = 0.003
        ax.annotate(f"D{pt['disease']}", (pt["coverage"] + offset_x, pt["mean_fdp"] + offset_y),
                    fontsize=7, color="gray")

    ax.axhline(y=alpha_A, color="steelblue", linestyle="--", linewidth=1, alpha=0.7,
               label=f"$\\alpha_A$ = {alpha_A}")
    if alpha_B != alpha_A:
        ax.axhline(y=alpha_B, color="coral", linestyle="--", linewidth=1, alpha=0.7,
                   label=f"$\\alpha_B$ = {alpha_B}")

    ax.set_xlabel("Coverage (retention rate)", fontsize=12)
    ax.set_ylabel("mean FDP", fontsize=12)
    ax.set_title(f"FDP vs Coverage per Disease ($\\alpha_A$={alpha_A}, $\\alpha_B$={alpha_B})", fontsize=14)
    ax.set_xlim(-0.02, 1.05)
    ax.set_ylim(-0.01, max(alpha_A, alpha_B) * 2.5)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.3)

    target = min(alpha_A, alpha_B)
    ax.fill_between([0, 1.05], 0, target, color="green", alpha=0.05)
    ax.text(0.85, target * 0.4, "controlled\nzone", fontsize=9, color="green",
            ha="center", style="italic", alpha=0.6)

    plt.tight_layout()
    save_path = os.path.join(img_dir, f"mfdr_coverage_scatter_alphaA_{alpha_A}_alphaB_{alpha_B}.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Saved: {save_path}")


def main():
    p = argparse.ArgumentParser(description="Plot mFDR vs coverage scatter per disease")
    p.add_argument("--output_dir", default="result/json")
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--alpha_A", type=float, default=0.1)
    p.add_argument("--alpha_B", type=float, default=0.1)
    p.add_argument("--img_dir", default="result/img/scatter")
    args = p.parse_args()

    os.makedirs(args.img_dir, exist_ok=True)
    points = collect_data(args.output_dir, args.beta, args.alpha_A, args.alpha_B)
    plot_scatter(points, args.alpha_A, args.alpha_B, args.img_dir)
    print(f"\n[Done] Scatter plot saved to {args.img_dir}")


if __name__ == "__main__":
    main()
