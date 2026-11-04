import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np

from evaluation.evaluate import evaluate_lec_result
from pipeline import step6_calibrate_phase2
from pipeline.config import NUM_DISEASES, DISEASE_NAMES


def compute_coverage(lec_result, disease_data, splits):
    disease_ids = set(disease_data.keys())
    coverages = []
    for split, lec_split in zip(splits["splits"], lec_result["splits"]):
        test_in_disease = [sid for sid in split["test"] if sid in disease_ids]
        total = len(test_in_disease)
        if total == 0:
            continue
        selected = lec_split["selected_num"]
        coverages.append(selected / total)
    if not coverages:
        return None, None
    mean_cov = np.mean(coverages)
    std_cov = np.std(coverages)
    return round(float(mean_cov), 6), round(float(std_cov), 6)


def sweep_coverage(output_dir, alpha_list, beta):
    dir_step4 = os.path.join(output_dir, "step4")
    dir_step5 = os.path.join(output_dir, "step5")
    dir_step6 = os.path.join(output_dir, "step6")

    with open(os.path.join(dir_step4, f"splits_branch_a_beta_{beta}.json")) as f:
        splits_a = json.load(f)
    with open(os.path.join(dir_step4, f"splits_branch_b_beta_{beta}.json")) as f:
        splits_b = json.load(f)

    disease_data = {}
    for branch in ["a", "b"]:
        disease_data[branch] = {}
        for k in range(NUM_DISEASES):
            path = os.path.join(dir_step5, f"branch_{branch}_disease_{k}.json")
            with open(path) as f:
                disease_data[branch][k] = json.load(f)

    results = {"branch_a": {k: [] for k in range(NUM_DISEASES)},
               "branch_b": {k: [] for k in range(NUM_DISEASES)}}

    for alpha in alpha_list:
        print(f"Running Step 6 with alpha_A={alpha}, alpha_B={alpha}...")
        step6_calibrate_phase2.run(dir_step5, dir_step6, splits_a, splits_b, alpha, alpha, beta)

        for k in range(NUM_DISEASES):
            for branch, splits, suffix in [("a", splits_a, f"alphaA_{alpha}"), ("b", splits_b, f"alphaB_{alpha}")]:
                path = os.path.join(dir_step6,
                                    f"lec_branch_{branch}_disease_{k}_beta_{beta}_{suffix}.json")
                if not os.path.exists(path):
                    results[f"branch_{branch}"][k].append((alpha, None, None))
                    continue
                with open(path) as f:
                    lec_result = json.load(f)
                mean_cov, std_cov = compute_coverage(lec_result, disease_data[branch][k], splits)
                results[f"branch_{branch}"][k].append((alpha, mean_cov, std_cov))

    return results


def plot_coverage(results, branch_key, alpha_list, img_dir):
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "stix",
        "font.size": 11,
    })

    fig, axes = plt.subplots(2, 5, figsize=(20, 8), sharey=True)
    axes = axes.flatten()

    alpha_label = r"$\alpha_A$" if branch_key == "branch_a" else r"$\alpha_B$"

    for k in range(NUM_DISEASES):
        ax = axes[k]
        entries = results[branch_key][k]

        alphas, means, stds = [], [], []
        for alpha, mean_cov, std_cov in entries:
            if mean_cov is not None:
                alphas.append(alpha)
                means.append(mean_cov)
                stds.append(std_cov if std_cov is not None else 0)

        if alphas:
            alphas = np.array(alphas)
            means = np.array(means)
            stds = np.array(stds)
            ax.plot(alphas, means, "o-", color="steelblue", markersize=4, linewidth=1.5)
            ax.fill_between(alphas, np.clip(means - stds, 0, 1),
                            np.clip(means + stds, 0, 1),
                            color="steelblue", alpha=0.2)

        ax.set_title(f"D{k}: {DISEASE_NAMES[k]}", fontsize=9)
        ax.set_xlim(0, max(alpha_list) + 0.05)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        if k >= 5:
            ax.set_xlabel(alpha_label, fontsize=9)
        if k % 5 == 0:
            ax.set_ylabel("Coverage", fontsize=9)
        ax.tick_params(labelsize=7)

    branch_label = "Branch A (AI Routing)" if branch_key == "branch_a" else "Branch B (Human Routing)"
    fig.suptitle(f"{branch_label}: Coverage vs {alpha_label} (mean $\\pm$ std)",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    save_path = os.path.join(img_dir, f"coverage_vs_alpha_{branch_key}.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Saved: {save_path}")


def main():
    p = argparse.ArgumentParser(description="Plot coverage vs alpha curves per disease")
    p.add_argument("--output_dir", default="result/json")
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--alpha_list", nargs="+", type=float,
                   default=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])
    p.add_argument("--img_dir", default="result/img/coverage")
    args = p.parse_args()

    os.makedirs(args.img_dir, exist_ok=True)
    results = sweep_coverage(args.output_dir, sorted(args.alpha_list), args.beta)
    plot_coverage(results, "branch_a", args.alpha_list, args.img_dir)
    plot_coverage(results, "branch_b", args.alpha_list, args.img_dir)
    print(f"\n[Done] Coverage plots saved to {args.img_dir}")


if __name__ == "__main__":
    main()
