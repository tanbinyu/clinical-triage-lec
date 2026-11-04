import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np

from evaluation.evaluate import evaluate_lec_result
from pipeline import step3_calibrate_phase1, step4_split_branches, step6_calibrate_phase2
from pipeline.config import NUM_DISEASES


def sweep_controlled(output_dir, beta_list, alpha_list):
    dir_step2 = os.path.join(output_dir, "step2")
    dir_step3 = os.path.join(output_dir, "step3")
    dir_step4 = os.path.join(output_dir, "step4")
    dir_step5 = os.path.join(output_dir, "step5")
    dir_step6 = os.path.join(output_dir, "step6")

    with open(os.path.join(dir_step2, "splits_phase_1.json")) as f:
        splits = json.load(f)
    with open(os.path.join(output_dir, "step1", "phase_1.json")) as f:
        phase1_data = json.load(f)

    n1, n2 = len(beta_list), len(alpha_list)
    grid_a = np.zeros((n1, n2), dtype=int)
    grid_b = np.zeros((n1, n2), dtype=int)

    for i, beta in enumerate(beta_list):
        print(f"\n[beta={beta}] Running Step 3 + Step 4...")
        lec_p1 = step3_calibrate_phase1.run(phase1_data, splits, beta)
        with open(os.path.join(dir_step3, f"lec_phase_1_beta_{beta}.json"), "w") as f:
            json.dump(lec_p1, f, indent=2)

        splits_a, splits_b = step4_split_branches.run(phase1_data, splits, lec_p1)
        with open(os.path.join(dir_step4, f"splits_branch_a_beta_{beta}.json"), "w") as f:
            json.dump(splits_a, f, indent=2)
        with open(os.path.join(dir_step4, f"splits_branch_b_beta_{beta}.json"), "w") as f:
            json.dump(splits_b, f, indent=2)

        for j, alpha in enumerate(alpha_list):
            print(f"  [alpha={alpha}] Running Step 6 + Evaluate...")
            step6_calibrate_phase2.run(dir_step5, dir_step6, splits_a, splits_b, alpha, alpha, beta)

            for k in range(NUM_DISEASES):
                for branch, grid, suffix in [("a", grid_a, f"alphaA_{alpha}"), ("b", grid_b, f"alphaB_{alpha}")]:
                    path = os.path.join(dir_step6,
                                        f"lec_branch_{branch}_disease_{k}_beta_{beta}_{suffix}.json")
                    if not os.path.exists(path):
                        continue
                    with open(path) as f:
                        data = json.load(f)
                    ev = evaluate_lec_result(data, alpha)
                    if ev["controlled"]:
                        grid[i, j] += 1

    return grid_a, grid_b


def plot_heatmap(grid, beta_list, alpha_list, branch_label, save_path):
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "stix",
        "font.size": 11,
    })

    fig, ax = plt.subplots(figsize=(8, 6.5))

    alpha_label = r"$\alpha_A$" if "A" in branch_label else r"$\alpha_B$"

    im = ax.imshow(grid.T, origin="lower", aspect="auto",
                   cmap="RdYlGn", vmin=0, vmax=NUM_DISEASES,
                   extent=[beta_list[0] - 0.025, beta_list[-1] + 0.025,
                           alpha_list[0] - 0.025, alpha_list[-1] + 0.025])

    for i in range(len(beta_list)):
        for j in range(len(alpha_list)):
            val = grid[i, j]
            color = "white" if val <= 3 else "black"
            ax.text(beta_list[i], alpha_list[j], str(val),
                    ha="center", va="center", fontsize=14, fontweight="bold", color=color)

    ax.set_xticks(beta_list)
    ax.set_yticks(alpha_list)
    ax.set_xlabel(r"$\beta$ (routing)", fontsize=13)
    ax.set_ylabel(f"{alpha_label} (diagnosis)", fontsize=13)
    ax.set_title(f"{branch_label}\nControlled Diseases (out of {NUM_DISEASES})", fontsize=14)
    ax.tick_params(labelsize=10)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.08)
    cbar.set_label("Number of controlled diseases", fontsize=11)
    cbar.set_ticks(range(0, NUM_DISEASES + 1, 2))
    cbar.ax.tick_params(labelsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Saved: {save_path}")


def main():
    p = argparse.ArgumentParser(description="Heatmap of controlled disease count over (beta, alpha)")
    p.add_argument("--output_dir", default="result/json")
    p.add_argument("--beta_list", nargs="+", type=float,
                   default=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3])
    p.add_argument("--alpha_list", nargs="+", type=float,
                   default=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3])
    p.add_argument("--img_dir", default="result/img/heatmap")
    args = p.parse_args()

    os.makedirs(args.img_dir, exist_ok=True)
    b = sorted(args.beta_list)
    a = sorted(args.alpha_list)

    grid_a, grid_b = sweep_controlled(args.output_dir, b, a)

    plot_heatmap(grid_a, b, a, "Branch A (AI Routing)",
                 os.path.join(args.img_dir, "controlled_heatmap_branch_a.png"))
    plot_heatmap(grid_b, b, a, "Branch B (Human Routing)",
                 os.path.join(args.img_dir, "controlled_heatmap_branch_b.png"))

    print(f"\n[Done] Heatmaps saved to {args.img_dir}")


if __name__ == "__main__":
    main()
