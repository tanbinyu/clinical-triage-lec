import argparse
import json
import os

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np

from evaluation.evaluate import evaluate_lec_result
from pipeline import step3_calibrate_phase1, step4_split_branches, step6_calibrate_phase2
from pipeline.config import NUM_DISEASES, DISEASE_NAMES


def sweep(output_dir, beta_list, alpha_list):
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
    grid_a = np.full((n1, n2), np.nan)
    grid_b = np.full((n1, n2), np.nan)

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

            fdrs_a, fdrs_b = [], []
            for k in range(NUM_DISEASES):
                for branch, fdr_list, suffix in [("a", fdrs_a, f"alphaA_{alpha}"), ("b", fdrs_b, f"alphaB_{alpha}")]:
                    path = os.path.join(dir_step6,
                                        f"lec_branch_{branch}_disease_{k}_beta_{beta}_{suffix}.json")
                    if not os.path.exists(path):
                        continue
                    with open(path) as f:
                        data = json.load(f)
                    ev = evaluate_lec_result(data, alpha)
                    if ev["mean_fdp"] is not None:
                        fdr_list.append(ev["mean_fdp"])

            if fdrs_a:
                grid_a[i, j] = np.mean(fdrs_a)
            if fdrs_b:
                grid_b[i, j] = np.mean(fdrs_b)

    return grid_a, grid_b


def plot_3d(grid, beta_list, alpha_list, branch_label, save_path):
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "stix",
        "font.size": 11,
        "axes.labelsize": 13,
        "axes.titlesize": 15,
    })

    X, Y = np.meshgrid(beta_list, alpha_list, indexing="ij")

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    Z_ref = Y.copy()
    ax.plot_surface(X, Y, Z_ref, color="gray", alpha=0.15, zorder=1)
    ax.plot_wireframe(X, Y, Z_ref, color="gray", alpha=0.2, linewidth=0.5, zorder=2)

    alpha_label = r"$\alpha_A$" if "A" in branch_label else r"$\alpha_B$"

    surf = ax.plot_surface(X, Y, grid, cmap="coolwarm", alpha=0.88,
                           edgecolor="k", linewidth=0.3, zorder=5,
                           vmin=0, vmax=max(alpha_list) + 0.05)

    ax.set_xlabel(r"$\beta$ (routing)", labelpad=10)
    ax.set_ylabel(f"{alpha_label} (diagnosis)", labelpad=10)
    ax.set_zlabel("mean FDP", labelpad=8)
    ax.set_title(f"{branch_label}\nFDP vs ($\\beta$, {alpha_label})", pad=15)

    ax.set_xlim(min(beta_list), max(beta_list))
    ax.set_ylim(min(alpha_list), max(alpha_list))
    ax.set_zlim(0, max(alpha_list) + 0.08)

    ax.view_init(elev=25, azim=225)
    ax.tick_params(labelsize=9)

    cbar = fig.colorbar(surf, ax=ax, shrink=0.55, pad=0.1, aspect=15)
    cbar.set_label("mean mFDR", fontsize=11)
    cbar.ax.tick_params(labelsize=9)

    ax.text2D(0.02, 0.02, f"gray plane: mFDR = {alpha_label} (target)",
              transform=ax.transAxes, fontsize=8, color="gray", style="italic")

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Saved: {save_path}")


def main():
    p = argparse.ArgumentParser(description="3D sweep of (beta, alpha) and plot mFDR surface")
    p.add_argument("--output_dir", default="result/json")
    p.add_argument("--beta_list", nargs="+", type=float,
                   default=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3])
    p.add_argument("--alpha_list", nargs="+", type=float,
                   default=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3])
    p.add_argument("--img_dir", default="result/img/3d_surface")
    args = p.parse_args()

    os.makedirs(args.img_dir, exist_ok=True)
    b = sorted(args.beta_list)
    a = sorted(args.alpha_list)

    grid_a, grid_b = sweep(args.output_dir, b, a)

    plot_3d(grid_a, b, a, "Branch A (AI Routing)",
            os.path.join(args.img_dir, "mfdr_3d_branch_a.png"))
    plot_3d(grid_b, b, a, "Branch B (Human Routing)",
            os.path.join(args.img_dir, "mfdr_3d_branch_b.png"))

    print(f"\n[Done] 3D plots saved to {args.img_dir}")


if __name__ == "__main__":
    main()
