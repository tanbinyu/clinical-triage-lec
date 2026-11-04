import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np

from evaluation.evaluate import evaluate_lec_result
from pipeline import step3_calibrate_phase1


def sweep_and_evaluate(output_dir, beta_list):
    dir_step2 = os.path.join(output_dir, "step2")
    dir_step3 = os.path.join(output_dir, "step3")

    with open(os.path.join(dir_step2, "splits_phase_1.json")) as f:
        splits = json.load(f)
    with open(os.path.join(output_dir, "step1", "phase_1.json")) as f:
        phase1_data = json.load(f)

    results = []
    for beta in beta_list:
        print(f"Running Step 3 with beta={beta}...")
        lec_p1 = step3_calibrate_phase1.run(phase1_data, splits, beta)

        out_path = os.path.join(dir_step3, f"lec_phase_1_beta_{beta}.json")
        with open(out_path, "w") as f:
            json.dump(lec_p1, f, indent=2)

        ev = evaluate_lec_result(lec_p1, beta)

        fdr_list = []
        for s in lec_p1["splits"]:
            if s["t_hat"] is not None:
                sel = s["selected_num"]
                err = s["error_num"]
                fdr_list.append(err / sel if sel > 0 else 0.0)

        results.append((beta, ev["mean_fdp"], ev["std_fdp"], ev["mfdr"], fdr_list))
        status = "PASS" if ev["controlled"] else "FAIL"
        print(f"  [{status}] mean_fdp={ev['mean_fdp']}, std={ev['std_fdp']}, "
              f"mfdr={ev['mfdr']}, feasible={ev['valid_epochs']}/{ev['total_epochs']}")

    return results


def plot_boxplot(results, beta_list, img_dir):
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "stix",
        "font.size": 11,
    })
    fig, ax = plt.subplots(figsize=(10, 6))

    sorted_betas = sorted(beta_list)
    n = len(sorted_betas)

    bp_data = []
    bp_positions = []
    for beta, _, _, _, fdr_list in results:
        idx = sorted_betas.index(beta)
        if fdr_list:
            bp_data.append(fdr_list)
            bp_positions.append(idx)

    if bp_data:
        bp = ax.boxplot(bp_data, positions=bp_positions, widths=0.6,
                        patch_artist=True, showmeans=True,
                        meanprops=dict(marker="D", markerfacecolor="red", markersize=5),
                        medianprops=dict(color="black", linewidth=1.5),
                        flierprops=dict(marker="o", markersize=3, alpha=0.5))
        for patch in bp["boxes"]:
            patch.set_facecolor("skyblue")
            patch.set_alpha(0.6)

    ax.plot(range(n), sorted_betas, "k--", alpha=0.5, linewidth=1,
            label=r"$y = \beta$ (target)")

    ax.set_xlabel(r"$\beta$ (target)", fontsize=13)
    ax.set_ylabel("FDP", fontsize=13)
    ax.set_title(r"Phase 1 (Router): FDP vs $\beta$", fontsize=14)
    ax.set_xticks(range(n))
    ax.set_xticklabels([str(b) for b in sorted_betas], rotation=45, ha="right")
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=10)

    plt.tight_layout()
    save_path = os.path.join(img_dir, "fdr_boxplot_phase1.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Saved: {save_path}")


def plot_mfdr_scatter(results, beta_list, img_dir, save_name="mfdr_scatter_phase1.png"):
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "stix",
        "font.size": 11,
    })
    fig, ax = plt.subplots(figsize=(10, 6))

    sorted_betas = sorted(beta_list)
    n = len(sorted_betas)

    xs, ys = [], []
    for beta, _, _, mfdr, _ in results:
        if mfdr is not None:
            xs.append(sorted_betas.index(beta))
            ys.append(mfdr)

    ax.scatter(xs, ys, c="steelblue", s=60, zorder=5, edgecolors="black", linewidths=0.8,
               label="mFDR")
    if len(xs) > 1:
        ax.plot(xs, ys, "-", color="steelblue", alpha=0.5, linewidth=1.5)

    ax.plot(range(n), sorted_betas, "k--", alpha=0.5, linewidth=1,
            label=r"$y = \beta$ (target)")

    ax.set_xlabel(r"$\beta$ (target mFDR)", fontsize=13)
    ax.set_ylabel("mFDR (E[errors]/E[selected])", fontsize=13)
    ax.set_title(r"Phase 1 (Router): mFDR vs $\beta$" "\n"
                 r"(mFDR = E[errors]/E[selected])", fontsize=14)
    ax.set_xticks(range(n))
    ax.set_xticklabels([str(b) for b in sorted_betas], rotation=45, ha="right")
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=10)

    plt.tight_layout()
    save_path = os.path.join(img_dir, save_name)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Saved: {save_path}")


def main():
    p = argparse.ArgumentParser(description="Sweep beta and plot Phase 1 FDP boxplot")
    p.add_argument("--output_dir", default="result/json")
    p.add_argument("--beta_list", nargs="+", type=float,
                   default=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])
    p.add_argument("--img_dir", default="result/img")
    args = p.parse_args()

    boxplot_dir = os.path.join(args.img_dir, "boxplot")
    scatter_dir = os.path.join(args.img_dir, "scatter")
    os.makedirs(boxplot_dir, exist_ok=True)
    os.makedirs(scatter_dir, exist_ok=True)

    results = sweep_and_evaluate(args.output_dir, sorted(args.beta_list))
    plot_boxplot(results, args.beta_list, boxplot_dir)
    plot_mfdr_scatter(results, args.beta_list, scatter_dir)
    plot_mfdr_scatter(results, args.beta_list, boxplot_dir, "mfdr_boxplot_phase1.png")

    print(f"\n[Done] Plots saved to {args.img_dir}")


if __name__ == "__main__":
    main()
