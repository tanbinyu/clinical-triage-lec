import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np

from evaluation.evaluate import evaluate_lec_result
from pipeline import step6_calibrate_phase2
from pipeline.config import (NUM_DISEASES, DISEASE_NAMES, DISEASE_TO_DATASETS,
                              DATASET_ERROR_RATES, DATASET_SAMPLE_COUNTS)


def sweep_and_evaluate(output_dir, alpha_list, beta):
    dir_step4 = os.path.join(output_dir, "step4")
    dir_step5 = os.path.join(output_dir, "step5")
    dir_step6 = os.path.join(output_dir, "step6")

    with open(os.path.join(dir_step4, f"splits_branch_a_beta_{beta}.json")) as f:
        splits_a = json.load(f)
    with open(os.path.join(dir_step4, f"splits_branch_b_beta_{beta}.json")) as f:
        splits_b = json.load(f)

    results = {"branch_a": {k: [] for k in range(NUM_DISEASES)},
               "branch_b": {k: [] for k in range(NUM_DISEASES)}}

    for alpha in alpha_list:
        print(f"Running Step 6 with alpha_A={alpha}, alpha_B={alpha}...")
        step6_calibrate_phase2.run(dir_step5, dir_step6, splits_a, splits_b, alpha, alpha, beta)

        for k in range(NUM_DISEASES):
            for branch, suffix in [("a", f"alphaA_{alpha}"), ("b", f"alphaB_{alpha}")]:
                path = os.path.join(dir_step6, f"lec_branch_{branch}_disease_{k}_beta_{beta}_{suffix}.json")
                if not os.path.exists(path):
                    results[f"branch_{branch}"][k].append((alpha, None, None, None, []))
                    continue
                with open(path) as f:
                    data = json.load(f)
                ev = evaluate_lec_result(data, alpha)

                fdr_list = []
                for s in data["splits"]:
                    if s["t_hat"] is not None:
                        sel = s["selected_num"]
                        err = s["error_num"]
                        fdr_list.append(err / sel if sel > 0 else 0.0)

                results[f"branch_{branch}"][k].append(
                    (alpha, ev["mean_fdp"], ev["std_fdp"], ev["mfdr"], fdr_list)
                )

    return results


def plot_branch(results, branch_key, alpha_list, img_dir):
    plt.rcParams.update({
        "figure.figsize": (12, 7),
        "font.size": 11,
        "axes.grid": True,
        "grid.alpha": 0.3,
    })

    fig, ax = plt.subplots()
    colors = plt.cm.tab10(np.linspace(0, 1, NUM_DISEASES))

    alpha_label = r"$\alpha_A$" if branch_key == "branch_a" else r"$\alpha_B$"

    for k in range(NUM_DISEASES):
        entries = results[branch_key][k]
        alphas, means, stds = [], [], []
        for alpha, mean_fdp, std_fdp, _mfdr, _fdr_list in entries:
            if mean_fdp is not None:
                alphas.append(alpha)
                means.append(mean_fdp)
                stds.append(std_fdp if std_fdp is not None else 0)

        if not alphas:
            continue

        alphas = np.array(alphas)
        means = np.array(means)
        stds = np.array(stds)

        label = f"D{k}: {DISEASE_NAMES[k]}"
        ax.plot(alphas, means, "o-", color=colors[k], label=label, markersize=4, linewidth=1.5)
        ax.fill_between(alphas, means - stds, means + stds, color=colors[k], alpha=0.15)

    max_alpha = max(alpha_list)
    ax.plot([0, max_alpha + 0.05], [0, max_alpha + 0.05], "k--", alpha=0.5, linewidth=1, label=r"y = $\alpha$ (ideal)")

    branch_label = "Branch A (AI Routing)" if branch_key == "branch_a" else "Branch B (Human Routing)"
    ax.set_xlabel(f"{alpha_label} (target FDP)", fontsize=12)
    ax.set_ylabel("Empirical FDP (mean ± std)", fontsize=12)
    ax.set_title(f"{branch_label}: FDP vs {alpha_label} by Disease", fontsize=14)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_xlim(0, max_alpha + 0.05)
    ax.set_ylim(0, max_alpha + 0.1)

    plt.tight_layout()
    save_path = os.path.join(img_dir, f"mfdr_vs_alpha_{branch_key}.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Saved: {save_path}")


def plot_branch_boxplot(results, branch_key, alpha_list, img_dir,
                        base_error_rates=None):
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "stix",
        "font.size": 11,
    })
    fig, axes = plt.subplots(2, 5, figsize=(20, 8), sharey=True)
    axes = axes.flatten()

    alpha_label = r"$\alpha_A$" if branch_key == "branch_a" else r"$\alpha_B$"
    sorted_alphas = sorted(alpha_list)
    n = len(sorted_alphas)

    for k in range(NUM_DISEASES):
        ax = axes[k]
        entries = results[branch_key][k]

        bp_data = []
        bp_positions = []
        for alpha, _, _, _, fdr_list in entries:
            idx = sorted_alphas.index(alpha)
            if fdr_list:
                bp_data.append(fdr_list)
                bp_positions.append(idx)

        if bp_data:
            bp = ax.boxplot(bp_data, positions=bp_positions, widths=0.6,
                            patch_artist=True, showmeans=True,
                            meanprops=dict(marker="D", markerfacecolor="red", markersize=3),
                            medianprops=dict(color="black", linewidth=1.5),
                            flierprops=dict(marker="o", markersize=2, alpha=0.3))
            for patch in bp["boxes"]:
                patch.set_facecolor("skyblue")
                patch.set_alpha(0.6)

        if base_error_rates is not None and k in base_error_rates:
            base_err = base_error_rates[k]
            ax.axhline(y=base_err, color="red", linestyle=":", linewidth=1.2, alpha=0.7)
            ax.text(n - 0.5, base_err, f"{base_err:.0%}",
                    fontsize=6, color="red", va="bottom")

        ax.plot(range(n), sorted_alphas, "k--", alpha=0.4, linewidth=1)

        ax.set_title(f"D{k}: {DISEASE_NAMES[k]}", fontsize=9)
        ax.set_xticks(range(n))
        ax.set_xticklabels([str(a) for a in sorted_alphas], rotation=45, ha="right")
        ax.grid(True, alpha=0.3)

        if k >= 5:
            ax.set_xlabel(alpha_label, fontsize=9)
        if k % 5 == 0:
            ax.set_ylabel("FDP", fontsize=9)
        ax.tick_params(labelsize=7)

    branch_label = "Branch A (AI Routing)" if branch_key == "branch_a" else "Branch B (Human Routing)"
    fig.suptitle(f"{branch_label}: FDP vs {alpha_label} by Disease",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    save_path = os.path.join(img_dir, f"fdr_boxplot_{branch_key}.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Saved: {save_path}")


def collect_unified_epoch_fdrs(output_dir, alpha_list, beta):
    dir_step6 = os.path.join(output_dir, "step6")
    unified = {}

    for alpha in alpha_list:
        disease_splits = {}
        num_epochs = None
        for k in range(NUM_DISEASES):
            path = os.path.join(dir_step6,
                                f"lec_branch_a_disease_{k}_beta_{beta}_alphaA_{alpha}.json")
            if not os.path.exists(path):
                continue
            with open(path) as f:
                data = json.load(f)
            disease_splits[k] = data["splits"]
            if num_epochs is None:
                num_epochs = len(data["splits"])

        if num_epochs is None:
            continue

        epoch_fdrs = []
        for e in range(num_epochs):
            total_selected = 0
            total_errors = 0
            for key, splits in disease_splits.items():
                s = splits[e]
                total_selected += s["selected_num"]
                total_errors += s["error_num"]
            if total_selected > 0:
                epoch_fdrs.append(total_errors / total_selected)

        unified[alpha] = epoch_fdrs

    return unified


def plot_unified_boxplot(unified, alpha_list, img_dir):
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "stix",
        "font.size": 11,
    })
    fig, ax = plt.subplots(figsize=(10, 6))

    sorted_alphas = sorted(alpha_list)
    n = len(sorted_alphas)

    bp_data = []
    bp_positions = []
    for i, alpha in enumerate(sorted_alphas):
        fdrs = unified.get(alpha, [])
        if fdrs:
            bp_data.append(fdrs)
            bp_positions.append(i)

    if bp_data:
        bp = ax.boxplot(bp_data, positions=bp_positions, widths=0.6,
                        patch_artist=True, showmeans=True,
                        meanprops=dict(marker="D", markerfacecolor="red", markersize=4),
                        medianprops=dict(color="black", linewidth=1.5),
                        flierprops=dict(marker="o", markersize=2, alpha=0.3))
        for patch in bp["boxes"]:
            patch.set_facecolor("steelblue")
            patch.set_alpha(0.5)

    ax.plot(range(n), sorted_alphas, "k--", alpha=0.5, linewidth=1,
            label=r"$y = \alpha$ (target)")

    ax.set_xlabel(r"$\alpha$ (target)", fontsize=13)
    ax.set_ylabel("Unified FDP (all diseases merged)", fontsize=13)
    ax.set_title("Unified FDP Distribution: All Diseases Merged per Epoch\n"
                 r"(Branch A, $\alpha_A = \alpha$)",
                 fontsize=14)
    ax.set_xticks(range(n))
    ax.set_xticklabels([str(a) for a in sorted_alphas], rotation=45, ha="right")
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=10)

    plt.tight_layout()
    save_path = os.path.join(img_dir, "fdr_boxplot_unified.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Saved: {save_path}")


def plot_branch_mfdr_scatter(results, branch_key, alpha_list, img_dir,
                             base_error_rates=None,
                             save_name=None):
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "stix",
        "font.size": 11,
    })
    fig, axes = plt.subplots(2, 5, figsize=(20, 8), sharey=True)
    axes = axes.flatten()

    alpha_label = r"$\alpha_A$" if branch_key == "branch_a" else r"$\alpha_B$"
    sorted_alphas = sorted(alpha_list)
    n = len(sorted_alphas)

    for k in range(NUM_DISEASES):
        ax = axes[k]
        entries = results[branch_key][k]

        xs, ys = [], []
        for alpha, _, _, mfdr, _ in entries:
            if mfdr is not None:
                xs.append(sorted_alphas.index(alpha))
                ys.append(mfdr)

        ax.scatter(xs, ys, c="steelblue", s=40, zorder=5, edgecolors="black", linewidths=0.5)
        if len(xs) > 1:
            ax.plot(xs, ys, "-", color="steelblue", alpha=0.5, linewidth=1)

        if base_error_rates is not None and k in base_error_rates:
            base_err = base_error_rates[k]
            ax.axhline(y=base_err, color="red", linestyle=":", linewidth=1.2, alpha=0.7)
            ax.text(n - 0.5, base_err, f"{base_err:.0%}",
                    fontsize=6, color="red", va="bottom")

        ax.plot(range(n), sorted_alphas, "k--", alpha=0.4, linewidth=1)

        ax.set_title(f"D{k}: {DISEASE_NAMES[k]}", fontsize=9)
        ax.set_xticks(range(n))
        ax.set_xticklabels([str(a) for a in sorted_alphas], rotation=45, ha="right")
        ax.grid(True, alpha=0.3)

        if k >= 5:
            ax.set_xlabel(alpha_label, fontsize=9)
        if k % 5 == 0:
            ax.set_ylabel("mFDR", fontsize=9)
        ax.tick_params(labelsize=7)

    branch_label = "Branch A (AI Routing)" if branch_key == "branch_a" else "Branch B (Human Routing)"
    fig.suptitle(f"{branch_label}: mFDR vs {alpha_label} (mFDR = E[errors]/E[selected])",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    if save_name is None:
        save_name = f"mfdr_scatter_{branch_key}.png"
    save_path = os.path.join(img_dir, save_name)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Saved: {save_path}")


def plot_unified_mfdr_scatter(unified, alpha_list, img_dir,
                              save_name="mfdr_scatter_unified.png"):
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "stix",
        "font.size": 11,
    })
    fig, ax = plt.subplots(figsize=(10, 6))

    sorted_alphas = sorted(alpha_list)
    n = len(sorted_alphas)

    xs, ys = [], []
    for i, alpha in enumerate(sorted_alphas):
        fdrs = unified.get(alpha, [])
        if fdrs:
            mfdr = sum(fdrs) / len(fdrs)
            xs.append(i)
            ys.append(mfdr)

    ax.scatter(xs, ys, c="steelblue", s=60, zorder=5, edgecolors="black", linewidths=0.8,
               label="mFDR")
    if len(xs) > 1:
        ax.plot(xs, ys, "-", color="steelblue", alpha=0.5, linewidth=1.5)

    ax.plot(range(n), sorted_alphas, "k--", alpha=0.5, linewidth=1,
            label=r"$y = \alpha$ (target)")

    ax.set_xlabel(r"$\alpha$ (target mFDR)", fontsize=13)
    ax.set_ylabel("Unified mFDR (E[errors]/E[selected])", fontsize=13)
    ax.set_title("Unified mFDR: All Diseases Merged\n"
                 r"(Branch A, $\alpha_A = \alpha$)",
                 fontsize=14)
    ax.set_xticks(range(n))
    ax.set_xticklabels([str(a) for a in sorted_alphas], rotation=45, ha="right")
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=10)

    plt.tight_layout()
    save_path = os.path.join(img_dir, save_name)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Saved: {save_path}")


def main():
    p = argparse.ArgumentParser(description="Sweep alpha_A/alpha_B and plot FDP/mFDR")
    p.add_argument("--output_dir", default="result/json")
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--alpha_list", nargs="+", type=float,
                   default=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])
    p.add_argument("--img_dir", default="result/img")
    args = p.parse_args()

    os.makedirs(args.img_dir, exist_ok=True)
    curve_dir = os.path.join(args.img_dir, "curve")
    boxplot_dir = os.path.join(args.img_dir, "boxplot")
    os.makedirs(curve_dir, exist_ok=True)
    os.makedirs(boxplot_dir, exist_ok=True)

    results = sweep_and_evaluate(args.output_dir, sorted(args.alpha_list), args.beta)

    base_err = {}
    dir_step6 = os.path.join(args.output_dir, "step6")
    first_alpha = sorted(args.alpha_list)[0]
    for k in range(NUM_DISEASES):
        for branch, suffix in [("a", f"alphaA_{first_alpha}"), ("b", f"alphaB_{first_alpha}")]:
            path = os.path.join(dir_step6, f"lec_branch_{branch}_disease_{k}_beta_{args.beta}_{suffix}.json")
            if os.path.exists(path):
                with open(path) as f:
                    base_err[(branch, k)] = json.load(f)["base_error_rate"]

    base_err_a = {k: base_err[(b, k)] for b, k in base_err if b == "a"}
    base_err_b = {k: base_err[(b, k)] for b, k in base_err if b == "b"}

    plot_branch(results, "branch_a", args.alpha_list, curve_dir)
    plot_branch(results, "branch_b", args.alpha_list, curve_dir)
    plot_branch_boxplot(results, "branch_a", args.alpha_list, boxplot_dir, base_err_a)
    plot_branch_boxplot(results, "branch_b", args.alpha_list, boxplot_dir, base_err_b)

    unified = collect_unified_epoch_fdrs(args.output_dir, sorted(args.alpha_list), args.beta)
    plot_unified_boxplot(unified, sorted(args.alpha_list), boxplot_dir)

    scatter_dir = os.path.join(args.img_dir, "scatter")
    os.makedirs(scatter_dir, exist_ok=True)
    plot_branch_mfdr_scatter(results, "branch_a", args.alpha_list, scatter_dir, base_err_a)
    plot_branch_mfdr_scatter(results, "branch_b", args.alpha_list, scatter_dir, base_err_b)
    plot_unified_mfdr_scatter(unified, sorted(args.alpha_list), scatter_dir)

    plot_branch_mfdr_scatter(results, "branch_a", args.alpha_list, boxplot_dir, base_err_a,
                             save_name="mfdr_boxplot_branch_a.png")
    plot_branch_mfdr_scatter(results, "branch_b", args.alpha_list, boxplot_dir, base_err_b,
                             save_name="mfdr_boxplot_branch_b.png")
    plot_unified_mfdr_scatter(unified, sorted(args.alpha_list), boxplot_dir,
                              save_name="mfdr_boxplot_unified.png")

    print(f"\n[Done] Plots saved to {args.img_dir}")


if __name__ == "__main__":
    main()
