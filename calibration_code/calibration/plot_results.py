"""
可视化模块：绘制 LEC 实验结果图表。

图表类型：
1. mFDR vs α 曲线（验证 mFDR ≤ α）
2. Retention vs α 曲线
3. 按疾病对比的 retention 柱状图
4. Branch A vs Branch B 对比图
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from calibration.config import (
    DISEASE_NAMES,
    DISEASE_TO_DATASETS,
    RESULTS_DIR,
    RESULTS_BRANCH_A_DIR,
    RESULTS_BRANCH_B_DIR,
    PROJECT_ROOT,
)

# 图片保存目录
PLOT_DIR = os.path.join(PROJECT_ROOT, "res", "res_img", "lec_plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# 绘图风格配置
plt.rcParams.update({
    "figure.figsize": (10, 6),
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
})


def plot_stage1_routing(results_path=None):
    """
    绘制实验①的结果：Routing mFDR 和 Retention vs α^R。

    生成两子图:
    - 左: mFDR vs α^R (带 y=x 参考线)
    - 右: Retention vs α^R
    """
    if results_path is None:
        results_path = os.path.join(RESULTS_DIR, "stage1_routing.json")

    if not os.path.exists(results_path):
        print(f"[SKIP] 未找到实验①结果: {results_path}")
        return

    with open(results_path) as f:
        results = json.load(f)

    # 过滤有效结果
    valid = [r for r in results if r.get("mean_fdp") is not None]
    if not valid:
        print("[SKIP] 实验①无有效结果")
        return

    alphas = [r["alpha"] for r in valid]
    fdp_means = [r["mean_fdp"] for r in valid]
    fdp_stds = [r["std_fdp"] for r in valid]
    retentions = [r["mean_retention_rate"] for r in valid]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 左图: FDP vs α^R
    ax1.errorbar(alphas, fdp_means, yerr=fdp_stds, fmt="o-", capsize=4,
                 color="tab:blue", label="Empirical FDP")
    ax1.plot([0, max(alphas) + 0.05], [0, max(alphas) + 0.05],
             "k--", alpha=0.5, label="y = α (target)")
    ax1.set_xlabel("α^R (target FDP)")
    ax1.set_ylabel("Empirical FDP")
    ax1.set_title("Stage 1: Routing FDP Control")
    ax1.legend()
    ax1.set_xlim(0, max(alphas) + 0.05)
    ax1.set_ylim(0, max(alphas) + 0.05)

    # 右图: Retention vs α^R
    ax2.plot(alphas, retentions, "s-", color="tab:green", markersize=7)
    ax2.set_xlabel("α^R (target FDP)")
    ax2.set_ylabel("Retention Rate")
    ax2.set_title("Stage 1: Router Retention Rate")
    ax2.set_xlim(0, max(alphas) + 0.05)
    ax2.set_ylim(0, 1.05)

    plt.tight_layout()
    save_path = os.path.join(PLOT_DIR, "stage1_routing.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Saved: {save_path}")


def plot_stage2_by_disease(branch="a", results_dir=None):
    """
    绘制实验②/③的结果：按疾病绘制 mFDR 和 Retention vs α。

    每种疾病一条线，所有疾病在同一张图上对比。

    参数:
        branch: str, "a" 或 "b"
        results_dir: str, 结果目录
    """
    if results_dir is None:
        results_dir = RESULTS_BRANCH_A_DIR if branch == "a" else RESULTS_BRANCH_B_DIR

    if not os.path.exists(results_dir):
        print(f"[SKIP] 未找到结果目录: {results_dir}")
        return

    # 加载所有疾病结果
    disease_results = {}
    for fname in sorted(os.listdir(results_dir)):
        if not fname.endswith(".json"):
            continue
        parts = fname.replace(".json", "").split("_", 2)
        if len(parts) >= 2 and parts[0] == "disease":
            try:
                disease_id = int(parts[1])
            except ValueError:
                continue
            with open(os.path.join(results_dir, fname)) as f:
                disease_results[disease_id] = json.load(f)

    if not disease_results:
        print(f"[SKIP] 目录 {results_dir} 中无有效结果")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(disease_results)))

    for idx, (disease_id, results) in enumerate(sorted(disease_results.items())):
        valid = [r for r in results if r.get("mean_fdp") is not None]
        if not valid:
            continue

        disease_name = DISEASE_NAMES.get(disease_id, f"D{disease_id}")
        alphas = [r["alpha"] for r in valid]
        fdp_means = [r["mean_fdp"] for r in valid]
        retentions = [r["mean_retention_rate"] for r in valid]
        color = colors[idx]

        ax1.plot(alphas, fdp_means, "o-", color=color, label=f"D{disease_id}: {disease_name}",
                 markersize=5)
        ax2.plot(alphas, retentions, "s-", color=color, label=f"D{disease_id}: {disease_name}",
                 markersize=5)

    # y=x 参考线
    max_alpha = 0.5
    ax1.plot([0, max_alpha + 0.05], [0, max_alpha + 0.05], "k--", alpha=0.5, label="y=α")
    ax1.set_xlabel("α (target FDP)")
    ax1.set_ylabel("Empirical FDP")
    branch_label = "Branch A" if branch == "a" else "Branch B"
    ax1.set_title(f"Stage 2 {branch_label}: FDP by Disease")
    ax1.legend(fontsize=8, loc="upper left")
    ax1.set_xlim(0, max_alpha + 0.05)

    ax2.set_xlabel("α (target FDP)")
    ax2.set_ylabel("Retention Rate")
    ax2.set_title(f"Stage 2 {branch_label}: Retention by Disease")
    ax2.legend(fontsize=8, loc="lower right")
    ax2.set_xlim(0, max_alpha + 0.05)
    ax2.set_ylim(0, 1.05)

    plt.tight_layout()
    save_path = os.path.join(PLOT_DIR, f"stage2_branch_{branch}_by_disease.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Saved: {save_path}")


def plot_combined_comparison(results_path=None):
    """
    绘制实验④整合结果：Branch A vs B 的 retention 柱状图。
    """
    if results_path is None:
        results_path = os.path.join(RESULTS_DIR, "combined.json")

    if not os.path.exists(results_path):
        print(f"[SKIP] 未找到整合结果: {results_path}")
        return

    with open(results_path) as f:
        results = json.load(f)

    if not results:
        print("[SKIP] 整合结果为空")
        return

    # 按疾病绘制柱状图
    diseases = [r["disease_name"] for r in results]
    a_retention = [r.get("branch_a_retention", 0) for r in results]
    b_retention = [r.get("branch_b_retention", 0) for r in results]
    total_retention = [r.get("weighted_total_retention", 0) for r in results]

    x = np.arange(len(diseases))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width, a_retention, width, label="Branch A", color="tab:blue", alpha=0.8)
    bars2 = ax.bar(x, b_retention, width, label="Branch B", color="tab:orange", alpha=0.8)
    bars3 = ax.bar(x + width, total_retention, width, label="Combined", color="tab:green", alpha=0.8)

    ax.set_xlabel("Disease")
    ax.set_ylabel("Retention Rate")
    ax.set_title("Retention Comparison: Branch A vs Branch B vs Combined")
    ax.set_xticks(x)
    ax.set_xticklabels(diseases, rotation=45, ha="right", fontsize=9)
    ax.legend()
    ax.set_ylim(0, 1.1)

    # 标注是否 controlled
    for idx, r in enumerate(results):
        if r.get("both_controlled"):
            ax.annotate("✓", (x[idx] + width, total_retention[idx] + 0.02),
                        ha="center", fontsize=12, color="green")
        else:
            ax.annotate("✗", (x[idx] + width, total_retention[idx] + 0.02),
                        ha="center", fontsize=12, color="red")

    plt.tight_layout()
    save_path = os.path.join(PLOT_DIR, "combined_comparison.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Saved: {save_path}")


def plot_all():
    """绘制所有可用的实验结果图表"""
    print("=" * 60)
    print("绘制实验结果图表")
    print("=" * 60)
    print(f"输出目录: {PLOT_DIR}\n")

    plot_stage1_routing()
    plot_stage2_by_disease(branch="a")
    plot_stage2_by_disease(branch="b")
    plot_combined_comparison()

    print(f"\n[DONE] 所有图表已保存到 {PLOT_DIR}")


if __name__ == "__main__":
    plot_all()

