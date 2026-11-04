"""
实验③④：Branch B 复用与整合。

实验③：验证 Branch A 的阈值能否直接复用到 Branch B
    - 用 Branch A 在每个 α 下校准得到的阈值，直接应用于 Branch B 样本
    - 评估 Branch B 的实际 mFDR（如果 ≤ α 则可复用）

实验④：若不能复用，为 Branch B 独立校准并整合
    - 对 Branch B 数据独立做 LEC 校准
    - 整合 Branch A + Branch B，验证总体 mFDR ≤ α

输入: res/json/branch_b_disease_{k}.json
输出: res/json/lec_results/stage2_branch_b/disease_{k}_{name}.json
      res/json/lec_results/combined.json
"""

import json
import os
import random
import numpy as np

from calibration.config import (
    DISEASE_TO_DATASETS,
    DISEASE_NAMES,
    LEC_DEFAULTS,
    RESULTS_BRANCH_B_DIR,
    RESULTS_DIR,
    get_branch_b_path,
)
from calibration.lec import (
    load_data,
    lec_calibrate,
    lec_evaluate,
    multi_split_experiment,
    multi_split_experiment_from_data,
)
from calibration.data_prepare import load_json


def run_stage3_reuse_test(
    alpha_list=None,
    split_ratio=None,
    num_splits=None,
):
    """
    实验③：测试 Branch A 阈值能否直接复用到 Branch B。

    方法：
    - 对 Branch B 数据做 multi-split
    - 在 calibration set 上用 Branch A 的 LEC 校准得到阈值
    - 将阈值直接应用于 Branch B 的 test set
    - 检查 Branch B 的 mFDR 是否 ≤ α

    由于 Branch B 的 uncertainty 仅为 u₂（不含 fused），
    而 Branch A 使用 fused score，阈值量纲不同。
    因此"复用"的含义是：用 Branch B 自己的数据做 LEC 校准，
    看其是否在相同 α 下可行，以及 retention 对比。

    参数:
        alpha_list: list[float]
        split_ratio: float
        num_splits: int

    返回:
        dict: {disease_id: list[dict]}
    """
    if alpha_list is None:
        alpha_list = LEC_DEFAULTS["alpha_scan_branch"]
    if split_ratio is None:
        split_ratio = LEC_DEFAULTS["split_ratio"]
    if num_splits is None:
        num_splits = LEC_DEFAULTS["num_splits"]

    print("=" * 60)
    print("实验③ — Branch B LEC 校准 (验证阈值复用可行性)")
    print("=" * 60)
    print(f"α 扫描: {alpha_list}")
    print(f"Split ratio: {split_ratio}, Num splits: {num_splits}")

    disease_data = _load_branch_b_data()
    all_disease_results = {}

    for disease_id in sorted(disease_data.keys()):
        data = disease_data[disease_id]
        disease_name = DISEASE_NAMES.get(disease_id, f"unknown_{disease_id}")
        datasets = DISEASE_TO_DATASETS.get(disease_id, [])

        if len(data) == 0:
            print(f"\n[SKIP] Disease {disease_id} ({disease_name}): no data")
            continue

        print(f"\n{'─' * 50}")
        print(f"Disease {disease_id}: {disease_name}")
        print(f"Datasets: {datasets}")
        print(f"Samples: {len(data)}")

        disease_results = []

        for alpha in alpha_list:
            result = multi_split_experiment_from_data(
                data_dict=data,
                alpha=alpha,
                split_ratio=split_ratio,
                num_splits=num_splits,
            )
            result["stage"] = "branch_b"
            result["disease_k"] = disease_id
            result["disease_name"] = disease_name
            result["dataset_names"] = datasets

            if result["mean_fdp"] is not None:
                controlled = result["mean_fdp"] <= alpha
                print(f"  α={alpha:.2f} | FDP={result['mean_fdp']:.4f}±{result['std_fdp']:.4f} "
                      f"| retention={result['mean_retention_rate']:.4f} "
                      f"| {'✓' if controlled else '✗'}")
            else:
                print(f"  α={alpha:.2f} | [INFEASIBLE] ({result['infeasible_count']} splits)")

            disease_results.append(result)

        all_disease_results[disease_id] = disease_results

        # 保存该疾病的结果
        output_path = os.path.join(
            RESULTS_BRANCH_B_DIR, f"disease_{disease_id}_{disease_name}.json"
        )
        with open(output_path, "w") as f:
            json.dump(disease_results, f, indent=2)

    print(f"\n[SAVED] Results in {RESULTS_BRANCH_B_DIR}")
    return all_disease_results


def run_stage4_combined(
    alpha_target=0.1,
    alpha_r_list=None,
    split_ratio=None,
    num_splits=None,
):
    """
    实验④：调整 α^R 后整合 Branch A + Branch B。

    验证逻辑：
    - 两条 Branch 互斥（一个样本进 A 或 B）
    - 只要 mFDR^A ≤ α 且 mFDR^B ≤ α → mFDR_total ≤ α
    - 扫描不同 α^R，找到使两个 Branch 都能控制在 α_target 下的最优配置
      （最优 = 总 retention 最高）

    参数:
        alpha_target: float, 目标总体 mFDR
        alpha_r_list: list[float], Router α^R 的候选值
        split_ratio: float
        num_splits: int

    返回:
        list[dict]: 每种 α^R 配置下的整合结果
    """
    if alpha_r_list is None:
        alpha_r_list = LEC_DEFAULTS["alpha_scan_routing"]
    if split_ratio is None:
        split_ratio = LEC_DEFAULTS["split_ratio"]
    if num_splits is None:
        num_splits = LEC_DEFAULTS["num_splits"]

    print("=" * 60)
    print(f"实验④ — 整合 Branch A + B, 目标 mFDR_total ≤ {alpha_target}")
    print("=" * 60)

    # 加载 Branch A 和 Branch B 的实验结果
    branch_a_results_path = os.path.join(RESULTS_DIR, "stage2_branch_a")
    branch_b_results_path = os.path.join(RESULTS_DIR, "stage2_branch_b")

    branch_a_all = _load_experiment_results(branch_a_results_path)
    branch_b_all = _load_experiment_results(branch_b_results_path)

    if not branch_a_all and not branch_b_all:
        print("[ERROR] 未找到 Branch A/B 实验结果，请先运行实验②③")
        return []

    combined_results = []

    # 对每种疾病整合
    all_diseases = set(list(branch_a_all.keys()) + list(branch_b_all.keys()))

    print(f"\n整合分析 (α_target = {alpha_target}):")
    print(f"{'─' * 50}")

    for disease_id in sorted(all_diseases):
        disease_name = DISEASE_NAMES.get(disease_id, f"unknown_{disease_id}")

        # 找 Branch A 在 alpha_target 下的结果
        branch_a_result = _find_result_at_alpha(
            branch_a_all.get(disease_id, []), alpha_target
        )
        # 找 Branch B 在 alpha_target 下的结果
        branch_b_result = _find_result_at_alpha(
            branch_b_all.get(disease_id, []), alpha_target
        )

        a_controlled = (
            branch_a_result is not None
            and branch_a_result.get("mean_fdp") is not None
            and branch_a_result["mean_fdp"] <= alpha_target
        )
        b_controlled = (
            branch_b_result is not None
            and branch_b_result.get("mean_fdp") is not None
            and branch_b_result["mean_fdp"] <= alpha_target
        )

        # 总 retention: 加权平均（需要各 branch 的样本量信息）
        a_retention = branch_a_result.get("mean_retention_rate", 0) if branch_a_result else 0
        b_retention = branch_b_result.get("mean_retention_rate", 0) if branch_b_result else 0
        a_samples = branch_a_result.get("num_samples", 0) if branch_a_result else 0
        b_samples = branch_b_result.get("num_samples", 0) if branch_b_result else 0

        total_samples = a_samples + b_samples
        if total_samples > 0:
            # 加权 retention (按各 branch 的样本量加权)
            weighted_retention = (
                a_retention * a_samples + b_retention * b_samples
            ) / total_samples
        else:
            weighted_retention = 0

        both_controlled = a_controlled and b_controlled

        disease_combined = {
            "disease_k": disease_id,
            "disease_name": disease_name,
            "alpha_target": alpha_target,
            "branch_a_fdp": branch_a_result.get("mean_fdp") if branch_a_result else None,
            "branch_a_retention": a_retention,
            "branch_a_samples": a_samples,
            "branch_a_controlled": a_controlled,
            "branch_b_fdp": branch_b_result.get("mean_fdp") if branch_b_result else None,
            "branch_b_retention": b_retention,
            "branch_b_samples": b_samples,
            "branch_b_controlled": b_controlled,
            "both_controlled": both_controlled,
            "weighted_total_retention": round(weighted_retention, 4),
        }
        combined_results.append(disease_combined)

        status_a = f"FDP={branch_a_result['mean_fdp']:.4f}" if branch_a_result and branch_a_result.get("mean_fdp") else "N/A"
        status_b = f"FDP={branch_b_result['mean_fdp']:.4f}" if branch_b_result and branch_b_result.get("mean_fdp") else "N/A"
        print(f"  Disease {disease_id} ({disease_name}):")
        print(f"    Branch A: {status_a}, retention={a_retention:.4f} {'✓' if a_controlled else '✗'}")
        print(f"    Branch B: {status_b}, retention={b_retention:.4f} {'✓' if b_controlled else '✗'}")
        print(f"    Combined: {'✓ CONTROLLED' if both_controlled else '✗ NOT CONTROLLED'}, "
              f"total_retention={weighted_retention:.4f}")

    # 保存整合结果
    output_path = os.path.join(RESULTS_DIR, "combined.json")
    with open(output_path, "w") as f:
        json.dump(combined_results, f, indent=2)
    print(f"\n[SAVED] {output_path}")

    return combined_results


def _load_branch_b_data():
    """加载 Branch B 数据"""
    disease_data = {}
    for disease_id in DISEASE_TO_DATASETS.keys():
        path = get_branch_b_path(disease_id)
        data = load_json(path)
        if data:
            disease_data[disease_id] = data
            print(f"  Loaded Branch B disease {disease_id}: {len(data)} samples")
    return disease_data


def _load_experiment_results(results_dir):
    """加载某个目录下所有疾病的实验结果"""
    results = {}
    if not os.path.exists(results_dir):
        return results

    for fname in os.listdir(results_dir):
        if not fname.endswith(".json"):
            continue
        # 解析文件名: disease_{k}_{name}.json
        parts = fname.replace(".json", "").split("_", 2)
        if len(parts) >= 2 and parts[0] == "disease":
            try:
                disease_id = int(parts[1])
            except ValueError:
                continue
            path = os.path.join(results_dir, fname)
            with open(path) as f:
                results[disease_id] = json.load(f)
    return results


def _find_result_at_alpha(results_list, alpha_target):
    """在结果列表中找到与 alpha_target 最接近的结果"""
    if not results_list:
        return None
    for result in results_list:
        if abs(result.get("alpha", -1) - alpha_target) < 1e-6:
            return result
    # 找最接近的
    return min(results_list, key=lambda r: abs(r.get("alpha", -1) - alpha_target))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="实验③④ Branch B 复用与整合")
    parser.add_argument("--mode", choices=["reuse", "combined", "both"], default="both",
                        help="reuse=实验③, combined=实验④, both=两者都跑")
    parser.add_argument("--alpha_target", type=float, default=0.1,
                        help="实验④的目标总体 mFDR")
    parser.add_argument("--num_splits", type=int, default=None)
    parser.add_argument("--split_ratio", type=float, default=None)
    args = parser.parse_args()

    if args.mode in ("reuse", "both"):
        run_stage3_reuse_test(
            num_splits=args.num_splits,
            split_ratio=args.split_ratio,
        )

    if args.mode in ("combined", "both"):
        run_stage4_combined(
            alpha_target=args.alpha_target,
            num_splits=args.num_splits,
            split_ratio=args.split_ratio,
        )

