"""
实验②：控制 Branch A Total mFDR。

对每种疾病独立做 LEC 校准，使用 fused score S_A = λ₁*u₁ + λ₂(ẑ)*u₂，
correctness = Router分对 AND 诊断正确。

当有 Router 结果时使用 Branch A fused 数据；
无 Router 时可直接使用已有 phase_2 数据运行简化版（仅用 u₂）。

输入: res/json/branch_a_disease_{k}.json 或 phase_2 数据
输出: res/json/lec_results/stage2_branch_a/disease_{k}_{name}.json
"""

import json
import os

from calibration.config import (
    DISEASE_TO_DATASETS,
    DISEASE_NAMES,
    LEC_DEFAULTS,
    RESULTS_BRANCH_A_DIR,
    get_branch_a_path,
    get_phase2_path,
    JSON_DIR,
)
from calibration.lec import multi_split_experiment, multi_split_experiment_from_data
from calibration.data_prepare import load_json, merge_phase2_by_disease, load_all_phase2


def run_stage2_branch_a(
    alpha_list=None,
    split_ratio=None,
    num_splits=None,
    use_fused=True,
):
    """
    实验②：对每种疾病做 Branch A LEC 校准。

    参数:
        alpha_list: list[float], 要扫描的 α 值
        split_ratio: float, calibration 占比
        num_splits: int, multi-split 重复次数
        use_fused: bool, True使用fused score(需Router), False使用简化版(仅u₂)

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
    if use_fused:
        print("实验② — Branch A Total mFDR 控制 (Fused Score)")
    else:
        print("实验② — 简化版: 仅用 u₂ 做 LEC 校准 (无 Router)")
    print("=" * 60)
    print(f"α 扫描: {alpha_list}")
    print(f"Split ratio: {split_ratio}, Num splits: {num_splits}")

    # 加载数据
    if use_fused:
        disease_data = _load_branch_a_data()
    else:
        disease_data = _load_simplified_data()

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
            result["stage"] = "branch_a" if use_fused else "branch_a_simplified"
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
            RESULTS_BRANCH_A_DIR, f"disease_{disease_id}_{disease_name}.json"
        )
        with open(output_path, "w") as f:
            json.dump(disease_results, f, indent=2)

    print(f"\n[SAVED] Results in {RESULTS_BRANCH_A_DIR}")
    return all_disease_results


def _load_branch_a_data():
    """加载 Branch A fused score 数据（需要先运行 data_prepare）"""
    disease_data = {}
    for disease_id in DISEASE_TO_DATASETS.keys():
        path = get_branch_a_path(disease_id)
        data = load_json(path)
        if data:
            disease_data[disease_id] = data
            print(f"  Loaded Branch A disease {disease_id}: {len(data)} samples")
    return disease_data


def _load_simplified_data():
    """加载简化版数据：直接合并已有 phase_2 文件"""
    print("\n加载 phase_2 数据:")
    phase2_all = load_all_phase2()
    merged = merge_phase2_by_disease(phase2_all)
    return merged


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="实验② Branch A mFDR 控制")
    parser.add_argument("--num_splits", type=int, default=None)
    parser.add_argument("--split_ratio", type=float, default=None)
    parser.add_argument("--simplified", action="store_true",
                        help="运行简化版(仅用u₂，不需要Router)")
    args = parser.parse_args()

    run_stage2_branch_a(
        num_splits=args.num_splits,
        split_ratio=args.split_ratio,
        use_fused=not args.simplified,
    )

