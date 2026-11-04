"""
实验①：控制 Stage One Routing mFDR。

对 Router 结果做 multi-split LEC 校准，扫描多个 α^R，
验证 Routing mFDR ≤ α^R 并记录 retention（被Router接受进入Branch A的比例）。

输入: res/json/router_phase_1.json
输出: res/json/lec_results/stage1_routing.json
"""

import json
import os

from calibration.config import (
    LEC_DEFAULTS,
    ROUTER_PHASE1_PATH,
    RESULTS_DIR,
)
from calibration.lec import multi_split_experiment


def run_stage1_routing(
    alpha_list=None,
    split_ratio=None,
    num_splits=None,
    router_json_path=None,
):
    """
    实验①：扫描多个 α^R 对 Router 结果做 LEC 校准。

    参数:
        alpha_list: list[float], 要扫描的 α^R 值
        split_ratio: float, calibration 占比
        num_splits: int, multi-split 重复次数
        router_json_path: str, Router phase_1 JSON 路径

    返回:
        list[dict]: 每个 α 下的实验结果
    """
    if alpha_list is None:
        alpha_list = LEC_DEFAULTS["alpha_scan_routing"]
    if split_ratio is None:
        split_ratio = LEC_DEFAULTS["split_ratio"]
    if num_splits is None:
        num_splits = LEC_DEFAULTS["num_splits"]
    if router_json_path is None:
        router_json_path = ROUTER_PHASE1_PATH

    if not os.path.exists(router_json_path):
        print(f"[ERROR] Router phase_1 文件不存在: {router_json_path}")
        print("请先运行 Router 推理: python -m calibration.router_inference")
        return []

    print("=" * 60)
    print("实验① — Stage One Routing mFDR 控制")
    print("=" * 60)
    print(f"数据: {router_json_path}")
    print(f"α^R 扫描: {alpha_list}")
    print(f"Split ratio: {split_ratio}, Num splits: {num_splits}")
    print()

    all_results = []

    for alpha_r in alpha_list:
        print(f"--- α^R = {alpha_r} ---")
        result = multi_split_experiment(
            json_path=router_json_path,
            alpha=alpha_r,
            split_ratio=split_ratio,
            num_splits=num_splits,
        )
        result["stage"] = "routing"
        result["disease_k"] = None

        # 打印结果
        if result["mean_fdp"] is not None:
            print(f"  FDP: {result['mean_fdp']:.4f} ± {result['std_fdp']:.4f} "
                  f"(target ≤ {alpha_r})")
            print(f"  Retention: {result['mean_retention_rate']:.4f} "
                  f"({result['mean_retained_count']:.1f} samples)")
            controlled = result["mean_fdp"] <= alpha_r
            print(f"  Controlled: {'✓' if controlled else '✗'}")
        else:
            print(f"  [INFEASIBLE] All {result['infeasible_count']} splits infeasible")

        all_results.append(result)
        print()

    # 保存结果
    output_path = os.path.join(RESULTS_DIR, "stage1_routing.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"[SAVED] {output_path}")

    # 找到最小可行 α^R
    feasible = [r for r in all_results if r["mean_fdp"] is not None]
    if feasible:
        min_feasible = min(feasible, key=lambda x: x["alpha"])
        print(f"\n最小可行 α^R = {min_feasible['alpha']}, "
              f"对应 retention = {min_feasible['mean_retention_rate']:.4f}")
    else:
        print("\n[WARNING] 所有 α^R 均 infeasible")

    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="实验① Routing mFDR 控制")
    parser.add_argument("--num_splits", type=int, default=None)
    parser.add_argument("--split_ratio", type=float, default=None)
    parser.add_argument("--router_json", type=str, default=None)
    args = parser.parse_args()

    run_stage1_routing(
        num_splits=args.num_splits,
        split_ratio=args.split_ratio,
        router_json_path=args.router_json,
    )

