"""
LEC 校准实验主入口：统一命令行接口，按步骤执行全部实验流程。

使用方式:
    # 简化版: 仅用已有 phase_2 数据做 LEC 校准 (不需要 Router)
    python -m calibration.run_all --mode simplified

    # 完整版: Router推理 → 数据准备 → 四步实验
    python -m calibration.run_all --mode full

    # 单步执行
    python -m calibration.run_all --mode router       # 仅运行 Router 推理
    python -m calibration.run_all --mode stage1       # 仅运行实验①
    python -m calibration.run_all --mode stage2       # 仅运行实验②
    python -m calibration.run_all --mode stage3       # 仅运行实验③④
    python -m calibration.run_all --mode plot         # 仅绘图
"""

import argparse
import json
import os
import sys

from calibration.config import (
    RESULTS_DIR,
    ROUTER_PHASE1_PATH,
    LEC_DEFAULTS,
)


def run_simplified(num_splits=None, split_ratio=None):
    """
    简化版实验：用已有的 4 个 phase_2.json 直接做 LEC 校准。

    不依赖 Router，直接使用专病模型的 uncertainty 和 correctness。
    相当于只验证 LEC 框架在已有数据上的有效性。
    """
    from calibration.experiment_stage2 import run_stage2_branch_a

    print("\n" + "█" * 60)
    print("█  简化版 LEC 实验 (仅用已有 phase_2 数据)")
    print("█" * 60)

    results = run_stage2_branch_a(
        num_splits=num_splits,
        split_ratio=split_ratio,
        use_fused=False,
    )

    # 绘图
    from calibration.plot_results import plot_stage2_by_disease
    plot_stage2_by_disease(branch="a")

    print("\n" + "=" * 60)
    print("简化版实验完成!")
    print("=" * 60)
    _print_summary_simplified(results)

    return results


def run_full(num_splits=None, split_ratio=None, alpha_target=0.1, skip_router=False):
    """
    完整版四步实验流程。

    步骤:
    1. Router 推理 (生成 router_phase_1.json)
    2. 实验① Routing mFDR 控制
    3. 数据准备 (生成 Branch A/B JSON)
    4. 实验② Branch A Total mFDR
    5. 实验③ Branch B 校准
    6. 实验④ 整合
    7. 绘图
    """
    print("\n" + "█" * 60)
    print("█  完整版 LEC 两阶段两分支实验")
    print("█" * 60)

    # Step 1: Router 推理
    if not skip_router and not os.path.exists(ROUTER_PHASE1_PATH):
        print("\n[Step 1] Router 推理...")
        from calibration.router_inference import run_router_inference
        run_router_inference()
    elif os.path.exists(ROUTER_PHASE1_PATH):
        print(f"\n[Step 1] Router 结果已存在: {ROUTER_PHASE1_PATH}")
    else:
        print("\n[Step 1] [SKIP] Router 推理 (--skip_router)")

    # Step 2: 实验① Routing mFDR
    print("\n[Step 2] 实验① — Routing mFDR 控制")
    from calibration.experiment_stage1 import run_stage1_routing
    stage1_results = run_stage1_routing(
        num_splits=num_splits,
        split_ratio=split_ratio,
    )

    # 确定 Router 阈值 τ_R (取最小可行 α^R 对应的结果)
    tau_r = _determine_tau_r(stage1_results)
    if tau_r is None:
        print("[WARNING] 无法确定 τ_R，使用默认值 (中位数 uncertainty)")
        tau_r = _get_median_uncertainty()

    print(f"\n确定 Router 阈值 τ_R = {tau_r}")

    # Step 3: 数据准备
    print("\n[Step 3] 数据准备 — 生成 Branch A/B 数据")
    from calibration.data_prepare import prepare_all_branch_data
    prepare_all_branch_data(tau_r)

    # Step 4: 实验② Branch A
    print("\n[Step 4] 实验② — Branch A Total mFDR 控制")
    from calibration.experiment_stage2 import run_stage2_branch_a
    run_stage2_branch_a(
        num_splits=num_splits,
        split_ratio=split_ratio,
        use_fused=True,
    )

    # Step 5: 实验③ Branch B
    print("\n[Step 5] 实验③ — Branch B LEC 校准")
    from calibration.experiment_stage3 import run_stage3_reuse_test
    run_stage3_reuse_test(
        num_splits=num_splits,
        split_ratio=split_ratio,
    )

    # Step 6: 实验④ 整合
    print("\n[Step 6] 实验④ — 整合 Branch A + B")
    from calibration.experiment_stage3 import run_stage4_combined
    run_stage4_combined(alpha_target=alpha_target)

    # Step 7: 绘图
    print("\n[Step 7] 绘制实验结果图表")
    from calibration.plot_results import plot_all
    plot_all()

    print("\n" + "█" * 60)
    print("█  完整版实验流程完毕!")
    print("█" * 60)


def run_single_step(mode, num_splits=None, split_ratio=None, alpha_target=0.1, tau_r=None):
    """运行单个步骤"""
    if mode == "router":
        from calibration.router_inference import run_router_inference
        run_router_inference()

    elif mode == "stage1":
        from calibration.experiment_stage1 import run_stage1_routing
        run_stage1_routing(num_splits=num_splits, split_ratio=split_ratio)

    elif mode == "prepare":
        if tau_r is None:
            tau_r = _determine_tau_r_from_saved()
        if tau_r is None:
            print("[ERROR] 无法确定 τ_R，请先运行实验① 或指定 --tau_r")
            return
        from calibration.data_prepare import prepare_all_branch_data
        prepare_all_branch_data(tau_r)

    elif mode == "stage2":
        from calibration.experiment_stage2 import run_stage2_branch_a
        run_stage2_branch_a(num_splits=num_splits, split_ratio=split_ratio, use_fused=True)

    elif mode == "stage3":
        from calibration.experiment_stage3 import run_stage3_reuse_test, run_stage4_combined
        run_stage3_reuse_test(num_splits=num_splits, split_ratio=split_ratio)
        run_stage4_combined(alpha_target=alpha_target)

    elif mode == "plot":
        from calibration.plot_results import plot_all
        plot_all()

    else:
        print(f"[ERROR] Unknown mode: {mode}")


def _determine_tau_r(stage1_results):
    """
    从实验①结果中确定 Router 阈值 τ_R。

    策略：使用 multi-split 中的中位数阈值。
    由于 multi_split_experiment 返回的是统计量而非具体阈值，
    我们通过 Router phase_1 数据和最小可行 α^R 来反推阈值。

    具体做法：找到 mean_retention_rate，用它反推对应的 uncertainty 分位数。
    """
    from calibration.data_prepare import load_json

    if not stage1_results:
        return None

    # 找最小可行 α^R (mean_fdp 不为 None 的最小 α)
    feasible = [r for r in stage1_results if r.get("mean_fdp") is not None]
    if not feasible:
        return None

    # 选择 α^R = 0.1 或最小可行 α
    target_result = None
    for r in feasible:
        if abs(r["alpha"] - 0.1) < 1e-6:
            target_result = r
            break
    if target_result is None:
        target_result = feasible[0]  # 最小可行 α

    # 通过 retention rate 反推阈值
    retention_rate = target_result["mean_retention_rate"]

    # 加载 Router phase_1 数据，找到对应分位数的 uncertainty 值
    phase1 = load_json(ROUTER_PHASE1_PATH)
    if not phase1:
        return None

    uncertainties = sorted([v["uncertainty"] for v in phase1.values()])
    # retention_rate 的含义：有 retention_rate 比例的样本被接受 (u ≤ τ)
    # 即阈值 = 第 retention_rate 分位数的 uncertainty
    idx = min(int(len(uncertainties) * retention_rate), len(uncertainties) - 1)
    tau_r = uncertainties[idx]

    return tau_r


def _determine_tau_r_from_saved():
    """从已保存的实验①结果中确定 τ_R"""
    results_path = os.path.join(RESULTS_DIR, "stage1_routing.json")
    if not os.path.exists(results_path):
        return None
    with open(results_path) as f:
        results = json.load(f)
    return _determine_tau_r(results)


def _get_median_uncertainty():
    """获取 Router phase_1 中 uncertainty 的中位数作为默认 τ_R"""
    from calibration.data_prepare import load_json
    import numpy as np

    phase1 = load_json(ROUTER_PHASE1_PATH)
    if not phase1:
        return 1.0  # 安全默认值（接受所有样本）

    uncertainties = [v["uncertainty"] for v in phase1.values()]
    return float(np.median(uncertainties))


def _print_summary_simplified(results):
    """打印简化版实验摘要"""
    print("\n┌─────────────────────────────────────────────────────────┐")
    print("│  LEC 校准实验摘要 (简化版, 仅 u₂)                       │")
    print("├─────────────────────────────────────────────────────────┤")

    for disease_id, disease_results in sorted(results.items()):
        from calibration.config import DISEASE_NAMES
        name = DISEASE_NAMES.get(disease_id, f"D{disease_id}")

        # 找 α=0.1 的结果
        target = None
        for r in disease_results:
            if abs(r.get("alpha", -1) - 0.1) < 1e-6:
                target = r
                break

        if target and target.get("mean_fdp") is not None:
            print(f"│  D{disease_id} {name:<25s} "
                  f"FDP={target['mean_fdp']:.3f} "
                  f"ret={target['mean_retention_rate']:.3f} "
                  f"n={target['num_samples']}")
        else:
            print(f"│  D{disease_id} {name:<25s} (infeasible or no data)")

    print("└─────────────────────────────────────────────────────────┘")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LEC 两阶段两分支 mFDR 控制实验系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
运行模式:
  simplified  - 简化版: 用已有 phase_2.json 做 LEC 校准 (无需Router)
  full        - 完整版: Router推理 → 四步实验
  router      - 仅运行 Router 推理
  stage1      - 仅运行实验① (Routing mFDR)
  prepare     - 仅运行数据准备 (Branch A/B)
  stage2      - 仅运行实验② (Branch A)
  stage3      - 仅运行实验③④ (Branch B + 整合)
  plot        - 仅绘图
        """,
    )
    parser.add_argument("--mode", type=str, default="simplified",
                        choices=["simplified", "full", "router", "stage1",
                                 "prepare", "stage2", "stage3", "plot"],
                        help="运行模式")
    parser.add_argument("--num_splits", type=int, default=None,
                        help="Multi-split 次数 (默认500)")
    parser.add_argument("--split_ratio", type=float, default=None,
                        help="Calibration 占比 (默认0.5)")
    parser.add_argument("--alpha_target", type=float, default=0.1,
                        help="实验④目标总体 mFDR (默认0.1)")
    parser.add_argument("--skip_router", action="store_true",
                        help="完整模式下跳过 Router 推理")
    parser.add_argument("--tau_r", type=float, default=None,
                        help="手动指定 Router 阈值 (用于 prepare 模式)")
    args = parser.parse_args()

    # 将项目根目录加入 sys.path (确保 calibration 包可导入)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    if args.mode == "simplified":
        run_simplified(num_splits=args.num_splits, split_ratio=args.split_ratio)
    elif args.mode == "full":
        run_full(
            num_splits=args.num_splits,
            split_ratio=args.split_ratio,
            alpha_target=args.alpha_target,
            skip_router=args.skip_router,
        )
    else:
        run_single_step(
            mode=args.mode,
            num_splits=args.num_splits,
            split_ratio=args.split_ratio,
            alpha_target=args.alpha_target,
            tau_r=args.tau_r,
        )

