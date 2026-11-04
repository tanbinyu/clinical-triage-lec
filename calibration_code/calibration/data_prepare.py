"""
数据准备模块：从 phase_1 (Router) + phase_2 (专病模型) 生成 LEC 校准输入数据。

功能：
- 生成 Branch A fused score JSON (按疾病分组)
- 生成 Branch B JSON (按疾病分组)
- 合并同一疾病下多个数据集的 phase_2 数据
"""

import json
from pathlib import Path

from calibration.config import (
    DISEASE_TO_DATASETS,
    DISEASE_NAMES,
    DATASET_TO_DISEASE,
    LAMBDA1,
    LAMBDA2_PER_DISEASE,
    JSON_DIR,
    ROUTER_PHASE1_PATH,
    get_phase2_path,
    get_branch_a_path,
    get_branch_b_path,
)


def load_json(path):
    """安全加载 JSON 文件，不存在则返回空字典"""
    p = Path(path)
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f)


def save_json(data, path):
    """保存字典为 JSON 文件"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_all_phase2():
    """
    加载所有已有的 phase_2 JSON 文件。

    返回:
        dict: {dataset_name: {sample_id: {correctness, uncertainty}}}
    """
    phase2_all = {}
    for disease_id, datasets in DISEASE_TO_DATASETS.items():
        for dataset_name in datasets:
            path = get_phase2_path(dataset_name)
            data = load_json(path)
            if data:
                phase2_all[dataset_name] = data
                print(f"  Loaded {dataset_name}_phase_2.json: {len(data)} samples")
    return phase2_all


def merge_phase2_by_disease(phase2_all):
    """
    将同一疾病下多个数据集的 phase_2 数据合并。

    参数:
        phase2_all: dict, {dataset_name: {sample_id: {correctness, uncertainty}}}

    返回:
        dict: {disease_id: {sample_id: {correctness, uncertainty}}}
    """
    merged = {}
    for dataset_name, samples in phase2_all.items():
        disease_id = DATASET_TO_DISEASE.get(dataset_name)
        if disease_id is None:
            continue
        if disease_id not in merged:
            merged[disease_id] = {}
        merged[disease_id].update(samples)
    return merged


def compute_fused_score(u1, u2, disease_id):
    """
    计算 Branch A 的 fused score。

    S_A(x) = λ₁ * u₁(x) + λ₂(ẑ) * u₂(x, ẑ)

    参数:
        u1: float, Router uncertainty (predictive entropy)
        u2: float, 专病模型 uncertainty (predictive entropy)
        disease_id: int, 预测的疾病编号 (用于选择 λ₂)

    返回:
        float: fused score
    """
    lambda2 = LAMBDA2_PER_DISEASE.get(disease_id, 0.5)
    return LAMBDA1 * u1 + lambda2 * u2


def generate_branch_a_data(phase1, phase2_all, tau_r):
    """
    生成 Branch A 的 fused score 数据（按疾病分组）。

    Branch A 样本条件: u₁(x) ≤ τ_R (Router 自信，AI分流)
    Fused score: S_A = λ₁*u₁ + λ₂(ẑ)*u₂
    Correctness: Router分对 AND 诊断正确

    参数:
        phase1: dict, Router 推理结果 {sample_id: {correctness, uncertainty, predicted_disease, ...}}
        phase2_all: dict, {dataset_name: {sample_id: {correctness, uncertainty}}}
        tau_r: float, Router 阈值 (由实验①确定)

    返回:
        dict: {disease_id: {sample_id: {correctness, uncertainty}}}
    """
    branch_a_data = {}

    for sample_id, p1 in phase1.items():
        # 只取 Router 接受的样本 (进入 Branch A)
        if p1["uncertainty"] > tau_r:
            continue

        predicted_k = p1["predicted_disease"]
        source_dataset = p1["source_dataset"]

        # 在对应数据集的 phase_2 中查找该样本
        if source_dataset not in phase2_all:
            continue
        if sample_id not in phase2_all[source_dataset]:
            continue

        p2 = phase2_all[source_dataset][sample_id]

        # 计算 fused score
        u1 = p1["uncertainty"]
        u2 = p2["uncertainty"]
        fused_uncertainty = compute_fused_score(u1, u2, predicted_k)

        # Branch A correctness: routing 正确 AND 诊断正确
        fused_correctness = int(p1["correctness"] == 1 and p2["correctness"] == 1)

        branch_a_data.setdefault(predicted_k, {})[sample_id] = {
            "correctness": fused_correctness,
            "uncertainty": round(fused_uncertainty, 6),
        }

    return branch_a_data


def generate_branch_b_data(phase1, phase2_all, tau_r):
    """
    生成 Branch B 的数据（按疾病分组）。

    Branch B 样本条件: u₁(x) > τ_R (Router 不确定，人工分流)
    人工分流到正确疾病 → 用 true_disease 查找 phase_2
    Uncertainty: 仅用专病模型 entropy u₂(x, z*)
    Correctness: 仅看诊断正确性

    参数:
        phase1: dict, Router 推理结果
        phase2_all: dict, {dataset_name: {sample_id: {correctness, uncertainty}}}
        tau_r: float, Router 阈值

    返回:
        dict: {disease_id: {sample_id: {correctness, uncertainty}}}
    """
    branch_b_data = {}

    for sample_id, p1 in phase1.items():
        # 只取 Router 拒绝的样本 (进入 Branch B)
        if p1["uncertainty"] <= tau_r:
            continue

        # Branch B: 人工分流到正确疾病
        true_k = p1["true_disease"]
        source_dataset = p1["source_dataset"]

        if source_dataset not in phase2_all:
            continue
        if sample_id not in phase2_all[source_dataset]:
            continue

        p2 = phase2_all[source_dataset][sample_id]

        # Branch B: 仅看诊断 uncertainty 和 correctness
        branch_b_data.setdefault(true_k, {})[sample_id] = {
            "correctness": p2["correctness"],
            "uncertainty": p2["uncertainty"],
        }

    return branch_b_data


def prepare_all_branch_data(tau_r):
    """
    一键生成 Branch A 和 Branch B 的所有数据文件。

    参数:
        tau_r: float, Router 阈值 (由实验①确定)
    """
    print("=" * 60)
    print("数据准备: 生成 Branch A/B 校准数据")
    print("=" * 60)

    # 加载 Router phase_1
    phase1 = load_json(ROUTER_PHASE1_PATH)
    if not phase1:
        print(f"[ERROR] Router phase_1 文件不存在: {ROUTER_PHASE1_PATH}")
        return

    print(f"\nRouter phase_1: {len(phase1)} samples loaded")
    print(f"Router 阈值 τ_R = {tau_r}")

    # 加载所有 phase_2
    print("\n加载 phase_2 数据:")
    phase2_all = load_all_phase2()

    # 生成 Branch A
    print(f"\n--- Branch A (Router 接受, u₁ ≤ {tau_r}) ---")
    branch_a_data = generate_branch_a_data(phase1, phase2_all, tau_r)
    for disease_id, data in sorted(branch_a_data.items()):
        path = get_branch_a_path(disease_id)
        save_json(data, path)
        error_rate = 1 - sum(v["correctness"] for v in data.values()) / len(data) if data else 0
        print(f"  Disease {disease_id} ({DISEASE_NAMES[disease_id]}): "
              f"{len(data)} samples, error_rate={error_rate:.3f}")

    # 生成 Branch B
    print(f"\n--- Branch B (Router 拒绝, u₁ > {tau_r}) ---")
    branch_b_data = generate_branch_b_data(phase1, phase2_all, tau_r)
    for disease_id, data in sorted(branch_b_data.items()):
        path = get_branch_b_path(disease_id)
        save_json(data, path)
        error_rate = 1 - sum(v["correctness"] for v in data.values()) / len(data) if data else 0
        print(f"  Disease {disease_id} ({DISEASE_NAMES[disease_id]}): "
              f"{len(data)} samples, error_rate={error_rate:.3f}")

    print("\n[DONE] Branch A/B 数据准备完成")


def prepare_simplified_branch_b():
    """
    简化版数据准备：不依赖 Router，直接用已有 phase_2 数据按疾病合并。
    用于在没有 Router 结果时验证 LEC 框架。

    生成的数据即 Branch B 的输入格式 (uncertainty=u₂, correctness=诊断正确性)。

    返回:
        dict: {disease_id: {sample_id: {correctness, uncertainty}}}
    """
    print("=" * 60)
    print("简化版数据准备: 仅用 phase_2 数据 (无 Router)")
    print("=" * 60)

    print("\n加载 phase_2 数据:")
    phase2_all = load_all_phase2()
    merged = merge_phase2_by_disease(phase2_all)

    print(f"\n按疾病合并结果:")
    for disease_id, data in sorted(merged.items()):
        num_correct = sum(v["correctness"] for v in data.values())
        error_rate = 1 - num_correct / len(data) if data else 0
        print(f"  Disease {disease_id} ({DISEASE_NAMES[disease_id]}): "
              f"{len(data)} samples, error_rate={error_rate:.3f}")

    return merged


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="数据准备: 生成 Branch A/B 校准数据")
    parser.add_argument("--tau_r", type=float, default=None,
                        help="Router 阈值。不指定时仅运行简化版(无Router)")
    args = parser.parse_args()

    if args.tau_r is not None:
        prepare_all_branch_data(args.tau_r)
    else:
        merged = prepare_simplified_branch_b()
        # 保存简化版数据
        for disease_id, data in merged.items():
            path = get_branch_b_path(disease_id)
            save_json(data, path)
        print("\n[DONE] 简化版 Branch B 数据已保存")
