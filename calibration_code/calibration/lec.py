"""
LEC (Learn then Test) 校准核心模块。

实现：
- load_data: 加载统一格式 {sample_id: {correctness, uncertainty}} JSON
- lec_calibrate: 在 calibration set 上二分搜索满足 sum(loss)≤-1 的最大阈值
- lec_evaluate: 在 test set 上评估实际 mFDR 和 retention
- multi_split_experiment: 完整的 multi-split LEC 实验流程
"""

import json
import random
import numpy as np
from pathlib import Path


def load_data(json_path):
    """
    加载统一格式的 {sample_id: {correctness, uncertainty}} JSON 文件。

    参数:
        json_path: str 或 Path, JSON 文件路径

    返回:
        data: list of dict, 每个含 'id' 字段
        id_to_uncertainty: dict, {sample_id: float}
        id_to_correctness: dict, {sample_id: int(0或1)}
    """
    with open(json_path) as f:
        raw = json.load(f)

    data = [{"id": k} for k in raw.keys()]
    id_to_uncertainty = {k: v["uncertainty"] for k, v in raw.items()}
    id_to_correctness = {k: int(v["correctness"]) for k, v in raw.items()}

    return data, id_to_uncertainty, id_to_correctness


def lec_calibrate(calib_set, id_to_uncertainty, id_to_correctness, alpha):
    """
    LEC 校准：在 calibration set 上找满足 sum(loss)≤-1 的最大阈值。

    LEC 准则推导:
        l_i(τ) = 1{u_i≤τ 且 correctness_i=0} - α * 1{u_i≤τ}
        要求: sum(l_i(τ)) ≤ -1

    二分搜索策略：在所有候选阈值中找满足准则的最大值（最大化 retention）。

    参数:
        calib_set: list of dict, 每个含 'id' 字段
        id_to_uncertainty: dict, {sample_id: uncertainty_score}
        id_to_correctness: dict, {sample_id: 0 或 1}
        alpha: float, 目标 mFDR 上界

    返回:
        t_hat: float 或 None (None 表示该风险水平不可达/infeasible)
    """
    # 获取校准集中所有唯一不确定性值并排序（作为候选阈值）
    sorted_uncertainties = sorted(set(
        id_to_uncertainty[d["id"]] for d in calib_set
    ))

    if not sorted_uncertainties:
        return None

    # 二分搜索最大可行阈值
    low = 0
    high = len(sorted_uncertainties) - 1
    t_hat = None

    while low <= high:
        mid = (low + high) // 2
        t_candidate = sorted_uncertainties[mid]

        # 计算每个校准样本的 loss
        total_loss = 0.0
        for calib_data in calib_set:
            sample_id = calib_data["id"]
            uncertainty = id_to_uncertainty[sample_id]
            correctness = id_to_correctness[sample_id]

            # l_i(τ) = 1{error ∩ selection} - α * 1{selection}
            if uncertainty <= t_candidate:
                total_loss += (1 - correctness) - alpha

        # LEC 准则: sum(loss) ≤ -1
        if total_loss <= -1:
            t_hat = t_candidate
            low = mid + 1  # 尝试更大阈值（接受更多样本）
        else:
            high = mid - 1

    return t_hat


def lec_evaluate(test_set, id_to_uncertainty, id_to_correctness, t_hat):
    """
    在 test set 上评估给定阈值的实际 mFDR 和 retention。

    参数:
        test_set: list of dict, 每个含 'id' 字段
        id_to_uncertainty: dict
        id_to_correctness: dict
        t_hat: float, 校准得到的阈值

    返回:
        fdr: float, 实际 mFDR
        selected_num: int, 被接受的样本数
        error_num: int, 错误且被接受的样本数
    """
    selected_num = 0
    error_num = 0

    for test_data in test_set:
        sample_id = test_data["id"]
        uncertainty = id_to_uncertainty[sample_id]
        correctness = id_to_correctness[sample_id]

        if uncertainty <= t_hat:
            selected_num += 1
            if correctness == 0:
                error_num += 1

    if selected_num == 0:
        fdr = 0.0
    else:
        fdr = error_num / selected_num

    return fdr, selected_num, error_num


def multi_split_experiment(json_path, alpha, split_ratio=0.5, num_splits=500):
    """
    对一个 JSON 数据文件执行完整的 multi-split LEC 实验。

    流程 (重复 num_splits 次):
        1. random.seed(epoch) 确保可复现
        2. 随机划分 calibration/test set (比例 split_ratio)
        3. 在 calibration set 上 LEC 校准阈值
        4. 在 test set 上评估 mFDR 和 retention

    参数:
        json_path: str, 数据文件路径 ({id: {correctness, uncertainty}})
        alpha: float, 目标 mFDR 上界
        split_ratio: float, calibration 占比
        num_splits: int, 重复次数

    返回:
        dict: 包含 mean_fdp, std_fdp, mean_retention, infeasible_count 等
    """
    data, id_to_uncertainty, id_to_correctness = load_data(json_path)

    num_samples = len(data)
    num_calib = int(num_samples * split_ratio)
    base_error_rate = 1 - sum(id_to_correctness.values()) / num_samples

    fdr_list = []
    retention_list = []
    retained_count_list = []
    infeasible_count = 0

    for epoch in range(num_splits):
        random.seed(epoch)  # 可复现
        shuffled = data.copy()
        random.shuffle(shuffled)
        calib_set = shuffled[:num_calib]
        test_set = shuffled[num_calib:]

        # 校准
        t_hat = lec_calibrate(calib_set, id_to_uncertainty, id_to_correctness, alpha)

        if t_hat is None:
            infeasible_count += 1
            continue

        # 评估
        fdr, selected_num, error_num = lec_evaluate(
            test_set, id_to_uncertainty, id_to_correctness, t_hat
        )

        fdr_list.append(fdr)
        if len(test_set) > 0:
            retention_list.append(selected_num / len(test_set))
        retained_count_list.append(selected_num)

    results = {
        "alpha": alpha,
        "num_samples": num_samples,
        "base_error_rate": round(base_error_rate, 4),
        "num_splits": num_splits,
        "infeasible_count": infeasible_count,
        "num_valid_splits": len(fdr_list),
    }

    if fdr_list:
        results["mean_fdp"] = round(float(np.mean(fdr_list)), 4)
        results["std_fdp"] = round(float(np.std(fdr_list)), 4)
        results["mean_retention_rate"] = round(float(np.mean(retention_list)), 4)
        results["mean_retained_count"] = round(float(np.mean(retained_count_list)), 2)
    else:
        results["mean_fdp"] = None
        results["std_fdp"] = None
        results["mean_retention_rate"] = None
        results["mean_retained_count"] = None

    return results


def multi_split_experiment_from_data(data_dict, alpha, split_ratio=0.5, num_splits=500):
    """
    直接传入数据字典执行 multi-split LEC 实验（无需读文件）。

    参数:
        data_dict: dict, {sample_id: {correctness: int, uncertainty: float}}
        alpha: float, 目标 mFDR 上界
        split_ratio: float, calibration 占比
        num_splits: int, 重复次数

    返回:
        dict: 同 multi_split_experiment
    """
    data = [{"id": k} for k in data_dict.keys()]
    id_to_uncertainty = {k: v["uncertainty"] for k, v in data_dict.items()}
    id_to_correctness = {k: int(v["correctness"]) for k, v in data_dict.items()}

    num_samples = len(data)
    if num_samples == 0:
        return {
            "alpha": alpha, "num_samples": 0, "base_error_rate": None,
            "num_splits": num_splits, "infeasible_count": num_splits,
            "num_valid_splits": 0, "mean_fdp": None, "std_fdp": None,
            "mean_retention_rate": None, "mean_retained_count": None,
        }

    num_calib = int(num_samples * split_ratio)
    base_error_rate = 1 - sum(id_to_correctness.values()) / num_samples

    fdr_list = []
    retention_list = []
    retained_count_list = []
    infeasible_count = 0

    for epoch in range(num_splits):
        random.seed(epoch)
        shuffled = data.copy()
        random.shuffle(shuffled)
        calib_set = shuffled[:num_calib]
        test_set = shuffled[num_calib:]

        t_hat = lec_calibrate(calib_set, id_to_uncertainty, id_to_correctness, alpha)

        if t_hat is None:
            infeasible_count += 1
            continue

        fdr, selected_num, error_num = lec_evaluate(
            test_set, id_to_uncertainty, id_to_correctness, t_hat
        )

        fdr_list.append(fdr)
        if len(test_set) > 0:
            retention_list.append(selected_num / len(test_set))
        retained_count_list.append(selected_num)

    results = {
        "alpha": alpha,
        "num_samples": num_samples,
        "base_error_rate": round(base_error_rate, 4),
        "num_splits": num_splits,
        "infeasible_count": infeasible_count,
        "num_valid_splits": len(fdr_list),
    }

    if fdr_list:
        results["mean_fdp"] = round(float(np.mean(fdr_list)), 4)
        results["std_fdp"] = round(float(np.std(fdr_list)), 4)
        results["mean_retention_rate"] = round(float(np.mean(retention_list)), 4)
        results["mean_retained_count"] = round(float(np.mean(retained_count_list)), 2)
    else:
        results["mean_fdp"] = None
        results["std_fdp"] = None
        results["mean_retention_rate"] = None
        results["mean_retained_count"] = None

    return results

