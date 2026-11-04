"""核心指标计算模块。

实现 metrics_summary.md 中定义的全部可直接计算指标：
  第一章 风险控制 (9项)
  第二章 覆盖率   (6项)
  第三章 临床效用 (4项)
  第四章 路由污染 (8项/疾病)
  第五章 校准可行性 (3项)
  第七章 统计报告 (4项)
  第八章 基线     (2项可直接计算)
  第六章 敏感性分析 (β扫描 + α扫描)
"""
import json
import os
import numpy as np

from pipeline.metrics.config import (
    NUM_DISEASES, DISEASE_NAMES, N_BOOT,
    BETA_SWEEP, ALPHA_SWEEP, C_ERR_LIST, C_REF_LIST, BETA_SWEEP_FULL,
)


# ============================================================
# 通用聚合工具
# ============================================================

def aggregate_pairs(pairs):
    """给定 per-split (numerator, denominator) 列表，返回 pooled/mean/std。"""
    nums = [p[0] for p in pairs if p[1] > 0]
    dens = [p[1] for p in pairs if p[1] > 0]
    total_num = sum(p[0] for p in pairs)
    total_den = sum(p[1] for p in pairs)
    pooled = total_num / total_den if total_den > 0 else None
    ratios = [n / d for n, d in zip(nums, dens)] if nums else []
    mean = float(np.mean(ratios)) if ratios else None
    std = float(np.std(ratios)) if ratios else None
    return {
        "pooled": round(pooled, 6) if pooled is not None else None,
        "mean": round(mean, 6) if mean is not None else None,
        "std": round(std, 6) if std is not None else None,
        "n_valid": len(ratios),
        "total_num": total_num,
        "total_den": total_den,
    }


def bootstrap_ci(pairs, n_boot=N_BOOT, seed=42):
    """对 per-split (num, den) 对做 bootstrap，返回 95% CI。"""
    rng = np.random.default_rng(seed)
    if not pairs or sum(p[1] for p in pairs) == 0:
        return {"ci_low": None, "ci_high": None}
    arr = np.array(pairs)
    n = len(arr)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sampled = arr[idx]
        den_sum = sampled[:, 1].sum()
        if den_sum > 0:
            boots.append(sampled[:, 0].sum() / den_sum)
    if not boots:
        return {"ci_low": None, "ci_high": None}
    return {
        "ci_low": round(float(np.percentile(boots, 2.5)), 6),
        "ci_high": round(float(np.percentile(boots, 97.5)), 6),
    }


def full_metric(pairs, n_boot=N_BOOT):
    """完整指标：pooled + mean + std + CI + controlled 判定。"""
    agg = aggregate_pairs(pairs)
    ci = bootstrap_ci(pairs, n_boot)
    return {**agg, **ci}


# ============================================================
# 第一章 风险控制 & 第五章 校准可行性 & 第七章 统计报告
# ============================================================

def compute_lec_metrics(lec_data, alpha):
    """从 lec 文件计算 mFDR、不可行频率、阈值变异性等。

    返回 dict 包含:
      pooled_mfdr, mean_fdp, std_fdp, ci_low, ci_high,
      infeasible_freq, threshold_mean, threshold_std,
      base_error_rate, n_splits
    """
    splits = lec_data["splits"]
    total_splits = len(splits)

    valid_splits = [s for s in splits if s["t_hat"] is not None]
    infeasible = total_splits - len(valid_splits)

    pairs = [(s["error_num"], s["selected_num"]) for s in valid_splits]
    fm = full_metric(pairs)

    t_hats = [s["t_hat"] for s in valid_splits]
    thresh_mean = float(np.mean(t_hats)) if t_hats else None
    thresh_std = float(np.std(t_hats)) if t_hats else None

    controlled = fm["pooled"] is not None and fm["pooled"] <= alpha

    return {
        "alpha": alpha,
        "base_error_rate": lec_data.get("base_error_rate"),
        "n_splits": total_splits,
        "n_valid": len(valid_splits),
        "infeasible_count": infeasible,
        "infeasible_freq": round(infeasible / total_splits, 4) if total_splits else None,
        "pooled_mfdr": fm["pooled"],
        "mean_fdp": fm["mean"],
        "std_fdp": fm["std"],
        "ci_low": fm["ci_low"],
        "ci_high": fm["ci_high"],
        "controlled": controlled,
        "threshold_mean": round(thresh_mean, 6) if thresh_mean is not None else None,
        "threshold_std": round(thresh_std, 6) if thresh_std is not None else None,
        "total_selected": fm["total_den"],
        "total_error": fm["total_num"],
    }


# ============================================================
# 逐样本系统级计算（mFDR_sys, 覆盖率, 效用, 路由污染）
# ============================================================

def compute_per_split(dl):
    """对每个 split 做逐样本计算，返回 per-split 系统级 + 路由污染指标。

    这是整个计算的核心：遍历 500 splits × ~7000 test samples。
    """
    splits = dl.splits_phase1["splits"]
    n_splits = len(splits)

    # per-split 系统级累积
    sys_pairs_mfdr = []      # (sum_S_sys_E, sum_S_sys) per split
    sys_pairs_cov_r = []     # (sum_S_R, n_test) per split
    sys_pairs_cov_sys = []   # (sum_S_sys, n_test) per split
    sys_pairs_ref_r = []     # (n_test - sum_S_R, n_test)
    sys_pairs_ref_spec = []  # (sum_S_R - sum_S_sys, n_test)

    # per-split 临床效用
    utility_per_split = []  # (S_sys_1_E, S_sys_E, 1-S_sys, n_test)

    # per-disease per-split 路由污染
    disease_per_split = {k: [] for k in range(NUM_DISEASES)}
    # 每个 disease_per_split[k] 是 list of dict

    for b in range(n_splits):
        lambda_r = dl.lec_phase1_t_hats[b]
        test_ids = splits[b]["test"]
        n_test = len(test_ids)

        # 系统级
        sum_s_r = 0
        sum_s_sys = 0
        sum_s_sys_e = 0
        sum_s_sys_1_e = 0
        sum_1_s_sys = 0

        # 疾病级
        dk = {k: {
            "n_route": 0, "n_route_wrong": 0,
            "n_sk": 0, "n_sk_route_err": 0, "n_sk_diag_err": 0,
            "n_sk_err": 0, "n_route_wrong_rejected": 0,
        } for k in range(NUM_DISEASES)}

        if lambda_r is not None:
            for sid in test_ids:
                p1 = dl.phase1_lookup.get(sid)
                if p1 is None:
                    continue
                u1, pred_k, _ = p1
                s_r = 1 if u1 <= lambda_r else 0

                if s_r:
                    ba = dl.branch_a_lookup[pred_k].get(sid)
                    if ba is not None:
                        fused_unc, routing_correct, pred_correct, correctness = ba
                        lambda_k = dl.lec_branch_a_t_hats[pred_k][b]
                        s_k = 1 if (lambda_k is not None and fused_unc <= lambda_k) else 0
                    else:
                        s_k = 0
                        routing_correct = 0
                        pred_correct = 0
                        correctness = 0
                else:
                    s_k = 0
                    routing_correct = 0
                    pred_correct = 0
                    correctness = 0
                    fused_unc = 0

                s_sys = s_r * s_k
                e_sys = 1 - correctness

                sum_s_r += s_r
                sum_s_sys += s_sys
                sum_s_sys_e += s_sys * e_sys
                sum_s_sys_1_e += s_sys * (1 - e_sys)
                sum_1_s_sys += (1 - s_sys)

                if s_r and ba is not None:
                    dk[pred_k]["n_route"] += 1
                    dk[pred_k]["n_route_wrong"] += (1 - routing_correct)
                    dk[pred_k]["n_sk"] += s_k
                    dk[pred_k]["n_sk_route_err"] += s_k * (1 - routing_correct)
                    dk[pred_k]["n_sk_diag_err"] += s_k * routing_correct * (1 - pred_correct)
                    dk[pred_k]["n_sk_err"] += s_k * e_sys
                    dk[pred_k]["n_route_wrong_rejected"] += (1 - routing_correct) * (1 - s_k)

        sys_pairs_mfdr.append((sum_s_sys_e, sum_s_sys))
        sys_pairs_cov_r.append((sum_s_r, n_test))
        sys_pairs_cov_sys.append((sum_s_sys, n_test))
        sys_pairs_ref_r.append((n_test - sum_s_r, n_test))
        sys_pairs_ref_spec.append((sum_s_r - sum_s_sys, n_test))
        utility_per_split.append((sum_s_sys_1_e, sum_s_sys_e, sum_1_s_sys, n_test))

        for k in range(NUM_DISEASES):
            disease_per_split[k].append(dk[k])

    return {
        "sys_pairs_mfdr": sys_pairs_mfdr,
        "sys_pairs_cov_r": sys_pairs_cov_r,
        "sys_pairs_cov_sys": sys_pairs_cov_sys,
        "sys_pairs_ref_r": sys_pairs_ref_r,
        "sys_pairs_ref_spec": sys_pairs_ref_spec,
        "utility_per_split": utility_per_split,
        "disease_per_split": disease_per_split,
    }


# ============================================================
# 汇总系统级指标
# ============================================================

def aggregate_system_metrics(per_split_data):
    """从 per-split 数据汇总所有系统级指标。"""
    fm_mfdr_sys = full_metric(per_split_data["sys_pairs_mfdr"])
    fm_cov_r = full_metric(per_split_data["sys_pairs_cov_r"])
    fm_cov_sys = full_metric(per_split_data["sys_pairs_cov_sys"])
    fm_ref_r = full_metric(per_split_data["sys_pairs_ref_r"])
    fm_ref_spec = full_metric(per_split_data["sys_pairs_ref_spec"])

    # Cov_spec|R = Cov_sys / Cov_R
    cov_r = fm_cov_r["pooled"]
    cov_sys = fm_cov_sys["pooled"]
    cov_spec_given_r = cov_sys / cov_r if (cov_r and cov_r > 0) else None

    # Ref_sys = 1 - Cov_sys
    ref_sys = 1 - cov_sys if cov_sys is not None else None

    return {
        "mFDR_sys": fm_mfdr_sys,
        "Cov_R": fm_cov_r,
        "Cov_spec_given_R": {
            "pooled": round(cov_spec_given_r, 6) if cov_spec_given_r is not None else None,
        },
        "Cov_sys": fm_cov_sys,
        "Ref_R": fm_ref_r,
        "Ref_spec": fm_ref_spec,
        "Ref_sys": {
            "pooled": round(ref_sys, 6) if ref_sys is not None else None,
        },
    }


# ============================================================
# 第三章 临床效用
# ============================================================

def aggregate_clinical_utility(per_split_data):
    """汇总临床效用指标。"""
    ups = per_split_data["utility_per_split"]
    total_correct = sum(u[0] for u in ups)
    total_error = sum(u[1] for u in ups)
    total_human = sum(u[2] for u in ups)
    total_n = sum(u[3] for u in ups)

    utility_grid = {}
    for c_err in C_ERR_LIST:
        for c_ref in C_REF_LIST:
            u = (total_correct - c_err * total_error - c_ref * total_human) / total_n if total_n > 0 else None
            utility_grid[f"c_err={c_err}_c_ref={c_ref}"] = round(u, 6) if u is not None else None

    return {
        "N_AI_correct": total_correct,
        "N_AI_error": total_error,
        "N_human": total_human,
        "N_total": total_n,
        "utility_grid": utility_grid,
    }


# ============================================================
# 第四章 路由污染分析
# ============================================================

def aggregate_contamination(per_split_data):
    """汇总每个疾病的路由污染指标。"""
    results = {}
    for k in range(NUM_DISEASES):
        dps = per_split_data["disease_per_split"][k]

        # pooled pairs
        pairs_rho_route = [(d["n_route_wrong"], d["n_route"]) for d in dps]
        pairs_rho_acc = [(d["n_sk_route_err"], d["n_sk"]) for d in dps]
        pairs_rej_wrong = [(d["n_route_wrong_rejected"], d["n_route_wrong"]) for d in dps]
        pairs_n_route_err = [(d["n_sk_route_err"], 1) for d in dps]  # sum only
        pairs_n_diag_err = [(d["n_sk_diag_err"], 1) for d in dps]
        pairs_n_err = [(d["n_sk_err"], 1) for d in dps]

        fm_rho_route = full_metric(pairs_rho_route)
        fm_rho_acc = full_metric(pairs_rho_acc)
        fm_rej_wrong = full_metric(pairs_rej_wrong)

        total_route_err = sum(d["n_sk_route_err"] for d in dps)
        total_diag_err = sum(d["n_sk_diag_err"] for d in dps)
        total_err = sum(d["n_sk_err"] for d in dps)
        total_route = sum(d["n_route"] for d in dps)
        total_sk = sum(d["n_sk"] for d in dps)

        q_route = total_route_err / total_err if total_err > 0 else None
        q_diag = total_diag_err / total_err if total_err > 0 else None

        results[k] = {
            "disease_name": DISEASE_NAMES[k],
            "rho_route": fm_rho_route,
            "rho_acc": fm_rho_acc,
            "Rej_wrong": fm_rej_wrong,
            "N_route_err": total_route_err,
            "N_diag_err": total_diag_err,
            "N_err": total_err,
            "q_route_err": round(q_route, 6) if q_route is not None else None,
            "q_diag_err": round(q_diag, 6) if q_diag is not None else None,
            "total_routed": total_route,
            "total_accepted": total_sk,
        }
    return results


# ============================================================
# 第五章 校准可行性
# ============================================================

def aggregate_calibration_feasibility(dl, per_split_data=None):
    """汇总校准可行性指标。"""
    # Router
    router_lec = compute_lec_metrics(dl.lec_phase1, dl.beta)

    # Specialist per disease
    specialist = {}
    for k in range(NUM_DISEASES):
        if dl.lec_branch_a[k] is not None:
            specialist[k] = compute_lec_metrics(dl.lec_branch_a[k], dl.alpha)
        else:
            specialist[k] = None

    return {
        "router": {
            "infeasible_freq": router_lec["infeasible_freq"],
            "calib_size": len(dl.splits_phase1["splits"][0]["calib"]),
            "threshold_mean": router_lec["threshold_mean"],
            "threshold_std": router_lec["threshold_std"],
        },
        "specialist": {
            str(k): {
                "infeasible_freq": s["infeasible_freq"],
                "threshold_mean": s["threshold_mean"],
                "threshold_std": s["threshold_std"],
            } if s else None
            for k, s in specialist.items()
        },
    }


# ============================================================
# 第八章 基线
# ============================================================

def compute_baselines(dl):
    """计算两个可直接计算的基线。"""
    # ---- 基线1: 完全自动化 (不做任何拒绝) ----
    # mFDR = base_error_rate, Cov = 100%, Ref = 0%
    router_base = dl.lec_phase1["base_error_rate"]

    spec_base = {}
    for k in range(NUM_DISEASES):
        if dl.lec_branch_a[k] is not None:
            spec_base[k] = dl.lec_branch_a[k]["base_error_rate"]

    # 系统级: 完全不拒绝，所有样本都通过
    # mFDR_sys = 总错误数 / 总样本数
    total_err = 0
    total_n = 0
    for k in range(NUM_DISEASES):
        for sid, v in dl.branch_a[k].items():
            total_err += (1 - v["correctness"])
            total_n += 1
    sys_base = total_err / total_n if total_n > 0 else None

    full_automation = {
        "mFDR_R": router_base,
        "mFDR_k": spec_base,
        "mFDR_sys": round(sys_base, 6) if sys_base is not None else None,
        "Cov_R": 1.0,
        "Cov_sys": 1.0,
        "Ref_sys": 0.0,
    }

    # ---- 基线2: 仅路由器拒绝 (专科全部接受 S_k=1) ----
    # 使用主实验的 λ_R，但专科不拒绝
    # mFDR_R 与主实验相同（路由器阈值一致）
    router_lec = compute_lec_metrics(dl.lec_phase1, dl.beta)
    per_split = compute_per_split_router_only(dl)
    fm_mfdr_sys = full_metric(per_split["sys_pairs_mfdr"])
    fm_cov_r = full_metric(per_split["sys_pairs_cov_r"])

    router_only = {
        "mFDR_R": {
            "pooled": router_lec["pooled_mfdr"],
            "ci_low": router_lec["ci_low"],
            "ci_high": router_lec["ci_high"],
        },
        "mFDR_sys": fm_mfdr_sys,
        "Cov_R": fm_cov_r,
        "Cov_sys": fm_cov_r,  # S_k=1, so Cov_sys = Cov_R
        "Ref_sys": {
            "pooled": round(1 - fm_cov_r["pooled"], 6) if fm_cov_r["pooled"] else None,
        },
    }

    return {
        "full_automation": full_automation,
        "router_only_rejection": router_only,
    }


def compute_per_split_router_only(dl):
    """仅路由器拒绝基线的 per-split 计算（专科全部接受 S_k=1）。"""
    splits = dl.splits_phase1["splits"]
    sys_pairs_mfdr = []
    sys_pairs_cov_r = []
    sys_pairs_cov_sys = []

    for b in range(len(splits)):
        lambda_r = dl.lec_phase1_t_hats[b]
        test_ids = splits[b]["test"]
        n_test = len(test_ids)

        sum_s_r = 0
        sum_s_sys_e = 0
        sum_s_sys = 0

        if lambda_r is not None:
            for sid in test_ids:
                p1 = dl.phase1_lookup.get(sid)
                if p1 is None:
                    continue
                u1, pred_k, _ = p1
                s_r = 1 if u1 <= lambda_r else 0
                if s_r:
                    ba = dl.branch_a_lookup[pred_k].get(sid)
                    if ba is not None:
                        _, _, _, correctness = ba
                    else:
                        correctness = 0
                    s_sys = 1  # specialist accepts all
                    e_sys = 1 - correctness
                else:
                    s_sys = 0
                    e_sys = 0
                sum_s_r += s_r
                sum_s_sys += s_sys
                sum_s_sys_e += s_sys * e_sys

        sys_pairs_mfdr.append((sum_s_sys_e, sum_s_sys))
        sys_pairs_cov_r.append((sum_s_r, n_test))
        sys_pairs_cov_sys.append((sum_s_sys, n_test))

    return {
        "sys_pairs_mfdr": sys_pairs_mfdr,
        "sys_pairs_cov_r": sys_pairs_cov_r,
        "sys_pairs_cov_sys": sys_pairs_cov_sys,
    }


# ============================================================
# 第六章 敏感性分析
# ============================================================

def compute_beta_sweep(dl):
    """β 扫描：计算不同 β 下的路由器级指标。

    由于仅 β=0.1 有完整下游文件，其他 β 值只计算路由器级指标。
    """
    test_sizes = [len(s["test"]) for s in dl.splits_phase1["splits"]]
    results = []
    for beta in BETA_SWEEP:
        lec = dl.load_lec_phase1(beta)
        if lec is None:
            results.append({"beta": beta, "status": "no_data"})
            continue
        m = compute_lec_metrics(lec, beta)
        # Cov_R = sum(selected_num for valid splits) / sum(test_sizes for valid splits)
        lec_splits = lec["splits"]
        if len(lec_splits) == len(test_sizes):
            valid_test_sum = sum(
                test_sizes[i] for i, s in enumerate(lec_splits) if s["t_hat"] is not None
            )
        else:
            valid_test_sum = sum(test_sizes)
        results.append({
            "beta": beta,
            "mFDR_R": m["pooled_mfdr"],
            "mean_fdp": m["mean_fdp"],
            "std_fdp": m["std_fdp"],
            "ci_low": m["ci_low"],
            "ci_high": m["ci_high"],
            "controlled": m["controlled"],
            "Cov_R": round(m["total_selected"] / valid_test_sum, 6) if valid_test_sum > 0 else None,
            "infeasible_freq": m["infeasible_freq"],
            "threshold_mean": m["threshold_mean"],
            "threshold_std": m["threshold_std"],
        })
    return results


def compute_alpha_sweep(dl):
    """α 扫描：计算不同 α 下的专科级指标。"""
    results = []
    for alpha in ALPHA_SWEEP:
        row = {"alpha": alpha, "diseases": {}}
        all_mfdr = []
        total_err_sum = 0
        total_sel_sum = 0
        for k in range(NUM_DISEASES):
            lec = dl.load_lec_branch_a(k, alpha)
            if lec is None:
                row["diseases"][k] = None
                continue
            m = compute_lec_metrics(lec, alpha)
            row["diseases"][k] = {
                "mFDR_k": m["pooled_mfdr"],
                "mean_fdp": m["mean_fdp"],
                "std_fdp": m["std_fdp"],
                "ci_low": m["ci_low"],
                "ci_high": m["ci_high"],
                "controlled": m["controlled"],
                "infeasible_freq": m["infeasible_freq"],
                "threshold_mean": m["threshold_mean"],
            }
            if m["pooled_mfdr"] is not None:
                all_mfdr.append(m["pooled_mfdr"])
                total_err_sum += m["total_error"]
                total_sel_sum += m["total_selected"]

        # macro / micro / worst
        row["mFDR_spec_macro"] = round(float(np.mean(all_mfdr)), 6) if all_mfdr else None
        row["mFDR_spec_micro"] = round(total_err_sum / total_sel_sum, 6) if total_sel_sum > 0 else None
        row["mFDR_spec_worst"] = round(max(all_mfdr), 6) if all_mfdr else None
        row["n_diseases_controlled"] = sum(
            1 for k in range(NUM_DISEASES)
            if row["diseases"].get(k) and row["diseases"][k]["controlled"]
        )
        results.append(row)
    return results


# ============================================================
# Branch B 补充指标
# ============================================================

def compute_branch_b_metrics(dl):
    """计算 Branch B (人工分流) 的补充指标。"""
    results = {}
    for k in range(NUM_DISEASES):
        if dl.lec_branch_b[k] is not None:
            m = compute_lec_metrics(dl.lec_branch_b[k], dl.alpha)
            results[k] = {
                "disease_name": DISEASE_NAMES[k],
                "mFDR_B_k": m["pooled_mfdr"],
                "controlled": m["controlled"],
                "infeasible_freq": m["infeasible_freq"],
                "base_error_rate": m["base_error_rate"],
                "n_samples": len(dl.branch_b[k]),
            }
        else:
            results[k] = None
    return results


# ============================================================
# 预计算 + 系统级 α 扫描（高效向量化实现）
# ============================================================

def precompute_routed_samples(dl):
    """预计算所有 S_R=1 测试样本的数据，用于高效 α 扫描。

    返回四个 numpy 数组: split_idx, pred_k, fused_unc, correctness
    """
    splits = dl.splits_phase1["splits"]
    n_splits = len(splits)

    split_idx_list = []
    pred_k_list = []
    fused_unc_list = []
    correctness_list = []

    for b in range(n_splits):
        lambda_r = dl.lec_phase1_t_hats[b]
        if lambda_r is None:
            continue
        for sid in splits[b]["test"]:
            p1 = dl.phase1_lookup.get(sid)
            if p1 is None:
                continue
            u1, pred_k, _ = p1
            if u1 > lambda_r:
                continue
            ba = dl.branch_a_lookup[pred_k].get(sid)
            if ba is None:
                continue
            fused_unc, _, _, correctness = ba
            split_idx_list.append(b)
            pred_k_list.append(pred_k)
            fused_unc_list.append(fused_unc)
            correctness_list.append(correctness)

    return (
        np.array(split_idx_list, dtype=np.int32),
        np.array(pred_k_list, dtype=np.int32),
        np.array(fused_unc_list, dtype=np.float64),
        np.array(correctness_list, dtype=np.int32),
    )


def compute_alpha_sweep_system(dl, precomputed, test_sizes):
    """计算不同 α 下的系统级指标（mFDR_sys, Cov_sys, Ref_sys）。

    使用预计算的 S_R=1 样本数据和向量化操作，高效计算每个 α 值的系统级指标。
    """
    split_idx, pred_k_arr, fused_unc_arr, correctness_arr = precomputed
    n_splits = len(dl.splits_phase1["splits"])
    total_n_test = sum(test_sizes)

    results = []
    for alpha in ALPHA_SWEEP:
        # 构建 t_hats 矩阵: [NUM_DISEASES, n_splits]
        t_hats_matrix = np.full((NUM_DISEASES, n_splits), np.nan)
        for k in range(NUM_DISEASES):
            lec = dl.load_lec_branch_a(k, alpha)
            if lec is not None:
                for b_idx, s in enumerate(lec["splits"]):
                    if s["t_hat"] is not None:
                        t_hats_matrix[k, b_idx] = s["t_hat"]

        # 查找每个预计算样本的 t_hat
        if len(split_idx) > 0:
            sample_t_hats = t_hats_matrix[pred_k_arr, split_idx]
            valid_mask = ~np.isnan(sample_t_hats)
            accept_mask = valid_mask & (fused_unc_arr <= sample_t_hats)
        else:
            accept_mask = np.array([], dtype=bool)

        # S_sys = S_k (因为 S_R=1)
        s_sys = accept_mask.astype(np.float64)
        e_sys = 1.0 - correctness_arr.astype(np.float64)
        s_sys_e = s_sys * e_sys

        # 按 split 聚合
        if len(split_idx) > 0:
            split_s_sys = np.bincount(split_idx, weights=s_sys, minlength=n_splits)
            split_s_sys_e = np.bincount(split_idx, weights=s_sys_e, minlength=n_splits)
        else:
            split_s_sys = np.zeros(n_splits)
            split_s_sys_e = np.zeros(n_splits)

        total_s_sys = float(split_s_sys.sum())
        total_s_sys_e = float(split_s_sys_e.sum())

        # 也计算每疾病的 S_k 和 S_k*E
        per_disease = {}
        for k in range(NUM_DISEASES):
            mask = (pred_k_arr == k) if len(pred_k_arr) > 0 else np.array([], dtype=bool)
            n_routed = int(mask.sum())
            n_accepted = int(accept_mask[mask].sum()) if len(mask) > 0 else 0
            n_error = int((accept_mask[mask] & (correctness_arr[mask] == 0)).sum()) if len(mask) > 0 and len(accept_mask) > 0 else 0
            per_disease[k] = {
                "n_routed": n_routed,
                "n_accepted": n_accepted,
                "n_error": n_error,
                "Cov_k": round(n_accepted / n_routed, 6) if n_routed > 0 else None,
                "mFDR_k_sys": round(n_error / n_accepted, 6) if n_accepted > 0 else None,
            }

        mfdr_sys = total_s_sys_e / total_s_sys if total_s_sys > 0 else None
        cov_sys = total_s_sys / total_n_test if total_n_test > 0 else None
        ref_sys = 1 - cov_sys if cov_sys is not None else None

        # Split-wise FDP for mFDR_sys
        valid_sp = split_s_sys > 0
        if np.any(valid_sp):
            fdp_vals = split_s_sys_e[valid_sp] / split_s_sys[valid_sp]
            mean_fdp_sys = round(float(np.mean(fdp_vals)), 6)
            std_fdp_sys = round(float(np.std(fdp_vals)), 6)
        else:
            mean_fdp_sys = None
            std_fdp_sys = None

        # Bootstrap CI for mFDR_sys and Cov_sys
        ts = np.array(test_sizes, dtype=np.float64)
        mfdr_pairs = [(float(split_s_sys_e[i]), float(split_s_sys[i]))
                       for i in range(n_splits) if split_s_sys[i] > 0]
        cov_pairs = [(float(split_s_sys[i]), ts[i]) for i in range(n_splits)]
        ci_mfdr = bootstrap_ci(mfdr_pairs)
        ci_cov = bootstrap_ci(cov_pairs)

        results.append({
            "alpha": alpha,
            "mFDR_sys": round(mfdr_sys, 6) if mfdr_sys is not None else None,
            "mean_fdp_sys": mean_fdp_sys,
            "std_fdp_sys": std_fdp_sys,
            "ci_low_mfdr_sys": ci_mfdr["ci_low"],
            "ci_high_mfdr_sys": ci_mfdr["ci_high"],
            "Cov_sys": round(cov_sys, 6) if cov_sys is not None else None,
            "ci_low_cov_sys": ci_cov["ci_low"],
            "ci_high_cov_sys": ci_cov["ci_high"],
            "Ref_sys": round(ref_sys, 6) if ref_sys is not None else None,
            "total_s_sys": int(total_s_sys),
            "total_s_sys_e": int(round(total_s_sys_e)),
            "per_disease": per_disease,
        })

    return results


# ============================================================
# β 扫描完整版：LEC 校准 + 全部下游指标
# ============================================================

def _lec_calibrate_np(uncertainties, correctness, alpha):
    """Numpy 优化的 LEC 校准。

    返回阈值 t_hat，或 None 表示不可行。
    准则: sum((unc<=t AND correct=0) - alpha*(unc<=t)) <= -1
    """
    if len(uncertainties) == 0:
        return None

    sort_idx = np.argsort(uncertainties)
    sorted_unc = uncertainties[sort_idx]
    sorted_corr = correctness[sort_idx]

    unique_unc, start_idx = np.unique(sorted_unc, return_index=True)
    boundaries = np.append(start_idx[1:], len(sorted_unc)).astype(np.int64)

    cum_errors = np.cumsum(1 - sorted_corr)
    errors_at_b = cum_errors[boundaries - 1]
    selected = boundaries.astype(np.float64)

    loss = errors_at_b - alpha * selected
    valid = loss <= -1

    if not np.any(valid):
        return None

    last_valid = np.where(valid)[0][-1]
    return float(unique_unc[last_valid])


def precompute_all_samples(dl):
    """预计算所有 calib 和 test 样本数据为 numpy 数组。"""
    splits = dl.splits_phase1["splits"]
    n_splits = len(splits)

    c_split, c_u1, c_pred, c_unc, c_corr = [], [], [], [], []
    t_split, t_u1, t_pred, t_unc, t_corr, t_rc = [], [], [], [], [], []

    for b in range(n_splits):
        for sid in splits[b]["calib"]:
            p1 = dl.phase1_lookup.get(sid)
            if p1 is None:
                continue
            u1, pred_k, _ = p1
            ba = dl.branch_a_lookup[pred_k].get(sid)
            c_split.append(b)
            c_u1.append(u1)
            c_pred.append(pred_k)
            if ba is not None:
                c_unc.append(ba[0])
                c_corr.append(ba[3])
            else:
                c_unc.append(np.nan)
                c_corr.append(0)

        for sid in splits[b]["test"]:
            p1 = dl.phase1_lookup.get(sid)
            if p1 is None:
                continue
            u1, pred_k, _ = p1
            ba = dl.branch_a_lookup[pred_k].get(sid)
            t_split.append(b)
            t_u1.append(u1)
            t_pred.append(pred_k)
            if ba is not None:
                t_unc.append(ba[0])
                t_rc.append(ba[1])
                t_corr.append(ba[3])
            else:
                t_unc.append(np.nan)
                t_rc.append(0)
                t_corr.append(0)

    return {
        "calib": {
            "split": np.array(c_split, dtype=np.int32),
            "u1": np.array(c_u1, dtype=np.float64),
            "pred_k": np.array(c_pred, dtype=np.int32),
            "unc": np.array(c_unc, dtype=np.float64),
            "corr": np.array(c_corr, dtype=np.int32),
        },
        "test": {
            "split": np.array(t_split, dtype=np.int32),
            "u1": np.array(t_u1, dtype=np.float64),
            "pred_k": np.array(t_pred, dtype=np.int32),
            "unc": np.array(t_unc, dtype=np.float64),
            "corr": np.array(t_corr, dtype=np.int32),
            "rc": np.array(t_rc, dtype=np.int32),
        },
    }


def compute_beta_sweep_full(dl):
    """β 扫描完整版：为每个 β 计算全部下游指标。

    对每个 β 值，直接在代码中执行 LEC 校准（无需运行管线），
    然后计算系统级、专科级、路由污染指标。
    """
    precomputed = precompute_all_samples(dl)
    calib = precomputed["calib"]
    test = precomputed["test"]

    splits = dl.splits_phase1["splits"]
    n_splits = len(splits)
    test_sizes = np.array([len(s["test"]) for s in splits], dtype=np.float64)
    total_n_test = float(test_sizes.sum())

    results = []

    for beta in BETA_SWEEP_FULL:
        lec_phase1 = dl.load_lec_phase1(beta)
        if lec_phase1 is None:
            results.append({"beta": beta, "status": "no_data"})
            continue

        lambda_r_arr = np.array(
            [s["t_hat"] if s["t_hat"] is not None else np.nan for s in lec_phase1["splits"]],
            dtype=np.float64,
        )
        router_m = compute_lec_metrics(lec_phase1, beta)

        # ---- LEC 校准：为每个疾病×split 计算专科阈值 ----
        t_hats_matrix = np.full((NUM_DISEASES, n_splits), np.nan)
        for k in range(NUM_DISEASES):
            for b in range(n_splits):
                lr = lambda_r_arr[b]
                if np.isnan(lr):
                    continue
                mask = (
                    (calib["split"] == b)
                    & (calib["u1"] <= lr)
                    & (calib["pred_k"] == k)
                    & ~np.isnan(calib["unc"])
                )
                unc_k = calib["unc"][mask]
                corr_k = calib["corr"][mask]
                if len(unc_k) == 0:
                    continue
                t_hat = _lec_calibrate_np(unc_k, corr_k, dl.alpha)
                if t_hat is not None:
                    t_hats_matrix[k, b] = t_hat

        # ---- 系统级指标（numpy 向量化）----
        t_lambda_r = lambda_r_arr[test["split"]]
        s_r = test["u1"] <= t_lambda_r

        t_hats_lookup = t_hats_matrix[test["pred_k"], test["split"]]
        has_unc = ~np.isnan(test["unc"])
        s_k = s_r & has_unc & ~np.isnan(t_hats_lookup) & (test["unc"] <= t_hats_lookup)

        s_sys = s_k.astype(np.float64)
        e_sys = 1.0 - test["corr"].astype(np.float64)
        s_sys_e = s_sys * e_sys

        t_sp = test["split"]
        split_s_r = np.bincount(t_sp, weights=s_r.astype(np.float64), minlength=n_splits)
        split_s_sys = np.bincount(t_sp, weights=s_sys, minlength=n_splits)
        split_s_sys_e = np.bincount(t_sp, weights=s_sys_e, minlength=n_splits)

        total_s_sys = float(split_s_sys.sum())
        total_s_sys_e = float(split_s_sys_e.sum())
        total_s_r = float(split_s_r.sum())

        mfdr_sys = total_s_sys_e / total_s_sys if total_s_sys > 0 else None
        cov_r = total_s_r / total_n_test if total_n_test > 0 else None
        cov_sys = total_s_sys / total_n_test if total_n_test > 0 else None
        ref_sys = 1 - cov_sys if cov_sys is not None else None

        # Split-wise FDP
        valid_sp = split_s_sys > 0
        if np.any(valid_sp):
            fdp_vals = split_s_sys_e[valid_sp] / split_s_sys[valid_sp]
            mean_fdp_sys = round(float(np.mean(fdp_vals)), 6)
            std_fdp_sys = round(float(np.std(fdp_vals)), 6)
        else:
            mean_fdp_sys = None
            std_fdp_sys = None

        # Bootstrap CI
        mfdr_pairs = [(float(split_s_sys_e[i]), float(split_s_sys[i]))
                       for i in range(n_splits) if split_s_sys[i] > 0]
        cov_pairs = [(float(split_s_sys[i]), test_sizes[i]) for i in range(n_splits)]
        ci_mfdr = bootstrap_ci(mfdr_pairs)
        ci_cov = bootstrap_ci(cov_pairs)

        # ---- 专科级指标 ----
        specialist = {}
        all_mfdr_k = []
        total_err_k_sum = 0
        total_sel_k_sum = 0

        for k in range(NUM_DISEASES):
            mask_k = (test["pred_k"] == k) & s_r & has_unc
            unc_k = test["unc"][mask_k]
            corr_k = test["corr"][mask_k]
            split_k = t_sp[mask_k]
            th_k = t_hats_lookup[mask_k]
            rc_k = test["rc"][mask_k]

            sk_k = (~np.isnan(th_k)) & (unc_k <= th_k)
            ek = 1 - corr_k

            if len(split_k) > 0:
                sp_sel = np.bincount(split_k, weights=sk_k.astype(np.float64), minlength=n_splits)
                sp_err = np.bincount(split_k, weights=(sk_k * ek).astype(np.float64), minlength=n_splits)
            else:
                sp_sel = np.zeros(n_splits)
                sp_err = np.zeros(n_splits)

            tot_sel = float(sp_sel.sum())
            tot_err = float(sp_err.sum())
            mfdr_k = tot_err / tot_sel if tot_sel > 0 else None

            vk = sp_sel > 0
            if np.any(vk):
                fv = sp_err[vk] / sp_sel[vk]
                mean_fdp_k = round(float(np.mean(fv)), 6)
                std_fdp_k = round(float(np.std(fv)), 6)
            else:
                mean_fdp_k = None
                std_fdp_k = None

            mk_pairs = [(float(sp_err[i]), float(sp_sel[i])) for i in range(n_splits) if sp_sel[i] > 0]
            ci_k = bootstrap_ci(mk_pairs)

            th_splits = t_hats_matrix[k]
            infeasible_k = int(np.sum(np.isnan(th_splits)))
            valid_t = th_splits[~np.isnan(th_splits)]

            specialist[k] = {
                "disease_name": DISEASE_NAMES[k],
                "mFDR_k": round(mfdr_k, 6) if mfdr_k is not None else None,
                "mean_fdp": mean_fdp_k,
                "std_fdp": std_fdp_k,
                "ci_low": ci_k["ci_low"],
                "ci_high": ci_k["ci_high"],
                "controlled": mfdr_k is not None and mfdr_k <= dl.alpha,
                "infeasible_freq": round(infeasible_k / n_splits, 4),
                "threshold_mean": round(float(np.mean(valid_t)), 6) if len(valid_t) > 0 else None,
                "threshold_std": round(float(np.std(valid_t)), 6) if len(valid_t) > 0 else None,
            }
            if mfdr_k is not None:
                all_mfdr_k.append(mfdr_k)
                total_err_k_sum += tot_err
                total_sel_k_sum += tot_sel

        mfdr_spec = {
            "macro": round(float(np.mean(all_mfdr_k)), 6) if all_mfdr_k else None,
            "micro": round(total_err_k_sum / total_sel_k_sum, 6) if total_sel_k_sum > 0 else None,
            "worst": round(max(all_mfdr_k), 6) if all_mfdr_k else None,
        }

        # ---- 路由污染 ----
        contamination = {}
        for k in range(NUM_DISEASES):
            mask_k = (test["pred_k"] == k) & s_r & has_unc
            rc_k = test["rc"][mask_k]
            unc_k = test["unc"][mask_k]
            th_k = t_hats_lookup[mask_k]
            corr_k = test["corr"][mask_k]

            sk_k = (~np.isnan(th_k)) & (unc_k <= th_k)
            n_route = int(mask_k.sum())
            n_rw = int((rc_k == 0).sum())
            n_sk = int(sk_k.sum())
            n_sk_re = int((sk_k & (rc_k == 0)).sum())
            n_sk_e = int((sk_k & (corr_k == 0)).sum())
            n_rw_rej = int(((rc_k == 0) & ~sk_k).sum())

            contamination[k] = {
                "disease_name": DISEASE_NAMES[k],
                "rho_route": round(n_rw / n_route, 6) if n_route > 0 else None,
                "rho_acc": round(n_sk_re / n_sk, 6) if n_sk > 0 else None,
                "Rej_wrong": round(n_rw_rej / n_rw, 6) if n_rw > 0 else None,
                "N_route_err": n_sk_re,
                "N_diag_err": n_sk_e - n_sk_re,
                "N_err": n_sk_e,
            }

        results.append({
            "beta": beta,
            "mFDR_R": router_m["pooled_mfdr"],
            "mean_fdp_R": router_m["mean_fdp"],
            "std_fdp_R": router_m["std_fdp"],
            "ci_low_R": router_m["ci_low"],
            "ci_high_R": router_m["ci_high"],
            "controlled_R": router_m["controlled"],
            "Cov_R": round(cov_r, 6) if cov_r is not None else None,
            "infeasible_freq_R": router_m["infeasible_freq"],
            "threshold_mean_R": router_m["threshold_mean"],
            "threshold_std_R": router_m["threshold_std"],
            "mFDR_sys": round(mfdr_sys, 6) if mfdr_sys is not None else None,
            "mean_fdp_sys": mean_fdp_sys,
            "std_fdp_sys": std_fdp_sys,
            "ci_low_mfdr_sys": ci_mfdr["ci_low"],
            "ci_high_mfdr_sys": ci_mfdr["ci_high"],
            "Cov_sys": round(cov_sys, 6) if cov_sys is not None else None,
            "ci_low_cov_sys": ci_cov["ci_low"],
            "ci_high_cov_sys": ci_cov["ci_high"],
            "Ref_sys": round(ref_sys, 6) if ref_sys is not None else None,
            "mfdr_spec": mfdr_spec,
            "specialist": {str(k): v for k, v in specialist.items()},
            "contamination": {str(k): v for k, v in contamination.items()},
        })

    return results
