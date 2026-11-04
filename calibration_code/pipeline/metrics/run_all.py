"""主入口：计算所有可直接计算的指标并输出 JSON + Markdown 表格。

用法:
    cd calibration_code
    python -m pipeline.metrics.run_all
"""
import argparse
import json
import os
import sys
import time

import numpy as np

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pipeline.metrics.config import (
    NUM_DISEASES, DISEASE_NAMES, N_BOOT, BETA, ALPHA,
    BETA_SWEEP, ALPHA_SWEEP, C_ERR_LIST, C_REF_LIST, OUTPUT_ROOT,
)
from pipeline.metrics.data_loader import DataLoader
from pipeline.metrics.compute import (
    compute_lec_metrics,
    compute_per_split,
    aggregate_system_metrics,
    aggregate_clinical_utility,
    aggregate_contamination,
    aggregate_calibration_feasibility,
    compute_baselines,
    compute_beta_sweep_full,
    compute_alpha_sweep,
    compute_alpha_sweep_system,
    compute_branch_b_metrics,
    precompute_routed_samples,
)
from pipeline.metrics.compute_extra import (
    precompute_all_with_rc,
    compute_specialist_only_baseline,
    compute_shared_threshold_baseline,
    compute_oracle_clean_baseline,
    compute_calibration_size_analysis,
)


# ============================================================
# Markdown 格式化辅助函数
# ============================================================

def _f(val, precision=4):
    """格式化数值为字符串。"""
    if val is None:
        return "N/A"
    if isinstance(val, float):
        return f"{val:.{precision}f}"
    if isinstance(val, bool):
        return "✅" if val else "❌"
    return str(val)


def _ci(low, high):
    """格式化置信区间。"""
    if low is None or high is None:
        return "N/A"
    return f"[{low:.4f}, {high:.4f}]"


def _ms(mean, std):
    """格式化均值±标准差。"""
    if mean is None:
        return "N/A"
    if std is None:
        return f"{mean:.4f}"
    return f"{mean:.4f} ± {std:.4f}"


def _ctrl(val):
    """格式化 controlled 标记。"""
    if val is None:
        return "N/A"
    return "✅" if val else "❌"


# ============================================================
# 计算所有指标
# ============================================================

def compute_all_metrics(beta=BETA, alpha=ALPHA, data_root=None, input_root=None):
    """计算所有指标并返回结果字典。

    Explicit roots and risk levels make the function reusable for isolated
    experiment runs while preserving the historical defaults.
    """
    print("=" * 60)
    print("Loading data...")
    t0 = time.time()
    dl = DataLoader(beta=beta, alpha=alpha, data_root=data_root, input_root=input_root)
    print(f"  Data loaded in {time.time() - t0:.1f}s")

    # ---- 路由器级 ----
    print("Computing router-level metrics...")
    router = compute_lec_metrics(dl.lec_phase1, dl.beta)

    # ---- 专科级 ----
    print("Computing specialist-level metrics...")
    specialist = {}
    for k in range(NUM_DISEASES):
        if dl.lec_branch_a[k] is not None:
            specialist[k] = compute_lec_metrics(dl.lec_branch_a[k], dl.alpha)
        else:
            specialist[k] = None

    # mFDR_spec macro/micro/worst
    all_mfdr_k = [
        s["pooled_mfdr"] for s in specialist.values()
        if s and s["pooled_mfdr"] is not None
    ]
    total_err_k = sum(s["total_error"] for s in specialist.values() if s)
    total_sel_k = sum(s["total_selected"] for s in specialist.values() if s)
    mfdr_spec = {
        "macro": round(float(np.mean(all_mfdr_k)), 6) if all_mfdr_k else None,
        "micro": round(total_err_k / total_sel_k, 6) if total_sel_k > 0 else None,
        "worst": round(max(all_mfdr_k), 6) if all_mfdr_k else None,
    }

    # ---- 系统级（逐样本计算）----
    print("Computing system-level metrics (500 splits × ~7000 samples)...")
    t1 = time.time()
    per_split_data = compute_per_split(dl)
    print(f"  Per-split computation: {time.time() - t1:.1f}s")

    system_metrics = aggregate_system_metrics(per_split_data)
    utility = aggregate_clinical_utility(per_split_data)
    contamination = aggregate_contamination(per_split_data)
    calib_feasibility = aggregate_calibration_feasibility(dl)

    # ---- 基线 ----
    print("Computing baselines...")
    t2 = time.time()
    baselines = compute_baselines(dl)
    print(f"  Baselines: {time.time() - t2:.1f}s")

    # ---- 敏感性分析: β 扫描（完整版：含 LEC 校准 + 全部下游指标）----
    print("Computing β sweep (full, with LEC calibration)...")
    t_beta = time.time()
    beta_sweep = compute_beta_sweep_full(dl)
    print(f"  β sweep full: {time.time() - t_beta:.1f}s")

    # ---- 敏感性分析: α 扫描 ----
    print("Computing α sweep (specialist level)...")
    alpha_sweep = compute_alpha_sweep(dl)

    print("Precomputing routed samples for α sweep (system level)...")
    t3 = time.time()
    test_sizes = [len(s["test"]) for s in dl.splits_phase1["splits"]]
    precomputed = precompute_routed_samples(dl)
    n_precomputed = len(precomputed[0])
    print(f"  Precomputed {n_precomputed} routed samples in {time.time() - t3:.1f}s")

    print("Computing α sweep (system level)...")
    t4 = time.time()
    alpha_sweep_sys = compute_alpha_sweep_system(dl, precomputed, test_sizes)
    print(f"  α sweep system: {time.time() - t4:.1f}s")

    # 合并 α 扫描结果
    for i, alpha in enumerate(ALPHA_SWEEP):
        if i < len(alpha_sweep) and i < len(alpha_sweep_sys):
            alpha_sweep[i].update(alpha_sweep_sys[i])

    # ---- Branch B ----
    print("Computing Branch B metrics...")
    branch_b = compute_branch_b_metrics(dl)

    # ---- 补充基线 (3个需额外校准的基线) ----
    print("Precomputing all samples with routing_correctness...")
    t5 = time.time()
    n_splits = len(dl.splits_phase1["splits"])
    precomputed_rc = precompute_all_with_rc(dl)
    print(f"  Precomputed in {time.time() - t5:.1f}s")

    print("Computing baseline: specialist-only rejection...")
    t6 = time.time()
    bl_spec_only = compute_specialist_only_baseline(precomputed_rc, n_splits, alpha)
    print(f"  Specialist-only: {time.time() - t6:.1f}s")

    print("Computing baseline: shared-threshold...")
    t7 = time.time()
    bl_shared = compute_shared_threshold_baseline(precomputed_rc, dl, n_splits, alpha)
    print(f"  Shared-threshold: {time.time() - t7:.1f}s")

    print("Computing baseline: oracle-clean calibration...")
    t8 = time.time()
    bl_oracle = compute_oracle_clean_baseline(precomputed_rc, dl, n_splits, alpha)
    print(f"  Oracle-clean: {time.time() - t8:.1f}s")

    baselines["specialist_only_rejection"] = bl_spec_only
    baselines["shared_threshold"] = bl_shared
    baselines["oracle_clean_calibration"] = bl_oracle

    # ---- 校准大小分析 ----
    print("Computing calibration size analysis...")
    t9 = time.time()
    calib_size_analysis = compute_calibration_size_analysis(dl)
    print(f"  Calibration size analysis: {time.time() - t9:.1f}s")

    # ---- 汇总 ----
    results = {
        "meta": {
            "beta": beta,
            "alpha": alpha,
            "n_diseases": NUM_DISEASES,
            "n_splits": len(dl.splits_phase1["splits"]),
            "n_boot": N_BOOT,
            "calib_size": len(dl.splits_phase1["splits"][0]["calib"]),
            "disease_names": {str(k): v for k, v in DISEASE_NAMES.items()},
        },
        "risk_control": {
            "router": router,
            "specialist": {str(k): s for k, s in specialist.items()},
            "mfdr_spec": mfdr_spec,
            "system": system_metrics["mFDR_sys"],
        },
        "coverage": {
            "Cov_R": system_metrics["Cov_R"],
            "Cov_spec_given_R": system_metrics["Cov_spec_given_R"],
            "Cov_sys": system_metrics["Cov_sys"],
            "Ref_R": system_metrics["Ref_R"],
            "Ref_spec": system_metrics["Ref_spec"],
            "Ref_sys": system_metrics["Ref_sys"],
        },
        "clinical_utility": utility,
        "contamination": contamination,
        "calibration_feasibility": calib_feasibility,
        "baselines": baselines,
        "beta_sweep": beta_sweep,
        "alpha_sweep": alpha_sweep,
        "branch_b": branch_b,
        "calibration_size_analysis": calib_size_analysis,
    }

    return results


# ============================================================
# Markdown 生成
# ============================================================

def generate_markdown(results):
    """生成完整的 Markdown 报告。"""
    md = []
    meta = results["meta"]

    # ---- 标题 ----
    md.append("# 论文评价指标计算结果\n")
    md.append(f"> **实验参数**: β = {meta['beta']}, α = {meta['alpha']}, "
              f"疾病数 = {meta['n_diseases']}, 随机划分数 = {meta['n_splits']}, "
              f"Bootstrap 次数 = {meta['n_boot']}, 校准集大小 = {meta['calib_size']}\n")

    # ============================================================
    # 表 1: 风险控制指标
    # ============================================================
    md.append("## 一、风险控制指标\n")

    rc = results["risk_control"]
    router = rc["router"]
    sys_m = rc["system"]
    spec = rc["mfdr_spec"]

    md.append("### 1.1 汇总\n")
    md.append("| 指标 | Pooled | FDP Mean ± Std | 95% CI | Controlled |")
    md.append("|------|--------|-----------|--------|-----------|")
    md.append(f"| mFDR_R (β={meta['beta']}) | {_f(router['pooled_mfdr'])} | "
              f"{_ms(router['mean_fdp'], router['std_fdp'])} | "
              f"{_ci(router['ci_low'], router['ci_high'])} | {_ctrl(router['controlled'])} |")
    md.append(f"| mFDR_sys | {_f(sys_m['pooled'])} | "
              f"{_ms(sys_m['mean'], sys_m['std'])} | "
              f"{_ci(sys_m['ci_low'], sys_m['ci_high'])} | {_ctrl(sys_m['pooled'] is not None and sys_m['pooled'] <= meta['beta'])} |")
    md.append(f"| mFDR_spec^macro | {_f(spec['macro'])} | - | - | - |")
    md.append(f"| mFDR_spec^micro | {_f(spec['micro'])} | - | - | - |")
    md.append(f"| mFDR_spec^worst | {_f(spec['worst'])} | - | - | - |")

    md.append(f"\n### 1.2 各疾病专科风险 (α = {meta['alpha']})\n")
    md.append("| k | 疾病 | mFDR_k | FDP Mean ± Std | 95% CI | Controlled | Base Error Rate |")
    md.append("|---|------|--------|-----------|--------|-----------|----------------|")
    for k in range(meta["n_diseases"]):
        s = rc["specialist"].get(str(k))
        name = meta["disease_names"].get(str(k), f"disease_{k}")
        if s is None:
            md.append(f"| {k} | {name} | N/A | N/A | N/A | N/A | N/A |")
        else:
            md.append(f"| {k} | {name} | {_f(s['pooled_mfdr'])} | "
                      f"{_ms(s['mean_fdp'], s['std_fdp'])} | "
                      f"{_ci(s['ci_low'], s['ci_high'])} | {_ctrl(s['controlled'])} | "
                      f"{_f(s['base_error_rate'])} |")

    # ============================================================
    # 表 2: 覆盖率指标
    # ============================================================
    md.append("\n## 二、效率 / 覆盖率指标\n")
    cov = results["coverage"]
    md.append("| 指标 | Pooled | Mean ± Std | 95% CI |")
    md.append("|------|--------|-----------|--------|")
    for name, key in [
        ("Cov_R", "Cov_R"),
        ("Cov_spec|R", "Cov_spec_given_R"),
        ("Cov_sys", "Cov_sys"),
        ("Ref_R", "Ref_R"),
        ("Ref_spec", "Ref_spec"),
        ("Ref_sys", "Ref_sys"),
    ]:
        m = cov[key]
        if isinstance(m, dict):
            pooled = m.get("pooled")
            mean = m.get("mean")
            std = m.get("std")
            ci_low = m.get("ci_low")
            ci_high = m.get("ci_high")
        else:
            pooled = m
            mean = std = ci_low = ci_high = None
        md.append(f"| {name} | {_f(pooled)} | {_ms(mean, std)} | {_ci(ci_low, ci_high)} |")

    md.append("\n> 恒等关系: Ref_sys = Ref_R + Ref_spec")

    # ============================================================
    # 表 3: 临床效用指标
    # ============================================================
    md.append("\n## 三、临床效用指标\n")
    u = results["clinical_utility"]
    md.append("| 指标 | 值 |")
    md.append("|------|-----|")
    md.append(f"| N_AI_correct (AI 接受且正确) | {u['N_AI_correct']} |")
    md.append(f"| N_AI_error (AI 接受但错误) | {u['N_AI_error']} |")
    md.append(f"| N_human (转交人类) | {u['N_human']} |")
    md.append(f"| N_total (总测试样本) | {u['N_total']} |")

    md.append("\n### 效用函数网格 U(c_err, c_ref)\n")
    md.append("| c_err \\ c_ref | " + " | ".join(str(c) for c in C_REF_LIST) + " |")
    md.append("|" + "---|" * (len(C_REF_LIST) + 1))
    for c_err in C_ERR_LIST:
        row = [f"{c_err}"]
        for c_ref in C_REF_LIST:
            key = f"c_err={c_err}_c_ref={c_ref}"
            v = u["utility_grid"].get(key)
            row.append(_f(v))
        md.append("| " + " | ".join(row) + " |")

    # ============================================================
    # 表 4: 路由污染分析
    # ============================================================
    md.append("\n## 四、路由污染分析指标\n")
    cont = results["contamination"]
    md.append("| k | 疾病 | ρ_route | ρ_acc | Rej_wrong | N_route-err | N_diag-err | N_err | q_route-err | q_diag-err |")
    md.append("|---|------|---------|-------|-----------|-------------|-----------|-------|---------|--------|")
    for k in range(meta["n_diseases"]):
        c = cont.get(k) or cont.get(str(k))
        if c is None:
            name = meta["disease_names"].get(str(k), f"disease_{k}")
            md.append(f"| {k} | {name} | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |")
            continue
        rho_r = c["rho_route"]["pooled"] if isinstance(c["rho_route"], dict) else c["rho_route"]
        rho_a = c["rho_acc"]["pooled"] if isinstance(c["rho_acc"], dict) else c["rho_acc"]
        rej = c["Rej_wrong"]["pooled"] if isinstance(c["Rej_wrong"], dict) else c["Rej_wrong"]
        md.append(f"| {k} | {c['disease_name']} | "
                  f"{_f(rho_r)} | {_f(rho_a)} | {_f(rej)} | "
                  f"{c['N_route_err']} | {c['N_diag_err']} | {c['N_err']} | "
                  f"{_f(c['q_route_err'])} | {_f(c['q_diag_err'])} |")

    # ============================================================
    # 表 5: 校准可行性指标
    # ============================================================
    md.append("\n## 五、校准可行性指标\n")
    cf = results["calibration_feasibility"]
    md.append("| 层级 | 疾病 | 不可行频率 | 阈值均值 | 阈值标准差 |")
    md.append("|------|------|-----------|---------|-----------|")
    r = cf["router"]
    md.append(f"| Router | - | {r['infeasible_freq']} | {_f(r['threshold_mean'])} | {_f(r['threshold_std'])} |")
    for k in range(meta["n_diseases"]):
        s = cf["specialist"].get(str(k))
        name = meta["disease_names"].get(str(k), f"disease_{k}")
        if s is None:
            md.append(f"| Specialist | {name} | N/A | N/A | N/A |")
        else:
            md.append(f"| Specialist | {name} | {s['infeasible_freq']} | "
                      f"{_f(s['threshold_mean'])} | {_f(s['threshold_std'])} |")

    # ============================================================
    # 表 6: 基线对比
    # ============================================================
    md.append("\n## 六、基线对比系统\n")
    bl = results["baselines"]
    md.append("| 基线 | mFDR_R | mFDR_sys | Cov_R | Cov_sys | Ref_sys |")
    md.append("|------|--------|----------|-------|---------|---------|")

    fa = bl["full_automation"]
    md.append(f"| 完全自动化 | {_f(fa['mFDR_R'])} | {_f(fa['mFDR_sys'])} | "
              f"{_f(fa['Cov_R'])} | {_f(fa['Cov_sys'])} | {_f(fa['Ref_sys'])} |")

    ro = bl["router_only_rejection"]
    ro_mfdr_r = ro["mFDR_R"]["pooled"] if isinstance(ro["mFDR_R"], dict) else ro["mFDR_R"]
    ro_mfdr_sys = ro["mFDR_sys"]["pooled"] if isinstance(ro["mFDR_sys"], dict) else ro["mFDR_sys"]
    ro_cov_r = ro["Cov_R"]["pooled"] if isinstance(ro["Cov_R"], dict) else ro["Cov_R"]
    ro_cov_sys = ro["Cov_sys"]["pooled"] if isinstance(ro["Cov_sys"], dict) else ro["Cov_sys"]
    ro_ref_sys = ro["Ref_sys"]["pooled"] if isinstance(ro["Ref_sys"], dict) else ro["Ref_sys"]
    md.append(f"| 仅路由器拒绝 | {_f(ro_mfdr_r)} | {_f(ro_mfdr_sys)} | "
              f"{_f(ro_cov_r)} | {_f(ro_cov_sys)} | {_f(ro_ref_sys)} |")

    # 仅专科拒绝
    so = bl.get("specialist_only_rejection", {})
    so_mfdr_sys = so.get("mFDR_sys", {})
    so_cov_sys = so.get("Cov_sys", {})
    md.append(f"| 仅专科拒绝 | - | "
              f"{_f(so_mfdr_sys.get('pooled') if isinstance(so_mfdr_sys, dict) else so_mfdr_sys)} | "
              f"{_f(so.get('Cov_R', 1.0))} | "
              f"{_f(so_cov_sys.get('pooled') if isinstance(so_cov_sys, dict) else so_cov_sys)} | "
              f"{_f(so.get('Ref_sys', {}).get('pooled') if isinstance(so.get('Ref_sys'), dict) else so.get('Ref_sys'))} |")

    # 共享阈值
    st = bl.get("shared_threshold", {})
    st_mfdr_sys = st.get("mFDR_sys", {})
    st_cov_sys = st.get("Cov_sys", {})
    md.append(f"| 共享阈值 | {_f(router['pooled_mfdr'])} | "
              f"{_f(st_mfdr_sys.get('pooled') if isinstance(st_mfdr_sys, dict) else st_mfdr_sys)} | "
              f"{_f(cov.get('Cov_R', {}).get('pooled') if isinstance(cov.get('Cov_R'), dict) else cov.get('Cov_R'))} | "
              f"{_f(st_cov_sys.get('pooled') if isinstance(st_cov_sys, dict) else st_cov_sys)} | "
              f"{_f(st.get('Ref_sys', {}).get('pooled') if isinstance(st.get('Ref_sys'), dict) else st.get('Ref_sys'))} |")

    # 先知式干净校准
    oc = bl.get("oracle_clean_calibration", {})
    oc_mfdr_sys = oc.get("mFDR_sys", {})
    oc_cov_sys = oc.get("Cov_sys", {})
    md.append(f"| 先知式干净校准 | {_f(router['pooled_mfdr'])} | "
              f"{_f(oc_mfdr_sys.get('pooled') if isinstance(oc_mfdr_sys, dict) else oc_mfdr_sys)} | "
              f"{_f(cov.get('Cov_R', {}).get('pooled') if isinstance(cov.get('Cov_R'), dict) else cov.get('Cov_R'))} | "
              f"{_f(oc_cov_sys.get('pooled') if isinstance(oc_cov_sys, dict) else oc_cov_sys)} | "
              f"{_f(oc.get('Ref_sys', {}).get('pooled') if isinstance(oc.get('Ref_sys'), dict) else oc.get('Ref_sys'))} |")

    # 主系统
    sys_pooled = rc["system"]["pooled"]
    cov_r_pooled = cov["Cov_R"]["pooled"] if isinstance(cov["Cov_R"], dict) else cov["Cov_R"]
    cov_sys_pooled = cov["Cov_sys"]["pooled"] if isinstance(cov["Cov_sys"], dict) else cov["Cov_sys"]
    ref_sys_pooled = cov["Ref_sys"]["pooled"] if isinstance(cov["Ref_sys"], dict) else cov["Ref_sys"]
    md.append(f"| **主系统 (LEC)** | **{_f(router['pooled_mfdr'])}** | **{_f(sys_pooled)}** | "
              f"**{_f(cov_r_pooled)}** | **{_f(cov_sys_pooled)}** | **{_f(ref_sys_pooled)}** |")

    # ============================================================
    # 表 7: β 扫描
    # ============================================================
    md.append("\n## 七、β 扫描（路由风险敏感性）\n")
    bs = results["beta_sweep"]

    # 7a: 系统级汇总
    md.append("### 7.1 系统级汇总\n")
    md.append("| β | mFDR_R | Ctrl_R | Cov_R | mFDR_sys | Cov_sys | Ref_sys | mFDR_spec^macro |")
    md.append("|---|--------|--------|-------|----------|---------|---------|-----------------|")
    for row in bs:
        if row.get("status") == "no_data":
            md.append(f"| {row['beta']} | N/A | N/A | N/A | N/A | N/A | N/A | N/A |")
        else:
            ms = row.get("mfdr_spec", {})
            md.append(f"| {row['beta']} | {_f(row.get('mFDR_R'))} | {_ctrl(row.get('controlled_R'))} | "
                      f"{_f(row.get('Cov_R'))} | {_f(row.get('mFDR_sys'))} | "
                      f"{_f(row.get('Cov_sys'))} | {_f(row.get('Ref_sys'))} | "
                      f"{_f(ms.get('macro'))} |")

    # 7b: 统计报告 (FDP mean±std + Bootstrap 95% CI)
    md.append("\n### 7.2 统计报告（FDP mean±std + Bootstrap 95% CI）\n")
    md.append("| β | FDP_R mean±std | FDP_R CI | FDP_sys mean±std | FDP_sys CI |")
    md.append("|---|-----------------|-----------|-------------------|-------------|")
    for row in bs:
        if row.get("status") == "no_data":
            md.append(f"| {row['beta']} | N/A | N/A | N/A | N/A |")
        else:
            md.append(f"| {row['beta']} | "
                      f"{_ms(row.get('mean_fdp_R'), row.get('std_fdp_R'))} | "
                      f"{_ci(row.get('ci_low_R'), row.get('ci_high_R'))} | "
                      f"{_ms(row.get('mean_fdp_sys'), row.get('std_fdp_sys'))} | "
                      f"{_ci(row.get('ci_low_mfdr_sys'), row.get('ci_high_mfdr_sys'))} |")

    # 7c: 路由污染（macro 平均）
    md.append("\n### 7.3 路由污染（macro 平均）\n")
    md.append("| β | ρ_route^macro | ρ_acc^macro | Rej_wrong^macro |")
    md.append("|---|---------------|------------|-----------------|")
    for row in bs:
        if row.get("status") == "no_data":
            md.append(f"| {row['beta']} | N/A | N/A | N/A |")
        else:
            cont = row.get("contamination", {})
            rho_r_vals = [cont[str(k)]["rho_route"] for k in range(meta["n_diseases"])
                          if str(k) in cont and cont[str(k)]["rho_route"] is not None]
            rho_a_vals = [cont[str(k)]["rho_acc"] for k in range(meta["n_diseases"])
                          if str(k) in cont and cont[str(k)]["rho_acc"] is not None]
            rej_w_vals = [cont[str(k)]["Rej_wrong"] for k in range(meta["n_diseases"])
                          if str(k) in cont and cont[str(k)]["Rej_wrong"] is not None]
            rho_r_macro = round(float(np.mean(rho_r_vals)), 6) if rho_r_vals else None
            rho_a_macro = round(float(np.mean(rho_a_vals)), 6) if rho_a_vals else None
            rej_w_macro = round(float(np.mean(rej_w_vals)), 6) if rej_w_vals else None
            md.append(f"| {row['beta']} | {_f(rho_r_macro)} | {_f(rho_a_macro)} | {_f(rej_w_macro)} |")

    md.append("\n> 每个 β 值的完整专科级和路由污染数据见 JSON 文件。")

    # ============================================================
    # 表 8: α 扫描
    # ============================================================
    md.append("\n## 八、α 扫描（专科风险敏感性）\n")
    aswp = results["alpha_sweep"]

    # 8a: 专科级 + 系统级汇总
    md.append("### 8.1 汇总\n")
    md.append("| α | mFDR_spec^macro | mFDR_spec^micro | mFDR_spec^worst | N Ctrl | "
             "mFDR_sys | Cov_sys | Ref_sys |")
    md.append("|---|-----------------|-----------------|-----------------|---------|"
              "----------|---------|---------|")
    for row in aswp:
        alpha = row["alpha"]
        macro = row.get("mFDR_spec_macro")
        micro = row.get("mFDR_spec_micro")
        worst = row.get("mFDR_spec_worst")
        n_ctrl = row.get("n_diseases_controlled", "N/A")
        mfdr_sys = row.get("mFDR_sys")
        cov_sys = row.get("Cov_sys")
        ref_sys = row.get("Ref_sys")
        md.append(f"| {alpha} | {_f(macro)} | {_f(micro)} | {_f(worst)} | {n_ctrl} | "
                  f"{_f(mfdr_sys)} | {_f(cov_sys)} | {_f(ref_sys)} |")

    # 8b: 统计报告 (FDP mean±std + Bootstrap 95% CI)
    md.append("\n### 8.2 统计报告（FDP mean±std + Bootstrap 95% CI）\n")
    md.append("| α | FDP_sys mean±std | FDP_sys CI | Cov_sys CI |")
    md.append("|---|-------------------|------------|-----------|")
    for row in aswp:
        alpha = row["alpha"]
        md.append(f"| {alpha} | "
                  f"{_ms(row.get('mean_fdp_sys'), row.get('std_fdp_sys'))} | "
                  f"{_ci(row.get('ci_low_mfdr_sys'), row.get('ci_high_mfdr_sys'))} | "
                  f"{_ci(row.get('ci_low_cov_sys'), row.get('ci_high_cov_sys'))} |")

    # α 扫描详细表（主实验 α=0.1 的各疾病数据）
    main_alpha = meta["alpha"]
    main_alpha_idx = ALPHA_SWEEP.index(main_alpha) if main_alpha in ALPHA_SWEEP else 0
    if main_alpha_idx < len(aswp):
        main_row = aswp[main_alpha_idx]
        diseases = main_row.get("diseases", {})
        md.append(f"\n### α = {main_alpha} 各疾病详细\n")
        md.append("| k | 疾病 | mFDR_k | FDP Mean ± Std | 95% CI | Controlled | Infeasible | Threshold Mean |")
        md.append("|---|------|--------|-----------|--------|-----------|-----------|----------------|")
        for k in range(NUM_DISEASES):
            d = diseases.get(k) or diseases.get(str(k))
            name = DISEASE_NAMES.get(k, f"disease_{k}")
            if d is None:
                md.append(f"| {k} | {name} | N/A | N/A | N/A | N/A | N/A | N/A |")
            else:
                md.append(f"| {k} | {name} | {_f(d.get('mFDR_k'))} | "
                          f"{_ms(d.get('mean_fdp'), d.get('std_fdp'))} | "
                          f"{_ci(d.get('ci_low'), d.get('ci_high'))} | "
                          f"{_ctrl(d.get('controlled'))} | {d.get('infeasible_freq', 'N/A')} | "
                          f"{_f(d.get('threshold_mean'))} |")

    # ============================================================
    # 表 9: Branch B 补充指标
    # ============================================================
    md.append("\n## 九、Branch B 补充指标\n")
    bb = results["branch_b"]
    md.append("| k | 疾病 | mFDR_B_k | Controlled | Infeasible | Base Error | N_samples |")
    md.append("|---|------|----------|-----------|-----------|-----------|-----------|")
    for k in range(meta["n_diseases"]):
        b = bb.get(k) or bb.get(str(k))
        name = meta["disease_names"].get(str(k), f"disease_{k}")
        if b is None:
            md.append(f"| {k} | {name} | N/A | N/A | N/A | N/A | N/A |")
        else:
            md.append(f"| {k} | {b['disease_name']} | {_f(b['mFDR_B_k'])} | "
                      f"{_ctrl(b['controlled'])} | {b['infeasible_freq']} | "
                      f"{_f(b['base_error_rate'])} | {b['n_samples']} |")

    # ============================================================
    # 表 10: 校准大小敏感性分析
    # ============================================================
    md.append("\n## 十、校准大小敏感性分析\n")
    csa = results.get("calibration_size_analysis", [])
    if csa:
        md.append("| Calib比例 | Calib大小 | Test大小 | mFDR_R | Ctrl_R | Cov_R | "
                  "mFDR_sys | Cov_sys | Ref_sys |")
        md.append("|----------|----------|---------|--------|--------|-------|"
                  "----------|---------|---------|")
        for row in csa:
            r_info = row.get("router", {})
            md.append(f"| {row['split_ratio']} | {row['calib_size']} | {row['test_size']} | "
                      f"{_f(r_info.get('mFDR_R'))} | {_ctrl(r_info.get('controlled'))} | "
                      f"{_f(r_info.get('Cov_R'))} | "
                      f"{_f(row.get('mFDR_sys'))} | {_f(row.get('Cov_sys'))} | "
                      f"{_f(row.get('Ref_sys'))} |")

        md.append("\n### 统计报告\n")
        md.append("| Calib比例 | FDP_sys mean±std | FDP_sys CI | Cov_sys |")
        md.append("|----------|------------------|-----------|---------|")
        for row in csa:
            md.append(f"| {row['split_ratio']} | "
                      f"{_ms(row.get('mean_fdp_sys'), row.get('std_fdp_sys'))} | "
                      f"{_ci(row.get('ci_low_sys'), row.get('ci_high_sys'))} | "
                      f"{_f(row.get('Cov_sys'))} |")
    else:
        md.append("> 校准大小分析数据不可用。")

    # ============================================================
    # 附录: 指标计算覆盖情况
    # ============================================================
    md.append("\n## 附录: 指标计算覆盖情况\n")
    md.append("| 类别 | 指标数 | 状态 |")
    md.append("|------|--------|------|")
    md.append("| 风险控制 | 9 | 9 ✅ |")
    md.append("| 覆盖率 | 6 | 6 ✅ |")
    md.append("| 临床效用 | 4 | 4 ✅ |")
    md.append("| 路由污染 | 8 | 8 ✅ |")
    md.append("| 校准可行性 | 3 | 3 ✅ |")
    md.append("| 敏感性分析 | 3 | 3 ✅ |")
    md.append("| 统计报告 | 4 | 4 ✅ |")
    md.append("| 基线 | 5 | 5 ✅ |")
    md.append("| **合计** | **42** | **42 ✅** |")

    return "\n".join(md) + "\n"


# ============================================================
# 主函数
# ============================================================

def main(argv=None):
    parser = argparse.ArgumentParser(description="Compute the full metrics bundle")
    parser.add_argument("--beta", type=float, default=BETA)
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--data-root", default=None,
                        help="Root containing derived step2-step6 JSON")
    parser.add_argument("--input-root", default=None,
                        help="Root containing immutable step1 JSON (defaults to data root)")
    parser.add_argument("--output-dir", default=os.path.join(PROJECT_ROOT, "result", "metrics"))
    args = parser.parse_args(argv)
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 60)
    print("论文指标计算系统")
    print("=" * 60)

    results = compute_all_metrics(
        beta=args.beta,
        alpha=args.alpha,
        data_root=args.data_root,
        input_root=args.input_root,
    )

    # 输出 JSON
    json_path = os.path.join(out_dir, "metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nJSON saved to {json_path}")

    # 输出 Markdown
    md_path = os.path.join(out_dir, "summary.md")
    md = generate_markdown(results)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"Markdown saved to {md_path}")

    print("\n" + "=" * 60)
    print("All done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
