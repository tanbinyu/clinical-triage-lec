"""补充指标计算：4项需额外校准运行的指标。

实现 metrics_summary.md 中标记为 ⚠️ 的指标:
  基线3: 仅专科拒绝 (Specialist-Only Rejection)
  基线4: 共享阈值 (Shared-Threshold)
  基线5: 先知式干净校准 (Oracle-Clean Calibration)
  敏感性分析: 校准大小分析 (Calibration Size Sensitivity)
"""
import random
from collections import defaultdict

import numpy as np

from pipeline.metrics.config import NUM_DISEASES, DISEASE_NAMES
from pipeline.metrics.compute import _lec_calibrate_np, full_metric

CALIB_SIZE_RATIOS = [0.1, 0.2, 0.3, 0.4, 0.5]


# ============================================================
# 共享预计算
# ============================================================

def precompute_all_with_rc(dl):
    """预计算所有 calib/test 样本数据（含 routing_correctness）。

    与 compute.precompute_all_samples 的区别：calib 也包含 rc 字段。
    """
    splits = dl.splits_phase1["splits"]
    n_splits = len(splits)

    keys = ("split", "u1", "pred_k", "unc", "corr", "rc")
    c = {k: [] for k in keys}
    t = {k: [] for k in keys}

    for b in range(n_splits):
        for role, buf in [("calib", c), ("test", t)]:
            for sid in splits[b][role]:
                p1 = dl.phase1_lookup.get(sid)
                if p1 is None:
                    continue
                u1, pred_k, _ = p1
                ba = dl.branch_a_lookup[pred_k].get(sid)
                buf["split"].append(b)
                buf["u1"].append(u1)
                buf["pred_k"].append(pred_k)
                if ba is not None:
                    buf["unc"].append(ba[0])
                    buf["corr"].append(ba[3])
                    buf["rc"].append(ba[1])
                else:
                    buf["unc"].append(np.nan)
                    buf["corr"].append(0)
                    buf["rc"].append(0)

    def _to_np(buf):
        return {
            "split": np.array(buf["split"], dtype=np.int32),
            "u1": np.array(buf["u1"], dtype=np.float64),
            "pred_k": np.array(buf["pred_k"], dtype=np.int32),
            "unc": np.array(buf["unc"], dtype=np.float64),
            "corr": np.array(buf["corr"], dtype=np.int32),
            "rc": np.array(buf["rc"], dtype=np.int32),
        }

    return {"calib": _to_np(c), "test": _to_np(t)}


# ============================================================
# 通用评估函数
# ============================================================

def _evaluate_specialist_system(t_hats_matrix, test, n_splits, alpha,
                                total_n_per_split=None):
    """给定专科阈值矩阵和测试样本，计算 per-disease + 系统级指标。

    Args:
        t_hats_matrix: [NUM_DISEASES, n_splits] 专科阈值
        test: dict of numpy arrays {split, pred_k, unc, corr}
        n_splits: 划分数
        alpha: 目标 FDR
        total_n_per_split: 各 split 总测试样本数（Cov_sys 分母）。
                           None 则从 test 数组推算。
    """
    t_sp = test["split"]
    t_pred = test["pred_k"]
    t_unc = test["unc"]
    t_corr = test["corr"]

    has_unc = ~np.isnan(t_unc)
    sample_th = t_hats_matrix[t_pred, t_sp]
    accept = has_unc & ~np.isnan(sample_th) & (t_unc <= sample_th)

    s_sys = accept.astype(np.float64)
    e_sys = 1.0 - t_corr.astype(np.float64)
    s_sys_e = s_sys * e_sys

    split_s_sys = np.bincount(t_sp, weights=s_sys, minlength=n_splits)
    split_s_sys_e = np.bincount(t_sp, weights=s_sys_e, minlength=n_splits)

    if total_n_per_split is None:
        total_n_per_split = np.bincount(t_sp, minlength=n_splits).astype(np.float64)
    else:
        total_n_per_split = np.asarray(total_n_per_split, dtype=np.float64)

    # ---- per-disease ----
    specialist = {}
    all_mfdr_k = []
    sum_err, sum_sel = 0, 0

    for k in range(NUM_DISEASES):
        mk = t_pred == k
        if not np.any(mk):
            specialist[k] = _empty_spec(k)
            continue

        k_sp = t_sp[mk]
        k_accept = accept[mk]
        k_corr = t_corr[mk]

        k_sel = np.bincount(k_sp, weights=k_accept.astype(np.float64),
                            minlength=n_splits)
        k_err = np.bincount(k_sp,
                            weights=(k_accept & (k_corr == 0)).astype(np.float64),
                            minlength=n_splits)
        pairs = [(float(k_err[b]), float(k_sel[b])) for b in range(n_splits)]
        fm = full_metric(pairs)

        tv = t_hats_matrix[k][~np.isnan(t_hats_matrix[k])]
        inf_n = int(np.isnan(t_hats_matrix[k]).sum())

        specialist[k] = {
            "disease_name": DISEASE_NAMES[k],
            "mFDR_k": fm["pooled"],
            "mean_fdp": fm["mean"],
            "std_fdp": fm["std"],
            "ci_low": fm["ci_low"],
            "ci_high": fm["ci_high"],
            "controlled": fm["pooled"] is not None and fm["pooled"] <= alpha,
            "infeasible_freq": round(inf_n / n_splits, 4),
            "threshold_mean": round(float(tv.mean()), 6) if len(tv) else None,
            "threshold_std": round(float(tv.std()), 6) if len(tv) else None,
            "total_selected": fm["total_den"],
            "total_error": fm["total_num"],
        }
        if fm["pooled"] is not None:
            all_mfdr_k.append(fm["pooled"])
            sum_err += fm["total_num"]
            sum_sel += fm["total_den"]

    mfdr_spec = {
        "macro": round(float(np.mean(all_mfdr_k)), 6) if all_mfdr_k else None,
        "micro": round(sum_err / sum_sel, 6) if sum_sel > 0 else None,
        "worst": round(float(max(all_mfdr_k)), 6) if all_mfdr_k else None,
    }

    # ---- system ----
    sys_pairs = [(float(split_s_sys_e[b]), float(split_s_sys[b]))
                 for b in range(n_splits)]
    fm_sys = full_metric(sys_pairs)

    cov_pairs = [(float(split_s_sys[b]), float(total_n_per_split[b]))
                 for b in range(n_splits)]
    fm_cov = full_metric(cov_pairs)

    total_acc = float(s_sys.sum())
    total_err_acc = float(s_sys_e.sum())
    total_n = float(total_n_per_split.sum())

    return {
        "specialist": {str(k): v for k, v in specialist.items()},
        "mfdr_spec": mfdr_spec,
        "mFDR_sys": fm_sys,
        "Cov_sys": fm_cov,
        "Ref_sys": {
            "pooled": round(1 - fm_cov["pooled"], 6) if fm_cov["pooled"] else None,
        },
        "N_AI_correct": int(round(total_acc - total_err_acc)),
        "N_AI_error": int(round(total_err_acc)),
        "N_human": int(round(total_n - total_acc)),
    }


def _empty_spec(k):
    return {
        "disease_name": DISEASE_NAMES[k],
        "mFDR_k": None, "mean_fdp": None, "std_fdp": None,
        "ci_low": None, "ci_high": None, "controlled": None,
        "infeasible_freq": 1.0,
        "threshold_mean": None, "threshold_std": None,
        "total_selected": 0, "total_error": 0,
    }


def _lambda_r_array(dl):
    return np.array(
        [t if t is not None else np.nan for t in dl.lec_phase1_t_hats],
        dtype=np.float64,
    )


# ============================================================
# 基线 3: 仅专科拒绝 (Specialist-Only Rejection)
# ============================================================

def compute_specialist_only_baseline(precomputed, n_splits, alpha):
    """S_R=1（路由器不拒绝），仅专科 LEC 校准。"""
    calib = precomputed["calib"]
    test = precomputed["test"]
    c_valid = ~np.isnan(calib["unc"])

    t_hats = np.full((NUM_DISEASES, n_splits), np.nan)
    for k in range(NUM_DISEASES):
        for b in range(n_splits):
            mask = c_valid & (calib["split"] == b) & (calib["pred_k"] == k)
            if not mask.any():
                continue
            th = _lec_calibrate_np(calib["unc"][mask], calib["corr"][mask], alpha)
            if th is not None:
                t_hats[k, b] = th

    result = _evaluate_specialist_system(t_hats, test, n_splits, alpha)
    result["Cov_R"] = 1.0
    result["Ref_R"] = 0.0
    return result


# ============================================================
# 基线 4: 共享阈值 (Shared-Threshold)
# ============================================================

def compute_shared_threshold_baseline(precomputed, dl, n_splits, alpha):
    """所有专科共用一个阈值（池化所有疾病的校准样本）。"""
    calib = precomputed["calib"]
    test = precomputed["test"]

    lr = _lambda_r_array(dl)
    c_valid = ~np.isnan(calib["unc"])
    c_sr = calib["u1"] <= lr[calib["split"]]

    shared = np.full(n_splits, np.nan)
    for b in range(n_splits):
        mask = c_valid & c_sr & (calib["split"] == b)
        if not mask.any():
            continue
        th = _lec_calibrate_np(calib["unc"][mask], calib["corr"][mask], alpha)
        if th is not None:
            shared[b] = th

    t_hats_matrix = np.tile(shared, (NUM_DISEASES, 1))

    t_lr = lr[test["split"]]
    s_r = test["u1"] <= t_lr
    test_sr = {k: test[k][s_r] for k in test}
    total_n = np.bincount(test["split"], minlength=n_splits).astype(np.float64)

    result = _evaluate_specialist_system(t_hats_matrix, test_sr, n_splits,
                                         alpha, total_n)

    tv = shared[~np.isnan(shared)]
    result["shared_threshold_mean"] = round(float(tv.mean()), 6) if len(tv) else None
    result["shared_threshold_std"] = round(float(tv.std()), 6) if len(tv) else None
    result["shared_infeasible_freq"] = round(
        int(np.isnan(shared).sum()) / n_splits, 4
    )
    return result


# ============================================================
# 基线 5: 先知式干净校准 (Oracle-Clean Calibration)
# ============================================================

def compute_oracle_clean_baseline(precomputed, dl, n_splits, alpha):
    """仅用 routing_correctness=1 的样本校准，应用于全部路由流。"""
    calib = precomputed["calib"]
    test = precomputed["test"]

    lr = _lambda_r_array(dl)
    c_valid = ~np.isnan(calib["unc"])
    c_sr = calib["u1"] <= lr[calib["split"]]

    t_hats = np.full((NUM_DISEASES, n_splits), np.nan)
    for k in range(NUM_DISEASES):
        for b in range(n_splits):
            mask = (c_valid & c_sr
                    & (calib["split"] == b)
                    & (calib["pred_k"] == k)
                    & (calib["rc"] == 1))
            if not mask.any():
                continue
            th = _lec_calibrate_np(calib["unc"][mask], calib["corr"][mask], alpha)
            if th is not None:
                t_hats[k, b] = th

    t_lr = lr[test["split"]]
    s_r = test["u1"] <= t_lr
    test_sr = {k: test[k][s_r] for k in test}
    total_n = np.bincount(test["split"], minlength=n_splits).astype(np.float64)

    return _evaluate_specialist_system(t_hats, test_sr, n_splits, alpha, total_n)


# ============================================================
# 校准大小分析 (Calibration Size Sensitivity)
# ============================================================

def compute_calibration_size_analysis(dl):
    """不同 calib/test 比例下的系统表现分析。

    对每个比例重新划分 → 路由器 LEC → 过滤 S_R=1 → 专科 LEC → 系统级评估。
    """
    all_sids = list(dl.phase1.keys())
    sid_to_idx = {sid: i for i, sid in enumerate(all_sids)}
    n_total = len(all_sids)

    u1 = np.array([dl.phase1[s]["uncertainty"] for s in all_sids])
    p1_corr = np.array([dl.phase1[s]["correctness"] for s in all_sids])
    pred_k = np.array([dl.phase1[s]["predicted_disease"] for s in all_sids])

    fused = np.full(n_total, np.nan)
    ba_corr = np.zeros(n_total, dtype=np.int32)
    for i, sid in enumerate(all_sids):
        ba = dl.branch_a_lookup[int(pred_k[i])].get(sid)
        if ba is not None:
            fused[i] = ba[0]
            ba_corr[i] = ba[3]

    ds_groups = defaultdict(list)
    for sid in all_sids:
        ds_groups[dl.phase1[sid]["source_dataset"]].append(sid)
    sorted_ds = sorted(ds_groups.keys())

    n_splits = len(dl.splits_phase1["splits"])
    results = []

    for ratio in CALIB_SIZE_RATIOS:
        router_t = np.full(n_splits, np.nan)
        spec_t = np.full((NUM_DISEASES, n_splits), np.nan)

        router_pairs = []
        cov_r_pairs = []
        sys_pairs = []
        cov_sys_pairs = []
        spec_pairs = {k: [] for k in range(NUM_DISEASES)}
        calib_size_0 = None

        for b in range(n_splits):
            random.seed(b)
            c_sids, t_sids = [], []
            for ds_name in sorted_ds:
                sids = ds_groups[ds_name].copy()
                random.shuffle(sids)
                nc = int(len(sids) * ratio)
                c_sids.extend(sids[:nc])
                t_sids.extend(sids[nc:])

            c_idx = np.array([sid_to_idx[s] for s in c_sids], dtype=np.int32)
            t_idx = np.array([sid_to_idx[s] for s in t_sids], dtype=np.int32)
            if calib_size_0 is None:
                calib_size_0 = len(c_idx)
            n_test = len(t_idx)

            # ---- Router LEC ----
            th_r = _lec_calibrate_np(u1[c_idx], p1_corr[c_idx], dl.beta)
            if th_r is not None:
                router_t[b] = th_r

            tu1 = u1[t_idx]
            tp1c = p1_corr[t_idx]
            if th_r is not None:
                r_sel = tu1 <= th_r
                router_pairs.append((int((r_sel & (tp1c == 0)).sum()),
                                     int(r_sel.sum())))
                cov_r_pairs.append((int(r_sel.sum()), n_test))
            else:
                router_pairs.append((0, 0))
                cov_r_pairs.append((0, n_test))
                sys_pairs.append((0, 0))
                cov_sys_pairs.append((0, n_test))
                for k in range(NUM_DISEASES):
                    spec_pairs[k].append((0, 0))
                continue

            # ---- Specialist LEC per disease ----
            c_u1 = u1[c_idx]
            c_pk = pred_k[c_idx]
            c_fused = fused[c_idx]
            c_ba = ba_corr[c_idx]
            c_sr = c_u1 <= th_r
            c_ok = ~np.isnan(c_fused)

            spec_th = np.full(NUM_DISEASES, np.nan)
            for k in range(NUM_DISEASES):
                mask = c_sr & (c_pk == k) & c_ok
                if not mask.any():
                    continue
                th_k = _lec_calibrate_np(c_fused[mask], c_ba[mask], dl.alpha)
                if th_k is not None:
                    spec_th[k] = th_k
                    spec_t[k, b] = th_k

            # ---- System eval ----
            t_pk = pred_k[t_idx]
            t_fused = fused[t_idx]
            t_ba = ba_corr[t_idx]
            t_sr = tu1 <= th_r

            sth = spec_th[t_pk]
            t_ok = ~np.isnan(t_fused) & ~np.isnan(sth)
            s_k = t_sr & t_ok & (t_fused <= sth)

            s_sys = s_k.astype(np.float64)
            e_sys = 1.0 - t_ba.astype(np.float64)
            sum_s = float(s_sys.sum())
            sum_se = float((s_sys * e_sys).sum())

            sys_pairs.append((sum_se, sum_s))
            cov_sys_pairs.append((sum_s, n_test))

            for k in range(NUM_DISEASES):
                mk = t_pk == k
                spec_pairs[k].append((int((s_sys[mk] * e_sys[mk]).sum()),
                                      int(s_sys[mk].sum())))

        # ---- Aggregate ----
        fm_router = full_metric(router_pairs)
        fm_cov_r = full_metric(cov_r_pairs)
        fm_sys = full_metric(sys_pairs)
        fm_cov_sys = full_metric(cov_sys_pairs)

        rv = router_t[~np.isnan(router_t)]
        r_inf = int(np.isnan(router_t).sum())

        specialist = {}
        for k in range(NUM_DISEASES):
            fm_k = full_metric(spec_pairs[k])
            tv = spec_t[k][~np.isnan(spec_t[k])]
            inf_k = int(np.isnan(spec_t[k]).sum())
            specialist[k] = {
                "disease_name": DISEASE_NAMES[k],
                "mFDR_k": fm_k["pooled"],
                "mean_fdp": fm_k["mean"],
                "std_fdp": fm_k["std"],
                "ci_low": fm_k["ci_low"],
                "ci_high": fm_k["ci_high"],
                "controlled": fm_k["pooled"] is not None and fm_k["pooled"] <= dl.alpha,
                "infeasible_freq": round(inf_k / n_splits, 4),
                "threshold_mean": round(float(tv.mean()), 6) if len(tv) else None,
                "threshold_std": round(float(tv.std()), 6) if len(tv) else None,
            }

        results.append({
            "split_ratio": ratio,
            "calib_size": calib_size_0,
            "test_size": n_total - calib_size_0,
            "router": {
                "mFDR_R": fm_router["pooled"],
                "mean_fdp": fm_router["mean"],
                "std_fdp": fm_router["std"],
                "ci_low": fm_router["ci_low"],
                "ci_high": fm_router["ci_high"],
                "controlled": (fm_router["pooled"] is not None
                               and fm_router["pooled"] <= dl.beta),
                "infeasible_freq": round(r_inf / n_splits, 4),
                "threshold_mean": round(float(rv.mean()), 6) if len(rv) else None,
                "threshold_std": round(float(rv.std()), 6) if len(rv) else None,
                "Cov_R": fm_cov_r["pooled"],
            },
            "specialist": {str(k): v for k, v in specialist.items()},
            "mFDR_sys": fm_sys["pooled"],
            "mean_fdp_sys": fm_sys["mean"],
            "std_fdp_sys": fm_sys["std"],
            "ci_low_sys": fm_sys["ci_low"],
            "ci_high_sys": fm_sys["ci_high"],
            "Cov_sys": fm_cov_sys["pooled"],
            "Ref_sys": (round(1 - fm_cov_sys["pooled"], 6)
                        if fm_cov_sys["pooled"] is not None else None),
        })

    return results
