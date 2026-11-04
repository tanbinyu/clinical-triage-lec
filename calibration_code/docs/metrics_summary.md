# 论文评价指标总结

> 论文：Risk-Controlled Clinical Deployment of Foundation Models for Triage and Specialist Diagnosis across Chest Disease Pathways

---

## 一、风险控制指标

### 1.1 路由器层面

| 指标 | 符号 | 含义 |
|------|------|------|
| 路由器接受案例错误率 | $\widehat{\mathrm{mFDR}}_R$ | 路由器接受的案例中，路由错误（分错诊室）的比例 |
| 路由目标风险水平 | $\beta$ | 用户指定的路由风险上限，需满足 $\mathrm{mFDR}_R(\lambda_R) \leq \beta$ |

### 1.2 专科层面

| 指标 | 符号 | 含义 |
|------|------|------|
| 专科 $k$ 接受案例错误率 | $\widehat{\mathrm{mFDR}}_k$ | 专科 $k$ 接受的案例中，路由错误 + 诊断错误的比例 |
| 宏平均专科风险 | $\widehat{\mathrm{mFDR}}_{\mathrm{spec}}^{\mathrm{macro}}$ | 所有可估专科风险的简单平均 |
| 微平均专科风险 | $\widehat{\mathrm{mFDR}}_{\mathrm{spec}}^{\mathrm{micro}}$ | 按接受案例数加权的平均 |
| 最差专科风险 | $\widehat{\mathrm{mFDR}}_{\mathrm{spec}}^{\mathrm{worst}}$ | 所有专科中最高的接受案例错误率 |
| 专科目标风险水平 | $\alpha_k$ | 专科 $k$ 的诊断风险上限，需满足 $\mathrm{mFDR}_k(\lambda_k;\lambda_R) \leq \alpha_k$；主实验中所有专科共用同一水平 $\alpha$ |

### 1.3 端到端系统层面

| 指标 | 符号 | 含义 |
|------|------|------|
| 端到端接受案例错误率 | $\widehat{\mathrm{mFDR}}_{\mathrm{sys}}$ | 完整两阶段 AI 管理的案例中，路由错误或诊断错误的总比例 |
| 分组级错误比例 | $\widehat{\mathrm{FDP}}_{\mathrm{sys}}^{(b)}$ | 单次 split 的端到端错误率（作为变异性摘要报告） |

---

## 二、效率 / 覆盖率指标

| 指标 | 符号 | 含义 |
|------|------|------|
| 路由覆盖率 | $\widehat{\mathrm{Cov}}_R$ | 被路由器接受的测试样本比例 |
| 条件专科覆盖率 | $\widehat{\mathrm{Cov}}_{\mathrm{spec}\mid R}$ | 路由器接受的案例中被专科也接受的比例 |
| 整体 AI 管理率 | $\widehat{\mathrm{Cov}}_{\mathrm{sys}}$ | 两阶段都被接受（完全由 AI 管理）的样本比例 |
| 人类转诊率 | $\widehat{\mathrm{Ref}}_{\mathrm{sys}}$ | $1 - \widehat{\mathrm{Cov}}_{\mathrm{sys}}$，被转交人类的总比例 |
| 路由器阶段转诊率 | $\widehat{\mathrm{Ref}}_R$ | 被路由器拒绝的比例 |
| 专科阶段转诊率 | $\widehat{\mathrm{Ref}}_{\mathrm{spec}}$ | 被路由器接受但被专科拒绝的比例 |

恒等关系：$\widehat{\mathrm{Ref}}_{\mathrm{sys}} = \widehat{\mathrm{Ref}}_R + \widehat{\mathrm{Ref}}_{\mathrm{spec}}$

---

## 三、临床效用指标

| 指标 | 符号 | 含义 |
|------|------|------|
| 正确 AI 管理案例数 | $N_{\mathrm{AI\text{-}correct}}$ | AI 接受且正确的案例数 |
| 错误 AI 管理案例数 | $N_{\mathrm{AI\text{-}error}}$ | AI 接受但错误的案例数 |
| 人类转诊案例数 | $N_{\mathrm{human}}$ | 被转交人类的案例数 |
| 效用函数 | $\widehat{U}(c_{\mathrm{err}}, c_{\mathrm{ref}})$ | 综合考虑正确收益、错误代价 $c_{\mathrm{err}}$、转诊代价 $c_{\mathrm{ref}}$ 的效用值，在多种代价比例下报告 |

---

## 四、路由污染分析指标

| 指标 | 符号 | 含义 |
|------|------|------|
| 路由污染率 | $\widehat{\rho}_{k}^{\mathrm{route}}$ | 专科 $k$ 的路由流中，真实诊室不是 $k$ 的案例比例 |
| 接受污染率 | $\widehat{\rho}_{k}^{\mathrm{acc}}$ | 专科 $k$ 接受的案例中，真实诊室不是 $k$ 的比例 |
| 错误路由拒绝率 | $\widehat{\mathrm{Rej}}_{k}^{\mathrm{wrong}}$ | 被错误路由到专科 $k$ 的案例中，被专科拒绝的比例 |
| 路由致错误数 | $N_{k}^{\mathrm{route\text{-}err}}$ | 专科 $k$ 接受的案例中，因路由错误导致的错误数 |
| 诊断错误数 | $N_{k}^{\mathrm{diag\text{-}err}}$ | 专科 $k$ 接受的案例中，因诊断错误导致的错误数 |
| 总接受错误数 | $N_{k}^{\mathrm{err}}$ | $N_{k}^{\mathrm{route\text{-}err}} + N_{k}^{\mathrm{diag\text{-}err}}$，专科 $k$ 接受的案例中的总错误数 |
| 路由错误占比 | $\widehat{q}_{k}^{\mathrm{route\text{-}err}}$ | 路由致错误占总接受错误的比例 |
| 诊断错误占比 | $\widehat{q}_{k}^{\mathrm{diag\text{-}err}}$ | 诊断致错误占总接受错误的比例 |

---

## 五、校准可行性指标

| 指标 | 含义 |
|------|------|
| 不可行频率 | 路由器或专科在校准数据上无法满足安全约束时，被标记为不可行的频率 |
| 校准集大小 | 路由校准集大小 $n_{\mathrm{cal}}^R$、专科路由校准流大小 $n_{\mathrm{cal},k}^{S,\mathrm{route}}(\lambda_R)$ |
| 阈值变异性 | 多次随机 split 间阈值 $\lambda_R^{(b)}$、$\lambda_k^{(b)}$ 的波动 |

---

## 六、敏感性分析实验

| 实验 | 变化参数 | 固定参数 | 观察指标 |
|------|----------|----------|----------|
| 路由风险敏感性 | $\beta$ 网格扫描 | $\alpha$ 固定 | 路由覆盖率、路由污染率、专科覆盖率、端到端错误率、人类转诊率 |
| 专科风险敏感性 | $\alpha$ 网格扫描 | $\beta$ 固定 | 各专科接受案例错误率、专科覆盖率、AI 管理率、端到端错误率、人类转诊率 |
| 校准大小分析 | 校准集大小变化 | — | 阈值变异性、接受案例错误率、AI 管理率、不可行频率 |

---

## 七、统计报告方式

- **主指标**：跨 $B$ 次随机 split 的 pooled 估计（合并所有测试预测）
- **变异性**：split 级别的均值和标准差
- **置信区间**：对 $B$ 次 split 非参数自助重采样，报告双侧 95% CI
- **零分母情况**：报告为不可估计（not estimable）

---

## 八、基线对比系统（5 个）

| 基线 | 描述 |
|------|------|
| 完全自动化 (Full Automation) | 不做任何不确定性拒绝 |
| 仅路由器拒绝 (Router-Only Rejection) | 仅校准路由器阈值，专科全部接受 |
| 仅专科拒绝 (Specialist-Only Rejection) | 路由器全部接受，仅校准专科阈值 |
| 共享阈值 (Shared-Threshold) | 所有专科用同一阈值，不做专科特异校准 |
| 先知式干净校准 (Oracle-Clean Calibration) | 仅用正确路由的案例校准专科（不可部署，作为参考上界） |

---

## 九、指标计算可行性分析（基于实验结果字段设计的 JSON 管线）

### 9.0 JSON 文件与论文概念对照

| 论文概念 | 对应 JSON 文件 | 关键字段 |
|----------|---------------|----------|
| 路由器不确定性 $u_i^R$ | `phase_1.json` | `uncertainty` (u₁) |
| 路由正确性 $A_i^R$ | `phase_1.json` | `correctness`（1=正确路由） |
| 预测诊室 $\hat{r}_i$ | `phase_1.json` | `predicted_disease` |
| 真实诊室 $r_i$ | `phase_1.json` | `true_disease` |
| 路由器阈值 $\lambda_R$ | `lec_phase_1_alpha_{β}.json` | `splits[].t_hat` |
| 路由器接受 $S_R$ | 由 `phase_1.uncertainty ≤ lec_phase_1.t_hat` 推导 | — |
| 专科不确定性 $u_i^k$ | `branch_a_disease_{k}.json` | `uncertainty` (λ₁·u₁+λ₂·u₂) |
| 路由正确性（分项） | `branch_a_disease_{k}.json` | `routing_correctness` |
| 诊断正确性（分项） | `branch_a_disease_{k}.json` | `pred_correctness` |
| 综合正确性 $A_i^k$ | `branch_a_disease_{k}.json` | `correctness`（= routing_correctness AND pred_correctness） |
| 专科阈值 $\lambda_k$ | `lec_branch_a_disease_{k}_alpha_{α}.json` | `splits[].t_hat` |
| 专科接受 $S_k$ | 由 `branch_a_disease.uncertainty ≤ lec_branch_a.t_hat` 推导 | — |
| 数据划分 | `splits_phase_1.json` / `splits_branch_a.json` | `epoch`, `calib[]`, `test[]` |

### 9.1 完全可计算的指标 ✅

以下指标可直接从现有 JSON 文件计算，无需额外校准运行：

#### 风险控制指标（第一至三章）

| 指标 | 所需 JSON 文件 | 计算方法 |
|------|----------------|----------|
| $\widehat{\mathrm{mFDR}}_R$ | `lec_phase_1` | pooled: $\sum$ `error_num` / $\sum$ `selected_num` |
| $\widehat{\mathrm{mFDR}}_k$ | `lec_branch_a_disease_{k}` 或 `branch_a_disease_{k}` + `splits_branch_a` + `lec_branch_a_disease_{k}` | pooled: $\sum$ `error_num` / $\sum$ `selected_num`；或逐样本计算 $S_k \cdot (1-\text{correctness})$ / $\sum S_k$ |
| $\widehat{\mathrm{mFDR}}_{\mathrm{spec}}^{\mathrm{macro}}$ | 由各 $\widehat{\mathrm{mFDR}}_k$ 派生 | $\frac{1}{|\mathcal{K}_{\mathrm{acc}}|} \sum_k \widehat{\mathrm{mFDR}}_k$ |
| $\widehat{\mathrm{mFDR}}_{\mathrm{spec}}^{\mathrm{micro}}$ | 由各 $\widehat{\mathrm{mFDR}}_k$ 派生 | $\sum_k \sum S_k E_k$ / $\sum_k \sum S_k$ |
| $\widehat{\mathrm{mFDR}}_{\mathrm{spec}}^{\mathrm{worst}}$ | 由各 $\widehat{\mathrm{mFDR}}_k$ 派生 | $\max_k \widehat{\mathrm{mFDR}}_k$ |
| $\widehat{\mathrm{mFDR}}_{\mathrm{sys}}$ | `phase_1` + `splits_phase_1` + `lec_phase_1` + `branch_a_disease_{k}` + `splits_branch_a` + `lec_branch_a_disease_{k}` | 逐样本：$S_{\mathrm{sys}} = S_R \cdot S_k$，$E_{\mathrm{sys}} = 1 - \text{correctness}$；pooled: $\sum S_{\mathrm{sys}} E_{\mathrm{sys}}$ / $\sum S_{\mathrm{sys}}$ |
| $\widehat{\mathrm{FDP}}_{\mathrm{sys}}^{(b)}$ | 同上 | 按 split 分组计算 |

#### 效率/覆盖率指标（第二章）

| 指标 | 所需 JSON 文件 | 计算方法 |
|------|----------------|----------|
| $\widehat{\mathrm{Cov}}_R$ | `phase_1` + `splits_phase_1` + `lec_phase_1` | $\sum S_R$ / $N_{\mathrm{test}}$ |
| $\widehat{\mathrm{Cov}}_{\mathrm{spec}\mid R}$ | `branch_a_disease_{k}` + `splits_branch_a` + `lec_branch_a_disease_{k}` + `lec_phase_1` | $\sum S_{\mathrm{sys}}$ / $\sum S_R$ |
| $\widehat{\mathrm{Cov}}_{\mathrm{sys}}$ | 同上 | $\sum S_{\mathrm{sys}}$ / $N_{\mathrm{test}}$ |
| $\widehat{\mathrm{Ref}}_{\mathrm{sys}}$ | 由 $\widehat{\mathrm{Cov}}_{\mathrm{sys}}$ 派生 | $1 - \widehat{\mathrm{Cov}}_{\mathrm{sys}}$ |
| $\widehat{\mathrm{Ref}}_R$ | `phase_1` + `splits_phase_1` + `lec_phase_1` | $\sum (1-S_R)$ / $N_{\mathrm{test}}$ |
| $\widehat{\mathrm{Ref}}_{\mathrm{spec}}$ | 同 Cov 指标 | $\sum S_R(1-S_k)$ / $N_{\mathrm{test}}$ |

#### 临床效用指标（第三章）

| 指标 | 所需 JSON 文件 | 计算方法 |
|------|----------------|----------|
| $N_{\mathrm{AI\text{-}correct}}$ | `branch_a_disease_{k}` + `splits_branch_a` + `lec_branch_a_disease_{k}` | $\sum S_{\mathrm{sys}} (1-E_{\mathrm{sys}})$ |
| $N_{\mathrm{AI\text{-}error}}$ | 同上 | $\sum S_{\mathrm{sys}} E_{\mathrm{sys}}$ |
| $N_{\mathrm{human}}$ | 由覆盖率派生 | $\sum (1-S_{\mathrm{sys}})$ |
| $\widehat{U}(c_{\mathrm{err}}, c_{\mathrm{ref}})$ | 由上述三者派生 | 按公式计算，扫描多组代价比例 |

#### 路由污染分析指标（第四章）

| 指标 | 所需 JSON 文件 | 计算方法 |
|------|----------------|----------|
| $\widehat{\rho}_{k}^{\mathrm{route}}$ | `branch_a_disease_{k}` + `splits_branch_a` | test 集中 `routing_correctness=0` 的比例 |
| $\widehat{\rho}_{k}^{\mathrm{acc}}$ | 同上 + `lec_branch_a_disease_{k}` | $S_k=1$ 且 `routing_correctness=0` 的样本 / $\sum S_k$ |
| $\widehat{\mathrm{Rej}}_{k}^{\mathrm{wrong}}$ | 同上 | `routing_correctness=0` 且 $S_k=0$ 的样本 / `routing_correctness=0` 总数 |
| $N_{k}^{\mathrm{route\text{-}err}}$ | 同上 | $\sum S_k \cdot \mathbf{1}\{\text{routing\_correctness}=0\}$ |
| $N_{k}^{\mathrm{diag\text{-}err}}$ | 同上 | $\sum S_k \cdot \mathbf{1}\{\text{routing\_correctness}=1, \text{pred\_correctness}=0\}$ |
| $N_{k}^{\mathrm{err}}$ | 由上两项派生 | $N_{k}^{\mathrm{route\text{-}err}} + N_{k}^{\mathrm{diag\text{-}err}}$ |
| $\widehat{q}_{k}^{\mathrm{route\text{-}err}}$ | 由上派生 | $N_{k}^{\mathrm{route\text{-}err}} / N_{k}^{\mathrm{err}}$ |
| $\widehat{q}_{k}^{\mathrm{diag\text{-}err}}$ | 由上派生 | $N_{k}^{\mathrm{diag\text{-}err}} / N_{k}^{\mathrm{err}}$ |

#### 校准可行性指标（第五章）

| 指标 | 所需 JSON 文件 | 计算方法 |
|------|----------------|----------|
| 不可行频率 | `lec_phase_1` / `lec_branch_a_disease_{k}` | `t_hat is null` 的 split 数 / 总 split 数 $B$。校准代码已保证 `t_hat` 只有两种状态：有效数值（校准成功）或 `null`（校准不可行） |
| 校准集大小 $n_{\mathrm{cal}}^R$ | `splits_phase_1` | `len(splits[].calib)` |
| 校准流大小 $n_{\mathrm{cal},k}^{S,\mathrm{route}}(\lambda_R)$ | `splits_branch_a` + `branch_a_disease_{k}` | 按疾病 $k$ 过滤 calib 集后计数 |
| 阈值变异性 $\lambda_R^{(b)}$ | `lec_phase_1` | `splits[].t_hat` 的均值与标准差（排除 `null`） |
| 阈值变异性 $\lambda_k^{(b)}$ | `lec_branch_a_disease_{k}` | `splits[].t_hat` 的均值与标准差（排除 `null`） |

#### 统计报告（第七章）

| 方法 | 可行性 | 说明 |
|------|----------|------|
| Pooled 估计 | ✅ | 跨 split 合并分子分母 |
| Split-wise 均值/标准差 | ✅ | 由 `lec` 文件的 per-split `error_num`/`selected_num` 或逐样本计算 |
| Bootstrap 95% CI | ✅ | 对 $B$ 次 split 非参数重采样，重算 pooled 估计 |

#### 基线（第八章，部分）

| 基线 | 可行性 | 计算方法 |
|------|----------|----------|
| 完全自动化 | ✅ | `lec_phase_1.base_error_rate`（路由器）+ `lec_branch_a.base_error_rate`（专科）；Cov = 100%，Ref = 0% |
| 仅路由器拒绝 | ✅ | 使用 `lec_phase_1` 的 $\lambda_R$，专科全部接受（$S_k=1$） |

### 9.2 需要额外校准运行的指标 ⚠️

以下指标所需的原始数据（`phase_1.json`、`phase_2_{dataset}.json`）已充分，但需要以不同参数或不同数据子集执行校准管线。其中 **β 扫描和 α 扫描已完成**，文件已保存；校准大小分析和 3 个基线仍需运行。步骤编号对应 `实验结果字段设计.md` 中的管线阶段。

---

#### 敏感性分析实验（第六章）

##### 1. 路由风险敏感性（$\beta$ 网格扫描）✅ 已完成

**已完成参数网格**：$\beta \in \{0.01, 0.025, 0.05, 0.1, 0.15, 0.2\}$，$\alpha$ 固定不变。

**已保存的文件**（每个 $\beta$ 值一组，共 6 组）：

| 文件 | 说明 |
|------|------|
| `lec_phase_1_alpha_{β}.json` | 该 $\beta$ 下的路由器校准结果（含 `t_hat`、`selected_num`、`error_num`、`base_error_rate`） |
| `splits_branch_a.json` | 该 $\beta$ 下基于新 $\lambda_R$ 重新过滤的 Branch A/B 划分 |
| `branch_a_disease_{k}.json` | 该 $\beta$ 下重新聚合的疾病文件（含 `uncertainty`、`routing_correctness`、`pred_correctness`、`correctness`） |
| `lec_branch_a_disease_{k}_alpha_{α}.json` | 该 $\beta$ 下、固定 $\alpha$ 重新校准的专科阈值 |

> **注意**：`phase_1.json` 和 `phase_2_{dataset}.json` 不随 $\beta$ 变化（模型输出不变），可复用主实验文件。

**指标计算方法**：对每个 $\beta$ 值，加载该 $\beta$ 对应的一组文件，用 9.1 中相同公式计算：

| 指标 | 所需文件（按 $\beta$ 加载） | 计算方法 |
|------|-------------------------------|----------|
| $\widehat{\mathrm{mFDR}}_R(\beta)$ | `lec_phase_1_alpha_{β}` | pooled: $\sum$ `error_num` / $\sum$ `selected_num` |
| $\widehat{\mathrm{Cov}}_R(\beta)$ | `lec_phase_1_alpha_{β}` + `splits_phase_1` | $\sum$ `selected_num` / $N_{\mathrm{test}}$（$N_{\mathrm{test}}$ = 各 split test 集大小之和） |
| $\widehat{\mathrm{mFDR}}_k(\beta)$ | 该 $\beta$ 的 `lec_branch_a_disease_{k}_alpha_{α}` | pooled: $\sum$ `error_num` / $\sum$ `selected_num` |
| $\widehat{\mathrm{mFDR}}_{\mathrm{sys}}(\beta)$ | 该 $\beta$ 的全部文件（按 9.3 数据流逐样本计算） | $\sum S_{\mathrm{sys}} E_{\mathrm{sys}}$ / $\sum S_{\mathrm{sys}}$ |
| $\widehat{\mathrm{Cov}}_{\mathrm{sys}}(\beta)$ | 同上 | $\sum S_{\mathrm{sys}}$ / $N_{\mathrm{test}}$ |
| $\widehat{\mathrm{Ref}}_{\mathrm{sys}}(\beta)$ | 由 $\widehat{\mathrm{Cov}}_{\mathrm{sys}}$ 派生 | $1 - \widehat{\mathrm{Cov}}_{\mathrm{sys}}(\beta)$ |
| 路由污染率 $\widehat{\rho}_{k}^{\mathrm{route}}(\beta)$ | 该 $\beta$ 的 `branch_a_disease_{k}` + `splits_branch_a` | test 集中 `routing_correctness=0` 的比例 |
| 阈值变异性 $\lambda_R^{(b)}$ | `lec_phase_1_alpha_{β}` | `splits[].t_hat` 的均值与标准差 |
| 不可行频率 | `lec_phase_1_alpha_{β}` / `lec_branch_a_disease_{k}` | `t_hat is null` 的 split 数 / 总 split 数 |

**预期趋势**：$\beta$ 增大 → $\widehat{\mathrm{Cov}}_R$ 上升但路由污染可能增加；$\beta$ 减小 → 路由更安全但更多转人类。绘制各指标随 $\beta$ 的变化曲线。

##### 2. 专科风险敏感性（$\alpha$ 网格扫描）✅ 已完成

**已完成参数网格**：$\alpha \in \{0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.25, 0.5, 0.8\}$，$\beta$ 固定不变。

**已保存的文件**（每个 $\alpha$ 值一组，共 9 组）：

| 文件 | 说明 |
|------|------|
| `lec_branch_a_disease_{k}_alpha_{α}.json` | 该 $\alpha$ 下的专科校准结果（含 `t_hat`、`selected_num`、`error_num`、`base_error_rate`） |

> **关键优势**：上游文件（`splits_branch_a.json`、`branch_a_disease_{k}.json`、`lec_phase_1_alpha_{β}.json`）不随 $\alpha$ 变化（$\beta$ 固定），全部复用主实验文件。只需替换 `lec_branch_a_disease_{k}` 即可。

**指标计算方法**：对每个 $\alpha$ 值，只需加载新的 `lec_branch_a_disease_{k}_alpha_{α}.json`，其余文件复用主实验：

| 指标 | 所需文件 | 计算方法 |
|------|----------|----------|
| $\widehat{\mathrm{mFDR}}_k(\alpha)$ | `lec_branch_a_disease_{k}_alpha_{α}` | pooled: $\sum$ `error_num` / $\sum$ `selected_num` |
| $\widehat{\mathrm{Cov}}_k(\alpha)$ | 同上 + `splits_branch_a` | $\sum$ `selected_num` / $N_{\mathrm{test},k}^{\mathrm{route}}$ |
| $\widehat{\mathrm{mFDR}}_{\mathrm{sys}}(\alpha)$ | 主实验上游文件 + 新 `lec_branch_a`（按 9.3 数据流，仅替换 $\lambda_k$） | $\sum S_{\mathrm{sys}} E_{\mathrm{sys}}$ / $\sum S_{\mathrm{sys}}$ |
| $\widehat{\mathrm{Cov}}_{\mathrm{sys}}(\alpha)$ | 同上 | $\sum S_{\mathrm{sys}}$ / $N_{\mathrm{test}}$ |
| $\widehat{\mathrm{Ref}}_{\mathrm{sys}}(\alpha)$ | 由 $\widehat{\mathrm{Cov}}_{\mathrm{sys}}$ 派生 | $1 - \widehat{\mathrm{Cov}}_{\mathrm{sys}}(\alpha)$ |
| 阈值变异性 $\lambda_k^{(b)}$ | `lec_branch_a_disease_{k}_alpha_{α}` | `splits[].t_hat` 的均值与标准差 |
| 不可行频率 | `lec_branch_a_disease_{k}_alpha_{α}` | `t_hat is null` 的 split 数 / 总 split 数 |

**预期趋势**：$\alpha$ 增大 → $\widehat{\mathrm{Cov}}_k$ 上升但 $\widehat{\mathrm{mFDR}}_k$ 可能上升；$\alpha$ 减小 → $\widehat{\mathrm{mFDR}}_k$ 下降但更多转人类。绘制各指标随 $\alpha$ 的变化曲线。

##### 3. 校准大小分析

**目标**：评估系统在不同校准数据可用性下的稳健性。校准/测试划分比例网格为 **10/90, 20/80, 30/70, 40/60, 50/50**（calib 占比从小到大）。$\alpha$、$\beta$ 固定不变。

**预期分析视角**：即使在仅 10% 数据用于校准的极端不平衡设置下，系统的风险控制保证（$\widehat{\mathrm{mFDR}}_R \leq \beta$、$\widehat{\mathrm{mFDR}}_k \leq \alpha_k$）仍应成立。这一特性对于风险感知型医疗 AI 系统的可扩展部署至关重要。重点关注：(1) 阈值 $\lambda_R^{(b)}$、$\lambda_k^{(b)}$ 在小校准集下是否剧烈波动；(2) 不可行频率是否随校准集缩小而上升；(3) AI 管理率 $\widehat{\mathrm{Cov}}_{\mathrm{sys}}$ 是否因阈值保守化而下降。

**对每个划分比例，执行完整管线**：

| 步骤 | 操作 | 输入文件 | 输出文件 |
|------|------|----------|----------|
| 第二步 | 以新比例重新划分（仍按数据集为单位，每个数据集内部按比例切分 calib/test） | `phase_2_{dataset}.json`（获取数据集列表） | `splits_phase_1.json` |
| 第三步 | 重新校准路由器 | `phase_1.json` + 新 `splits_phase_1.json` | `lec_phase_1_alpha_{β}.json` |
| 第四步 | 重新过滤 Branch A/B | 新 `splits_phase_1.json` + `phase_1.json` + 新 `lec` | `splits_branch_a.json` + `splits_branch_b.json` |
| 第五步 | 重新聚合疾病文件 | `phase_1.json` + `phase_2_{dataset}.json` + 新 `splits` | `branch_a/b_disease_{k}.json` |
| 第六步 | 重新校准专科 | `branch_a/b_disease_{k}.json` + 新 `splits` | `lec_branch_a/b_disease_{k}_alpha_{α}.json` |
| 计算 | 用 9.1 方法计算，重点关注阈值变异性、不可行频率与 AI 管理率 | 上述全部新文件 | — |

> **注意**：需要完整重新运行管线（第二步到第六步），但原始模型输出 `phase_1.json`、`phase_2_{dataset}.json` 不变。默认实验使用 50/50 划分，其余比例为敏感性分析。

---

#### 基线（第八章，部分）

##### 4. 仅专科拒绝 (Specialist-Only Rejection)

**目标**：路由器不做拒绝（$S_R=1, \lambda_R=+\infty$），仅靠专科阈值做不确定性拒绝。

**操作步骤**：

| 步骤 | 操作 | 输入文件 | 输出文件 |
|------|------|----------|----------|
| 跳过第三步 | 不需要路由器校准（等价于 $\lambda_R=+\infty$，全部样本 $S_R=1$） | — | — |
| 跳过 Branch B | 不存在路由器拒绝的样本 | — | — |
| 新中间步 | 构建全样本疾病聚合文件 | `phase_1.json`（取 `predicted_disease` 分流、`uncertainty` u₁）+ `phase_2_{dataset}.json`（取 `uncertainty` u₂、`correctness`） | `all_disease_{k}.json` |
| 第六步（变体） | 在全样本上执行 LEC 校准 | `all_disease_{k}.json` + `splits_phase_1.json`（用原始划分） | `lec_all_disease_{k}_alpha_{α}.json` |
| 计算 | 见下方特殊计算方式 | 上述文件 | — |

**`all_disease_{k}.json` 格式**（与 `branch_a_disease_{k}.json` 相同）：
```json
{
    "sample_001": {
        "uncertainty": "λ₁·u₁ + λ₂·u₂",
        "routing_correctness": 1,
        "pred_correctness": 1,
        "correctness": 1
    }
}
```

**指标计算差异**：
- $\widehat{\mathrm{Cov}}_R = 100\%$（路由器全部接受）
- $\widehat{\mathrm{mFDR}}_R$ = `phase_1` 的 `base_error_rate`（不拒绝即等于全部错误率）
- $\widehat{\mathrm{Cov}}_{\mathrm{sys}} = \widehat{\mathrm{Cov}}_{\mathrm{spec}\mid R}$（系统覆盖率 = 专科覆盖率）
- $\widehat{\mathrm{Ref}}_R = 0$，$\widehat{\mathrm{Ref}}_{\mathrm{sys}} = \widehat{\mathrm{Ref}}_{\mathrm{spec}}$
- 其余指标（$\widehat{\mathrm{mFDR}}_k$、$\widehat{\mathrm{mFDR}}_{\mathrm{sys}}$、路由污染等）从 `lec_all_disease_{k}` 计算

##### 5. 共享阈值 (Shared-Threshold)

**目标**：所有 $K$ 个专科共用一个阈值 $\lambda_{\mathrm{shared}}$，而非按疾病分别校准。

**操作步骤**：

| 步骤 | 操作 | 输入文件 | 输出文件 |
|------|------|----------|----------|
| 新中间步 | 池化所有疾病文件 | `branch_a_disease_{0}.json` ~ `branch_a_disease_{9}.json`（合并所有样本，保留 `uncertainty`、`correctness` 字段） | `pooled_branch_a_disease.json` |
| 第六步（变体） | 在池化文件上执行一次 LEC 校准 | `pooled_branch_a_disease.json` + `splits_branch_a.json` | `lec_pooled_branch_a_alpha_{α}.json` |
| 计算 | 见下方特殊计算方式 | 上述文件 + `branch_a_disease_{k}.json` | — |

**`pooled_branch_a_disease.json` 格式**：
```json
{
    "sample_001": {
        "uncertainty": 0.527,
        "correctness": 1
    }
}
```

**指标计算差异**：
- 从 `lec_pooled_branch_a` 获取唯一的 $\lambda_{\mathrm{shared}}$（每个 split 一个值）
- 对每个专科 $k$，用同一个 $\lambda_{\mathrm{shared}}$ 判定 $S_k = (\text{uncertainty} \leq \lambda_{\mathrm{shared}})$
- $\widehat{\mathrm{mFDR}}_k$、$\widehat{\mathrm{Cov}}_k$ 等按疾病分别计算分子分母，但阈值统一
- 对比主系统（按疾病分别校准）vs 此基线（共享阈值），观察专科特异校准的增益

##### 6. 先知式干净校准 (Oracle-Clean Calibration)

**目标**：仅用 `routing_correctness=1` 的样本校准专科阈值（模拟完美路由），再应用于全部路由流。这是不可部署的参考上界。

**操作步骤**：

| 步骤 | 操作 | 输入文件 | 输出文件 |
|------|------|----------|----------|
| 新中间步 | 过滤干净样本 | `branch_a_disease_{k}.json`（仅保留 `routing_correctness=1` 的样本） | `clean_branch_a_disease_{k}.json` |
| 新中间步 | 过滤划分 | `splits_branch_a.json` + `clean_branch_a_disease_{k}.json`（calib 和 test 集均过滤为干净子集） | `splits_clean_branch_a_{k}.json` |
| 第六步（变体） | 在干净子集上执行 LEC 校准 | `clean_branch_a_disease_{k}.json` + `splits_clean_branch_a_{k}.json` | `lec_clean_branch_a_disease_{k}_alpha_{α}.json` |
| 计算 | 见下方特殊计算方式 | 上述文件 + 原始 `branch_a_disease_{k}.json` | — |

**`clean_branch_a_disease_{k}.json` 格式**（与原文件相同，但仅含路由正确的样本）：
```json
{
    "sample_001": {
        "uncertainty": 0.527,
        "routing_correctness": 1,
        "pred_correctness": 1,
        "correctness": 1
    }
}
```

**指标计算差异**：
- 从 `lec_clean_branch_a_disease_{k}` 获取 $\lambda_k^{\mathrm{clean}}$（干净校准阈值）
- $S_k$ 判定仍应用于**全部** `branch_a_disease_{k}.json` 样本（包括 `routing_correctness=0` 的样本）
  - 即：$S_k = \mathbf{1}\{\text{uncertainty} \leq \lambda_k^{\mathrm{clean}}\}$，在全部路由流样本上计算
- 这是"先知校准 + 现实应用"的设计：校准用干净数据，应用在真实路由流上
- $\widehat{\mathrm{mFDR}}_k$ 从全部样本计算，对比主系统可观察路由污染对风险控制的影响
- 预期结果：干净校准的阈值更宽松（因校准集中无路由错误样本），但应用在真实路由流上时 $\widehat{\mathrm{mFDR}}_k$ 可能超标

### 9.3 计算复杂指标时的数据流总结

对于需要逐样本计算的复杂指标（如 $\widehat{\mathrm{mFDR}}_{\mathrm{sys}}$、$\widehat{U}$、路由污染分析等），计算流程如下：

```
对每个 split b:
  1. 从 splits_phase_1.json 获取 test 样本列表
  2. 从 phase_1.json 获取每个 test 样本的 u₁, predicted_disease(k̂), correctness
  3. 从 lec_phase_1 获取该 split 的 t_hat(λ_R)
  4. 计算 S_R = (u₁ ≤ λ_R)
  5. 对 S_R=1 的样本：
     a. 从 branch_a_disease_{k̂} 获取 uncertainty, routing_correctness, pred_correctness, correctness
     b. 从 lec_branch_a_disease_{k̂} 获取该 split 的 t_hat(λ_k)
     c. 计算 S_k = (uncertainty ≤ λ_k)
     d. 计算 S_sys = S_R · S_k
     e. 计算 E_sys = 1 - correctness
  6. 对 S_R=0 的样本（Branch B）：
     S_sys = 0，计入人类转诊
  7. 汇总该 split 的各类分子分母

跨 split 汇总：
  pooled = Σ(分子) / Σ(分母)
  bootstrap: 重采样 split → 重算 pooled → 取 2.5%/97.5% 分位数
```

### 9.4 总结

| 类别 | 指标数量 | 可直接计算 | 需额外运行 | 无法计算 |
|------|----------|-----------|-----------|----------|
| 风险控制（第一章） | 9 | 9 ✅ | 0 | 0 |
| 效率/覆盖率（第二章） | 6 | 6 ✅ | 0 | 0 |
| 临床效用（第三章） | 4 | 4 ✅ | 0 | 0 |
| 路由污染（第四章） | 8 | 8 ✅ | 0 | 0 |
| 校准可行性（第五章） | 3 | 3 ✅ | 0 | 0 |
| 敏感性分析（第六章） | 3 | 2 ✅ | 1 ⚠️（校准大小分析） | 0 |
| 统计报告（第七章） | 4 | 4 ✅ | 0 | 0 |
| 基线（第八章） | 5 | 2 ✅ | 3 ⚠️（需额外校准） | 0 |
| **合计** | **42** | **38 ✅** | **4 ⚠️** | **0** |

> **结论**：当前 JSON 管线可覆盖论文全部 42 项指标中的 38 项直接计算（含已完成的 β 扫描与 α 扫描）。4 项仍需额外运行：校准大小分析（需重跑完整管线）和 3 个基线（需不同校准策略）。不可行频率通过 `t_hat is null` 判定，校准代码已保证该字段的二值语义。
