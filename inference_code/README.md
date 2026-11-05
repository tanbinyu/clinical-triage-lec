# NatureHelp：10 个真实数据集的两阶段推理流水线

本项目完成以下完整链路：原始数据校验与必要解压 → 模型专用预处理 → 固化样本清单 → 患者互斥的 train/val/downstream 划分 → downstream 内 CR/CS/T 划分 → 每数据集独立 adapter 训练 → UniMed-CLIP 路由推理 → 路由目标专科模型推理 → 11 份 JSON 的一致性校验与原子发布。

最终文件仍严格保持既有格式，不添加 provenance 或“synthetic”等字段：

- `phase_1.json`：10 类路由结果；
- `phase_2_{dataset}.json`：10 个数据集各一份专科结果。

## 数据集和模型

| disease id | 数据集 | phase-2 骨干 | 任务与真值粒度 |
|---:|---|---|---|
| 0 | LungHist700 | UNI | 3 类，图像级 |
| 1 | BreakHis | UNI | 2 类，患者级 |
| 2 | COVID_CT | DINOv2-large | 2 类，患者级 |
| 3 | LDFC | DINOv2-large | 2 类，结节级；原始 0=良性、1/2=恶性 |
| 4 | ImageCHD | DINOv2-large | 16 类多标签，volume 级 |
| 5 | ChestX_ray14 | RAD-DINO | 14 类多标签，图像级；No Finding=`[]` |
| 6 | CBIS-DDSM | RAD-DINO | 2 类，病灶级 |
| 7 | BUS-BRA | USFM | 2 类，患者级 |
| 8 | PU2756 | USFM | 2 类，患者级 |
| 9 | ACDC | CineMA | 5 类，患者级 |

## 工程结构

```text
configs/default.toml       路径、种子、checkpoint ID 与两层划分配置
environment.yml           Python/CUDA/依赖的参考锁定环境
prepare_datasets.py        BreakHis/ImageCHD 的幂等安全解压
train_adapter.py           单数据集训练、验证、早停和断点续训
run_inference.py           doctor/prepare/split/run/validate/evaluate 命令
run_full_pipeline.py       端到端总入口
naturehelp_inference/
  datasets.py              数据集标签和 patient ID 解析
  preprocessing.py         CineMA 官方 ACDC 预处理
  manifest.py              稳定 sample_id 与 JSONL manifest
  splits.py                患者互斥且类别平衡的持久化划分
  media.py                 ZIP/TAR/DICOM/NIfTI 媒体读取
  models.py                六个模型及 adapter 注册/严格权重校验
  metrics.py               多分类和多标签指标、阈值校准
  pipeline.py              两阶段批量推理与 staging 发布
  output.py                JSON 契约、跨文件一致性与回滚
  evaluation.py            phase 1、phase 2、级联系统准确率
work/                      manifest、split、缓存、checkpoint、staging
```

## 安装和预检

推荐使用锁定的 Python 3.11、PyTorch 2.5.1、CUDA 12.4 环境：

```powershell
Set-Location inference_code
conda env create -f environment.yml
conda activate naturehelp-paper
python run_inference.py doctor
```

`requirements.txt` 也锁定了 Python 包版本，但 GPU 主机应优先使用
`environment.yml`，避免 pip 自动选择与本机 CUDA 不一致的 PyTorch wheel。

训练前可不要求 adapter 已存在；推理前应执行：

```powershell
python run_inference.py doctor --require-adapters
```

`doctor` 同时报告依赖、原始数据完整性、骨干权重和 adapter 缺失，不会静默改用随机权重。

## 推荐分阶段流程

1. 解压需要预处理的归档：

   ```powershell
   python prepare_datasets.py all
   ```

2. 固化清单和患者级划分：

   ```powershell
   python run_inference.py prepare --rebuild-split
   python run_inference.py split
   python run_inference.py partition --reference
   ```

   此步骤也会调用 CineMA 官方 ACDC 预处理：1×1×10 mm 重采样、基于 LV 的 192×192 中心裁剪和强度归一化。原始 ACDC 文件不会被修改。

   固定层按患者使用 seed `20261019` 分成 35% adapter train、5% adapter
   validation 和 60% downstream pool。第二层只在 downstream pool 内划分：
   参考划分用 seed `20261019`；重复实验用 seed `20260929 + repetition`；
   每个来源最多抽取 1,850 名患者，再按来源分层分配
   CR/CS/T=30%/30%/40%，并精确匹配全局人数 1,659/1,659/2,211。
   `partition` 文件同时保存每名患者的角色和在角色确定后等概率选中的一个原生 decision。

3. 训练每个数据集自己的 adapter：

   ```powershell
   python train_adapter.py ACDC --epochs 100 --patience 10
   ```

   训练只使用 `train`；`val` 仅用于 early stopping、最佳 checkpoint 和多标签逐类阈值校准；`downstream` 完全隔离。默认 AdamW、lr=1e-3、weight decay=1e-4、batch=256、最多 100 epochs，训练种子为 `20261019`。`--resume` 恢复最后一个训练 checkpoint，`--rebuild-cache` 明确重建冻结特征。

4. 生成全量 11 份 JSON：

   ```powershell
   python run_inference.py run --scope all --phase all
   python run_inference.py validate-output
   python run_inference.py evaluate
   python run_inference.py provenance --hash-checkpoints
   ```

   `--scope all` 生成论文所述的冻结全量决策导出。统计分析必须按
   `work/downstream_partitions/*.json` 只选择 downstream 患者，不能把 adapter
   train/val 中的预测当作泛化性能。若只需要隔离池，可使用 `--scope downstream`。

   推理进度默认按样本写入 `work/runs/progress/*.sqlite`，中断后同一数据、权重和 scope 会自动续跑；需要明确忽略旧进度时使用 `--no-resume`。

## 一条命令运行全链路

```powershell
python run_full_pipeline.py --scope all
```

该命令默认执行解压、manifest/split、10 个 adapter 训练、两阶段推理、发布和级联指标计算。已有合格 adapter 时可加 `--skip-training`。只选部分数据集时只生成相应 phase 2，避免用不完整数据覆盖全局 `phase_1.json`。

## 可复现性和数据安全

- `sample_id` 在完整、稳定排序的 manifest 中一次性生成；筛选 train/val/downstream 后不会重新编号。
- split 文件记录 manifest SHA-256、算法版本、随机种子和每个 patient 的唯一归属；原始数据变化会使运行失败，必须显式重建。
- patient 是不可拆分单位；患者级数据还会检查同一 patient 是否出现冲突标签。图像级/病灶级数据允许同一患者的不同样本有不同标签。
- CBIS-DDSM 只接受四份官方 pathology CSV，并严格使用 `cropped image file path` 匹配 DICOM；不做患者名模糊兜底。
- 特征缓存绑定 manifest、split、骨干与预处理版本；过期缓存不会被复用。
- 训练保存 best adapter、last checkpoint 和验证指标；metadata 包含数据/split digest、随机种子、优化器超参数、基础 checkpoint ID。推理严格检查 checkpoint metadata 与模型结构。
- `provenance` 记录 Python、全部依赖版本、三个随机种子、模型 checkpoint ID、源码 commit 和文件大小；`--hash-checkpoints` 额外计算 SHA-256。
- 11 个文件先写入 staging，全部通过字段、样本集合、patient ID 和 disease ID 的跨文件检查后才发布。旧输出移入 `_previous/<时间>/`；发布失败则自动恢复。

## 指标语义

- phase 1：10 类路由 softmax 熵与路由正确率；
- phase 2 多分类：softmax 熵；
- phase 2 多标签：各标签 Bernoulli 熵之和，`correctness` 采用标签集合完全匹配；
- phase 2 根据 `phase_1.predicted_disease` 实际加载目标 adapter；其 `correctness`
  只有在路由正确且目标专科预测正确时才为 1；
- `specialist_accuracy_given_correct_route` 仅在路由正确的样本中计算；
- 多标签 uncertainty 是各标签 Bernoulli entropy **之和**，不是除以标签数后的均值。

## 跨模态错误路由的边界

普通 2D/CT/X-ray/超声骨干可接收由统一媒体层产生的 2D 图像或切片；CineMA
则只接受经过官方 ACDC 流程得到的 ED/ES cine MRI 张量。若非 ACDC 样本被路由到
CineMA，程序会停止并给出样本 ID，不会用重复切片伪造 cine MRI，也不会冒充完成了
目标模型推理。若实际实验需要继续，必须先在研究方案中定义并验证跨模态拒绝策略。

## Required local assets

- UNI weights are access-controlled and must be obtained from the official
  provider before running the corresponding pathways.
- CBIS-DDSM requires the four official case-description CSV files and the
  complete DICOM collection referenced by their cropped-lesion paths.
- ImageCHD multipart extraction requires 7-Zip.
- CineMA requires the documented ACDC preprocessing outputs.
- Trained adapters and foundation-model checkpoints belong under the local
  `weights/` directory defined by `configs/default.toml`.

The pipeline never substitutes random weights, guesses missing labels or
silently changes a frozen cohort. Missing or inconsistent prerequisites cause
an explicit validation error.

