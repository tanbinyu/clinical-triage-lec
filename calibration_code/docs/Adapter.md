## 简化后的统一方案

每个输入样本都按单张 2D 图像处理：

```text
样本 x
  ↓
对应基础模型 Fₘ（冻结）
  ↓
特征 z
  ↓
数据集专属 Adapter A_d（训练）
  ↓
数据集专属分类头 H_d（训练）
  ↓
预测类别、类别概率、预测熵
```

其中：

- 病理：UNI
- CT：DINOv2-large
- 胸部 X 光：RAD-DINO
- 超声：USFM
- 心脏 MRI：CineMA

基础模型参数全部冻结。每个数据集训练自己的 Adapter 与分类头，不合并不同数据集的标签空间。

## Adapter

使用统一的残差 bottleneck Adapter：

\[
A_d(z)=z+W_{up}\,\mathrm{GELU}(W_{down}\,\mathrm{LN}(z))
\]

训练参数只有：

- Adapter 参数 \(W_{down},W_{up}\)
- 该数据集的分类头 \(H_d\)

建议默认 bottleneck 维度设为 64。

因此每个数据集保存一组独立参数：

```text
{foundation_model}_{dataset}_adapter
{foundation_model}_{dataset}_head
```

例如：

```text
UNI_LungHist700_adapter + UNI_LungHist700_head
UNI_TCGA_BRCA_adapter + UNI_TCGA_BRCA_head
USFM_BUSI_adapter + USFM_BUSI_head
USFM_BUS_BRA_adapter + USFM_BUS_BRA_head
```

即使它们属于同一种疾病，也不共享分类头、不硬合标签。

## 训练

单标签二分类/多分类：

\[
\hat{y}=H_d(A_d(F_m(x)))
\]

使用交叉熵损失：

\[
\mathcal{L}=-\sum_{k=1}^{K} y_k\log p_k
\]

ChestX-ray14 这类多标签数据集使用每标签二元交叉熵。

## 预测熵

对于单标签任务，分类头输出 softmax 概率：

\[
p_k=\frac{e^{l_k}}{\sum_{j=1}^{K}e^{l_j}}
\]

预测类别：

\[
\hat{y}=\arg\max_k p_k
\]

预测熵：

\[
u(x)=-\sum_{k=1}^{K}p_k\log p_k
\]

为便于记录，可同时保存归一化熵：

\[
u_{\text{norm}}(x)=\frac{u(x)}{\log K}
\]

多标签任务对每个标签概率 \(p_\ell\) 计算二元熵后取平均：

\[
u(x)=\frac{1}{L}\sum_{\ell=1}^{L}
[-p_\ell\log p_\ell-(1-p_\ell)\log(1-p_\ell)]
\]

## 每个样本的输出

```json
{
  "sample_id": "sample_001",
  "foundation_model": "USFM",
  "adapter_id": "USFM_BUSI",
  "dataset_id": "BUSI",
  "label_space_id": "BUSI_3class",
  "pred": "benign",
  "probabilities": [0.03, 0.91, 0.06],
  "pred_entropy": 0.36,
  "normalized_pred_entropy": 0.33,
  "correctness": true
}
```

核心原则：**一个数据集对应一个 Adapter、一个分类头、一个标签空间；每个样本输出预测类别、概率和预测熵。**











## 具体定义

`数据集专属 Adapter A_d` **不是 linear head**。

它是放在冻结基础模型输出之后的、两层残差 MLP；分类头是其后的独立单层 Linear。

```text
图像 x
  ↓
冻结基础模型 Fₘ
  ↓
pooled feature h ∈ Rᴰ
  ↓
残差 MLP Adapter A_d
  ↓
adapted feature h' ∈ Rᴰ
  ↓
Linear Head H_d
  ↓
K_d 个 logits
```

对于当前这些以 ViT 为主的基础模型，输入特征统一取最终层的全局 pooled feature；若模型没有显式 pooled output，则取最终层 CLS token。

\[
h=F_m(x)\in \mathbb{R}^{D_m}
\]

\[
a=W_{up}\left[\mathrm{Dropout}(\mathrm{GELU}(W_{down}\cdot \mathrm{LN}(h)+b_{down}))\right]+b_{up}
\]

\[
h'=\mathrm{LN}(h+\mathrm{Dropout}(a))
\]

\[
\mathrm{logits}=W_{\text{head}}h'+b_{\text{head}}
\]

其中：

- `W_down`: \(D_m \rightarrow 64\)
- `W_up`: \(64 \rightarrow D_m\)
- `Dropout = 0.1`
- `Linear Head`: \(D_m \rightarrow K_d\)
- \(K_d\) 是该数据集的实际标签数

`W_up` 初始化为零，使 Adapter 在训练开始时近似恒等映射；分类头使用 Xavier 初始化。

## 每个数据集训练什么

| 模块 | 是否训练 |
|---|---|
| 对应基础模型 \(F_m\) | 否，完全冻结 |
| LayerNorm（基础模型内部） | 否 |
| 数据集专属 Adapter \(A_d\) | 是 |
| 数据集专属 Linear Head \(H_d\) | 是 |

例如 BUSI 与 BUS-BRA 使用同一个 USFM 骨干，但分别训练：

```text
USFM_BUSI_adapter + USFM_BUSI_head
USFM_BUS_BRA_adapter + USFM_BUS_BRA_head
```

不共享 Adapter，也不共享 Head。

## 固定训练配置

```yaml
optimizer: AdamW

adapter:
  learning_rate: 3e-4
  weight_decay: 1e-4

linear_head:
  learning_rate: 1e-3
  weight_decay: 1e-4

betas: [0.9, 0.999]
gradient_clip_norm: 1.0

batch_size:
  physical: 32
  effective: 64
  gradient_accumulation: 2

scheduler:
  type: cosine_decay
  warmup_epochs: 5

max_epochs: 50
min_epochs_before_early_stop: 10
early_stopping_patience: 8
```

显存不足时用 `physical batch=16`、`gradient accumulation=4`，保持有效 batch size 为 64。

## 损失与 checkpoint

- 二分类、多分类：类别加权交叉熵；
- 多标签数据集：`BCEWithLogitsLoss`，每个标签的 `pos_weight` 只由训练集计算；
- 不同时使用过采样和很强的类别权重，二者选一种作为不均衡处理方式。

checkpoint 选择规则：

1. 每个数据集单独保留验证集；
2. 以验证集的 **macro-AUROC** 最高者作为 `best_checkpoint`；
3. 若 macro-AUROC 持平，选择验证损失更低者；
4. 连续 8 个 epoch 未提升则停止；
5. 最终只使用 `best_checkpoint` 为每个样本输出 logits、预测类别、概率和预测熵。

每个 checkpoint 应保存：

```json
{
  "foundation_model": "USFM",
  "dataset_id": "BUSI",
  "feature_dim": 1024,
  "adapter_bottleneck_dim": 64,
  "num_classes": 3,
  "best_epoch": 23,
  "best_val_macro_auc": 0.914,
  "class_mapping": ["normal", "benign", "malignant"]
}
```

因此最终主方法可明确写成：**使用冻结的对应视觉基础模型提取 pooled feature，训练一个 bottleneck=64 的数据集专属残差 MLP Adapter，并以数据集专属单层 Linear Head 完成分类。**