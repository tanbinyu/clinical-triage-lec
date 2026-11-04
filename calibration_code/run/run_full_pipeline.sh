#!/bin/bash

# ============================================================
# 完整实验流程一键运行
#
# 步骤:
#   1. 训练所有数据集的 adapter
#   2. 推理所有数据集，生成 phase_2 JSON
#   3. 运行 Router 推理（需要 open_clip）
#   4. 运行 LEC 四步校准实验
#
# 用法:
#   bash run/run_full_pipeline.sh              # 完整流程
#   bash run/run_full_pipeline.sh --skip-train # 跳过训练（已有权重时）
#   bash run/run_full_pipeline.sh --simplified # 简化版（无 Router）
# ============================================================

set -e

SKIP_TRAIN=0
SIMPLIFIED=0

for arg in "$@"; do
    case $arg in
        --skip-train) SKIP_TRAIN=1 ;;
        --simplified) SIMPLIFIED=1 ;;
    esac
done

echo "████████████████████████████████████████████████████████████"
echo "█  LEC 两阶段两分支 mFDR 控制 — 完整实验流程"
echo "████████████████████████████████████████████████████████████"

# Step 1: 训练
if [ "$SKIP_TRAIN" -eq 0 ]; then
    echo ""
    echo "[Step 1] 训练所有 adapter..."
    bash run/train_all.sh
else
    echo ""
    echo "[Step 1] 跳过训练 (--skip-train)"
fi

# Step 2: 推理
echo ""
echo "[Step 2] 推理所有数据集，生成 phase_2 JSON..."
bash run/inference_all.sh

# Step 3 & 4: LEC 实验
if [ "$SIMPLIFIED" -eq 1 ]; then
    echo ""
    echo "[Step 3] 运行简化版 LEC 实验 (仅用 phase_2, 无 Router)..."
    python -m calibration.run_all --mode simplified
else
    echo ""
    echo "[Step 3] 运行完整版 LEC 四步实验..."
    python -m calibration.run_all --mode full
fi

echo ""
echo "████████████████████████████████████████████████████████████"
echo "█  实验流程完毕!"
echo "█  结果: res/json/lec_results/"
echo "█  图表: res/res_img/lec_plots/"
echo "████████████████████████████████████████████████████████████"
