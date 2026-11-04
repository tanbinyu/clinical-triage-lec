#!/bin/bash

# ============================================================
# 推理 / 测试
# ============================================================

DATASET="LungHist700"
NUM_CLASSES=7
GPU_IDS="0"
BATCH_SIZE=32
NUM_WORKERS=4
IMG_SIZE=224
MANUAL_SEED=1

RESUME_PATH="trials/LungHist700/auto/adapter_epoch_99.pth.tar"

python inference.py \
    --dataset ${DATASET} \
    --num_classes ${NUM_CLASSES} \
    --gpu_id ${GPU_IDS} \
    --batch_size ${BATCH_SIZE} \
    --num_workers ${NUM_WORKERS} \
    --img_size ${IMG_SIZE} \
    --manual_seed ${MANUAL_SEED} \
    --resume_path "${RESUME_PATH}" \
    --pin_memory
