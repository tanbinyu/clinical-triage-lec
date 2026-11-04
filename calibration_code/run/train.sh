#!/bin/bash

# ============================================================
# 训练 adapter (Linear Probe)
# 大模型编码器自动根据数据集选择，也可通过 --model 手动指定
#
# 可用模型: UNI, CT_CLIP, CheXFusion, USFM, BrainMVP
# 可用数据集: LungHist700, TCGA_luadlusc, BRACS, TCGA_BRCA,
#             COVID_CT, LIDC_IDRI, LNDb, CT_Heart,
#             Chest_X_ray_14, CheXpert,
#             BUSI, BUS_BRA, TMED, HMC_QU,
#             ACDC, MMs
# ============================================================

DATASET="LungHist700"
NUM_CLASSES=7
GPU_IDS="0"
BATCH_SIZE=32
NUM_WORKERS=4
LR=0.001
IMG_SIZE=224
EPOCHS=100
VAL_EVERY=1
MANUAL_SEED=1

python train.py \
    --dataset ${DATASET} \
    --num_classes ${NUM_CLASSES} \
    --gpu_id ${GPU_IDS} \
    --batch_size ${BATCH_SIZE} \
    --num_workers ${NUM_WORKERS} \
    --learning_rate ${LR} \
    --img_size ${IMG_SIZE} \
    --n_epochs ${EPOCHS} \
    --val_every ${VAL_EVERY} \
    --manual_seed ${MANUAL_SEED} \
    --pin_memory
