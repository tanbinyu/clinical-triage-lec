#!/bin/bash

# ============================================================
# 批量训练所有数据集的 adapter (Linear Probe)
# 大模型编码器根据数据集自动选择
#
# 用法:
#   bash run/train_all.sh          # 训练全部
#   bash run/train_all.sh 0 2      # 仅训练疾病0和疾病2的数据集
# ============================================================

GPU_IDS="0"
BATCH_SIZE=32
NUM_WORKERS=4
LR=0.001
IMG_SIZE=224
EPOCHS=100
VAL_EVERY=1
MANUAL_SEED=1

# 数据集配置: "数据集名 类别数 疾病ID"
DATASETS=(
    "LungHist700 7 0"
    "TCGA_luadlusc 2 0"
    "BRACS 7 1"
    "TCGA_BRCA 2 1"
    "COVID_CT 2 2"
    "LIDC_IDRI 2 2"
    "LNDb 2 2"
    "CT_Heart 2 3"
    "Chest_X_ray_14 15 4"
    "CheXpert 14 4"
    "BUSI 3 5"
    "BUS_BRA 2 5"
    "TMED 3 6"
    "HMC_QU 2 6"
    "ACDC 5 7"
    "MMs 5 7"
)

# 如果有参数，只训练指定疾病ID的数据集
FILTER_DISEASES="$@"

for entry in "${DATASETS[@]}"; do
    read -r DATASET NUM_CLASSES DISEASE_ID <<< "$entry"

    # 如果指定了过滤条件，跳过不匹配的
    if [ -n "$FILTER_DISEASES" ]; then
        MATCH=0
        for d in $FILTER_DISEASES; do
            if [ "$d" = "$DISEASE_ID" ]; then
                MATCH=1
                break
            fi
        done
        if [ "$MATCH" -eq 0 ]; then
            continue
        fi
    fi

    echo ""
    echo "============================================================"
    echo "  Training: ${DATASET} (${NUM_CLASSES} classes, disease=${DISEASE_ID})"
    echo "============================================================"

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

    if [ $? -ne 0 ]; then
        echo "[ERROR] Training failed for ${DATASET}, continuing..."
    fi
done

echo ""
echo "============================================================"
echo "  All training complete!"
echo "============================================================"
