#!/bin/bash

# ============================================================
# 批量推理所有数据集，生成 phase_2 JSON
#
# 用法:
#   bash run/inference_all.sh          # 推理全部（自动查找权重）
#   bash run/inference_all.sh 0 2      # 仅推理疾病0和疾病2的数据集
# ============================================================

GPU_IDS="0"
BATCH_SIZE=32
NUM_WORKERS=4
IMG_SIZE=224
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

FILTER_DISEASES="$@"

for entry in "${DATASETS[@]}"; do
    read -r DATASET NUM_CLASSES DISEASE_ID <<< "$entry"

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

    # 自动查找最新的 checkpoint
    TRIALS_DIR="trials/${DATASET}"
    RESUME_PATH=""

    if [ -d "$TRIALS_DIR" ]; then
        RESUME_PATH=$(find "$TRIALS_DIR" -name "*.pth.tar" -type f -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2)
        # macOS 兼容
        if [ -z "$RESUME_PATH" ]; then
            RESUME_PATH=$(find "$TRIALS_DIR" -name "*.pth.tar" -type f | head -1)
        fi
    fi

    if [ -z "$RESUME_PATH" ]; then
        echo "[SKIP] ${DATASET}: no checkpoint found in ${TRIALS_DIR}"
        continue
    fi

    echo ""
    echo "============================================================"
    echo "  Inference: ${DATASET} (${NUM_CLASSES} classes)"
    echo "  Checkpoint: ${RESUME_PATH}"
    echo "============================================================"

    python run_inference_phase2.py \
        --dataset ${DATASET} \
        --num_classes ${NUM_CLASSES} \
        --resume_path "${RESUME_PATH}" \
        --gpu_id ${GPU_IDS} \
        --batch_size ${BATCH_SIZE} \
        --num_workers ${NUM_WORKERS}

    if [ $? -ne 0 ]; then
        echo "[ERROR] Inference failed for ${DATASET}, continuing..."
    fi
done

echo ""
echo "============================================================"
echo "  All inference complete!"
echo "  Phase 2 JSON files saved to res/json/"
echo "============================================================"
