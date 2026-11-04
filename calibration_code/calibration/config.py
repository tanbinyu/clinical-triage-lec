"""
全局配置：疾病-数据集映射、Router参数、LEC实验参数等。
"""

# ============================================================
# 疾病-数据集映射（一个疾病对应一个专病模型，可对应多个数据集）
# ============================================================
DISEASE_TO_DATASETS = {
    0: ["LungHist700", "TCGA_luadlusc"],       # 肺癌(病理)
    1: ["BRACS", "TCGA_BRCA"],                 # 乳腺癌(病理)
    2: ["COVID_CT", "LIDC_IDRI", "LNDb"],      # 新冠/肺部CT
    3: ["CT_Heart"],                           # 心脏CT
    4: ["Chest_X_ray_14", "CheXpert"],         # 胸部X光疾病
    5: ["BUSI", "BUS_BRA"],                    # 乳腺超声
    6: ["TMED", "HMC_QU"],                     # 心脏超声
    7: ["ACDC", "MMs"],                        # 心脏MRI
}

DISEASE_NAMES = {
    0: "lung_cancer_pathology",
    1: "breast_cancer_pathology",
    2: "lung_ct",
    3: "cardiac_ct",
    4: "chest_xray",
    5: "breast_ultrasound",
    6: "cardiac_ultrasound",
    7: "cardiac_mri",
}

# 反向映射：数据集名 → 疾病编号
DATASET_TO_DISEASE = {}
for disease_id, datasets in DISEASE_TO_DATASETS.items():
    for dataset_name in datasets:
        DATASET_TO_DISEASE[dataset_name] = disease_id

NUM_DISEASES = len(DISEASE_TO_DATASETS)  # K=8

# ============================================================
# 疾病对应的基础 Encoder
# ============================================================
DISEASE_TO_ENCODER = {
    0: "UNI",
    1: "UNI",
    2: "CT_CLIP",
    3: "CT_CLIP",
    4: "CheXFusion",
    5: "USFM",
    6: "USFM",
    7: "BrainMVP",
}

# ============================================================
# LEC 实验参数
# ============================================================
LEC_DEFAULTS = {
    "split_ratio": 0.5,          # calibration set 占比
    "num_splits": 500,           # multi-split 重复次数
    "alpha_scan_routing": [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5],
    "alpha_scan_branch": [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5],
}

# ============================================================
# Fused Score 权重参数
# ============================================================
LAMBDA1 = 0.5  # routing uncertainty 权重

# λ2 按疾病可不同（因为不同疾病类别数不同，entropy范围不同）
LAMBDA2_PER_DISEASE = {k: 0.5 for k in range(NUM_DISEASES)}

# ============================================================
# 文件路径
# ============================================================
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JSON_DIR = os.path.join(PROJECT_ROOT, "res", "json")
RESULTS_DIR = os.path.join(JSON_DIR, "lec_results")
RESULTS_BRANCH_A_DIR = os.path.join(RESULTS_DIR, "stage2_branch_a")
RESULTS_BRANCH_B_DIR = os.path.join(RESULTS_DIR, "stage2_branch_b")

# Router 推理结果
ROUTER_PHASE1_PATH = os.path.join(JSON_DIR, "router_phase_1.json")

# 确保输出目录存在
for directory in [RESULTS_DIR, RESULTS_BRANCH_A_DIR, RESULTS_BRANCH_B_DIR]:
    os.makedirs(directory, exist_ok=True)


def get_phase2_path(dataset_name):
    """获取指定数据集的 phase_2 JSON 文件路径"""
    return os.path.join(JSON_DIR, f"{dataset_name}_phase_2.json")


def get_branch_a_path(disease_id):
    """获取 Branch A 某疾病的 fused score JSON 路径"""
    return os.path.join(JSON_DIR, f"branch_a_disease_{disease_id}.json")


def get_branch_b_path(disease_id):
    """获取 Branch B 某疾病的 JSON 路径"""
    return os.path.join(JSON_DIR, f"branch_b_disease_{disease_id}.json")
