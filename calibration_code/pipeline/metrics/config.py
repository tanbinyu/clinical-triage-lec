"""metrics 模块的全局配置：路径、常量、参数网格。"""
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_ROOT = os.path.join(PROJECT_ROOT, "result", "json")
OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "result", "new_res")

NUM_DISEASES = 10
NUM_SPLITS = 500
BETA = 0.1
ALPHA = 0.2
N_BOOT = 1000

DISEASE_NAMES = {
    0: "pathology_lung_cancer",
    1: "pathology_breast_cancer",
    2: "ct_pneumonia",
    3: "ct_lung_nodule",
    4: "ct_congenital_heart",
    5: "xray_lung",
    6: "mammography_breast_lesion_benign_malignant",
    7: "us_breast_cancer",
    8: "us_lung_disease",
    9: "mri_cardiomyopathy",
}

BETA_SWEEP = [0.01, 0.02, 0.025, 0.03, 0.04, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
# β 扫描完整版：仅对这些 β 值计算全部下游指标（含 LEC 校准）
BETA_SWEEP_FULL = [0.01, 0.025, 0.05, 0.1, 0.15, 0.2]
ALPHA_SWEEP = [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.8]

# 效用函数的代价参数网格
C_ERR_LIST = [1, 5, 10, 20, 50]
C_REF_LIST = [0.1, 0.5, 1.0, 2.0]
