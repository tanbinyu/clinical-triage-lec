"""Canonical disease metadata and default experiment parameters."""

DISEASE_TO_DATASETS = {
    0: ["LungHist700"],
    1: ["BreakHis"],
    2: ["COVID_CT"],
    3: ["LIDC_IDRI"],
    4: ["ImageCHD"],
    5: ["ChestX_ray14"],
    6: ["BUSI", "BUS_BRA"],
    7: ["HMC_QU"],
    8: ["TMED_2"],
    9: ["ACDC"],
}

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

NUM_DISEASES = 10

DATASET_TO_DISEASE = {
    ds: did
    for did, datasets in DISEASE_TO_DATASETS.items()
    for ds in datasets
}

DATASET_SAMPLE_COUNTS = {
    "LungHist700": 691,
    "BreakHis": 2000,
    "COVID_CT": 812,
    "LIDC_IDRI": 2000,
    "ImageCHD": 800,
    "ChestX_ray14": 2000,
    "BUSI": 780,
    "BUS_BRA": 1875,
    "HMC_QU": 800,
    "TMED_2": 2000,
    "ACDC": 150,
}

DATASET_NUM_CLASSES = {
    "LungHist700": 3,
    "BreakHis": 2,
    "COVID_CT": 2,
    "LIDC_IDRI": 3,
    "ImageCHD": 5,
    "ChestX_ray14": 14,
    "BUSI": 3,
    "BUS_BRA": 2,
    "HMC_QU": 2,
    "TMED_2": 3,
    "ACDC": 5,
}

DATASET_ERROR_RATES = {
    "LungHist700": 0.0478,
    "BreakHis": 0.0635,
    "COVID_CT": 0.1234,
    "LIDC_IDRI": 0.1733,
    "ImageCHD": 0.1127,
    "ChestX_ray14": 0.1683,
    "BUSI": 0.1430,
    "BUS_BRA": 0.1430,
    "HMC_QU": 0.1965,
    "TMED_2": 0.2062,
    "ACDC": 0.2656,
}

DEFAULTS = {
    "num_splits": 500,
    "split_ratio": 0.5,
    "beta": 0.1,
    "alpha_A": 0.1,
    "alpha_B": 0.1,
    "lambda_1": 0.5,
    "lambda_2": 0.5,
}
