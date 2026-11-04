"""Metadata for the patient-ID Step-1 cohort."""

DISEASE_NAMES = {
    0: "Pathology—lung cancer",
    1: "Pathology—breast cancer",
    2: "CT—pneumonia",
    3: "CT—pulmonary nodule malignancy",
    4: "CT—congenital heart disease",
    5: "Chest X-ray—thoracic findings",
    6: "Mammography—breast-lesion malignancy",
    7: "Ultrasound—breast cancer",
    8: "Ultrasound—pulmonary-tumour malignancy",
    9: "Cardiac MRI—five-class cardiac diagnosis",
}

DISEASE_SHORT_NAMES = {
    0: "Path—lung cancer",
    1: "Path—breast cancer",
    2: "CT—pneumonia",
    3: "CT—nodule malignancy",
    4: "CT—CHD",
    5: "CXR—thoracic findings",
    6: "Mammo—breast malignancy",
    7: "US—breast cancer",
    8: "US—pulmonary tumour",
    9: "Cardiac MRI—diagnosis",
}

DATASET_TO_DISEASE = {
    "LungHist700": 0,
    "BreakHis": 1,
    "COVID_CT": 2,
    "LIDC_IDRI": 3,
    "LDFC": 3,
    "ImageCHD": 4,
    "ChestX_ray14": 5,
    "CBIS-DDSM": 6,
    "BUS-BRA": 7,
    "PU2756": 8,
    "ACDC": 9,
}

DISEASE_TO_DATASET = {
    disease_id: dataset for dataset, disease_id in DATASET_TO_DISEASE.items()
    if dataset != "LIDC_IDRI"
}

DISEASE_TO_SPECIALIST_MODEL = {
    0: "UNI/LungHist700_adapter",
    1: "UNI/BreakHis_adapter",
    2: "DINOv2-large/COVID_CT_adapter",
    3: "DINOv2-large/LDFC_adapter",
    4: "DINOv2-large/ImageCHD_adapter",
    5: "RAD-DINO/ChestX_ray14_adapter",
    6: "RAD-DINO/CBIS-DDSM_adapter",
    7: "USFM/BUS-BRA_adapter",
    8: "USFM/PU2756_adapter",
    9: "CineMA/ACDC_adapter",
}

NUM_DISEASES = 10
