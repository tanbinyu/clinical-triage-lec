"""
Router 推理模块：使用 UniMed-CLIP 作为 K 分类 Router。

Router 功能：
- 输入任意模态的病例图像（病理/CT/X光/超声/MRI）
- 输出该图像属于哪种疾病类型（K=8 分类）+ routing uncertainty (predictive entropy)
- 保存结果为 router_phase_1.json

Router 实现：
- 使用 UniMed-CLIP 模型做 zero-shot 分类
- 为每种疾病定义文本描述 (prompt)
- 计算图像与各疾病 prompt 的余弦相似度 → softmax → predictive entropy
"""

import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm

from calibration.config import (
    DISEASE_TO_DATASETS,
    DISEASE_NAMES,
    DATASET_TO_DISEASE,
    NUM_DISEASES,
    JSON_DIR,
    ROUTER_PHASE1_PATH,
    PROJECT_ROOT,
)


# 每种疾病的文本 prompt（用于 UniMed-CLIP zero-shot 分类）
# UniMed-CLIP 在医学多模态数据上预训练，对医学文本描述有更好的理解
DISEASE_PROMPTS = {
    0: "a histopathology image of lung cancer tissue",
    1: "a histopathology image of breast cancer tissue",
    2: "a CT scan image of the lung showing pulmonary disease",
    3: "a CT scan image of the heart showing cardiac structure",
    4: "a chest X-ray radiograph showing thoracic findings",
    5: "a breast ultrasound image showing breast lesion",
    6: "an echocardiography ultrasound image of the heart",
    7: "a cardiac MRI image showing heart structure",
}


class UniMedCLIPRouter:
    """
    基于 UniMed-CLIP 的 K 分类 Router。

    通过计算输入图像与 K 个疾病 prompt 的余弦相似度，
    经 softmax 得到概率分布，预测疾病类型并计算 predictive entropy。

    与通用 CLIP 相比，UniMed-CLIP 在医学图像上有更强的跨模态对齐能力，
    能更准确地区分不同模态（病理/CT/X光/超声/MRI）的医学图像。
    """

    def __init__(self, weights_path=None, device=None):
        """
        初始化 UniMed-CLIP Router。

        参数:
            weights_path: str 或 None, 模型权重路径。None 时使用默认路径
            device: torch.device 或 None
        """
        from model.UniMed_CLIP import load_unimed_clip

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        print(f"Loading UniMed-CLIP Router on {self.device}")
        self.model, self.preprocess, self.tokenizer = load_unimed_clip(
            weights_path=weights_path,
            device=self.device,
        )

        # 预计算疾病 prompt 的文本特征
        self.text_features = self._encode_prompts()
        print(f"Router initialized with {NUM_DISEASES} disease categories")

    def _encode_prompts(self):
        """
        预计算所有疾病 prompt 的文本嵌入。

        使用 UniMed-CLIP 的 HFTokenizer (BiomedBERT) 编码文本，
        然后通过模型的文本编码器提取特征并归一化。
        """
        prompts = [DISEASE_PROMPTS[k] for k in range(NUM_DISEASES)]

        # UniMed-CLIP 的 tokenizer 逐条编码后 cat
        tokens = torch.cat([
            self.tokenizer(prompt).to(self.device) for prompt in prompts
        ], dim=0)

        with torch.no_grad():
            text_features = self.model.encode_text(tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return text_features

    @torch.no_grad()
    def predict(self, image):
        """
        对单张 PIL 图像预测疾病类型和 uncertainty。

        参数:
            image: PIL.Image, 输入图像

        返回:
            predicted_disease: int, 预测的疾病编号 (0~K-1)
            uncertainty: float, predictive entropy H = -sum(p*log(p))
            probs: np.ndarray, (K,) 各疾病的概率分布
        """
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        image_features = self.model.encode_image(image_tensor)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # 余弦相似度 → softmax 概率
        similarity = (image_features @ self.text_features.T).squeeze(0)
        probs = similarity.softmax(dim=0).cpu().numpy()

        predicted_disease = int(np.argmax(probs))

        # predictive entropy: H = -sum(p * log(p))
        eps = 1e-12
        uncertainty = float(-np.sum(probs * np.log(probs + eps)))

        return predicted_disease, uncertainty, probs

    @torch.no_grad()
    def predict_batch(self, images):
        """
        批量预测（输入为已预处理的 tensor）。

        参数:
            images: torch.Tensor, (B, C, H, W) 已预处理的批量图像

        返回:
            predicted_diseases: np.ndarray, (B,) 预测的疾病编号
            uncertainties: np.ndarray, (B,) predictive entropy
            all_probs: np.ndarray, (B, K) 概率分布
        """
        images = images.to(self.device)
        image_features = self.model.encode_image(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        similarity = image_features @ self.text_features.T
        all_probs = similarity.softmax(dim=1).cpu().numpy()

        predicted_diseases = np.argmax(all_probs, axis=1)

        eps = 1e-12
        uncertainties = -np.sum(all_probs * np.log(all_probs + eps), axis=1)

        return predicted_diseases, uncertainties, all_probs


def run_router_inference(data_root=None, weights_path=None, device=None):
    """
    对所有数据集的 test split 运行 Router 推理并保存结果。

    遍历每个数据集，用 UniMed-CLIP Router 预测疾病类型，
    保存为 router_phase_1.json。

    每个样本记录:
        correctness: Router 是否分对了疾病 (1=正确, 0=错误)
        uncertainty: Router 的 predictive entropy (用于 LEC 校准)
        predicted_disease: Router 预测的疾病编号
        true_disease: 该样本所属数据集对应的真实疾病编号
        source_dataset: 该样本来自哪个数据集

    参数:
        data_root: str, 数据根目录 (默认: 项目下 data/)
        weights_path: str, UniMed-CLIP 权重路径 (默认: model/UniMed_CLIP/weights/)
        device: str 或 None, 设备 (如 'cuda:0', 'cpu')
    """
    from PIL import Image

    if data_root is None:
        data_root = os.path.join(PROJECT_ROOT, "data")

    device_obj = torch.device(device) if device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    router = UniMedCLIPRouter(weights_path=weights_path, device=device_obj)
    router_results = {}

    for disease_id, datasets in DISEASE_TO_DATASETS.items():
        for dataset_name in datasets:
            print(f"\n--- Processing {dataset_name} (disease={disease_id}) ---")

            samples = _load_dataset_samples(dataset_name, data_root)
            if not samples:
                print(f"  [SKIP] No samples found for {dataset_name}")
                continue

            print(f"  Found {len(samples)} samples")

            for sample_path in tqdm(samples, desc=f"  Router inference"):
                try:
                    img = Image.open(sample_path)
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                except Exception as e:
                    print(f"  [WARN] Cannot open {sample_path}: {e}")
                    continue

                predicted_disease, uncertainty, probs = router.predict(img)

                router_results[sample_path] = {
                    "correctness": int(predicted_disease == disease_id),
                    "uncertainty": round(float(uncertainty), 6),
                    "predicted_disease": predicted_disease,
                    "true_disease": disease_id,
                    "source_dataset": dataset_name,
                }

    # 保存结果
    os.makedirs(os.path.dirname(ROUTER_PHASE1_PATH), exist_ok=True)
    with open(ROUTER_PHASE1_PATH, "w") as f:
        json.dump(router_results, f, indent=2)

    total = len(router_results)
    correct = sum(1 for v in router_results.values() if v["correctness"] == 1)
    print(f"\n{'=' * 60}")
    print(f"Router inference complete: {total} samples")
    print(f"Routing accuracy: {correct}/{total} = {correct / total:.4f}" if total > 0 else "No samples")
    print(f"Saved to: {ROUTER_PHASE1_PATH}")

    return router_results


def _load_dataset_samples(dataset_name, data_root):
    """
    加载指定数据集的 test split 样本路径列表。

    通过扫描该数据集的 Data_split 目录中的 test.txt 文件获取样本列表。

    参数:
        dataset_name: str, 数据集名称
        data_root: str, 数据根目录

    返回:
        list[str]: 样本文件的完整路径列表
    """
    dataset_dir = os.path.join(data_root, dataset_name)
    split_dir = os.path.join(dataset_dir, "Data_split")
    image_dir = os.path.join(dataset_dir, "images")

    samples = []

    if not os.path.exists(split_dir):
        # 尝试直接从 image_dir 扫描
        if os.path.exists(image_dir):
            for root, dirs, files in os.walk(image_dir):
                for fname in files:
                    if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")):
                        samples.append(os.path.join(root, fname))
        return samples

    # 通过 Data_split/class/test.txt 获取样本
    categories = sorted([
        cls for cls in os.listdir(split_dir)
        if os.path.isdir(os.path.join(split_dir, cls))
    ])

    for cls_name in categories:
        test_txt = os.path.join(split_dir, cls_name, "test.txt")
        if not os.path.exists(test_txt):
            continue

        cls_img_dir = os.path.join(image_dir, cls_name)
        with open(test_txt) as f:
            for line in f:
                img_name = line.strip()
                if img_name:
                    full_path = os.path.join(cls_img_dir, img_name)
                    if os.path.exists(full_path):
                        samples.append(full_path)

    return samples


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="UniMed-CLIP Router 推理")
    parser.add_argument("--data_root", type=str, default=None,
                        help="数据根目录 (默认: 项目下 data/)")
    parser.add_argument("--weights_path", type=str, default=None,
                        help="UniMed-CLIP 权重路径 (默认: model/UniMed_CLIP/weights/unimed-clip-vit-b16.pt)")
    parser.add_argument("--device", type=str, default=None,
                        help="设备 (如 cuda:0, cpu)")
    args = parser.parse_args()

    run_router_inference(
        data_root=args.data_root,
        weights_path=args.weights_path,
        device=args.device,
    )

