from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Iterable

from .config import DatasetConfig, PipelineConfig, ROUTER_PROMPTS


class ModelError(RuntimeError):
    pass


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ModelError("推理需要 PyTorch；请先安装 requirements.txt 中的依赖") from exc
    return torch


def resolve_device(requested: str):
    torch = _torch()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ModelError(f"配置要求 {requested}，但当前 PyTorch 无可用 CUDA")
    return device


def _freeze(model: Any) -> None:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def _load_torch_state(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    torch = _torch()
    if not path.exists():
        raise ModelError(f"权重不存在: {path}")
    if path.suffix.casefold() == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise ModelError("读取 safetensors 需要 safetensors") from exc
        return load_file(str(path), device="cpu"), {}
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ModelError(f"权重文件顶层必须是 dict: {path}")
    metadata = payload.get("metadata", {}) if isinstance(payload.get("metadata"), dict) else {}
    for key in ("state_dict", "model", "model_state_dict"):
        if key in payload and isinstance(payload[key], dict):
            return payload[key], metadata
    return payload, metadata


def _strip_prefix(state: dict[str, Any], prefixes: tuple[str, ...]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in state.items():
        normalized = key
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if normalized.startswith(prefix):
                    normalized = normalized[len(prefix):]
                    changed = True
        result[normalized] = value
    return result


class ImageFeatureExtractor:
    feature_dim: int

    def encode_images(self, images: list[Any]):
        raise NotImplementedError


class HuggingFaceDinoExtractor(ImageFeatureExtractor):
    def __init__(self, model_dir: Path, device: Any) -> None:
        if not (model_dir / "config.json").exists():
            raise ModelError(f"DINO 模型目录不完整: {model_dir}")
        try:
            from transformers import AutoImageProcessor, AutoModel
        except ImportError as exc:
            raise ModelError("DINO/RAD-DINO 需要 transformers") from exc
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(model_dir, local_files_only=True)
        self.model = AutoModel.from_pretrained(model_dir, local_files_only=True).to(device)
        _freeze(self.model)
        self.feature_dim = int(self.model.config.hidden_size)

    def encode_images(self, images: list[Any]):
        torch = _torch()
        batch = self.processor(images=images, return_tensors="pt")
        batch = {key: value.to(self.device) for key, value in batch.items()}
        with torch.inference_mode():
            output = self.model(**batch)
            features = output.last_hidden_state[:, 0]
        return features.float()


class UNIExtractor(ImageFeatureExtractor):
    def __init__(self, model_dir: Path, device: Any) -> None:
        candidates = [
            model_dir / "pytorch_model.bin", model_dir / "model.safetensors",
            model_dir / "uni.bin", model_dir / "UNI.pth",
        ]
        checkpoint = next((path for path in candidates if path.exists()), None)
        if checkpoint is None:
            raise ModelError(
                f"UNI 权重缺失（官方仓库受权限控制）: {model_dir}。"
                "请放入 pytorch_model.bin 或 model.safetensors。"
            )
        try:
            import timm
            from torchvision import transforms
            from torchvision.transforms import InterpolationMode
        except ImportError as exc:
            raise ModelError("UNI 需要 timm 和 torchvision") from exc
        self.device = device
        self.model = timm.create_model(
            "vit_large_patch16_224",
            img_size=224,
            patch_size=16,
            init_values=1e-5,
            num_classes=0,
            dynamic_img_size=True,
        )
        state, _ = _load_torch_state(checkpoint)
        state = _strip_prefix(state, ("module.", "model.", "backbone."))
        self.model.load_state_dict(state, strict=True)
        self.model.to(device)
        _freeze(self.model)
        self.feature_dim = int(self.model.num_features)
        self.transform = transforms.Compose([
            transforms.Resize(224, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    def encode_images(self, images: list[Any]):
        torch = _torch()
        batch = torch.stack([self.transform(image) for image in images]).to(self.device)
        with torch.inference_mode():
            output = self.model(batch)
        return output.float()


class USFMExtractor(ImageFeatureExtractor):
    def __init__(self, root: Path, device: Any) -> None:
        source = root / "source"
        checkpoint = root / "USFM_latest.pth"
        if not (source / "usdsgen").exists() or not checkpoint.exists():
            raise ModelError(f"USFM 源码或权重不完整: {root}")
        sys.path.insert(0, str(source))
        try:
            from functools import partial
            import torch.nn as nn
            from torchvision import transforms
            from torchvision.transforms import InterpolationMode
            from usdsgen.modules.backbone.vision_transformer import VisionTransformer
        except ImportError as exc:
            raise ModelError("USFM 依赖未安装或官方源码不可导入") from exc
        self.device = device
        self.model = VisionTransformer(
            img_size=224, patch_size=16, in_chans=3, num_classes=0,
            embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
            qkv_bias=True, attn_drop_rate=0.0, drop_path_rate=0.1,
            init_values=0.1, use_abs_pos_emb=False, use_rel_pos_bias=True,
            use_shared_rel_pos_bias=False, use_mean_pooling=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
        )
        state, _ = _load_torch_state(checkpoint)
        state = _strip_prefix(state, ("module.", "model.", "backbone."))
        incompatible = self.model.load_state_dict(state, strict=False)
        bad_missing = [key for key in incompatible.missing_keys if not key.startswith("head.")]
        bad_unexpected = [key for key in incompatible.unexpected_keys if not key.startswith("head.")]
        if bad_missing or bad_unexpected:
            raise ModelError(
                f"USFM 权重与官方 ViT-B 结构不匹配；missing={bad_missing[:8]}, "
                f"unexpected={bad_unexpected[:8]}"
            )
        self.model.to(device)
        _freeze(self.model)
        self.feature_dim = 768
        self.transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    def encode_images(self, images: list[Any]):
        torch = _torch()
        batch = torch.stack([self.transform(image) for image in images]).to(self.device)
        with torch.inference_mode():
            output = self.model.forward_features(batch)
        return output.float()


class CineMAExtractor:
    # ED and ES are encoded independently by the one-channel pretrained model,
    # then concatenated so the adapter can use cardiac-cycle differences.
    feature_dim = 1536

    def __init__(self, root: Path, device: Any) -> None:
        source = root / "source"
        checkpoint = root / "pretrained" / "cinema.safetensors"
        config_path = root / "pretrained" / "config.yaml"
        for path in (source / "cinema", checkpoint, config_path):
            if not path.exists():
                raise ModelError(f"CineMA 文件不完整: {path}")
        sys.path.insert(0, str(source))
        try:
            from omegaconf import OmegaConf
            from safetensors.torch import load_file
            from cinema.mae.mae import get_model
        except ImportError as exc:
            raise ModelError("CineMA 需要其官方源码、omegaconf 和 safetensors") from exc
        config = OmegaConf.load(config_path)
        self.model = get_model(config)
        self.model.load_state_dict(load_file(str(checkpoint), device="cpu"), strict=True)
        self.model.to(device)
        _freeze(self.model)
        self.device = device

    def encode_volume(self, tensor: Any):
        torch = _torch()
        tensor = tensor.to(self.device)
        with torch.inference_mode():
            features = self.model.feature_forward({"sax": tensor})["cls"]
        return features[:, 0].float().reshape(1, -1)


def create_extractor(config: PipelineConfig, spec: DatasetConfig, device: Any):
    root = config.weights_root
    if spec.backbone == "UNI":
        return UNIExtractor(root / "UNI", device)
    if spec.backbone == "DINOv2-large":
        return HuggingFaceDinoExtractor(root / "DINOv2-large", device)
    if spec.backbone == "RAD-DINO":
        return HuggingFaceDinoExtractor(root / "RAD-DINO", device)
    if spec.backbone == "USFM":
        return USFMExtractor(root / "USFM", device)
    if spec.backbone == "CineMA":
        return CineMAExtractor(root / "CineMA", device)
    raise ModelError(f"未知 backbone: {spec.backbone}")


def build_adapter_head(
    config: PipelineConfig,
    spec: DatasetConfig,
    feature_dim: int,
):
    torch = _torch()
    nn = torch.nn

    class AdapterHead(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            bottleneck = config.adapter_bottleneck
            self.adapter = nn.Sequential(
                nn.LayerNorm(feature_dim),
                nn.Linear(feature_dim, bottleneck),
                nn.GELU(),
                nn.Linear(bottleneck, feature_dim),
            )
            self.head = nn.Sequential(
                nn.LayerNorm(feature_dim),
                nn.Linear(feature_dim, spec.num_classes),
            )

        def forward(self, features: Any):
            return self.head(features + self.adapter(features))

    return AdapterHead()


def create_adapter_head(
    config: PipelineConfig,
    spec: DatasetConfig,
    feature_dim: int,
    device: Any,
):
    model = build_adapter_head(config, spec, feature_dim)
    checkpoint = config.adapters_dir / f"{spec.name}.pt"
    if not checkpoint.exists():
        raise ModelError(
            f"缺少 {spec.name} adapter 权重: {checkpoint}。"
            "本程序不会以随机初始化代替已训练权重。"
        )
    state, metadata = _load_torch_state(checkpoint)
    expected_metadata = {
        "dataset": spec.name,
        "backbone": spec.backbone,
        "feature_dim": feature_dim,
        "num_classes": spec.num_classes,
        "task_type": spec.task_type,
        "preprocessing_version": config.preprocessing_version,
        "split_seed": config.split_seed,
        "checkpoint_id": config.checkpoint_ids.get(spec.backbone, "unregistered"),
    }
    for key, expected in expected_metadata.items():
        if metadata.get(key) != expected:
            raise ModelError(
                f"{checkpoint} metadata.{key}={metadata.get(key)!r}，期望 {expected!r}"
            )
    state = _strip_prefix(state, ("module.", "model."))
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise ModelError(f"{spec.name} adapter 结构与权重不匹配: {exc}") from exc
    model.to(device).eval()
    # Inference can use validation-calibrated per-class thresholds without adding
    # fields to the public JSON contract.
    model.naturehelp_metadata = metadata
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


class UniMedRouter:
    def __init__(self, config: PipelineConfig, device: Any) -> None:
        for path, description in (
            (config.router_source / "src" / "open_clip", "UniMed-CLIP 官方源码"),
            (config.router_checkpoint, "UniMed-CLIP 检查点"),
            (config.router_text_encoder / "config.json", "本地 BiomedBERT"),
        ):
            if not path.exists():
                raise ModelError(f"缺少{description}: {path}")
        sys.path.insert(0, str(config.router_source / "src"))
        try:
            from open_clip import HFTokenizer, create_model_and_transforms
        except ImportError as exc:
            raise ModelError("无法导入 UniMed-CLIP 官方 open_clip") from exc
        self.device = device
        self.tokenizer = HFTokenizer(str(config.router_text_encoder), context_length=256)
        mean = (0.48145466, 0.4578275, 0.40821073)
        std = (0.26862954, 0.26130258, 0.27577711)
        self.model, _, self.transform = create_model_and_transforms(
            "ViT-L-14-336-quickgelu",
            str(config.router_checkpoint),
            precision="fp32",
            device=device,
            force_quick_gelu=True,
            pretrained_image=False,
            mean=mean,
            std=std,
            inmem=True,
            text_encoder_name=str(config.router_text_encoder),
        )
        _freeze(self.model)
        prompts = [ROUTER_PROMPTS[name] for name in config.datasets]
        torch = _torch()
        tokens = self.tokenizer(prompts).to(device)
        with torch.inference_mode():
            text = self.model.encode_text(tokens).float()
            self.text_features = torch.nn.functional.normalize(text, dim=-1)

    def predict(self, images: list[Any]):
        torch = _torch()
        batch = torch.stack([self.transform(image) for image in images]).to(self.device)
        with torch.inference_mode():
            image_features = self.model.encode_image(batch).float()
            image_features = torch.nn.functional.normalize(image_features, dim=-1)
            scale = self.model.logit_scale.exp().float().clamp(max=100.0)
            probabilities = (scale * image_features @ self.text_features.T).softmax(dim=-1)
        return probabilities


def verify_weight_layout(
    config: PipelineConfig,
    dataset_names: Iterable[str],
    require_adapters: bool = True,
) -> list[str]:
    errors: list[str] = []
    router_paths = [
        config.router_source / "src" / "open_clip",
        config.router_checkpoint,
        config.router_text_encoder / "config.json",
    ]
    for path in router_paths:
        if not path.exists():
            errors.append(f"missing router file: {path}")
    for name in dataset_names:
        spec = config.datasets[name]
        root = config.weights_root / spec.backbone
        required: dict[str, tuple[Path, ...]] = {
            "UNI": (
                root / "pytorch_model.bin", root / "model.safetensors",
                root / "uni.bin", root / "UNI.pth",
            ),
            "DINOv2-large": (root / "model.safetensors", root / "config.json"),
            "RAD-DINO": (root / "model.safetensors", root / "config.json"),
            "USFM": (root / "USFM_latest.pth", root / "source" / "usdsgen"),
            "CineMA": (
                root / "pretrained" / "cinema.safetensors",
                root / "pretrained" / "config.yaml",
                root / "source" / "cinema",
            ),
        }
        paths = required[spec.backbone]
        if spec.backbone == "UNI":
            if not any(path.exists() for path in paths):
                errors.append(f"missing UNI checkpoint in: {root}")
        else:
            for path in paths:
                if not path.exists():
                    errors.append(f"missing backbone file: {path}")
        if require_adapters:
            adapter = config.adapters_dir / f"{name}.pt"
            if not adapter.exists():
                errors.append(f"missing adapter: {adapter}")
    return list(dict.fromkeys(errors))

