"""Train one patient-disjoint adapter with validation and best-checkpoint selection."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import tempfile
from pathlib import Path
from typing import Any

from naturehelp_inference.config import DATASET_ORDER, load_config
from naturehelp_inference.manifest import manifest_digest
from naturehelp_inference.media import MediaReader
from naturehelp_inference.metrics import (
    calibrate_multilabel_thresholds,
    multiclass_metrics,
    multilabel_metrics,
)
from naturehelp_inference.models import build_adapter_head, create_extractor, resolve_device
from naturehelp_inference.pipeline import _encode_records
from naturehelp_inference.output import atomic_write_document
from naturehelp_inference.splits import build_or_load_split, records_for_split, split_digest
from naturehelp_inference.workflow import load_prepared_manifests


def _targets(records, spec, torch):
    if spec.task_type == "multilabel":
        result = torch.zeros((len(records), spec.num_classes), dtype=torch.float32)
        for row, record in enumerate(records):
            for label in record.target:
                result[row, label] = 1.0
        return result
    return torch.tensor([record.target for record in records], dtype=torch.long)


def _loss_function(targets, spec, device, torch):
    if spec.task_type == "multilabel":
        positives = targets.sum(dim=0)
        negatives = targets.shape[0] - positives
        pos_weight = negatives / positives.clamp_min(1.0)
        return torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    counts = torch.bincount(targets, minlength=spec.num_classes).float()
    weights = counts.sum() / counts.clamp_min(1.0)
    weights = weights / weights.mean()
    return torch.nn.CrossEntropyLoss(weight=weights.to(device))


def _validate_train_coverage(all_records, train_records, spec) -> None:
    def present(records) -> set[int]:
        labels: set[int] = set()
        for record in records:
            if isinstance(record.target, tuple):
                labels.update(record.target)
            else:
                labels.add(record.target)
        return labels

    missing = sorted(present(all_records) - present(train_records))
    if missing:
        raise RuntimeError(
            f"{spec.name}: train split 缺少真实存在的类别 {missing}；请调整 split seed 或划分算法"
        )


def _atomic_torch_save(torch, payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        torch.save(payload, temp_name)
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def _cache_metadata(config, spec, records, assignments, split_name: str) -> dict[str, Any]:
    return {
        "dataset": spec.name,
        "split": split_name,
        "manifest_digest": manifest_digest(records),
        "split_digest": split_digest(assignments),
        "split_seed": config.split_seed,
        "split_fractions": [
            config.train_fraction, config.val_fraction, config.downstream_fraction,
        ],
        "backbone": spec.backbone,
        "preprocessing_version": config.preprocessing_version,
    }


def _extract_or_load_features(
    config, spec, all_records, selected, assignments, split_name,
    extractor, cache_dir, rebuild_cache, torch,
):
    cache = cache_dir / f"{spec.name}_{split_name}.pt"
    expected_metadata = _cache_metadata(config, spec, all_records, assignments, split_name)
    expected_ids = [record.sample_id for record in selected]
    if cache.exists() and not rebuild_cache:
        payload = torch.load(cache, map_location="cpu", weights_only=True)
        if payload.get("metadata") != expected_metadata or payload.get("sample_ids") != expected_ids:
            raise RuntimeError(f"特征缓存已过期，请加 --rebuild-cache: {cache}")
        return payload["features"], payload["targets"]

    feature_parts = []
    with MediaReader(config.volume_slices) as reader:
        for start in range(0, len(selected), config.batch_size):
            batch = selected[start:start + config.batch_size]
            feature_parts.append(_encode_records(extractor, batch, reader, config.batch_size).cpu())
            print(f"[{split_name}] features {start + len(batch)}/{len(selected)}", flush=True)
    features = torch.cat(feature_parts, dim=0)
    targets = _targets(selected, spec, torch)
    _atomic_torch_save(torch, {
        "metadata": expected_metadata,
        "sample_ids": expected_ids,
        "features": features,
        "targets": targets,
    }, cache)
    return features, targets


def _evaluate(model, features, targets, spec, batch_size, device, torch):
    model.eval()
    logits = []
    with torch.inference_mode():
        for indices in torch.arange(len(features)).split(batch_size):
            logits.append(model(features[indices].to(device)).cpu())
    all_logits = torch.cat(logits, dim=0)
    if spec.task_type == "multilabel":
        metrics = multilabel_metrics(all_logits, targets)
    else:
        metrics = multiclass_metrics(all_logits, targets, spec.num_classes)
    return all_logits, metrics


def train(args) -> Path:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch

    config = load_config(args.config)
    spec = config.datasets[args.dataset]
    all_records = load_prepared_manifests(config, [args.dataset], verify_raw=True)[args.dataset]
    assignments = build_or_load_split(
        config, args.dataset, all_records, rebuild=args.rebuild_split
    )
    train_records = records_for_split(all_records, assignments, "train")
    val_records = records_for_split(all_records, assignments, "val")
    if not train_records or not val_records:
        raise RuntimeError(f"{args.dataset}: train 或 val 为空")
    _validate_train_coverage(all_records, train_records, spec)

    random.seed(args.seed)
    try:
        import numpy as np
        np.random.seed(args.seed)
    except ImportError:
        pass
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if config.deterministic_algorithms:
        torch.use_deterministic_algorithms(True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    device = resolve_device(config.device)
    extractor = create_extractor(config, spec, device)
    cache_dir = args.cache_dir or (config.work_dir / "feature_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    train_features, train_targets = _extract_or_load_features(
        config, spec, all_records, train_records, assignments, "train", extractor,
        cache_dir, args.rebuild_cache, torch,
    )
    val_features, val_targets = _extract_or_load_features(
        config, spec, all_records, val_records, assignments, "val", extractor,
        cache_dir, args.rebuild_cache, torch,
    )

    model = build_adapter_head(config, spec, extractor.feature_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = _loss_function(train_targets, spec, device, torch)
    generator = torch.Generator().manual_seed(args.seed)
    last_checkpoint = config.work_dir / "checkpoints" / f"{spec.name}.last.pt"
    start_epoch = 0
    best_score = -1.0
    best_epoch = -1
    stale_epochs = 0
    history: list[dict[str, Any]] = []
    training_metadata = {
        **_cache_metadata(config, spec, all_records, assignments, "training"),
        "training_seed": args.seed,
        "deterministic_algorithms": config.deterministic_algorithms,
        "optimizer": "AdamW",
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "train_batch_size": args.train_batch_size,
        "max_epochs": args.epochs,
        "patience": args.patience,
        "min_delta": args.min_delta,
        "checkpoint_id": config.checkpoint_ids.get(spec.backbone, "unregistered"),
    }
    if args.resume and last_checkpoint.exists():
        state = torch.load(last_checkpoint, map_location=device, weights_only=False)
        if state["metadata"] != training_metadata:
            raise RuntimeError(f"resume checkpoint 与当前数据/配置不一致: {last_checkpoint}")
        model.load_state_dict(state["state_dict"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch = int(state["epoch"]) + 1
        best_score = float(state["best_score"])
        best_epoch = int(state["best_epoch"])
        stale_epochs = int(state["stale_epochs"])
        history = list(state.get("history", []))
        if "generator_state" in state:
            generator.set_state(state["generator_state"])

    output = config.adapters_dir / f"{spec.name}.pt"
    for epoch in range(start_epoch, args.epochs):
        model.train()
        permutation = torch.randperm(len(train_features), generator=generator)
        total_loss = 0.0
        for indices in permutation.split(args.train_batch_size):
            batch_features = train_features[indices].to(device)
            batch_targets = train_targets[indices].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_features), batch_targets)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(indices)
        _, val_metrics = _evaluate(
            model, val_features, val_targets, spec, args.train_batch_size, device, torch
        )
        score = val_metrics["macro_f1"]
        row = {
            "epoch": epoch + 1,
            "train_loss": total_loss / len(train_features),
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if score > best_score + args.min_delta:
            best_score = score
            best_epoch = epoch
            stale_epochs = 0
            checkpoint_metadata = {
                **training_metadata,
                "dataset": spec.name,
                "backbone": spec.backbone,
                "feature_dim": extractor.feature_dim,
                "num_classes": spec.num_classes,
                "task_type": spec.task_type,
                "best_epoch": epoch + 1,
                "validation_metrics": val_metrics,
            }
            _atomic_torch_save(torch, {
                "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                "metadata": checkpoint_metadata,
            }, output)
        else:
            stale_epochs += 1

        _atomic_torch_save(torch, {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_score": best_score,
            "best_epoch": best_epoch,
            "stale_epochs": stale_epochs,
            "history": history,
            "generator_state": generator.get_state(),
            "metadata": training_metadata,
        }, last_checkpoint)
        if stale_epochs >= args.patience:
            print(f"early stopping at epoch {epoch + 1}; best={best_epoch + 1}", flush=True)
            break

    if not output.exists():
        raise RuntimeError("训练未生成 best checkpoint")
    best = torch.load(output, map_location=device, weights_only=False)
    model.load_state_dict(best["state_dict"])
    val_logits, calibrated_metrics = _evaluate(
        model, val_features, val_targets, spec, args.train_batch_size, device, torch
    )
    if spec.task_type == "multilabel":
        threshold_values = calibrate_multilabel_thresholds(val_logits, val_targets)
        calibrated_metrics = multilabel_metrics(
            val_logits, val_targets, torch.tensor(threshold_values)
        )
        best["metadata"]["multilabel_thresholds"] = threshold_values
        best["metadata"]["calibrated_validation_metrics"] = calibrated_metrics
        _atomic_torch_save(torch, best, output)

    metrics_path = config.work_dir / "metrics" / f"{spec.name}.json"
    atomic_write_document(metrics_path, {
        "dataset": spec.name,
        "best_epoch": int(best["metadata"]["best_epoch"]),
        "validation": calibrated_metrics,
        "history": history,
    })
    return output


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Train one frozen-backbone adapter")
    parser.add_argument("dataset", choices=DATASET_ORDER)
    parser.add_argument(
        "--config", type=Path,
        default=Path(__file__).resolve().parent / "configs" / "default.toml",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--train-batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--rebuild-split", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.seed is None:
        args.seed = load_config(args.config).training_seed
    print(f"saved: {train(args)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
