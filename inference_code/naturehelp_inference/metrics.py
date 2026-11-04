from __future__ import annotations

from typing import Any


def _safe_f1(tp: int, fp: int, fn: int) -> float:
    denominator = 2 * tp + fp + fn
    return 0.0 if denominator == 0 else (2.0 * tp) / denominator


def multiclass_metrics(logits: Any, targets: Any, num_classes: int) -> dict[str, float]:
    predictions = logits.argmax(dim=1)
    accuracy = float((predictions == targets).float().mean().item())
    per_class: list[float] = []
    for label in range(num_classes):
        tp = int(((predictions == label) & (targets == label)).sum().item())
        fp = int(((predictions == label) & (targets != label)).sum().item())
        fn = int(((predictions != label) & (targets == label)).sum().item())
        per_class.append(_safe_f1(tp, fp, fn))
    return {"accuracy": accuracy, "macro_f1": sum(per_class) / len(per_class)}


def multilabel_metrics(
    logits: Any, targets: Any, thresholds: Any | float = 0.5
) -> dict[str, float]:
    probabilities = logits.sigmoid()
    predictions = probabilities >= thresholds
    truth = targets.bool()
    exact_match = float((predictions == truth).all(dim=1).float().mean().item())
    tp_all = int((predictions & truth).sum().item())
    fp_all = int((predictions & ~truth).sum().item())
    fn_all = int((~predictions & truth).sum().item())
    per_class: list[float] = []
    for label in range(targets.shape[1]):
        predicted = predictions[:, label]
        actual = truth[:, label]
        tp = int((predicted & actual).sum().item())
        fp = int((predicted & ~actual).sum().item())
        fn = int((~predicted & actual).sum().item())
        per_class.append(_safe_f1(tp, fp, fn))
    return {
        "exact_match": exact_match,
        "micro_f1": _safe_f1(tp_all, fp_all, fn_all),
        "macro_f1": sum(per_class) / len(per_class),
    }


def calibrate_multilabel_thresholds(logits: Any, targets: Any) -> list[float]:
    probabilities = logits.sigmoid()
    truth = targets.bool()
    result: list[float] = []
    for label in range(targets.shape[1]):
        actual = truth[:, label]
        if int(actual.sum().item()) == 0:
            result.append(0.5)
            continue
        best_threshold = 0.5
        best_f1 = -1.0
        for step in range(10, 91, 5):
            threshold = step / 100.0
            predicted = probabilities[:, label] >= threshold
            tp = int((predicted & actual).sum().item())
            fp = int((predicted & ~actual).sum().item())
            fn = int((~predicted & actual).sum().item())
            score = _safe_f1(tp, fp, fn)
            if score > best_f1:
                best_f1 = score
                best_threshold = threshold
        result.append(best_threshold)
    return result

