"""
Evaluation metric helpers for safety and calibration reporting.
"""

from typing import Dict, Sequence
import numpy as np


def safety_metrics(y_true: Sequence[float], y_pred: Sequence[float], threshold: float = 0.5) -> Dict[str, float]:
    """Compute basic safety metrics for binary collision predictions."""
    y_true = np.asarray(y_true, dtype=np.int32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    y_pred_binary = (y_pred >= threshold).astype(np.int32)

    tp = int(((y_true == 1) & (y_pred_binary == 1)).sum())
    tn = int(((y_true == 0) & (y_pred_binary == 0)).sum())
    fp = int(((y_true == 0) & (y_pred_binary == 1)).sum())
    fn = int(((y_true == 1) & (y_pred_binary == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    collision_detection_rate = recall

    return {
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'precision': precision,
        'recall': recall,
        'false_alarm_rate': false_alarm_rate,
        'collision_detection_rate': collision_detection_rate
    }


def calibration_metrics(y_true: Sequence[float], y_pred: Sequence[float], n_bins: int = 10) -> Dict[str, float]:
    """Compute calibration metrics such as ECE and MCE."""
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    mce = 0.0

    for i in range(n_bins):
        mask = (y_pred >= bins[i]) & (y_pred < bins[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = y_true[mask].mean()
        bin_conf = y_pred[mask].mean()
        gap = abs(bin_acc - bin_conf)
        ece += gap * mask.sum() / len(y_pred)
        mce = max(mce, gap)

    return {
        'ece': float(ece),
        'mce': float(mce)
    }
