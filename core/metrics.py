import numpy as np
from typing import Dict


EPS = 1e-8  # numerical stability


# =========================================================
# REGRESSION METRICS
# =========================================================
def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    name: str = "Regression"
) -> Dict[str, float]:
    """
    Compute regression metrics: MSE, RMSE, MAE.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        name: Optional label

    Returns:
        Dict of regression metrics
    """
    y_true = np.asarray(y_true, dtype=np.float32).flatten()
    y_pred = np.asarray(y_pred, dtype=np.float32).flatten()

    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(y_true - y_pred)))

    return {
        "name": name,
        "mse": mse,
        "rmse": rmse,
        "mae": mae
    }


# =========================================================
# FULL EVALUATION (REGRESSION + SAFETY)
# =========================================================
def evaluate_model_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    name: str = "Model",
    danger_threshold: float = 0.7
) -> Dict[str, Dict]:
    """
    Evaluate model predictions for:
    - Risk regression
    - Danger classification

    Args:
        y_true: Ground truth risk values
        y_pred: Predicted risk values
        name: Model/experiment name
        danger_threshold: Risk threshold for "danger"

    Returns:
        Dictionary of evaluation metrics
    """
    y_true = np.asarray(y_true, dtype=np.float32).flatten()
    y_pred = np.asarray(y_pred, dtype=np.float32).flatten()

    # -----------------------------
    # CLASSIFICATION LABELS
    # -----------------------------
    true_danger = (y_true > danger_threshold).astype(np.int32)
    pred_danger = (y_pred > danger_threshold).astype(np.int32)

    # -----------------------------
    # CONFUSION MATRIX
    # -----------------------------
    tp = int(np.sum((true_danger == 1) & (pred_danger == 1)))
    tn = int(np.sum((true_danger == 0) & (pred_danger == 0)))
    fp = int(np.sum((true_danger == 0) & (pred_danger == 1)))
    fn = int(np.sum((true_danger == 1) & (pred_danger == 0)))

    # -----------------------------
    # METRICS (SAFE COMPUTATION)
    # -----------------------------
    precision = tp / (tp + fp + EPS)
    recall = tp / (tp + fn + EPS)
    f1_score = 2 * precision * recall / (precision + recall + EPS)
    fnr = fn / (tp + fn + EPS)
    accuracy = (tp + tn) / (len(y_true) + EPS)

    # -----------------------------
    # REGRESSION METRICS
    # -----------------------------
    risk_metrics = compute_regression_metrics(y_true, y_pred, name=name)

    # -----------------------------
    # SAFETY METRICS
    # -----------------------------
    safety_metrics_dict = {
        "threshold": danger_threshold,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "false_negative_rate": fnr,
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn
    }

    return {
        "risk_metrics": risk_metrics,
        "safety_metrics": safety_metrics_dict
    }


# =========================================================
# LIGHTWEIGHT SAFETY METRICS
# =========================================================
def safety_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute binary safety metrics.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        threshold: Decision threshold

    Returns:
        Dict of safety metrics
    """
    y_true = np.asarray(y_true, dtype=np.float32).flatten()
    y_pred = np.asarray(y_pred, dtype=np.float32).flatten()

    y_true_binary = (y_true >= threshold).astype(np.int32)
    y_pred_binary = (y_pred >= threshold).astype(np.int32)

    tp = int(np.sum((y_true_binary == 1) & (y_pred_binary == 1)))
    tn = int(np.sum((y_true_binary == 0) & (y_pred_binary == 0)))
    fp = int(np.sum((y_true_binary == 0) & (y_pred_binary == 1)))
    fn = int(np.sum((y_true_binary == 1) & (y_pred_binary == 0)))

    precision = tp / (tp + fp + EPS)
    recall = tp / (tp + fn + EPS)
    false_alarm_rate = fp / (fp + tn + EPS)

    return {
        "threshold": threshold,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "false_alarm_rate": false_alarm_rate,
        "collision_detection_rate": recall  # alias
    }