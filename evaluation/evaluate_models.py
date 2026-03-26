"""
Trajectory Model Evaluation (FINAL + ROBUST)

Fixes:
- Safe model loading (PyTorch 2.6 compatible)
- Correct logits → probability handling
- Batch inference (scalable)
- Strong numerical stability
- Collapse detection
- Calibration insights (research-grade)
"""

import torch
import numpy as np

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from utils.tle_loader import load_all_satellites
from data.features.trajectory_dataset import build_trajectory_dataset
from models.trajectory_risk_model import TrajectoryRiskModel
import models.trajectory_risk_model as m


# ============================================
# SAFE LOAD (PYTORCH 2.6 FIX)
# ============================================
def safe_load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except Exception:
        print("Falling back to unsafe load (trusted source only)")
        return torch.load(path, map_location=device, weights_only=False)


# ============================================
# BATCH PREDICTION
# ============================================
def predict_in_batches(model, X, batch_size=128):
    preds = []

    model.eval()
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = X[i:i + batch_size]
            logits = model(xb)

            # handle both cases (logits or sigmoid output)
            if logits.min() < 0 or logits.max() > 1:
                probs = torch.sigmoid(logits)
            else:
                probs = logits

            preds.append(probs.cpu().numpy())

    preds = np.concatenate(preds).flatten()

    preds = np.nan_to_num(preds, nan=0.0, posinf=1.0, neginf=0.0)
    preds = np.clip(preds, 0.0, 1.0)

    return preds


# ============================================
# MAIN EVALUATION
# ============================================
def evaluate():

    # ============================================
    # DEVICE
    # ============================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ============================================
    # LOAD DATA
    # ============================================
    sats = load_all_satellites()["starlink"][:500]

    print("Building test dataset...")
    X, y = build_trajectory_dataset(sats, num_samples=1000)

    X = torch.tensor(X, dtype=torch.float32).to(device)
    y = np.array(y, dtype=np.float32)

    # ---- Clean labels ----
    mask = np.isfinite(y)
    X = X[mask]
    y = y[mask]

    if len(y) == 0:
        print("No valid samples")
        return

    print(f"Dataset shape: {X.shape}")
    print(
        f"Risk stats → min: {y.min():.6f}, "
        f"mean: {y.mean():.6f}, max: {y.max():.6f}"
    )

    # ============================================
    # LOAD MODEL
    # ============================================
    model = TrajectoryRiskModel().to(device)

    try:
        state_dict = safe_load("models/trajectory_model.pth", device)

        if hasattr(model, "load_safe"):
            model.load_safe(state_dict)
        else:
            model.load_state_dict(state_dict, strict=False)

        print("Model loaded successfully")

    except Exception as e:
        print("Model loading failed:", e)
        return

    # ============================================
    # PREDICTION
    # ============================================
    preds = predict_in_batches(model, X)

    # ============================================
    # REGRESSION METRICS
    # ============================================
    mse = mean_squared_error(y, preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, preds)

    # ---- Correlation ----
    if np.std(preds) < 1e-6 or np.std(y) < 1e-6:
        corr = 0.0
    else:
        corr = np.corrcoef(y, preds)[0, 1]

    print("\n=== Evaluation Results ===")

    print("\n--- Regression ---")
    print(f"MSE:  {mse:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE:  {mae:.6f}")
    print(f"Correlation: {corr:.4f}")

    # ============================================
    # COLLAPSE DETECTION
    # ============================================
    if preds.max() - preds.min() < 1e-3:
        print("\n WARNING: Model predictions collapsed (constant output)")

    # ============================================
    # CLASSIFICATION METRICS
    # ============================================
    thresholds = [0.1, 0.2, 0.3, 0.5]

    print("\n--- Classification (multi-threshold) ---")

    for th in thresholds:
        y_true_cls = (y > th).astype(int)
        y_pred_cls = (preds > th).astype(int)

        if y_true_cls.sum() == 0:
            print(f"\nThreshold = {th}")
            print("  No positive samples in ground truth")
            continue

        acc = accuracy_score(y_true_cls, y_pred_cls)
        precision = precision_score(y_true_cls, y_pred_cls, zero_division=0)
        recall = recall_score(y_true_cls, y_pred_cls, zero_division=0)
        f1 = f1_score(y_true_cls, y_pred_cls, zero_division=0)

        print(f"\nThreshold = {th}")
        print(f"  Accuracy:  {acc:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1 Score:  {f1:.4f}")

    # ============================================
    # CALIBRATION INSIGHT (VERY IMPORTANT)
    # ============================================
    print("\n--- Calibration Check ---")
    print(f"Mean Prediction: {preds.mean():.6f}")
    print(f"Mean Ground Truth: {y.mean():.6f}")

    if abs(preds.mean() - y.mean()) > 0.05:
        print("⚠️ Model is poorly calibrated")

    # ============================================
    # DISTRIBUTIONS
    # ============================================
    print("\n--- Prediction Stats ---")
    print(f"Pred min:  {preds.min():.6f}")
    print(f"Pred mean: {preds.mean():.6f}")
    print(f"Pred max:  {preds.max():.6f}")

    print("\n--- Label Distribution ---")
    print(f"% > 0.1: {(y > 0.1).mean():.4f}")
    print(f"% > 0.2: {(y > 0.2).mean():.4f}")
    print(f"% > 0.3: {(y > 0.3).mean():.4f}")
    print(f"% > 0.5: {(y > 0.5).mean():.4f}")

    # ============================================
    # DEBUG
    # ============================================
    print("\nEVAL MODEL FILE:", m.__file__)


# ============================================
# MAIN
# ============================================
if __name__ == "__main__":
    evaluate()