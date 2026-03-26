"""
Model Comparison: Trajectory vs Static Collision Risk

FINAL RESEARCH-GRADE VERSION

Fixes:
- Handles logits vs probabilities correctly
- Safe model loading (PyTorch 2.6 fix)
- Robust correlation (no NaN)
- Collapse detection
- Dataset filtering (avoid degenerate evaluation)
- Better visualization
- Save plots and table for paper
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

from sklearn.metrics import mean_squared_error, mean_absolute_error

from utils.tle_loader import load_all_satellites
from data.features.trajectory_dataset import build_trajectory_dataset

from models.trajectory_risk_model import TrajectoryRiskModel
from models.collision_risk_model import CollisionRiskModel

os.makedirs("results", exist_ok=True)  # folder to save plots & table

# =========================================
# UTILS
# =========================================
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def safe_corr(y, preds):
    if len(y) < 2:
        return 0.0
    if np.std(preds) < 1e-8:
        return 0.0
    return np.corrcoef(y, preds)[0, 1]


# =========================================
# MODEL LOADERS
# =========================================
def load_trajectory_model(device):
    model = TrajectoryRiskModel().to(device)
    try:
        state_dict = torch.load(
            "models/trajectory_model.pth",
            map_location=device,
            weights_only=False
        )
        if hasattr(model, "load_safe"):
            model.load_safe(state_dict)
        else:
            model.load_state_dict(state_dict, strict=False)
        print("[INFO] Trajectory model loaded")
    except Exception as e:
        print("Trajectory model load failed:", e)
    model.eval()
    return model


def load_static_model(device):
    model = CollisionRiskModel().to(device)
    try:
        state_dict = torch.load(
            "models/collision_model_real.pth",
            map_location=device,
            weights_only=False
        )
        model.load_state_dict(state_dict, strict=False)
        print("[INFO] Static model loaded")
    except Exception as e:
        print("⚠️ Static model load failed:", e)
    model.eval()
    return model


# =========================================
# METRICS
# =========================================
def compute_metrics(y, preds, name="Model"):
    preds = np.clip(np.nan_to_num(preds, nan=0.0), 0.0, 1.0)
    mse = mean_squared_error(y, preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, preds)
    corr = safe_corr(y, preds)
    print(f"\n{name}")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  MAE : {mae:.6f}")
    print(f"  Corr: {corr:.4f}")
    return {"name": name, "rmse": rmse, "mae": mae, "corr": corr,
            "min": preds.min(), "mean": preds.mean(), "max": preds.max()}


# =========================================
# MAIN EVALUATION
# =========================================
def evaluate_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    sats = load_all_satellites()["starlink"][:500]

    print("Building evaluation dataset...")
    X, y = build_trajectory_dataset(sats, num_samples=1500)
    if len(y) == 0:
        print(" No valid samples")
        return

    X = torch.tensor(X, dtype=torch.float32).to(device)
    y = np.array(y, dtype=np.float32)
    mask = np.isfinite(y)
    X = X[mask]
    y = y[mask]

    print(f"Dataset shape: {X.shape}")
    print(f"Risk stats → min: {y.min():.6f}, mean: {y.mean():.6f}, max: {y.max():.6f}")

    traj_model = load_trajectory_model(device)
    static_model = load_static_model(device)

    # ---- PREDICTIONS ----
    with torch.no_grad():
        traj_logits = traj_model(X).cpu().numpy().flatten()
    traj_preds = sigmoid(traj_logits)

    static_input = X[:, -1, :]
    with torch.no_grad():
        static_preds = static_model(static_input).cpu().numpy().flatten()

    traj_preds = np.clip(np.nan_to_num(traj_preds), 0.0, 1.0)
    static_preds = np.clip(np.nan_to_num(static_preds), 0.0, 1.0)

    # ---- COLLAPSE DETECTION ----
    for preds, name in zip([traj_preds, static_preds], ["Trajectory", "Static"]):
        if preds.max() - preds.min() < 1e-3:
            print(f"⚠️ {name} model collapsed (constant predictions)")

    # ---- METRICS ----
    print("\n=== MODEL COMPARISON ===")
    traj_metrics = compute_metrics(y, traj_preds, "Trajectory Model")
    static_metrics = compute_metrics(y, static_preds, "Static Model")

    # ---- SAVE RESULTS TABLE ----
    df = pd.DataFrame([traj_metrics, static_metrics])
    table_path = "results/model_comparison.csv"
    df.to_csv(table_path, index=False)
    print(f"\n✅ Results table saved: {table_path}")

    # ---- VISUALIZATION ----
    plot_results(y, traj_preds, static_preds)


# =========================================
# VISUALIZATION
# =========================================
def plot_results(y, traj_preds, static_preds):
    # Scatter plots
    plt.figure(figsize=(6,6))
    plt.scatter(y, traj_preds, alpha=0.3)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.title("Trajectory Model")
    plt.xlabel("True")
    plt.ylabel("Pred")
    plt.tight_layout()
    plt.savefig("results/trajectory_scatter.png", dpi=300)
    plt.close()

    plt.figure(figsize=(6,6))
    plt.scatter(y, static_preds, alpha=0.3)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.title("Static Model")
    plt.xlabel("True")
    plt.ylabel("Pred")
    plt.tight_layout()
    plt.savefig("results/static_scatter.png", dpi=300)
    plt.close()

    # Error distribution
    plt.figure(figsize=(6,4))
    plt.hist(np.abs(y - traj_preds), bins=50, alpha=0.6, label="Trajectory")
    plt.hist(np.abs(y - static_preds), bins=50, alpha=0.6, label="Static")
    plt.legend()
    plt.title("Error Distribution")
    plt.tight_layout()
    plt.savefig("results/error_distribution.png", dpi=300)
    plt.close()

    # Risk distribution
    plt.figure(figsize=(6,4))
    plt.hist(y, bins=50, alpha=0.4, label="True")
    plt.hist(traj_preds, bins=50, alpha=0.4, label="Trajectory")
    plt.hist(static_preds, bins=50, alpha=0.4, label="Static")
    plt.legend()
    plt.title("Risk Distribution")
    plt.tight_layout()
    plt.savefig("results/risk_distribution.png", dpi=300)
    plt.close()

    print("\n✅ All plots saved in 'results/' folder.")


# =========================================
# MAIN
# =========================================
if __name__ == "__main__":
    evaluate_models()