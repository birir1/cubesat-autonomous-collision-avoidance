"""
Model Comparison: Trajectory vs Static vs Fusion

FINAL MULTI-MODEL RESEARCH VERSION

Features:
- Trajectory vs Static vs Fusion
- Safe model loading (PyTorch 2.6)
- Logits → probability handling
- Pearson + Spearman correlation
- Calibration curve
- Scatter (density + regression line)
- Error distribution
- CSV + LaTeX export
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import spearmanr

from utils.tle_loader import load_all_satellites
from data.features.trajectory_dataset import build_trajectory_dataset

from models.trajectory_risk_model import TrajectoryRiskModel
from models.static_collision_risk_model import StaticCollisionRiskModel
from models.fusion.fusion_model import FusionModel

os.makedirs("results", exist_ok=True)


# =========================================
# UTILS
# =========================================
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def safe_load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except Exception:
        print("⚠️ Falling back to unsafe load (trusted source only)")
        return torch.load(path, map_location=device, weights_only=False)


def safe_corr(y, preds):
    if len(y) < 2 or np.std(preds) < 1e-8:
        return 0.0
    return np.corrcoef(y, preds)[0, 1]


def safe_spearman(y, preds):
    if len(y) < 2:
        return 0.0
    return spearmanr(y, preds).correlation


# =========================================
# METRICS
# =========================================
def compute_metrics(y, preds, name):
    preds = np.clip(np.nan_to_num(preds), 0.0, 1.0)

    mse = mean_squared_error(y, preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, preds)
    pearson = safe_corr(y, preds)
    spearman = safe_spearman(y, preds)

    print(f"\n{name}")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  MAE : {mae:.6f}")
    print(f"  Pearson: {pearson:.4f}")
    print(f"  Spearman: {spearman:.4f}")

    return {
        "Model": name,
        "RMSE": rmse,
        "MAE": mae,
        "Pearson": pearson,
        "Spearman": spearman,
        "MeanPred": preds.mean()
    }


# =========================================
# CALIBRATION
# =========================================
def plot_calibration(y, preds, name):
    bins = np.linspace(0, 1, 10)
    digitized = np.digitize(preds, bins)

    bin_true, bin_pred = [], []

    for i in range(1, len(bins)):
        mask = digitized == i
        if np.sum(mask) > 0:
            bin_true.append(y[mask].mean())
            bin_pred.append(preds[mask].mean())

    plt.figure()
    plt.plot(bin_pred, bin_true, marker='o')
    plt.plot([0, 1], [0, 1], '--')
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Calibration ({name})")
    plt.tight_layout()
    plt.savefig(f"results/calibration_{name}.png", dpi=300)
    plt.close()


# =========================================
# VISUALIZATION
# =========================================
def plot_scatter(y, preds, name):
    plt.figure(figsize=(6, 6))

    plt.hexbin(y, preds, gridsize=40)
    plt.colorbar()

    # regression line
    z = np.polyfit(y, preds, 1)
    p = np.poly1d(z)
    plt.plot(y, p(y), linestyle='--')

    plt.plot([0, 1], [0, 1], '--')
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title(name)

    plt.tight_layout()
    plt.savefig(f"results/{name}_scatter.png", dpi=300)
    plt.close()


def plot_error_distribution(y, preds_dict):
    plt.figure()

    for name, preds in preds_dict.items():
        plt.hist(np.abs(y - preds), bins=50, alpha=0.5, label=name)

    plt.legend()
    plt.title("Error Distribution")
    plt.tight_layout()
    plt.savefig("results/error_distribution.png", dpi=300)
    plt.close()


# =========================================
# MAIN
# =========================================
def evaluate_models():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # =========================================
    # DATA
    # =========================================
    sats = load_all_satellites()["starlink"][:500]

    print("Building dataset...")
    X, y = build_trajectory_dataset(sats, num_samples=2000)

    X = torch.tensor(X, dtype=torch.float32).to(device)
    y = np.array(y, dtype=np.float32)

    # clean labels
    mask = np.isfinite(y)
    X = X[mask]
    y = y[mask]

    print(f"Dataset shape: {X.shape}")
    print(f"Risk stats → min: {y.min():.4f}, mean: {y.mean():.4f}, max: {y.max():.4f}")

    # =========================================
    # LOAD MODELS
    # =========================================
    traj_model = TrajectoryRiskModel().to(device)
    static_model = StaticCollisionRiskModel().to(device)
    fusion_model = FusionModel().to(device)

    traj_model.load_state_dict(safe_load("models/trajectory_model.pth", device), strict=False)
    # static_model.load_state_dict(safe_load("models/collision_model_real.pth", device), strict=False)
    fusion_model.load_state_dict(safe_load("models/fusion_model.pth", device), strict=False)

    traj_model.eval()
    static_model.eval()
    fusion_model.eval()

    # =========================================
    # PREDICTIONS
    # =========================================
    with torch.no_grad():

        traj_logits = traj_model(X).cpu().numpy().flatten()
        traj_preds = sigmoid(traj_logits)

        static_preds = static_model(X[:, -1, :]).cpu().numpy().flatten()
        static_preds = np.clip(static_preds, 0.0, 1.0)

        fusion_preds = fusion_model(
            torch.tensor(traj_preds).unsqueeze(1).to(device),
            torch.tensor(static_preds).unsqueeze(1).to(device)
        )[0].cpu().numpy().flatten()

    # collapse detection
    for name, preds in {
        "Trajectory": traj_preds,
        "Static": static_preds,
        "Fusion": fusion_preds
    }.items():
        if preds.max() - preds.min() < 1e-3:
            print(f"⚠️ {name} collapsed")

    # =========================================
    # METRICS
    # =========================================
    print("\n=== MODEL COMPARISON ===")

    traj_metrics = compute_metrics(y, traj_preds, "Trajectory")
    static_metrics = compute_metrics(y, static_preds, "Static")
    fusion_metrics = compute_metrics(y, fusion_preds, "Fusion")

    df = pd.DataFrame([traj_metrics, static_metrics, fusion_metrics])
    df_rounded = df.round(4)

    df_rounded.to_csv("results/model_comparison.csv", index=False)
    df_rounded.to_latex("results/model_comparison.tex", index=False)

    print("\n✅ Tables saved (CSV + LaTeX)")

    # =========================================
    # PLOTS
    # =========================================
    plot_scatter(y, traj_preds, "trajectory")
    plot_scatter(y, static_preds, "static")
    plot_scatter(y, fusion_preds, "fusion")

    plot_error_distribution(y, {
        "Trajectory": traj_preds,
        "Static": static_preds,
        "Fusion": fusion_preds
    })

    plot_calibration(y, traj_preds, "trajectory")
    plot_calibration(y, static_preds, "static")
    plot_calibration(y, fusion_preds, "fusion")

    print("\n✅ All plots saved in results/")


# =========================================
# MAIN
# =========================================
if __name__ == "__main__":
    evaluate_models()