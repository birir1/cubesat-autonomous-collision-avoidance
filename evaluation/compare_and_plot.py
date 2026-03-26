import torch
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error
)

from utils.tle_loader import load_all_satellites
from data.features.trajectory_dataset import build_trajectory_dataset

from models.trajectory_risk_model import TrajectoryRiskModel
from models.collision_risk_model import CollisionRiskModel


# ===============================
# MODEL LOADERS
# ===============================
def load_trajectory_model(device):
    model = TrajectoryRiskModel().to(device)

    try:
        state_dict = torch.load("models/trajectory_model.pth", map_location=device)

        if hasattr(model, "load_safe"):
            model.load_safe(state_dict)
        else:
            model.load_state_dict(state_dict, strict=False)

        print("[INFO] Trajectory model loaded")

    except Exception as e:
        print("⚠️ Trajectory model load failed:", e)

    model.eval()
    return model


def load_static_model(device):
    model = CollisionRiskModel().to(device)

    try:
        state_dict = torch.load(
            "models/collision_model_real.pth",
            map_location=device
        )

        model.load_state_dict(state_dict, strict=False)
        print("[INFO] Static model loaded")

    except Exception as e:
        print("⚠️ Static model load failed:", e)

    model.eval()
    return model


# ===============================
# SAFE CORRELATION
# ===============================
def safe_corr(y, preds):
    if len(y) < 2:
        return 0.0
    if np.std(preds) < 1e-6:
        return 0.0  # model collapse case
    return float(np.corrcoef(y, preds)[0, 1])


# ===============================
# MAIN EVALUATION
# ===============================
def evaluate_models():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    sats = load_all_satellites()["starlink"][:500]

    print("Building evaluation dataset...")
    X, y = build_trajectory_dataset(sats, num_samples=1000)

    X = torch.tensor(X, dtype=torch.float32).to(device)
    y = np.array(y)

    # ---- Clean dataset ----
    mask = np.isfinite(y)
    X = X[mask]
    y = y[mask]

    print(f"Dataset shape: {X.shape}")
    print(
        f"Risk stats → min: {y.min():.6f}, "
        f"mean: {y.mean():.6f}, max: {y.max():.6f}"
    )

    # ---- Load models ----
    traj_model = load_trajectory_model(device)
    static_model = load_static_model(device)

    # ===============================
    # PREDICTIONS
    # ===============================
    with torch.no_grad():
        traj_preds = traj_model(X).detach().cpu().numpy().flatten()

    # Static model uses last timestep
    static_input = X[:, -1, :]
    with torch.no_grad():
        static_preds = static_model(static_input).detach().cpu().numpy().flatten()

    # ---- Clean predictions ----
    traj_preds = np.clip(np.nan_to_num(traj_preds), 0.0, 1.0)
    static_preds = np.clip(np.nan_to_num(static_preds), 0.0, 1.0)

    # ===============================
    # METRICS
    # ===============================
    def compute_metrics(name, preds):
        mse = mean_squared_error(y, preds)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y, preds)
        corr = safe_corr(y, preds)

        print(f"\n{name}")
        print(f"  RMSE: {rmse:.6f}")
        print(f"  MAE : {mae:.6f}")
        print(f"  Corr: {corr:.4f}")

        return rmse, mae, corr

    print("\n=== MODEL COMPARISON ===")

    traj_rmse, traj_mae, traj_corr = compute_metrics(
        "Trajectory Model", traj_preds
    )

    static_rmse, static_mae, static_corr = compute_metrics(
        "Static Model", static_preds
    )

    # ===============================
    # DIAGNOSTICS (VERY IMPORTANT)
    # ===============================
    print("\n=== MODEL DIAGNOSTICS ===")

    def describe_preds(name, preds):
        print(f"\n{name}")
        print(f"  min:  {preds.min():.6f}")
        print(f"  mean: {preds.mean():.6f}")
        print(f"  max:  {preds.max():.6f}")
        print(f"  std:  {preds.std():.6f}")

        if preds.std() < 1e-4:
            print("  ⚠️ WARNING: Model predictions collapsed (constant output)")

    describe_preds("Trajectory Predictions", traj_preds)
    describe_preds("Static Predictions", static_preds)

    # ===============================
    # INTERPRETATION (FOR PAPER)
    # ===============================
    print("\n=== INTERPRETATION ===")

    if traj_corr < 0.05:
        print("🚨 Trajectory model is NOT learning meaningful signal")
        print("→ Likely causes:")
        print("  - Extreme class imbalance")
        print("  - Too few close encounters")
        print("  - Temporal features not informative")
        print("  - Model collapsed to mean prediction")

    elif traj_corr < static_corr:
        print("⚠️ Static model outperforms trajectory model")
        print("→ Temporal modeling not yet beneficial")

    else:
        print("✅ Trajectory model shows improvement over static baseline")

    # ===============================
    # PLOTS (FOR PAPER)
    # ===============================
    plot_results(y, traj_preds, static_preds)


# ===============================
# VISUALIZATION
# ===============================
def plot_results(y, traj_preds, static_preds):

    # ---- 1. Predicted vs True ----
    plt.figure()
    plt.scatter(y, traj_preds, alpha=0.3)
    plt.plot([0, 1], [0, 1])
    plt.title("Trajectory Model: Predicted vs True")
    plt.xlabel("True Risk")
    plt.ylabel("Predicted Risk")
    plt.show()

    plt.figure()
    plt.scatter(y, static_preds, alpha=0.3)
    plt.plot([0, 1], [0, 1])
    plt.title("Static Model: Predicted vs True")
    plt.xlabel("True Risk")
    plt.ylabel("Predicted Risk")
    plt.show()

    # ---- 2. Error Distribution ----
    traj_error = np.abs(y - traj_preds)
    static_error = np.abs(y - static_preds)

    plt.figure()
    plt.hist(traj_error, bins=50, alpha=0.6, label="Trajectory")
    plt.hist(static_error, bins=50, alpha=0.6, label="Static")
    plt.title("Error Distribution")
    plt.xlabel("Absolute Error")
    plt.ylabel("Count")
    plt.legend()
    plt.show()

    # ---- 3. Risk Distribution ----
    plt.figure()
    plt.hist(y, bins=50, alpha=0.4, label="True")
    plt.hist(traj_preds, bins=50, alpha=0.4, label="Trajectory")
    plt.hist(static_preds, bins=50, alpha=0.4, label="Static")
    plt.title("Risk Distribution")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    evaluate_models()