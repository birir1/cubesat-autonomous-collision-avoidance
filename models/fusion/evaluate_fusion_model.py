"""
Evaluate Fusion Model (Trajectory + Static)

FINAL ROBUST VERSION (UPDATED)

Fixes:
- Uses AttentionFusionModel (correct class)
- PyTorch 2.6 safe loading
- Stable inference (no tensor re-creation bugs)
- Proper sigmoid handling
- Clean evaluation metrics
"""

import torch
import numpy as np

from sklearn.metrics import mean_squared_error, mean_absolute_error

from utils.tle_loader import load_all_satellites
from data.features.trajectory_dataset import build_trajectory_dataset

from models.trajectory_risk_model import TrajectoryRiskModel
from models.collision_risk_model import CollisionRiskModel
from models.fusion.confidence_fusion_model import ConfidenceWeightedFusion


# =========================================
# SAFE LOAD (PYTORCH 2.6 FIX)
# =========================================
def safe_load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except Exception:
        print("⚠️ Falling back to unsafe load (trusted source only)")
        return torch.load(path, map_location=device, weights_only=False)


# =========================================
# SAFE PROBABILITY HANDLING
# =========================================
def to_prob(x):
    if x.min() < 0 or x.max() > 1:
        return torch.sigmoid(x)
    return x


# =========================================
# METRICS
# =========================================
def compute_metrics(name, y_true, preds):
    preds = np.clip(np.nan_to_num(preds), 0.0, 1.0)

    mse = mean_squared_error(y_true, preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, preds)

    corr = 0.0
    if np.std(preds) > 1e-6:
        corr = np.corrcoef(y_true, preds)[0, 1]

    print(f"\n{name}")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  MAE : {mae:.6f}")
    print(f"  Corr: {corr:.4f}")


# =========================================
# MAIN EVALUATION
# =========================================
def evaluate():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # =========================================
    # DATA
    # =========================================
    sats = load_all_satellites()["starlink"][:500]

    print("Building trajectory dataset...")
    X, y = build_trajectory_dataset(sats, num_samples=1000)

    X = torch.tensor(X, dtype=torch.float32).to(device)
    y = np.array(y, dtype=np.float32)

    print(f"Dataset shape: {X.shape}")

    # =========================================
    # LOAD MODELS
    # =========================================

    # ---- trajectory ----
    traj_model = TrajectoryRiskModel().to(device)
    traj_model.load_state_dict(
        safe_load("models/trajectory_model.pth", device),
        strict=False
    )
    traj_model.eval()
    print("[INFO] Trajectory model loaded")

    # ---- static ----
    static_model = CollisionRiskModel().to(device)
    static_model.load_state_dict(
        safe_load("models/collision_model_real.pth", device),
        strict=False
    )
    static_model.eval()
    print("[INFO] Static model loaded")

    # ---- fusion ----
    fusion_model = ConfidenceWeightedFusion().to(device)
    fusion_model.load_state_dict(
        safe_load("models/fusion_model.pth", device),
        strict=False
    )
    fusion_model.eval()
    print("[INFO] Fusion model loaded")

    # =========================================
    # PREDICTIONS
    # =========================================
    with torch.no_grad():

        # ---- trajectory ----
        traj_preds = to_prob(traj_model(X))

        # ---- static ----
        static_preds = to_prob(static_model(X[:, -1, :]))
        static_preds = torch.clamp(static_preds, 0.0, 1.0)

        # ---- fusion (with same normalization as training!) ----
        # This is CRITICAL - must match training preprocessing
        traj_preds_norm = (traj_preds - traj_preds.mean()) / (traj_preds.std() + 1e-6)
        static_preds_norm = (static_preds - static_preds.mean()) / (static_preds.std() + 1e-6)
        
        fusion_preds, alpha_preds = fusion_model(traj_preds_norm, static_preds_norm)
        fusion_preds = to_prob(fusion_preds)

        # convert to numpy
        traj_preds = traj_preds.cpu().numpy().flatten()
        static_preds = static_preds.cpu().numpy().flatten()
        fusion_preds = fusion_preds.cpu().numpy().flatten()
        alpha_preds = alpha_preds.cpu().numpy().flatten()

    # =========================================
    # RESULTS
    # =========================================
    print("\n=== MODEL COMPARISON ===")

    compute_metrics("Trajectory", y, traj_preds)
    compute_metrics("Static", y, static_preds)
    compute_metrics("Fusion", y, fusion_preds)

    print("\n=== FUSION GATE STATS ===")
    print(f"Alpha mean: {alpha_preds.mean():.4f}, std: {alpha_preds.std():.4f}, min: {alpha_preds.min():.4f}, max: {alpha_preds.max():.4f}")

    # =========================================
    # SANITY CHECK
    # =========================================
    print("\n--- Prediction Stats ---")
    print(f"Fusion min:  {fusion_preds.min():.6f}")
    print(f"Fusion mean: {fusion_preds.mean():.6f}")
    print(f"Fusion max:  {fusion_preds.max():.6f}")


# =========================================
# MAIN
# =========================================
if __name__ == "__main__":
    evaluate()