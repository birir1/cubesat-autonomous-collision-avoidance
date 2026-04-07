"""
Evaluate Fusion Model

- Loads trained fusion, trajectory, and static models
- Computes RMSE, MAE, correlation
- Reports alpha statistics for confidence-weighted fusion
"""

import torch
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

from models.fusion.confidence_fusion_model import ConfidenceWeightedFusion
from models.trajectory_risk_model import TrajectoryRiskModel
from models.collision_risk_model import CollisionRiskModel

from utils.tle_loader import load_all_satellites
from data.features.trajectory_dataset import build_trajectory_dataset


# =========================================
# SAFE LOAD
# =========================================
def safe_load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except Exception:
        print("⚠️ Falling back to unsafe load (trusted source only)")
        return torch.load(path, map_location=device, weights_only=False)


# =========================================
# EVALUATION
# =========================================
def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # =========================================
    # DATA
    # =========================================
    sats = load_all_satellites()["starlink"][:500]

    print("Building trajectory dataset...")
    X, y = build_trajectory_dataset(sats, num_samples=1000)

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    dataset = torch.utils.data.TensorDataset(X, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)

    # =========================================
    # LOAD MODELS
    # =========================================
    traj_model = TrajectoryRiskModel().to(device)
    static_model = CollisionRiskModel().to(device)
    fusion_model = ConfidenceWeightedFusion().to(device)

    traj_model.load_state_dict(
        safe_load("models/trajectory_model.pth", device),
        strict=False
    )
    static_model.load_state_dict(
        safe_load("models/collision_model_real.pth", device),
        strict=False
    )
    fusion_model.load_state_dict(
        safe_load("models/fusion_model.pth", device),
        strict=False
    )

    traj_model.eval()
    static_model.eval()
    fusion_model.eval()

    # =========================================
    # EVALUATION LOOP
    # =========================================
    all_preds = []
    all_targets = []
    all_alphas = []

    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)

            traj_pred = torch.sigmoid(traj_model(xb))
            static_pred = torch.clamp(static_model(xb[:, -1, :]), 0, 1)

            fused, alpha = fusion_model(traj_pred, static_pred)

            all_preds.append(fused.cpu().numpy())
            all_targets.append(yb.cpu().numpy())
            all_alphas.append(alpha.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    all_alphas = np.concatenate(all_alphas)

    # =========================================
    # METRICS
    # =========================================
    rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
    mae = mean_absolute_error(all_targets, all_preds)
    corr = np.corrcoef(all_targets.flatten(), all_preds.flatten())[0, 1]

    print("=== Fusion Model Evaluation ===")
    print(f"Samples evaluated: {len(all_targets)}")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE : {mae:.6f}")
    print(f"Correlation: {corr:.4f}")
    print(f"Prediction stats -> min: {all_preds.min():.4f}, mean: {all_preds.mean():.4f}, max: {all_preds.max():.4f}")
    print(f"Alpha stats -> mean: {all_alphas.mean():.4f}, min: {all_alphas.min():.4f}, max: {all_alphas.max():.4f}")


# =========================================
# MAIN
# =========================================
if __name__ == "__main__":
    evaluate()