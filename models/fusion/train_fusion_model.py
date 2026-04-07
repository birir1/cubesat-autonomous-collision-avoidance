"""
Train Fusion Model (FINAL - STABLE + NON-COLLAPSING)

Key Fixes:
- Confidence-weighted fusion (learn alpha for trajectory vs static)
- Residual fusion (learn correction instead of full prediction)
- Stable input distributions
- Anti-collapse variance constraint
- Improved hybrid loss
"""

import torch
import numpy as np
import random

from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

from models.fusion.confidence_fusion_model import ConfidenceWeightedFusion
from models.trajectory_risk_model import TrajectoryRiskModel
from models.collision_risk_model import CollisionRiskModel

from utils.tle_loader import load_all_satellites
from data.features.trajectory_dataset import build_trajectory_dataset


# =========================================
# SEED
# =========================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
# HYBRID LOSS (FIXED: Always Positive + Stable)
# =========================================
def hybrid_loss(preds, targets, mse_weight=1.0, corr_weight=0.1, var_reg_weight=0.01):
    """
    Stable hybrid loss that always produces positive values.
    
    Components:
    - MSE: prediction error (primary learning signal)
    - (1-correlation): ensures outputs correlate with targets
    - Variance penalty: soft threshold prevents mode collapse
    """
    
    # Primary loss: prediction error
    mse = torch.nn.functional.mse_loss(preds, targets)

    # Secondary loss: correlation bonus
    preds_centered = preds - preds.mean()
    targets_centered = targets - targets.mean()

    corr = torch.sum(preds_centered * targets_centered) / (
        torch.sqrt(torch.sum(preds_centered ** 2)) *
        torch.sqrt(torch.sum(targets_centered ** 2)) + 1e-8
    )
    
    corr = torch.clamp(corr, -1.0, 1.0)
    corr_loss = (1.0 - corr)

    # Tertiary loss: variance floor (prevent collapse)
    var = torch.var(preds)
    var_threshold = 0.01
    var_penalty = torch.clamp(var_threshold - var, min=0.0)

    return mse_weight * mse + corr_weight * corr_loss + var_reg_weight * var_penalty


# =========================================
# TRAIN
# =========================================
def train():
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # =========================================
    # DATA
    # =========================================
    sats = load_all_satellites()["starlink"][:500]

    print("Building dataset...")
    X, y = build_trajectory_dataset(sats, num_samples=2000)

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=64,
        shuffle=True
    )

    val_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=64,
        shuffle=False
    )

    # =========================================
    # BASE MODELS
    # =========================================
    traj_model = TrajectoryRiskModel().to(device)
    static_model = CollisionRiskModel().to(device)

    traj_model.load_state_dict(
        safe_load("models/trajectory_model.pth", device),
        strict=False
    )
    print("[INFO] Trajectory model loaded")

    static_model.load_state_dict(
        safe_load("models/collision_model_real.pth", device),
        strict=False
    )
    print("[INFO] Static model loaded")

    traj_model.eval()
    static_model.eval()

    # =========================================
    # FUSION MODEL
    # =========================================
    fusion_model = ConfidenceWeightedFusion().to(device)

    optimizer = torch.optim.Adam(fusion_model.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    best_val = float("inf")
    patience = 0
    max_patience = 15

    print("\nTraining Fusion Model (STABLE + CONFIDENCE-WEIGHTED)...\n")

    # =========================================
    # LOOP
    # =========================================
    for epoch in range(100):
        fusion_model.train()
        train_loss = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            with torch.no_grad():
                traj_pred = torch.sigmoid(traj_model(xb))          # (B,1)
                static_pred = torch.clamp(
                    static_model(xb[:, -1, :]), 0.0, 1.0
                )                                                  # (B,1)

            # =========================================
            # CONFIDENCE-WEIGHTED FUSION
            # =========================================
            fused, alpha = fusion_model(traj_pred, static_pred)

            loss = hybrid_loss(fused, yb)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(fusion_model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # -------- VALIDATION --------
        fusion_model.eval()
        val_loss = 0
        preds = []
        alphas = []

        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)

                traj_pred = torch.sigmoid(traj_model(xb))
                static_pred = torch.clamp(
                    static_model(xb[:, -1, :]), 0, 1
                )

                fused, alpha = fusion_model(traj_pred, static_pred)

                loss = hybrid_loss(fused, yb)
                val_loss += loss.item()

                preds.append(fused.cpu().numpy())
                alphas.append(alpha.cpu().numpy())

        val_loss /= len(val_loader)
        preds = np.concatenate(preds)
        alphas = np.concatenate(alphas)

        print(
            f"Epoch {epoch:02d} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Std: {preds.std():.6f} | "
            f"Alpha: {alphas.mean():.4f}"
        )

        if preds.std() < 0.01:
            print("⚠️ WARNING: Low variance (possible collapse)")

        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            patience = 0
            torch.save(fusion_model.state_dict(), "models/fusion_model.pth")
        else:
            patience += 1

        if patience >= max_patience:
            print(f"Early stopping triggered (patience={patience})")
            break

    print("\nFusion model saved: models/fusion_model.pth")


# =========================================
# MAIN
# =========================================
if __name__ == "__main__":
    train()