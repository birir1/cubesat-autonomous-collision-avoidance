"""
Train Transformer-based Trajectory Collision Risk Model

FINAL STABLE VERSION (REGRESSION FIX)

Key Fixes:
- Uses regression (HuberLoss) instead of BCE
- Removes incorrect pos_weight (caused collapse)
- Matches model output (Sigmoid)
- Improved monitoring (std-based)
- Stable training dynamics
"""

import torch
import numpy as np
import random

from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

import models.trajectory_risk_model as m
from models.trajectory_risk_model import TrajectoryRiskModel

from utils.tle_loader import load_all_satellites
from data.features.trajectory_dataset import build_trajectory_dataset


# ============================================
# REPRODUCIBILITY
# ============================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================
# TRAIN
# ============================================

def train():

    set_seed(42)

    # ---- device ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ---- load satellites ----
    sats = load_all_satellites()["starlink"][:1000]

    # ---- build dataset ----
    X, y = build_trajectory_dataset(sats, num_samples=3000)

    print("Dataset shape:", X.shape)

    # ---- convert ----
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    # ---- split ----
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, shuffle=True, random_state=42
    )

    # ---- dataloaders ----
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

    # ---- model ----
    model = TrajectoryRiskModel().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # REGRESSION LOSS (CRITICAL FIX)
    criterion = torch.nn.HuberLoss(delta=0.1)

    # ---- scheduler ----
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.5
    )

    print("\nStarting training...\n")

    # ============================================
    # TRAIN LOOP
    # ============================================

    for epoch in range(25):

        # ================= TRAIN =================
        model.train()
        train_loss = 0.0
        train_preds = []

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            preds = model(xb)  # already sigmoid output
            loss = criterion(preds, yb)

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            train_preds.append(preds.detach().cpu().numpy())

        train_loss /= len(train_loader)
        train_preds = np.concatenate(train_preds)

        # ================= VALIDATE =================
        model.eval()
        val_loss = 0.0
        val_preds = []

        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)

                preds = model(xb)
                loss = criterion(preds, yb)

                val_loss += loss.item()
                val_preds.append(preds.cpu().numpy())

        val_loss /= len(val_loader)
        val_preds = np.concatenate(val_preds)

        # ============================================
        # MONITORING (CRITICAL)
        # ============================================

        pred_mean = val_preds.mean()
        pred_std = val_preds.std()
        pred_min = val_preds.min()
        pred_max = val_preds.max()

        print(
            f"Epoch {epoch:02d} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Mean: {pred_mean:.4f} | Std: {pred_std:.4f} | "
            f"Min: {pred_min:.4f} | Max: {pred_max:.4f}"
        )

        # ---- collapse detection ----
        if pred_std < 1e-3:
            print("⚠️ WARNING: Model collapsing (low variance predictions)")

        scheduler.step(val_loss)

    # ============================================
    # SAVE
    # ============================================

    torch.save(model.state_dict(), "models/trajectory_model.pth")

    print("\nModel saved: models/trajectory_model.pth")
    print("TRAIN MODEL FILE:", m.__file__)


# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    train()