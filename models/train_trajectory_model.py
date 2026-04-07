"""
Train Transformer-based Trajectory Collision Risk Model

FINAL RESEARCH-GRADE VERSION (STABLE)

Fixes:
- Correct logits → sigmoid handling
- Stable hybrid loss (MSE + correlation)
- Safe correlation computation
- Proper LR for Transformer
- Early stopping (robust)
- Improved monitoring
"""

import torch
import numpy as np
import random

from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

from scipy.stats import spearmanr

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
# HYBRID LOSS (STABLE VERSION)
# ============================================

def hybrid_loss(preds, targets):
    """
    Combines:
    - MSE (accuracy)
    - Correlation (ranking quality)

    NOTE:
    Final loss can be negative → THIS IS OK.
    """

    # ---- MSE ----
    mse = torch.nn.functional.mse_loss(preds, targets)

    # ---- CORRELATION (stable) ----
    preds_centered = preds - preds.mean()
    targets_centered = targets - targets.mean()

    numerator = torch.sum(preds_centered * targets_centered)

    denom = (
        torch.sqrt(torch.sum(preds_centered ** 2)) *
        torch.sqrt(torch.sum(targets_centered ** 2))
    ) + 1e-8

    corr = numerator / denom

    # ---- combined ----
    loss = mse - 0.1 * corr

    return loss, mse.detach(), corr.detach()


# ============================================
# TRAIN
# ============================================

def train():

    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ============================================
    # DATA
    # ============================================

    sats = load_all_satellites()["starlink"][:1000]

    X, y = build_trajectory_dataset(sats, num_samples=3000)

    print("Dataset shape:", X.shape)

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, shuffle=True, random_state=42
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

    # ============================================
    # MODEL
    # ============================================

    model = TrajectoryRiskModel().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.5
    )

    print("\nStarting training...\n")

    best_val_loss = float("inf")
    patience_counter = 0

    # ============================================
    # TRAIN LOOP
    # ============================================

    for epoch in range(30):

        # ================= TRAIN =================
        model.train()
        train_loss = 0.0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            logits = model(xb)
            preds = torch.sigmoid(logits)  # CRITICAL FIX

            loss, _, _ = hybrid_loss(preds, yb)

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ================= VALIDATION =================
        model.eval()
        val_loss = 0.0
        val_preds = []

        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)

                logits = model(xb)
                preds = torch.sigmoid(logits)

                loss, _, _ = hybrid_loss(preds, yb)

                val_loss += loss.item()
                val_preds.append(preds.cpu().numpy())

        val_loss /= len(val_loader)
        val_preds = np.concatenate(val_preds).flatten()
        y_val_np = y_val.numpy().flatten()

        # ============================================
        # METRICS (SAFE)
        # ============================================

        if np.std(val_preds) > 1e-6 and np.std(y_val_np) > 1e-6:
            pearson = np.corrcoef(val_preds, y_val_np)[0, 1]
            spearman = spearmanr(val_preds, y_val_np).correlation
        else:
            pearson = 0.0
            spearman = 0.0

        # ============================================
        # MONITORING
        # ============================================

        print(
            f"Epoch {epoch:02d} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Pearson: {pearson:.4f} | Spearman: {spearman:.4f} | "
            f"Std: {val_preds.std():.4f}"
        )

        # ---- collapse detection ----
        if val_preds.std() < 1e-3:
            print("⚠️ WARNING: Model collapsing (low variance predictions)")

        # ---- scheduler ----
        scheduler.step(val_loss)

        # ---- early stopping ----
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "models/trajectory_model.pth")
        else:
            patience_counter += 1

        if patience_counter >= 5:
            print("Early stopping triggered")
            break

    print("\nModel saved: models/trajectory_model.pth")
    print("TRAIN MODEL FILE:", m.__file__)


# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    train()