"""
Hybrid Training Pipeline (REAL + SYNTHETIC)

Key Features:
- Uses hybrid dataset
- Fixes class imbalance
- Stable training (no collapse)
- Proper logits handling
"""

import torch
import numpy as np
import random

from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

from utils.tle_loader import load_all_satellites
from data.synthetic.hybrid_dataset import build_hybrid_dataset
from models.trajectory_risk_model import TrajectoryRiskModel


# ============================================
# SEED
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ============================================
    # LOAD SATELLITES
    # ============================================
    sats = load_all_satellites()["starlink"][:1000]

    # ============================================
    # BUILD HYBRID DATASET
    # ============================================
    X, y = build_hybrid_dataset(
        sats,
        num_real=2000,
        num_synthetic=4000
    )

    print("Dataset shape:", X.shape)

    # ============================================
    # TORCH CONVERSION
    # ============================================
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    # ============================================
    # SPLIT
    # ============================================
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

    # ============================================
    # MODEL
    # ============================================
    model = TrajectoryRiskModel().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # ============================================
    # LOSS (NO pos_weight needed now!)
    # ============================================
    criterion = torch.nn.BCEWithLogitsLoss()

    print("\nStarting HYBRID training...\n")

    # ============================================
    # TRAIN LOOP
    # ============================================
    for epoch in range(20):

        # ===== TRAIN =====
        model.train()
        train_loss = 0.0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            logits = model(xb)
            loss = criterion(logits, yb)

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ===== VALIDATE =====
        model.eval()
        val_loss = 0.0
        preds = []

        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)

                logits = model(xb)
                loss = criterion(logits, yb)

                val_loss += loss.item()

                probs = torch.sigmoid(logits)
                preds.append(probs.cpu().numpy())

        val_loss /= len(val_loader)
        preds = np.concatenate(preds)

        print(
            f"Epoch {epoch:02d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Pred Mean: {preds.mean():.4f}"
        )

        # ---- collapse detection ----
        if preds.max() - preds.min() < 1e-3:
            print(" WARNING: Model collapsing")

    # ============================================
    # SAVE
    # ============================================
    torch.save(model.state_dict(), "models/trajectory_model_hybrid.pth")

    print("\n Hybrid model saved → models/trajectory_model_hybrid.pth")


# ============================================
# MAIN
# ============================================
if __name__ == "__main__":
    train()