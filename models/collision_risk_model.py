"""
Collision Risk Prediction Model (FINAL RESEARCH VERSION)

Upgrades:
- Consistent normalization (train + inference)
- Safe save/load structure
- Improved stability (scheduler + monitoring)
- Collapse detection
- Robust inference pipeline
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader


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
# MODEL
# ============================================
class CollisionRiskModel(nn.Module):
    def __init__(self, input_dim=6):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),

            nn.Linear(64, 128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


# ============================================
# NORMALIZATION
# ============================================
def compute_normalization_stats(X):
    mean = np.mean(X, axis=0, keepdims=True)
    std = np.std(X, axis=0, keepdims=True) + 1e-8
    return mean.astype(np.float32), std.astype(np.float32)


def apply_normalization(X, mean, std):
    return (X - mean) / std


# ============================================
# LABEL FUNCTION
# ============================================
def compute_risk(distance):
    d = max(distance, 1e-3)
    risk = 1.0 / (1.0 + d / 50.0)
    return float(np.clip(risk, 0.0, 1.0))


# ============================================
# DATASET
# ============================================
def build_pairwise_dataset(features, max_pairs=5000):

    X, y = [], []

    n = min(len(features), 200)

    for i in range(n):
        for j in range(i + 1, n):

            s1 = features[i]
            s2 = features[j]

            rel_pos = s1[:3] - s2[:3]
            rel_vel = s1[3:] - s2[3:]

            sample = np.concatenate([rel_pos, rel_vel])

            if not np.isfinite(sample).all():
                continue

            sample = np.clip(sample, -1e4, 1e4)

            distance = np.linalg.norm(rel_pos)
            risk = compute_risk(distance)

            X.append(sample)
            y.append(risk)

            if len(X) >= max_pairs:
                break

        if len(X) >= max_pairs:
            break

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    return X, y


# ============================================
# TRAINING
# ============================================
def train_model(features, save_path="models/collision_model_real.pth"):

    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---- dataset ----
    X, y = build_pairwise_dataset(features)
    print(f"Dataset size: {X.shape}")

    # ---- normalization ----
    mean, std = compute_normalization_stats(X)
    X = apply_normalization(X, mean, std)

    norm_stats = {"mean": mean, "std": std}

    # ---- tensors ----
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    # ---- split ----
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, shuffle=True, random_state=42
    )

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=64)

    # ---- model ----
    model = CollisionRiskModel().to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.BCELoss()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.5
    )

    print("\nStarting training...\n")

    # ============================================
    # TRAIN LOOP
    # ============================================
    for epoch in range(20):

        # ===== TRAIN =====
        model.train()
        train_loss = 0.0
        train_preds = []

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            preds = model(xb)
            loss = criterion(preds, yb)

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            train_preds.append(preds.detach().cpu().numpy())

        train_loss /= len(train_loader)
        train_preds = np.concatenate(train_preds)

        # ===== VALIDATION =====
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

        pred_mean = val_preds.mean()
        pred_min = val_preds.min()
        pred_max = val_preds.max()

        print(
            f"Epoch {epoch:02d} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Pred Mean: {pred_mean:.4f} | "
            f"Min: {pred_min:.4f} | Max: {pred_max:.4f}"
        )

        # ---- collapse detection ----
        if pred_max - pred_min < 1e-4:
            print("WARNING: Model collapsing (constant predictions)")

        scheduler.step(val_loss)

    # ---- save ----
    torch.save({
        "model_state_dict": model.state_dict(),
        "norm": norm_stats
    }, save_path)

    print(f"\nModel saved: {save_path}")

    return model


# ============================================
# LOAD MODEL (NEW - IMPORTANT)
# ============================================
def load_model(model_path, device="cpu"):
    model = CollisionRiskModel().to(device)

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])
    norm_stats = checkpoint["norm"]

    model.eval()

    return model, norm_stats


# ============================================
# INFERENCE
# ============================================
def predict_risk(model, sat1, sat2, norm_stats, device="cpu"):

    sat1 = np.array(sat1)
    sat2 = np.array(sat2)

    rel_pos = sat1[:3] - sat2[:3]
    rel_vel = sat1[3:] - sat2[3:]

    x = np.concatenate([rel_pos, rel_vel])
    x = np.clip(x, -1e4, 1e4)

    mean = norm_stats["mean"].squeeze()
    std = norm_stats["std"].squeeze()

    x = (x - mean) / std

    x = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        risk = model(x).item()

    return float(np.clip(risk, 0.0, 1.0))