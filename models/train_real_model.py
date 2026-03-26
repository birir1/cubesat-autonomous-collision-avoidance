"""
Train Collision Risk Model using REAL satellite data

Improvements:
- Dataset balancing (prevents collapse)
- Feature normalization
- Train/validation split
- Better loss monitoring
"""

import torch
import numpy as np

from utils.tle_loader import load_all_satellites
from data.features.collision_labels import build_real_dataset
from models.collision_risk_model import CollisionRiskModel


def normalize(X):
    """
    Normalize features (important for ML stability)
    """
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-6
    return (X - mean) / std


def train_real():

    # -----------------------------
    # 1. Load satellites
    # -----------------------------
    sats = load_all_satellites()["starlink"]

    print("Building real dataset (this may take time)...")
    X, y = build_real_dataset(sats)

    if len(X) == 0:
        raise ValueError("Dataset is empty — check dataset builder")

    print("Dataset shape:", X.shape)

    # -----------------------------
    # 2. Balance dataset
    # -----------------------------
    print("Balancing dataset...")

    X_balanced = []
    y_balanced = []

    for xi, yi in zip(X, y):
        if yi > 0:
            # keep all risky samples
            X_balanced.append(xi)
            y_balanced.append(yi)
        else:
            # downsample safe samples
            if np.random.rand() < 0.1:
                X_balanced.append(xi)
                y_balanced.append(yi)

    X = np.array(X_balanced)
    y = np.array(y_balanced)

    print("Balanced dataset shape:", X.shape)

    # -----------------------------
    # 3. Normalize features
    # -----------------------------
    X = normalize(X)

    # -----------------------------
    # 4. Train / Validation split
    # -----------------------------
    split = int(0.8 * len(X))

    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    # Convert to tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

    # -----------------------------
    # 5. Model
    # -----------------------------
    model = CollisionRiskModel()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()

    # -----------------------------
    # 6. Training loop
    # -----------------------------
    print("\nStarting training...\n")

    for epoch in range(30):

        # Train
        model.train()
        optimizer.zero_grad()

        preds = model(X_train)
        loss = criterion(preds, y_train)

        loss.backward()
        optimizer.step()

        # Validate
        model.eval()
        with torch.no_grad():
            val_preds = model(X_val)
            val_loss = criterion(val_preds, y_val)

        print(
            f"Epoch {epoch:02d} | "
            f"Train Loss: {loss.item():.4f} | "
            f"Val Loss: {val_loss.item():.4f}"
        )

    # -----------------------------
    # 7. Save model
    # -----------------------------
    torch.save(model.state_dict(), "models/collision_model_real.pth")
    print("\nModel saved!")

    return model


if __name__ == "__main__":
    train_real()