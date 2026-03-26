"""
Training Script for GNN Collision Predictor

Pipeline:
Graph Dataset -> GNN Training -> Collision Probability Model

Input:
results/collision_dataset/graph_data.pt

Outputs:
results/saved_models/gnn_collision_model.pth
results/metrics/gnn_collision_training_metrics.csv
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from tqdm import tqdm

from phases.phase5_collision_risk_estimation.models.gnn_collision_predictor import GNNCollisionPredictor


# ----------------------------------------------------
# CONFIG
# ----------------------------------------------------

DATASET_PATH = "results/collision_dataset/graph_data.pt"

MODEL_SAVE_PATH = "results/saved_models/gnn_collision_model.pth"
METRICS_PATH = "results/metrics/gnn_collision_training_metrics.csv"

EPOCHS = 40
LR = 1e-3

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs("results/saved_models", exist_ok=True)
os.makedirs("results/metrics", exist_ok=True)


# ----------------------------------------------------
# LOAD DATASET
# ----------------------------------------------------

def load_dataset():

    print("Loading collision graph dataset...")

    graph_data = torch.load(DATASET_PATH)

    print("Total graph samples:", len(graph_data))

    return graph_data


# ----------------------------------------------------
# TRAIN
# ----------------------------------------------------

def train():

    dataset = load_dataset()

    model = GNNCollisionPredictor().to(DEVICE)

    criterion = nn.BCELoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=LR
    )

    metrics = []

    print("Starting GNN collision training...")

    for epoch in range(EPOCHS):

        model.train()

        total_loss = 0

        loop = tqdm(dataset)

        for graph in loop:

            node_features = graph["node_features"].to(DEVICE)
            adj = graph["adjacency"].to(DEVICE)

            labels = graph["labels"]

            outputs = model(node_features, adj)

            loss = 0
            count = 0

            for pred, label in zip(outputs, labels):

                y_pred = pred["collision_probability"]

                y_true = torch.tensor(
                    [label["collision"]],
                    dtype=torch.float32
                ).to(DEVICE)

                loss += criterion(y_pred, y_true)

                count += 1

            if count == 0:
                continue

            loss = loss / count

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

            loop.set_description(f"Epoch {epoch+1}/{EPOCHS}")
            loop.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(dataset)

        print(f"Epoch {epoch+1} Loss: {avg_loss:.6f}")

        metrics.append({
            "epoch": epoch + 1,
            "loss": avg_loss
        })

        torch.save(model.state_dict(), MODEL_SAVE_PATH)

    pd.DataFrame(metrics).to_csv(METRICS_PATH, index=False)

    print("Training complete.")
    print("Model saved:", MODEL_SAVE_PATH)


# ----------------------------------------------------

if __name__ == "__main__":

    train()