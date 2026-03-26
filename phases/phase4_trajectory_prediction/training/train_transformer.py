"""
Training Script for Transformer Trajectory Prediction

Pipeline:
Tracked Objects CSV -> Sequence Dataset -> Transformer Training

Input:
results/tracking_outputs/tracked_objects.csv

Outputs:
results/saved_models/transformer_trajectory_model.pth
results/metrics/transformer_training_metrics.csv
"""

import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from phases.phase4_trajectory_prediction.models.transformer_predictor import TransformerTrajectoryPredictor


# -----------------------------------------------------
# CONFIG
# -----------------------------------------------------

DATA_PATH = "results/tracking_outputs/tracked_objects.csv"

MODEL_SAVE_PATH = "results/saved_models/transformer_trajectory_model.pth"
METRICS_PATH = "results/metrics/transformer_training_metrics.csv"

SEQUENCE_LENGTH = 20
PREDICTION_HORIZON = 10

BATCH_SIZE = 32
EPOCHS = 50
LR = 1e-4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs("results/saved_models", exist_ok=True)
os.makedirs("results/metrics", exist_ok=True)


# -----------------------------------------------------
# DATASET
# -----------------------------------------------------

class TrajectoryDataset(Dataset):

    def __init__(self, dataframe):

        self.samples = []

        grouped = dataframe.groupby("track_id")

        for _, track in grouped:

            track = track.sort_values("frame")

            data = track[["pos_x","pos_y","vel_x","vel_y"]].values

            for i in range(len(data) - SEQUENCE_LENGTH - PREDICTION_HORIZON):

                seq = data[i:i+SEQUENCE_LENGTH]

                future = data[
                    i+SEQUENCE_LENGTH:
                    i+SEQUENCE_LENGTH+PREDICTION_HORIZON
                ]

                future_pos = future[:, :2]

                self.samples.append((seq, future_pos))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        seq, target = self.samples[idx]

        seq = torch.tensor(seq, dtype=torch.float32)
        target = torch.tensor(target, dtype=torch.float32)

        return seq, target


# -----------------------------------------------------
# LOAD DATA
# -----------------------------------------------------

def load_dataset():

    print("Loading trajectory dataset...")

    df = pd.read_csv(DATA_PATH)

    dataset = TrajectoryDataset(df)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2
    )

    print("Total samples:", len(dataset))

    return loader


# -----------------------------------------------------
# TRAIN
# -----------------------------------------------------

def train():

    dataloader = load_dataset()

    model = TransformerTrajectoryPredictor(
        prediction_horizon=PREDICTION_HORIZON
    ).to(DEVICE)

    criterion = nn.MSELoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=LR
    )

    metrics = []

    print("Starting Transformer training...")

    for epoch in range(EPOCHS):

        model.train()

        epoch_loss = 0

        loop = tqdm(dataloader)

        for seq, target in loop:

            seq = seq.to(DEVICE)
            target = target.to(DEVICE)

            pred = model(seq)

            loss = criterion(pred, target)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            epoch_loss += loss.item()

            loop.set_description(f"Epoch {epoch+1}/{EPOCHS}")
            loop.set_postfix(loss=loss.item())

        avg_loss = epoch_loss / len(dataloader)

        print(f"Epoch {epoch+1} Loss: {avg_loss:.6f}")

        metrics.append({
            "epoch": epoch+1,
            "loss": avg_loss
        })

        torch.save(model.state_dict(), MODEL_SAVE_PATH)

    pd.DataFrame(metrics).to_csv(METRICS_PATH, index=False)

    print("Training complete.")
    print("Model saved:", MODEL_SAVE_PATH)


# -----------------------------------------------------

if __name__ == "__main__":
    train()