"""
Collision Dataset Generator

Converts predicted trajectories into graph datasets
for training the GNN collision predictor.

Output:
results/collision_dataset/graph_data.pt
"""

import os
import torch
import pandas as pd
import numpy as np
from itertools import combinations
from tqdm import tqdm


# ----------------------------------------------------
# CONFIG
# ----------------------------------------------------

TRAJECTORY_FILE = "results/tracking_outputs/tracked_objects.csv"

OUTPUT_PATH = "results/collision_dataset/graph_data.pt"

DISTANCE_THRESHOLD = 20.0

os.makedirs("results/collision_dataset", exist_ok=True)


# ----------------------------------------------------
# LOAD TRAJECTORY DATA
# ----------------------------------------------------

def load_data():

    print("Loading trajectory dataset...")

    df = pd.read_csv(TRAJECTORY_FILE)

    return df


# ----------------------------------------------------
# BUILD NODE FEATURES
# ----------------------------------------------------

def build_node_features(frame_df):

    features = frame_df[["pos_x", "pos_y", "vel_x", "vel_y"]].values

    return torch.tensor(features, dtype=torch.float32)


# ----------------------------------------------------
# BUILD ADJACENCY MATRIX
# ----------------------------------------------------

def build_adjacency_matrix(node_features):

    N = node_features.shape[0]

    adj = torch.zeros((N, N))

    for i in range(N):
        for j in range(N):

            if i == j:
                continue

            dist = torch.norm(
                node_features[i, :2] -
                node_features[j, :2]
            )

            if dist < DISTANCE_THRESHOLD:

                adj[i, j] = 1

    return adj


# ----------------------------------------------------
# GENERATE COLLISION LABELS
# ----------------------------------------------------

def generate_collision_labels(node_features):

    labels = []

    N = node_features.shape[0]

    for i, j in combinations(range(N), 2):

        dist = torch.norm(
            node_features[i, :2] -
            node_features[j, :2]
        )

        label = 1 if dist < DISTANCE_THRESHOLD else 0

        labels.append({
            "pair": (i, j),
            "collision": label
        })

    return labels


# ----------------------------------------------------
# BUILD GRAPH DATASET
# ----------------------------------------------------

def build_graph_dataset(df):

    graph_data = []

    frames = sorted(df["frame"].unique())

    print("Generating graph dataset...")

    for frame in tqdm(frames):

        frame_df = df[df["frame"] == frame]

        if len(frame_df) < 2:
            continue

        node_features = build_node_features(frame_df)

        adj = build_adjacency_matrix(node_features)

        labels = generate_collision_labels(node_features)

        graph_data.append({

            "node_features": node_features,
            "adjacency": adj,
            "labels": labels
        })

    return graph_data


# ----------------------------------------------------
# MAIN
# ----------------------------------------------------

def main():

    df = load_data()

    graph_data = build_graph_dataset(df)

    torch.save(graph_data, OUTPUT_PATH)

    print("Collision dataset saved to:", OUTPUT_PATH)


# ----------------------------------------------------

if __name__ == "__main__":

    main()