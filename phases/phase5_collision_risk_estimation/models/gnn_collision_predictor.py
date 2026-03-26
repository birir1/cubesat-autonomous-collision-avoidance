"""
Graph Neural Network Collision Risk Predictor

This model predicts the probability of collision between
space objects based on predicted trajectories.

Nodes:
satellites / debris

Edges:
relative motion relationships

Node features:
[x, y, vx, vy]

Output:
collision probability
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------------------------------------
# GRAPH CONVOLUTION LAYER
# -------------------------------------------------------

class GraphConvLayer(nn.Module):

    def __init__(self, in_features, out_features):

        super().__init__()

        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adj):

        """
        x: node features (N, F)
        adj: adjacency matrix (N, N)
        """

        agg = torch.matmul(adj, x)

        out = self.linear(agg)

        return F.relu(out)


# -------------------------------------------------------
# COLLISION GNN MODEL
# -------------------------------------------------------

class GNNCollisionPredictor(nn.Module):

    def __init__(
        self,
        input_dim=4,
        hidden_dim=64,
        num_layers=3
    ):
        super().__init__()

        self.layers = nn.ModuleList()

        self.layers.append(
            GraphConvLayer(input_dim, hidden_dim)
        )

        for _ in range(num_layers - 1):

            self.layers.append(
                GraphConvLayer(hidden_dim, hidden_dim)
            )

        self.edge_predictor = nn.Sequential(

            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    # ---------------------------------------------------

    def forward(self, node_features, adj_matrix):

        """
        node_features: (N, 4)
        adj_matrix: (N, N)
        """

        x = node_features

        for layer in self.layers:

            x = layer(x, adj_matrix)

        collisions = []

        N = x.shape[0]

        for i in range(N):
            for j in range(i + 1, N):

                pair = torch.cat([x[i], x[j]])

                prob = self.edge_predictor(pair)

                collisions.append({
                    "object_a": i,
                    "object_b": j,
                    "collision_probability": prob
                })

        return collisions


# -------------------------------------------------------
# TRAJECTORY-AWARE GNN (Enhanced)
# -------------------------------------------------------

class TrajectoryGNNCollisionPredictor(nn.Module):

    def __init__(
        self,
        input_dim=4,
        trajectory_length=10,
        hidden_dim=64,
        num_layers=3,
        nhead=4
    ):
        super().__init__()

        self.trajectory_length = trajectory_length

        # Process trajectory sequences with transformer
        self.trajectory_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=input_dim,
                nhead=min(nhead, input_dim // 2),
                dim_feedforward=hidden_dim,
                batch_first=True
            ),
            num_layers=1
        )

        # Graph convolution layers
        self.layers = nn.ModuleList()
        self.layers.append(GraphConvLayer(hidden_dim, hidden_dim))

        for _ in range(num_layers - 1):
            self.layers.append(GraphConvLayer(hidden_dim, hidden_dim))

        # Enhanced edge predictor with temporal features
        self.edge_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2 + trajectory_length, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, trajectories, adj_matrix):

        """
        trajectories: (N, T, 4) where T is trajectory length
        adj_matrix: (N, N)
        """

        # Encode trajectories
        trajectory_features = self.trajectory_encoder(trajectories)  # (N, T, hidden_dim)

        # Aggregate temporal features
        node_features = trajectory_features.mean(dim=1)  # (N, hidden_dim)

        # Graph convolution
        x = node_features
        for layer in self.layers:
            x = layer(x, adj_matrix)

        collisions = []
        N = x.shape[0]

        for i in range(N):
            for j in range(i + 1, N):

                # Concatenate node features
                pair_features = torch.cat([x[i], x[j]], dim=0)  # (2*hidden_dim,)

                # Add trajectory distance features
                traj_i = trajectories[i]  # (T, 4)
                traj_j = trajectories[j]  # (T, 4)

                # Compute minimum distance over trajectory
                distances = torch.sqrt(
                    ((traj_i[:, :2] - traj_j[:, :2]) ** 2).sum(dim=1)
                )
                min_distance = distances.min()

                # Create distance history feature
                distance_history = torch.sigmoid(10.0 * (distances - 5.0))  # Normalize

                # Combine features
                edge_features = torch.cat([
                    pair_features,
                    distance_history
                ])

                prob = self.edge_predictor(edge_features)

                collisions.append({
                    "object_a": i,
                    "object_b": j,
                    "collision_probability": prob.item(),
                    "min_distance": min_distance.item(),
                    "distance_history": distance_history.tolist()
                })

        return collisions