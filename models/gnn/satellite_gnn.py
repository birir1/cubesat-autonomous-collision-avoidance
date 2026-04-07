"""
Graph Neural Network for Satellite Neighbor Interactions

This model captures dynamic neighbor relationships and interactions among satellites
using graph neural networks. It models satellites as nodes in a graph where edges
represent proximity, communication, or interaction relationships.

Key Features:
- Dynamic graph construction based on satellite positions
- GNN layers to aggregate neighbor information
- Temporal modeling of graph evolution
- Integration with trajectory and vision features
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.data import Data, Batch
from torch_geometric.utils import dense_to_sparse
import numpy as np


class SatelliteGNN(nn.Module):
    """
    Graph Neural Network for modeling satellite neighbor interactions.

    The model takes satellite states as node features and proximity relationships
    as edges, then learns to aggregate neighbor information for better prediction
    of collision risk and trajectory evolution.
    """

    def __init__(
        self,
        node_dim=6,  # position (3) + velocity (3)
        edge_dim=3,  # relative distance, velocity, angle
        hidden_dim=64,
        output_dim=32,
        num_layers=3,
        gnn_type='gcn',  # 'gcn', 'gat', 'sage'
        dropout=0.1
    ):
        super().__init__()

        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.gnn_type = gnn_type

        # Node feature projection
        self.node_proj = nn.Linear(node_dim, hidden_dim)

        # Edge feature projection
        self.edge_proj = nn.Linear(edge_dim, hidden_dim)

        # GNN layers
        self.gnn_layers = nn.ModuleList()
        for i in range(num_layers):
            in_dim = hidden_dim if i > 0 else hidden_dim
            out_dim = hidden_dim

            if gnn_type == 'gcn':
                self.gnn_layers.append(GCNConv(in_dim, out_dim))
            elif gnn_type == 'gat':
                self.gnn_layers.append(GATConv(in_dim, out_dim // 4, heads=4))
            elif gnn_type == 'sage':
                self.gnn_layers.append(SAGEConv(in_dim, out_dim))
            else:
                raise ValueError(f"Unknown GNN type: {gnn_type}")

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

        # Activation and dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def build_graph(self, positions, velocities, communication_range=500.0):
        """
        Build dynamic graph from satellite positions and velocities.

        Args:
            positions: (num_satellites, 3) tensor of positions
            velocities: (num_satellites, 3) tensor of velocities
            communication_range: maximum distance for edge creation

        Returns:
            edge_index: (2, num_edges) tensor
            edge_attr: (num_edges, edge_dim) tensor
        """
        num_satellites = positions.shape[0]

        # Compute pairwise distances
        pos_diff = positions.unsqueeze(0) - positions.unsqueeze(1)  # (N, N, 3)
        distances = torch.norm(pos_diff, dim=2)  # (N, N)

        # Compute relative velocities
        vel_diff = velocities.unsqueeze(0) - velocities.unsqueeze(1)  # (N, N, 3)
        rel_velocities = torch.norm(vel_diff, dim=2)  # (N, N)

        # Compute angles between relative position and velocity
        pos_unit = pos_diff / (distances.unsqueeze(-1) + 1e-8)
        vel_unit = vel_diff / (rel_velocities.unsqueeze(-1) + 1e-8)
        cos_angles = torch.sum(pos_unit * vel_unit, dim=2)  # (N, N)

        # Create adjacency matrix based on communication range
        adj_matrix = (distances < communication_range).float()

        # Remove self-loops
        adj_matrix.fill_diagonal_(0)

        # Convert to edge index and attributes
        edge_index = dense_to_sparse(adj_matrix)[0]  # (2, num_edges)

        # Edge attributes: [normalized_distance, relative_velocity, cos_angle]
        edge_attr = torch.stack([
            distances[edge_index[0], edge_index[1]] / communication_range,
            rel_velocities[edge_index[0], edge_index[1]] / 10.0,  # normalize velocity
            cos_angles[edge_index[0], edge_index[1]]
        ], dim=1)

        return edge_index, edge_attr

    def forward(self, positions, velocities, communication_range=500.0):
        """
        Forward pass through the GNN.

        Args:
            positions: (batch_size, num_satellites, 3) or (num_satellites, 3)
            velocities: (batch_size, num_satellites, 3) or (num_satellites, 3)
            communication_range: communication range for graph construction

        Returns:
            node_embeddings: (batch_size, num_satellites, output_dim)
        """
        # Handle batch dimension
        if positions.dim() == 2:
            positions = positions.unsqueeze(0)
            velocities = velocities.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        batch_size = positions.shape[0]
        num_satellites = positions.shape[1]

        # Process each batch item
        all_node_embeddings = []

        for b in range(batch_size):
            pos_b = positions[b]  # (num_satellites, 3)
            vel_b = velocities[b]  # (num_satellites, 3)

            # Build graph
            edge_index, edge_attr = self.build_graph(pos_b, vel_b, communication_range)

            # Node features: concatenate position and velocity
            node_features = torch.cat([pos_b, vel_b], dim=1)  # (num_satellites, 6)

            # Project node and edge features
            x = self.node_proj(node_features)
            edge_attr = self.edge_proj(edge_attr)

            # Apply GNN layers
            for gnn_layer in self.gnn_layers:
                if isinstance(gnn_layer, GATConv):
                    x = gnn_layer(x, edge_index, edge_attr=edge_attr)
                else:
                    x = gnn_layer(x, edge_index)
                x = self.relu(x)
                x = self.dropout(x)

            # Output projection
            node_embeddings = self.output_proj(x)  # (num_satellites, output_dim)
            all_node_embeddings.append(node_embeddings)

        # Stack batch
        output = torch.stack(all_node_embeddings, dim=0)  # (batch_size, num_satellites, output_dim)

        if squeeze_output:
            output = output.squeeze(0)

        return output


class TemporalSatelliteGNN(nn.Module):
    """
    Temporal GNN that processes sequences of satellite states over time.
    """

    def __init__(self, gnn_config, temporal_config):
        super().__init__()

        self.gnn = SatelliteGNN(**gnn_config)

        # Temporal aggregation (simple mean pooling for now)
        self.temporal_pool = nn.AdaptiveAvgPool1d(1)

        # Final output projection
        self.final_proj = nn.Linear(gnn_config['output_dim'], 1)

    def forward(self, position_sequence, velocity_sequence, communication_range=500.0):
        """
        Args:
            position_sequence: (batch, time_steps, num_satellites, 3)
            velocity_sequence: (batch, time_steps, num_satellites, 3)

        Returns:
            risk_predictions: (batch, 1)
        """
        batch_size, time_steps, num_satellites, _ = position_sequence.shape

        # Process each time step
        temporal_embeddings = []
        for t in range(time_steps):
            pos_t = position_sequence[:, t]  # (batch, num_satellites, 3)
            vel_t = velocity_sequence[:, t]  # (batch, num_satellites, 3)

            # Get GNN embeddings for this timestep
            embeddings_t = self.gnn(pos_t, vel_t, communication_range)  # (batch, num_satellites, output_dim)

            # Pool across satellites (take mean for global representation)
            global_embedding_t = embeddings_t.mean(dim=1)  # (batch, output_dim)
            temporal_embeddings.append(global_embedding_t)

        # Stack temporal embeddings
        temporal_embeddings = torch.stack(temporal_embeddings, dim=1)  # (batch, time_steps, output_dim)

        # Temporal pooling
        pooled = self.temporal_pool(temporal_embeddings.transpose(1, 2))  # (batch, output_dim, 1)
        pooled = pooled.squeeze(-1)  # (batch, output_dim)

        # Final prediction
        risk_pred = self.final_proj(pooled)  # (batch, 1)

        return risk_pred


if __name__ == "__main__":
    # Test the GNN
    model = SatelliteGNN()

    # Sample data
    positions = torch.randn(10, 3)  # 10 satellites
    velocities = torch.randn(10, 3)

    embeddings = model(positions, velocities)
    print(f"Node embeddings shape: {embeddings.shape}")  # Should be (10, 32)