"""
Graph Builder for Satellite Neighbor Relationships

Constructs dynamic graphs representing satellite interactions for GNN-based collision avoidance.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import logging
from pathlib import Path
import yaml
from scipy.spatial.distance import cdist

class SatelliteGraphBuilder:
    """
    Build dynamic neighbor graphs for satellite constellation modeling.
    """

    def __init__(self, config_path: str = 'configs/data_config.yaml'):
        """
        Initialize graph builder.

        Args:
            config_path: Path to configuration file
        """
        self.logger = logging.getLogger(__name__)

        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Graph parameters
        self.max_neighbors = self.config['graph']['max_neighbors']
        self.distance_threshold_km = self.config['communication']['range_km']
        self.edge_features = self.config['graph']['edge_features']

    def build_graph_from_positions(self, positions: np.ndarray,
                                 velocities: Optional[np.ndarray] = None,
                                 satellite_ids: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]]]:
        """
        Build graph from satellite positions.

        Args:
            positions: Shape (num_satellites, 3) - [x, y, z] in km
            velocities: Optional shape (num_satellites, 3) - velocities in km/s
            satellite_ids: Optional list of satellite identifiers

        Returns:
            Tuple of (adjacency_matrix, node_features, edge_list)
        """
        num_satellites = positions.shape[0]

        # Compute pairwise distances
        distances = cdist(positions, positions)

        # Build adjacency matrix based on communication range
        adjacency = np.zeros((num_satellites, num_satellites), dtype=np.float32)

        for i in range(num_satellites):
            # Find neighbors within communication range
            neighbor_indices = np.where(distances[i] <= self.distance_threshold_km)[0]
            neighbor_indices = neighbor_indices[neighbor_indices != i]  # Exclude self

            # Limit to max neighbors
            if len(neighbor_indices) > self.max_neighbors:
                # Sort by distance and take closest
                sorted_indices = neighbor_indices[np.argsort(distances[i, neighbor_indices])]
                neighbor_indices = sorted_indices[:self.max_neighbors]

            # Set adjacency weights (inverse distance)
            for j in neighbor_indices:
                if distances[i, j] > 0:  # Avoid division by zero
                    adjacency[i, j] = 1.0 / distances[i, j]
                    adjacency[j, i] = adjacency[i, j]  # Symmetric

        # Build node features
        node_features = []

        for i in range(num_satellites):
            features = [
                positions[i, 0], positions[i, 1], positions[i, 2],  # Position
            ]

            if velocities is not None:
                features.extend([
                    velocities[i, 0], velocities[i, 1], velocities[i, 2]  # Velocity
                ])
            else:
                features.extend([0.0, 0.0, 0.0])  # Zero velocity

            # Add satellite ID as one-hot if available
            if satellite_ids:
                # Simple ID encoding (can be improved)
                id_hash = hash(satellite_ids[i]) % 100
                features.append(float(id_hash))
            else:
                features.append(float(i))

            node_features.append(features)

        node_features = np.array(node_features, dtype=np.float32)

        # Build edge list for sparse representation
        edge_list = []
        for i in range(num_satellites):
            for j in range(i + 1, num_satellites):  # Upper triangle only
                if adjacency[i, j] > 0:
                    edge_list.append((i, j))

        self.logger.info(f"Built graph with {num_satellites} nodes, {len(edge_list)} edges")
        self.logger.info(f"Average degree: {adjacency.sum(axis=1).mean():.2f}")

        return adjacency, node_features, edge_list

    def build_random_graph(self, num_satellites: int,
                          connectivity: float = 0.3) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build a random graph for testing/synthetic data generation.

        Args:
            num_satellites: Number of satellites/nodes
            connectivity: Probability of edge between any two nodes

        Returns:
            Tuple of (adjacency_matrix, node_features)
        """
        # Create random positions in LEO
        positions = np.random.uniform(-1000, 1000, (num_satellites, 3))
        positions[:, 2] = np.abs(positions[:, 2])  # Ensure positive altitude

        # Create random velocities
        velocities = np.random.normal(0, 1, (num_satellites, 3))

        # Build graph from positions
        adjacency, node_features, _ = self.build_graph_from_positions(positions, velocities)

        return adjacency, node_features

    def build_temporal_graph_sequence(self, trajectory_data: np.ndarray,
                                    satellite_ids: Optional[List[str]] = None,
                                    time_window: int = 5) -> List[Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]]]]:
        """
        Build sequence of graphs from trajectory data.

        Args:
            trajectory_data: Shape (num_timesteps, num_satellites, 6) - [pos, vel]
            satellite_ids: Optional satellite identifiers
            time_window: Number of timesteps to consider for temporal features

        Returns:
            List of (adjacency, node_features, edge_list) tuples for each timestep
        """
        num_timesteps, num_satellites, feature_dim = trajectory_data.shape

        if feature_dim < 6:
            raise ValueError(f"Trajectory data must have at least 6 features (pos+vel), got {feature_dim}")

        graph_sequence = []

        for t in range(num_timesteps):
            # Extract positions and velocities at current timestep
            positions = trajectory_data[t, :, :3]  # (num_satellites, 3)
            velocities = trajectory_data[t, :, 3:6]  # (num_satellites, 3)

            # Add temporal features (velocity changes over time window)
            if t >= time_window - 1:
                # Compute average velocity over time window
                window_velocities = trajectory_data[t-time_window+1:t+1, :, 3:6]
                avg_velocity = np.mean(window_velocities, axis=0)
                temporal_features = avg_velocity - velocities  # Velocity change
            else:
                temporal_features = np.zeros_like(velocities)

            # Build graph
            adjacency, node_features, edge_list = self.build_graph_from_positions(
                positions, velocities, satellite_ids
            )

            # Add temporal features to node features
            extended_features = np.concatenate([node_features, temporal_features], axis=1)

            graph_sequence.append((adjacency, extended_features, edge_list))

        self.logger.info(f"Built temporal graph sequence with {len(graph_sequence)} timesteps")
        return graph_sequence

    def compute_graph_metrics(self, adjacency: np.ndarray) -> Dict[str, float]:
        """
        Compute graph structure metrics.

        Args:
            adjacency: Adjacency matrix

        Returns:
            Dictionary of graph metrics
        """
        num_nodes = adjacency.shape[0]

        # Degree distribution
        degrees = adjacency.sum(axis=1)
        avg_degree = float(degrees.mean())
        max_degree = int(degrees.max())
        min_degree = int(degrees.min())

        # Clustering coefficient (simplified)
        triangles = 0
        possible_triangles = 0
        for i in range(num_nodes):
            neighbors_i = np.where(adjacency[i] > 0)[0]
            for j in neighbors_i:
                if j > i:
                    neighbors_j = np.where(adjacency[j] > 0)[0]
                    common = np.intersect1d(neighbors_i, neighbors_j)
                    triangles += len(common)
                    possible_triangles += len(neighbors_i) - 1  # Exclude j

        clustering_coeff = triangles / max(possible_triangles, 1)

        # Connected components (simplified)
        visited = np.zeros(num_nodes, dtype=bool)
        components = 0

        for i in range(num_nodes):
            if not visited[i]:
                components += 1
                # Simple DFS
                stack = [i]
                while stack:
                    node = stack.pop()
                    if not visited[node]:
                        visited[node] = True
                        neighbors = np.where(adjacency[node] > 0)[0]
                        stack.extend(neighbors[~np.isin(neighbors, np.where(visited)[0])])

        metrics = {
            'num_nodes': num_nodes,
            'num_edges': int(np.sum(adjacency > 0) / 2),  # Undirected
            'avg_degree': avg_degree,
            'max_degree': max_degree,
            'min_degree': min_degree,
            'clustering_coefficient': clustering_coeff,
            'connected_components': components,
            'sparsity': float(np.sum(adjacency == 0) / adjacency.size)
        }

        return metrics

    def build_collision_focused_graph(self, positions: np.ndarray,
                                    target_satellite_idx: int,
                                    collision_candidates: List[int]) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]]]:
        """
        Build graph focused on collision scenario.

        Args:
            positions: All satellite positions
            target_satellite_idx: Index of satellite of interest
            collision_candidates: Indices of potential collision satellites

        Returns:
            Tuple of (adjacency, node_features, edge_list) focused on collision scenario
        """
        # Include target and collision candidates
        relevant_indices = [target_satellite_idx] + collision_candidates
        relevant_positions = positions[relevant_indices]

        # Build graph with higher connectivity for collision candidates
        adjacency, node_features, edge_list = self.build_graph_from_positions(relevant_positions)

        # Enhance edges between target and collision candidates
        for i, idx in enumerate(relevant_indices):
            if idx == target_satellite_idx:
                target_local_idx = i
                break

        # Mark collision candidate nodes with special feature
        collision_flags = np.zeros(len(relevant_indices))
        for i, idx in enumerate(relevant_indices):
            if idx in collision_candidates:
                collision_flags[i] = 1.0

        # Add collision flags to node features
        node_features = np.concatenate([node_features, collision_flags.reshape(-1, 1)], axis=1)

        self.logger.info(f"Built collision-focused graph with {len(relevant_indices)} nodes")
        return adjacency, node_features, edge_list

    def save_graph_data(self, graph_data: Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]]],
                       output_path: str) -> None:
        """
        Save graph data to file.

        Args:
            graph_data: Tuple of (adjacency, node_features, edge_list)
            output_path: Output file path
        """
        adjacency, node_features, edge_list = graph_data

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Save as numpy arrays
        np.savez(output_path,
                adjacency=adjacency,
                node_features=node_features,
                edge_list=np.array(edge_list))

        self.logger.info(f"Saved graph data to {output_path}")

    def load_graph_data(self, input_path: str) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]]]:
        """
        Load graph data from file.

        Args:
            input_path: Input file path

        Returns:
            Tuple of (adjacency, node_features, edge_list)
        """
        data = np.load(input_path)
        adjacency = data['adjacency']
        node_features = data['node_features']
        edge_list = data['edge_list'].tolist()

        self.logger.info(f"Loaded graph data from {input_path}")
        return adjacency, node_features, edge_list

    def visualize_graph(self, adjacency: np.ndarray,
                       positions: np.ndarray,
                       output_path: Optional[str] = None) -> None:
        """
        Visualize satellite graph (optional, requires matplotlib).

        Args:
            adjacency: Adjacency matrix
            positions: Node positions for visualization
            output_path: Optional path to save plot
        """
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D

            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')

            # Plot nodes
            ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                      c='blue', s=50, alpha=0.7)

            # Plot edges
            num_nodes = adjacency.shape[0]
            for i in range(num_nodes):
                for j in range(i + 1, num_nodes):
                    if adjacency[i, j] > 0:
                        ax.plot([positions[i, 0], positions[j, 0]],
                               [positions[i, 1], positions[j, 1]],
                               [positions[i, 2], positions[j, 2]],
                               'gray', alpha=0.3, linewidth=0.5)

            ax.set_xlabel('X (km)')
            ax.set_ylabel('Y (km)')
            ax.set_zlabel('Z (km)')
            ax.set_title('Satellite Constellation Graph')

            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"Graph visualization saved to {output_path}")
            else:
                plt.show()

        except ImportError:
            self.logger.warning("Matplotlib not available for graph visualization")