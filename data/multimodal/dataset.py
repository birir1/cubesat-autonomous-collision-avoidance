import torch
import numpy as np
from torch.utils.data import Dataset
from torch_geometric.data import Data
import pickle


class MultimodalSatelliteDataset(Dataset):
    """
    Multimodal dataset for satellite collision prediction.
    Combines trajectory sequences, graph structures, and vision data.
    """

    def __init__(self, data_path=None, split='train', transform=None):
        """
        Initialize multimodal dataset.

        Args:
            data_path: Path to processed multimodal data
            split: 'train', 'val', or 'test'
            transform: Optional data transformations
        """
        self.split = split
        self.transform = transform

        if data_path:
            with open(data_path, 'rb') as f:
                self.data = pickle.load(f)
        else:
            # Generate synthetic data for testing
            self.data = self._generate_synthetic_data()

    def _generate_synthetic_data(self, n_samples=1000):
        """Generate synthetic multimodal data for testing."""
        data = []

        for i in range(n_samples):
            # Trajectory data (sequence of states)
            trajectory = torch.randn(20, 6)  # 20 timesteps, 6D state

            # Graph data (satellite positions and edges)
            n_satellites = np.random.randint(5, 15)
            positions = torch.randn(n_satellites, 3)
            velocities = torch.randn(n_satellites, 3)

            # Create edges based on proximity
            edges = []
            edge_attr = []
            for j in range(n_satellites):
                for k in range(j+1, n_satellites):
                    dist = torch.norm(positions[j] - positions[k])
                    if dist < 2.0:  # Proximity threshold
                        edges.append([j, k])
                        edges.append([k, j])
                        edge_attr.extend([dist.item()] * 2)

            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attr, dtype=torch.float)

            # Vision data (simulated image features)
            vision_features = torch.randn(2048)  # ResNet features

            # Label (collision risk)
            risk = torch.rand(1).item()

            sample = {
                'trajectory': trajectory,
                'positions': positions,
                'velocities': velocities,
                'edge_index': edge_index,
                'edge_attr': edge_attr,
                'vision_features': vision_features,
                'risk': risk
            }

            data.append(sample)

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        # Convert to tensors
        trajectory = torch.tensor(sample['trajectory'], dtype=torch.float32)
        positions = torch.tensor(sample['positions'], dtype=torch.float32)
        velocities = torch.tensor(sample['velocities'], dtype=torch.float32)
        edge_index = sample['edge_index']
        edge_attr = torch.tensor(sample['edge_attr'], dtype=torch.float32)
        vision_features = torch.tensor(sample['vision_features'], dtype=torch.float32)
        risk = torch.tensor(sample['risk'], dtype=torch.float32)

        # Create PyG Data object for graph
        graph_data = Data(
            x=torch.cat([positions, velocities], dim=1),
            edge_index=edge_index,
            edge_attr=edge_attr
        )

        if self.transform:
            trajectory, graph_data, vision_features, risk = self.transform(
                trajectory, graph_data, vision_features, risk
            )

        return {
            'trajectory': trajectory,
            'graph': graph_data,
            'vision': vision_features,
            'risk': risk
        }