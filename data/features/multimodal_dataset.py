"""
Multimodal Dataset Builder for CubeSat Collision Avoidance

Combines trajectory, graph, and vision data into unified datasets.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Any, Union, Iterator
from pathlib import Path
import logging
import yaml
import json
from datetime import datetime
import h5py

from .feature_pipeline import FeaturePipeline
from .vision_processor import VisionProcessor
from .graph_builder import SatelliteGraphBuilder
from ..processed.trajectory_from_tle import TLETrajectoryGenerator
from ..processed.conjunction_processor import ConjunctionProcessor

class MultimodalDataset(Dataset):
    """
    PyTorch Dataset for multimodal collision avoidance data.
    """

    def __init__(self, data_dict: Dict[str, Any], transform: Optional[Any] = None):
        """
        Initialize multimodal dataset.

        Args:
            data_dict: Dictionary containing all modality data
            transform: Optional transform to apply
        """
        self.data = data_dict
        self.transform = transform
        self.length = len(data_dict['labels'])

        # Validate data consistency
        self._validate_data_consistency()

    def _validate_data_consistency(self) -> None:
        """Validate that all modalities have consistent sample counts."""
        modalities = ['trajectory', 'graph', 'vision', 'conjunction', 'labels']
        lengths = []

        for modality in modalities:
            if modality in self.data and self.data[modality] is not None:
                if isinstance(self.data[modality], (list, np.ndarray, torch.Tensor)):
                    lengths.append(len(self.data[modality]))
                elif isinstance(self.data[modality], dict):
                    # For dict data, check first value
                    first_key = next(iter(self.data[modality].keys()))
                    lengths.append(len(self.data[modality][first_key]))

        if len(set(lengths)) > 1:
            raise ValueError(f"Inconsistent sample counts across modalities: {dict(zip(modalities, lengths))}")

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single multimodal sample.

        Args:
            idx: Sample index

        Returns:
            Dictionary containing all modalities for the sample
        """
        sample = {}

        # Extract data from each modality
        if 'trajectory' in self.data and self.data['trajectory'] is not None:
            sample['trajectory'] = self.data['trajectory'][idx]

        if 'graph' in self.data and self.data['graph'] is not None:
            sample['graph'] = {
                'adjacency': self.data['graph']['adjacency'][idx],
                'node_features': self.data['graph']['node_features'][idx]
            }

        if 'vision' in self.data and self.data['vision'] is not None:
            sample['vision'] = self.data['vision'][idx]

        if 'conjunction' in self.data and self.data['conjunction'] is not None:
            # Conjunction data is stored as dict of arrays
            conj_sample = {}
            for key, value in self.data['conjunction'].items():
                conj_sample[key] = value[idx]
            sample['conjunction'] = conj_sample

        # Labels
        if 'labels' in self.data:
            sample['label'] = self.data['labels'][idx]

        # Additional metadata
        if 'metadata' in self.data and idx < len(self.data['metadata']):
            sample['metadata'] = self.data['metadata'][idx]

        # Apply transform if provided
        if self.transform:
            sample = self.transform(sample)

        return sample

class MultimodalDatasetBuilder:
    """
    Builder for creating multimodal datasets from various data sources.
    """

    def __init__(self, config_path: str = 'configs/data_config.yaml'):
        """
        Initialize dataset builder.

        Args:
            config_path: Path to configuration file
        """
        self.logger = logging.getLogger(__name__)

        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Initialize component processors
        self.feature_pipeline = FeaturePipeline(config_path)
        self.vision_processor = VisionProcessor(config_path)
        self.graph_builder = SatelliteGraphBuilder(config_path)
        self.trajectory_generator = TLETrajectoryGenerator(config_path)
        self.conjunction_processor = ConjunctionProcessor(config_path)

        # Dataset configuration
        self.dataset_config = self.config['dataset']

    def build_from_components(self, trajectory_data: Optional[np.ndarray] = None,
                            graph_data: Optional[Dict[str, np.ndarray]] = None,
                            vision_data: Optional[torch.Tensor] = None,
                            conjunction_data: Optional[pd.DataFrame] = None,
                            labels: Optional[np.ndarray] = None) -> MultimodalDataset:
        """
        Build dataset from individual component data.

        Args:
            trajectory_data: Trajectory sequences
            graph_data: Graph adjacency and node features
            vision_data: Vision tensors
            conjunction_data: Conjunction assessment data
            labels: Target labels

        Returns:
            MultimodalDataset instance
        """
        data_dict = {}

        # Add each modality if provided
        if trajectory_data is not None:
            data_dict['trajectory'] = trajectory_data

        if graph_data is not None:
            data_dict['graph'] = graph_data

        if vision_data is not None:
            data_dict['vision'] = vision_data

        if conjunction_data is not None:
            # Convert DataFrame to dict of arrays
            conj_dict = {}
            for col in conjunction_data.columns:
                conj_dict[col] = conjunction_data[col].values
            data_dict['conjunction'] = conj_dict

        if labels is not None:
            data_dict['labels'] = labels

        return MultimodalDataset(data_dict)

    def build_from_files(self, data_dir: str, modalities: List[str] = None) -> MultimodalDataset:
        """
        Build dataset from saved data files.

        Args:
            data_dir: Directory containing saved data
            modalities: List of modalities to include

        Returns:
            MultimodalDataset instance
        """
        if modalities is None:
            modalities = ['trajectory', 'graph', 'vision', 'conjunction']

        data_path = Path(data_dir)
        data_dict = {}

        # Load trajectory data
        if 'trajectory' in modalities:
            traj_file = data_path / 'trajectory_data.h5'
            if traj_file.exists():
                with h5py.File(traj_file, 'r') as f:
                    data_dict['trajectory'] = np.array(f['trajectories'])

        # Load graph data
        if 'graph' in modalities:
            graph_file = data_path / 'graph_data.h5'
            if graph_file.exists():
                with h5py.File(graph_file, 'r') as f:
                    data_dict['graph'] = {
                        'adjacency': np.array(f['adjacency']),
                        'node_features': np.array(f['node_features'])
                    }

        # Load vision data
        if 'vision' in modalities:
            vision_tensors, vision_detections, vision_features = self.vision_processor.load_processed_data(
                data_path / 'vision'
            )
            if vision_tensors is not None:
                data_dict['vision'] = vision_tensors

        # Load conjunction data
        if 'conjunction' in modalities:
            conj_file = data_path / 'conjunction_data.csv'
            if conj_file.exists():
                conj_df = pd.read_csv(conj_file)
                conj_dict = {}
                for col in conj_df.columns:
                    conj_dict[col] = conj_df[col].values
                data_dict['conjunction'] = conj_dict

        # Load labels
        labels_file = data_path / 'labels.npy'
        if labels_file.exists():
            data_dict['labels'] = np.load(labels_file)

        # Load metadata
        metadata_file = data_path / 'metadata.json'
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                data_dict['metadata'] = json.load(f)

        return MultimodalDataset(data_dict)

    def create_synthetic_dataset(self, num_samples: int = 1000,
                               sequence_length: int = 100,
                               num_satellites: int = 10) -> MultimodalDataset:
        """
        Create a synthetic multimodal dataset for testing.

        Args:
            num_samples: Number of samples to generate
            sequence_length: Length of trajectory sequences
            num_satellites: Number of satellites in graph

        Returns:
            MultimodalDataset with synthetic data
        """
        self.logger.info(f"Creating synthetic dataset with {num_samples} samples")

        data_dict = {}

        # Generate trajectory data
        trajectory_data = []
        for _ in range(num_samples):
            # Simulate relative trajectory with some collision scenarios
            trajectory = self._generate_synthetic_trajectory(sequence_length)
            trajectory_data.append(trajectory)
        data_dict['trajectory'] = np.array(trajectory_data)

        # Generate graph data
        adjacency_matrices = []
        node_features_list = []

        for _ in range(num_samples):
            adjacency, node_features = self.graph_builder.build_random_graph(num_satellites)
            adjacency_matrices.append(adjacency)
            node_features_list.append(node_features)

        data_dict['graph'] = {
            'adjacency': np.array(adjacency_matrices),
            'node_features': np.array(node_features_list)
        }

        # Generate vision data
        vision_data = []
        for _ in range(num_samples):
            image, detections = self.vision_processor.simulate_satellite_detection(
                num_satellites=np.random.randint(1, 5)
            )
            vision_data.append(torch.tensor(image))
        data_dict['vision'] = torch.stack(vision_data)

        # Generate conjunction data
        conjunction_data = []
        for _ in range(num_samples):
            conj_sample = self._generate_synthetic_conjunction()
            conjunction_data.append(conj_sample)

        conj_df = pd.DataFrame(conjunction_data)
        conj_dict = {}
        for col in conj_df.columns:
            conj_dict[col] = conj_df[col].values
        data_dict['conjunction'] = conj_dict

        # Generate labels (collision risk: 0=safe, 1=warning, 2=critical)
        labels = []
        for i in range(num_samples):
            # Base risk on minimum distance in trajectory
            min_distance = np.min(np.linalg.norm(trajectory_data[i][:, :3], axis=1))

            if min_distance < 1.0:  # Very close
                risk = 2
            elif min_distance < 5.0:  # Close
                risk = 1
            else:  # Safe
                risk = 0
            labels.append(risk)

        data_dict['labels'] = np.array(labels)

        # Generate metadata
        metadata = []
        for i in range(num_samples):
            meta = {
                'sample_id': i,
                'timestamp': datetime.now().isoformat(),
                'data_type': 'synthetic',
                'risk_level': int(labels[i])
            }
            metadata.append(meta)
        data_dict['metadata'] = metadata

        return MultimodalDataset(data_dict)

    def _generate_synthetic_trajectory(self, sequence_length: int) -> np.ndarray:
        """
        Generate synthetic trajectory data.

        Args:
            sequence_length: Number of time steps

        Returns:
            Trajectory array (seq_len, 6)
        """
        # Start with some initial relative state
        initial_pos = np.random.normal(0, 10, 3)  # km
        initial_vel = np.random.normal(0, 1, 3)  # km/s

        trajectory = []
        pos = initial_pos.copy()
        vel = initial_vel.copy()

        for t in range(sequence_length):
            # Simple orbital dynamics (circular approximation)
            # Add some random perturbations
            pos += vel * 0.1  # Time step of 0.1 hours
            vel += np.random.normal(0, 0.01, 3)  # Small random accelerations

            # Add collision scenario occasionally
            if np.random.random() < 0.1:  # 10% chance
                # Make trajectory approach closer
                pos *= 0.9
                vel *= 0.9

            trajectory.append(np.concatenate([pos, vel]))

        return np.array(trajectory)

    def _generate_synthetic_conjunction(self) -> Dict[str, Any]:
        """
        Generate synthetic conjunction data.

        Returns:
            Conjunction data dictionary
        """
        return {
            'miss_distance_km': np.random.exponential(10),  # Exponential distribution
            'relative_velocity_kms': np.random.normal(0, 2),
            'time_to_tca_hours': np.random.uniform(0, 24),
            'collision_probability': np.random.beta(1, 100),  # Low probability
            'relative_position_x_km': np.random.normal(0, 5),
            'relative_position_y_km': np.random.normal(0, 5),
            'relative_position_z_km': np.random.normal(0, 5),
            'sat1_inclination_deg': np.random.uniform(0, 180),
            'sat2_inclination_deg': np.random.uniform(0, 180),
            'sat1_altitude_km': 6371 + np.random.uniform(400, 2000),  # LEO
            'sat2_altitude_km': 6371 + np.random.uniform(400, 2000)
        }

    def save_dataset(self, dataset: MultimodalDataset, output_dir: str) -> None:
        """
        Save multimodal dataset to disk.

        Args:
            dataset: Dataset to save
            output_dir: Output directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save trajectory data
        if 'trajectory' in dataset.data:
            with h5py.File(output_path / 'trajectory_data.h5', 'w') as f:
                f.create_dataset('trajectories', data=dataset.data['trajectory'])

        # Save graph data
        if 'graph' in dataset.data:
            with h5py.File(output_path / 'graph_data.h5', 'w') as f:
                f.create_dataset('adjacency', data=dataset.data['graph']['adjacency'])
                f.create_dataset('node_features', data=dataset.data['graph']['node_features'])

        # Save vision data
        if 'vision' in dataset.data:
            vision_dir = output_path / 'vision'
            vision_dir.mkdir(exist_ok=True)

            # Save as individual tensors for memory efficiency
            for i, tensor in enumerate(dataset.data['vision']):
                torch.save(tensor, vision_dir / f'vision_{i:06d}.pt')

            # Save metadata
            vision_metadata = {'num_samples': len(dataset.data['vision'])}
            with open(vision_dir / 'metadata.json', 'w') as f:
                json.dump(vision_metadata, f)

        # Save conjunction data
        if 'conjunction' in dataset.data:
            conj_data = dataset.data['conjunction']
            if isinstance(conj_data, dict):
                conj_df = pd.DataFrame(conj_data)
                conj_df.to_csv(output_path / 'conjunction_data.csv', index=False)

        # Save labels
        if 'labels' in dataset.data:
            np.save(output_path / 'labels.npy', dataset.data['labels'])

        # Save metadata
        if 'metadata' in dataset.data:
            with open(output_path / 'metadata.json', 'w') as f:
                json.dump(dataset.data['metadata'], f, indent=2)

        # Save dataset info
        dataset_info = {
            'num_samples': len(dataset),
            'modalities': list(dataset.data.keys()),
            'created_at': datetime.now().isoformat(),
            'config': self.dataset_config
        }

        with open(output_path / 'dataset_info.json', 'w') as f:
            json.dump(dataset_info, f, indent=2)

        self.logger.info(f"Saved dataset with {len(dataset)} samples to {output_path}")

    def load_dataset(self, input_dir: str) -> MultimodalDataset:
        """
        Load multimodal dataset from disk.

        Args:
            input_dir: Input directory

        Returns:
            Loaded MultimodalDataset
        """
        return self.build_from_files(input_dir)

    def create_data_loaders(self, dataset: MultimodalDataset,
                          batch_size: int = 32,
                          train_split: float = 0.7,
                          val_split: float = 0.15,
                          shuffle: bool = True) -> Dict[str, DataLoader]:
        """
        Create train/validation/test data loaders.

        Args:
            dataset: MultimodalDataset instance
            batch_size: Batch size
            train_split: Fraction for training
            val_split: Fraction for validation
            shuffle: Whether to shuffle data

        Returns:
            Dictionary of DataLoaders
        """
        dataset_size = len(dataset)
        indices = list(range(dataset_size))

        if shuffle:
            np.random.shuffle(indices)

        # Calculate split sizes
        train_size = int(train_split * dataset_size)
        val_size = int(val_split * dataset_size)
        test_size = dataset_size - train_size - val_size

        # Create subsets
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]

        # Create subset datasets
        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        val_dataset = torch.utils.data.Subset(dataset, val_indices)
        test_dataset = torch.utils.data.Subset(dataset, test_indices)

        # Create data loaders
        loaders = {
            'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
            'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
            'test': DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        }

        self.logger.info(f"Created data loaders: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")

        return loaders

    def get_dataset_statistics(self, dataset: MultimodalDataset) -> Dict[str, Any]:
        """
        Compute statistics for the dataset.

        Args:
            dataset: Dataset to analyze

        Returns:
            Dictionary of statistics
        """
        stats = {
            'num_samples': len(dataset),
            'modalities': list(dataset.data.keys()),
            'label_distribution': {}
        }

        # Label distribution
        if 'labels' in dataset.data:
            labels = dataset.data['labels']
            unique, counts = np.unique(labels, return_counts=True)
            for label, count in zip(unique, counts):
                stats['label_distribution'][str(label)] = int(count)

        # Modality-specific statistics
        if 'trajectory' in dataset.data:
            trajectories = dataset.data['trajectory']
            stats['trajectory'] = {
                'shape': trajectories.shape,
                'mean_distance': float(np.mean(np.linalg.norm(trajectories[:, -1, :3], axis=1))),
                'std_distance': float(np.std(np.linalg.norm(trajectories[:, -1, :3], axis=1)))
            }

        if 'graph' in dataset.data:
            adjacency = dataset.data['graph']['adjacency']
            stats['graph'] = {
                'num_nodes': adjacency.shape[-1],
                'avg_degree': float(np.mean(np.sum(adjacency, axis=-1))),
                'sparsity': float(np.mean(adjacency == 0))
            }

        if 'vision' in dataset.data:
            vision_data = dataset.data['vision']
            stats['vision'] = {
                'shape': vision_data.shape,
                'mean_intensity': float(torch.mean(vision_data)),
                'std_intensity': float(torch.std(vision_data))
            }

        return stats