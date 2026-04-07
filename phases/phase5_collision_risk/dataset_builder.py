"""
Collision Risk Dataset Builder

Builds datasets for collision risk assessment and prediction models.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import json
from datetime import datetime, timedelta

class CollisionRiskDataset(Dataset):
    """
    Dataset for collision risk assessment.
    """

    def __init__(self, conjunctions: List[Dict], trajectories: Dict[str, np.ndarray],
                 time_window: int = 100, prediction_horizon: int = 10):
        """
        Initialize collision risk dataset.

        Args:
            conjunctions: List of conjunction data
            trajectories: Dictionary of satellite trajectories
            time_window: Number of timesteps to use as input
            prediction_horizon: Number of timesteps to predict
        """
        self.conjunctions = conjunctions
        self.trajectories = trajectories
        self.time_window = time_window
        self.prediction_horizon = prediction_horizon

        # Process conjunctions
        self.processed_data = self._process_conjunctions()

    def _process_conjunctions(self) -> List[Dict]:
        """Process conjunction data for model input."""
        processed = []

        for conj in self.conjunctions:
            sat1_id = conj['satellite1_id']
            sat2_id = conj['satellite2_id']
            tca = conj['tca']  # Time of closest approach

            # Get trajectories around TCA
            traj1 = self._get_trajectory_window(sat1_id, tca, self.time_window)
            traj2 = self._get_trajectory_window(sat2_id, tca, self.time_window)

            if traj1 is not None and traj2 is not None:
                # Calculate relative trajectory
                relative_traj = traj1 - traj2

                # Get future trajectories for prediction
                future_traj1 = self._get_trajectory_window(
                    sat1_id, tca, self.prediction_horizon, offset=self.time_window
                )
                future_traj2 = self._get_trajectory_window(
                    sat2_id, tca, self.prediction_horizon, offset=self.time_window
                )

                if future_traj1 is not None and future_traj2 is not None:
                    future_relative = future_traj1 - future_traj2

                    # Calculate collision risk label
                    min_distance = np.min(np.linalg.norm(future_relative, axis=1))
                    collision_risk = 1.0 if min_distance < 1.0 else 0.0  # 1km threshold

                    processed.append({
                        'input_trajectory': relative_traj,
                        'future_trajectory': future_relative,
                        'collision_risk': collision_risk,
                        'min_distance': min_distance,
                        'conjunction_id': conj.get('id', f"{sat1_id}-{sat2_id}"),
                        'tca': tca
                    })

        return processed

    def _get_trajectory_window(self, sat_id: str, center_time: datetime,
                              window_size: int, offset: int = 0) -> Optional[np.ndarray]:
        """Get trajectory window around a specific time."""
        if sat_id not in self.trajectories:
            return None

        trajectory = self.trajectories[sat_id]
        # Assume trajectory has timestamps and positions

        # Find index closest to center_time
        # This is a simplified implementation
        # In practice, would need proper time indexing

        start_idx = max(0, len(trajectory) // 2 - window_size // 2 + offset)
        end_idx = min(len(trajectory), start_idx + window_size)

        if end_idx - start_idx < window_size:
            return None

        return trajectory[start_idx:end_idx]

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, idx):
        sample = self.processed_data[idx]

        return {
            'trajectory': torch.tensor(sample['input_trajectory'], dtype=torch.float32),
            'future_trajectory': torch.tensor(sample['future_trajectory'], dtype=torch.float32),
            'collision_risk': torch.tensor(sample['collision_risk'], dtype=torch.float32),
            'min_distance': torch.tensor(sample['min_distance'], dtype=torch.float32),
            'conjunction_id': sample['conjunction_id']
        }


class CollisionRiskDatasetBuilder:
    """
    Builder for collision risk datasets.
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize dataset builder.

        Args:
            config: Configuration dictionary
        """
        self.config = config or self._default_config()
        self.logger = logging.getLogger(__name__)

    def _default_config(self) -> Dict:
        """Default configuration."""
        return {
            'time_window': 100,
            'prediction_horizon': 10,
            'min_conjunctions': 1000,
            'max_conjunctions': 10000,
            'risk_threshold': 1.0,  # km
            'data_path': './data',
            'save_path': './datasets/collision_risk'
        }

    def build_dataset_from_trajectories(self, trajectories: Dict[str, np.ndarray],
                                       conjunctions: List[Dict]) -> CollisionRiskDataset:
        """
        Build dataset from trajectory and conjunction data.

        Args:
            trajectories: Dictionary of satellite trajectories
            conjunctions: List of conjunction events

        Returns:
            Collision risk dataset
        """
        self.logger.info(f"Building dataset from {len(conjunctions)} conjunctions")

        dataset = CollisionRiskDataset(
            conjunctions=conjunctions,
            trajectories=trajectories,
            time_window=self.config['time_window'],
            prediction_horizon=self.config['prediction_horizon']
        )

        self.logger.info(f"Created dataset with {len(dataset)} samples")
        return dataset

    def generate_synthetic_dataset(self, n_samples: int = 1000) -> CollisionRiskDataset:
        """
        Generate synthetic collision risk dataset.

        Args:
            n_samples: Number of samples to generate

        Returns:
            Synthetic collision risk dataset
        """
        self.logger.info(f"Generating synthetic dataset with {n_samples} samples")

        # Generate synthetic conjunctions and trajectories
        conjunctions = []
        trajectories = {}

        for i in range(n_samples):
            # Generate two satellite trajectories
            sat1_id = f"SAT_{2*i}"
            sat2_id = f"SAT_{2*i+1}"

            # Generate orbital trajectories
            traj1 = self._generate_orbital_trajectory()
            traj2 = self._generate_orbital_trajectory()

            trajectories[sat1_id] = traj1
            trajectories[sat2_id] = traj2

            # Generate conjunction
            tca = datetime.now() + timedelta(hours=np.random.uniform(-24, 24))

            conjunction = {
                'id': f"CONJ_{i}",
                'satellite1_id': sat1_id,
                'satellite2_id': sat2_id,
                'tca': tca,
                'miss_distance': np.random.exponential(10),  # km
                'relative_velocity': np.random.uniform(1, 15)  # km/s
            }
            conjunctions.append(conjunction)

        dataset = CollisionRiskDataset(
            conjunctions=conjunctions,
            trajectories=trajectories,
            time_window=self.config['time_window'],
            prediction_horizon=self.config['prediction_horizon']
        )

        return dataset

    def _generate_orbital_trajectory(self, n_points: int = 200) -> np.ndarray:
        """Generate synthetic orbital trajectory."""
        # Simplified orbital trajectory generation
        t = np.linspace(0, 4*np.pi, n_points)

        # Orbital parameters
        a = np.random.uniform(6671, 42164)  # Semi-major axis (LEO to GEO)
        e = np.random.uniform(0, 0.1)       # Eccentricity
        i = np.random.uniform(0, np.pi)     # Inclination

        # Position in orbital plane
        r = a * (1 - e * np.cos(t))
        x_orb = r * np.cos(t)
        y_orb = r * np.sin(t)
        z_orb = np.zeros_like(t)

        # Rotate by inclination
        x = x_orb
        y = y_orb * np.cos(i)
        z = y_orb * np.sin(i)

        # Velocity (simplified)
        vx = -a * e * np.sin(t)
        vy = a * (1 + e * np.cos(t))
        vz = np.zeros_like(t)

        # Combine position and velocity
        trajectory = np.column_stack([x, y, z, vx, vy, vz])

        # Add noise
        noise_level = 0.01
        trajectory += np.random.normal(0, noise_level, trajectory.shape)

        return trajectory

    def save_dataset(self, dataset: CollisionRiskDataset, path: str):
        """
        Save dataset to disk.

        Args:
            dataset: Dataset to save
            path: Save path
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        # Save processed data
        data_to_save = []
        for sample in dataset.processed_data:
            data_to_save.append({
                'input_trajectory': sample['input_trajectory'].tolist(),
                'future_trajectory': sample['future_trajectory'].tolist(),
                'collision_risk': sample['collision_risk'],
                'min_distance': sample['min_distance'],
                'conjunction_id': sample['conjunction_id'],
                'tca': sample['tca'].isoformat() if isinstance(sample['tca'], datetime) else sample['tca']
            })

        with open(path, 'w') as f:
            json.dump(data_to_save, f, indent=2)

        self.logger.info(f"Dataset saved to {path}")

    def load_dataset(self, path: str) -> CollisionRiskDataset:
        """
        Load dataset from disk.

        Args:
            path: Path to saved dataset

        Returns:
            Loaded dataset
        """
        with open(path, 'r') as f:
            data = json.load(f)

        # Convert back to numpy arrays
        processed_data = []
        for sample in data:
            processed_data.append({
                'input_trajectory': np.array(sample['input_trajectory']),
                'future_trajectory': np.array(sample['future_trajectory']),
                'collision_risk': sample['collision_risk'],
                'min_distance': sample['min_distance'],
                'conjunction_id': sample['conjunction_id'],
                'tca': sample['tca']
            })

        # Create dataset object
        dataset = CollisionRiskDataset.__new__(CollisionRiskDataset)
        dataset.processed_data = processed_data
        dataset.time_window = self.config['time_window']
        dataset.prediction_horizon = self.config['prediction_horizon']

        self.logger.info(f"Dataset loaded from {path}")
        return dataset

    def create_data_loaders(self, dataset: CollisionRiskDataset,
                           batch_size: int = 32, train_split: float = 0.8):
        """
        Create train/validation data loaders.

        Args:
            dataset: Dataset to split
            batch_size: Batch size
            train_split: Fraction for training

        Returns:
            Train and validation data loaders
        """
        # Split dataset
        n_train = int(len(dataset) * train_split)
        n_val = len(dataset) - n_train

        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(42)
        )

        # Create loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader


if __name__ == "__main__":
    # Example usage
    builder = CollisionRiskDatasetBuilder()

    # Generate synthetic dataset
    dataset = builder.generate_synthetic_dataset(n_samples=100)

    print(f"Generated dataset with {len(dataset)} samples")

    # Create data loaders
    train_loader, val_loader = builder.create_data_loaders(dataset)

    print(f"Train loader: {len(train_loader)} batches")
    print(f"Val loader: {len(val_loader)} batches")

    # Save dataset
    builder.save_dataset(dataset, './collision_risk_dataset.json')