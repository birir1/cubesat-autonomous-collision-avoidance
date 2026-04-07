"""
Fusion Data Loader for CubeSat Collision Avoidance

Time-aligned multimodal data loading with fusion capabilities.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from typing import Dict, List, Tuple, Optional, Any, Union, Iterator
from pathlib import Path
import logging
import yaml
from datetime import datetime, timedelta
import h5py

class TimeAlignedSampler(Sampler):
    """
    Sampler that ensures time-aligned batches across modalities.
    """

    def __init__(self, timestamps: np.ndarray, batch_size: int = 32,
                 time_window: float = 1.0):
        """
        Initialize time-aligned sampler.

        Args:
            timestamps: Array of timestamps for each sample
            batch_size: Batch size
            time_window: Maximum time difference for alignment (hours)
        """
        self.timestamps = timestamps
        self.batch_size = batch_size
        self.time_window = time_window

        # Sort indices by timestamp
        self.sorted_indices = np.argsort(timestamps)

    def __iter__(self) -> Iterator[List[int]]:
        """
        Iterate over time-aligned batches.

        Yields:
            Lists of sample indices for each batch
        """
        n_samples = len(self.sorted_indices)

        # Group samples by time windows
        current_batch = []
        current_time = None

        for idx in self.sorted_indices:
            sample_time = self.timestamps[idx]

            if current_time is None:
                current_time = sample_time
                current_batch = [idx]
            elif abs(sample_time - current_time) <= self.time_window:
                current_batch.append(idx)
            else:
                # Yield current batch if it meets minimum size
                if len(current_batch) >= max(1, self.batch_size // 4):
                    # Pad or truncate to batch_size
                    if len(current_batch) < self.batch_size:
                        # Duplicate samples to fill batch
                        while len(current_batch) < self.batch_size:
                            current_batch.extend(current_batch[:self.batch_size - len(current_batch)])
                    elif len(current_batch) > self.batch_size:
                        current_batch = current_batch[:self.batch_size]

                    yield current_batch

                # Start new batch
                current_batch = [idx]
                current_time = sample_time

        # Yield final batch
        if current_batch and len(current_batch) >= max(1, self.batch_size // 4):
            while len(current_batch) < self.batch_size:
                current_batch.extend(current_batch[:self.batch_size - len(current_batch)])
            current_batch = current_batch[:self.batch_size]
            yield current_batch

    def __len__(self) -> int:
        """Return number of batches."""
        # Estimate based on time windows
        time_span = self.timestamps.max() - self.timestamps.min()
        estimated_batches = max(1, int(time_span / self.time_window))
        return estimated_batches

class FusionDataset(Dataset):
    """
    Dataset for time-aligned multimodal fusion.
    """

    def __init__(self, data_dict: Dict[str, Any],
                 timestamps: np.ndarray,
                 sequence_length: int = 10,
                 prediction_horizon: int = 1):
        """
        Initialize fusion dataset.

        Args:
            data_dict: Dictionary containing all modality data
            timestamps: Timestamps for each sample
            sequence_length: Number of time steps in input sequence
            prediction_horizon: Steps ahead to predict
        """
        self.data = data_dict
        self.timestamps = timestamps
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon

        # Validate data
        self._validate_data()

        # Create sequence indices
        self.valid_sequences = self._find_valid_sequences()

    def _validate_data(self) -> None:
        """Validate data consistency."""
        required_keys = ['trajectory', 'graph', 'vision', 'conjunction', 'labels']
        for key in required_keys:
            if key not in self.data:
                raise ValueError(f"Missing required data: {key}")

        # Check sequence lengths
        n_samples = len(self.data['labels'])
        for key, value in self.data.items():
            if isinstance(value, (np.ndarray, list)):
                if len(value) != n_samples:
                    raise ValueError(f"Data length mismatch for {key}: {len(value)} vs {n_samples}")

    def _find_valid_sequences(self) -> List[Tuple[int, int]]:
        """
        Find valid sequence start and end indices.

        Returns:
            List of (start_idx, end_idx) tuples for valid sequences
        """
        valid_sequences = []
        n_samples = len(self.data['labels'])

        for start_idx in range(n_samples - self.sequence_length - self.prediction_horizon + 1):
            end_idx = start_idx + self.sequence_length
            target_idx = end_idx + self.prediction_horizon - 1

            # Check if all data in sequence is valid
            if self._is_sequence_valid(start_idx, end_idx):
                valid_sequences.append((start_idx, target_idx))

        return valid_sequences

    def _is_sequence_valid(self, start_idx: int, end_idx: int) -> bool:
        """
        Check if a sequence contains valid data.

        Args:
            start_idx: Start index of sequence
            end_idx: End index of sequence

        Returns:
            True if sequence is valid
        """
        # Check for NaN values in trajectory data
        trajectory_seq = self.data['trajectory'][start_idx:end_idx]
        if np.any(np.isnan(trajectory_seq)):
            return False

        # Check timestamp continuity (basic check)
        times = self.timestamps[start_idx:end_idx]
        time_diffs = np.diff(times)
        if np.any(time_diffs <= 0):  # Non-monotonic or duplicate times
            return False

        return True

    def __len__(self) -> int:
        return len(self.valid_sequences)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a time-aligned multimodal sequence.

        Args:
            idx: Sequence index

        Returns:
            Dictionary containing sequence data
        """
        start_idx, target_idx = self.valid_sequences[idx]

        # Extract input sequence
        sequence_data = {}

        # Trajectory sequence
        sequence_data['trajectory'] = self.data['trajectory'][start_idx:start_idx + self.sequence_length]

        # Graph sequence (use last graph in sequence)
        sequence_data['graph'] = {
            'adjacency': self.data['graph']['adjacency'][start_idx + self.sequence_length - 1],
            'node_features': self.data['graph']['node_features'][start_idx + self.sequence_length - 1]
        }

        # Vision sequence (use last image in sequence)
        sequence_data['vision'] = self.data['vision'][start_idx + self.sequence_length - 1]

        # Conjunction sequence
        conj_data = {}
        for key, value in self.data['conjunction'].items():
            conj_data[key] = value[start_idx:start_idx + self.sequence_length]
        sequence_data['conjunction'] = conj_data

        # Target labels (prediction horizon)
        sequence_data['target'] = self.data['labels'][target_idx]

        # Metadata
        sequence_data['timestamps'] = self.timestamps[start_idx:start_idx + self.sequence_length]
        sequence_data['target_timestamp'] = self.timestamps[target_idx]

        return sequence_data

class FusionDataLoader:
    """
    Data loader for time-aligned multimodal fusion.
    """

    def __init__(self, config_path: str = 'configs/data_config.yaml'):
        """
        Initialize fusion data loader.

        Args:
            config_path: Path to configuration file
        """
        self.logger = logging.getLogger(__name__)

        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.fusion_config = self.config['fusion']

    def create_fusion_dataset(self, data_dict: Dict[str, Any],
                            timestamps: np.ndarray) -> FusionDataset:
        """
        Create a fusion dataset from data dictionary.

        Args:
            data_dict: Dictionary containing all modality data
            timestamps: Timestamps for each sample

        Returns:
            FusionDataset instance
        """
        sequence_length = self.fusion_config.get('sequence_length', 10)
        prediction_horizon = self.fusion_config.get('prediction_horizon', 1)

        return FusionDataset(
            data_dict=data_dict,
            timestamps=timestamps,
            sequence_length=sequence_length,
            prediction_horizon=prediction_horizon
        )

    def create_time_aligned_loaders(self, dataset: FusionDataset,
                                  batch_size: int = 32,
                                  time_window: float = 1.0) -> Dict[str, DataLoader]:
        """
        Create time-aligned data loaders.

        Args:
            dataset: FusionDataset instance
            batch_size: Batch size
            time_window: Time alignment window (hours)

        Returns:
            Dictionary of DataLoaders
        """
        # Create time-aligned sampler
        sampler = TimeAlignedSampler(
            timestamps=dataset.timestamps,
            batch_size=batch_size,
            time_window=time_window
        )

        # Create data loader
        loader = DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=self.fusion_config.get('num_workers', 0),
            pin_memory=self.fusion_config.get('pin_memory', True)
        )

        return {'fusion': loader}

    def create_temporal_loaders(self, dataset: FusionDataset,
                              batch_size: int = 32,
                              train_split: float = 0.7,
                              val_split: float = 0.15) -> Dict[str, DataLoader]:
        """
        Create temporal train/val/test split loaders.

        Args:
            dataset: FusionDataset instance
            batch_size: Batch size
            train_split: Fraction for training
            val_split: Fraction for validation

        Returns:
            Dictionary of DataLoaders
        """
        n_samples = len(dataset)
        indices = list(range(n_samples))

        # Temporal split (respecting time order)
        train_size = int(train_split * n_samples)
        val_size = int(val_split * n_samples)

        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]

        # Create subset datasets
        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        val_dataset = torch.utils.data.Subset(dataset, val_indices)
        test_dataset = torch.utils.data.Subset(dataset, test_indices)

        # Create data loaders
        loaders = {
            'train': DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=self.fusion_config.get('num_workers', 0),
                pin_memory=self.fusion_config.get('pin_memory', True)
            ),
            'val': DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=self.fusion_config.get('num_workers', 0),
                pin_memory=self.fusion_config.get('pin_memory', True)
            ),
            'test': DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=self.fusion_config.get('num_workers', 0),
                pin_memory=self.fusion_config.get('pin_memory', True)
            )
        }

        self.logger.info(f"Created temporal loaders: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")

        return loaders

    def create_rolling_window_loaders(self, dataset: FusionDataset,
                                    window_size: int = 100,
                                    step_size: int = 10,
                                    batch_size: int = 32) -> DataLoader:
        """
        Create rolling window data loader for online learning.

        Args:
            dataset: FusionDataset instance
            window_size: Size of rolling window
            step_size: Step size for window movement
            batch_size: Batch size

        Returns:
            DataLoader with rolling windows
        """
        class RollingWindowSampler(Sampler):
            def __init__(self, dataset_size, window_size, step_size):
                self.dataset_size = dataset_size
                self.window_size = window_size
                self.step_size = step_size

            def __iter__(self):
                for start_idx in range(0, self.dataset_size - self.window_size + 1, self.step_size):
                    end_idx = min(start_idx + self.window_size, self.dataset_size)
                    window_indices = list(range(start_idx, end_idx))
                    yield window_indices

            def __len__(self):
                return (self.dataset_size - self.window_size) // self.step_size + 1

        sampler = RollingWindowSampler(len(dataset), window_size, step_size)

        loader = DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=self.fusion_config.get('num_workers', 0),
            pin_memory=self.fusion_config.get('pin_memory', True)
        )

        return loader

    def fuse_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fuse multimodal batch data.

        Args:
            batch: Batch dictionary from DataLoader

        Returns:
            Fused batch data
        """
        fused_batch = {}

        # Stack trajectory sequences
        if 'trajectory' in batch:
            fused_batch['trajectory'] = torch.stack(batch['trajectory'])

        # Handle graph data
        if 'graph' in batch:
            fused_batch['graph'] = {
                'adjacency': torch.stack([item['adjacency'] for item in batch['graph']]),
                'node_features': torch.stack([item['node_features'] for item in batch['graph']])
            }

        # Stack vision tensors
        if 'vision' in batch:
            fused_batch['vision'] = torch.stack(batch['vision'])

        # Handle conjunction data
        if 'conjunction' in batch:
            conj_fused = {}
            first_item = batch['conjunction'][0]
            for key in first_item.keys():
                conj_fused[key] = torch.stack([item[key] for item in batch['conjunction']])
            fused_batch['conjunction'] = conj_fused

        # Stack targets
        if 'target' in batch:
            fused_batch['target'] = torch.stack(batch['target']) if isinstance(batch['target'][0], torch.Tensor) else torch.tensor(batch['target'])

        # Handle metadata
        if 'timestamps' in batch:
            fused_batch['timestamps'] = torch.stack(batch['timestamps'])

        if 'target_timestamp' in batch:
            fused_batch['target_timestamp'] = batch['target_timestamp']

        return fused_batch

    def save_fusion_data(self, dataset: FusionDataset, output_dir: str) -> None:
        """
        Save fusion dataset to disk.

        Args:
            dataset: FusionDataset to save
            output_dir: Output directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save data in HDF5 format for efficiency
        with h5py.File(output_path / 'fusion_data.h5', 'w') as f:
            # Save trajectory data
            f.create_dataset('trajectory', data=dataset.data['trajectory'])

            # Save graph data
            graph_group = f.create_group('graph')
            graph_group.create_dataset('adjacency', data=dataset.data['graph']['adjacency'])
            graph_group.create_dataset('node_features', data=dataset.data['graph']['node_features'])

            # Save vision data (as individual datasets for large tensors)
            vision_group = f.create_group('vision')
            for i, tensor in enumerate(dataset.data['vision']):
                vision_group.create_dataset(f'tensor_{i}', data=tensor.numpy())

            # Save conjunction data
            conj_group = f.create_group('conjunction')
            for key, value in dataset.data['conjunction'].items():
                conj_group.create_dataset(key, data=value)

            # Save labels and timestamps
            f.create_dataset('labels', data=dataset.data['labels'])
            f.create_dataset('timestamps', data=dataset.timestamps)

        # Save metadata
        metadata = {
            'sequence_length': dataset.sequence_length,
            'prediction_horizon': dataset.prediction_horizon,
            'num_sequences': len(dataset.valid_sequences),
            'config': self.fusion_config,
            'created_at': datetime.now().isoformat()
        }

        with open(output_path / 'fusion_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        self.logger.info(f"Saved fusion dataset with {len(dataset)} sequences to {output_path}")

    def load_fusion_data(self, input_dir: str) -> Tuple[FusionDataset, np.ndarray]:
        """
        Load fusion dataset from disk.

        Args:
            input_dir: Input directory

        Returns:
            Tuple of (FusionDataset, timestamps)
        """
        input_path = Path(input_dir)

        # Load HDF5 data
        with h5py.File(input_path / 'fusion_data.h5', 'r') as f:
            data_dict = {}

            # Load trajectory
            data_dict['trajectory'] = np.array(f['trajectory'])

            # Load graph
            data_dict['graph'] = {
                'adjacency': np.array(f['graph']['adjacency']),
                'node_features': np.array(f['graph']['node_features'])
            }

            # Load vision
            vision_tensors = []
            vision_group = f['vision']
            for key in sorted(vision_group.keys()):
                vision_tensors.append(torch.tensor(np.array(vision_group[key])))
            data_dict['vision'] = torch.stack(vision_tensors)

            # Load conjunction
            conj_dict = {}
            conj_group = f['conjunction']
            for key in conj_group.keys():
                conj_dict[key] = np.array(conj_group[key])
            data_dict['conjunction'] = conj_dict

            # Load labels and timestamps
            data_dict['labels'] = np.array(f['labels'])
            timestamps = np.array(f['timestamps'])

        # Load metadata
        with open(input_path / 'fusion_metadata.json', 'r') as f:
            metadata = json.load(f)

        # Create dataset
        dataset = FusionDataset(
            data_dict=data_dict,
            timestamps=timestamps,
            sequence_length=metadata['sequence_length'],
            prediction_horizon=metadata['prediction_horizon']
        )

        return dataset, timestamps