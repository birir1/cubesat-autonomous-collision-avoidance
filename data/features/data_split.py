"""
Data Split Module for CubeSat Collision Avoidance

Handles splitting data into real vs synthetic, train/val/test sets.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import logging
import yaml
import json
from datetime import datetime
from collections import Counter

class DataSplitter:
    """
    Advanced data splitting for multimodal collision avoidance datasets.
    """

    def __init__(self, config_path: str = 'configs/data_config.yaml'):
        """
        Initialize data splitter.

        Args:
            config_path: Path to configuration file
        """
        self.logger = logging.getLogger(__name__)

        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.split_config = self.config['data_split']

    def split_real_synthetic(self, data_dict: Dict[str, Any],
                           data_types: np.ndarray) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Split data into real and synthetic subsets.

        Args:
            data_dict: Dictionary containing all modality data
            data_types: Array indicating 'real' or 'synthetic' for each sample

        Returns:
            Tuple of (real_data, synthetic_data) dictionaries
        """
        # Find real and synthetic indices
        real_indices = np.where(data_types == 'real')[0]
        synthetic_indices = np.where(data_types == 'synthetic')[0]

        self.logger.info(f"Found {len(real_indices)} real samples and {len(synthetic_indices)} synthetic samples")

        # Split each modality
        real_data = self._extract_subset(data_dict, real_indices)
        synthetic_data = self._extract_subset(data_dict, synthetic_indices)

        return real_data, synthetic_data

    def _extract_subset(self, data_dict: Dict[str, Any], indices: np.ndarray) -> Dict[str, Any]:
        """
        Extract subset of data using given indices.

        Args:
            data_dict: Full data dictionary
            indices: Indices to extract

        Returns:
            Subset data dictionary
        """
        subset = {}

        for key, value in data_dict.items():
            if isinstance(value, (np.ndarray, torch.Tensor)):
                subset[key] = value[indices]
            elif isinstance(value, dict):
                # Handle nested dictionaries (like graph data)
                subset[key] = {}
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, (np.ndarray, torch.Tensor)):
                        subset[key][sub_key] = sub_value[indices]
            elif isinstance(value, list):
                subset[key] = [value[i] for i in indices]
            else:
                # Copy other data types as-is
                subset[key] = value

        return subset

    def stratified_temporal_split(self, data_dict: Dict[str, Any],
                                labels: np.ndarray,
                                timestamps: Optional[np.ndarray] = None,
                                train_ratio: float = 0.7,
                                val_ratio: float = 0.15,
                                test_ratio: float = 0.15) -> Tuple[Dict[str, Any], ...]:
        """
        Perform stratified temporal split maintaining class distribution and time order.

        Args:
            data_dict: Dictionary containing all modality data
            labels: Target labels
            timestamps: Timestamps for temporal ordering
            train_ratio: Fraction for training
            val_ratio: Fraction for validation
            test_ratio: Fraction for testing

        Returns:
            Tuple of (train_data, val_data, test_data) dictionaries
        """
        n_samples = len(labels)

        # Sort by time if timestamps provided
        if timestamps is not None:
            sort_indices = np.argsort(timestamps)
            labels = labels[sort_indices]
            data_dict = self._extract_subset(data_dict, sort_indices)

        # Calculate split points
        train_end = int(train_ratio * n_samples)
        val_end = train_end + int(val_ratio * n_samples)

        # Extract splits
        train_data = self._extract_subset(data_dict, np.arange(0, train_end))
        val_data = self._extract_subset(data_dict, np.arange(train_end, val_end))
        test_data = self._extract_subset(data_dict, np.arange(val_end, n_samples))

        # Add labels to each split
        train_data['labels'] = labels[:train_end]
        val_data['labels'] = labels[train_end:val_end]
        test_data['labels'] = labels[val_end:]

        self.logger.info(f"Temporal split: train={train_end}, val={val_end - train_end}, test={n_samples - val_end}")

        return train_data, val_data, test_data

    def risk_stratified_split(self, data_dict: Dict[str, Any],
                            labels: np.ndarray,
                            risk_levels: Optional[np.ndarray] = None,
                            train_ratio: float = 0.7,
                            val_ratio: float = 0.15) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """
        Split data stratified by risk levels to ensure balanced representation.

        Args:
            data_dict: Dictionary containing all modality data
            labels: Target labels
            risk_levels: Risk level categories (if different from labels)
            train_ratio: Fraction for training
            val_ratio: Fraction for validation

        Returns:
            Tuple of (train_data, val_data, test_data) dictionaries
        """
        if risk_levels is None:
            risk_levels = labels

        # Use stratified split to maintain risk distribution
        train_indices, temp_indices = train_test_split(
            np.arange(len(labels)),
            test_size=(1 - train_ratio),
            stratify=risk_levels,
            random_state=42
        )

        val_indices, test_indices = train_test_split(
            temp_indices,
            test_size=(1 - val_ratio / (1 - train_ratio)),
            stratify=risk_levels[temp_indices],
            random_state=42
        )

        # Extract splits
        train_data = self._extract_subset(data_dict, train_indices)
        val_data = self._extract_subset(data_dict, val_indices)
        test_data = self._extract_subset(data_dict, test_indices)

        # Add labels
        train_data['labels'] = labels[train_indices]
        val_data['labels'] = labels[val_indices]
        test_data['labels'] = labels[test_indices]

        self.logger.info(f"Risk-stratified split: train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}")

        return train_data, val_data, test_data

    def cross_validation_splits(self, data_dict: Dict[str, Any],
                              labels: np.ndarray,
                              n_folds: int = 5,
                              stratified: bool = True) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
        """
        Create cross-validation splits.

        Args:
            data_dict: Dictionary containing all modality data
            labels: Target labels
            n_folds: Number of CV folds
            stratified: Whether to use stratified CV

        Returns:
            List of (train_data, val_data) tuples for each fold
        """
        if stratified:
            kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        else:
            kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

        cv_splits = []

        for train_indices, val_indices in kf.split(np.arange(len(labels)), labels):
            train_data = self._extract_subset(data_dict, train_indices)
            val_data = self._extract_subset(data_dict, val_indices)

            train_data['labels'] = labels[train_indices]
            val_data['labels'] = labels[val_indices]

            cv_splits.append((train_data, val_data))

        self.logger.info(f"Created {n_folds} cross-validation splits")

        return cv_splits

    def balance_classes(self, data_dict: Dict[str, Any],
                       labels: np.ndarray,
                       target_samples: Optional[int] = None,
                       method: str = 'undersample') -> Tuple[Dict[str, Any], np.ndarray]:
        """
        Balance class distribution in the dataset.

        Args:
            data_dict: Dictionary containing all modality data
            labels: Target labels
            target_samples: Target number of samples per class
            method: Balancing method ('undersample', 'oversample', 'hybrid')

        Returns:
            Tuple of (balanced_data, balanced_labels)
        """
        unique_labels, counts = np.unique(labels, return_counts=True)
        label_counts = dict(zip(unique_labels, counts))

        self.logger.info(f"Original class distribution: {label_counts}")

        if target_samples is None:
            # Balance to minimum class size
            target_samples = min(counts)

        balanced_indices = []

        for label in unique_labels:
            label_indices = np.where(labels == label)[0]
            current_count = len(label_indices)

            if method == 'undersample':
                # Random undersampling
                selected_indices = np.random.choice(
                    label_indices,
                    size=min(current_count, target_samples),
                    replace=False
                )
            elif method == 'oversample':
                # Random oversampling with replacement
                selected_indices = np.random.choice(
                    label_indices,
                    size=target_samples,
                    replace=True
                )
            elif method == 'hybrid':
                # Hybrid approach
                if current_count < target_samples:
                    # Oversample minority classes
                    selected_indices = np.random.choice(
                        label_indices,
                        size=target_samples,
                        replace=True
                    )
                else:
                    # Undersample majority classes
                    selected_indices = np.random.choice(
                        label_indices,
                        size=target_samples,
                        replace=False
                    )
            else:
                raise ValueError(f"Unknown balancing method: {method}")

            balanced_indices.extend(selected_indices)

        # Shuffle balanced indices
        balanced_indices = np.array(balanced_indices)
        np.random.shuffle(balanced_indices)

        # Extract balanced data
        balanced_data = self._extract_subset(data_dict, balanced_indices)
        balanced_labels = labels[balanced_indices]

        # Check final distribution
        final_counts = Counter(balanced_labels)
        self.logger.info(f"Balanced class distribution: {dict(final_counts)}")

        return balanced_data, balanced_labels

    def create_domain_splits(self, data_dict: Dict[str, Any],
                           domain_labels: np.ndarray,
                           test_domains: List[Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Create domain generalization splits (train on some domains, test on others).

        Args:
            data_dict: Dictionary containing all modality data
            domain_labels: Domain labels for each sample
            test_domains: List of domains to use for testing

        Returns:
            Tuple of (train_data, test_data) dictionaries
        """
        test_domains = set(test_domains)

        # Find train and test indices
        train_indices = []
        test_indices = []

        for i, domain in enumerate(domain_labels):
            if domain in test_domains:
                test_indices.append(i)
            else:
                train_indices.append(i)

        train_indices = np.array(train_indices)
        test_indices = np.array(test_indices)

        # Extract splits
        train_data = self._extract_subset(data_dict, train_indices)
        test_data = self._extract_subset(data_dict, test_indices)

        # Add domain labels
        train_data['domain_labels'] = domain_labels[train_indices]
        test_data['domain_labels'] = domain_labels[test_indices]

        self.logger.info(f"Domain split: train_domains={set(domain_labels[train_indices])}, test_domains={set(domain_labels[test_indices])}")

        return train_data, test_data

    def save_split_indices(self, split_indices: Dict[str, np.ndarray],
                          output_path: str) -> None:
        """
        Save data split indices for reproducibility.

        Args:
            split_indices: Dictionary with split names and indices
            output_path: Output file path
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Convert to serializable format
        serializable_splits = {}
        for split_name, indices in split_indices.items():
            serializable_splits[split_name] = indices.tolist()

        split_info = {
            'splits': serializable_splits,
            'created_at': datetime.now().isoformat(),
            'config': self.split_config
        }

        with open(output_path, 'w') as f:
            json.dump(split_info, f, indent=2)

        self.logger.info(f"Saved split indices to {output_path}")

    def load_split_indices(self, input_path: str) -> Dict[str, np.ndarray]:
        """
        Load data split indices.

        Args:
            input_path: Input file path

        Returns:
            Dictionary with split names and indices
        """
        with open(input_path, 'r') as f:
            split_info = json.load(f)

        split_indices = {}
        for split_name, indices in split_info['splits'].items():
            split_indices[split_name] = np.array(indices)

        self.logger.info(f"Loaded split indices from {input_path}")

        return split_indices

    def get_split_statistics(self, data_dict: Dict[str, Any],
                           labels: np.ndarray,
                           split_indices: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Compute statistics for each data split.

        Args:
            data_dict: Full data dictionary
            labels: Target labels
            split_indices: Dictionary with split indices

        Returns:
            Dictionary with split statistics
        """
        stats = {}

        for split_name, indices in split_indices.items():
            split_labels = labels[indices]
            unique_labels, counts = np.unique(split_labels, return_counts=True)

            stats[split_name] = {
                'num_samples': len(indices),
                'class_distribution': dict(zip(unique_labels.tolist(), counts.tolist())),
                'class_ratios': (counts / len(indices)).tolist()
            }

        return stats

    def validate_splits(self, data_dict: Dict[str, Any],
                       split_indices: Dict[str, np.ndarray]) -> bool:
        """
        Validate that data splits are properly formed.

        Args:
            data_dict: Full data dictionary
            split_indices: Dictionary with split indices

        Returns:
            True if splits are valid
        """
        all_indices = []
        n_samples = len(data_dict['labels'])

        for split_name, indices in split_indices.items():
            # Check for valid indices
            if np.any(indices < 0) or np.any(indices >= n_samples):
                self.logger.error(f"Invalid indices in split {split_name}")
                return False

            # Check for duplicates across splits
            if set(indices) & set(all_indices):
                self.logger.error(f"Overlapping indices between splits")
                return False

            all_indices.extend(indices)

        # Check that all samples are covered
        if set(all_indices) != set(range(n_samples)):
            self.logger.error("Not all samples are covered by splits")
            return False

        self.logger.info("Data splits validation passed")
        return True

    def create_incremental_splits(self, data_dict: Dict[str, Any],
                                timestamps: np.ndarray,
                                n_increments: int = 5) -> List[Dict[str, Any]]:
        """
        Create incremental learning splits based on time.

        Args:
            data_dict: Dictionary containing all modality data
            timestamps: Timestamps for temporal ordering
            n_increments: Number of incremental steps

        Returns:
            List of data dictionaries for each increment
        """
        # Sort by time
        sort_indices = np.argsort(timestamps)
        sorted_data = self._extract_subset(data_dict, sort_indices)
        sorted_timestamps = timestamps[sort_indices]

        n_samples = len(sorted_timestamps)
        increment_size = n_samples // n_increments

        increments = []

        for i in range(1, n_increments + 1):
            end_idx = min(i * increment_size, n_samples)
            increment_indices = np.arange(0, end_idx)

            increment_data = self._extract_subset(sorted_data, increment_indices)
            increment_data['increment'] = i
            increment_data['timestamps'] = sorted_timestamps[:end_idx]

            increments.append(increment_data)

        self.logger.info(f"Created {n_increments} incremental splits")

        return increments