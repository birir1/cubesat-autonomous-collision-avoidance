"""
Data Validation Module for CubeSat Collision Avoidance

Ensures data quality, consistency, and scientific validity across all modalities.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import yaml

class DataValidator:
    """
    Comprehensive data validation for multimodal collision avoidance datasets.
    """

    def __init__(self, config_path: str = 'configs/data_config.yaml'):
        """
        Initialize validator with configuration.

        Args:
            config_path: Path to data configuration file
        """
        self.logger = logging.getLogger(__name__)

        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Validation thresholds
        self.max_distance_km = self.config['validation']['max_distance_km']
        self.min_distance_km = self.config['validation']['min_distance_km']
        self.max_velocity_kms = self.config['validation']['max_velocity_kms']

    def validate_trajectory_data(self, trajectories: np.ndarray,
                               timestamps: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Validate trajectory time-series data.

        Args:
            trajectories: Shape (batch, seq_len, 6) - [pos_x, pos_y, pos_z, vel_x, vel_y, vel_z]
            timestamps: Optional timestamps for each trajectory point

        Returns:
            Validation results dictionary
        """
        results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'stats': {}
        }

        # Check shape
        if trajectories.ndim != 3 or trajectories.shape[-1] != 6:
            results['errors'].append(f"Invalid trajectory shape: {trajectories.shape}, expected (batch, seq_len, 6)")
            results['valid'] = False
            return results

        # Check for NaNs
        if self.config['validation']['check_nans']:
            nan_mask = np.isnan(trajectories)
            if nan_mask.any():
                results['errors'].append(f"Found {nan_mask.sum()} NaN values in trajectories")
                results['valid'] = False

        # Check ranges
        if self.config['validation']['check_ranges']:
            positions = trajectories[:, :, :3]  # km
            velocities = trajectories[:, :, 3:]  # km/s

            # Position ranges (LEO orbits)
            pos_magnitude = np.linalg.norm(positions, axis=-1)
            if pos_magnitude.max() > self.max_distance_km or pos_magnitude.min() < self.min_distance_km:
                results['errors'].append(f"Position magnitudes out of range: min={pos_magnitude.min():.1f}, max={pos_magnitude.max():.1f} km")
                results['valid'] = False

            # Velocity ranges
            vel_magnitude = np.linalg.norm(velocities, axis=-1)
            if vel_magnitude.max() > self.max_velocity_kms:
                results['errors'].append(f"Velocity magnitudes too high: max={vel_magnitude.max():.3f} km/s")
                results['valid'] = False

        # Compute statistics
        results['stats'] = {
            'num_trajectories': trajectories.shape[0],
            'seq_length': trajectories.shape[1],
            'position_range_km': [float(positions.min()), float(positions.max())],
            'velocity_range_kms': [float(velocities.min()), float(velocities.max())],
            'mean_distance_km': float(pos_magnitude.mean()),
            'mean_velocity_kms': float(vel_magnitude.mean())
        }

        return results

    def validate_conjunction_data(self, conjunctions: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate conjunction event data.

        Args:
            conjunctions: DataFrame with conjunction events

        Returns:
            Validation results
        """
        results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'stats': {}
        }

        required_columns = ['satellite1_id', 'satellite2_id', 'tca', 'miss_distance_km',
                          'relative_velocity_kms', 'collision_probability']

        # Check required columns
        missing_cols = [col for col in required_columns if col not in conjunctions.columns]
        if missing_cols:
            results['errors'].append(f"Missing required columns: {missing_cols}")
            results['valid'] = False
            return results

        # Check for NaNs
        if self.config['validation']['check_nans']:
            nan_counts = conjunctions.isnull().sum()
            if nan_counts.any():
                results['errors'].append(f"Found NaN values: {nan_counts[nan_counts > 0].to_dict()}")
                results['valid'] = False

        # Check ranges
        if self.config['validation']['check_ranges']:
            # Miss distance
            miss_dist = conjunctions['miss_distance_km']
            if (miss_dist < 0).any():
                results['errors'].append("Negative miss distances found")
                results['valid'] = False

            # Collision probability
            prob = conjunctions['collision_probability']
            if (prob < 0).any() or (prob > 1).any():
                results['errors'].append("Collision probabilities out of [0,1] range")
                results['valid'] = False

        # Compute statistics
        results['stats'] = {
            'num_conjunctions': len(conjunctions),
            'miss_distance_stats': {
                'min': float(conjunctions['miss_distance_km'].min()),
                'max': float(conjunctions['miss_distance_km'].max()),
                'mean': float(conjunctions['miss_distance_km'].mean()),
                'median': float(conjunctions['miss_distance_km'].median())
            },
            'collision_prob_stats': {
                'min': float(conjunctions['collision_probability'].min()),
                'max': float(conjunctions['collision_probability'].max()),
                'mean': float(conjunctions['collision_probability'].mean())
            },
            'high_risk_count': int((conjunctions['miss_distance_km'] < self.config['risk']['collision_threshold_m'] / 1000).sum())
        }

        return results

    def validate_graph_data(self, adjacency_matrix: np.ndarray,
                          node_features: np.ndarray) -> Dict[str, Any]:
        """
        Validate graph structure data.

        Args:
            adjacency_matrix: Shape (batch, num_nodes, num_nodes)
            node_features: Shape (batch, num_nodes, feature_dim)

        Returns:
            Validation results
        """
        results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'stats': {}
        }

        # Check shapes
        if adjacency_matrix.shape[:-1] != node_features.shape[:-1]:
            results['errors'].append(f"Shape mismatch: adjacency {adjacency_matrix.shape}, features {node_features.shape}")
            results['valid'] = False
            return results

        # Check adjacency matrix properties
        if not np.allclose(adjacency_matrix, adjacency_matrix.transpose((0, 2, 1))):
            results['warnings'].append("Adjacency matrix is not symmetric")

        # Check for self-loops (should be zero)
        diagonal = np.diagonal(adjacency_matrix, axis1=1, axis2=2)
        if np.any(diagonal != 0):
            results['warnings'].append("Self-loops detected in adjacency matrix")

        # Check for isolated nodes
        degree = adjacency_matrix.sum(axis=-1)
        isolated = (degree == 0).sum(axis=-1)
        if isolated.max() > 0:
            results['stats']['isolated_nodes_max'] = int(isolated.max())

        # Compute statistics
        results['stats'].update({
            'num_graphs': adjacency_matrix.shape[0],
            'num_nodes': adjacency_matrix.shape[1],
            'node_feature_dim': node_features.shape[2],
            'avg_degree': float(degree.mean()),
            'sparsity': float((adjacency_matrix == 0).mean())
        })

        return results

    def validate_image_data(self, images: np.ndarray,
                          labels: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Validate image data for vision modality.

        Args:
            images: Shape (batch, channels, height, width)
            labels: Optional detection labels

        Returns:
            Validation results
        """
        results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'stats': {}
        }

        # Check shape (assuming RGB images)
        if images.ndim != 4 or images.shape[1] not in [1, 3]:
            results['errors'].append(f"Invalid image shape: {images.shape}, expected (batch, channels, height, width)")
            results['valid'] = False

        # Check value range (assuming normalized 0-1 or 0-255)
        if images.min() < -0.1 or images.max() > 255.1:
            results['warnings'].append(f"Image values out of expected range: min={images.min():.3f}, max={images.max():.3f}")

        # Check for NaNs
        if np.isnan(images).any():
            results['errors'].append("NaN values found in images")
            results['valid'] = False

        # Validate labels if provided
        if labels is not None:
            if len(labels) != len(images):
                results['errors'].append(f"Label/image count mismatch: {len(labels)} vs {len(images)}")
                results['valid'] = False

        # Compute statistics
        results['stats'] = {
            'num_images': images.shape[0],
            'channels': images.shape[1],
            'height': images.shape[2],
            'width': images.shape[3],
            'value_range': [float(images.min()), float(images.max())],
            'mean_intensity': float(images.mean()),
            'std_intensity': float(images.std())
        }

        return results

    def validate_multimodal_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate a complete multimodal sample.

        Args:
            sample: Dictionary containing all modalities

        Returns:
            Validation results
        """
        results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'modality_results': {}
        }

        # Validate each modality
        if 'trajectory' in sample:
            traj_result = self.validate_trajectory_data(sample['trajectory'][None, :, :])
            results['modality_results']['trajectory'] = traj_result
            if not traj_result['valid']:
                results['valid'] = False
                results['errors'].extend([f"Trajectory: {e}" for e in traj_result['errors']])

        if 'graph' in sample:
            adj = sample['graph']['adjacency']
            features = sample['graph']['node_features']
            graph_result = self.validate_graph_data(adj[None, :, :], features[None, :, :])
            results['modality_results']['graph'] = graph_result
            if not graph_result['valid']:
                results['valid'] = False
                results['errors'].extend([f"Graph: {e}" for e in graph_result['errors']])

        if 'image' in sample:
            img_result = self.validate_image_data(sample['image'][None, :, :, :])
            results['modality_results']['image'] = img_result
            if not img_result['valid']:
                results['valid'] = False
                results['errors'].extend([f"Image: {e}" for e in img_result['errors']])

        return results

    def generate_validation_report(self, validation_results: List[Dict[str, Any]],
                                output_path: str) -> None:
        """
        Generate comprehensive validation report.

        Args:
            validation_results: List of validation result dictionaries
            output_path: Path to save report
        """
        report = {
            'summary': {
                'total_samples': len(validation_results),
                'valid_samples': sum(1 for r in validation_results if r['valid']),
                'invalid_samples': sum(1 for r in validation_results if not r['valid']),
                'total_errors': sum(len(r.get('errors', [])) for r in validation_results),
                'total_warnings': sum(len(r.get('warnings', [])) for r in validation_results)
            },
            'modality_stats': {},
            'error_summary': {}
        }

        # Aggregate modality statistics
        for result in validation_results:
            if result['valid'] and 'modality_results' in result:
                for modality, mod_result in result['modality_results'].items():
                    if modality not in report['modality_stats']:
                        report['modality_stats'][modality] = []
                    if 'stats' in mod_result:
                        report['modality_stats'][modality].append(mod_result['stats'])

        # Aggregate errors
        for result in validation_results:
            for error in result.get('errors', []):
                if error not in report['error_summary']:
                    report['error_summary'][error] = 0
                report['error_summary'][error] += 1

        # Save report
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            yaml.dump(report, f, default_flow_style=False)

        self.logger.info(f"Validation report saved to {output_path}")