"""
Feature Engineering for Collision Risk Assessment

Extracts and engineers features for collision risk prediction models.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from scipy.spatial.distance import mahalanobis
from scipy.stats import norm
import torch
import torch.nn.functional as F

class CollisionRiskFeatureEngineer:
    """
    Feature engineering for collision risk assessment.
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize feature engineer.

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
            'feature_types': ['kinematic', 'geometric', 'statistical', 'orbital'],
            'normalization': True,
            'pca_components': None
        }

    def extract_features(self, trajectory1: np.ndarray, trajectory2: np.ndarray,
                        conjunction_data: Optional[Dict] = None) -> Dict[str, np.ndarray]:
        """
        Extract features from satellite trajectories.

        Args:
            trajectory1: Trajectory of first satellite [n_timesteps, 6] (x,y,z,vx,vy,vz)
            trajectory2: Trajectory of second satellite [n_timesteps, 6]
            conjunction_data: Additional conjunction information

        Returns:
            Dictionary of extracted features
        """
        features = {}

        # Calculate relative trajectory
        relative_traj = trajectory1 - trajectory2

        # Extract different feature types
        if 'kinematic' in self.config['feature_types']:
            features.update(self._extract_kinematic_features(relative_traj))

        if 'geometric' in self.config['feature_types']:
            features.update(self._extract_geometric_features(relative_traj))

        if 'statistical' in self.config['feature_types']:
            features.update(self._extract_statistical_features(relative_traj))

        if 'orbital' in self.config['feature_types']:
            features.update(self._extract_orbital_features(trajectory1, trajectory2, conjunction_data))

        # Combine all features
        combined_features = np.concatenate([features[key] for key in sorted(features.keys())])

        if self.config['normalization']:
            combined_features = self._normalize_features(combined_features)

        return {
            'features': combined_features,
            'feature_names': sorted(features.keys()),
            'individual_features': features
        }

    def _extract_kinematic_features(self, relative_traj: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract kinematic features from relative trajectory."""
        features = {}

        # Position features
        positions = relative_traj[:, :3]
        features['position_mean'] = np.mean(positions, axis=0)
        features['position_std'] = np.std(positions, axis=0)
        features['position_min'] = np.min(positions, axis=0)
        features['position_max'] = np.max(positions, axis=0)

        # Velocity features
        velocities = relative_traj[:, 3:6]
        features['velocity_mean'] = np.mean(velocities, axis=0)
        features['velocity_std'] = np.std(velocities, axis=0)
        features['velocity_min'] = np.min(velocities, axis=0)
        features['velocity_max'] = np.max(velocities, axis=0)

        # Acceleration (numerical differentiation)
        if len(relative_traj) > 1:
            acceleration = np.diff(velocities, axis=0)
            features['acceleration_mean'] = np.mean(acceleration, axis=0)
            features['acceleration_std'] = np.std(acceleration, axis=0)

        # Relative speed
        speed = np.linalg.norm(velocities, axis=1)
        features['speed_mean'] = np.array([np.mean(speed)])
        features['speed_std'] = np.array([np.std(speed)])
        features['speed_max'] = np.array([np.max(speed)])

        return features

    def _extract_geometric_features(self, relative_traj: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract geometric features from relative trajectory."""
        features = {}

        positions = relative_traj[:, :3]

        # Distance features
        distances = np.linalg.norm(positions, axis=1)
        features['distance_mean'] = np.array([np.mean(distances)])
        features['distance_std'] = np.array([np.std(distances)])
        features['distance_min'] = np.array([np.min(distances)])
        features['distance_max'] = np.array([np.max(distances)])

        # Closest approach
        min_distance_idx = np.argmin(distances)
        features['min_distance'] = np.array([distances[min_distance_idx]])
        features['min_distance_time'] = np.array([min_distance_idx / len(distances)])

        # Trajectory curvature (simplified)
        if len(positions) > 2:
            # Calculate angles between consecutive position vectors
            vectors = positions[1:] - positions[:-1]
            angles = []
            for i in range(len(vectors) - 1):
                v1 = vectors[i]
                v2 = vectors[i + 1]
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                cos_angle = np.clip(cos_angle, -1, 1)
                angles.append(np.arccos(cos_angle))

            features['trajectory_curvature'] = np.array([np.mean(angles)])
            features['trajectory_curvature_std'] = np.array([np.std(angles)])

        # Orbital plane features
        # Simplified: assume trajectories are in orbital planes
        # Calculate relative inclination (simplified)
        features['relative_inclination'] = np.array([0.0])  # Placeholder

        return features

    def _extract_statistical_features(self, relative_traj: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract statistical features from relative trajectory."""
        features = {}

        positions = relative_traj[:, :3]
        velocities = relative_traj[:, 3:6]

        # Position statistics
        features['position_skewness'] = self._skewness(positions, axis=0)
        features['position_kurtosis'] = self._kurtosis(positions, axis=0)

        # Velocity statistics
        features['velocity_skewness'] = self._skewness(velocities, axis=0)
        features['velocity_kurtosis'] = self._kurtosis(velocities, axis=0)

        # Cross-correlations
        for i in range(3):
            for j in range(3):
                if i != j:
                    corr = np.corrcoef(positions[:, i], positions[:, j])[0, 1]
                    features[f'position_corr_{i}{j}'] = np.array([corr])

                    corr = np.corrcoef(velocities[:, i], velocities[:, j])[0, 1]
                    features[f'velocity_corr_{i}{j}'] = np.array([corr])

        # Autocorrelation
        if len(positions) > 10:
            autocorr = np.correlate(positions[:, 0], positions[:, 0], mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            autocorr = autocorr[:10]  # First 10 lags
            features['position_autocorr'] = autocorr / autocorr[0] if autocorr[0] != 0 else autocorr

        return features

    def _extract_orbital_features(self, trajectory1: np.ndarray, trajectory2: np.ndarray,
                                conjunction_data: Optional[Dict] = None) -> Dict[str, np.ndarray]:
        """Extract orbital mechanics features."""
        features = {}

        # Orbital elements (simplified estimation)
        orb_elements1 = self._estimate_orbital_elements(trajectory1)
        orb_elements2 = self._estimate_orbital_elements(trajectory2)

        # Relative orbital elements
        for key in orb_elements1.keys():
            if key in orb_elements2:
                features[f'relative_{key}'] = orb_elements1[key] - orb_elements2[key]

        # Conjunction-specific features
        if conjunction_data:
            features['miss_distance'] = np.array([conjunction_data.get('miss_distance', 0)])
            features['relative_velocity'] = np.array([conjunction_data.get('relative_velocity', 0)])
            features['time_to_tca'] = np.array([conjunction_data.get('time_to_tca', 0)])

        # Collision probability using simplified Pc calculation
        pc = self._calculate_collision_probability(trajectory1, trajectory2, conjunction_data)
        features['collision_probability'] = np.array([pc])

        return features

    def _estimate_orbital_elements(self, trajectory: np.ndarray) -> Dict[str, float]:
        """Estimate orbital elements from trajectory (simplified)."""
        # This is a highly simplified estimation
        # In practice, would use proper orbital element calculation

        positions = trajectory[:, :3]
        velocities = trajectory[:, 3:6]

        # Semi-major axis (approximate)
        r = np.linalg.norm(positions, axis=1)
        v = np.linalg.norm(velocities, axis=1)
        a = 1 / (2/np.mean(r) - np.mean(v**2) / 398600.4418)  # mu_earth = 398600.4418 km^3/s^2

        # Eccentricity (simplified)
        e = np.std(r) / np.mean(r)

        # Inclination (simplified - assume equatorial if z components are small)
        inclination = np.arctan2(np.std(positions[:, 2]), np.sqrt(np.std(positions[:, 0])**2 + np.std(positions[:, 1])**2))

        return {
            'semi_major_axis': a,
            'eccentricity': e,
            'inclination': inclination
        }

    def _calculate_collision_probability(self, trajectory1: np.ndarray, trajectory2: np.ndarray,
                                       conjunction_data: Optional[Dict] = None) -> float:
        """Calculate collision probability using simplified method."""
        # Simplified Pc calculation
        # In practice, would use proper conjunction assessment

        if conjunction_data and 'miss_distance' in conjunction_data:
            # Use provided miss distance
            miss_distance = conjunction_data['miss_distance']
            relative_velocity = conjunction_data.get('relative_velocity', 10)  # km/s

            # Hard body radius (simplified)
            radius1 = radius2 = 0.1  # 100m radius for satellites

            # Combined radius
            combined_radius = radius1 + radius2

            # Pc calculation (simplified cylindrical approximation)
            if miss_distance <= combined_radius:
                pc = 1.0
            else:
                # Simplified probability calculation
                sigma = combined_radius / 3  # Assume 3-sigma containment
                pc = 2 * (1 - norm.cdf(miss_distance / sigma))

            return min(pc, 1.0)

        # Fallback: calculate from trajectories
        relative_positions = trajectory1[:, :3] - trajectory2[:, :3]
        distances = np.linalg.norm(relative_positions, axis=1)
        min_distance = np.min(distances)

        # Simple threshold-based probability
        if min_distance < 1.0:  # 1km threshold
            return 0.1  # 10% probability for close approaches
        elif min_distance < 10.0:
            return 0.01
        else:
            return 0.001

    def _skewness(self, data: np.ndarray, axis: int = 0) -> np.ndarray:
        """Calculate skewness of data."""
        mean = np.mean(data, axis=axis, keepdims=True)
        std = np.std(data, axis=axis, keepdims=True)
        skewness = np.mean(((data - mean) / std)**3, axis=axis)
        return skewness

    def _kurtosis(self, data: np.ndarray, axis: int = 0) -> np.ndarray:
        """Calculate kurtosis of data."""
        mean = np.mean(data, axis=axis, keepdims=True)
        std = np.std(data, axis=axis, keepdims=True)
        kurtosis = np.mean(((data - mean) / std)**4, axis=axis) - 3
        return kurtosis

    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize features using z-score normalization."""
        mean = np.mean(features)
        std = np.std(features)

        if std > 0:
            return (features - mean) / std
        else:
            return features

    def extract_batch_features(self, trajectories1: List[np.ndarray],
                              trajectories2: List[np.ndarray],
                              conjunction_data: Optional[List[Dict]] = None) -> torch.Tensor:
        """
        Extract features for a batch of trajectory pairs.

        Args:
            trajectories1: List of first satellite trajectories
            trajectories2: List of second satellite trajectories
            conjunction_data: List of conjunction data

        Returns:
            Batch of features [batch_size, feature_dim]
        """
        batch_features = []

        for i, (traj1, traj2) in enumerate(zip(trajectories1, trajectories2)):
            conj_data = conjunction_data[i] if conjunction_data else None
            features_dict = self.extract_features(traj1, traj2, conj_data)
            batch_features.append(features_dict['features'])

        # Pad to same length if necessary
        max_len = max(len(f) for f in batch_features)
        padded_features = []
        for f in batch_features:
            if len(f) < max_len:
                # Pad with zeros
                padding = np.zeros(max_len - len(f))
                f = np.concatenate([f, padding])
            padded_features.append(f)

        return torch.tensor(np.array(padded_features), dtype=torch.float32)


if __name__ == "__main__":
    # Example usage
    engineer = CollisionRiskFeatureEngineer()

    # Generate sample trajectories
    np.random.seed(42)
    t = np.linspace(0, 10, 100)
    traj1 = np.column_stack([
        7000 + 100*np.sin(t),  # x
        100*np.cos(t),         # y
        50*np.sin(0.1*t),      # z
        -100*np.cos(t),        # vx
        100*np.sin(t),         # vy
        5*np.cos(0.1*t)        # vz
    ])

    traj2 = np.column_stack([
        7000 + 80*np.sin(t + 0.5),  # x
        80*np.cos(t + 0.5),         # y
        40*np.sin(0.1*t + 0.5),     # z
        -80*np.cos(t + 0.5),        # vx
        80*np.sin(t + 0.5),         # vy
        4*np.cos(0.1*t + 0.5)       # vz
    ])

    # Extract features
    features = engineer.extract_features(traj1, traj2)

    print(f"Extracted {len(features['features'])} features")
    print(f"Feature types: {features['feature_names']}")

    # Show some feature values
    for name, value in list(features['individual_features'].items())[:5]:
        print(f"{name}: {value}")