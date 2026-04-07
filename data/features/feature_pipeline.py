"""
Feature Engineering Pipeline for CubeSat Collision Avoidance

Centralized feature computation across all modalities and models.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime, timedelta
import logging
from pathlib import Path
import yaml

class FeaturePipeline:
    """
    Unified feature engineering pipeline for multimodal collision avoidance.
    """

    def __init__(self, config_path: str = 'configs/data_config.yaml'):
        """
        Initialize feature pipeline.

        Args:
            config_path: Path to configuration file
        """
        self.logger = logging.getLogger(__name__)

        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Feature normalization parameters
        self.normalize = self.config['features']['normalize']
        self.feature_names = self.config['features']['feature_names']

        # Initialize normalization statistics
        self.feature_stats = {}

    def extract_trajectory_features(self, trajectory: np.ndarray) -> np.ndarray:
        """
        Extract features from trajectory time-series.

        Args:
            trajectory: Shape (seq_len, 6) - [rel_pos_x, rel_pos_y, rel_pos_z, rel_vel_x, rel_vel_y, rel_vel_z]

        Returns:
            Feature vector
        """
        if trajectory.shape[1] != 6:
            raise ValueError(f"Trajectory must have 6 features, got {trajectory.shape[1]}")

        features = []

        # Basic relative state features
        final_rel_pos = trajectory[-1, :3]  # Final relative position
        final_rel_vel = trajectory[-1, 3:6]  # Final relative velocity

        features.extend(final_rel_pos)
        features.extend(final_rel_vel)

        # Distance features
        distances = np.linalg.norm(trajectory[:, :3], axis=1)
        final_distance = distances[-1]
        min_distance = np.min(distances)
        max_distance = np.max(distances)
        mean_distance = np.mean(distances)

        features.extend([final_distance, min_distance, max_distance, mean_distance])

        # Velocity features
        velocities = np.linalg.norm(trajectory[:, 3:6], axis=1)
        final_speed = velocities[-1]
        max_speed = np.max(velocities)
        mean_speed = np.mean(velocities)

        features.extend([final_speed, max_speed, mean_speed])

        # Time to closest approach (TCA) estimation
        min_dist_idx = np.argmin(distances)
        tca_hours = min_dist_idx * self.config['trajectory']['time_step_minutes'] / 60.0
        features.append(tca_hours)

        # Approach rate (change in distance)
        if len(distances) > 1:
            approach_rate = (distances[-1] - distances[0]) / (len(distances) - 1)
        else:
            approach_rate = 0.0
        features.append(approach_rate)

        # Collision probability proxy (based on minimum distance)
        collision_prob = np.exp(-min_distance / 10.0)  # Simple exponential decay
        features.append(collision_prob)

        return np.array(features, dtype=np.float32)

    def extract_graph_features(self, adjacency: np.ndarray,
                             node_features: np.ndarray,
                             target_node_idx: int = 0) -> np.ndarray:
        """
        Extract features from graph structure.

        Args:
            adjacency: Adjacency matrix
            node_features: Node feature matrix
            target_node_idx: Index of target satellite

        Returns:
            Graph feature vector
        """
        features = []

        # Local neighborhood features
        target_neighbors = np.where(adjacency[target_node_idx] > 0)[0]
        num_neighbors = len(target_neighbors)

        features.append(num_neighbors)

        # Neighbor density
        total_possible_neighbors = len(adjacency) - 1  # Exclude self
        neighbor_density = num_neighbors / max(total_possible_neighbors, 1)
        features.append(neighbor_density)

        # Average neighbor distance
        if num_neighbors > 0:
            neighbor_distances = adjacency[target_node_idx, target_neighbors]
            avg_neighbor_dist = np.mean(neighbor_distances)
            min_neighbor_dist = np.min(neighbor_distances)
            max_neighbor_dist = np.max(neighbor_distances)
        else:
            avg_neighbor_dist = min_neighbor_dist = max_neighbor_dist = 0.0

        features.extend([avg_neighbor_dist, min_neighbor_dist, max_neighbor_dist])

        # Graph centrality measures
        degrees = adjacency.sum(axis=1)
        target_degree = degrees[target_node_idx]
        avg_degree = np.mean(degrees)
        degree_centrality = target_degree / max(np.sum(degrees), 1)

        features.extend([target_degree, avg_degree, degree_centrality])

        # Clustering coefficient approximation
        if num_neighbors > 1:
            # Count triangles involving target node
            triangles = 0
            for i in target_neighbors:
                for j in target_neighbors:
                    if i != j and adjacency[i, j] > 0:
                        triangles += 1
            clustering_coeff = triangles / (num_neighbors * (num_neighbors - 1))
        else:
            clustering_coeff = 0.0

        features.append(clustering_coeff)

        return np.array(features, dtype=np.float32)

    def extract_vision_features(self, image: np.ndarray,
                              detections: Optional[List[Dict]] = None) -> np.ndarray:
        """
        Extract features from vision data.

        Args:
            image: Image array (channels, height, width)
            detections: Optional object detection results

        Returns:
            Vision feature vector
        """
        features = []

        # Basic image statistics
        if len(image.shape) == 3:
            # Multi-channel image
            for channel in range(image.shape[0]):
                channel_data = image[channel].flatten()
                features.extend([
                    np.mean(channel_data),
                    np.std(channel_data),
                    np.min(channel_data),
                    np.max(channel_data)
                ])
        else:
            # Single channel or processed image
            image_data = image.flatten()
            features.extend([
                np.mean(image_data),
                np.std(image_data),
                np.min(image_data),
                np.max(image_data)
            ])

        # Detection-based features
        if detections:
            num_detections = len(detections)
            features.append(num_detections)

            if num_detections > 0:
                # Average confidence
                avg_confidence = np.mean([d.get('confidence', 0) for d in detections])
                features.append(avg_confidence)

                # Detection density
                image_area = image.shape[-2] * image.shape[-1] if len(image.shape) >= 2 else 1
                total_bbox_area = sum([
                    (d.get('bbox', [0, 0, 1, 1])[2] - d.get('bbox', [0, 0, 1, 1])[0]) *
                    (d.get('bbox', [0, 0, 1, 1])[3] - d.get('bbox', [0, 0, 1, 1])[1])
                    for d in detections
                ])
                detection_density = total_bbox_area / image_area
                features.append(detection_density)

                # Closest detection distance (proxy for threat level)
                if 'distance' in detections[0]:
                    min_distance = min(d.get('distance', float('inf')) for d in detections)
                    features.append(min_distance)
                else:
                    features.append(0.0)  # Placeholder
            else:
                features.extend([0.0, 0.0, 0.0])  # No detections
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])  # No detection data

        return np.array(features, dtype=np.float32)

    def extract_conjunction_features(self, conjunction_data: Dict[str, Any]) -> np.ndarray:
        """
        Extract features from conjunction assessment data.

        Args:
            conjunction_data: Conjunction event data

        Returns:
            Conjunction feature vector
        """
        features = []

        # Basic conjunction features
        features.append(conjunction_data.get('miss_distance_km', 0))
        features.append(conjunction_data.get('relative_velocity_kms', 0))
        features.append(conjunction_data.get('time_to_tca_hours', 0))
        features.append(conjunction_data.get('collision_probability', 0))

        # Relative position components
        features.append(conjunction_data.get('relative_position_x_km', 0))
        features.append(conjunction_data.get('relative_position_y_km', 0))
        features.append(conjunction_data.get('relative_position_z_km', 0))

        # Orbital parameters
        features.append(conjunction_data.get('sat1_inclination_deg', 0))
        features.append(conjunction_data.get('sat2_inclination_deg', 0))
        features.append(conjunction_data.get('sat1_altitude_km', 6371))
        features.append(conjunction_data.get('sat2_altitude_km', 6371))

        # Derived features
        rel_pos = np.array([
            conjunction_data.get('relative_position_x_km', 0),
            conjunction_data.get('relative_position_y_km', 0),
            conjunction_data.get('relative_position_z_km', 0)
        ])
        distance_3d = np.linalg.norm(rel_pos)
        features.append(distance_3d)

        # Risk indicators
        miss_distance_m = conjunction_data.get('miss_distance_km', 0) * 1000
        collision_threshold_m = self.config['risk']['collision_threshold_m']

        risk_score = 1.0 / (1.0 + miss_distance_m / collision_threshold_m)
        features.append(risk_score)

        return np.array(features, dtype=np.float32)

    def combine_multimodal_features(self, trajectory_features: Optional[np.ndarray] = None,
                                  graph_features: Optional[np.ndarray] = None,
                                  vision_features: Optional[np.ndarray] = None,
                                  conjunction_features: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Combine features from all modalities into unified feature vector.

        Args:
            trajectory_features: Features from trajectory analysis
            graph_features: Features from graph analysis
            vision_features: Features from vision analysis
            conjunction_features: Features from conjunction data

        Returns:
            Combined feature vector
        """
        combined_features = []

        if trajectory_features is not None:
            combined_features.extend(trajectory_features)

        if graph_features is not None:
            combined_features.extend(graph_features)

        if vision_features is not None:
            combined_features.extend(vision_features)

        if conjunction_features is not None:
            combined_features.extend(conjunction_features)

        return np.array(combined_features, dtype=np.float32)

    def normalize_features(self, features: np.ndarray, feature_type: str = 'combined') -> np.ndarray:
        """
        Normalize features using pre-computed statistics.

        Args:
            features: Feature vector or matrix
            feature_type: Type of features for normalization

        Returns:
            Normalized features
        """
        if not self.normalize:
            return features

        if feature_type not in self.feature_stats:
            # Compute statistics if not available
            if features.ndim == 1:
                self.feature_stats[feature_type] = {
                    'mean': features,
                    'std': np.ones_like(features)
                }
            else:
                self.feature_stats[feature_type] = {
                    'mean': np.mean(features, axis=0),
                    'std': np.std(features, axis=0)
                }

        stats = self.feature_stats[feature_type]

        # Avoid division by zero
        safe_std = np.where(stats['std'] == 0, 1.0, stats['std'])

        if features.ndim == 1:
            normalized = (features - stats['mean']) / safe_std
        else:
            normalized = (features - stats['mean']) / safe_std

        return normalized

    def fit_normalization(self, feature_matrix: np.ndarray, feature_type: str = 'combined') -> None:
        """
        Fit normalization parameters from training data.

        Args:
            feature_matrix: Training feature matrix (samples, features)
            feature_type: Type of features
        """
        if not self.normalize:
            return

        self.feature_stats[feature_type] = {
            'mean': np.mean(feature_matrix, axis=0),
            'std': np.std(feature_matrix, axis=0)
        }

        self.logger.info(f"Fitted normalization for {feature_type} features: mean={self.feature_stats[feature_type]['mean'].shape}")

    def process_sample(self, trajectory: Optional[np.ndarray] = None,
                      graph_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                      image: Optional[np.ndarray] = None,
                      conjunction_data: Optional[Dict[str, Any]] = None,
                      detections: Optional[List[Dict]] = None) -> np.ndarray:
        """
        Process a complete multimodal sample.

        Args:
            trajectory: Trajectory time-series
            graph_data: Tuple of (adjacency, node_features)
            image: Image data
            conjunction_data: Conjunction assessment data
            detections: Vision detections

        Returns:
            Unified feature vector
        """
        # Extract features from each modality
        trajectory_features = None
        if trajectory is not None:
            trajectory_features = self.extract_trajectory_features(trajectory)

        graph_features = None
        if graph_data is not None:
            adjacency, node_features = graph_data
            graph_features = self.extract_graph_features(adjacency, node_features)

        vision_features = None
        if image is not None:
            vision_features = self.extract_vision_features(image, detections)

        conjunction_features = None
        if conjunction_data is not None:
            conjunction_features = self.extract_conjunction_features(conjunction_data)

        # Combine all features
        combined_features = self.combine_multimodal_features(
            trajectory_features, graph_features, vision_features, conjunction_features
        )

        # Normalize
        combined_features = self.normalize_features(combined_features)

        return combined_features

    def save_feature_stats(self, output_path: str) -> None:
        """
        Save feature normalization statistics.

        Args:
            output_path: Output file path
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Convert numpy arrays to lists for JSON serialization
        serializable_stats = {}
        for feature_type, stats in self.feature_stats.items():
            serializable_stats[feature_type] = {
                'mean': stats['mean'].tolist(),
                'std': stats['std'].tolist()
            }

        with open(output_path, 'w') as f:
            json.dump(serializable_stats, f, indent=2)

        self.logger.info(f"Saved feature statistics to {output_path}")

    def load_feature_stats(self, input_path: str) -> None:
        """
        Load feature normalization statistics.

        Args:
            input_path: Input file path
        """
        with open(input_path, 'r') as f:
            serializable_stats = json.load(f)

        self.feature_stats = {}
        for feature_type, stats in serializable_stats.items():
            self.feature_stats[feature_type] = {
                'mean': np.array(stats['mean']),
                'std': np.array(stats['std'])
            }

        self.logger.info(f"Loaded feature statistics from {input_path}")

    def get_feature_names(self) -> List[str]:
        """
        Get list of feature names in order.

        Returns:
            List of feature names
        """
        return self.feature_names

    def validate_feature_vector(self, features: np.ndarray) -> bool:
        """
        Validate feature vector for NaNs and reasonable ranges.

        Args:
            features: Feature vector

        Returns:
            True if valid, False otherwise
        """
        # Check for NaNs
        if np.any(np.isnan(features)):
            self.logger.warning("Feature vector contains NaN values")
            return False

        # Check for infinities
        if np.any(np.isinf(features)):
            self.logger.warning("Feature vector contains infinite values")
            return False

        # Check for reasonable ranges (not too extreme)
        if np.any(np.abs(features) > 1e10):
            self.logger.warning("Feature vector contains extremely large values")
            return False

        return True