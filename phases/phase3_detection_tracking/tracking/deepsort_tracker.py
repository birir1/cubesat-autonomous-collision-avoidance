"""
DeepSORT Satellite Object Tracker

Implements DeepSORT (Deep Learning-based SORT) for tracking satellites
and space objects across image sequences.
"""

import numpy as np
import torch
import torch.nn as nn
from collections import deque
import logging
from typing import List, Dict, Tuple, Optional, Any
from scipy.spatial.distance import cdist

try:
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    cosine_similarity = None

class KalmanFilter:
    """
    Kalman filter for object state estimation.
    """

    def __init__(self, dim_x=4, dim_z=2):
        """
        Initialize Kalman filter.

        Args:
            dim_x: State dimension (x, y, vx, vy)
            dim_z: Measurement dimension (x, y)
        """
        self.dim_x = dim_x
        self.dim_z = dim_z

        # State transition matrix
        self.F = np.eye(dim_x)
        self.F[0, 2] = 1  # x += vx
        self.F[1, 3] = 1  # y += vy

        # Measurement matrix
        self.H = np.zeros((dim_z, dim_x))
        self.H[0, 0] = 1  # measure x
        self.H[1, 1] = 1  # measure y

        # Process noise
        self.Q = np.eye(dim_x) * 0.01

        # Measurement noise
        self.R = np.eye(dim_z) * 0.1

        # Initial state covariance
        self.P = np.eye(dim_x) * 100

    def predict(self, x):
        """Predict next state."""
        x_pred = self.F @ x
        P_pred = self.F @ self.P @ self.F.T + self.Q
        return x_pred, P_pred

    def update(self, x_pred, P_pred, z):
        """Update state with measurement."""
        # Innovation
        y = z - self.H @ x_pred
        S = self.H @ P_pred @ self.H.T + self.R

        # Kalman gain
        K = P_pred @ self.H.T @ np.linalg.inv(S)

        # Update state
        x_updated = x_pred + K @ y
        P_updated = (np.eye(self.dim_x) - K @ self.H) @ P_pred

        return x_updated, P_updated


class Track:
    """
    Represents a single object track.
    """

    def __init__(self, track_id: int, bbox: np.ndarray, feature: Optional[np.ndarray] = None):
        """
        Initialize track.

        Args:
            track_id: Unique track identifier
            bbox: Initial bounding box [x1, y1, x2, y2]
            feature: Appearance feature vector
        """
        self.track_id = track_id
        self.bbox = bbox
        self.feature = feature
        self.age = 1
        self.time_since_update = 0
        self.hits = 1
        self.hit_streak = 1

        # Kalman filter state [x, y, vx, vy]
        center = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
        self.kf = KalmanFilter()
        self.kf_state = np.concatenate([center, np.zeros(2)])  # [x, y, 0, 0]

        # Track history
        self.history = deque(maxlen=30)
        self.history.append(bbox)

    def update(self, bbox: np.ndarray, feature: Optional[np.ndarray] = None):
        """Update track with new detection."""
        self.bbox = bbox
        self.feature = feature if feature is not None else self.feature
        self.age += 1
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1

        # Update Kalman filter
        center = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
        self.kf_state, _ = self.kf.update(self.kf_state, self.kf.P, center)

        # Add to history
        self.history.append(bbox)

    def predict(self):
        """Predict next state."""
        self.kf_state, _ = self.kf.predict(self.kf_state)
        self.age += 1
        self.time_since_update += 1

        # Update bbox from predicted state
        center = self.kf_state[:2]
        # Use last known size (simplified)
        if len(self.history) > 0:
            last_bbox = self.history[-1]
            w = last_bbox[2] - last_bbox[0]
            h = last_bbox[3] - last_bbox[1]
            self.bbox = np.array([
                center[0] - w/2, center[1] - h/2,
                center[0] + w/2, center[1] + h/2
            ])

    def get_state(self) -> Dict:
        """Get current track state."""
        return {
            'id': self.track_id,
            'bbox': self.bbox,
            'age': self.age,
            'hits': self.hits,
            'time_since_update': self.time_since_update,
            'feature': self.feature
        }


class DeepSORTTracker:
    """
    DeepSORT tracker for satellite object tracking.
    """

    def __init__(self, max_age: int = 30, n_init: int = 3,
                 max_iou_distance: float = 0.7, max_cosine_distance: float = 0.2,
                 nn_budget: int = 100):
        """
        Initialize DeepSORT tracker.

        Args:
            max_age: Maximum age of track before deletion
            n_init: Number of frames to wait before confirming track
            max_iou_distance: Maximum IoU distance for matching
            max_cosine_distance: Maximum cosine distance for appearance matching
            nn_budget: Budget for nearest neighbor search
        """
        self.max_age = max_age
        self.n_init = n_init
        self.max_iou_distance = max_iou_distance
        self.max_cosine_distance = max_cosine_distance
        self.nn_budget = nn_budget

        self.tracks: List[Track] = []
        self.next_id = 0
        self.logger = logging.getLogger(__name__)

        # Feature extractor (placeholder)
        self.feature_extractor = self._create_feature_extractor()

    def _create_feature_extractor(self):
        """Create CNN feature extractor for appearance matching."""
        class SimpleFeatureExtractor(nn.Module):
            def __init__(self):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, 64, 3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(64, 128, 3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1, 1))
                )

            def forward(self, x):
                return self.features(x).view(x.size(0), -1)

        return SimpleFeatureExtractor()

    def update(self, detections: List[Dict], timestamp: Optional[float] = None) -> List[Dict]:
        """
        Update tracker with new detections.

        Args:
            detections: List of detections [{'bbox': [x1,y1,x2,y2], 'confidence': float, 'class': int}, ...]
            timestamp: Current timestamp

        Returns:
            List of active tracks
        """
        # Extract bboxes and features from detections
        bboxes = []
        features = []

        for det in detections:
            bbox = np.array(det['bbox'])
            bboxes.append(bbox)

            # Extract features (placeholder - in practice would crop and process image)
            feature = np.random.randn(128)  # Random feature for now
            features.append(feature)

        bboxes = np.array(bboxes) if bboxes else np.empty((0, 4))
        features = np.array(features) if features else np.empty((0, 128))

        # Predict new locations for existing tracks
        for track in self.tracks:
            track.predict()

        # Associate detections with tracks
        matches, unmatched_detections, unmatched_tracks = self._associate_detections_to_tracks(
            bboxes, features
        )

        # Update matched tracks
        for track_idx, det_idx in matches:
            self.tracks[track_idx].update(bboxes[det_idx], features[det_idx])

        # Create new tracks for unmatched detections
        for det_idx in unmatched_detections:
            if len(bboxes) > det_idx:
                self._create_track(bboxes[det_idx], features[det_idx])

        # Mark unmatched tracks as lost
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].time_since_update += 1

        # Remove dead tracks
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        # Return confirmed tracks
        active_tracks = [t.get_state() for t in self.tracks if t.hits >= self.n_init]

        return active_tracks

    def _associate_detections_to_tracks(self, bboxes: np.ndarray, features: np.ndarray):
        """
        Associate detections to existing tracks using IoU and appearance.

        Args:
            bboxes: Detection bounding boxes
            features: Detection features

        Returns:
            matches, unmatched_detections, unmatched_tracks
        """
        if len(self.tracks) == 0:
            return [], list(range(len(bboxes))), []

        if len(bboxes) == 0:
            return [], [], list(range(len(self.tracks)))

        # Calculate IoU distance matrix
        track_bboxes = np.array([t.bbox for t in self.tracks])
        iou_matrix = self._iou_distance(track_bboxes, bboxes)
        iou_matrix = np.where(iou_matrix > self.max_iou_distance, 1.0, iou_matrix)

        # Calculate appearance distance matrix
        if len(features) > 0 and cosine_similarity is not None:
            track_features = np.array([t.feature for t in self.tracks if t.feature is not None])
            if len(track_features) > 0:
                appearance_matrix = 1 - cosine_similarity(track_features, features)
                appearance_matrix = np.where(appearance_matrix > self.max_cosine_distance,
                                           1.0, appearance_matrix)
            else:
                appearance_matrix = np.ones((len(self.tracks), len(features)))
        else:
            appearance_matrix = np.ones((len(self.tracks), len(features)))

        # Combine distances (simple average)
        distance_matrix = (iou_matrix + appearance_matrix) / 2

        # Find matches using Hungarian algorithm
        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(distance_matrix)

        matches = []
        unmatched_tracks = list(range(len(self.tracks)))
        unmatched_detections = list(range(len(bboxes)))

        for row, col in zip(row_ind, col_ind):
            if distance_matrix[row, col] < 0.5:  # Matching threshold
                matches.append((row, col))
                if row in unmatched_tracks:
                    unmatched_tracks.remove(row)
                if col in unmatched_detections:
                    unmatched_detections.remove(col)

        return matches, unmatched_detections, unmatched_tracks

    def _iou_distance(self, bboxes1: np.ndarray, bboxes2: np.ndarray) -> np.ndarray:
        """
        Calculate IoU distance between bounding boxes.

        Args:
            bboxes1: First set of bboxes (N, 4)
            bboxes2: Second set of bboxes (M, 4)

        Returns:
            IoU distance matrix (N, M)
        """
        def bbox_iou(box1, box2):
            x1 = max(box1[0], box2[0])
            y1 = max(box1[1], box2[1])
            x2 = min(box1[2], box2[2])
            y2 = min(box1[3], box2[3])

            intersection = max(0, x2 - x1) * max(0, y2 - y1)
            area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
            area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
            union = area1 + area2 - intersection

            return intersection / union if union > 0 else 0

        n1, n2 = len(bboxes1), len(bboxes2)
        iou_matrix = np.zeros((n1, n2))

        for i in range(n1):
            for j in range(n2):
                iou_matrix[i, j] = 1 - bbox_iou(bboxes1[i], bboxes2[j])  # Distance = 1 - IoU

        return iou_matrix

    def _create_track(self, bbox: np.ndarray, feature: np.ndarray):
        """Create new track."""
        track = Track(self.next_id, bbox, feature)
        self.tracks.append(track)
        self.next_id += 1

    def get_track_history(self, track_id: int) -> Optional[List[np.ndarray]]:
        """Get trajectory history for a track."""
        for track in self.tracks:
            if track.track_id == track_id:
                return list(track.history)
        return None

    def get_all_tracks(self) -> List[Dict]:
        """Get all active tracks."""
        return [t.get_state() for t in self.tracks]

    def reset(self):
        """Reset tracker state."""
        self.tracks = []
        self.next_id = 0


class SatelliteTrack:
    """
    Extended track class for satellite-specific tracking.
    """

    def __init__(self, track_id: int, bbox: np.ndarray, feature: Optional[np.ndarray] = None,
                 orbital_elements: Optional[Dict] = None):
        """
        Initialize satellite track.

        Args:
            track_id: Unique track identifier
            bbox: Initial bounding box
            feature: Appearance feature
            orbital_elements: Orbital elements (a, e, i, etc.)
        """
        super().__init__(track_id, bbox, feature)
        self.orbital_elements = orbital_elements or {}
        self.velocity_estimate = np.zeros(3)  # 3D velocity
        self.position_estimate = np.zeros(3)  # 3D position

    def update_orbital_state(self, new_elements: Dict):
        """Update orbital elements."""
        self.orbital_elements.update(new_elements)

    def predict_orbit(self, dt: float):
        """Predict orbital position using Kepler's laws (simplified)."""
        # Simplified orbital prediction
        # In practice, would use SGP4 or numerical integration
        if 'a' in self.orbital_elements:  # semi-major axis
            a = self.orbital_elements['a']
            n = np.sqrt(3.986004418e14 / a**3)  # mean motion
            M = n * dt  # mean anomaly
            self.position_estimate[0] += a * np.cos(M) * 0.01  # Simplified
            self.position_estimate[1] += a * np.sin(M) * 0.01


if __name__ == "__main__":
    # Example usage
    tracker = DeepSORTTracker()

    # Create dummy detections
    detections = [
        {'bbox': [100, 100, 150, 150], 'confidence': 0.9, 'class': 0},
        {'bbox': [200, 200, 250, 250], 'confidence': 0.8, 'class': 1}
    ]

    # Update tracker
    tracks = tracker.update(detections)

    print(f"Active tracks: {len(tracks)}")
    for track in tracks:
        print(f"Track {track['id']}: bbox={track['bbox']}, age={track['age']}")

    # Update with new detections
    new_detections = [
        {'bbox': [105, 105, 155, 155], 'confidence': 0.85, 'class': 0},
        {'bbox': [300, 300, 350, 350], 'confidence': 0.7, 'class': 1}
    ]

    tracks = tracker.update(new_detections)
    print(f"Updated tracks: {len(tracks)}")
    for track in tracks:
        print(f"Track {track['id']}: bbox={track['bbox']}, age={track['age']}")