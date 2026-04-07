"""
Kalman Filter Satellite Object Tracker

Implements Kalman filter-based tracking for satellites and space objects
with orbital mechanics integration.
"""

import numpy as np
import logging
from typing import List, Dict, Tuple, Optional, Any
from collections import deque
import time

class KalmanTracker:
    """
    Kalman filter-based tracker for satellite object tracking.
    """

    def __init__(self, dt: float = 1.0, process_noise: float = 0.1,
                 measurement_noise: float = 0.1, max_age: int = 30):
        """
        Initialize Kalman tracker.

        Args:
            dt: Time step between measurements
            process_noise: Process noise variance
            measurement_noise: Measurement noise variance
            max_age: Maximum track age before deletion
        """
        self.dt = dt
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.max_age = max_age

        # Track management
        self.tracks: List[KalmanTrack] = []
        self.next_id = 0
        self.logger = logging.getLogger(__name__)

    def update(self, detections: List[Dict], timestamp: Optional[float] = None) -> List[Dict]:
        """
        Update tracker with new detections.

        Args:
            detections: List of detections [{'bbox': [x1,y1,x2,y2], 'confidence': float, 'class': int}, ...]
            timestamp: Current timestamp

        Returns:
            List of active tracks
        """
        if timestamp is None:
            timestamp = time.time()

        # Predict all existing tracks
        for track in self.tracks:
            track.predict()

        # Associate detections to tracks
        matches, unmatched_detections, unmatched_tracks = self._associate_detections(
            detections
        )

        # Update matched tracks
        for track_idx, det_idx in matches:
            self.tracks[track_idx].update(detections[det_idx], timestamp)

        # Create new tracks for unmatched detections
        for det_idx in unmatched_detections:
            self._create_track(detections[det_idx], timestamp)

        # Mark unmatched tracks as missed
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].missed()

        # Remove dead tracks
        self.tracks = [t for t in self.tracks if t.age <= self.max_age and not t.is_dead()]

        # Return active tracks
        active_tracks = [t.get_state() for t in self.tracks if t.is_confirmed()]

        return active_tracks

    def _associate_detections(self, detections: List[Dict]) -> Tuple[List, List, List]:
        """
        Associate detections to existing tracks using nearest neighbor.

        Args:
            detections: List of detections

        Returns:
            matches, unmatched_detections, unmatched_tracks
        """
        if len(self.tracks) == 0:
            return [], list(range(len(detections))), []

        if len(detections) == 0:
            return [], [], list(range(len(self.tracks)))

        # Calculate distance matrix
        distance_matrix = np.zeros((len(self.tracks), len(detections)))

        for i, track in enumerate(self.tracks):
            track_center = track.get_position()
            for j, det in enumerate(detections):
                det_center = np.array([
                    (det['bbox'][0] + det['bbox'][2]) / 2,
                    (det['bbox'][1] + det['bbox'][3]) / 2
                ])
                distance_matrix[i, j] = np.linalg.norm(track_center - det_center)

        # Find matches using greedy assignment
        matches = []
        unmatched_tracks = set(range(len(self.tracks)))
        unmatched_detections = set(range(len(detections)))

        # Sort by distance
        sorted_indices = np.argsort(distance_matrix.flatten())
        rows, cols = np.unravel_index(sorted_indices, distance_matrix.shape)

        for row, col in zip(rows, cols):
            if row in unmatched_tracks and col in unmatched_detections:
                if distance_matrix[row, col] < 50:  # Distance threshold
                    matches.append((row, col))
                    unmatched_tracks.remove(row)
                    unmatched_detections.remove(col)

        return matches, list(unmatched_detections), list(unmatched_tracks)

    def _create_track(self, detection: Dict, timestamp: float):
        """Create new track from detection."""
        track = KalmanTrack(self.next_id, detection, timestamp,
                          self.dt, self.process_noise, self.measurement_noise)
        self.tracks.append(track)
        self.next_id += 1

    def get_track_history(self, track_id: int) -> Optional[List[np.ndarray]]:
        """Get trajectory history for a track."""
        for track in self.tracks:
            if track.track_id == track_id:
                return list(track.position_history)
        return None

    def get_all_tracks(self) -> List[Dict]:
        """Get all active tracks."""
        return [t.get_state() for t in self.tracks]

    def reset(self):
        """Reset tracker state."""
        self.tracks = []
        self.next_id = 0


class KalmanTrack:
    """
    Individual track with Kalman filter state estimation.
    """

    def __init__(self, track_id: int, detection: Dict, timestamp: float,
                 dt: float, process_noise: float, measurement_noise: float):
        """
        Initialize track.

        Args:
            track_id: Unique track identifier
            detection: Initial detection
            timestamp: Initial timestamp
            dt: Time step
            process_noise: Process noise variance
            measurement_noise: Measurement noise variance
        """
        self.track_id = track_id
        self.timestamp = timestamp
        self.age = 1
        self.hits = 1
        self.misses = 0
        self.max_misses = 3

        # Extract initial measurement
        bbox = np.array(detection['bbox'])
        self.bbox = bbox
        measurement = np.array([
            (bbox[0] + bbox[2]) / 2,  # center x
            (bbox[1] + bbox[3]) / 2   # center y
        ])

        # Initialize Kalman filter
        self.kf = SatelliteKalmanFilter(dt, process_noise, measurement_noise)
        self.kf.initialize(measurement)

        # Track history
        self.position_history = deque(maxlen=100)
        self.position_history.append(measurement.copy())

        # Additional tracking info
        self.velocity_history = deque(maxlen=100)
        self.confidence = detection.get('confidence', 1.0)
        self.object_class = detection.get('class', 0)

    def predict(self):
        """Predict next state."""
        self.kf.predict()
        self.age += 1

    def update(self, detection: Dict, timestamp: float):
        """Update track with new detection."""
        bbox = np.array(detection['bbox'])
        measurement = np.array([
            (bbox[0] + bbox[2]) / 2,
            (bbox[1] + bbox[3]) / 2
        ])

        # Update Kalman filter
        self.kf.update(measurement)

        # Update track info
        self.bbox = bbox
        self.timestamp = timestamp
        self.hits += 1
        self.misses = 0
        self.confidence = detection.get('confidence', self.confidence)
        self.object_class = detection.get('class', self.object_class)

        # Add to history
        position = self.kf.get_position()
        velocity = self.kf.get_velocity()
        self.position_history.append(position.copy())
        self.velocity_history.append(velocity.copy())

    def missed(self):
        """Handle missed detection."""
        self.misses += 1
        # Keep predicting without measurement update

    def get_position(self) -> np.ndarray:
        """Get current position estimate."""
        return self.kf.get_position()

    def get_velocity(self) -> np.ndarray:
        """Get current velocity estimate."""
        return self.kf.get_velocity()

    def get_state(self) -> Dict:
        """Get current track state."""
        position = self.get_position()
        velocity = self.get_velocity()

        # Estimate bbox from position (simplified)
        bbox_center = position
        bbox_size = np.array([20, 20])  # Default size
        if len(self.position_history) > 1:
            # Use average size from history
            sizes = []
            for i in range(1, len(self.position_history)):
                prev_pos = self.position_history[i-1]
                curr_pos = self.position_history[i]
                size = np.abs(curr_pos - prev_pos) * 2
                sizes.append(size)
            if sizes:
                bbox_size = np.mean(sizes, axis=0)

        bbox = np.array([
            bbox_center[0] - bbox_size[0]/2,
            bbox_center[1] - bbox_size[1]/2,
            bbox_center[0] + bbox_size[0]/2,
            bbox_center[1] + bbox_size[1]/2
        ])

        return {
            'id': self.track_id,
            'bbox': bbox,
            'position': position,
            'velocity': velocity,
            'age': self.age,
            'hits': self.hits,
            'misses': self.misses,
            'confidence': self.confidence,
            'class': self.object_class,
            'timestamp': self.timestamp
        }

    def is_confirmed(self) -> bool:
        """Check if track is confirmed (enough hits)."""
        return self.hits >= 3

    def is_dead(self) -> bool:
        """Check if track should be deleted."""
        return self.misses >= self.max_misses or self.age > 100


class SatelliteKalmanFilter:
    """
    Kalman filter for satellite tracking with orbital dynamics.
    """

    def __init__(self, dt: float, process_noise: float, measurement_noise: float):
        """
        Initialize Kalman filter.

        Args:
            dt: Time step
            process_noise: Process noise variance
            measurement_noise: Measurement noise variance
        """
        # State: [x, y, vx, vy] (position and velocity)
        self.dim_x = 4
        self.dim_z = 2  # measurement: [x, y]

        # State transition matrix F
        self.F = np.eye(self.dim_x)
        self.F[0, 2] = dt  # x += vx * dt
        self.F[1, 3] = dt  # y += vy * dt

        # Measurement matrix H
        self.H = np.zeros((self.dim_z, self.dim_x))
        self.H[0, 0] = 1  # measure x
        self.H[1, 1] = 1  # measure y

        # Process noise covariance Q
        self.Q = np.eye(self.dim_x) * process_noise
        # Add more noise to velocity components
        self.Q[2, 2] = process_noise * 10
        self.Q[3, 3] = process_noise * 10

        # Measurement noise covariance R
        self.R = np.eye(self.dim_z) * measurement_noise

        # Initial state covariance P
        self.P = np.eye(self.dim_x) * 100

        # State vector
        self.x = np.zeros(self.dim_x)

        self.initialized = False

    def initialize(self, measurement: np.ndarray):
        """Initialize filter with first measurement."""
        self.x[:2] = measurement  # position
        self.x[2:] = 0  # initial velocity estimate
        self.initialized = True

    def predict(self):
        """Predict next state."""
        if not self.initialized:
            return

        # State prediction
        self.x = self.F @ self.x

        # Covariance prediction
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, measurement: np.ndarray):
        """Update state with measurement."""
        if not self.initialized:
            self.initialize(measurement)
            return

        # Innovation
        y = measurement - self.H @ self.x

        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # State update
        self.x = self.x + K @ y

        # Covariance update
        I = np.eye(self.dim_x)
        self.P = (I - K @ self.H) @ self.P

    def get_position(self) -> np.ndarray:
        """Get current position estimate."""
        return self.x[:2].copy()

    def get_velocity(self) -> np.ndarray:
        """Get current velocity estimate."""
        return self.x[2:].copy()

    def get_state_covariance(self) -> np.ndarray:
        """Get state covariance matrix."""
        return self.P.copy()


class OrbitalKalmanFilter(SatelliteKalmanFilter):
    """
    Extended Kalman filter with orbital mechanics.
    """

    def __init__(self, dt: float, process_noise: float, measurement_noise: float,
                 mu: float = 3.986004418e14):  # Earth's gravitational parameter
        """
        Initialize orbital Kalman filter.

        Args:
            dt: Time step
            process_noise: Process noise variance
            measurement_noise: Measurement noise variance
            mu: Gravitational parameter
        """
        super().__init__(dt, process_noise, measurement_noise)
        self.mu = mu

        # Extended state for orbital elements
        self.dim_x = 6  # [x, y, z, vx, vy, vz]
        self.dim_z = 3  # 3D position measurement

        # Reinitialize matrices for 3D
        self._initialize_matrices_3d()

    def _initialize_matrices_3d(self):
        """Initialize matrices for 3D orbital tracking."""
        # State transition (simplified - would need proper orbital propagation)
        self.F = np.eye(self.dim_x)
        for i in range(3):
            self.F[i, i+3] = self.dt

        # Measurement matrix (3D position)
        self.H = np.zeros((self.dim_z, self.dim_x))
        self.H[0, 0] = 1  # x
        self.H[1, 1] = 1  # y
        self.H[2, 2] = 1  # z

        # Process noise (higher for orbital dynamics)
        self.Q = np.eye(self.dim_x) * self.process_noise * 100

        # Measurement noise
        self.R = np.eye(self.dim_z) * self.measurement_noise

        # State covariance
        self.P = np.eye(self.dim_x) * 1000

        # State vector
        self.x = np.zeros(self.dim_x)

    def predict(self):
        """Predict with orbital dynamics."""
        if not self.initialized:
            return

        # Simple orbital prediction (would use numerical integration in practice)
        r = np.linalg.norm(self.x[:3])  # distance from Earth center
        if r > 0:
            # Gravitational acceleration
            accel = -self.mu / r**3 * self.x[:3]
            self.x[3:6] += accel * self.dt  # velocity update
            self.x[:3] += self.x[3:6] * self.dt  # position update

        # Covariance prediction
        self.P = self.F @ self.P @ self.F.T + self.Q


if __name__ == "__main__":
    # Example usage
    tracker = KalmanTracker()

    # Create dummy detections
    detections = [
        {'bbox': [100, 100, 150, 150], 'confidence': 0.9, 'class': 0},
        {'bbox': [200, 200, 250, 250], 'confidence': 0.8, 'class': 1}
    ]

    # Update tracker
    tracks = tracker.update(detections)

    print(f"Active tracks: {len(tracks)}")
    for track in tracks:
        print(f"Track {track['id']}: position={track['position']:.1f}, velocity={track['velocity']}")

    # Update with new detections
    new_detections = [
        {'bbox': [105, 105, 155, 155], 'confidence': 0.85, 'class': 0},
        {'bbox': [300, 300, 350, 350], 'confidence': 0.7, 'class': 1}
    ]

    tracks = tracker.update(new_detections)
    print(f"Updated tracks: {len(tracks)}")
    for track in tracks:
        print(f"Track {track['id']}: position={track['position']:.1f}, velocity={track['velocity']}")