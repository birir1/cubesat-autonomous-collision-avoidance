"""
Kalman filter tracker for satellite object tracking.
"""

import numpy as np
from typing import List, Dict, Any


class KalmanTracker:
    """Simple Kalman filter-based object tracker."""

    def __init__(self, dt: float = 1.0):
        self.dt = dt
        self.tracks: List[Dict[str, Any]] = []
        self.next_id = 1

    def _predict_state(self, state: np.ndarray) -> np.ndarray:
        F = np.array([[1, 0, self.dt, 0],
                      [0, 1, 0, self.dt],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])
        return F @ state

    def _update_state(self, state: np.ndarray, measurement: np.ndarray) -> np.ndarray:
        return measurement

    def track(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        tracks = []
        for det in detections:
            measurement = np.array([det['x'], det['y'], det.get('vx', 0.0), det.get('vy', 0.0)])
            if not self.tracks:
                self.tracks.append({'id': self.next_id, 'state': measurement, 'history': [measurement.tolist()]})
                self.next_id += 1
            else:
                best = min(self.tracks, key=lambda t: np.linalg.norm(self._predict_state(t['state'])[:2] - measurement[:2]))
                best['state'] = self._update_state(self._predict_state(best['state']), measurement)
                best['history'].append(best['state'].tolist())
            tracks = self.tracks
        return tracks
