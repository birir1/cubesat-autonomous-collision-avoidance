"""
Simplified DeepSORT-inspired tracker for phase 3 object tracking.
"""

import numpy as np
from typing import Any, Dict, List


class DeepSORTTracker:
    """Lightweight tracking class using appearance-free association."""

    def __init__(self, max_distance: float = 50.0):
        self.max_distance = max_distance
        self.next_id = 1
        self.tracks: List[Dict[str, Any]] = []

    def _similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return np.linalg.norm(a - b)

    def track(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        for detection in detections:
            position = np.array([detection['x'], detection['y']])
            matched = False
            for track in self.tracks:
                prior = np.array(track['position'])
                if self._similarity(prior, position) < self.max_distance:
                    track['position'] = position.tolist()
                    track['history'].append(position.tolist())
                    matched = True
                    break
            if not matched:
                self.tracks.append({
                    'id': self.next_id,
                    'position': position.tolist(),
                    'history': [position.tolist()]
                })
                self.next_id += 1
        return self.tracks
