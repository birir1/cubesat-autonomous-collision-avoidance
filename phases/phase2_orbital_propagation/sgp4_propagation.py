"""
Lightweight SGP4-style propagation support for phase 2.
"""

import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple


class SGP4Propagator:
    """Simplified wrapper for SGP4-style orbital propagation."""

    def propagate(self, tle_data: Dict[str, float], times: List[datetime]) -> List[Tuple[np.ndarray, np.ndarray]]:
        trajectory = []
        for time in times:
            r = float(tle_data.get('semi_major_axis', 6771.0))
            theta = float(tle_data.get('mean_anomaly', 0.0)) + 0.001 * len(trajectory)
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            z = 0.0
            pos = np.array([x, y, z], dtype=np.float32)
            vel = np.array([-np.sin(theta), np.cos(theta), 0.0], dtype=np.float32) * 7.5
            trajectory.append((pos, vel))
        return trajectory
