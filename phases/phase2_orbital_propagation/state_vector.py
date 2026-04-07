"""
State vector abstraction for orbital propagation.
"""

from dataclasses import dataclass
import numpy as np
from datetime import datetime


@dataclass
class StateVector:
    position: np.ndarray
    velocity: np.ndarray
    epoch: datetime

    def as_array(self) -> np.ndarray:
        return np.concatenate([self.position, self.velocity])

    def distance(self) -> float:
        return np.linalg.norm(self.position)

    def speed(self) -> float:
        return np.linalg.norm(self.velocity)
