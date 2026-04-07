"""
Trajectory simulation utilities for orbital motion.
"""

import numpy as np
from datetime import datetime, timedelta
from .state_vector import StateVector


class TrajectorySimulator:
    """Generate synthetic orbit trajectories for satellite analysis."""

    def __init__(self, mu: float = 398600.4418):
        self.mu = mu

    def simulate_circular_orbit(self, semi_major_axis_km: float, inclination_deg: float, duration_seconds: int, dt_seconds: int = 60):
        steps = max(1, duration_seconds // dt_seconds)
        inclination = np.deg2rad(inclination_deg)
        radius = semi_major_axis_km
        omega = np.sqrt(self.mu / radius**3)
        trajectory = []
        start = datetime.utcnow()
        for step in range(steps):
            theta = omega * (step * dt_seconds)
            x = radius * np.cos(theta)
            y = radius * np.sin(theta) * np.cos(inclination)
            z = radius * np.sin(theta) * np.sin(inclination)
            vx = -radius * omega * np.sin(theta)
            vy = radius * omega * np.cos(theta) * np.cos(inclination)
            vz = radius * omega * np.cos(theta) * np.sin(inclination)
            trajectory.append(StateVector(
                position=np.array([x, y, z], dtype=np.float32),
                velocity=np.array([vx, vy, vz], dtype=np.float32),
                epoch=start + timedelta(seconds=step * dt_seconds)
            ))
        return trajectory

    def sample_trajectory(self, state: StateVector, steps: int = 10, dt_seconds: int = 60):
        return self.simulate_circular_orbit(np.linalg.norm(state.position), np.rad2deg(np.arccos(state.position[2] / np.linalg.norm(state.position))), steps*dt_seconds, dt_seconds)
