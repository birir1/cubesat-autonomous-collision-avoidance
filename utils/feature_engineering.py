"""
Feature engineering helpers for orbital and collision risk data.
"""

import numpy as np
from typing import Dict, List, Optional


def safe_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Normalize a tensor while avoiding division by zero."""
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / (norm + eps)


def compute_relative_state(positions: np.ndarray, velocities: np.ndarray) -> np.ndarray:
    """Compute relative state features for neighbor satellites."""
    relative_positions = positions[:, :, None, :] - positions[:, None, :, :]
    relative_velocities = velocities[:, :, None, :] - velocities[:, None, :, :]
    distances = np.linalg.norm(relative_positions, axis=-1)
    speed_deltas = np.linalg.norm(relative_velocities, axis=-1)
    return np.stack([distances, speed_deltas], axis=-1)


def compute_kinematic_features(positions: np.ndarray, velocities: np.ndarray) -> np.ndarray:
    """Compute canonical kinematic features from position and velocity vectors."""
    speed = np.linalg.norm(velocities, axis=-1)
    altitude = np.linalg.norm(positions, axis=-1) - 6371.0
    radial_velocity = np.sum(positions * velocities, axis=-1) / np.clip(np.linalg.norm(positions, axis=-1), 1e-8)
    return np.stack([speed, altitude, radial_velocity], axis=-1)


def compute_orbital_elements(positions: np.ndarray, velocities: np.ndarray) -> np.ndarray:
    """Compute simple orbital element proxies for feature extraction."""
    r = np.linalg.norm(positions, axis=-1)
    v = np.linalg.norm(velocities, axis=-1)
    specific_energy = 0.5 * v**2 - 398600.4418 / r
    angular_momentum = np.linalg.norm(np.cross(positions, velocities), axis=-1)
    return np.stack([r, v, specific_energy, angular_momentum], axis=-1)


def build_risk_features(sample: Dict[str, np.ndarray]) -> np.ndarray:
    """Build a compact feature vector for collision risk models."""
    positions = sample['positions']
    velocities = sample['velocities']
    kinematic = compute_kinematic_features(positions, velocities)
    orbital = compute_orbital_elements(positions, velocities)
    relative = compute_relative_state(positions, velocities)
    pairwise_stats = np.stack([
        np.min(relative[..., 0], axis=(-1, -2)),
        np.mean(relative[..., 0], axis=(-1, -2)),
        np.max(relative[..., 0], axis=(-1, -2)),
        np.min(relative[..., 1], axis=(-1, -2)),
        np.mean(relative[..., 1], axis=(-1, -2)),
        np.max(relative[..., 1], axis=(-1, -2))
    ], axis=-1)
    summary = np.concatenate([kinematic.reshape(kinematic.shape[0], -1), orbital.reshape(orbital.shape[0], -1), pairwise_stats], axis=-1)
    return summary
