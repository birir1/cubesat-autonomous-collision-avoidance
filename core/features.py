import numpy as np
from .utils import relative_position, relative_velocity, safe_norm
from typing import Tuple, Union

def compute_relative_state(
    state1: Union[np.ndarray, list, tuple],
    state2: Union[np.ndarray, list, tuple],
    return_full: bool = False
) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
    """
    Compute relative state between two satellites.

    Parameters
    ----------
    state1 : array-like (6,)
        [x, y, z, vx, vy, vz] of object 1 (meters, m/s)
    state2 : array-like (6,)
        [x, y, z, vx, vy, vz] of object 2 (meters, m/s)
    return_full : bool
        If False → returns (rel_r, rel_v)
        If True  → returns extended feature vector

    Returns
    -------
    If return_full=False:
        rel_r : np.ndarray (3,)
        rel_v : np.ndarray (3,)
    If return_full=True:
        features : np.ndarray (8,)
            [rel_rx, rel_ry, rel_rz,
             rel_vx, rel_vy, rel_vz,
             distance, relative_speed]
    """

    # Ensure numpy arrays
    state1 = np.asarray(state1, dtype=np.float64).flatten()
    state2 = np.asarray(state2, dtype=np.float64).flatten()

    if state1.shape[0] != 6 or state2.shape[0] != 6:
        raise ValueError("State vectors must be 6-dimensional (position and velocity)")

    # Split position and velocity
    r1, v1 = state1[:3], state1[3:]
    r2, v2 = state2[:3], state2[3:]

    # Relative quantities
    rel_r = relative_position(r1, r2)
    rel_v = relative_velocity(v1, v2)

    if not return_full:
        return rel_r, rel_v

    # --- Extended physically meaningful features ---
    distance = safe_norm(rel_r)
    relative_speed = safe_norm(rel_v)

    radial_velocity = 0.0
    if distance > 1e-8:
        radial_velocity = np.dot(rel_r, rel_v) / distance

    features = np.array([
        rel_r[0], rel_r[1], rel_r[2],
        rel_v[0], rel_v[1], rel_v[2],
        distance,
        relative_speed
    ], dtype=np.float64)

    return features


def compute_time_to_closest_approach(rel_r: np.ndarray, rel_v: np.ndarray) -> float:
    """
    Compute Time of Closest Approach (TCA)

    Returns
    -------
    tca : float (seconds)
    """
    rel_r = np.asarray(rel_r, dtype=np.float64)
    rel_v = np.asarray(rel_v, dtype=np.float64)

    v_norm_sq = np.dot(rel_v, rel_v)
    if v_norm_sq < 1e-12:
        return 0.0

    tca = -np.dot(rel_r, rel_v) / v_norm_sq
    return float(max(tca, 0.0))


def compute_miss_distance(rel_r: np.ndarray, rel_v: np.ndarray) -> float:
    """
    Compute miss distance at closest approach

    Returns
    -------
    miss_distance : float (meters)
    """
    tca = compute_time_to_closest_approach(rel_r, rel_v)
    closest_point = rel_r + tca * rel_v
    return float(np.linalg.norm(closest_point))


def compute_covariance(std_pos: float = 100.0, std_vel: float = 1.0) -> np.ndarray:
    """
    Realistic diagonal covariance model.

    Parameters
    ----------
    std_pos : float
        Position uncertainty (meters)
    std_vel : float
        Velocity uncertainty (m/s)

    Returns
    -------
    cov : np.ndarray (6x6)
    """
    pos_cov = np.eye(3) * (std_pos ** 2)
    vel_cov = np.eye(3) * (std_vel ** 2)

    cov = np.block([
        [pos_cov, np.zeros((3, 3))],
        [np.zeros((3, 3)), vel_cov]
    ])

    return cov