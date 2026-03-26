# orbital_mechanics.py
"""
Orbital Mechanics Utilities

Provides fundamental orbital mechanics calculations used across the
CubeSat Autonomous Collision Avoidance framework.

Includes:
- Cartesian ↔ Keplerian conversions
- Orbital velocity computation
- Distance calculations
- Closest approach estimation
- Orbital energy calculations
"""

import numpy as np
from numpy.linalg import norm
from typing import Tuple

# Earth's gravitational parameter (km^3/s^2)
MU_EARTH = 398600.4418

# Earth's radius (km)
EARTH_RADIUS = 6378.137


# ------------------------------------------------------------
# Basic Vector Utilities
# ------------------------------------------------------------

def unit_vector(v: np.ndarray) -> np.ndarray:
    """Return the unit vector of a vector."""
    return v / norm(v)


def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute angle between two vectors in radians."""
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)

    dot_product = np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)
    return np.arccos(dot_product)


# ------------------------------------------------------------
# Orbital Energy
# ------------------------------------------------------------

def specific_orbital_energy(position: np.ndarray, velocity: np.ndarray) -> float:
    """
    Compute specific orbital energy.

    ε = v^2/2 - μ/r
    """

    r = norm(position)
    v = norm(velocity)

    return (v ** 2) / 2 - MU_EARTH / r


# ------------------------------------------------------------
# Cartesian -> Keplerian Conversion
# ------------------------------------------------------------

def cartesian_to_keplerian(
    position: np.ndarray,
    velocity: np.ndarray,
    mu: float = MU_EARTH
) -> dict:
    """
    Convert Cartesian state vectors to Keplerian orbital elements.

    Returns:
        dict containing:
        - semi_major_axis
        - eccentricity
        - inclination
        - raan
        - argument_of_perigee
        - true_anomaly
    """

    r = position
    v = velocity

    r_norm = norm(r)
    v_norm = norm(v)

    # Specific angular momentum
    h = np.cross(r, v)
    h_norm = norm(h)

    # Inclination
    inclination = np.arccos(h[2] / h_norm)

    # Node vector
    K = np.array([0, 0, 1])
    n = np.cross(K, h)
    n_norm = norm(n)

    # Eccentricity vector
    e_vec = (1 / mu) * (
        (v_norm ** 2 - mu / r_norm) * r - np.dot(r, v) * v
    )

    eccentricity = norm(e_vec)

    # Semi-major axis
    energy = specific_orbital_energy(r, v)
    semi_major_axis = -mu / (2 * energy)

    # RAAN
    if n_norm != 0:
        raan = np.arccos(n[0] / n_norm)
        if n[1] < 0:
            raan = 2 * np.pi - raan
    else:
        raan = 0

    # Argument of Perigee
    if n_norm != 0 and eccentricity > 1e-8:
        arg_perigee = np.arccos(np.dot(n, e_vec) / (n_norm * eccentricity))
        if e_vec[2] < 0:
            arg_perigee = 2 * np.pi - arg_perigee
    else:
        arg_perigee = 0

    # True Anomaly
    if eccentricity > 1e-8:
        true_anomaly = np.arccos(np.dot(e_vec, r) / (eccentricity * r_norm))
        if np.dot(r, v) < 0:
            true_anomaly = 2 * np.pi - true_anomaly
    else:
        true_anomaly = 0

    return {
        "semi_major_axis": semi_major_axis,
        "eccentricity": eccentricity,
        "inclination": inclination,
        "raan": raan,
        "argument_of_perigee": arg_perigee,
        "true_anomaly": true_anomaly
    }


# ------------------------------------------------------------
# Keplerian -> Cartesian
# ------------------------------------------------------------

def keplerian_to_cartesian(
    a: float,
    e: float,
    i: float,
    raan: float,
    arg_perigee: float,
    true_anomaly: float,
    mu: float = MU_EARTH
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert Keplerian elements to Cartesian position and velocity.
    """

    p = a * (1 - e ** 2)

    r_orb = (p / (1 + e * np.cos(true_anomaly))) * np.array([
        np.cos(true_anomaly),
        np.sin(true_anomaly),
        0
    ])

    v_orb = np.sqrt(mu / p) * np.array([
        -np.sin(true_anomaly),
        e + np.cos(true_anomaly),
        0
    ])

    # Rotation matrices
    R3_W = np.array([
        [np.cos(raan), -np.sin(raan), 0],
        [np.sin(raan),  np.cos(raan), 0],
        [0, 0, 1]
    ])

    R1_i = np.array([
        [1, 0, 0],
        [0, np.cos(i), -np.sin(i)],
        [0, np.sin(i),  np.cos(i)]
    ])

    R3_w = np.array([
        [np.cos(arg_perigee), -np.sin(arg_perigee), 0],
        [np.sin(arg_perigee),  np.cos(arg_perigee), 0],
        [0, 0, 1]
    ])

    rotation_matrix = R3_W @ R1_i @ R3_w

    r = rotation_matrix @ r_orb
    v = rotation_matrix @ v_orb

    return r, v


# ------------------------------------------------------------
# Distance Between Objects
# ------------------------------------------------------------

def relative_distance(
    pos1: np.ndarray,
    pos2: np.ndarray
) -> float:
    """Compute Euclidean distance between two objects."""
    return norm(pos1 - pos2)


# ------------------------------------------------------------
# Relative Velocity
# ------------------------------------------------------------

def relative_velocity(
    vel1: np.ndarray,
    vel2: np.ndarray
) -> float:
    """Compute relative velocity magnitude."""
    return norm(vel1 - vel2)


# ------------------------------------------------------------
# Closest Approach Computation
# ------------------------------------------------------------

def closest_approach(
    r1: np.ndarray,
    v1: np.ndarray,
    r2: np.ndarray,
    v2: np.ndarray
) -> Tuple[float, float]:
    """
    Estimate closest approach distance between two objects
    assuming linear relative motion.

    Returns:
        (time_of_closest_approach, minimum_distance)
    """

    r_rel = r1 - r2
    v_rel = v1 - v2

    t_ca = -np.dot(r_rel, v_rel) / np.dot(v_rel, v_rel)

    closest_position = r_rel + v_rel * t_ca
    min_distance = norm(closest_position)

    return t_ca, min_distance


# ------------------------------------------------------------
# Escape Velocity
# ------------------------------------------------------------

def escape_velocity(radius: float, mu: float = MU_EARTH) -> float:
    """Compute escape velocity at a given radius."""
    return np.sqrt(2 * mu / radius)


# ------------------------------------------------------------
# Circular Orbit Velocity
# ------------------------------------------------------------

def circular_orbit_velocity(radius: float, mu: float = MU_EARTH) -> float:
    """Velocity required for circular orbit."""
    return np.sqrt(mu / radius)