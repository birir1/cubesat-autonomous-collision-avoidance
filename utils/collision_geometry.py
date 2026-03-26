# collision_geometry.py
"""
Collision Geometry Utilities

Provides analytical tools for computing conjunction risk between
space objects using relative motion and probabilistic models.

Used by:
- Collision Risk Estimation (GNN training)
- RL Maneuver Planning
- Orbital Simulation
"""

import numpy as np
from numpy.linalg import norm
from scipy.stats import multivariate_normal

from utils.orbital_mechanics import relative_distance, relative_velocity
from utils.coordinate_transforms import eci_to_lvlh


# ---------------------------------------------------------
# Relative Motion
# ---------------------------------------------------------

def relative_state(r1, v1, r2, v2):
    """
    Compute relative state vectors between two objects.

    Returns:
        relative_position
        relative_velocity
    """

    r_rel = r2 - r1
    v_rel = v2 - v1

    return r_rel, v_rel


# ---------------------------------------------------------
# Time of Closest Approach (TCA)
# ---------------------------------------------------------

def time_of_closest_approach(r1, v1, r2, v2):
    """
    Compute time of closest approach assuming linear motion.
    """

    r_rel, v_rel = relative_state(r1, v1, r2, v2)

    denom = np.dot(v_rel, v_rel)

    if denom == 0:
        return 0.0

    tca = - np.dot(r_rel, v_rel) / denom

    return tca


# ---------------------------------------------------------
# Miss Distance
# ---------------------------------------------------------

def miss_distance(r1, v1, r2, v2):
    """
    Compute miss distance at closest approach.
    """

    tca = time_of_closest_approach(r1, v1, r2, v2)

    r1_ca = r1 + v1 * tca
    r2_ca = r2 + v2 * tca

    return norm(r1_ca - r2_ca), tca


# ---------------------------------------------------------
# Conjunction Detection
# ---------------------------------------------------------

def detect_conjunction(
    r1,
    v1,
    r2,
    v2,
    threshold_km=5.0
):
    """
    Detect potential conjunction between two objects.

    threshold_km: miss distance threshold for warning
    """

    miss_dist, tca = miss_distance(r1, v1, r2, v2)

    conjunction = miss_dist < threshold_km

    return {
        "conjunction": conjunction,
        "miss_distance": miss_dist,
        "tca": tca
    }


# ---------------------------------------------------------
# Relative Motion in LVLH Frame
# ---------------------------------------------------------

def relative_motion_lvlh(r_sat, v_sat, r_obj, v_obj):
    """
    Express relative motion in LVLH frame.

    Useful for collision analysis.
    """

    r_rel = r_obj - r_sat
    v_rel = v_obj - v_sat

    T = eci_to_lvlh(r_sat, v_sat)

    r_lvlh = T.T @ r_rel
    v_lvlh = T.T @ v_rel

    return r_lvlh, v_lvlh


# ---------------------------------------------------------
# Collision Probability (Gaussian Model)
# ---------------------------------------------------------

def collision_probability(
    r1,
    v1,
    r2,
    v2,
    cov1,
    cov2,
    hard_body_radius=0.01
):
    """
    Estimate probability of collision using Gaussian uncertainty.

    r1,v1 : satellite state
    r2,v2 : debris state

    cov1,cov2 : covariance matrices

    hard_body_radius : combined object radius (km)
    """

    r_rel = r2 - r1

    combined_cov = cov1 + cov2

    # Probability density of relative position
    rv = multivariate_normal(mean=[0,0,0], cov=combined_cov)

    # Collision sphere
    prob = rv.pdf(r_rel) * (4/3) * np.pi * hard_body_radius**3

    return prob


# ---------------------------------------------------------
# Conjunction Data Generator (for ML training)
# ---------------------------------------------------------

def generate_conjunction_features(
    r1,
    v1,
    r2,
    v2
):
    """
    Generate features used for training ML collision models.
    """

    r_rel, v_rel = relative_state(r1, v1, r2, v2)

    dist = norm(r_rel)
    rel_speed = norm(v_rel)

    miss_dist, tca = miss_distance(r1, v1, r2, v2)

    features = {
        "distance": dist,
        "relative_speed": rel_speed,
        "miss_distance": miss_dist,
        "time_to_closest_approach": tca
    }

    return features


# ---------------------------------------------------------
# Multi-object conjunction scan
# ---------------------------------------------------------

def scan_conjunctions(objects, threshold_km=5.0):
    """
    Scan multiple objects for conjunctions.

    objects = list of dictionaries:
    {
        "id": object_id,
        "position": np.array,
        "velocity": np.array
    }
    """

    conjunctions = []

    n = len(objects)

    for i in range(n):
        for j in range(i + 1, n):

            obj1 = objects[i]
            obj2 = objects[j]

            result = detect_conjunction(
                obj1["position"],
                obj1["velocity"],
                obj2["position"],
                obj2["velocity"],
                threshold_km
            )

            if result["conjunction"]:
                conjunctions.append({
                    "object_1": obj1["id"],
                    "object_2": obj2["id"],
                    "miss_distance": result["miss_distance"],
                    "tca": result["tca"]
                })

    return conjunctions