"""
Real Collision Risk Dataset Builder

Generates realistic collision risk labels using:
- Relative distance
- Relative velocity
- Continuous risk scoring (NOT binary)

This is critical for training meaningful ML models.
"""

import numpy as np
from tqdm import tqdm


def compute_relative_features(state1, state2):
    """
    Compute relative position & velocity

    Args:
        state1, state2: [x, y, z, vx, vy, vz]

    Returns:
        distance, relative_speed
    """

    pos1, vel1 = state1[:3], state1[3:]
    pos2, vel2 = state2[:3], state2[3:]

    distance = np.linalg.norm(pos1 - pos2)
    rel_velocity = np.linalg.norm(vel1 - vel2)

    return distance, rel_velocity


def compute_collision_risk(distance, rel_velocity):
    """
    Continuous collision risk function

    Key idea:
    - Closer distance → higher risk
    - Higher relative velocity → higher risk

    Returns:
        float in [0, 1]
    """

    # Tunable parameters (important for realism)
    d_scale = 50.0      # km (close encounter scale)
    v_scale = 5.0       # km/s (typical LEO rel speed)

    distance_term = np.exp(-distance / d_scale)
    velocity_term = np.tanh(rel_velocity / v_scale)

    risk = distance_term * velocity_term

    return float(risk)


def build_real_dataset(satellites, max_pairs=10000):
    """
    Build dataset from real satellite constellation

    Args:
        satellites: list of satellites
        max_pairs: limit for computation

    Returns:
        X: feature matrix (N, 6)
        y: risk scores (N,)
    """

    print("Building real collision dataset...")

    from data.features.orbital_features import tle_to_state_vector

    states = []
    valid_sats = []

    # ---------------------------
    # Step 1: Convert satellites → state vectors
    # ---------------------------
    for sat in satellites:
        state = tle_to_state_vector(sat)

        if state is not None:
            states.append(state)
            valid_sats.append(sat)

    states = np.array(states)

    print(f"Valid satellites: {len(states)}")

    # ---------------------------
    # Step 2: Generate pairs (SAFE VERSION)
    # ---------------------------
    X = []
    y = []

    count = 0

    for i in tqdm(range(len(states))):
        for j in range(i + 1, len(states)):

            if count >= max_pairs:
                break

            s1 = states[i]
            s2 = states[j]

            # Skip invalid states
            if not np.isfinite(s1).all() or not np.isfinite(s2).all():
                continue

            distance, rel_velocity = compute_relative_features(s1, s2)

            # Skip invalid physics
            if not np.isfinite(distance) or not np.isfinite(rel_velocity):
                continue

            # Skip zero or extreme values
            if distance <= 0 or distance > 1e6:
                continue

            if rel_velocity < 0 or rel_velocity > 20:
                continue

            # Feature
            feature = s1 - s2

            # Final safety check
            if not np.isfinite(feature).all():
                continue

            risk = compute_collision_risk(distance, rel_velocity)

            # Skip bad risk values
            if not np.isfinite(risk):
                continue

            X.append(feature)
            y.append(risk)

            count += 1

        if count >= max_pairs:
            break

    X = np.array(X)
    y = np.array(y)

    # FINAL CLEANING (IMPORTANT)
    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)

    X = X[mask]
    y = y[mask]

    print(f"Built {len(X)} samples after cleaning")

    if len(y) > 0:
        print(f"Risk stats → min: {y.min():.6f}, max: {y.max():.6f}, mean: {y.mean():.6f}")
    else:
        print("WARNING: No valid samples!")

    return X, y