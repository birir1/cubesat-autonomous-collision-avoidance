"""
Orbital Feature Extraction from TLE Data

Converts Skyfield EarthSatellite objects into ML-ready state vectors:
[x, y, z, vx, vy, vz]

Uses Skyfield (SGP4 internally) for real orbital mechanics.
"""

import numpy as np
from skyfield.api import load
from datetime import datetime

# Initialize timescale ONCE
ts = load.timescale()


def tle_to_state_vector(sat, timestamp=None):
    """
    Convert EarthSatellite → state vector [x, y, z, vx, vy, vz]
    """

    try:
        if timestamp is None:
            t = ts.now()
        else:
            t = ts.utc(
                timestamp.year,
                timestamp.month,
                timestamp.day,
                timestamp.hour,
                timestamp.minute,
                timestamp.second,
            )

        geocentric = sat.at(t)

        position = geocentric.position.km
        velocity = geocentric.velocity.km_per_s

        return np.concatenate([position, velocity])

    except Exception:
        return None


def extract_constellation_features(satellites, max_sats=100):
    """
    Extract features for a subset of satellites
    """

    features = []

    for sat in satellites[:max_sats]:
        state = tle_to_state_vector(sat)

        if state is not None:
            features.append(state)

    if len(features) == 0:
        return np.empty((0, 6))

    return np.array(features)


def build_feature_dataset(all_sats):
    """
    Build dataset from all constellations
    """

    dataset = {}

    for name, sats in all_sats.items():
        print(f"Processing {name} ({len(sats)} satellites)...")

        features = extract_constellation_features(sats)

        print(f"  → Extracted {features.shape[0]} valid state vectors")

        dataset[name] = features

    return dataset