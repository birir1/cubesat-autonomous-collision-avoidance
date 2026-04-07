# experiments/baselines/run_physics_baselines.py

import numpy as np

from models.baselines.gaussian_pc import GaussianPcModel
from models.baselines.mahalanobis_distance import MahalanobisRisk

from core.features import (
    compute_time_to_closest_approach,
    compute_miss_distance
)
from core.utils import safe_norm


pc_model = GaussianPcModel()
mah_model = MahalanobisRisk()


def _validate_sample(sample):
    """
    Ensure required fields exist and are numerically stable
    """
    if "rel_pos" not in sample or "cov" not in sample:
        raise ValueError("Sample must contain 'rel_pos' and 'cov'")

    rel_pos = np.asarray(sample["rel_pos"], dtype=np.float64)
    cov = np.asarray(sample["cov"], dtype=np.float64)

    # Fix NaNs / Infs (common in real pipelines)
    rel_pos = np.nan_to_num(rel_pos, nan=0.0, posinf=1e6, neginf=-1e6)
    cov = np.nan_to_num(cov, nan=0.0, posinf=1e6, neginf=-1e6)

    # Ensure covariance shape
    if cov.shape[0] >= 3 and cov.shape[1] >= 3:
        cov = cov[:3, :3]
    else:
        raise ValueError("Covariance must be at least 3x3")

    return rel_pos, cov


def run_physics_baselines(samples, return_dict=True):
    """
    Run physically grounded baseline models:
    - Gaussian Probability of Collision (Pc)
    - Mahalanobis Risk

    Parameters
    ----------
    samples : list of dict
        Each sample must contain:
            - rel_pos: (3,)
            - rel_vel: (3,) [optional but recommended]
            - cov: (3x3 or 6x6)

    return_dict : bool
        If True → returns structured dict (recommended)
        If False → returns tuple (pcs, mahal_risks)

    Returns
    -------
    dict or tuple
    """

    pcs = []
    mahal_risks = []
    distances = []
    tcas = []
    miss_distances = []

    for i, sample in enumerate(samples):

        try:
            rel_pos, cov = _validate_sample(sample)

            # Optional velocity
            rel_vel = sample.get("rel_vel", None)
            if rel_vel is not None:
                rel_vel = np.asarray(rel_vel, dtype=np.float64)
                rel_vel = np.nan_to_num(rel_vel, nan=0.0, posinf=1e6, neginf=-1e6)

            # --- Core physics metrics ---
            distance = safe_norm(rel_pos)
            distances.append(distance)

            if rel_vel is not None:
                tca = compute_time_to_closest_approach(rel_pos, rel_vel)
                miss_dist = compute_miss_distance(rel_pos, rel_vel)
            else:
                tca = 0.0
                miss_dist = distance

            tcas.append(tca)
            miss_distances.append(miss_dist)

            # --- Baseline models ---
            pc = pc_model.compute_pc(rel_pos, cov)
            risk = mah_model.compute_risk(rel_pos, cov)

            pcs.append(pc)
            mahal_risks.append(risk)

        except Exception as e:
            # Robust to bad samples (important in real datasets)
            print(f"[WARNING] Skipping sample {i}: {e}")

            pcs.append(0.0)
            mahal_risks.append(0.0)
            distances.append(0.0)
            tcas.append(0.0)
            miss_distances.append(0.0)

    if return_dict:
        return {
            "pc": np.array(pcs),
            "mahalanobis_risk": np.array(mahal_risks),
            "distance": np.array(distances),
            "tca": np.array(tcas),
            "miss_distance": np.array(miss_distances),
        }

    return pcs, mahal_risks