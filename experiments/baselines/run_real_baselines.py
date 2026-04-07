import numpy as np
from models.physics.baseline_models import (
    PhysicsCollisionProbability,
    MissDistanceBaseline,
    MahalanobisRisk
)
from core.features import compute_relative_state, compute_covariance


def generate_dummy_states(n=1000):
    """
    Replace this with real dataset loader
    """
    states1 = np.random.randn(n, 6) * 1000
    states2 = np.random.randn(n, 6) * 1000
    return states1, states2


def run_baselines():
    states1, states2 = generate_dummy_states()

    pc_model = PhysicsCollisionProbability()
    md_model = MissDistanceBaseline()
    maha_model = MahalanobisRisk()

    pc_scores = []
    md_scores = []
    maha_scores = []

    for s1, s2 in zip(states1, states2):
        rel_pos, rel_vel = compute_relative_state(s1, s2)
        cov = compute_covariance(rel_pos, rel_vel)

        pc_scores.append(pc_model.compute_probability(rel_pos, cov))
        md_scores.append(md_model.risk_score(rel_pos))
        maha_scores.append(maha_model.risk_score(rel_pos, cov))

    print("=== BASELINE RESULTS ===")
    print(f"PC Mean:   {np.mean(pc_scores):.4f}")
    print(f"Miss Dist: {np.mean(md_scores):.4f}")
    print(f"Mahalanobis: {np.mean(maha_scores):.4f}")


if __name__ == "__main__":
    run_baselines()