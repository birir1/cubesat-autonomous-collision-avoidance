import numpy as np
from scipy.linalg import inv, det
from scipy.stats import multivariate_normal

class PhysicsCollisionProbability:
    """
    Gaussian collision probability approximation (2D encounter plane)
    Inspired by NASA CARA / CDM methods
    """

    def __init__(self, hard_body_radius=10.0):
        self.hbr = hard_body_radius  # meters

    def compute_probability(self, rel_pos, cov_matrix):
        """
        rel_pos: (3,) relative position vector
        cov_matrix: (3,3) combined covariance matrix
        """

        # Project to encounter plane (x, y)
        pos_2d = rel_pos[:2]
        cov_2d = cov_matrix[:2, :2]

        try:
            inv_cov = inv(cov_2d)
            det_cov = det(cov_2d)

            # Probability density at origin
            exponent = -0.5 * pos_2d.T @ inv_cov @ pos_2d
            norm_factor = 1 / (2 * np.pi * np.sqrt(det_cov))

            pc = norm_factor * np.exp(exponent) * np.pi * self.hbr**2
            return float(np.clip(pc, 0, 1))

        except Exception:
            return 0.0


class MissDistanceBaseline:
    """
    Simple closest approach distance baseline
    """

    def compute_distance(self, rel_pos):
        return np.linalg.norm(rel_pos)

    def risk_score(self, rel_pos, threshold=1000):
        """
        Convert distance to risk score
        """
        d = self.compute_distance(rel_pos)
        return float(np.exp(-d / threshold))


class MahalanobisRisk:
    """
    Uncertainty-aware distance metric
    """

    def compute_distance(self, rel_pos, cov_matrix):
        inv_cov = inv(cov_matrix)
        return float(np.sqrt(rel_pos.T @ inv_cov @ rel_pos))

    def risk_score(self, rel_pos, cov_matrix):
        d = self.compute_distance(rel_pos, cov_matrix)
        return float(np.exp(-d))