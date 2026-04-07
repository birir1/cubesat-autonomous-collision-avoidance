import numpy as np
from scipy.stats import multivariate_normal
from scipy.linalg import sqrtm


class GaussianPcModel:
    """
    Gaussian Probability of Collision (Pc)

    Supports:
    - Analytical approximation (fast)
    - Monte Carlo integration (robust fallback)

    Based on encounter-plane Gaussian assumption (Alfano-style simplification).
    """

    def __init__(self, hard_body_radius=10.0, method="auto", mc_samples=10000):
        """
        Args:
            hard_body_radius (float): combined object radius (meters)
            method (str): "analytic", "mc", or "auto"
            mc_samples (int): samples for Monte Carlo
        """
        self.radius = hard_body_radius
        self.method = method
        self.mc_samples = mc_samples

    # -------------------------------------------------
    # Public API
    # -------------------------------------------------
    def compute_pc(self, rel_pos, cov):
        """
        rel_pos: (3,)
        cov: (3x3)

        Returns:
            Pc (float)
        """
        # Project into encounter plane (XY approximation)
        r = rel_pos[:2]
        C = cov[:2, :2]

        # Ensure numerical stability
        C = self._regularize_covariance(C)

        if self.method == "analytic":
            return self._analytic_pc(r, C)

        elif self.method == "mc":
            return self._monte_carlo_pc(r, C)

        else:  # auto
            try:
                return self._analytic_pc(r, C)
            except Exception:
                return self._monte_carlo_pc(r, C)

    # -------------------------------------------------
    # Analytical Approximation
    # -------------------------------------------------
    def _analytic_pc(self, r, C):
        """
        Fast Gaussian Pc approximation

        Pc ≈ exp(-0.5 * d^2) * (R^2 / (2π√|C|))
        where d is Mahalanobis distance
        """
        inv_C = np.linalg.inv(C)
        det_C = np.linalg.det(C)

        # Mahalanobis distance
        d2 = r.T @ inv_C @ r

        norm_factor = 1.0 / (2 * np.pi * np.sqrt(det_C))
        pc = norm_factor * np.exp(-0.5 * d2) * (np.pi * self.radius**2)

        return float(np.clip(pc, 0.0, 1.0))

    # -------------------------------------------------
    # Monte Carlo Method (robust)
    # -------------------------------------------------
    def _monte_carlo_pc(self, r, C):
        """
        Monte Carlo integration over Gaussian
        """
        rv = multivariate_normal(mean=r, cov=C)

        samples = rv.rvs(size=self.mc_samples)

        distances = np.linalg.norm(samples, axis=1)
        collisions = distances <= self.radius

        return float(np.mean(collisions))

    # -------------------------------------------------
    # Utilities
    # -------------------------------------------------
    def _regularize_covariance(self, C):
        """
        Ensure covariance is positive definite
        """
        eps = 1e-6
        return C + np.eye(C.shape[0]) * eps