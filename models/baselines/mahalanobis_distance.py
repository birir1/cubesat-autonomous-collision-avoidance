import numpy as np
from scipy.linalg import cho_factor, cho_solve


class MahalanobisRisk:
    """
    Covariance-aware distance metric for conjunction assessment.

    This is a core statistical baseline used in:
    - Satellite conjunction screening
    - Track association
    - Collision risk estimation

    d^2 follows a Chi-square distribution.
    """

    def __init__(self, eps=1e-6):
        self.eps = eps

    # -------------------------------------------------
    # Core Distance (Stable)
    # -------------------------------------------------
    def compute_distance(self, rel_pos, cov):
        """
        Mahalanobis distance using Cholesky decomposition

        Args:
            rel_pos: (3,)
            cov: (3x3)

        Returns:
            distance (float)
        """
        cov = self._regularize_covariance(cov)

        try:
            # Cholesky-based solve (more stable than inverse)
            c, lower = cho_factor(cov)
            sol = cho_solve((c, lower), rel_pos)
            d2 = rel_pos.T @ sol
        except Exception:
            # Fallback to pseudo-inverse
            cov_inv = np.linalg.pinv(cov)
            d2 = rel_pos.T @ cov_inv @ rel_pos

        return float(np.sqrt(max(d2, 0.0)))

    # -------------------------------------------------
    # Risk Mapping
    # -------------------------------------------------
    def compute_risk(self, rel_pos, cov):
        """
        Convert Mahalanobis distance → probability-like risk

        Uses Gaussian assumption:
            risk = exp(-0.5 * d^2)
        """
        d = self.compute_distance(rel_pos, cov)
        risk = np.exp(-0.5 * d**2)

        return float(np.clip(risk, 0.0, 1.0))

    # -------------------------------------------------
    # Chi-square Interpretation (IMPORTANT)
    # -------------------------------------------------
    def compute_chi_square(self, rel_pos, cov):
        """
        Returns d^2 (chi-square statistic)

        Useful for statistical gating
        """
        cov = self._regularize_covariance(cov)

        try:
            c, lower = cho_factor(cov)
            sol = cho_solve((c, lower), rel_pos)
            d2 = rel_pos.T @ sol
        except Exception:
            cov_inv = np.linalg.pinv(cov)
            d2 = rel_pos.T @ cov_inv @ rel_pos

        return float(max(d2, 0.0))

    # -------------------------------------------------
    # Gating Decision (Operational)
    # -------------------------------------------------
    def is_high_risk(self, rel_pos, cov, threshold=9.21):
        """
        Chi-square gating (3 DOF default ≈ 99%)

        threshold ≈ 9.21 → 99% confidence region (3D)

        Returns:
            True if inside danger region
        """
        d2 = self.compute_chi_square(rel_pos, cov)
        return d2 <= threshold

    # -------------------------------------------------
    # Batch Processing
    # -------------------------------------------------
    def compute_batch(self, rel_positions, covariances):
        """
        Args:
            rel_positions: (N, 3)
            covariances: (N, 3, 3)

        Returns:
            risks: (N,)
        """
        risks = []

        for r, C in zip(rel_positions, covariances):
            risks.append(self.compute_risk(r, C))

        return np.array(risks)

    # -------------------------------------------------
    # Utilities
    # -------------------------------------------------
    def _regularize_covariance(self, cov):
        """
        Ensure covariance is positive definite
        """
        return cov + np.eye(cov.shape[0]) * self.eps