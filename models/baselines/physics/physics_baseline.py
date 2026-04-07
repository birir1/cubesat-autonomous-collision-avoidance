"""
Physics-based baseline models for collision risk assessment.

Implements traditional methods like Pc (Probability of Collision)
and Mahalanobis distance for comparison with learning-based approaches.
"""

import numpy as np
from typing import Tuple, Optional
from scipy.stats import multivariate_normal
from core.utils import mahalanobis_distance, ensure_positive_definite


class PhysicsBaseline:
    """
    Physics-based collision risk assessment using traditional methods.

    Implements:
    - Pc (Probability of Collision) using Gaussian approximation
    - Mahalanobis distance for outlier detection
    - Minimum distance calculations
    """

    def __init__(self, collision_radius: float = 1.0):
        """
        Initialize physics baseline.

        Args:
            collision_radius: Collision radius in meters (default: 1.0)
        """
        self.collision_radius = collision_radius

    def compute_collision_probability(
        self,
        primary_state: np.ndarray,
        secondary_state: np.ndarray,
        primary_cov: Optional[np.ndarray] = None,
        secondary_cov: Optional[np.ndarray] = None,
        time_horizon: float = 3600.0
    ) -> float:
        """
        Compute collision probability using Gaussian approximation.

        Args:
            primary_state: Primary satellite state [x, y, z, vx, vy, vz]
            secondary_state: Secondary satellite state [x, y, z, vx, vy, vz]
            primary_cov: Primary satellite covariance matrix (6x6)
            secondary_cov: Secondary satellite covariance matrix (6x6)
            time_horizon: Time horizon for conjunction in seconds

        Returns:
            Collision probability (0.0 to 1.0)
        """
        # Split position and velocity
        r1, v1 = primary_state[:3], primary_state[3:]
        r2, v2 = secondary_state[:3], secondary_state[3:]

        # Relative state
        rel_r = r2 - r1
        rel_v = v2 - v1

        # Default covariances if not provided
        if primary_cov is None:
            primary_cov = self._default_covariance()
        if secondary_cov is None:
            secondary_cov = self._default_covariance()

        # Combined covariance for relative motion
        rel_cov = primary_cov + secondary_cov
        rel_cov = ensure_positive_definite(rel_cov)

        # Project to position space (simplified 2D projection)
        # In practice, this would use full 3D conjunction geometry
        pos_cov = rel_cov[:3, :3]

        # Compute minimum separation distance
        try:
            mahal_dist = mahalanobis_distance(rel_r, pos_cov)
            pc = 1.0 - multivariate_normal.cdf(self.collision_radius, cov=pos_cov)
            pc = min(pc, 1.0)  # Clamp to valid range
        except np.linalg.LinAlgError:
            # Fallback for singular matrices
            distance = np.linalg.norm(rel_r)
            pc = 1.0 if distance < self.collision_radius else 0.0

        return pc

    def compute_mahalanobis_distance(
        self,
        primary_state: np.ndarray,
        secondary_state: np.ndarray,
        combined_cov: Optional[np.ndarray] = None
    ) -> float:
        """
        Compute Mahalanobis distance between satellite states.

        Args:
            primary_state: Primary satellite state
            secondary_state: Secondary satellite state
            combined_cov: Combined covariance matrix

        Returns:
            Mahalanobis distance
        """
        if combined_cov is None:
            combined_cov = self._default_covariance()

        combined_cov = ensure_positive_definite(combined_cov)
        rel_state = secondary_state - primary_state

        try:
            mahal_dist = mahalanobis_distance(rel_state, combined_cov)
        except np.linalg.LinAlgError:
            # Fallback to Euclidean distance
            mahal_dist = np.linalg.norm(rel_state)

        return mahal_dist

    def compute_minimum_distance(
        self,
        primary_trajectory: np.ndarray,
        secondary_trajectory: np.ndarray
    ) -> Tuple[float, int]:
        """
        Compute minimum distance between two trajectories.

        Args:
            primary_trajectory: Primary satellite trajectory (T x 6)
            secondary_trajectory: Secondary satellite trajectory (T x 6)

        Returns:
            Tuple of (minimum_distance, time_step)
        """
        min_distance = float('inf')
        min_time = 0

        for t in range(min(len(primary_trajectory), len(secondary_trajectory))):
            r1 = primary_trajectory[t, :3]
            r2 = secondary_trajectory[t, :3]
            distance = np.linalg.norm(r2 - r1)

            if distance < min_distance:
                min_distance = distance
                min_time = t

        return min_distance, min_time

    def predict_risk_batch(
        self,
        primary_states: np.ndarray,
        secondary_states: np.ndarray,
        primary_covs: Optional[np.ndarray] = None,
        secondary_covs: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Batch prediction of collision risk.

        Args:
            primary_states: Batch of primary states (N x 6)
            secondary_states: Batch of secondary states (N x 6)
            primary_covs: Batch of primary covariances (N x 6 x 6)
            secondary_covs: Batch of secondary covariances (N x 6 x 6)

        Returns:
            Array of collision probabilities (N,)
        """
        n_samples = len(primary_states)
        risks = np.zeros(n_samples)

        for i in range(n_samples):
            primary_cov = primary_covs[i] if primary_covs is not None else None
            secondary_cov = secondary_covs[i] if secondary_covs is not None else None

            risks[i] = self.compute_collision_probability(
                primary_states[i],
                secondary_states[i],
                primary_cov,
                secondary_cov
            )

        return risks

    def _default_covariance(self) -> np.ndarray:
        """
        Default covariance matrix for satellite state uncertainty.

        Returns:
            6x6 covariance matrix
        """
        # Position uncertainties (meters^2)
        pos_sigma = 100.0  # 10m standard deviation
        # Velocity uncertainties (m/s)^2
        vel_sigma = 1.0    # 1m/s standard deviation

        cov = np.diag([
            pos_sigma, pos_sigma, pos_sigma,  # position
            vel_sigma, vel_sigma, vel_sigma   # velocity
        ])

        return cov

    def get_baseline_name(self) -> str:
        """Get the name of this baseline method."""
        return "Physics-based Pc"


class KalmanFilterBaseline:
    """
    Kalman filter-based collision risk assessment.
    """

    def __init__(self, process_noise: float = 0.1, measurement_noise: float = 1.0):
        """
        Initialize Kalman filter baseline.

        Args:
            process_noise: Process noise parameter
            measurement_noise: Measurement noise parameter
        """
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise

    def predict_trajectory(self, initial_state: np.ndarray, n_steps: int) -> np.ndarray:
        """
        Predict satellite trajectory using Kalman filter.

        Args:
            initial_state: Initial state [x, y, z, vx, vy, vz]
            n_steps: Number of prediction steps

        Returns:
            Predicted trajectory (n_steps x 6)
        """
        # Simplified Kalman prediction (constant velocity model)
        trajectory = np.zeros((n_steps, 6))
        trajectory[0] = initial_state

        dt = 10.0  # 10 second time steps

        for t in range(1, n_steps):
            # Constant velocity propagation
            trajectory[t, :3] = trajectory[t-1, :3] + trajectory[t-1, 3:] * dt
            trajectory[t, 3:] = trajectory[t-1, 3:]  # Constant velocity

        return trajectory

    def get_baseline_name(self) -> str:
        """Get the name of this baseline method."""
        return "Kalman Filter"