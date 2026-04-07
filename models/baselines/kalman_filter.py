import numpy as np
from filterpy.kalman import KalmanFilter


class OrbitalKalmanFilter:
    """
    Orbital Kalman Filter (Constant Velocity Model)

    State:
        x = [x, y, z, vx, vy, vz]

    Measurement:
        z = [x, y, z]
    """

    def __init__(self, dt=1.0, process_acc_std=1e-3, meas_pos_std=100.0):
        """
        Args:
            dt (float): time step (seconds)
            process_acc_std (float): process noise (m/s^2)
            meas_pos_std (float): measurement noise (meters)
        """
        self.dt = dt

        self.kf = KalmanFilter(dim_x=6, dim_z=3)

        # -------------------------------------------------
        # State Transition (Constant Velocity)
        # -------------------------------------------------
        self.kf.F = np.array([
            [1, 0, 0, dt, 0, 0],
            [0, 1, 0, 0, dt, 0],
            [0, 0, 1, 0, 0, dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ])

        # -------------------------------------------------
        # Measurement Model (Position only)
        # -------------------------------------------------
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
        ])

        # -------------------------------------------------
        # Process Noise (Derived from acceleration model)
        # -------------------------------------------------
        q = process_acc_std**2

        dt2 = dt**2
        dt3 = dt**3
        dt4 = dt**4

        Q_pos = (dt4 / 4) * np.eye(3)
        Q_cross = (dt3 / 2) * np.eye(3)
        Q_vel = dt2 * np.eye(3)

        self.kf.Q = q * np.block([
            [Q_pos, Q_cross],
            [Q_cross, Q_vel]
        ])

        # -------------------------------------------------
        # Measurement Noise
        # -------------------------------------------------
        self.kf.R = (meas_pos_std**2) * np.eye(3)

        # -------------------------------------------------
        # Initial Covariance
        # -------------------------------------------------
        self.kf.P = np.eye(6) * 1e6  # large initial uncertainty

        self.initialized = False

    # -------------------------------------------------
    # Initialization
    # -------------------------------------------------
    def initialize(self, state):
        """
        Initialize filter with first full state [x,y,z,vx,vy,vz]
        """
        self.kf.x = np.asarray(state).reshape(6, 1)
        self.initialized = True

    # -------------------------------------------------
    # Single Step
    # -------------------------------------------------
    def step(self, measurement):
        """
        measurement: [x, y, z]
        """
        if not self.initialized:
            raise RuntimeError("Kalman Filter not initialized")

        z = np.asarray(measurement).reshape(3, 1)

        self.kf.predict()
        self.kf.update(z)

        return self.kf.x.flatten(), self.kf.P.copy()

    # -------------------------------------------------
    # Sequence Processing (REAL USE CASE)
    # -------------------------------------------------
    def run_sequence(self, measurements, init_state):
        """
        Run filter over full trajectory

        Args:
            measurements: (T, 3)
            init_state: (6,)

        Returns:
            states: (T, 6)
            covariances: (T, 6, 6)
        """
        self.initialize(init_state)

        states = []
        covariances = []

        for z in measurements:
            x, P = self.step(z)
            states.append(x)
            covariances.append(P)

        return np.array(states), np.array(covariances)

    # -------------------------------------------------
    # Prediction Only (for propagation)
    # -------------------------------------------------
    def predict_only(self, steps=1):
        """
        Propagate state without measurement updates
        """
        preds = []

        for _ in range(steps):
            self.kf.predict()
            preds.append(self.kf.x.flatten())

        return np.array(preds)