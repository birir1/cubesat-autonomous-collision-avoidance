import numpy as np


class DistanceThresholdRisk:
    """
    Operational distance-based collision risk model.

    Instead of a hard threshold, we use a smooth risk function:
    - Close distance → high risk
    - Far distance → low risk

    This mimics real-world screening logic used in early conjunction filtering.
    """

    def __init__(self, threshold=1000.0, decay_scale=500.0):
        """
        Args:
            threshold (float): characteristic safety distance (meters)
            decay_scale (float): controls how fast risk decays
        """
        self.threshold = threshold
        self.decay_scale = decay_scale

    def predict(self, distance):
        """
        Convert distance → continuous risk score in [0,1]

        Uses exponential decay:
            risk = exp(-distance / scale)
        """
        distance = np.asarray(distance)

        risk = np.exp(-distance / self.decay_scale)

        return np.clip(risk, 0.0, 1.0)

    def predict_batch(self, distances):
        return self.predict(distances)


# -----------------------------
# Utility function (for scripts)
# -----------------------------
def compute_risk(distance, threshold=1000.0):
    """
    Backward-compatible function (used in your scripts)

    Uses smooth exponential risk instead of binary cutoff.
    """
    model = DistanceThresholdRisk(threshold=threshold)
    return model.predict(distance)