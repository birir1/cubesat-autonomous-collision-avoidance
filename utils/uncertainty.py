"""
Uncertainty estimation utilities for collision risk prediction.
"""

import numpy as np


def sigmoid(x):
    """Convert logits to probabilities."""
    return 1.0 / (1.0 + np.exp(-x))


def predictive_entropy(probs: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Compute entropy for probability distributions."""
    probs = np.clip(probs, eps, 1.0 - eps)
    return -np.sum(probs * np.log(probs), axis=-1)


def expected_calibration_error(y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 15) -> float:
    """Compute expected calibration error (ECE) for probabilistic predictions."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_pred >= bins[i]) & (y_pred < bins[i + 1])
        if mask.sum() == 0:
            continue
        bin_accuracy = y_true[mask].mean()
        bin_confidence = y_pred[mask].mean()
        ece += np.abs(bin_accuracy - bin_confidence) * mask.sum() / len(y_pred)
    return float(ece)


def aleatoric_uncertainty(variance: np.ndarray) -> np.ndarray:
    """Aleatoric uncertainty from predictive variance estimates."""
    return np.clip(variance, 0.0, None)


def epistemic_uncertainty(predictions: np.ndarray) -> np.ndarray:
    """Estimate epistemic uncertainty from ensemble predictions."""
    return np.var(predictions, axis=0)


def ensemble_uncertainty(ensemble_probs: np.ndarray) -> np.ndarray:
    """Compute total uncertainty from an ensemble of probability predictions."""
    mean_probs = np.mean(ensemble_probs, axis=0)
    entropy_mean = predictive_entropy(mean_probs)
    entropy_samples = np.mean([predictive_entropy(p) for p in ensemble_probs], axis=0)
    return entropy_mean - entropy_samples
