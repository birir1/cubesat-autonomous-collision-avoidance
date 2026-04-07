"""
Calibration metrics for probabilistic predictions.
Ensures predicted probabilities match observed frequencies.
"""

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
import matplotlib.pyplot as plt


def expected_calibration_error(y_true, y_prob, n_bins=10):
    """
    Compute Expected Calibration Error (ECE).

    ECE measures the difference between predicted confidence and
    actual accuracy across confidence bins.

    Args:
        y_true: True binary labels (0 or 1)
        y_prob: Predicted probabilities
        n_bins: Number of bins for calibration curve

    Returns:
        ece: Expected calibration error
        bin_confidences: Confidence values for each bin
        bin_accuracies: Accuracy values for each bin
        bin_counts: Number of samples in each bin
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    # Create bins based on predicted probabilities
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    bin_confidences = []
    bin_accuracies = []
    bin_counts = []

    ece = 0.0
    total_samples = len(y_true)

    for bin_idx in range(n_bins):
        mask = bin_indices == bin_idx
        bin_count = np.sum(mask)

        if bin_count == 0:
            continue

        bin_prob = np.mean(y_prob[mask])
        bin_acc = np.mean(y_true[mask])

        bin_confidences.append(bin_prob)
        bin_accuracies.append(bin_acc)
        bin_counts.append(bin_count)

        # Weighted ECE contribution
        ece += (bin_count / total_samples) * abs(bin_prob - bin_acc)

    return ece, bin_confidences, bin_accuracies, bin_counts


def maximum_calibration_error(y_true, y_prob, n_bins=10):
    """
    Compute Maximum Calibration Error (MCE).

    MCE measures the maximum difference between predicted confidence
    and actual accuracy across any confidence bin.

    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        n_bins: Number of bins

    Returns:
        mce: Maximum calibration error
    """
    _, bin_confidences, bin_accuracies, _ = expected_calibration_error(
        y_true, y_prob, n_bins
    )

    if len(bin_confidences) == 0:
        return 0.0

    mce = max(abs(c - a) for c, a in zip(bin_confidences, bin_accuracies))
    return mce


def brier_score(y_true, y_prob):
    """
    Compute Brier Score for probabilistic predictions.

    Lower scores indicate better calibration.

    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities

    Returns:
        brier: Brier score
    """
    return brier_score_loss(y_true, y_prob)


def calibration_metrics(y_true, y_prob, n_bins=10):
    """
    Compute comprehensive calibration metrics.

    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        n_bins: Number of bins for ECE/MCE

    Returns:
        dict: Dictionary containing all calibration metrics
    """
    ece, bin_confidences, bin_accuracies, bin_counts = expected_calibration_error(
        y_true, y_prob, n_bins
    )

    mce = maximum_calibration_error(y_true, y_prob, n_bins)
    brier = brier_score(y_true, y_prob)

    return {
        'ece': ece,
        'mce': mce,
        'brier_score': brier,
        'bin_confidences': bin_confidences,
        'bin_accuracies': bin_accuracies,
        'bin_counts': bin_counts
    }


def plot_calibration_curve(y_true, y_prob, n_bins=10, save_path=None):
    """
    Plot calibration curve showing predicted vs actual probabilities.

    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        n_bins: Number of bins
        save_path: Path to save plot (optional)
    """
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)

    plt.figure(figsize=(8, 6))
    plt.plot(prob_pred, prob_true, marker='o', label='Model')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly calibrated')

    plt.xlabel('Predicted probability')
    plt.ylabel('Actual probability')
    plt.title('Calibration Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

    plt.close()


def reliability_diagram(y_true, y_prob, n_bins=10, save_path=None):
    """
    Plot reliability diagram showing calibration quality.

    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        n_bins: Number of bins
        save_path: Path to save plot (optional)
    """
    ece, bin_confidences, bin_accuracies, bin_counts = expected_calibration_error(
        y_true, y_prob, n_bins
    )

    plt.figure(figsize=(8, 6))

    # Plot bars for each bin
    x_pos = np.arange(len(bin_confidences))
    plt.bar(x_pos, bin_accuracies, alpha=0.7, label='Accuracy', color='blue')
    plt.scatter(x_pos, bin_confidences, color='red', s=50, label='Confidence', zorder=5)

    # Add bin counts as text
    for i, count in enumerate(bin_counts):
        plt.text(i, max(bin_accuracies[i], bin_confidences[i]) + 0.02,
                f'n={count}', ha='center', va='bottom', fontsize=8)

    plt.xlabel('Confidence Bin')
    plt.ylabel('Probability')
    plt.title('.3f')
    plt.xticks(x_pos, [f'{c:.2f}' for c in bin_confidences])
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.1)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

    plt.close()


def temperature_scaling_calibration(y_true, y_prob, validation_split=0.2):
    """
    Apply temperature scaling for calibration.

    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        validation_split: Fraction of data for validation

    Returns:
        calibrated_probs: Temperature-scaled probabilities
        temperature: Learned temperature parameter
    """
    from scipy.optimize import minimize_scalar

    # Split data
    n_val = int(len(y_true) * validation_split)
    y_val, y_val_prob = y_true[:n_val], y_prob[:n_val]
    y_train, y_train_prob = y_true[n_val:], y_prob[n_val:]

    # Find optimal temperature
    def nll_loss(temp):
        scaled_probs = y_train_prob / temp
        scaled_probs = np.clip(scaled_probs, 1e-7, 1 - 1e-7)
        nll = -np.mean(y_train * np.log(scaled_probs) +
                      (1 - y_train) * np.log(1 - scaled_probs))
        return nll

    result = minimize_scalar(nll_loss, bounds=(0.1, 10), method='bounded')
    temperature = result.x

    # Apply scaling
    calibrated_probs = np.concatenate([
        y_val_prob / temperature,
        y_train_prob / temperature
    ])

    # Sort back to original order
    indices = np.argsort(np.concatenate([np.arange(n_val), np.arange(n_val, len(y_true))]))
    calibrated_probs = calibrated_probs[indices]

    return np.clip(calibrated_probs, 0, 1), temperature