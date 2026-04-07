"""
Trajectory Prediction Evaluation

Evaluates trajectory prediction models for satellite collision avoidance,
including prediction accuracy, uncertainty quantification, and safety metrics.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import gaussian_kde
import logging
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import json
import pandas as pd

class TrajectoryEvaluator:
    """
    Comprehensive evaluator for trajectory prediction models.
    """

    def __init__(self, save_dir: Optional[str] = None):
        """
        Initialize evaluator.

        Args:
            save_dir: Directory to save evaluation results
        """
        self.save_dir = Path(save_dir) if save_dir else None
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger(__name__)
        self.results = {}

    def evaluate_predictions(self, predictions: torch.Tensor,
                           targets: torch.Tensor,
                           uncertainties: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        Evaluate trajectory predictions.

        Args:
            predictions: Predicted trajectories (batch_size, seq_len, features)
            targets: Ground truth trajectories (batch_size, seq_len, features)
            uncertainties: Prediction uncertainties (optional)

        Returns:
            Dictionary of evaluation metrics
        """
        # Convert to numpy
        pred_np = predictions.detach().cpu().numpy()
        target_np = targets.detach().cpu().numpy()

        metrics = {}

        # Basic error metrics
        mse = mean_squared_error(target_np.flatten(), pred_np.flatten())
        mae = mean_absolute_error(target_np.flatten(), pred_np.flatten())
        rmse = np.sqrt(mse)

        metrics['mse'] = mse
        metrics['mae'] = mae
        metrics['rmse'] = rmse

        # Trajectory-specific metrics
        metrics.update(self._calculate_trajectory_metrics(pred_np, target_np))

        # Uncertainty metrics
        if uncertainties is not None:
            unc_np = uncertainties.detach().cpu().numpy()
            metrics.update(self._calculate_uncertainty_metrics(pred_np, target_np, unc_np))

        # Safety metrics for collision avoidance
        metrics.update(self._calculate_safety_metrics(pred_np, target_np))

        self.results = metrics
        return metrics

    def _calculate_trajectory_metrics(self, predictions: np.ndarray,
                                    targets: np.ndarray) -> Dict[str, float]:
        """Calculate trajectory-specific metrics."""
        metrics = {}

        # Position error over time
        position_errors = np.sqrt(np.sum((predictions - targets)**2, axis=-1))
        metrics['mean_position_error'] = np.mean(position_errors)
        metrics['max_position_error'] = np.max(position_errors)
        metrics['median_position_error'] = np.median(position_errors)

        # Velocity error (if available)
        if predictions.shape[-1] >= 6:  # position + velocity
            vel_pred = predictions[..., 3:6]
            vel_target = targets[..., 3:6]
            vel_errors = np.sqrt(np.sum((vel_pred - vel_target)**2, axis=-1))
            metrics['mean_velocity_error'] = np.mean(vel_errors)
            metrics['max_velocity_error'] = np.max(vel_errors)

        # Trajectory smoothness (jerk)
        if predictions.shape[1] > 2:  # Need at least 3 points for acceleration
            jerk = self._calculate_jerk(predictions)
            metrics['mean_jerk'] = np.mean(np.abs(jerk))
            metrics['max_jerk'] = np.max(np.abs(jerk))

        return metrics

    def _calculate_jerk(self, trajectories: np.ndarray) -> np.ndarray:
        """Calculate jerk (rate of change of acceleration)."""
        # Simple finite difference
        accel = np.diff(trajectories, axis=1, n=2)  # acceleration
        jerk = np.diff(accel, axis=1)  # jerk
        return jerk

    def _calculate_uncertainty_metrics(self, predictions: np.ndarray,
                                     targets: np.ndarray,
                                     uncertainties: np.ndarray) -> Dict[str, float]:
        """Calculate uncertainty quantification metrics."""
        metrics = {}

        # Prediction Interval Coverage Probability (PICP)
        errors = np.abs(predictions - targets)
        within_interval = errors <= 2 * uncertainties  # 95% confidence interval
        picp = np.mean(within_interval)
        metrics['picp'] = picp

        # Mean Prediction Interval Width (MPIW)
        mpiw = np.mean(2 * uncertainties)
        metrics['mpiw'] = mpiw

        # Negative Log Likelihood (NLL)
        # Assuming Gaussian uncertainty
        nll = 0.5 * np.log(2 * np.pi * uncertainties**2) + (errors**2) / (2 * uncertainties**2)
        metrics['nll'] = np.mean(nll)

        # Calibration error
        expected_coverage = 0.95
        calibration_error = abs(picp - expected_coverage)
        metrics['calibration_error'] = calibration_error

        return metrics

    def _calculate_safety_metrics(self, predictions: np.ndarray,
                                targets: np.ndarray) -> Dict[str, float]:
        """Calculate safety-critical metrics for collision avoidance."""
        metrics = {}

        # Minimum separation distance
        distances = np.sqrt(np.sum(predictions**2, axis=-1))  # Assuming relative trajectories
        min_distance = np.min(distances)
        metrics['min_separation'] = min_distance

        # Collision risk (distance < safety threshold)
        safety_threshold = 1.0  # km
        collision_predictions = np.sum(distances < safety_threshold)
        metrics['collision_predictions'] = collision_predictions

        # False positive/negative rates (would need ground truth collision labels)
        # This is a placeholder for when collision labels are available
        metrics['collision_false_positives'] = 0.0
        metrics['collision_false_negatives'] = 0.0

        # Time to closest approach prediction accuracy
        tca_pred = self._predict_time_to_closest_approach(predictions)
        tca_true = self._predict_time_to_closest_approach(targets)
        tca_error = np.abs(tca_pred - tca_true)
        metrics['tca_prediction_error'] = np.mean(tca_error)

        return metrics

    def _predict_time_to_closest_approach(self, trajectories: np.ndarray) -> np.ndarray:
        """Predict time to closest approach for each trajectory."""
        distances = np.sqrt(np.sum(trajectories**2, axis=-1))
        tca_indices = np.argmin(distances, axis=1)
        return tca_indices.astype(float)

    def plot_prediction_errors(self, predictions: np.ndarray,
                             targets: np.ndarray,
                             save_path: Optional[str] = None):
        """Plot prediction error analysis."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Error distribution
        errors = predictions - targets
        error_magnitudes = np.sqrt(np.sum(errors**2, axis=-1)).flatten()

        axes[0, 0].hist(error_magnitudes, bins=50, alpha=0.7, color='blue', edgecolor='black')
        axes[0, 0].set_xlabel('Prediction Error Magnitude')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Error Distribution')
        axes[0, 0].grid(True, alpha=0.3)

        # Error vs time
        time_steps = np.arange(predictions.shape[1])
        mean_errors = np.mean(np.sqrt(np.sum(errors**2, axis=-1)), axis=0)

        axes[0, 1].plot(time_steps, mean_errors, 'r-', linewidth=2, marker='o')
        axes[0, 1].set_xlabel('Time Step')
        axes[0, 1].set_ylabel('Mean Prediction Error')
        axes[0, 1].set_title('Error vs Time')
        axes[0, 1].grid(True, alpha=0.3)

        # Q-Q plot for normality check
        from scipy.stats import probplot
        probplot(error_magnitudes, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot (Normality Check)')
        axes[1, 0].grid(True, alpha=0.3)

        # Error autocorrelation
        if len(error_magnitudes) > 50:
            autocorr = np.correlate(error_magnitudes - np.mean(error_magnitudes),
                                   error_magnitudes - np.mean(error_magnitudes),
                                   mode='full')
            autocorr = autocorr[autocorr.size // 2:] / autocorr[autocorr.size // 2]
            lags = np.arange(len(autocorr))

            axes[1, 1].plot(lags[:50], autocorr[:50], 'g-', linewidth=2)
            axes[1, 1].set_xlabel('Lag')
            axes[1, 1].set_ylabel('Autocorrelation')
            axes[1, 1].set_title('Error Autocorrelation')
            axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
            axes[1, 1].grid(True, alpha=0.3)

        plt.suptitle('Trajectory Prediction Error Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_trajectory_comparison(self, predictions: np.ndarray,
                                 targets: np.ndarray,
                                 sample_indices: Optional[List[int]] = None,
                                 save_path: Optional[str] = None):
        """Plot trajectory prediction vs ground truth."""
        if sample_indices is None:
            sample_indices = np.random.choice(len(predictions), min(5, len(predictions)), replace=False)

        n_samples = len(sample_indices)
        fig, axes = plt.subplots(n_samples, 1, figsize=(12, 4*n_samples))

        if n_samples == 1:
            axes = [axes]

        time_steps = np.arange(predictions.shape[1])

        for i, idx in enumerate(sample_indices):
            ax = axes[i]

            # Plot trajectories
            if predictions.shape[-1] >= 3:  # 3D position
                ax.plot(time_steps, predictions[idx, :, 0], 'r--', label='Pred X', linewidth=2)
                ax.plot(time_steps, targets[idx, :, 0], 'r-', label='True X', linewidth=2)
                ax.plot(time_steps, predictions[idx, :, 1], 'g--', label='Pred Y', linewidth=2)
                ax.plot(time_steps, targets[idx, :, 1], 'g-', label='True Y', linewidth=2)
                ax.plot(time_steps, predictions[idx, :, 2], 'b--', label='Pred Z', linewidth=2)
                ax.plot(time_steps, targets[idx, :, 2], 'b-', label='True Z', linewidth=2)
            else:
                ax.plot(time_steps, predictions[idx, :, 0], 'r--', label='Predicted', linewidth=2)
                ax.plot(time_steps, targets[idx, :, 0], 'b-', label='Ground Truth', linewidth=2)

            ax.set_xlabel('Time Step')
            ax.set_ylabel('Position')
            ax.set_title(f'Trajectory Sample {idx+1}')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.suptitle('Trajectory Prediction Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_uncertainty_calibration(self, predictions: np.ndarray,
                                   targets: np.ndarray,
                                   uncertainties: np.ndarray,
                                   save_path: Optional[str] = None):
        """Plot uncertainty calibration analysis."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        errors = np.abs(predictions - targets)
        normalized_errors = errors / (uncertainties + 1e-6)

        # Uncertainty vs Error
        axes[0, 0].scatter(uncertainties.flatten(), errors.flatten(),
                          alpha=0.6, color='blue', s=30)
        axes[0, 0].set_xlabel('Predicted Uncertainty')
        axes[0, 0].set_ylabel('Prediction Error')
        axes[0, 0].set_title('Uncertainty vs Error')
        axes[0, 0].grid(True, alpha=0.3)

        # Normalized error distribution
        axes[0, 1].hist(normalized_errors.flatten(), bins=50, alpha=0.7,
                       color='green', edgecolor='black')
        axes[0, 1].axvline(x=1.96, color='red', linestyle='--', linewidth=2,
                          label='95% Confidence')
        axes[0, 1].set_xlabel('Normalized Error')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Normalized Error Distribution')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Reliability diagram
        confidence_levels = np.linspace(0.1, 0.9, 9)
        observed_confidence = []

        for conf in confidence_levels:
            threshold = np.percentile(uncertainties.flatten(), (1-conf)*100)
            mask = uncertainties.flatten() <= threshold
            if np.any(mask):
                coverage = np.mean(normalized_errors.flatten()[mask] <= 1.96)
                observed_confidence.append(coverage)
            else:
                observed_confidence.append(0)

        axes[1, 0].plot(confidence_levels, observed_confidence, 'bo-', linewidth=2, markersize=8)
        axes[1, 0].plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect Calibration')
        axes[1, 0].set_xlabel('Expected Confidence')
        axes[1, 0].set_ylabel('Observed Confidence')
        axes[1, 0].set_title('Reliability Diagram')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Prediction Interval Coverage
        coverage_levels = []
        interval_widths = []

        for alpha in [0.8, 0.9, 0.95, 0.99]:
            z_score = np.abs(np.percentile(np.random.normal(0, 1, 10000), (1-alpha)/2))
            within_interval = errors <= z_score * uncertainties
            coverage = np.mean(within_interval)
            width = np.mean(2 * z_score * uncertainties)

            coverage_levels.append(coverage)
            interval_widths.append(width)

        alphas = [0.8, 0.9, 0.95, 0.99]
        axes[1, 1].scatter(interval_widths, coverage_levels, s=100, alpha=0.7, color='purple')
        axes[1, 1].set_xlabel('Mean Prediction Interval Width')
        axes[1, 1].set_ylabel('Coverage Probability')
        axes[1, 1].set_title('Coverage vs Width')
        axes[1, 1].grid(True, alpha=0.3)

        plt.suptitle('Uncertainty Calibration Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def save_results(self, filename: str = 'trajectory_evaluation.json'):
        """Save evaluation results to file."""
        if self.save_dir:
            filepath = self.save_dir / filename
        else:
            filepath = Path(filename)

        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)

        self.logger.info(f"Results saved to {filepath}")

    def generate_report(self, model_name: str = 'Trajectory Model') -> str:
        """Generate evaluation report."""
        report = f"""
# Trajectory Prediction Evaluation Report

## Model: {model_name}

## Summary Metrics
- RMSE: {self.results.get('rmse', 'N/A'):.4f}
- MAE: {self.results.get('mae', 'N/A'):.4f}
- Mean Position Error: {self.results.get('mean_position_error', 'N/A'):.4f}
- Max Position Error: {self.results.get('max_position_error', 'N/A'):.4f}

## Safety Metrics
- Minimum Separation: {self.results.get('min_separation', 'N/A'):.4f} km
- Collision Predictions: {self.results.get('collision_predictions', 'N/A')}
- TCA Prediction Error: {self.results.get('tca_prediction_error', 'N/A'):.2f} steps

## Uncertainty Metrics
- PICP (95%): {self.results.get('picp', 'N/A'):.3f}
- MPIW: {self.results.get('mpiw', 'N/A'):.4f}
- Calibration Error: {self.results.get('calibration_error', 'N/A'):.4f}
- NLL: {self.results.get('nll', 'N/A'):.4f}
"""

        return report


def evaluate_trajectory_model(model, test_loader, device='cuda', save_dir=None):
    """
    Complete evaluation pipeline for trajectory prediction models.

    Args:
        model: Trained trajectory model
        test_loader: Test data loader
        device: Device to run evaluation on
        save_dir: Directory to save results

    Returns:
        Dictionary of evaluation results
    """
    evaluator = TrajectoryEvaluator(save_dir)
    model.eval()

    all_predictions = []
    all_targets = []
    all_uncertainties = []

    with torch.no_grad():
        for batch in test_loader:
            # Move to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)

            # Forward pass
            outputs = model(batch)

            # Extract predictions and targets
            if isinstance(outputs, dict):
                predictions = outputs['predictions']
                targets = batch['trajectory']
                uncertainties = outputs.get('uncertainties')
            else:
                predictions = outputs
                targets = batch['trajectory']
                uncertainties = None

            all_predictions.append(predictions.cpu())
            all_targets.append(targets.cpu())
            if uncertainties is not None:
                all_uncertainties.append(uncertainties.cpu())

    # Concatenate all batches
    predictions = torch.cat(all_predictions, dim=0)
    targets = torch.cat(all_targets, dim=0)
    uncertainties = torch.cat(all_uncertainties, dim=0) if all_uncertainties else None

    # Evaluate
    results = evaluator.evaluate_predictions(predictions, targets, uncertainties)

    # Generate plots
    if save_dir:
        evaluator.plot_prediction_errors(predictions.numpy(), targets.numpy(),
                                       save_path=f"{save_dir}/error_analysis.png")
        evaluator.plot_trajectory_comparison(predictions.numpy(), targets.numpy(),
                                          save_path=f"{save_dir}/trajectory_comparison.png")

        if uncertainties is not None:
            evaluator.plot_uncertainty_calibration(
                predictions.numpy(), targets.numpy(), uncertainties.numpy(),
                save_path=f"{save_dir}/uncertainty_calibration.png"
            )

        evaluator.save_results()

    return results


if __name__ == "__main__":
    # Example usage with synthetic data
    np.random.seed(42)

    # Generate synthetic trajectory data
    batch_size, seq_len, features = 100, 50, 6  # position + velocity
    predictions = np.random.normal(0, 1, (batch_size, seq_len, features))
    targets = predictions + np.random.normal(0, 0.1, (batch_size, seq_len, features))
    uncertainties = np.abs(np.random.normal(0.1, 0.05, (batch_size, seq_len, features)))

    # Evaluate
    evaluator = TrajectoryEvaluator()
    results = evaluator.evaluate_predictions(
        torch.tensor(predictions),
        torch.tensor(targets),
        torch.tensor(uncertainties)
    )

    print("Evaluation Results:")
    for metric, value in results.items():
        print(f"  {metric}: {value:.4f}")

    # Generate report
    report = evaluator.generate_report("Example Trajectory Model")
    print("\n" + report)