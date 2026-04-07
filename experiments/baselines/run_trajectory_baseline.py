"""
Trajectory Baseline Experiment

This script runs baseline experiments using the trajectory transformer model
for satellite collision risk prediction.
"""

import sys
from pathlib import Path
import logging
import json
from datetime import datetime
import torch
from typing import Dict, Any
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from models.trajectory import train_trajectory_transformer
from core.metrics import evaluate_model_predictions
from core.dataset import create_data_loaders

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_trajectory_baseline_experiment(data_path: str = "data",
                                       save_dir: str = "results/baselines",
                                       num_runs: int = 3) -> Dict[str, Any]:
    """
    Run trajectory baseline experiments with multiple runs for statistical significance.

    Args:
        data_path: Path to data directory
        save_dir: Directory to save results
        num_runs: Number of experimental runs

    Returns:
        dict: Experiment results
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"trajectory_baseline_{timestamp}"

    logger.info(f"Starting trajectory baseline experiment: {experiment_name}")

    all_results = []
    best_rmse = float('inf')
    best_model_path = None

    for run in range(num_runs):
        logger.info(f"\n--- Run {run + 1}/{num_runs} ---")
        model_save_path = save_dir / f"trajectory_model_run_{run + 1}.pth"

        try:
            result = train_trajectory_transformer(
                data_path=data_path,
                model_save_path=str(model_save_path),
                num_epochs=50,  # Shorter for baseline
                batch_size=32,
                learning_rate=1e-4,
                sequence_length=10
            )
            all_results.append(result)

            # Track best model
            test_rmse = result['test_metrics']['basic_metrics']['rmse']
            if test_rmse < best_rmse:
                best_rmse = test_rmse
                best_model_path = model_save_path

            logger.info(f"Run {run + 1} completed - Test RMSE: {test_rmse:.6f}")

        except Exception as e:
            logger.exception(f"Run {run + 1} failed")
            all_results.append({'error': str(e)})

    successful_runs = [r for r in all_results if 'error' not in r]
    if not successful_runs:
        logger.error("All runs failed!")
        return {'error': 'All experimental runs failed'}

    # Aggregate metrics
    test_rmses = [r['test_metrics']['basic_metrics']['rmse'] for r in successful_runs]
    test_maes = [r['test_metrics']['basic_metrics']['mae'] for r in successful_runs]
    test_r2s = [r['test_metrics']['basic_metrics']['r2'] for r in successful_runs]

    aggregated_results = {
        'experiment_name': experiment_name,
        'model_type': 'trajectory_baseline',
        'num_runs': num_runs,
        'successful_runs': len(successful_runs),
        'test_rmse': {
            'mean': float(np.mean(test_rmses)),
            'std': float(np.std(test_rmses)),
            'min': float(np.min(test_rmses)),
            'max': float(np.max(test_rmses))
        },
        'test_mae': {
            'mean': float(np.mean(test_maes)),
            'std': float(np.std(test_maes)),
            'min': float(np.min(test_maes)),
            'max': float(np.max(test_maes))
        },
        'test_r2': {
            'mean': float(np.mean(test_r2s)),
            'std': float(np.std(test_r2s)),
            'min': float(np.min(test_r2s)),
            'max': float(np.max(test_r2s))
        },
        'best_model_path': str(best_model_path),
        'best_rmse': best_rmse,
        'individual_runs': all_results,
        'timestamp': timestamp
    }

    results_file = save_dir / f"{experiment_name}_results.json"
    with open(results_file, 'w') as f:
        json.dump(aggregated_results, f, indent=2, default=str)

    logger.info(f"\n{'='*50}")
    logger.info("TRAJECTORY BASELINE EXPERIMENT COMPLETED")
    logger.info(f"Results saved to: {results_file}")
    logger.info(f"Best model: {best_model_path}")

    return aggregated_results


def evaluate_trajectory_baseline_detailed(model_path: str, data_path: str = "data") -> Dict[str, Any]:
    """
    Perform detailed evaluation of a trained trajectory baseline model.

    Args:
        model_path: Path to trained model
        data_path: Path to data directory

    Returns:
        dict: Detailed evaluation results
    """
    from models.trajectory import TrajectoryTransformerPredictor

    logger.info(f"Performing detailed evaluation of trajectory model: {model_path}")

    predictor = TrajectoryTransformerPredictor(model_path)
    _, _, test_loader, _ = create_data_loaders(data_path, batch_size=32, sequence_length=10)

    all_predictions, all_targets, all_features = [], [], []

    predictor.model.eval()
    with torch.no_grad():
        for batch in test_loader:
            features = batch['features']
            targets = batch['raw_target']
            features_seq = features.unsqueeze(1)
            predictions = predictor.model(features_seq.to(predictor.device)).squeeze().cpu().numpy()
            all_predictions.extend(predictions)
            all_targets.extend(targets.numpy())
            all_features.extend(features.numpy())

    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    all_features = np.array(all_features)

    detailed_metrics = evaluate_model_predictions(all_targets, all_predictions, "Trajectory Baseline (Detailed)")

    # Risk-level evaluation
    risk_levels = [0.001, 0.01, 0.1]
    risk_analysis = {}
    for threshold in risk_levels:
        mask = all_targets >= threshold
        if np.sum(mask) > 0:
            level_metrics = evaluate_model_predictions(
                all_targets[mask], all_predictions[mask],
                f"Trajectory Baseline (Risk ≥ {threshold})"
            )
            risk_analysis[f'risk_{threshold}'] = level_metrics['basic_metrics']

    # Feature importance analysis
    feature_importance = {}
    n_features = all_features.shape[1]
    for i in range(n_features):
        error = np.abs(all_targets - all_predictions)
        corr = np.corrcoef(all_features[:, i], error)[0, 1]
        feature_importance[f'feature_{i}'] = abs(corr)

    return {
        'model_path': model_path,
        'metrics': detailed_metrics,
        'risk_level_analysis': risk_analysis,
        'feature_importance': feature_importance,
        'sample_size': len(all_predictions),
        'prediction_stats': {
            'mean': float(np.mean(all_predictions)),
            'std': float(np.std(all_predictions)),
            'min': float(np.min(all_predictions)),
            'max': float(np.max(all_predictions))
        },
        'target_stats': {
            'mean': float(np.mean(all_targets)),
            'std': float(np.std(all_targets)),
            'min': float(np.min(all_targets)),
            'max': float(np.max(all_targets))
        }
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Run Trajectory Baseline Experiment')
    parser.add_argument('--data_path', type=str, default='data', help='Path to data directory')
    parser.add_argument('--save_dir', type=str, default='results/baselines', help='Directory to save results')
    parser.add_argument('--num_runs', type=int, default=3, help='Number of experimental runs')
    parser.add_argument('--detailed_eval', action='store_true', help='Perform detailed evaluation of best model')
    parser.add_argument('--model_path', type=str, default=None, help='Path to model for detailed evaluation')

    args = parser.parse_args()

    if args.detailed_eval and args.model_path:
        try:
            results = evaluate_trajectory_baseline_detailed(args.model_path, args.data_path)
            print("\n" + "="*50)
            print("DETAILED TRAJECTORY BASELINE EVALUATION")
            print("="*50)
            print(f"Model: {args.model_path}")
            print(f"Samples: {results['sample_size']}")
            print(f"Mean Prediction: {results['prediction_stats']['mean']:.6f}")
            print(f"Std Prediction: {results['prediction_stats']['std']:.6f}")
            print(f"Max Prediction: {results['prediction_stats']['max']:.6f}")
            results_file = Path(args.save_dir) / "trajectory_detailed_eval.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"Detailed results saved to: {results_file}")
        except Exception as e:
            logger.exception("Detailed evaluation failed")
    else:
        results = run_trajectory_baseline_experiment(
            data_path=args.data_path,
            save_dir=args.save_dir,
            num_runs=args.num_runs
        )
        print("\nExperiment completed successfully!")
        print(f"Results saved to: {args.save_dir}")
        print(f"Best model: {results['best_model_path']}")