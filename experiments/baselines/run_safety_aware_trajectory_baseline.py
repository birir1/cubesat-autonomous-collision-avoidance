"""
Safety-Aware Trajectory Baseline Experiment (FIXED)
"""

import sys
from pathlib import Path
import logging
import json
from datetime import datetime
import torch
import numpy as np

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from models.trajectory.safety_aware_train import train_safety_aware_trajectory_transformer
from core.dataset import create_data_loaders

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_balanced_data_loaders(data_path: str, batch_size: int = 32, danger_threshold: float = 0.7):

    train_loader, val_loader, test_loader, _ = create_data_loaders(data_path, batch_size=batch_size)

    train_targets = np.concatenate([
        batch['raw_target'].numpy().flatten() for batch in train_loader
    ])

    danger_labels = (train_targets > danger_threshold).astype(int)
    class_counts = np.bincount(danger_labels)

    total_samples = len(danger_labels)
    class_weights = total_samples / (len(class_counts) * class_counts)
    sample_weights = class_weights[danger_labels]

    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    train_loader_balanced = torch.utils.data.DataLoader(
        train_loader.dataset,
        batch_size=batch_size,
        sampler=sampler
    )

    return train_loader_balanced, val_loader, test_loader


def run_safety_aware_trajectory_baseline_experiment(
        data_path="data",
        save_dir="results/baselines",
        num_runs=3):

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    all_results = []
    best_fnr = float('inf')
    best_model_path = None

    for run in range(num_runs):
        logger.info(f"\n--- Run {run + 1}/{num_runs} ---")

        train_loader, val_loader, test_loader = create_balanced_data_loaders(data_path)

        model_save_path = save_dir / f"model_run_{run + 1}.pth"

        try:
            result = train_safety_aware_trajectory_transformer(
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                model_save_path=str(model_save_path),
                num_epochs=50
            )

            fnr = result['test_safety_metrics']['false_negative_rate']

            if fnr < best_fnr:
                best_fnr = fnr
                best_model_path = model_save_path

            all_results.append(result)

            logger.info(f"Run {run + 1} - RMSE={result['test_rmse']:.4f}, FNR={fnr:.4f}")

        except Exception as e:
            logger.exception("Run failed")
            all_results.append({'error': str(e)})

    successful_runs = [r for r in all_results if 'error' not in r]

    rmses = [r['test_rmse'] for r in successful_runs]
    fnrs = [r['test_safety_metrics']['false_negative_rate'] for r in successful_runs]

    results = {
        "rmse_mean": float(np.mean(rmses)),
        "fnr_mean": float(np.mean(fnrs)),
        "best_model_path": str(best_model_path)
    }

    with open(save_dir / f"results_{timestamp}.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    results = run_safety_aware_trajectory_baseline_experiment()

    print("\nExperiment completed!")
    print(f"Best model: {results['best_model_path']}")