"""
Physics Baseline Experiment

This script runs baseline experiments using physics-constrained models
for satellite collision risk prediction.
"""

import sys
from pathlib import Path
import logging
import json
import torch
from datetime import datetime
from typing import Dict, Any
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from models.physics.pc_model import PhysicsConstrainedPredictor, create_physics_constrained_model
from core.metrics import compute_regression_metrics
from core.dataset import create_data_loaders, SatelliteConjunctionDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PhysicsBaselineTrainer:
    """
    Trainer for physics-constrained baseline model.
    """

    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        self.criterion = torch.nn.BCEWithLogitsLoss()  # Binary outputs for risk

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        for batch in train_loader:
            features = batch['features'].to(self.device)
            targets = batch['target'].to(self.device)

            if features.ndim == 3:
                features = features[:, -1, :]  # Last timestep for physics baseline

            self.optimizer.zero_grad()
            outputs = self.model(features)
            loss = self.criterion(outputs.squeeze(), targets.squeeze())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(train_loader)

    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        predictions, targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                features = batch['features'].to(self.device)
                target_vals = batch['target'].to(self.device)
                if features.ndim == 3:
                    features = features[:, -1, :]
                outputs = self.model(features)
                loss = self.criterion(outputs.squeeze(), target_vals.squeeze())
                total_loss += loss.item()
                predictions.extend(outputs.squeeze().cpu().numpy())
                targets.extend(batch['target'].cpu().numpy())
        return total_loss / len(val_loader), np.array(predictions), np.array(targets)

    def train(self, train_loader, val_loader, num_epochs=50, patience=10, save_path=None):
        best_loss = float('inf')
        patience_counter = 0
        for epoch in range(num_epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss, _, _ = self.validate(val_loader) if val_loader else (train_loss, None, None)
            self.scheduler.step(val_loss)

            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                if save_path:
                    torch.save({
                        'model_state_dict': self.model.state_dict(),
                        'epoch': epoch,
                        'loss': val_loss
                    }, save_path)
            else:
                patience_counter += 1

            if patience_counter >= patience:
                break
        return best_loss


def run_physics_baseline_experiment(data_path: str = "data",
                                    save_dir: str = "results/baselines",
                                    num_runs: int = 3) -> Dict[str, Any]:
    """
    Run physics baseline experiments.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"physics_baseline_{timestamp}"

    logger.info(f"Starting physics baseline experiment: {experiment_name}")

    all_results = []
    best_rmse = float('inf')
    best_model_path = None

    # Load dataset once
    dataset = SatelliteConjunctionDataset(data_path)
    train_loader, val_loader = create_data_loaders(dataset, batch_size=32)
    test_loader = val_loader if val_loader else train_loader

    for run in range(num_runs):
        logger.info(f"\n--- Run {run + 1}/{num_runs} ---")
        model_save_path = save_dir / f"physics_model_run_{run + 1}.pth"

        try:
            model = create_physics_constrained_model(
                input_dim=6,
                hidden_dims=[128, 64, 32],
                physics_weight=0.1
            )
            trainer = PhysicsBaselineTrainer(model)
            best_loss = trainer.train(train_loader, val_loader, num_epochs=50, save_path=str(model_save_path))

            predictor = PhysicsConstrainedPredictor(str(model_save_path))
            test_predictions, test_targets = [], []

            for batch in test_loader:
                features = batch['features'].numpy()
                if features.ndim == 3:
                    features = features[:, -1, :]
                test_predictions.extend(predictor.predict_batch(features))
                test_targets.extend(batch['raw_target'].numpy())

            test_metrics = compute_regression_metrics(np.array(test_targets), np.array(test_predictions),
                                                      f"Physics Baseline Run {run + 1}")

            result = {
                'run': run + 1,
                'best_val_loss': float(best_loss),
                'test_metrics': test_metrics
            }
            all_results.append(result)

            test_rmse = test_metrics['basic_metrics']['rmse']
            if test_rmse < best_rmse:
                best_rmse = test_rmse
                best_model_path = model_save_path

            logger.info(f"Run {run + 1} completed - Test RMSE: {test_rmse:.6f}")

        except Exception as e:
            logger.exception(f"Run {run + 1} failed")
            all_results.append({'run': run + 1, 'error': str(e)})

    successful_runs = [r for r in all_results if 'error' not in r]
    if not successful_runs:
        return {'error': 'All runs failed'}

    test_rmses = [r['test_metrics']['basic_metrics']['rmse'] for r in successful_runs]
    aggregated_results = {
        'experiment_name': experiment_name,
        'model_type': 'physics_baseline',
        'num_runs': num_runs,
        'successful_runs': len(successful_runs),
        'test_rmse': {k: float(v) for k, v in {
            'mean': np.mean(test_rmses),
            'std': np.std(test_rmses),
            'min': np.min(test_rmses),
            'max': np.max(test_rmses)
        }.items()},
        'best_model_path': str(best_model_path),
        'best_rmse': float(best_rmse),
        'individual_runs': all_results,
        'timestamp': timestamp
    }

    results_file = save_dir / f"{experiment_name}_results.json"
    with open(results_file, 'w') as f:
        json.dump(aggregated_results, f, indent=2, default=str)

    logger.info(f"\n{'='*50}")
    logger.info("PHYSICS BASELINE EXPERIMENT COMPLETED")
    logger.info(f"Results saved to: {results_file}")
    logger.info(f"Best model: {best_model_path}")

    return aggregated_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Run Physics Baseline Experiment')
    parser.add_argument('--data_path', type=str, default='data', help='Path to data directory')
    parser.add_argument('--save_dir', type=str, default='results/baselines', help='Directory to save results')
    parser.add_argument('--num_runs', type=int, default=3, help='Number of experimental runs')

    args = parser.parse_args()

    results = run_physics_baseline_experiment(
        data_path=args.data_path,
        save_dir=args.save_dir,
        num_runs=args.num_runs
    )

    print("\nPhysics baseline experiment completed successfully!")
    print(f"Results saved to: {args.save_dir}")
    print(f"Best model: {results['best_model_path']}")