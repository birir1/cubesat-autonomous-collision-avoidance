"""
Fusion Model Experiment (FIXED)

Compatible with new MultimodalCollisionPredictor API.
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import json
from typing import Dict, Any

from models.multimodal.multimodal_predictor import MultimodalCollisionPredictor
from data.multimodal.dataset import MultimodalSatelliteDataset
from evaluation.metrics import safety_metrics, calibration_metrics


def run_fusion_experiment(config: Dict[str, Any] = None) -> Dict[str, Any]:

    if config is None:
        config = {
            'fusion_types': ['early', 'late', 'cross_attention'],  # kept for logging only
            'modalities': ['trajectory', 'graph', 'vision'],
            'n_trials': 3,
            'batch_size': 32,
            'epochs': 50
        }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    results_dir = Path('results/fusion_experiment')
    results_dir.mkdir(parents=True, exist_ok=True)

    # Dataset
    dataset = MultimodalSatelliteDataset(split='test')
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=False
    )

    results = {}

    for fusion_type in config['fusion_types']:
        print(f"\nRunning {fusion_type} fusion experiment...")

        fusion_results = []

        for trial in range(config['n_trials']):
            print(f"Trial {trial + 1}/{config['n_trials']}")

            # ✅ FIX: use config-based constructor
            model = MultimodalCollisionPredictor(
                trajectory_config=None,
                gnn_config=None,
                vision_config=None,
                fusion_dim=256
            ).to(device)

            model.eval()

            predictions = []
            labels = []

            with torch.no_grad():
                for batch in dataloader:

                    # -----------------------------
                    # Extract inputs (SAFE VERSION)
                    # -----------------------------
                    trajectory = batch.get('trajectory')
                    positions = batch.get('positions')
                    velocities = batch.get('velocities')
                    images = batch.get('vision')
                    risk = batch.get('risk')

                    if trajectory is not None:
                        trajectory = trajectory.to(device)

                    if positions is not None:
                        positions = positions.to(device)

                    if velocities is not None:
                        velocities = velocities.to(device)

                    if images is not None:
                        images = images.to(device)

                    if risk is not None:
                        risk = risk.to(device)

                    # -----------------------------
                    # Forward pass (FIXED)
                    # -----------------------------
                    risk_pred, _, _ = model(
                        trajectory_sequence=trajectory,
                        positions=positions,
                        velocities=velocities,
                        images=images
                    )

                    predictions.extend(risk_pred.cpu().numpy().flatten())
                    labels.extend(risk.cpu().numpy().flatten())

            predictions = np.array(predictions)
            labels = np.array(labels)

            # -----------------------------
            # Metrics
            # -----------------------------
            safety = safety_metrics(labels, predictions, threshold=0.1)
            calib = calibration_metrics(labels, predictions)

            trial_result = {
                'trial': trial,
                'fusion_type': fusion_type,
                'collision_detection_rate': safety.get('collision_detection_rate', 0),
                'false_alarm_rate': safety.get('false_alarm_rate', 0),
                'precision': safety.get('precision', 0),
                'recall': safety.get('recall', 0),
                'ece': calib.get('ece', 0),
                'mce': calib.get('mce', 0),
                'brier_score': calib.get('brier_score', 0)
            }

            fusion_results.append(trial_result)

        # -----------------------------
        # Aggregate
        # -----------------------------
        df = pd.DataFrame(fusion_results)

        results[fusion_type] = {
            'trials': fusion_results,
            'mean_metrics': df.mean(numeric_only=True).to_dict(),
            'std_metrics': df.std(numeric_only=True).to_dict()
        }

        print(f"{fusion_type.upper()} Results:")
        print(f"  CDR: {df['collision_detection_rate'].mean():.3f}")
        print(f"  FAR: {df['false_alarm_rate'].mean():.3f}")
        print(f"  ECE: {df['ece'].mean():.3f}")

    # -----------------------------
    # Save
    # -----------------------------
    with open(results_dir / "fusion_experiment_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    summary = generate_fusion_summary(results)

    with open(results_dir / "fusion_experiment_summary.md", 'w') as f:
        f.write(summary)

    return results


def generate_fusion_summary(results: Dict[str, Any]) -> str:

    summary = []
    summary.append("# Multimodal Fusion Experiment Results\n")

    for fusion_type, data in results.items():
        summary.append(f"## {fusion_type.upper()}\n")

        mean = data['mean_metrics']
        std = data['std_metrics']

        summary.append(f"- CDR: {mean.get('collision_detection_rate', 0):.3f} ± {std.get('collision_detection_rate', 0):.3f}")
        summary.append(f"- FAR: {mean.get('false_alarm_rate', 0):.3f} ± {std.get('false_alarm_rate', 0):.3f}")
        summary.append(f"- ECE: {mean.get('ece', 0):.3f} ± {std.get('ece', 0):.3f}\n")

    return "\n".join(summary)


if __name__ == "__main__":
    results = run_fusion_experiment()
    print("\nFusion experiment completed!")