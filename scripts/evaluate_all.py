"""
Evaluate all trained models and report safety metrics.
ALSO exposes dataset builder for training reuse (GNN).
"""

import argparse
import numpy as np

from utils.config import load_yaml_config
from evaluation.evaluate_models import evaluate as evaluate_trajectory_model


# =========================================================
# 🔥 DATASET BUILDER (SHARED WITH TRAINING)
# =========================================================
def build_test_dataset(config=None):
    """
    Builds dataset using SAME logic as evaluation pipeline.

    Returns:
        X: numpy array (N, T, F)
        y: numpy array (N,)
    """

    print("🔧 Building dataset from evaluation pipeline...")

    # 👇 import your real dataset builder logic
    from evaluation.evaluate_models import build_dataset  # MUST exist in your eval code

    dataset = build_dataset()

    # Expecting dict format from your logs
    X = dataset["features"]   # shape (N, T, F)
    y = dataset["risk"]       # shape (N,)

    print(f"✅ Dataset built: X={X.shape}, y={y.shape}")

    return np.array(X), np.array(y)


# =========================================================
# 🔥 MAIN EVALUATION
# =========================================================
def evaluate_all_models(config_path: str = 'configs/evaluation.yaml'):
    config = load_yaml_config(config_path)

    print(f"Running evaluation from {config_path}")

    evaluate_trajectory_model()

    print('Completed trajectory model evaluation.')


# =========================================================
# CLI
# =========================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run evaluation for all configured models.')
    parser.add_argument('--config', type=str, default='configs/evaluation.yaml')
    args = parser.parse_args()

    evaluate_all_models(args.config)