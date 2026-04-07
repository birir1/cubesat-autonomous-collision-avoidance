"""
Phase 4: Trajectory Prediction

Implements trajectory prediction models for satellite collision avoidance,
including LSTM, Transformer, and physics-informed neural networks.
"""

from .train import TrajectoryTrainer, train_trajectory_model
from .evaluate import TrajectoryEvaluator, evaluate_trajectory_model
from .models import TrajectoryLSTM, TrajectoryTransformer

__all__ = [
    'TrajectoryTrainer',
    'train_trajectory_model',
    'TrajectoryEvaluator',
    'evaluate_trajectory_model',
    'TrajectoryLSTM',
    'TrajectoryTransformer'
]