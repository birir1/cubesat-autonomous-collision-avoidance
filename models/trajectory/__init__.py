"""
Trajectory Models Module

This module provides trajectory prediction models for satellite collision avoidance,
including transformer-based architectures for temporal sequence modeling.
"""

from .transformer import (
    TrajectoryTransformer,
    TrajectoryTransformerPredictor,
    create_trajectory_transformer,
    PositionalEncoding
)
from .safety_aware_transformer import (
    SafetyAwareTrajectoryTransformer,
    SafetyAwareLoss,
    SafetyAwareTrajectoryPredictor
)
from .train import (
    TrajectoryTrainer,
    train_trajectory_transformer
)
from .safety_aware_train import (
    SafetyAwareTrajectoryTrainer,
    train_safety_aware_trajectory_transformer
)

__all__ = [
    'TrajectoryTransformer',
    'TrajectoryTransformerPredictor',
    'create_trajectory_transformer',
    'PositionalEncoding',
    'SafetyAwareTrajectoryTransformer',
    'SafetyAwareLoss',
    'SafetyAwareTrajectoryPredictor',
    'TrajectoryTrainer',
    'train_trajectory_transformer',
    'SafetyAwareTrajectoryTrainer',
    'train_safety_aware_trajectory_transformer'
]