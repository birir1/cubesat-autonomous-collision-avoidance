"""
Trajectory Prediction Models

Provides Transformer and LSTM-based models for satellite trajectory prediction.
"""

from .lstm import (
    TrajectoryLSTM,
    AttentionLSTM,
    UncertaintyLSTM,
    create_lstm_model
)

from .transformer import (
    TrajectoryTransformer,
    UncertaintyTransformer,
    TemporalConvolutionalTransformer,
    create_transformer_model
)

__all__ = [
    # LSTM models
    'TrajectoryLSTM',
    'AttentionLSTM',
    'UncertaintyLSTM',
    'create_lstm_model',

    # Transformer models
    'TrajectoryTransformer',
    'UncertaintyTransformer',
    'TemporalConvolutionalTransformer',
    'create_transformer_model'
]