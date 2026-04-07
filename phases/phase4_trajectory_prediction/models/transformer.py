"""
Transformer Trajectory Prediction Model

Implements Transformer-based trajectory prediction for satellite collision avoidance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional, Tuple

class PositionalEncoding(nn.Module):
    """
    Positional encoding for Transformer inputs.
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        """
        Initialize positional encoding.

        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
        """
        super(PositionalEncoding, self).__init__()

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.

        Args:
            x: Input tensor (seq_len, batch_size, d_model)

        Returns:
            Tensor with positional encoding added
        """
        return x + self.pe[:x.size(0), :]


class TrajectoryTransformer(nn.Module):
    """
    Transformer-based trajectory prediction model.
    """

    def __init__(self, input_dim: int = 6, hidden_dim: int = 128,
                 num_layers: int = 2, output_dim: int = 6,
                 num_heads: int = 8, dropout: float = 0.1,
                 max_seq_len: int = 100):
        """
        Initialize Transformer trajectory model.

        Args:
            input_dim: Input feature dimension (position + velocity)
            hidden_dim: Hidden/model dimension
            num_layers: Number of Transformer layers
            output_dim: Output feature dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
            max_seq_len: Maximum sequence length
        """
        super(TrajectoryTransformer, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.max_seq_len = max_seq_len

        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)

        # Positional encoding
        self.positional_encoding = PositionalEncoding(hidden_dim, max_seq_len)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection
        self.output_projection = nn.Linear(hidden_dim, output_dim)

        # Optional: Physics-informed constraints
        self.physics_layer = PhysicsConstraintLayer()

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize model weights."""
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

    def forward(self, x: torch.Tensor, return_attention: bool = False) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input trajectory sequence (batch_size, seq_len, input_dim)
            return_attention: Whether to return attention weights

        Returns:
            Predicted trajectory (batch_size, seq_len, output_dim)
        """
        batch_size, seq_len, _ = x.shape

        # Input projection
        x_proj = self.input_projection(x)  # (batch_size, seq_len, hidden_dim)

        # Add positional encoding
        x_proj = x_proj.transpose(0, 1)  # (seq_len, batch_size, hidden_dim)
        x_encoded = self.positional_encoding(x_proj)

        # Transformer encoder
        transformer_out = self.transformer_encoder(x_encoded)

        # Transpose back
        transformer_out = transformer_out.transpose(0, 1)  # (batch_size, seq_len, hidden_dim)

        # Output projection
        predictions = self.output_projection(transformer_out)

        # Apply physics constraints
        predictions = self.physics_layer(predictions, x)

        return predictions

    def predict_next(self, trajectory: torch.Tensor, n_steps: int = 1) -> torch.Tensor:
        """
        Predict next n steps autoregressively.

        Args:
            trajectory: Input trajectory (batch_size, seq_len, input_dim)
            n_steps: Number of steps to predict

        Returns:
            Predicted trajectory continuation (batch_size, n_steps, output_dim)
        """
        self.eval()
        predictions = []

        current_input = trajectory

        with torch.no_grad():
            for _ in range(n_steps):
                # Predict next step
                next_step = self(current_input)[:, -1:]  # Take last timestep prediction

                # Append to predictions
                predictions.append(next_step)

                # Update input for next prediction
                current_input = torch.cat([current_input[:, 1:], next_step], dim=1)

        return torch.cat(predictions, dim=1)


class PhysicsConstraintLayer(nn.Module):
    """
    Physics-informed constraint layer for trajectory predictions.
    """

    def __init__(self, mu: float = 3.986004418e14):  # Earth's gravitational parameter
        """
        Initialize physics constraint layer.

        Args:
            mu: Gravitational parameter (m^3/s^2)
        """
        super(PhysicsConstraintLayer, self).__init__()
        self.mu = mu

    def forward(self, predictions: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        """
        Apply physics constraints to predictions.

        Args:
            predictions: Raw model predictions
            inputs: Input trajectory

        Returns:
            Physics-constrained predictions
        """
        # For now, just return predictions unchanged
        # In a full implementation, this would enforce:
        # - Orbital mechanics constraints
        # - Energy conservation
        # - Angular momentum conservation

        return predictions


class UncertaintyTransformer(TrajectoryTransformer):
    """
    Transformer with uncertainty estimation for trajectory prediction.
    """

    def __init__(self, input_dim: int = 6, hidden_dim: int = 128,
                 num_layers: int = 2, output_dim: int = 6,
                 num_heads: int = 8, dropout: float = 0.1,
                 max_seq_len: int = 100, num_samples: int = 50):
        """
        Initialize Uncertainty Transformer.

        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden/model dimension
            num_layers: Number of Transformer layers
            output_dim: Output feature dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
            max_seq_len: Maximum sequence length
            num_samples: Number of Monte Carlo samples for uncertainty
        """
        super(UncertaintyTransformer, self).__init__(
            input_dim, hidden_dim, num_layers, output_dim,
            num_heads, dropout, max_seq_len
        )

        self.num_samples = num_samples

        # Uncertainty estimation layers
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)  # Log variance
        )

        # Enable dropout during inference for uncertainty
        self.dropout_layers = nn.ModuleList([
            nn.Dropout(dropout) for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor, return_uncertainty: bool = True) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with uncertainty estimation.

        Args:
            x: Input trajectory sequence (batch_size, seq_len, input_dim)
            return_uncertainty: Whether to return uncertainty estimates

        Returns:
            Tuple of (predictions, uncertainties)
        """
        batch_size, seq_len, _ = x.shape

        # Input projection
        x_proj = self.input_projection(x)

        # Add positional encoding
        x_proj = x_proj.transpose(0, 1)
        x_encoded = self.positional_encoding(x_proj)

        # Transformer encoder with dropout for uncertainty
        for i, layer in enumerate(self.transformer_encoder.layers):
            x_encoded = self.dropout_layers[i](x_encoded)
            x_encoded = layer(x_encoded, x_encoded)

        # Transpose back
        transformer_out = x_encoded.transpose(0, 1)

        # Mean predictions
        predictions = self.output_projection(transformer_out)

        if return_uncertainty:
            # Uncertainty (log variance)
            log_var = self.uncertainty_head(transformer_out)
            uncertainties = torch.exp(log_var)

            return predictions, uncertainties
        else:
            return predictions

    def predict_with_uncertainty(self, trajectory: torch.Tensor, n_steps: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict with uncertainty estimation using Monte Carlo dropout.

        Args:
            trajectory: Input trajectory (batch_size, seq_len, input_dim)
            n_steps: Number of steps to predict

        Returns:
            Tuple of (mean_predictions, uncertainties)
        """
        self.train()  # Enable dropout
        predictions = []

        for _ in range(self.num_samples):
            pred = self.predict_next(trajectory, n_steps)
            predictions.append(pred)

        predictions = torch.stack(predictions, dim=0)

        # Calculate mean and variance
        mean_pred = torch.mean(predictions, dim=0)
        uncertainty = torch.var(predictions, dim=0)

        return mean_pred, uncertainty


class TemporalConvolutionalTransformer(TrajectoryTransformer):
    """
    Transformer with temporal convolutional layers for local feature extraction.
    """

    def __init__(self, input_dim: int = 6, hidden_dim: int = 128,
                 num_layers: int = 2, output_dim: int = 6,
                 num_heads: int = 8, dropout: float = 0.1,
                 max_seq_len: int = 100, kernel_size: int = 3):
        """
        Initialize Temporal Convolutional Transformer.

        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden/model dimension
            num_layers: Number of Transformer layers
            output_dim: Output feature dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
            max_seq_len: Maximum sequence length
            kernel_size: Convolutional kernel size
        """
        super(TemporalConvolutionalTransformer, self).__init__(
            input_dim, hidden_dim, num_layers, output_dim,
            num_heads, dropout, max_seq_len
        )

        # Temporal convolutional layers
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )

        # Update input projection (now takes temporal conv output)
        self.input_projection = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor, return_attention: bool = False) -> torch.Tensor:
        """
        Forward pass with temporal convolution.

        Args:
            x: Input trajectory sequence (batch_size, seq_len, input_dim)
            return_attention: Whether to return attention weights

        Returns:
            Predicted trajectory (batch_size, seq_len, output_dim)
        """
        batch_size, seq_len, _ = x.shape

        # Temporal convolution
        x_conv = x.transpose(1, 2)  # (batch_size, input_dim, seq_len)
        x_conv = self.temporal_conv(x_conv)
        x_conv = x_conv.transpose(1, 2)  # (batch_size, seq_len, hidden_dim)

        # Input projection
        x_proj = self.input_projection(x_conv)

        # Add positional encoding
        x_proj = x_proj.transpose(0, 1)
        x_encoded = self.positional_encoding(x_proj)

        # Transformer encoder
        transformer_out = self.transformer_encoder(x_encoded)

        # Transpose back
        transformer_out = transformer_out.transpose(0, 1)

        # Output projection
        predictions = self.output_projection(transformer_out)

        # Apply physics constraints
        predictions = self.physics_layer(predictions, x)

        return predictions


def create_transformer_model(config: dict) -> TrajectoryTransformer:
    """
    Factory function to create Transformer model from configuration.

    Args:
        config: Model configuration dictionary

    Returns:
        Configured Transformer model
    """
    model_type = config.get('model_type', 'standard')

    if model_type == 'uncertainty':
        model = UncertaintyTransformer(
            input_dim=config.get('input_dim', 6),
            hidden_dim=config.get('hidden_dim', 128),
            num_layers=config.get('num_layers', 2),
            output_dim=config.get('output_dim', 6),
            num_heads=config.get('num_heads', 8),
            dropout=config.get('dropout', 0.1),
            max_seq_len=config.get('max_seq_len', 100),
            num_samples=config.get('num_samples', 50)
        )
    elif model_type == 'temporal_conv':
        model = TemporalConvolutionalTransformer(
            input_dim=config.get('input_dim', 6),
            hidden_dim=config.get('hidden_dim', 128),
            num_layers=config.get('num_layers', 2),
            output_dim=config.get('output_dim', 6),
            num_heads=config.get('num_heads', 8),
            dropout=config.get('dropout', 0.1),
            max_seq_len=config.get('max_seq_len', 100),
            kernel_size=config.get('kernel_size', 3)
        )
    else:
        model = TrajectoryTransformer(
            input_dim=config.get('input_dim', 6),
            hidden_dim=config.get('hidden_dim', 128),
            num_layers=config.get('num_layers', 2),
            output_dim=config.get('output_dim', 6),
            num_heads=config.get('num_heads', 8),
            dropout=config.get('dropout', 0.1),
            max_seq_len=config.get('max_seq_len', 100)
        )

    return model


if __name__ == "__main__":
    # Example usage
    model = TrajectoryTransformer(input_dim=6, hidden_dim=128, num_layers=2, output_dim=6)

    # Create dummy input
    batch_size, seq_len, input_dim = 4, 50, 6
    x = torch.randn(batch_size, seq_len, input_dim)

    # Forward pass
    predictions = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {predictions.shape}")

    # Test autoregressive prediction
    next_steps = model.predict_next(x[:, :10], n_steps=5)
    print(f"Next steps shape: {next_steps.shape}")

    # Test uncertainty model
    uncertainty_model = UncertaintyTransformer(input_dim=6, hidden_dim=128, num_layers=2, output_dim=6)
    pred_mean, pred_var = uncertainty_model(x)
    print(f"Mean shape: {pred_mean.shape}, Variance shape: {pred_var.shape}")

    # Test Monte Carlo uncertainty
    mc_mean, mc_uncertainty = uncertainty_model.predict_with_uncertainty(x[:, :10], n_steps=5)
    print(f"MC Mean shape: {mc_mean.shape}, MC Uncertainty shape: {mc_uncertainty.shape}")

    # Test temporal convolutional model
    temp_conv_model = TemporalConvolutionalTransformer(input_dim=6, hidden_dim=128, num_layers=2, output_dim=6)
    temp_pred = temp_conv_model(x)
    print(f"Temporal conv output shape: {temp_pred.shape}")