"""
LSTM Trajectory Prediction Model

Implements LSTM-based trajectory prediction for satellite collision avoidance.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple

class TrajectoryLSTM(nn.Module):
    """
    LSTM-based trajectory prediction model.
    """

    def __init__(self, input_dim: int = 6, hidden_dim: int = 128,
                 num_layers: int = 2, output_dim: int = 6,
                 dropout: float = 0.1):
        """
        Initialize LSTM trajectory model.

        Args:
            input_dim: Input feature dimension (position + velocity)
            hidden_dim: Hidden state dimension
            num_layers: Number of LSTM layers
            output_dim: Output feature dimension
            dropout: Dropout probability
        """
        super(TrajectoryLSTM, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        # Output projection
        self.output_projection = nn.Linear(hidden_dim, output_dim)

        # Optional: Physics-informed constraints
        self.physics_layer = PhysicsConstraintLayer()

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize model weights."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

    def forward(self, x: torch.Tensor, return_hidden: bool = False) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input trajectory sequence (batch_size, seq_len, input_dim)
            return_hidden: Whether to return hidden states

        Returns:
            Predicted trajectory (batch_size, seq_len, output_dim)
        """
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Apply output projection
        predictions = self.output_projection(lstm_out)

        # Apply physics constraints
        predictions = self.physics_layer(predictions, x)

        if return_hidden:
            return predictions, (h_n, c_n)
        else:
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


class AttentionLSTM(TrajectoryLSTM):
    """
    LSTM with attention mechanism for trajectory prediction.
    """

    def __init__(self, input_dim: int = 6, hidden_dim: int = 128,
                 num_layers: int = 2, output_dim: int = 6,
                 num_heads: int = 8, dropout: float = 0.1):
        """
        Initialize Attention LSTM.

        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden state dimension
            num_layers: Number of LSTM layers
            output_dim: Output feature dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super(AttentionLSTM, self).__init__(input_dim, hidden_dim, num_layers, output_dim, dropout)

        # Attention layer
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Layer norm for attention
        self.attention_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, return_hidden: bool = False) -> torch.Tensor:
        """
        Forward pass with attention.

        Args:
            x: Input trajectory sequence (batch_size, seq_len, input_dim)
            return_hidden: Whether to return hidden states

        Returns:
            Predicted trajectory (batch_size, seq_len, output_dim)
        """
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Apply attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        attn_out = self.attention_norm(attn_out + lstm_out)  # Residual connection

        # Apply output projection
        predictions = self.output_projection(attn_out)

        # Apply physics constraints
        predictions = self.physics_layer(predictions, x)

        if return_hidden:
            return predictions, (h_n, c_n)
        else:
            return predictions


class UncertaintyLSTM(TrajectoryLSTM):
    """
    LSTM with uncertainty estimation for trajectory prediction.
    """

    def __init__(self, input_dim: int = 6, hidden_dim: int = 128,
                 num_layers: int = 2, output_dim: int = 6,
                 dropout: float = 0.1, num_samples: int = 50):
        """
        Initialize Uncertainty LSTM.

        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden state dimension
            num_layers: Number of LSTM layers
            output_dim: Output feature dimension
            dropout: Dropout probability
            num_samples: Number of Monte Carlo samples for uncertainty
        """
        super(UncertaintyLSTM, self).__init__(input_dim, hidden_dim, num_layers, output_dim, dropout)

        self.num_samples = num_samples

        # Uncertainty estimation layers
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)  # Log variance
        )

        # Enable dropout during inference for uncertainty
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, return_uncertainty: bool = True) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with uncertainty estimation.

        Args:
            x: Input trajectory sequence (batch_size, seq_len, input_dim)
            return_uncertainty: Whether to return uncertainty estimates

        Returns:
            Tuple of (predictions, uncertainties)
        """
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Apply dropout for uncertainty
        lstm_out = self.dropout(lstm_out)

        # Mean predictions
        predictions = self.output_projection(lstm_out)

        if return_uncertainty:
            # Uncertainty (log variance)
            log_var = self.uncertainty_head(lstm_out)
            uncertainties = torch.exp(log_var)  # Convert to variance

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


def create_lstm_model(config: dict) -> TrajectoryLSTM:
    """
    Factory function to create LSTM model from configuration.

    Args:
        config: Model configuration dictionary

    Returns:
        Configured LSTM model
    """
    model_type = config.get('model_type', 'standard')

    if model_type == 'attention':
        model = AttentionLSTM(
            input_dim=config.get('input_dim', 6),
            hidden_dim=config.get('hidden_dim', 128),
            num_layers=config.get('num_layers', 2),
            output_dim=config.get('output_dim', 6),
            num_heads=config.get('num_heads', 8),
            dropout=config.get('dropout', 0.1)
        )
    elif model_type == 'uncertainty':
        model = UncertaintyLSTM(
            input_dim=config.get('input_dim', 6),
            hidden_dim=config.get('hidden_dim', 128),
            num_layers=config.get('num_layers', 2),
            output_dim=config.get('output_dim', 6),
            dropout=config.get('dropout', 0.1),
            num_samples=config.get('num_samples', 50)
        )
    else:
        model = TrajectoryLSTM(
            input_dim=config.get('input_dim', 6),
            hidden_dim=config.get('hidden_dim', 128),
            num_layers=config.get('num_layers', 2),
            output_dim=config.get('output_dim', 6),
            dropout=config.get('dropout', 0.1)
        )

    return model


if __name__ == "__main__":
    # Example usage
    model = TrajectoryLSTM(input_dim=6, hidden_dim=128, num_layers=2, output_dim=6)

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
    uncertainty_model = UncertaintyLSTM(input_dim=6, hidden_dim=128, num_layers=2, output_dim=6)
    pred_mean, pred_var = uncertainty_model(x)
    print(f"Mean shape: {pred_mean.shape}, Variance shape: {pred_var.shape}")

    # Test Monte Carlo uncertainty
    mc_mean, mc_uncertainty = uncertainty_model.predict_with_uncertainty(x[:, :10], n_steps=5)
    print(f"MC Mean shape: {mc_mean.shape}, MC Uncertainty shape: {mc_uncertainty.shape}")