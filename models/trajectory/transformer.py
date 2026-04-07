"""
Trajectory Transformer Model for Satellite Collision Avoidance

This module implements a transformer-based model for predicting satellite
trajectories and collision risks using temporal sequence data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional, Tuple, Dict, Any


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer inputs.
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

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
        Add positional encoding to input tensor.

        Args:
            x: Input tensor [seq_len, batch_size, d_model]

        Returns:
            torch.Tensor: Encoded tensor
        """
        return x + self.pe[:x.size(0), :]


class TrajectoryTransformer(nn.Module):
    """
    Transformer model for satellite trajectory prediction and collision risk assessment.
    """

    def __init__(self, input_dim: int = 6, d_model: int = 128, nhead: int = 8,
                 num_layers: int = 6, dim_feedforward: int = 512,
                 dropout: float = 0.1, max_seq_len: int = 100,
                 output_dim: int = 1):
        """
        Initialize the Trajectory Transformer.

        Args:
            input_dim: Dimension of input features (position + velocity)
            d_model: Model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Feedforward dimension
            dropout: Dropout rate
            max_seq_len: Maximum sequence length
            output_dim: Output dimension (collision risk)
        """
        super().__init__()

        self.input_dim = input_dim
        self.d_model = d_model
        self.output_dim = output_dim

        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False  # [seq_len, batch, d_model]
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Output layers
        self.output_projection = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.final_layer = nn.Linear(dim_feedforward, output_dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights."""
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the transformer.

        Args:
            src: Input tensor [seq_len, batch_size, input_dim]
            src_mask: Source mask for attention

        Returns:
            torch.Tensor: Collision risk prediction [batch_size, output_dim]
        """
        # Input projection
        src = self.input_projection(src)  # [seq_len, batch_size, d_model]

        # Add positional encoding
        src = self.pos_encoder(src)

        # Transformer encoding
        memory = self.transformer_encoder(src, src_key_padding_mask=src_mask)

        # Global average pooling across sequence dimension
        # [seq_len, batch_size, d_model] -> [batch_size, d_model]
        pooled = torch.mean(memory, dim=0)

        # Output projection
        output = self.output_projection(pooled)
        output = F.relu(output)
        output = self.dropout(output)
        output = self.final_layer(output)

        # Apply sigmoid for collision risk (0-1 range)
        if self.output_dim == 1:
            output = torch.sigmoid(output)

        return output

    def get_attention_weights(self, src: torch.Tensor) -> torch.Tensor:
        """
        Get attention weights from the transformer layers.

        Args:
            src: Input tensor

        Returns:
            torch.Tensor: Attention weights
        """
        # This is a simplified version - in practice, you'd need to modify
        # the transformer to return attention weights
        with torch.no_grad():
            src = self.input_projection(src)
            src = self.pos_encoder(src)
            # Get attention from first layer
            attn_weights = []
            for layer in self.transformer_encoder.layers:
                # This requires modifying TransformerEncoderLayer to return attention
                # For now, return dummy weights
                pass
        return torch.ones(src.size(0), src.size(0))  # Placeholder


class TrajectoryTransformerPredictor:
    """
    High-level predictor class using the Trajectory Transformer.
    """

    def __init__(self, model_path: Optional[str] = None, device: str = 'cpu'):
        """
        Initialize the predictor.

        Args:
            model_path: Path to saved model weights
            device: Device to run model on
        """
        self.device = torch.device(device)
        self.model = None

        if model_path:
            self.load_model(model_path)

    def build_model(self, **kwargs):
        """Build the transformer model."""
        self.model = TrajectoryTransformer(**kwargs).to(self.device)

    def load_model(self, model_path: str):
        """Load model weights from file."""
        if self.model is None:
            # Default model configuration
            self.build_model()

        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        print(f"Loaded model from {model_path}")

    def save_model(self, model_path: str, optimizer: Optional[torch.optim.Optimizer] = None,
                   epoch: int = 0, loss: float = 0.0):
        """Save model weights to file."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'epoch': epoch,
            'loss': loss
        }

        if optimizer:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()

        torch.save(checkpoint, model_path)
        print(f"Saved model to {model_path}")

    def predict(self, trajectory_sequence: np.ndarray) -> np.ndarray:
        """
        Predict collision risk from trajectory sequence.

        Args:
            trajectory_sequence: Trajectory data [seq_len, input_dim]

        Returns:
            np.ndarray: Collision risk prediction
        """
        self.model.eval()

        # Convert to tensor
        trajectory = torch.tensor(trajectory_sequence, dtype=torch.float32).unsqueeze(1)  # [seq_len, 1, input_dim]
        trajectory = trajectory.to(self.device)

        with torch.no_grad():
            prediction = self.model(trajectory)
            prediction = prediction.squeeze().cpu().numpy()

        return prediction

    def predict_batch(self, trajectory_batch: np.ndarray) -> np.ndarray:
        """
        Predict collision risk for batch of trajectories.

        Args:
            trajectory_batch: Batch of trajectories [batch_size, seq_len, input_dim]

        Returns:
            np.ndarray: Collision risk predictions [batch_size]
        """
        self.model.eval()

        # Convert to tensor [seq_len, batch_size, input_dim]
        trajectories = torch.tensor(trajectory_batch, dtype=torch.float32)
        trajectories = trajectories.permute(1, 0, 2).to(self.device)

        with torch.no_grad():
            predictions = self.model(trajectories)
            predictions = predictions.squeeze().cpu().numpy()

        return predictions


def create_trajectory_transformer(input_dim: int = 6, d_model: int = 128,
                                nhead: int = 8, num_layers: int = 6) -> TrajectoryTransformer:
    """
    Factory function to create a trajectory transformer model.

    Args:
        input_dim: Input feature dimension
        d_model: Model dimension
        nhead: Number of attention heads
        num_layers: Number of transformer layers

    Returns:
        TrajectoryTransformer: Configured model
    """
    model = TrajectoryTransformer(
        input_dim=input_dim,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers
    )
    return model