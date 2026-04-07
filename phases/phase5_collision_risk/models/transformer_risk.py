"""
Transformer-based Model for Collision Risk Assessment

Uses transformer architecture for sequence modeling of satellite trajectories.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import logging

class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer inputs.
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
            x: Input tensor [seq_len, batch_size, d_model]

        Returns:
            Encoded tensor [seq_len, batch_size, d_model]
        """
        return x + self.pe[:x.size(0), :]


class TrajectoryTransformerBlock(nn.Module):
    """
    Transformer block for trajectory processing.
    """

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048,
                 dropout: float = 0.1):
        """
        Initialize transformer block.

        Args:
            d_model: Model dimension
            nhead: Number of attention heads
            dim_feedforward: Feedforward dimension
            dropout: Dropout rate
        """
        super(TrajectoryTransformerBlock, self).__init__()

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        # Feedforward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through transformer block.

        Args:
            src: Input tensor [batch_size, seq_len, d_model]
            src_mask: Attention mask

        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        # Self-attention with residual connection
        src2 = self.self_attn(src, src, src, attn_mask=src_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # Feedforward with residual connection
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src


class TransformerRiskModel(nn.Module):
    """
    Transformer-based model for collision risk assessment.
    """

    def __init__(self, config: Dict):
        """
        Initialize transformer risk model.

        Args:
            config: Model configuration
        """
        super(TransformerRiskModel, self).__init__()

        self.config = config
        self.logger = logging.getLogger(__name__)

        # Model dimensions
        self.trajectory_dim = config.get('trajectory_dim', 6)  # x,y,z,vx,vy,vz
        self.time_window = config.get('time_window', 100)
        self.d_model = config.get('d_model', 128)
        self.nhead = config.get('nhead', 8)
        self.num_layers = config.get('num_layers', 4)
        self.dim_feedforward = config.get('dim_feedforward', 512)
        self.dropout = config.get('dropout', 0.1)

        # Input projection
        self.input_projection = nn.Linear(self.trajectory_dim, self.d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(self.d_model, self.time_window)

        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TrajectoryTransformerBlock(
                d_model=self.d_model,
                nhead=self.nhead,
                dim_feedforward=self.dim_feedforward,
                dropout=self.dropout
            )
            for _ in range(self.num_layers)
        ])

        # Output layers
        self.output_projection = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model // 2, 1)
        )

        # Attention pooling
        self.attention_pool = nn.Sequential(
            nn.Linear(self.d_model, 1),
            nn.Softmax(dim=1)
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize model weights."""
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

    def forward(self, trajectory: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through transformer model.

        Args:
            trajectory: Input trajectory [batch_size, time_window, trajectory_dim]
            mask: Attention mask [batch_size, time_window]

        Returns:
            Collision risk logits [batch_size, 1]
        """
        batch_size, seq_len, _ = trajectory.size()

        # Project input to model dimension
        x = self.input_projection(trajectory)  # [batch_size, seq_len, d_model]

        # Add positional encoding
        x = x.transpose(0, 1)  # [seq_len, batch_size, d_model]
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)  # [batch_size, seq_len, d_model]

        # Apply transformer layers
        for transformer_layer in self.transformer_layers:
            x = transformer_layer(x, mask)

        # Attention-based pooling
        attention_weights = self.attention_pool(x)  # [batch_size, seq_len, 1]
        attention_weights = attention_weights.squeeze(-1)  # [batch_size, seq_len]

        # Apply mask if provided
        if mask is not None:
            attention_weights = attention_weights.masked_fill(mask == 0, float('-inf'))
            attention_weights = F.softmax(attention_weights, dim=1)
            attention_weights = attention_weights.masked_fill(mask == 0, 0)
        else:
            attention_weights = F.softmax(attention_weights, dim=1)

        # Weighted sum pooling
        pooled = torch.bmm(attention_weights.unsqueeze(1), x)  # [batch_size, 1, d_model]
        pooled = pooled.squeeze(1)  # [batch_size, d_model]

        # Final prediction
        output = self.output_projection(pooled)
        return output

    def get_attention_weights(self, trajectory: torch.Tensor) -> torch.Tensor:
        """
        Get attention weights for interpretability.

        Args:
            trajectory: Input trajectory [batch_size, time_window, trajectory_dim]

        Returns:
            Attention weights [batch_size, time_window]
        """
        self.eval()

        with torch.no_grad():
            batch_size, seq_len, _ = trajectory.size()

            # Project input
            x = self.input_projection(trajectory)
            x = x.transpose(0, 1)
            x = self.pos_encoder(x)
            x = x.transpose(0, 1)

            # Apply transformer layers
            for transformer_layer in self.transformer_layers:
                x = transformer_layer(x)

            # Get attention weights
            attention_weights = self.attention_pool(x)
            attention_weights = attention_weights.squeeze(-1)
            attention_weights = F.softmax(attention_weights, dim=1)

        return attention_weights

    def extract_features(self, trajectory: torch.Tensor) -> torch.Tensor:
        """
        Extract features from trajectory for analysis.

        Args:
            trajectory: Input trajectory [batch_size, time_window, trajectory_dim]

        Returns:
            Features [batch_size, d_model]
        """
        self.eval()

        with torch.no_grad():
            batch_size, seq_len, _ = trajectory.size()

            # Project input
            x = self.input_projection(trajectory)
            x = x.transpose(0, 1)
            x = self.pos_encoder(x)
            x = x.transpose(0, 1)

            # Apply transformer layers
            for transformer_layer in self.transformer_layers:
                x = transformer_layer(x)

            # Attention pooling
            attention_weights = self.attention_pool(x)
            attention_weights = attention_weights.squeeze(-1)
            attention_weights = F.softmax(attention_weights, dim=1)

            # Weighted features
            features = torch.bmm(attention_weights.unsqueeze(1), x)
            features = features.squeeze(1)

        return features


class UncertaintyAwareTransformer(TransformerRiskModel):
    """
    Transformer model with uncertainty estimation.
    """

    def __init__(self, config: Dict):
        super(UncertaintyAwareTransformer, self).__init__(config)

        # Uncertainty estimation head
        self.uncertainty_projection = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model // 2, 1),
            nn.Softplus()  # Ensure positive uncertainty
        )

    def forward(self, trajectory: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with uncertainty estimation.

        Args:
            trajectory: Input trajectory [batch_size, time_window, trajectory_dim]
            mask: Attention mask

        Returns:
            Tuple of (logits, uncertainty)
        """
        batch_size, seq_len, _ = trajectory.size()

        # Get base features
        x = self.input_projection(trajectory)
        x = x.transpose(0, 1)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)

        # Apply transformer layers
        for transformer_layer in self.transformer_layers:
            x = transformer_layer(x, mask)

        # Attention pooling
        attention_weights = self.attention_pool(x)
        attention_weights = attention_weights.squeeze(-1)

        if mask is not None:
            attention_weights = attention_weights.masked_fill(mask == 0, float('-inf'))
            attention_weights = F.softmax(attention_weights, dim=1)
            attention_weights = attention_weights.masked_fill(mask == 0, 0)
        else:
            attention_weights = F.softmax(attention_weights, dim=1)

        pooled = torch.bmm(attention_weights.unsqueeze(1), x)
        pooled = pooled.squeeze(1)

        # Predictions and uncertainty
        logits = self.output_projection(pooled)
        uncertainty = self.uncertainty_projection(pooled)

        return logits, uncertainty


class TemporalConvolutionTransformer(TransformerRiskModel):
    """
    Transformer with temporal convolution for local feature extraction.
    """

    def __init__(self, config: Dict):
        super(TemporalConvolutionTransformer, self).__init__(config)

        # Temporal convolution layers
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(self.d_model, self.d_model, kernel_size=3, padding=1)
            for _ in range(2)
        ])

        self.conv_norms = nn.ModuleList([
            nn.LayerNorm(self.d_model)
            for _ in range(2)
        ])

    def forward(self, trajectory: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with temporal convolution.
        """
        batch_size, seq_len, _ = trajectory.size()

        # Project input
        x = self.input_projection(trajectory)
        x = x.transpose(0, 1)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)

        # Apply temporal convolutions
        for conv, norm in zip(self.conv_layers, self.conv_norms):
            x_conv = conv(x.transpose(1, 2)).transpose(1, 2)  # Conv1d expects [batch, channels, seq]
            x = x + x_conv  # Residual connection
            x = norm(x)
            x = F.relu(x)

        # Apply transformer layers
        for transformer_layer in self.transformer_layers:
            x = transformer_layer(x, mask)

        # Attention pooling
        attention_weights = self.attention_pool(x)
        attention_weights = attention_weights.squeeze(-1)

        if mask is not None:
            attention_weights = attention_weights.masked_fill(mask == 0, float('-inf'))
            attention_weights = F.softmax(attention_weights, dim=1)
            attention_weights = attention_weights.masked_fill(mask == 0, 0)
        else:
            attention_weights = F.softmax(attention_weights, dim=1)

        pooled = torch.bmm(attention_weights.unsqueeze(1), x)
        pooled = pooled.squeeze(1)

        # Final prediction
        output = self.output_projection(pooled)
        return output


if __name__ == "__main__":
    # Example usage
    config = {
        'trajectory_dim': 6,
        'time_window': 100,
        'd_model': 128,
        'nhead': 8,
        'num_layers': 4,
        'dim_feedforward': 512,
        'dropout': 0.1
    }

    model = TransformerRiskModel(config)

    # Dummy input
    batch_size = 4
    trajectory = torch.randn(batch_size, 100, 6)

    output = model(trajectory)
    print(f"Output shape: {output.shape}")

    # Test attention weights
    attention_weights = model.get_attention_weights(trajectory)
    print(f"Attention weights shape: {attention_weights.shape}")

    # Test uncertainty model
    uncertainty_model = UncertaintyAwareTransformer(config)
    logits, uncertainty = uncertainty_model(trajectory)
    print(f"Logits shape: {logits.shape}, Uncertainty shape: {uncertainty.shape}")

    # Test temporal convolution model
    conv_model = TemporalConvolutionTransformer(config)
    conv_output = conv_model(trajectory)
    print(f"Conv output shape: {conv_output.shape}")