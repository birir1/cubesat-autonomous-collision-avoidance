"""
Safety-Aware Trajectory Transformer Model

This module implements a multi-task transformer model for satellite collision avoidance
with both regression (risk prediction) and classification (danger detection) heads.
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


class SafetyAwareTrajectoryTransformer(nn.Module):
    """
    Multi-task transformer model for satellite collision avoidance.

    Features:
    - Regression head: Continuous collision risk prediction
    - Classification head: Binary danger detection
    - Safety-aware loss functions
    """

    def __init__(self, input_dim: int = 6, d_model: int = 128, nhead: int = 8,
                 num_layers: int = 6, dim_feedforward: int = 512,
                 dropout: float = 0.1, max_seq_len: int = 100):
        """
        Initialize the Safety-Aware Trajectory Transformer.

        Args:
            input_dim: Dimension of input features (position + velocity)
            d_model: Model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Feedforward dimension
            dropout: Dropout rate
            max_seq_len: Maximum sequence length
        """
        super().__init__()

        self.input_dim = input_dim
        self.d_model = d_model

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

        # Shared representation layer
        self.shared_projection = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)

        # Regression head (collision risk)
        self.regression_head = nn.Sequential(
            nn.Linear(dim_feedforward, dim_feedforward // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 2, 1),
            nn.Sigmoid()  # Output in [0, 1]
        )

        # Classification head (danger detection)
        self.classification_head = nn.Sequential(
            nn.Linear(dim_feedforward, dim_feedforward // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 2, 1)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights."""
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the multi-task transformer.

        Args:
            src: Input tensor [seq_len, batch_size, input_dim]
            src_mask: Source mask for attention

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (risk_prediction, danger_logits)
                - risk_prediction: [batch_size, 1] collision risk in [0, 1]
                - danger_logits: [batch_size, 1] danger classification logits
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

        # Shared representation
        shared_repr = self.shared_projection(pooled)
        shared_repr = F.relu(shared_repr)
        shared_repr = self.dropout(shared_repr)

        # Regression head (collision risk)
        risk_prediction = self.regression_head(shared_repr)

        # Classification head (danger detection)
        danger_logits = self.classification_head(shared_repr)

        return risk_prediction, danger_logits

    def predict_risk(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Predict only collision risk (for inference).

        Args:
            src: Input tensor [seq_len, batch_size, input_dim]
            src_mask: Source mask for attention

        Returns:
            torch.Tensor: Collision risk prediction [batch_size, 1]
        """
        risk_pred, _ = self.forward(src, src_mask)
        return risk_pred

    def predict_danger(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Predict only danger probability (for inference).

        Args:
            src: Input tensor [seq_len, batch_size, input_dim]
            src_mask: Source mask for attention

        Returns:
            torch.Tensor: Danger probability [batch_size, 1]
        """
        _, danger_logits = self.forward(src, src_mask)
        return torch.sigmoid(danger_logits)

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


class SafetyAwareLoss(nn.Module):
    """
    Safety-aware loss function for collision risk prediction.

    Combines regression loss with asymmetric penalties and classification loss.
    """

    def __init__(self, underestimation_penalty: float = 5.0,
                 risk_weight_factor: float = 5.0,
                 classification_weight: float = 0.5,
                 danger_threshold: float = 0.7):
        """
        Initialize safety-aware loss.

        Args:
            underestimation_penalty: Penalty multiplier for underestimating risk
            risk_weight_factor: Factor for risk-weighted loss
            classification_weight: Weight for classification loss
            danger_threshold: Threshold for danger classification
        """
        super().__init__()
        self.underestimation_penalty = underestimation_penalty
        self.risk_weight_factor = risk_weight_factor
        self.classification_weight = classification_weight
        self.danger_threshold = danger_threshold

        # Loss functions
        self.mse_loss = nn.MSELoss(reduction='none')
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, risk_pred: torch.Tensor, danger_logits: torch.Tensor,
                targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute safety-aware loss.

        Args:
            risk_pred: Predicted risk [batch_size, 1]
            danger_logits: Danger classification logits [batch_size, 1]
            targets: True risk values [batch_size, 1]

        Returns:
            Dict[str, torch.Tensor]: Loss components and total loss
        """
        # Regression loss with asymmetric penalties
        error = risk_pred - targets
        under_penalty = torch.where(error < 0, self.underestimation_penalty, 1.0)
        risk_weight = 1 + self.risk_weight_factor * targets

        regression_loss = torch.mean(under_penalty * risk_weight * (error ** 2))

        # Classification loss
        danger_labels = (targets > self.danger_threshold).float()
        classification_loss = torch.mean(self.bce_loss(danger_logits.squeeze(), danger_labels))

        # Total loss
        total_loss = regression_loss + self.classification_weight * classification_loss

        return {
            'total_loss': total_loss,
            'regression_loss': regression_loss,
            'classification_loss': classification_loss,
            'underestimation_penalty': torch.mean(under_penalty),
            'risk_weight': torch.mean(risk_weight)
        }


class SafetyAwareTrajectoryPredictor:
    """
    High-level predictor class using the Safety-Aware Trajectory Transformer.
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
        """Build the safety-aware transformer model."""
        self.model = SafetyAwareTrajectoryTransformer(**kwargs).to(self.device)

    def load_model(self, model_path: str):
        """Load model weights from file."""
        if self.model is None:
            # Default model configuration
            self.build_model()

        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        print(f"Loaded safety-aware trajectory model from {model_path}")

    def save_model(self, model_path: str, optimizer: Optional[torch.optim.Optimizer] = None,
                   epoch: int = 0, loss: float = 0.0):
        """Save model to file."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'epoch': epoch,
            'loss': loss
        }

        if optimizer:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()

        torch.save(checkpoint, model_path)
        print(f"Saved safety-aware trajectory model to {model_path}")

    def predict(self, features: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Predict collision risk and danger probability.

        Args:
            features: Input features [seq_len, input_dim] or [batch_size, seq_len, input_dim]

        Returns:
            Dict[str, np.ndarray]: Risk and danger predictions
        """
        self.model.eval()

        # Handle single sequence
        if features.ndim == 2:
            features = features.unsqueeze(0)  # [1, seq_len, input_dim]

        # Permute to [seq_len, batch_size, input_dim]
        if features.shape[0] != features.shape[1]:  # Not already [seq_len, batch, dim]
            features = features.permute(1, 0, 2)

        features_tensor = features.to(self.device)

        with torch.no_grad():
            risk_pred, danger_logits = self.model(features_tensor)
            danger_prob = torch.sigmoid(danger_logits)

            risk_pred = risk_pred.squeeze().cpu().numpy()
            danger_prob = danger_prob.squeeze().cpu().numpy()

        return {
            'risk': risk_pred,
            'danger_probability': danger_prob,
            'danger_class': (danger_prob > 0.5).astype(int)
        }

    def predict_batch(self, features_batch: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Predict collision risk for batch.

        Args:
            features_batch: Batch of features [batch_size, seq_len, input_dim]

        Returns:
            Dict[str, np.ndarray]: Risk and danger predictions
        """
        self.model.eval()

        # Permute to [seq_len, batch_size, input_dim]
        features_tensor = torch.tensor(features_batch, dtype=torch.float32).to(self.device)
        features_tensor = features_tensor.permute(1, 0, 2)

        with torch.no_grad():
            risk_pred, danger_logits = self.model(features_tensor)
            danger_prob = torch.sigmoid(danger_logits)

            risk_pred = risk_pred.squeeze().cpu().numpy()
            danger_prob = danger_prob.squeeze().cpu().numpy()

        return {
            'risk': risk_pred,
            'danger_probability': danger_prob,
            'danger_class': (danger_prob > 0.5).astype(int)
        }