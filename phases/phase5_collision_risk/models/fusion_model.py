"""
Fusion Model for Collision Risk Assessment

Combines multiple modalities (trajectory, features, physics) for collision risk prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import logging

class CollisionRiskFusionModel(nn.Module):
    """
    Multimodal fusion model for collision risk assessment.
    """

    def __init__(self, config: Dict):
        """
        Initialize fusion model.

        Args:
            config: Model configuration
        """
        super(CollisionRiskFusionModel, self).__init__()

        self.config = config
        self.logger = logging.getLogger(__name__)

        # Model dimensions
        self.trajectory_dim = config.get('trajectory_dim', 6)  # x,y,z,vx,vy,vz
        self.time_window = config.get('time_window', 100)
        self.feature_dim = config.get('feature_dim', 50)
        self.hidden_dim = config.get('hidden_dim', 128)
        self.num_layers = config.get('num_layers', 2)

        # Feature engineering flag
        self.feature_engineering = config.get('feature_engineering', True)

        # Trajectory encoder (LSTM)
        self.trajectory_encoder = nn.LSTM(
            input_size=self.trajectory_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=config.get('dropout', 0.1)
        )

        # Feature encoder (MLP)
        if self.feature_engineering:
            self.feature_encoder = nn.Sequential(
                nn.Linear(self.feature_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.get('dropout', 0.1)),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.get('dropout', 0.1))
            )

        # Physics-based features encoder
        self.physics_encoder = nn.Sequential(
            nn.Linear(10, self.hidden_dim // 2),  # Physics features
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, self.hidden_dim // 2),
            nn.ReLU()
        )

        # Fusion layers
        fusion_input_dim = self.hidden_dim * 2  # trajectory + features
        if self.feature_engineering:
            fusion_input_dim += self.hidden_dim

        self.fusion_network = nn.Sequential(
            nn.Linear(fusion_input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.get('dropout', 0.1)),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.get('dropout', 0.1)),
            nn.Linear(self.hidden_dim // 2, 1)  # Binary classification
        )

        # Attention mechanism for temporal features
        self.attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=config.get('num_heads', 8),
            dropout=config.get('dropout', 0.1),
            batch_first=True
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

    def forward(self, inputs: torch.Tensor, features: Optional[torch.Tensor] = None,
                physics_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the fusion model.

        Args:
            inputs: Trajectory inputs [batch_size, time_window, trajectory_dim]
            features: Engineered features [batch_size, feature_dim]
            physics_features: Physics-based features [batch_size, physics_dim]

        Returns:
            Collision risk logits [batch_size, 1]
        """
        batch_size = inputs.size(0)

        # Encode trajectory
        trajectory_encoded, (h_n, c_n) = self.trajectory_encoder(inputs)
        trajectory_features = h_n[-1]  # Use last hidden state

        # Apply attention to trajectory encoding
        attn_output, _ = self.attention(
            trajectory_encoded, trajectory_encoded, trajectory_encoded
        )
        trajectory_features = torch.mean(attn_output, dim=1)  # Average pooling

        # Prepare fusion inputs
        fusion_inputs = [trajectory_features]

        # Encode engineered features if provided
        if self.feature_engineering and features is not None:
            feature_encoded = self.feature_encoder(features)
            fusion_inputs.append(feature_encoded)

        # Encode physics features if provided
        if physics_features is not None:
            physics_encoded = self.physics_encoder(physics_features)
            fusion_inputs.append(physics_encoded)

        # Concatenate all features
        if len(fusion_inputs) > 1:
            combined_features = torch.cat(fusion_inputs, dim=1)
        else:
            combined_features = fusion_inputs[0]

        # Fusion and prediction
        output = self.fusion_network(combined_features)
        return output

    def extract_trajectory_features(self, trajectory: torch.Tensor) -> torch.Tensor:
        """
        Extract features from trajectory for interpretability.

        Args:
            trajectory: Input trajectory [batch_size, time_window, trajectory_dim]

        Returns:
            Trajectory features [batch_size, hidden_dim]
        """
        with torch.no_grad():
            _, (h_n, _) = self.trajectory_encoder(trajectory)
            return h_n[-1]

    def get_feature_importance(self, inputs: torch.Tensor,
                              features: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        Calculate feature importance using gradients.

        Args:
            inputs: Trajectory inputs
            features: Engineered features

        Returns:
            Feature importance scores
        """
        self.eval()

        # Enable gradient computation
        inputs.requires_grad_(True)
        if features is not None:
            features.requires_grad_(True)

        # Forward pass
        output = self.forward(inputs, features)

        # Calculate gradients
        output.backward(torch.ones_like(output))

        importance = {
            'trajectory': torch.mean(torch.abs(inputs.grad)).item()
        }

        if features is not None:
            importance['features'] = torch.mean(torch.abs(features.grad)).item()

        return importance


class EarlyFusionModel(CollisionRiskFusionModel):
    """
    Early fusion model that combines inputs at the input level.
    """

    def __init__(self, config: Dict):
        super(EarlyFusionModel, self).__init__(config)

        # Early fusion concatenates inputs before encoding
        early_input_dim = self.trajectory_dim
        if self.feature_engineering:
            early_input_dim += self.feature_dim

        self.early_encoder = nn.LSTM(
            input_size=early_input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=config.get('dropout', 0.1)
        )

        # Simplified fusion network
        self.fusion_network = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.get('dropout', 0.1)),
            nn.Linear(self.hidden_dim // 2, 1)
        )

    def forward(self, inputs: torch.Tensor, features: Optional[torch.Tensor] = None,
                physics_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Early fusion forward pass.
        """
        # Concatenate trajectory and features early
        if self.feature_engineering and features is not None:
            # Expand features to match time dimension
            features_expanded = features.unsqueeze(1).expand(-1, inputs.size(1), -1)
            combined_input = torch.cat([inputs, features_expanded], dim=2)
        else:
            combined_input = inputs

        # Encode combined input
        _, (h_n, _) = self.early_encoder(combined_input)
        combined_features = h_n[-1]

        # Final prediction
        output = self.fusion_network(combined_features)
        return output


class LateFusionModel(CollisionRiskFusionModel):
    """
    Late fusion model that combines predictions from separate models.
    """

    def __init__(self, config: Dict):
        super(LateFusionModel, self).__init__(config)

        # Separate models for different modalities
        self.trajectory_model = nn.Sequential(
            nn.LSTM(self.trajectory_dim, self.hidden_dim, batch_first=True),
            nn.Linear(self.hidden_dim, 1)
        )

        if self.feature_engineering:
            self.feature_model = nn.Sequential(
                nn.Linear(self.feature_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, 1)
            )

        # Late fusion layer
        num_models = 1 + (1 if self.feature_engineering else 0)
        self.fusion_layer = nn.Sequential(
            nn.Linear(num_models, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, 1)
        )

    def forward(self, inputs: torch.Tensor, features: Optional[torch.Tensor] = None,
                physics_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Late fusion forward pass.
        """
        predictions = []

        # Trajectory model prediction
        traj_out, _ = self.trajectory_model[0](inputs)
        traj_pred = self.trajectory_model[1](traj_out[:, -1])
        predictions.append(traj_pred)

        # Feature model prediction
        if self.feature_engineering and features is not None:
            feat_pred = self.feature_model(features)
            predictions.append(feat_pred)

        # Combine predictions
        combined_pred = torch.cat(predictions, dim=1)
        output = self.fusion_layer(combined_pred)

        return output


class UncertaintyAwareFusionModel(CollisionRiskFusionModel):
    """
    Fusion model with uncertainty estimation.
    """

    def __init__(self, config: Dict):
        super(UncertaintyAwareFusionModel, self).__init__(config)

        # Additional output for uncertainty
        self.uncertainty_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, 1),
            nn.Softplus()  # Ensure positive uncertainty
        )

    def forward(self, inputs: torch.Tensor, features: Optional[torch.Tensor] = None,
                physics_features: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with uncertainty estimation.

        Returns:
            Tuple of (logits, uncertainty)
        """
        # Get base features
        combined_features = self._get_combined_features(inputs, features, physics_features)

        # Prediction and uncertainty
        logits = self.fusion_network[:-1](combined_features)  # Remove last layer
        logits = self.fusion_network[-1](logits)

        uncertainty = self.uncertainty_head(combined_features)

        return logits, uncertainty

    def _get_combined_features(self, inputs: torch.Tensor, features: Optional[torch.Tensor] = None,
                              physics_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Get combined features (helper method)."""
        # Encode trajectory
        _, (h_n, _) = self.trajectory_encoder(inputs)
        trajectory_features = h_n[-1]

        # Prepare fusion inputs
        fusion_inputs = [trajectory_features]

        if self.feature_engineering and features is not None:
            feature_encoded = self.feature_encoder(features)
            fusion_inputs.append(feature_encoded)

        if physics_features is not None:
            physics_encoded = self.physics_encoder(physics_features)
            fusion_inputs.append(physics_encoded)

        return torch.cat(fusion_inputs, dim=1)


if __name__ == "__main__":
    # Example usage
    config = {
        'trajectory_dim': 6,
        'time_window': 100,
        'feature_dim': 50,
        'hidden_dim': 128,
        'num_layers': 2,
        'dropout': 0.1,
        'feature_engineering': True
    }

    model = CollisionRiskFusionModel(config)

    # Dummy input
    batch_size = 4
    trajectory = torch.randn(batch_size, 100, 6)
    features = torch.randn(batch_size, 50)

    output = model(trajectory, features)
    print(f"Output shape: {output.shape}")

    # Test uncertainty model
    uncertainty_model = UncertaintyAwareFusionModel(config)
    logits, uncertainty = uncertainty_model(trajectory, features)
    print(f"Logits shape: {logits.shape}, Uncertainty shape: {uncertainty.shape}")