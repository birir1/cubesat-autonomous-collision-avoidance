import torch
import torch.nn as nn


class AttentionFusionModel(nn.Module):
    """
    Confidence-Weighted Residual Fusion Model

    Uses gate alpha to mix static correction into strong trajectory signal.

    Final = Traj + 0.1 * alpha * (Static - Traj)
    """

    def __init__(self, hidden_dim=32):
        super().__init__()

        self.gate = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, traj_pred, static_pred):
        # traj_pred: (B, 1)
        # static_pred: (B, 1)

        x = torch.cat([traj_pred, static_pred], dim=1)  # (B, 2)
        alpha = self.gate(x)  # (B, 1), in [0,1]

        # Residual correction oriented toward trajectory
        fused = traj_pred + 0.1 * alpha * (static_pred - traj_pred)
        fused = torch.clamp(fused, 0.0, 1.0)
        return fused