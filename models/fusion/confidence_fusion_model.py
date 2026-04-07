"""
Confidence-Weighted Fusion Model (FINAL - STABLE)

Design Goals:
- Do NOT override the strong trajectory model
- Learn small corrections instead of full prediction
- Dynamically weight models based on confidence
- Prevent collapse and saturation
"""

import torch
import torch.nn as nn


class ConfidenceWeightedFusion(nn.Module):
    def __init__(self):
        super().__init__()

        # =========================================
        # CONFIDENCE ESTIMATOR
        # =========================================
        # Learns how much to trust static correction
        self.conf_net = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 1),  # outputs alpha scalar
            nn.Sigmoid()       # alpha in [0,1]
        )

        # =========================================
        # RESIDUAL CORRECTION NETWORK
        # =========================================
        # Learns a small correction instead of full output
        self.residual_net = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Tanh()  # bounded output
        )

        # scale for residual correction (IMPORTANT)
        self.res_scale = 0.1

    def forward(self, traj_pred, static_pred):
        """
        Inputs:
            traj_pred   : (B, 1)
            static_pred : (B, 1)

        Output:
            fused prediction (B, 1)
        """

        # =========================================
        # CONCAT INPUTS
        # =========================================
        x = torch.cat([traj_pred, static_pred], dim=1)  # (B, 2)

        # =========================================
        # CONFIDENCE WEIGHTS
        # =========================================
        alpha = self.conf_net(x)  # (B, 1), in [0,1]

        # =========================================
        # BASE (TRAJECTORY-FOCUSED WITH GATED STATIC CORRECTION)
        # =========================================
        base = traj_pred + 0.1 * alpha * (static_pred - traj_pred)

        # =========================================
        # RESIDUAL CORRECTION (CONTROLLED)
        # =========================================
        residual = self.residual_net(x)  # (B, 1)

        uncertainty = torch.abs(traj_pred - static_pred)
        gate = torch.sigmoid(5.0 * (uncertainty - 0.1))
        residual = self.res_scale * residual * gate

        # =========================================
        # FINAL OUTPUT
        # =========================================
        fused = base + residual
        fused = torch.clamp(fused, 0.0, 1.0)

        return fused, alpha