"""Simple Static Collision Risk Model for baseline comparison."""

import torch
import torch.nn as nn
import numpy as np


class StaticCollisionRiskModel(nn.Module):
    """Simple baseline model that predicts risk based on distance."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(6, 1)  # 6 features: pos + vel for 2 satellites

    def forward(self, x):
        # x: (batch, 6) - already the last timestep
        logits = self.linear(x)
        return torch.sigmoid(logits).squeeze(-1)