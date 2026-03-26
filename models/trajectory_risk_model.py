"""
Transformer-based Trajectory Risk Model
(STABLE + BACKWARD COMPATIBLE + SAFE LOADING + LOGITS FIX)
"""

import torch
import torch.nn as nn


# ============================================
# POSITIONAL ENCODING
# ============================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()

        pe = torch.zeros(max_len, d_model, dtype=torch.float32)

        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register buffer → moves with model automatically
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, T, d_model)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


# ============================================
# MODEL
# ============================================

class TrajectoryRiskModel(nn.Module):
    def __init__(self, input_dim=6, d_model=64, nhead=4, num_layers=2):
        super().__init__()

        # ---- Input projection ----
        self.input_proj = nn.Linear(input_dim, d_model)

        # ---- Positional encoding ----
        self.positional_encoding = PositionalEncoding(d_model)

        # ---- Transformer encoder ----
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=128,
            dropout=0.1,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # ---- Prediction head (NO SIGMOID → logits) ----
        self.fc = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 1)  # logits output
        )

    # ============================================
    # FORWARD
    # ============================================

    def forward(self, x, apply_sigmoid=False):
        """
        Args:
            x: (B, T, 6)
            apply_sigmoid: bool → return probability if True
        """

        # Input projection
        x = self.input_proj(x)

        # Positional encoding
        x = self.positional_encoding(x)

        # Transformer
        x = self.transformer(x)

        # Temporal pooling (robust)
        x = x.mean(dim=1)

        # Logits
        logits = self.fc(x)

        if apply_sigmoid:
            return torch.sigmoid(logits)

        return logits

    # ============================================
    # SAFE LOADING
    # ============================================

    def load_safe(self, state_dict):
        """
        Safe loader:
        - Handles mismatched architectures
        - Handles wrapped checkpoints
        - Ignores incompatible layers
        """

        # Handle wrapped checkpoints
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        model_dict = self.state_dict()

        # Keep only matching keys
        filtered_dict = {
            k: v for k, v in state_dict.items()
            if k in model_dict and v.shape == model_dict[k].shape
        }

        missing = [k for k in model_dict.keys() if k not in filtered_dict]
        unexpected = [k for k in state_dict.keys() if k not in model_dict]

        print(f"[INFO] Loaded {len(filtered_dict)} layers")
        print(f"[INFO] Missing: {len(missing)} | Unexpected: {len(unexpected)}")

        model_dict.update(filtered_dict)
        self.load_state_dict(model_dict, strict=False)