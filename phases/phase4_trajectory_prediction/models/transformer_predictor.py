"""
Transformer Trajectory Prediction Model

Predicts future positions of satellites/debris using a
sequence-to-sequence Transformer architecture.

Input:
Sequence of state vectors
[x, y, vx, vy]

Output:
Future positions for prediction horizon
"""

import torch
import torch.nn as nn
import math


# -----------------------------------------------------
# POSITIONAL ENCODING
# -----------------------------------------------------

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=500):

        super().__init__()

        pe = torch.zeros(max_len, d_model)

        position = torch.arange(0, max_len).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2) *
            (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x):

        x = x + self.pe[:, :x.size(1)]

        return x


# -----------------------------------------------------
# TRANSFORMER TRAJECTORY MODEL
# -----------------------------------------------------

class TransformerTrajectoryPredictor(nn.Module):

    def __init__(
        self,
        input_dim=4,
        d_model=128,
        num_heads=8,
        num_layers=4,
        prediction_horizon=10,
        dropout=0.1
    ):
        super().__init__()

        self.prediction_horizon = prediction_horizon

        # input projection
        self.input_projection = nn.Linear(input_dim, d_model)

        # positional encoding
        self.pos_encoder = PositionalEncoding(d_model)

        # transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # prediction head
        self.decoder = nn.Sequential(

            nn.Linear(d_model, 256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, prediction_horizon * 2)
        )

    # -------------------------------------------------

    def forward(self, x):

        """
        x shape:
        (batch, sequence_length, 4)
        """

        batch_size = x.size(0)

        x = self.input_projection(x)

        x = self.pos_encoder(x)

        x = self.transformer_encoder(x)

        # take representation of last timestep
        x = x[:, -1, :]

        pred = self.decoder(x)

        pred = pred.view(
            batch_size,
            self.prediction_horizon,
            2
        )

        return pred