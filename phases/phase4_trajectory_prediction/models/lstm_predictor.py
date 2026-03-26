"""
LSTM Trajectory Prediction Model

Predicts future satellite/debris trajectories from historical state vectors.

Input sequence:
[x, y, vx, vy]

Output:
future positions over prediction horizon

Used for:
- collision prediction
- autonomous maneuver planning
"""

import torch
import torch.nn as nn


# ----------------------------------------------------
# LSTM TRAJECTORY MODEL
# ----------------------------------------------------

class LSTMTrajectoryPredictor(nn.Module):

    def __init__(
        self,
        input_dim=4,
        hidden_dim=128,
        num_layers=2,
        dropout=0.2,
        prediction_horizon=10
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.prediction_horizon = prediction_horizon

        # LSTM encoder
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )

        # Fully connected decoder
        self.decoder = nn.Sequential(

            nn.Linear(hidden_dim, 256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, prediction_horizon * 2)
        )

    # ------------------------------------------------

    def forward(self, x):

        """
        x shape:
        (batch, sequence_length, input_dim)

        Example:
        (32, 20, 4)
        """

        batch_size = x.size(0)

        lstm_out, _ = self.lstm(x)

        # use final hidden state
        last_hidden = lstm_out[:, -1, :]

        predictions = self.decoder(last_hidden)

        predictions = predictions.view(
            batch_size,
            self.prediction_horizon,
            2
        )

        return predictions