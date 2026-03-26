import torch
import torch.nn as nn


class TrajectoryTransformerModel(nn.Module):
    def __init__(
        self,
        input_dim=6,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.1
    ):
        super().__init__()

        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)

        # Positional encoding
        self.positional_encoding = nn.Parameter(
            torch.randn(1, 100, d_model)
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Output head
        self.fc = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        x: (batch, time_steps, 6)
        """

        # Project input
        x = self.input_proj(x)

        # Add positional encoding
        x = x + self.positional_encoding[:, :x.size(1), :]

        # Transformer
        x = self.transformer(x)

        # Take last timestep
        x = x[:, -1, :]

        return self.fc(x)