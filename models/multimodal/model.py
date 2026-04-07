# models/multimodal/model.py
import torch
import torch.nn as nn

class MultimodalTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc(x))
        return self.out(x)