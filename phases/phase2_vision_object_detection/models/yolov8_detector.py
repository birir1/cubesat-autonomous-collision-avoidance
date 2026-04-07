"""
Lightweight YOLOv8-inspired detector stub for integration tests.
"""

import torch
import torch.nn as nn


class YOLOv8Detector(nn.Module):
    """Small detection backbone for satellite imagery."""

    def __init__(self, num_classes: int = 1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.head = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.head(x)

    def predict(self, images):
        self.eval()
        with torch.no_grad():
            logits = self.forward(images)
            return torch.sigmoid(logits)
