"""
Lightweight EfficientDet-inspired detector stub for satellite image detection.
"""

import torch
import torch.nn as nn


class EfficientDetDetector(nn.Module):
    """Simple detection backbone that mimics EfficientDet design."""

    def __init__(self, num_classes: int = 1):
        super().__init__()
        self.backbone = nn.Sequential(
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
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        return self.head(features)

    def predict(self, images):
        self.eval()
        with torch.no_grad():
            logits = self(images)
            return torch.sigmoid(logits)
