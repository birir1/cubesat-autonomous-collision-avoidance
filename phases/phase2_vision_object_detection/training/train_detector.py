"""
Training script for vision-based satellite object detectors.
"""

import argparse
import os
import torch
from torch.utils.data import DataLoader
from phases.phase2_vision_object_detection.dataset.space_object_dataset import SpaceObjectDataset
from phases.phase2_vision_object_detection.models.efficientdet_detector import EfficientDetDetector
from phases.phase2_vision_object_detection.models.yolov8_detector import YOLOv8Detector


def train_detector(config: dict):
    detector_type = config.get('detector_type', 'efficientdet')
    data_root = config.get('data_root', 'data/vision')
    batch_size = config.get('batch_size', 8)
    epochs = config.get('epochs', 5)
    lr = float(config.get('learning_rate', 1e-4))

    dataset = SpaceObjectDataset(root_dir=data_root, split='train')
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    if detector_type == 'yolov8':
        model = YOLOv8Detector(num_classes=config.get('num_classes', 1))
    else:
        model = EfficientDetDetector(num_classes=config.get('num_classes', 1))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in loader:
            images = batch['image'].to(device)
            labels = batch['label'].float().to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs.squeeze(-1), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, loss={total_loss / len(loader):.4f}")

    os.makedirs(config.get('output_dir', 'models/vision'), exist_ok=True)
    checkpoint_path = os.path.join(config.get('output_dir', 'models/vision'), f'{detector_type}_detector.pth')
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Saved detector checkpoint to {checkpoint_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a vision object detector.')
    parser.add_argument('--config', type=str, default='configs/dataset.yaml', help='Path to config YAML')
    args = parser.parse_args()
    import yaml
    with open(args.config, 'r') as handle:
        config = yaml.safe_load(handle) or {}
    train_detector(config)
