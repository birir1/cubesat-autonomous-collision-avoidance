"""
Evaluation entrypoints for vision object detection models.
"""

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score
from torch.utils.data import DataLoader


def evaluate_detector(model, dataset, device, batch_size: int = 8):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    model.eval()
    preds = []
    labels = []
    with torch.no_grad():
        for batch in loader:
            images = batch['image'].to(device)
            label = batch['label'].cpu().numpy()
            output = model.predict(images).cpu().numpy().flatten()
            preds.append(output)
            labels.append(label)
    if not preds:
        return {}
    preds = np.concatenate(preds)
    labels = np.concatenate(labels)
    binary_preds = (preds >= 0.5).astype(int)
    return {
        'accuracy': float(accuracy_score(labels, binary_preds)),
        'precision': float(precision_score(labels, binary_preds, zero_division=0)),
        'recall': float(recall_score(labels, binary_preds, zero_division=0))
    }
