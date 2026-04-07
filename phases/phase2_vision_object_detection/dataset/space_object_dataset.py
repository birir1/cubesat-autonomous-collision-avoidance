"""
Dataset for space object imagery and detection labels.
"""

import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class SpaceObjectDataset(Dataset):
    """Synthetic or real dataset for space object detection."""

    def __init__(self, root_dir: str, split: str = 'train', transform: Optional[Any] = None, num_samples: int = 1000):
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.num_samples = num_samples
        self.image_size = (3, 128, 128)
        self.samples = self._load_samples()

    def _load_samples(self) -> List[Dict[str, Any]]:
        if not self.root_dir.exists():
            return self._create_synthetic_samples()

        image_dir = self.root_dir / self.split
        if not image_dir.exists():
            return self._create_synthetic_samples()

        images = sorted(image_dir.glob('*.npy'))
        dataset = []
        for image_path in images:
            label_path = image_path.with_suffix('.label')
            label = 1 if label_path.exists() and label_path.read_text().strip() == '1' else 0
            dataset.append({'image_path': image_path, 'label': label})
        if not dataset:
            return self._create_synthetic_samples()
        return dataset

    def _create_synthetic_samples(self) -> List[Dict[str, Any]]:
        samples = []
        for i in range(self.num_samples):
            samples.append({'image': np.random.rand(*self.image_size).astype(np.float32), 'label': int(random.random() > 0.5)})
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        if 'image_path' in sample:
            image = np.load(sample['image_path']).astype(np.float32)
        else:
            image = sample['image']
        label = sample['label']
        tensor = torch.from_numpy(image).float()
        if self.transform is not None:
            tensor = self.transform(tensor)
        return {'image': tensor, 'label': torch.tensor(label, dtype=torch.float32)}
