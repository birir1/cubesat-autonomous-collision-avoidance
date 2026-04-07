import os
import cv2
import numpy as np
from typing import List, Optional


def load_space_images(image_folder: str) -> np.ndarray:
    """Load all space imagery from a folder."""
    images: List[np.ndarray] = []

    if not os.path.exists(image_folder):
        print("Image folder not found:", image_folder)
        return np.array(images)

    for file in sorted(os.listdir(image_folder)):
        if file.lower().endswith(('.jpg', '.png', '.tiff')):
            path = os.path.join(image_folder, file)
            img = cv2.imread(path)
            if img is None:
                continue
            img = cv2.resize(img, (640, 640))
            images.append(img.astype(np.float32) / 255.0)

    return np.array(images)


class SatelliteDataLoader:
    """Simple loader for satellite imagery and metadata."""

    def __init__(self, data_dir: str):
        self.data_dir = data_dir

    def load_images(self, split: str = 'train') -> np.ndarray:
        folder = os.path.join(self.data_dir, split)
        return load_space_images(folder)

    def load_image(self, path: str) -> Optional[np.ndarray]:
        if not os.path.exists(path):
            return None
        img = cv2.imread(path)
        if img is None:
            return None
        img = cv2.resize(img, (640, 640))
        return img.astype(np.float32) / 255.0


class SatelliteTrajectoryDataset:
    """Simple dataset wrapper for trajectory training data."""

    def __init__(self, trajectories: np.ndarray, targets: np.ndarray):
        self.trajectories = trajectories
        self.targets = targets

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int):
        return {
            'trajectory': self.trajectories[idx],
            'target': self.targets[idx]
        }
