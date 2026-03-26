import os
import numpy as np


def save_feature_dataset(dataset, output_dir="data/features"):
    """
    Save extracted features to disk

    Args:
        dataset: dict of numpy arrays
        output_dir: where to save
    """

    os.makedirs(output_dir, exist_ok=True)

    for name, data in dataset.items():
        path = os.path.join(output_dir, f"{name}_features.npy")
        np.save(path, data)
        print(f"Saved {name} features → {path} | shape={data.shape}")