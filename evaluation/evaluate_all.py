"""
evaluate_all.py
Corrected to work with build_dataset(config) from evaluate_models.py
Provides build_test_dataset for train_all.py (GNN input)
"""

import torch
from evaluation.evaluate_models import build_dataset

def build_test_dataset(config):
    """
    Build a dataset (X, y) for GNN training/evaluation.
    Converts output to PyTorch tensors if not already.

    Args:
        config (dict): Configuration dictionary with dataset parameters.

    Returns:
        X (torch.FloatTensor): Feature tensor of shape [num_samples, seq_len, features].
        y (torch.FloatTensor): Target tensor of shape [num_samples].
    """
    X, y = build_dataset(config)

    # Ensure PyTorch tensors
    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X, dtype=torch.float32)
    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y, dtype=torch.float32)

    return X, y


if __name__ == "__main__":
    # Simple standalone test
    test_config = {"num_sats": 500, "num_samples": 1000}
    X, y = build_test_dataset(test_config)
    print(f"Test dataset built: X={X.shape}, y={y.shape}")