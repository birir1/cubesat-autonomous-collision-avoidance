"""
Data utility helpers for preprocessing and dataset splitting.
"""

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
from sklearn.model_selection import train_test_split


def safe_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Normalize an array safely to avoid division by zero."""
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / (norm + eps)


def train_test_split_stratified(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: Optional[int] = None,
    stratify: Optional[np.ndarray] = None
) -> Dict[str, np.ndarray]:
    """Perform a stratified train/validation/test split."""
    if stratify is None:
        stratify = y

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=test_size + val_size,
        random_state=random_state,
        stratify=stratify
    )

    relative_val_size = val_size / (test_size + val_size) if (test_size + val_size) > 0 else 0.0
    stratify_temp = y_temp if stratify is None else y_temp

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=relative_val_size,
        random_state=random_state,
        stratify=stratify_temp
    )

    return {
        'train': X_train,
        'val': X_val,
        'test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test
    }
