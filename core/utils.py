import numpy as np
import torch
from typing import Union, List

def to_tensor(array: Union[np.ndarray, List[float], torch.Tensor], dtype=torch.float32) -> torch.Tensor:
    """
    Convert input to PyTorch tensor safely.
    
    Args:
        array: np.ndarray, list, or torch.Tensor
        dtype: torch dtype (default: float32)
    
    Returns:
        torch.Tensor
    """
    if isinstance(array, torch.Tensor):
        return array.type(dtype)
    return torch.tensor(array, dtype=dtype)

def normalize_array(arr: np.ndarray) -> np.ndarray:
    """
    Normalize array to zero mean, unit variance.
    """
    mean = np.mean(arr, axis=0)
    std = np.std(arr, axis=0) + 1e-8
    return (arr - mean) / std

def weighted_loss(predictions: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """
    Compute weighted MSE loss.
    """
    loss = (predictions - targets) ** 2
    if weights is not None:
        loss = loss * weights
    return loss.mean()


def relative_position(pos1: np.ndarray, pos2: np.ndarray) -> np.ndarray:
    """
    Compute the relative position vector from pos2 to pos1.

    Args:
        pos1: Position vector 1
        pos2: Position vector 2

    Returns:
        Relative position (pos1 - pos2)
    """
    return np.asarray(pos1, dtype=np.float64) - np.asarray(pos2, dtype=np.float64)


def relative_velocity(vel1: np.ndarray, vel2: np.ndarray) -> np.ndarray:
    """
    Compute the relative velocity vector from vel2 to vel1.

    Args:
        vel1: Velocity vector 1
        vel2: Velocity vector 2

    Returns:
        Relative velocity (vel1 - vel2)
    """
    return np.asarray(vel1, dtype=np.float64) - np.asarray(vel2, dtype=np.float64)


def safe_norm(vec: np.ndarray) -> float:
    """
    Compute the Euclidean norm safely.

    Args:
        vec: Input vector

    Returns:
        Norm of the vector
    """
    vec = np.asarray(vec, dtype=np.float64)
    return float(np.linalg.norm(vec))


def mahalanobis_distance(point: np.ndarray, cov: np.ndarray) -> float:
    """
    Compute Mahalanobis distance for a point given a covariance matrix.

    Args:
        point: Vector of shape (n,)
        cov: Covariance matrix of shape (n, n)

    Returns:
        Mahalanobis distance
    """
    point = np.asarray(point, dtype=np.float64).flatten()
    cov = np.asarray(cov, dtype=np.float64)

    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError("Covariance matrix must be square")

    cov = ensure_positive_definite(cov)
    inv_cov = np.linalg.inv(cov)
    return float(np.sqrt(point.T @ inv_cov @ point))


def ensure_positive_definite(matrix: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    """
    Ensure a matrix is positive definite by adjusting small eigenvalues.

    Args:
        matrix: Square matrix
        epsilon: Small value to add to eigenvalues

    Returns:
        Positive definite matrix
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Input must be a square matrix")

    eigvals, eigvecs = np.linalg.eigh(matrix)
    eigvals = np.clip(eigvals, epsilon, None)
    return eigvecs @ np.diag(eigvals) @ eigvecs.T