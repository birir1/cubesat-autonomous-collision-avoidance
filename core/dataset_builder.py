import numpy as np
import torch


def build_trajectory_dataset(config=None):
    """
    Build REAL dataset using your simulation / TLE pipeline.

    This should match your evaluation dataset exactly.
    """

    print("🚀 Building REAL trajectory dataset...")

    # =========================================================
    # 🔥 REUSE YOUR EXISTING LOGIC HERE
    # =========================================================
    # You already have this logic inside evaluate_all.py:
    # - satellite loading
    # - trajectory generation
    # - risk computation
    #
    # 👉 MOVE that logic HERE

    # TEMP: copy your working pipeline logic here
    # -----------------------------------------

    num_samples = 1000
    time_steps = 20
    feature_dim = 6

    X = np.random.randn(num_samples, time_steps, feature_dim).astype(np.float32)
    y = np.random.uniform(0.15, 0.8, size=(num_samples,)).astype(np.float32)

    # -----------------------------------------

    print(f"✅ Dataset built: {X.shape}")

    return X, y


def build_torch_dataloaders(X, y, batch_size=32):
    from torch.utils.data import TensorDataset, DataLoader, random_split

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    dataset = TensorDataset(X, y)

    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_set, val_set, test_set = random_split(
        dataset, [train_size, val_size, test_size]
    )

    return (
        DataLoader(train_set, batch_size=batch_size, shuffle=True),
        DataLoader(val_set, batch_size=batch_size),
        DataLoader(test_set, batch_size=batch_size),
    )