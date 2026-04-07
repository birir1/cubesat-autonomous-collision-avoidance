import numpy as np
import torch
import pickle
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, random_split
from .features import compute_relative_state
from typing import Optional


class SatelliteConjunctionDataset(Dataset):
    """
    Dataset for satellite conjunction / collision prediction.
    Supports:
    - .pkl datasets
    - .npy datasets (X.npy, y.npy)
    - dictionary samples (baseline compatibility)
    """

    def __init__(self,
                 trajectories: Optional[np.ndarray] = None,
                 labels: Optional[np.ndarray] = None,
                 path: Optional[str] = None):

        # Allow passing path as first arg
        if isinstance(trajectories, str):
            path = trajectories
            trajectories = None

        # =========================================================
        # LOAD FROM FILE OR DIRECTORY
        # =========================================================
        if path is not None:
            path = Path(path)

            # -----------------------------
            # 🔥 CASE 1: DIRECTORY INPUT
            # -----------------------------
            if path.is_dir():

                # --- Try NPY first (your case) ---
                x_file = path / "X.npy"
                y_file = path / "y.npy"

                if x_file.exists() and y_file.exists():
                    print(f"[INFO] Loading NumPy dataset: {x_file}, {y_file}")

                    X = np.load(x_file)
                    y = np.load(y_file)

                    self.X = torch.tensor(X, dtype=torch.float32)
                    self.y = torch.tensor(y, dtype=torch.float32)
                    self.mode = 'tensor'
                    return

                # --- Fallback: look for PKL ---
                pkl_files = list(path.glob("*.pkl"))
                if pkl_files:
                    path = pkl_files[0]
                    print(f"[INFO] Loading PKL dataset: {path}")
                else:
                    raise FileNotFoundError(
                        f"No supported dataset found in directory: {path}\n"
                        f"Expected: X.npy & y.npy OR .pkl file"
                    )

            # -----------------------------
            # 🔥 CASE 2: LOAD PKL FILE
            # -----------------------------
            if path.suffix == ".pkl":
                with open(path, "rb") as f:
                    self.data = pickle.load(f)

                # Case: list of dicts
                if isinstance(self.data, list) and len(self.data) > 0 and isinstance(self.data[0], dict):
                    self.samples = self.data
                    self.y = [d.get('collision_risk', d.get('risk', 0.0)) for d in self.data]
                    self.mode = 'dict'
                    return

                # Case: dict
                elif isinstance(self.data, dict):
                    self.X = torch.tensor(self.data.get('trajectories', []), dtype=torch.float32)
                    self.y = torch.tensor(self.data.get('labels', []), dtype=torch.float32)
                    self.mode = 'tensor'
                    return

                # Case: tuple/list
                elif isinstance(self.data, (tuple, list)) and len(self.data) == 2:
                    self.X = torch.tensor(self.data[0], dtype=torch.float32)
                    self.y = torch.tensor(self.data[1], dtype=torch.float32)
                    self.mode = 'tensor'
                    return

                # Fallback
                else:
                    self.X = torch.tensor(self.data, dtype=torch.float32)
                    self.y = torch.zeros(len(self.X))
                    self.mode = 'tensor'
                    return

        # =========================================================
        # DIRECT INPUT MODE
        # =========================================================
        self.X = torch.tensor(trajectories, dtype=torch.float32) if trajectories is not None else []
        self.y = torch.tensor(labels, dtype=torch.float32) if labels is not None else []
        self.mode = 'tensor'

    def __len__(self):
        return len(self.samples) if hasattr(self, 'samples') else len(self.X)

    def __getitem__(self, idx):

        # -----------------------------
        # DICT MODE (baseline support)
        # -----------------------------
        if hasattr(self, 'mode') and self.mode == 'dict':
            sample = self.samples[idx]

            s1 = sample['state1'][-1]
            s2 = sample['state2'][-1]

            rel_r, rel_v = compute_relative_state(s1, s2)
            feat = np.concatenate([rel_r, rel_v])

            label = sample.get('collision_risk', sample.get('risk', 0.0))

            return {
                'features': torch.tensor(feat, dtype=torch.float32),
                'target': torch.tensor(label, dtype=torch.float32),
                'raw_target': torch.tensor(label, dtype=torch.float32)
            }

        # -----------------------------
        # TENSOR MODE
        # -----------------------------
        return {
            'features': self.X[idx],
            'target': self.y[idx],
            'raw_target': self.y[idx]
        }


# =========================================================
# UTILITY FUNCTIONS
# =========================================================

def build_conjunction_dataset(states_a: np.ndarray, states_b: np.ndarray) -> np.ndarray:
    N, T, _ = states_a.shape
    features = []

    for i in range(N):
        traj_features = []
        for t in range(T):
            f = compute_relative_state(states_a[i, t], states_b[i, t])
            traj_features.append(f)
        features.append(traj_features)

    return np.array(features)


def generate_synthetic_labels(features: np.ndarray) -> np.ndarray:
    risks = []

    for traj in features:
        distances = traj[:, -2]
        min_dist = np.min(distances)
        risk = np.exp(-min_dist / 1000.0)
        risks.append(risk)

    return np.array(risks)


# =========================================================
# DATA LOADER CREATOR
# =========================================================

def create_data_loaders(data,
                        val_dataset=None,
                        batch_size=32,
                        shuffle=True,
                        sequence_length=None):
    """
    Flexible loader supporting:
    - directory path
    - dataset object
    """

    # -----------------------------
    # MODE 1: PATH INPUT
    # -----------------------------
    if isinstance(data, str):
        dataset = SatelliteConjunctionDataset(path=data)

        total_size = len(dataset)

        if total_size < 10:
            raise ValueError("Dataset too small for splitting")

        train_size = int(0.8 * total_size)
        val_size = int(0.1 * total_size)
        test_size = total_size - train_size - val_size

        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size]
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        scalers = None

        return train_loader, val_loader, test_loader, scalers

    # -----------------------------
    # MODE 2: DATASET INPUT
    # -----------------------------
    train_dataset = data

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

    val_loader_out = None
    if val_dataset is not None:
        val_loader_out = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader_out