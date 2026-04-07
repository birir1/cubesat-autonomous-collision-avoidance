"""
Training Script for Multimodal Collision Risk Predictor

This script trains the complete multimodal model that integrates:
- Transformer for trajectory modeling
- GNN for neighbor interactions
- Vision model for perception
- Cross-modal fusion for risk prediction
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import yaml
import argparse
import os
from tqdm import tqdm
import wandb

from models.multimodal.multimodal_predictor import MultimodalCollisionPredictor
from core.dataset import SatelliteConjunctionDataset
from core.metrics import safety_metrics


class MultimodalSatelliteDataset(Dataset):
    """
    Dataset for multimodal satellite collision prediction training.
    """

    def __init__(self, data_path, mode='train', sequence_length=20, num_satellites=10):
        self.data_path = data_path
        self.mode = mode
        self.sequence_length = sequence_length
        self.num_satellites = num_satellites

        # Load data
        self.load_data()

    def load_data(self):
        """Load trajectory, graph, and vision data."""
        # Load trajectory data
        traj_data = np.load(os.path.join(self.data_path, f'{self.mode}_trajectory.npy'))
        self.trajectory_data = torch.FloatTensor(traj_data)

        # Load graph data (positions and velocities)
        pos_data = np.load(os.path.join(self.data_path, f'{self.mode}_positions.npy'))
        vel_data = np.load(os.path.join(self.data_path, f'{self.mode}_velocities.npy'))
        self.positions = torch.FloatTensor(pos_data)
        self.velocities = torch.FloatTensor(vel_data)

        # Load vision data (if available)
        vision_path = os.path.join(self.data_path, f'{self.mode}_images.npy')
        if os.path.exists(vision_path):
            self.vision_data = np.load(vision_path)
            self.has_vision = True
        else:
            self.vision_data = None
            self.has_vision = False

        # Load labels
        labels = np.load(os.path.join(self.data_path, f'{self.mode}_labels.npy'))
        self.labels = torch.FloatTensor(labels)

        self.num_samples = len(self.labels)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Get trajectory sequence
        traj_seq = self.trajectory_data[idx]  # (sequence_length, 6)

        # Get current positions and velocities (for graph)
        pos = self.positions[idx]  # (num_satellites, 3)
        vel = self.velocities[idx]  # (num_satellites, 3)

        # Get vision data
        if self.has_vision:
            images = self.vision_data[idx]  # This would need proper handling
            if isinstance(images, np.ndarray):
                images = torch.from_numpy(images).float()
        else:
            images = torch.empty(0)

        # Get label
        label = self.labels[idx]  # scalar

        return {
            'trajectory': traj_seq,
            'positions': pos,
            'velocities': vel,
            'images': images,
            'label': label
        }


def create_synthetic_multimodal_data(num_samples=1000, sequence_length=20, num_satellites=10, data_dir='data/synthetic', val_fraction=0.15, test_fraction=0.15):
    """
    Create synthetic multimodal training, validation, and test data for testing.
    """
    os.makedirs(data_dir, exist_ok=True)

    print("Creating synthetic multimodal training data...")

    # Generate trajectory data
    trajectory_data = []
    for _ in range(num_samples):
        t = np.linspace(0, 2*np.pi, sequence_length)
        x = 6371 + 500 + 50*np.sin(t) + np.random.normal(0, 10, sequence_length)
        y = np.random.normal(0, 100, sequence_length)
        z = 200*np.cos(t) + np.random.normal(0, 20, sequence_length)
        vx = 50*np.cos(t) + np.random.normal(0, 5, sequence_length)
        vy = np.random.normal(0, 10, sequence_length)
        vz = -200*np.sin(t) + np.random.normal(0, 5, sequence_length)

        trajectory_data.append(np.stack([x, y, z, vx, vy, vz], axis=1))

    trajectory_data = np.array(trajectory_data)

    # Generate positions and velocities for graph
    positions = np.random.uniform(6371, 6371 + 2000, (num_samples, num_satellites, 3))
    velocities = np.random.uniform(-7.8, 7.8, (num_samples, num_satellites, 3))

    # Generate synthetic labels (collision risk) based on neighbor proximity
    labels = []
    for i in range(num_samples):
        pos = positions[i]
        dist_matrix = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
        np.fill_diagonal(dist_matrix, np.inf)
        min_distance = np.min(dist_matrix)
        risk = 1 / (1 + np.exp(-(350.0 - min_distance) / 80.0))
        risk += np.random.normal(0, 0.05)
        labels.append(np.clip(risk, 0, 1))

    labels = np.array(labels)

    # Split into train/val/test
    idx = np.arange(num_samples)
    np.random.shuffle(idx)
    train_end = int((1.0 - val_fraction - test_fraction) * num_samples)
    val_end = train_end + int(val_fraction * num_samples)

    train_idx = idx[:train_end]
    val_idx = idx[train_end:val_end]
    test_idx = idx[val_end:]

    splits = {
        'train': train_idx,
        'val': val_idx,
        'test': test_idx
    }

    for split_name, split_idx in splits.items():
        np.save(os.path.join(data_dir, f'{split_name}_trajectory.npy'), trajectory_data[split_idx])
        np.save(os.path.join(data_dir, f'{split_name}_positions.npy'), positions[split_idx])
        np.save(os.path.join(data_dir, f'{split_name}_velocities.npy'), velocities[split_idx])
        np.save(os.path.join(data_dir, f'{split_name}_labels.npy'), labels[split_idx])

    print(f"Created {num_samples} synthetic samples in {data_dir} with splits train/val/test")
    return data_dir


def train_multimodal_model(config):
    """
    Train the multimodal collision predictor.
    """
    # Set random seeds
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])

    # Initialize wandb if enabled
    if config.get('use_wandb', False):
        wandb.init(project="cubesat-collision-prediction", config=config)

    # Create data directory and synthetic data if needed
    data_dir = config['data_dir']
    if not os.path.exists(data_dir):
        data_dir = create_synthetic_multimodal_data(
            num_samples=config['num_samples'],
            sequence_length=config['sequence_length'],
            num_satellites=config['num_satellites'],
            data_dir=data_dir
        )

    # Validate splits and generate synthetic validation/test data if needed
    required_files = [
        os.path.join(data_dir, 'train_trajectory.npy'),
        os.path.join(data_dir, 'val_trajectory.npy'),
        os.path.join(data_dir, 'test_trajectory.npy')
    ]
    if not all(os.path.exists(path) for path in required_files):
        data_dir = create_synthetic_multimodal_data(
            num_samples=config['num_samples'],
            sequence_length=config['sequence_length'],
            num_satellites=config['num_satellites'],
            data_dir=data_dir
        )

    # Create datasets
    train_dataset = MultimodalSatelliteDataset(
        data_dir, mode='train',
        sequence_length=config['sequence_length'],
        num_satellites=config['num_satellites']
    )

    val_dataset = MultimodalSatelliteDataset(
        data_dir, mode='val',
        sequence_length=config['sequence_length'],
        num_satellites=config['num_satellites']
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers']
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers']
    )

    # Initialize model
    model = MultimodalCollisionPredictor(
        trajectory_config=config['trajectory_config'],
        gnn_config=config['gnn_config'],
        vision_config=config['vision_config'],
        fusion_dim=config['fusion_dim'],
        dropout=config['dropout']
    )

    # Move to device with CUDA fallback
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    # Training loop
    best_val_auc = 0.0
    patience = config['patience']
    patience_counter = 0

    for epoch in range(config['num_epochs']):
        # Training phase
        model.train()
        train_loss = 0.0
        train_preds = []
        train_labels = []

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']}"):
            # Move data to device
            trajectory = batch['trajectory'].to(device)
            positions = batch['positions'].to(device)
            velocities = batch['velocities'].to(device)
            labels = batch['label'].to(device)

            images = batch['images']
            if isinstance(images, torch.Tensor) and images.numel() == 0:
                images = None

            # Forward pass
            risk_pred, _, _ = model(
                trajectory_sequence=trajectory,
                positions=positions,
                velocities=velocities,
                images=images
            )

            # Calculate loss
            loss = criterion(risk_pred.squeeze(), labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_preds.extend(risk_pred.squeeze().cpu().detach().numpy())
            train_labels.extend(labels.cpu().numpy())

        # Calculate training metrics
        train_labels_np = np.array(train_labels)
        train_label_binary = (train_labels_np >= 0.5).astype(int)
        train_auc = roc_auc_score(train_label_binary, train_preds) if len(np.unique(train_label_binary)) > 1 else float('nan')
        train_loss /= len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for batch in val_loader:
                trajectory = batch['trajectory'].to(device)
                positions = batch['positions'].to(device)
                velocities = batch['velocities'].to(device)
                labels = batch['label'].to(device)
                images = batch['images']
                if isinstance(images, torch.Tensor) and images.numel() == 0:
                    images = None

                risk_pred, _, _ = model(
                    trajectory_sequence=trajectory,
                    positions=positions,
                    velocities=velocities,
                    images=images
                )

                loss = criterion(risk_pred.squeeze(), labels)

                val_loss += loss.item()
                val_preds.extend(risk_pred.squeeze().cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_labels_np = np.array(val_labels)
        val_label_binary = (val_labels_np >= 0.5).astype(int)
        val_auc = roc_auc_score(val_label_binary, val_preds) if len(np.unique(val_label_binary)) > 1 else float('nan')
        val_loss /= len(val_loader)

        # Update learning rate
        scheduler.step()

        # Log metrics
        if config.get('use_wandb', False):
            wandb.log({
                'epoch': epoch,
                'train_loss': train_loss,
                'train_auc': train_auc,
                'val_loss': val_loss,
                'val_auc': val_auc,
                'learning_rate': optimizer.param_groups[0]['lr']
            })

        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train AUC={train_auc:.4f}, "
              f"Val Loss={val_loss:.4f}, Val AUC={val_auc:.4f}")

        # Save best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0

            # Save model
            os.makedirs(config['checkpoint_dir'], exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auc': val_auc,
                'config': config
            }, os.path.join(config['checkpoint_dir'], 'best_multimodal_model.pth'))

            print(f"Saved best model with Val AUC: {val_auc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    print("Training completed!")
    return model


def main():
    parser = argparse.ArgumentParser(description='Train Multimodal Collision Predictor')
    parser.add_argument('--config', type=str, default='configs/multimodal_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--data_dir', type=str, default='data/synthetic_multimodal',
                       help='Directory containing training data')
    parser.add_argument('--checkpoint_dir', type=str, default='results/models/multimodal',
                       help='Directory to save model checkpoints')
    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Override with command line args
    config['data_dir'] = args.data_dir
    config['checkpoint_dir'] = args.checkpoint_dir

    # Train model
    model = train_multimodal_model(config)


if __name__ == "__main__":
    main()