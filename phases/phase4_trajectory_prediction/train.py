"""
Trajectory Prediction Training

Trains trajectory prediction models for satellite collision avoidance,
including LSTM, Transformer, and physics-informed neural networks.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import json
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

from phases.phase4_trajectory_prediction.models import TrajectoryLSTM, TrajectoryTransformer
from utils.data_loader import SatelliteTrajectoryDataset
from phases.phase4_trajectory_prediction.evaluate import evaluate_trajectory_model

class TrajectoryTrainer:
    """
    Trainer for trajectory prediction models.
    """

    def __init__(self, model_type: str = 'transformer', config: Optional[Dict] = None):
        """
        Initialize trainer.

        Args:
            model_type: Type of model ('lstm', 'transformer')
            config: Training configuration
        """
        self.model_type = model_type
        self.config = config or self._default_config()
        self.logger = logging.getLogger(__name__)

        # Initialize model
        self.model = self._build_model()
        self.device = torch.device(self.config['device'])
        self.model.to(self.device)

        # Initialize optimizer and loss
        self.optimizer = self._build_optimizer()
        self.criterion = self._build_loss()

        # Training state
        self.epoch = 0
        self.best_loss = float('inf')
        self.training_history = []

    def _default_config(self) -> Dict:
        """Default training configuration."""
        return {
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'batch_size': 32,
            'seq_length': 50,
            'input_dim': 6,  # position + velocity
            'hidden_dim': 128,
            'num_layers': 2,
            'output_dim': 6,
            'learning_rate': 1e-3,
            'weight_decay': 1e-4,
            'num_epochs': 100,
            'patience': 10,
            'save_dir': './checkpoints',
            'log_interval': 10
        }

    def _build_model(self):
        """Build the trajectory prediction model."""
        if self.model_type == 'lstm':
            return TrajectoryLSTM(
                input_dim=self.config['input_dim'],
                hidden_dim=self.config['hidden_dim'],
                num_layers=self.config['num_layers'],
                output_dim=self.config['output_dim']
            )
        elif self.model_type == 'transformer':
            return TrajectoryTransformer(
                input_dim=self.config['input_dim'],
                hidden_dim=self.config['hidden_dim'],
                num_layers=self.config['num_layers'],
                output_dim=self.config['output_dim'],
                num_heads=8,
                dropout=0.1
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def _build_optimizer(self):
        """Build optimizer."""
        return optim.Adam(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )

    def _build_loss(self):
        """Build loss function."""
        return nn.MSELoss()

    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {self.epoch+1}")

        for batch in progress_bar:
            # Move to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(self.device)

            # Forward pass
            self.optimizer.zero_grad()

            # Prepare input/output
            input_seq = batch['trajectory'][:, :-1]  # All but last timestep
            target_seq = batch['trajectory'][:, 1:]   # All but first timestep

            # Model prediction
            predictions = self.model(input_seq)

            # Calculate loss
            loss = self.criterion(predictions, target_seq)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            # Update progress bar
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / num_batches
        return avg_loss

    def validate(self, val_loader: DataLoader) -> float:
        """Validate model."""
        self.model.eval()
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                # Move to device
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(self.device)

                # Prepare input/output
                input_seq = batch['trajectory'][:, :-1]
                target_seq = batch['trajectory'][:, 1:]

                # Model prediction
                predictions = self.model(input_seq)

                # Calculate loss
                loss = self.criterion(predictions, target_seq)

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches
        return avg_loss

    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None,
              save_path: Optional[str] = None):
        """
        Train the model.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            save_path: Path to save best model
        """
        self.logger.info(f"Starting training for {self.config['num_epochs']} epochs")

        patience_counter = 0
        best_val_loss = float('inf')

        for epoch in range(self.config['num_epochs']):
            self.epoch = epoch

            # Train epoch
            start_time = time.time()
            train_loss = self.train_epoch(train_loader)
            epoch_time = time.time() - start_time

            # Validate
            val_loss = self.validate(val_loader) if val_loader else train_loss

            # Log progress
            if (epoch + 1) % self.config['log_interval'] == 0:
                self.logger.info(
                    f"Epoch {epoch+1}/{self.config['num_epochs']} - "
                    f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                    f"Time: {epoch_time:.2f}s"
                )

            # Save training history
            self.training_history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'time': epoch_time
            })

            # Early stopping and model saving
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0

                if save_path:
                    self.save_model(save_path)
                    self.logger.info(f"Saved best model to {save_path}")
            else:
                patience_counter += 1

            if patience_counter >= self.config['patience']:
                self.logger.info(f"Early stopping at epoch {epoch+1}")
                break

        self.logger.info("Training completed")

    def save_model(self, path: str):
        """Save model checkpoint."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': self.epoch,
            'config': self.config,
            'training_history': self.training_history,
            'best_loss': self.best_loss
        }

        torch.save(checkpoint, path)

    def load_model(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.config = checkpoint['config']
        self.training_history = checkpoint.get('training_history', [])

    def plot_training_history(self, save_path: Optional[str] = None):
        """Plot training history."""
        if not self.training_history:
            return

        epochs = [h['epoch'] for h in self.training_history]
        train_losses = [h['train_loss'] for h in self.training_history]
        val_losses = [h['val_loss'] for h in self.training_history]

        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
        plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        plt.plot(epochs, val_losses, 'r-', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Validation Loss')
        plt.title('Validation Loss (Log Scale)')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)

        plt.suptitle(f'{self.model_type.upper()} Training History', fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def create_trajectory_dataset(n_samples: int = 1000, seq_length: int = 50,
                            noise_level: float = 0.01) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create synthetic trajectory dataset for training.

    Args:
        n_samples: Number of trajectory samples
        seq_length: Length of each trajectory
        noise_level: Amount of noise to add

    Returns:
        Tuple of (trajectories, labels) - labels are next timestep predictions
    """
    trajectories = []

    for _ in range(n_samples):
        # Generate orbital trajectory (simplified Keplerian orbit)
        t = np.linspace(0, 4*np.pi, seq_length)  # orbital period

        # Orbital elements (randomized)
        a = np.random.uniform(6671, 42164)  # semi-major axis (LEO to GEO)
        e = np.random.uniform(0, 0.1)       # eccentricity
        i = np.random.uniform(0, np.pi)     # inclination
        omega = np.random.uniform(0, 2*np.pi)  # argument of perigee
        Omega = np.random.uniform(0, 2*np.pi)  # RAAN

        # Simplified orbital position calculation
        r = a * (1 - e * np.cos(t))
        x = r * (np.cos(Omega) * np.cos(omega + t) - np.sin(Omega) * np.sin(omega + t) * np.cos(i))
        y = r * (np.sin(Omega) * np.cos(omega + t) + np.cos(Omega) * np.sin(omega + t) * np.cos(i))
        z = r * np.sin(omega + t) * np.sin(i)

        # Velocity (simplified)
        vx = -a * e * np.sin(t) / np.sqrt(a * (1 - e**2))
        vy = a * (1 + e * np.cos(t)) / np.sqrt(a * (1 - e**2))
        vz = np.zeros_like(t)  # simplified

        # Combine position and velocity
        trajectory = np.column_stack([x, y, z, vx, vy, vz])

        # Add noise
        trajectory += np.random.normal(0, noise_level, trajectory.shape)

        trajectories.append(trajectory)

    trajectories = np.array(trajectories)
    trajectories = torch.tensor(trajectories, dtype=torch.float32)

    return trajectories


def train_trajectory_model(model_type: str = 'transformer',
                          save_dir: str = './checkpoints',
                          **kwargs):
    """
    Complete training pipeline for trajectory prediction.

    Args:
        model_type: Type of model to train
        save_dir: Directory to save checkpoints
        **kwargs: Additional training arguments
    """
    # Setup logging
    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Create save directory
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Create synthetic dataset
    print("Creating synthetic trajectory dataset...")
    trajectories = create_trajectory_dataset(n_samples=2000, seq_length=50)

    # Create dataset and dataloaders
    dataset = TensorDataset(trajectories)
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [0.8, 0.2], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Initialize trainer
    trainer = TrajectoryTrainer(model_type=model_type)

    # Update config with kwargs
    trainer.config.update(kwargs)

    # Train model
    checkpoint_path = save_dir / f'{model_type}_trajectory_model.pth'
    trainer.train(train_loader, val_loader, save_path=str(checkpoint_path))

    # Plot training history
    trainer.plot_training_history(save_path=str(save_dir / 'training_history.png'))

    # Evaluate final model
    print("Evaluating final model...")
    results = evaluate_trajectory_model(trainer.model, val_loader,
                                      device=trainer.device,
                                      save_dir=str(save_dir))

    # Save results
    with open(save_dir / 'evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("Training completed!")
    print(f"Best model saved to: {checkpoint_path}")
    print(f"Results saved to: {save_dir}")

    return trainer, results


if __name__ == "__main__":
    # Train both LSTM and Transformer models
    models_to_train = ['lstm', 'transformer']

    for model_type in models_to_train:
        print(f"\n{'='*50}")
        print(f"Training {model_type.upper()} Trajectory Model")
        print(f"{'='*50}")

        try:
            trainer, results = train_trajectory_model(
                model_type=model_type,
                save_dir=f'./trajectory_checkpoints/{model_type}',
                num_epochs=50,
                batch_size=32,
                learning_rate=1e-3
            )

            print(f"\n{model_type.upper()} Results:")
            for metric, value in results.items():
                print(f"  {metric}: {value:.4f}")

        except Exception as e:
            print(f"Error training {model_type}: {e}")
            continue