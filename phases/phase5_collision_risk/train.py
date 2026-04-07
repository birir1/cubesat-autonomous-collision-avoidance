"""
Training Script for Collision Risk Assessment Models

Trains models to predict collision risk from satellite trajectories.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

from .dataset_builder import CollisionRiskDatasetBuilder
from .feature_engineering import CollisionRiskFeatureEngineer
from .models.fusion_model import CollisionRiskFusionModel
from .models.static_baseline import StaticCollisionRiskModel
from .models.transformer_risk import TransformerRiskModel

class CollisionRiskTrainer:
    """
    Trainer for collision risk assessment models.
    """

    def __init__(self, config: Dict):
        """
        Initialize trainer.

        Args:
            config: Training configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Using device: {self.device}")

        # Initialize components
        self.dataset_builder = CollisionRiskDatasetBuilder(config.get('dataset_config', {}))
        self.feature_engineer = CollisionRiskFeatureEngineer(config.get('feature_config', {}))

        # Initialize model
        self.model = self._initialize_model()
        self.model.to(self.device)

        # Initialize optimizer and loss
        self.optimizer = self._initialize_optimizer()
        self.criterion = self._initialize_criterion()

        # Initialize metrics tracking
        self.metrics_history = {
            'train_loss': [],
            'val_loss': [],
            'train_auc': [],
            'val_auc': [],
            'train_accuracy': [],
            'val_accuracy': []
        }

    def _initialize_model(self) -> nn.Module:
        """Initialize the model based on configuration."""
        model_type = self.config.get('model_type', 'fusion')

        if model_type == 'fusion':
            model = CollisionRiskFusionModel(self.config.get('model_config', {}))
        elif model_type == 'static':
            model = StaticCollisionRiskModel(self.config.get('model_config', {}))
        elif model_type == 'transformer':
            model = TransformerRiskModel(self.config.get('model_config', {}))
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        self.logger.info(f"Initialized {model_type} model")
        return model

    def _initialize_optimizer(self) -> optim.Optimizer:
        """Initialize optimizer."""
        optimizer_type = self.config.get('optimizer', 'adam')
        lr = self.config.get('learning_rate', 1e-3)
        weight_decay = self.config.get('weight_decay', 1e-4)

        if optimizer_type.lower() == 'adam':
            optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_type.lower() == 'adamw':
            optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_type.lower() == 'sgd':
            momentum = self.config.get('momentum', 0.9)
            optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")

        return optimizer

    def _initialize_criterion(self) -> nn.Module:
        """Initialize loss criterion."""
        loss_type = self.config.get('loss_type', 'bce')

        if loss_type == 'bce':
            # Binary Cross Entropy with logits
            pos_weight = torch.tensor([self.config.get('pos_weight', 10.0)]).to(self.device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        elif loss_type == 'focal':
            # Focal loss for imbalanced data
            criterion = FocalLoss(alpha=self.config.get('focal_alpha', 0.25),
                                gamma=self.config.get('focal_gamma', 2.0))
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

        return criterion

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_loss = 0.0
        predictions = []
        targets = []

        for batch in train_loader:
            # Move to device
            trajectories = batch['trajectory'].to(self.device)
            targets_batch = batch['collision_risk'].to(self.device)

            # Extract features if needed
            if hasattr(self.model, 'feature_engineering') and self.model.feature_engineering:
                features = self.feature_engineer.extract_batch_features(
                    trajectories[:, 0], trajectories[:, 1]  # Assuming batched trajectories
                ).to(self.device)
                inputs = features
            else:
                inputs = trajectories

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)

            # Calculate loss
            loss = self.criterion(outputs.squeeze(), targets_batch)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Accumulate metrics
            epoch_loss += loss.item()
            predictions.extend(torch.sigmoid(outputs).cpu().detach().numpy().flatten())
            targets.extend(targets_batch.cpu().numpy())

        # Calculate epoch metrics
        epoch_loss /= len(train_loader)
        metrics = self._calculate_metrics(np.array(predictions), np.array(targets))
        metrics['loss'] = epoch_loss

        return metrics

    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        val_loss = 0.0
        predictions = []
        targets = []

        with torch.no_grad():
            for batch in val_loader:
                # Move to device
                trajectories = batch['trajectory'].to(self.device)
                targets_batch = batch['collision_risk'].to(self.device)

                # Extract features if needed
                if hasattr(self.model, 'feature_engineering') and self.model.feature_engineering:
                    features = self.feature_engineer.extract_batch_features(
                        trajectories[:, 0], trajectories[:, 1]
                    ).to(self.device)
                    inputs = features
                else:
                    inputs = trajectories

                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs.squeeze(), targets_batch)

                # Accumulate metrics
                val_loss += loss.item()
                predictions.extend(torch.sigmoid(outputs).cpu().numpy().flatten())
                targets.extend(targets_batch.cpu().numpy())

        # Calculate validation metrics
        val_loss /= len(val_loader)
        metrics = self._calculate_metrics(np.array(predictions), np.array(targets))
        metrics['loss'] = val_loss

        return metrics

    def _calculate_metrics(self, predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        """Calculate classification metrics."""
        from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_curve, auc

        # Convert to binary predictions
        pred_binary = (predictions > 0.5).astype(int)

        # Calculate metrics
        accuracy = accuracy_score(targets, pred_binary)

        # AUC (only if both classes present)
        if len(np.unique(targets)) > 1:
            auc_score = roc_auc_score(targets, predictions)
        else:
            auc_score = 0.5  # Default for single class

        # Precision-Recall AUC
        precision, recall, _ = precision_recall_curve(targets, predictions)
        pr_auc = auc(recall, precision)

        return {
            'accuracy': accuracy,
            'auc': auc_score,
            'pr_auc': pr_auc
        }

    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              num_epochs: int, save_path: str = './checkpoints'):
        """
        Train the model.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs to train
            save_path: Path to save checkpoints
        """
        Path(save_path).mkdir(parents=True, exist_ok=True)
        best_val_auc = 0.0

        self.logger.info(f"Starting training for {num_epochs} epochs")

        for epoch in range(num_epochs):
            # Train epoch
            train_metrics = self.train_epoch(train_loader)

            # Validate
            val_metrics = self.validate(val_loader)

            # Log metrics
            self.logger.info(f"Epoch {epoch+1}/{num_epochs}")
            self.logger.info(f"Train - Loss: {train_metrics['loss']:.4f}, AUC: {train_metrics['auc']:.4f}")
            self.logger.info(f"Val - Loss: {val_metrics['loss']:.4f}, AUC: {val_metrics['auc']:.4f}")

            # Store metrics
            self.metrics_history['train_loss'].append(train_metrics['loss'])
            self.metrics_history['val_loss'].append(val_metrics['loss'])
            self.metrics_history['train_auc'].append(train_metrics['auc'])
            self.metrics_history['val_auc'].append(val_metrics['auc'])
            self.metrics_history['train_accuracy'].append(train_metrics['accuracy'])
            self.metrics_history['val_accuracy'].append(val_metrics['accuracy'])

            # Save best model
            if val_metrics['auc'] > best_val_auc:
                best_val_auc = val_metrics['auc']
                self.save_checkpoint(epoch, val_metrics, f"{save_path}/best_model.pth")

            # Save latest model
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch, val_metrics, f"{save_path}/model_epoch_{epoch+1}.pth")

        # Save final metrics
        self.save_metrics(f"{save_path}/training_metrics.json")

        self.logger.info("Training completed")

    def save_checkpoint(self, epoch: int, metrics: Dict, path: str):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config
        }
        torch.save(checkpoint, path)
        self.logger.info(f"Checkpoint saved to {path}")

    def save_metrics(self, path: str):
        """Save training metrics."""
        with open(path, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
        self.logger.info(f"Metrics saved to {path}")

    def plot_training_history(self, save_path: str = './plots'):
        """Plot training history."""
        Path(save_path).mkdir(parents=True, exist_ok=True)

        # Plot loss
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.plot(self.metrics_history['train_loss'], label='Train')
        plt.plot(self.metrics_history['val_loss'], label='Validation')
        plt.title('Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        # Plot AUC
        plt.subplot(1, 3, 2)
        plt.plot(self.metrics_history['train_auc'], label='Train')
        plt.plot(self.metrics_history['val_auc'], label='Validation')
        plt.title('AUC')
        plt.xlabel('Epoch')
        plt.ylabel('AUC')
        plt.legend()

        # Plot accuracy
        plt.subplot(1, 3, 3)
        plt.plot(self.metrics_history['train_accuracy'], label='Train')
        plt.plot(self.metrics_history['val_accuracy'], label='Validation')
        plt.title('Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.tight_layout()
        plt.savefig(f"{save_path}/training_history.png", dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"Training plots saved to {save_path}/training_history.png")


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train collision risk assessment model')
    parser.add_argument('--config', type=str, default='./configs/collision_risk_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--model_type', type=str, default='fusion',
                       choices=['fusion', 'static', 'transformer'],
                       help='Type of model to train')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--save_path', type=str, default='./results/collision_risk',
                       help='Path to save results')
    parser.add_argument('--use_gpu', action='store_true', help='Use GPU if available')

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Load configuration
    import yaml
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Override config with command line args
    config['model_type'] = args.model_type
    config['batch_size'] = args.batch_size
    config['num_epochs'] = args.num_epochs
    config['learning_rate'] = args.learning_rate
    config['save_path'] = args.save_path

    # Create dataset
    builder = CollisionRiskDatasetBuilder(config)
    dataset = builder.generate_synthetic_dataset(n_samples=5000)

    # Create data loaders
    train_loader, val_loader = builder.create_data_loaders(
        dataset, batch_size=args.batch_size, train_split=0.8
    )

    # Initialize trainer
    trainer = CollisionRiskTrainer(config)

    # Train model
    trainer.train(train_loader, val_loader, args.num_epochs, args.save_path)

    # Plot training history
    trainer.plot_training_history(args.save_path)

    print(f"Training completed. Results saved to {args.save_path}")


if __name__ == "__main__":
    main()