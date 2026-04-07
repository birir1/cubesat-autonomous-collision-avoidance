"""
Training Script for Trajectory Transformer Model

This module provides training functionality for the trajectory transformer
model used in satellite collision avoidance.
"""

import sys
import os
from pathlib import Path
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, Any, Optional, Tuple
import json
from datetime import datetime

# Handle imports for both direct execution and module import
try:
    from .transformer import TrajectoryTransformer, TrajectoryTransformerPredictor
    from ...core.dataset import create_data_loaders
    from ...core.metrics import evaluate_model_predictions
except ImportError:
    # Fallback for direct execution
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from models.trajectory.transformer import TrajectoryTransformer, TrajectoryTransformerPredictor
    from core.dataset import create_data_loaders
    from core.metrics import evaluate_model_predictions

logger = logging.getLogger(__name__)


class TrajectoryTrainer:
    """
    Trainer class for the Trajectory Transformer model.
    """

    def __init__(self, model: TrajectoryTransformer, device: str = 'auto',
                 learning_rate: float = 1e-4, weight_decay: float = 1e-5):
        """
        Initialize the trainer.

        Args:
            model: TrajectoryTransformer model
            device: Device to train on ('auto', 'cpu', 'cuda')
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for regularization
        """
        self.model = model

        # Device configuration
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.model.to(self.device)

        # Optimizer and loss
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Learning rate scheduler
        # Torch versions may or may not support verbose flag
        try:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
            )
        except TypeError:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=5
            )

        # Loss function - binary cross entropy for collision risk
        self.criterion = nn.BCELoss()  # Since we use sigmoid in the model

        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_rmse': [],
            'val_rmse': [],
            'learning_rates': []
        }

    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader

        Returns:
            float: Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            # Get batch data
            features = batch['features'].to(self.device)  # [batch_size, seq_len, input_dim]
            targets = batch['target'].to(self.device)     # [batch_size, 1]

            # For transformer, we need [seq_len, batch_size, input_dim]
            features = features.permute(1, 0, 2)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(features)

            # Compute loss
            loss = self.criterion(outputs.squeeze(), targets.squeeze())

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches if num_batches > 0 else 0.0

    def validate(self, val_loader: DataLoader) -> Tuple[float, Dict[str, Any]]:
        """
        Validate the model.

        Args:
            val_loader: Validation data loader

        Returns:
            tuple: (validation_loss, metrics_dict)
        """
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        num_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                features = batch['features'].to(self.device)
                targets = batch['target'].to(self.device)
                raw_targets = batch['raw_target'].cpu().numpy()

                # Permute for transformer
                features = features.permute(1, 0, 2)

                # Forward pass
                outputs = self.model(features)

                # Compute loss
                loss = self.criterion(outputs.squeeze(), targets.squeeze())
                total_loss += loss.item()

                # Store predictions and targets
                predictions = outputs.squeeze().cpu().numpy()
                all_predictions.extend(predictions)
                all_targets.extend(raw_targets)

                num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)

        # Compute metrics
        metrics = evaluate_model_predictions(all_targets, all_predictions, "Trajectory Transformer")

        return avg_loss, metrics

    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              num_epochs: int = 100, patience: int = 10,
              save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Train the model.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Maximum number of epochs
            patience: Early stopping patience
            save_path: Path to save the best model

        Returns:
            dict: Training results and metrics
        """
        best_val_loss = float('inf')
        best_epoch = 0
        patience_counter = 0

        logger.info("Starting training...")
        logger.info(f"Training on device: {self.device}")
        logger.info(f"Number of parameters: {sum(p.numel() for p in self.model.parameters())}")

        for epoch in range(num_epochs):
            # Train epoch
            train_loss = self.train_epoch(train_loader)
            val_loss, val_metrics = self.validate(val_loader)

            # Update learning rate
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']

            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_rmse'].append(val_metrics['basic_metrics']['rmse'])  # Approximation
            self.history['val_rmse'].append(val_metrics['basic_metrics']['rmse'])
            self.history['learning_rates'].append(current_lr)

            # Logging
            logger.info(f"Epoch {epoch+1}/{num_epochs}: Train loss={train_loss:.4f}, Val loss={val_loss:.4f}, Val RMSE={val_metrics['basic_metrics']['rmse']:.4f}, LR={current_lr:.6f}")

            # Check for improvement
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0

                # Save best model
                if save_path:
                    predictor = TrajectoryTransformerPredictor()
                    predictor.model = self.model
                    predictor.save_model(save_path, self.optimizer, epoch, val_loss)
                    logger.info(f"Saved best model to {save_path}")
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

        # Load best model
        if save_path and os.path.exists(save_path):
            predictor = TrajectoryTransformerPredictor()
            predictor.load_model(save_path)
            self.model = predictor.model.to(self.device)

        # Final evaluation
        final_val_loss, final_metrics = self.validate(val_loader)

        results = {
            'best_epoch': best_epoch,
            'best_val_loss': best_val_loss,
            'final_val_loss': final_val_loss,
            'final_metrics': final_metrics,
            'training_history': self.history,
            'total_epochs': epoch + 1
        }

        logger.info("Training completed!")
        logger.info(f"Best epoch: {best_epoch}, Best val loss: {best_val_loss:.4f}, Final val loss: {final_val_loss:.4f}")

        return results

    def save_training_history(self, save_path: str):
        """Save training history to JSON file."""
        history_path = Path(save_path).with_suffix('.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        logger.info(f"Saved training history to {history_path}")


def train_trajectory_transformer(data_path: str, model_save_path: str,
                               num_epochs: int = 100, batch_size: int = 32,
                               learning_rate: float = 1e-4,
                               sequence_length: int = None) -> Dict[str, Any]:
    """
    High-level function to train a trajectory transformer model.

    Args:
        data_path: Path to the data directory
        model_save_path: Path to save the trained model
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        sequence_length: Optional sequence length for trajectory modeling

    Returns:
        dict: Training results
    """
    # Create data loaders
    train_loader, val_loader, test_loader, scalers = create_data_loaders(
        data_path, batch_size=batch_size, sequence_length=sequence_length
    )

    # Get input dimension from data
    sample_batch = next(iter(train_loader))
    input_dim = sample_batch['features'].shape[-1]

    # Create model
    model = TrajectoryTransformer(
        input_dim=input_dim,
        d_model=128,
        nhead=8,
        num_layers=6,
        dropout=0.1
    )

    # Create trainer
    trainer = TrajectoryTrainer(
        model=model,
        learning_rate=learning_rate
    )

    # Train model
    results = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        save_path=model_save_path
    )

    # Save training history
    trainer.save_training_history(model_save_path)

    # Final evaluation on test set
    predictor = TrajectoryTransformerPredictor(model_save_path)
    test_predictions = []
    test_targets = []

    predictor.model.eval()
    with torch.no_grad():
        for batch in test_loader:
            features = batch['features'].to(predictor.device)
            features = features.permute(1, 0, 2)
            predictions = predictor.model(features)
            test_predictions.extend(predictions.squeeze().cpu().numpy())
            test_targets.extend(batch['raw_target'].cpu().numpy())

    test_metrics = evaluate_model_predictions(
        np.array(test_targets), np.array(test_predictions), "Trajectory Transformer (Test)"
    )

    results['test_metrics'] = test_metrics

    return results


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description='Train Trajectory Transformer')
    parser.add_argument('--data_path', type=str, default='data',
                       help='Path to data directory')
    parser.add_argument('--save_path', type=str, default='models/trajectory/trajectory_transformer.pth',
                       help='Path to save trained model')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')

    args = parser.parse_args()

    # Train model
    results = train_trajectory_transformer(
        data_path=args.data_path,
        model_save_path=args.save_path,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )

    print("Training completed!")
    print(f"Best validation loss: {results['best_val_loss']:.4f}")
    print(f"Test RMSE: {results['test_metrics']['basic_metrics']['rmse']:.4f}")