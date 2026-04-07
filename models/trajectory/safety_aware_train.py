"""
Safety-Aware Training for Trajectory Transformer (FIXED + RESEARCH-GRADE)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging

from models.trajectory.safety_aware_transformer import (
    SafetyAwareTrajectoryTransformer,
    SafetyAwareLoss
)

from core.metrics import evaluate_model_predictions

logger = logging.getLogger(__name__)


# =========================================================
# TRAINER CLASS
# =========================================================
class SafetyAwareTrajectoryTrainer:

    def __init__(self, model, lr=1e-4):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)

        self.criterion = SafetyAwareLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr)

    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0.0
        preds, targets = [], []

        for batch in loader:
            x = batch['features'].float().to(self.device)
            y = batch['target'].float().to(self.device)

            x = x.permute(1, 0, 2)

            self.optimizer.zero_grad()

            risk_pred, danger_logits = self.model(x)

            loss_dict = self.criterion(risk_pred, danger_logits, y)
            loss = loss_dict['total_loss']

            loss.backward()

            # 🔥 Gradient clipping (stability)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            self.optimizer.step()

            total_loss += loss.item()

            # For RMSE (regression head)
            preds.append(risk_pred.detach().cpu().numpy())
            targets.append(y.detach().cpu().numpy())

        preds = np.concatenate(preds).flatten()
        targets = np.concatenate(targets).flatten()

        rmse = np.sqrt(np.mean((preds - targets) ** 2))

        return total_loss / len(loader), rmse

    def validate(self, loader):
        self.model.eval()
        total_loss = 0.0
        preds, targets = [], []

        with torch.no_grad():
            for batch in loader:
                x = batch['features'].float().to(self.device)
                y = batch['target'].float().to(self.device)

                x = x.permute(1, 0, 2)

                risk_pred, danger_logits = self.model(x)

                loss_dict = self.criterion(risk_pred, danger_logits, y)
                total_loss += loss_dict['total_loss'].item()

                preds.append(risk_pred.cpu().numpy())
                targets.append(y.cpu().numpy())

        preds = np.concatenate(preds).flatten()
        targets = np.concatenate(targets).flatten()

        rmse = np.sqrt(np.mean((preds - targets) ** 2))

        return total_loss / len(loader), rmse

    def train(self, train_loader, val_loader, epochs=50, save_path=None):
        best_loss = float('inf')
        best_path = None

        for epoch in range(epochs):
            train_loss, train_rmse = self.train_epoch(train_loader)
            val_loss, val_rmse = self.validate(val_loader)

            logger.info(
                f"Epoch {epoch+1}: Train={train_loss:.4f}, "
                f"Val={val_loss:.4f}, RMSE={val_rmse:.4f}"
            )

            if val_loss < best_loss:
                best_loss = val_loss
                best_path = save_path

                if save_path is not None:
                    torch.save(self.model.state_dict(), save_path)

        return {
            "best_val_loss": best_loss,
            "best_model_path": best_path
        }


# =========================================================
# TRAIN FUNCTION
# =========================================================
def train_safety_aware_trajectory_transformer(
        train_loader,
        val_loader,
        test_loader,
        model_save_path: str,
        num_epochs: int = 50):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # -----------------------------
    # MODEL INIT
    # -----------------------------
    sample = next(iter(train_loader))
    input_dim = sample['features'].shape[-1]

    model = SafetyAwareTrajectoryTransformer(input_dim=input_dim)
    trainer = SafetyAwareTrajectoryTrainer(model)

    # -----------------------------
    # TRAIN
    # -----------------------------
    results = trainer.train(
        train_loader,
        val_loader,
        epochs=num_epochs,
        save_path=model_save_path
    )

    # -----------------------------
    # LOAD BEST MODEL
    # -----------------------------
    if model_save_path is not None:
        model.load_state_dict(
            torch.load(model_save_path, map_location=device)
        )

    model.to(device)
    model.eval()

    # -----------------------------
    # TEST EVALUATION (FIXED)
    # -----------------------------
    risk_preds, cls_preds, targets = [], [], []

    with torch.no_grad():
        for batch in test_loader:
            x = batch['features'].float().to(device)
            y = batch['target'].float().to(device)

            x = x.permute(1, 0, 2)

            risk_pred, danger_logits = model(x)

            # 🔥 Regression head
            risk_preds.append(risk_pred.cpu().numpy())

            # 🔥 Classification head (FIXED)
            probs = torch.sigmoid(danger_logits)
            cls_preds.append(probs.cpu().numpy())

            targets.append(y.cpu().numpy())

    risk_preds = np.concatenate(risk_preds).flatten()
    cls_preds = np.concatenate(cls_preds).flatten()
    targets = np.concatenate(targets).flatten()

    # 🔍 DEBUG (VERY IMPORTANT)
    print("\n[DEBUG] Prediction stats:")
    print(f"Risk mean={risk_preds.mean():.4f}, min={risk_preds.min():.4f}, max={risk_preds.max():.4f}")
    print(f"Cls  mean={cls_preds.mean():.4f}, min={cls_preds.min():.4f}, max={cls_preds.max():.4f}")

    # -----------------------------
    # FULL METRICS (USE CLASSIFICATION HEAD)
    # -----------------------------
    eval_metrics = evaluate_model_predictions(
        y_true=targets,
        y_pred=cls_preds,   # 🔥 FIXED
        name="SafetyAwareTransformer"
    )

    rmse = np.sqrt(np.mean((risk_preds - targets) ** 2))
    fnr = eval_metrics['safety_metrics']['false_negative_rate']
    recall = eval_metrics['safety_metrics']['recall']

    # Store results
    results['test_rmse'] = rmse
    results['test_metrics'] = eval_metrics

    # -----------------------------
    # LOGGING
    # -----------------------------
    logger.info(
        f"[TEST] RMSE={rmse:.4f}, "
        f"FNR={fnr:.4f}, "
        f"Recall={recall:.4f}"
    )

    return results