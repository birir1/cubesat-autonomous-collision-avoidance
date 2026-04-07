import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import logging

from models.gnn.satellite_gnn import SatelliteGNN
from core.metrics import evaluate_model_predictions

logger = logging.getLogger(__name__)


class GNNCollisionTrainer:
    def __init__(self, lr=1e-3):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = SatelliteGNN().to(self.device)

        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0.0

        preds, targets = [], []

        for batch in loader:
            pos = batch['positions'].to(self.device)   # (B, N, 3)
            vel = batch['velocities'].to(self.device)  # (B, N, 3)
            y = batch['target'].to(self.device)        # (B,)

            self.optimizer.zero_grad()

            embeddings = self.model(pos, vel)  # (B, N, D)

            # GLOBAL POOLING
            pooled = embeddings.mean(dim=1)  # (B, D)

            logits = pooled.mean(dim=1, keepdim=True)  # (B, 1)

            loss = self.criterion(logits.squeeze(), y)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            preds.append(torch.sigmoid(logits).detach().cpu().numpy())
            targets.append(y.cpu().numpy())

        preds = np.concatenate(preds).flatten()
        targets = np.concatenate(targets).flatten()

        return total_loss / len(loader), preds, targets

    def validate(self, loader):
        self.model.eval()
        preds, targets = [], []

        with torch.no_grad():
            for batch in loader:
                pos = batch['positions'].to(self.device)
                vel = batch['velocities'].to(self.device)
                y = batch['target'].to(self.device)

                embeddings = self.model(pos, vel)
                pooled = embeddings.mean(dim=1)
                logits = pooled.mean(dim=1, keepdim=True)

                preds.append(torch.sigmoid(logits).cpu().numpy())
                targets.append(y.cpu().numpy())

        preds = np.concatenate(preds).flatten()
        targets = np.concatenate(targets).flatten()

        return preds, targets


def train_gnn(train_loader, val_loader, test_loader, save_dir="results/metrics"):
    os.makedirs(save_dir, exist_ok=True)

    trainer = GNNCollisionTrainer()

    best_auc = 0.0

    for epoch in range(10):
        loss, preds, targets = trainer.train_epoch(train_loader)

        val_preds, val_targets = trainer.validate(val_loader)

        metrics = evaluate_model_predictions(
            y_true=val_targets,
            y_pred=val_preds,
            name="GNN"
        )

        val_auc = metrics['classification_metrics']['roc_auc']

        logger.info(f"[GNN] Epoch {epoch+1}: Loss={loss:.4f}, Val AUC={val_auc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(trainer.model.state_dict(), "results/models/gnn_best.pt")

    # -----------------------------
    # TEST EVALUATION
    # -----------------------------
    test_preds, test_targets = trainer.validate(test_loader)

    final_metrics = evaluate_model_predictions(
        y_true=test_targets,
        y_pred=test_preds,
        name="GNN"
    )

    # SAVE CSV 🔥
    metrics_path = os.path.join(save_dir, "gnn_collision_metrics.csv")

    with open(metrics_path, "w") as f:
        f.write("metric,value\n")
        f.write(f"roc_auc,{final_metrics['classification_metrics']['roc_auc']}\n")
        f.write(f"pr_auc,{final_metrics['classification_metrics']['pr_auc']}\n")
        f.write(f"recall,{final_metrics['safety_metrics']['recall']}\n")
        f.write(f"false_alarm_rate,{final_metrics['safety_metrics']['false_alarm_rate']}\n")

    logger.info(f"✅ GNN metrics saved to {metrics_path}")

    return final_metrics