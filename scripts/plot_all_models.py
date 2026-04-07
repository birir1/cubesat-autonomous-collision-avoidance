"""
plot_all_models.py
Generate plots and statistical analysis for CubeSat Collision Risk paper
- Multimodal Transformer: Train/Val Loss
- GNN: Risk distribution, confusion matrix, risk categories
- Saves figures to results/figures and tables to results/tables
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix

# -------------------------
# Ensure output folders
# -------------------------
os.makedirs("results/figures", exist_ok=True)
os.makedirs("results/tables", exist_ok=True)

# -------------------------
# 1. MULTIMODAL TRANSFORMER TRAIN/VAL LOSS CURVE
# -------------------------
train_val_file = "results/metrics/train_val_loss.csv"
if os.path.exists(train_val_file):
    df = pd.read_csv(train_val_file)
    plt.figure(figsize=(6,4))
    plt.plot(df['epoch'], df['train_loss'], label='Train Loss', marker='o')
    plt.plot(df['epoch'], df['val_auc'], label='Val AUC', marker='x')
    plt.xlabel("Epoch")
    plt.ylabel("Loss / AUC")
    plt.title("Multimodal Transformer Training")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/figures/multimodal_train_val_curve.png")
    plt.close()
    print("✅ Saved Multimodal Transformer train/val curve")
else:
    print(f"⚠️ File not found: {train_val_file}")

# -------------------------
# 2. GNN RISK DISTRIBUTION & METRICS TABLE
# -------------------------
gnn_file = "results/metrics/gnn_collision_metrics.csv"
if os.path.exists(gnn_file):
    gnn_df = pd.read_csv(gnn_file)

    # Risk distribution
    if 'y_pred' in gnn_df.columns:
        plt.figure(figsize=(6,4))
        sns.histplot(gnn_df['y_pred'], bins=20, kde=True)
        plt.xlabel("Predicted Risk")
        plt.ylabel("Count")
        plt.title("GNN Predicted Risk Distribution")
        plt.tight_layout()
        plt.savefig("results/figures/gnn_risk_distribution.png")
        plt.close()
        print("✅ Saved GNN predicted risk distribution")

    # Summary statistics table
    numeric_cols = gnn_df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        summary = gnn_df[numeric_cols].agg(['mean', 'std', 'min', 'max'])
        summary.to_csv("results/tables/gnn_metrics_summary.csv")
        print("✅ Saved GNN metrics summary table")
    else:
        # fallback if no numeric columns
        gnn_df.describe().to_csv("results/tables/gnn_metrics_summary.csv")
        print("⚠️ No numeric columns found; saved generic summary")

    # Confusion matrix
    if {'y_true','y_pred'}.issubset(gnn_df.columns):
        y_true = gnn_df['y_true'].round().astype(int)
        y_pred = gnn_df['y_pred'].round().astype(int)
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("GNN Confusion Matrix")
        plt.tight_layout()
        plt.savefig("results/figures/gnn_confusion_matrix.png")
        plt.close()
        print("✅ Saved GNN confusion matrix")
    else:
        print("⚠️ y_true/y_pred not found; skipping confusion matrix")

    # Risk categories
    if 'risk_category' in gnn_df.columns:
        plt.figure(figsize=(5,4))
        sns.countplot(x='risk_category', data=gnn_df, order=['high','medium','low'])
        plt.title("GNN Risk Category Counts")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig("results/figures/gnn_risk_category_counts.png")
        plt.close()
        print("✅ Saved GNN risk category counts")
    else:
        print("⚠️ risk_category column not found; skipping risk category plot")

else:
    print(f"⚠️ File not found: {gnn_file}")

print("✅ All plots and statistical analysis completed!")