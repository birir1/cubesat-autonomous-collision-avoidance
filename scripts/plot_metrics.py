"""
plot_metrics.py
Generate plots and tables for CubeSat Collision Risk paper
- Training/Validation loss curves
- GNN risk distribution
- Confusion matrix / metrics table
- Saves all figures to results/figures and tables
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------
# Ensure output folders
# -------------------------
os.makedirs("results/figures", exist_ok=True)
os.makedirs("results/tables", exist_ok=True)

# -------------------------
# 1. TRAIN/VAL LOSS CURVE
# -------------------------
loss_file = "results/metrics/train_val_loss.csv"
if os.path.exists(loss_file):
    df = pd.read_csv(loss_file)
    plt.figure(figsize=(6,4))
    if 'train_loss' in df.columns:
        plt.plot(df['epoch'], df['train_loss'], label='Train Loss', marker='o')
    if 'val_auc' in df.columns:
        plt.plot(df['epoch'], df['val_auc'], label='Validation AUC', marker='x')
    plt.xlabel("Epoch")
    plt.ylabel("Loss / AUC")
    plt.title("Multimodal Transformer Training")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/figures/train_val_curve.png")
    plt.close()
    print("✅ Saved train/val loss curve")
else:
    print(f"⚠️ File not found: {loss_file}")

# -------------------------
# 2. GNN METRICS
# -------------------------
metrics_file = "results/metrics/gnn_collision_metrics.csv"
if os.path.exists(metrics_file):
    gnn_df = pd.read_csv(metrics_file)
    
    # 2a. RISK DISTRIBUTION PLOT
    if 'risk_score' in gnn_df.columns:
        plt.figure(figsize=(6,4))
        sns.histplot(gnn_df['risk_score'], bins=20, kde=True)
        plt.xlabel("Predicted Risk")
        plt.ylabel("Count")
        plt.title("GNN Predicted Risk Distribution")
        plt.tight_layout()
        plt.savefig("results/figures/gnn_risk_distribution.png")
        plt.close()
        print("✅ Saved GNN risk distribution")

    # 2b. METRICS TABLE (if exists)
    if {'metric','value'}.issubset(gnn_df.columns):
        metrics_table = gnn_df[['metric','value']]
        metrics_table.to_csv("results/tables/gnn_metrics_table.csv", index=False)
        print("✅ Saved GNN metrics table")
    else:
        # fallback: save numeric summary
        numeric_cols = gnn_df.select_dtypes(include=['number']).columns.tolist()
        summary_df = gnn_df[numeric_cols].describe().transpose()
        summary_df.to_csv("results/tables/gnn_metrics_table.csv")
        print("⚠️ 'metric/value' not found. Saved numeric summary instead.")

    # 2c. CONFUSION MATRIX
    if {'y_true','y_pred'}.issubset(gnn_df.columns):
        from sklearn.metrics import confusion_matrix
        import numpy as np
        cm = confusion_matrix(gnn_df['y_true'].round(), gnn_df['y_pred'].round())
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
        print("⚠️ y_true/y_pred not found, skipping confusion matrix")

    # 2d. RISK CATEGORY DISTRIBUTION
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
        print("⚠️ risk_category column not found, skipping risk category plot")
else:
    print(f"⚠️ File not found: {metrics_file}")

print("✅ All plots and tables generation completed!")