"""
Evaluation Script for Multimodal Collision Risk Predictor

This script evaluates the trained multimodal model on test data,
comparing it against baseline methods and computing safety-critical metrics.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc, roc_curve,
    confusion_matrix, classification_report
)
import yaml
import argparse
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from scripts.train_multimodal import MultimodalSatelliteDataset
from models.multimodal.multimodal_predictor import MultimodalCollisionPredictor
from core.metrics import safety_metrics


def load_model(checkpoint_path, config):
    """
    Load trained multimodal model from checkpoint.
    """
    model = MultimodalCollisionPredictor(
        trajectory_config=config['trajectory_config'],
        gnn_config=config['gnn_config'],
        vision_config=config['vision_config'],
        fusion_dim=config['fusion_dim'],
        dropout=config['dropout']
    )

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model


def evaluate_multimodal_model(model, test_loader, device, config):
    """
    Evaluate the multimodal model on test data.
    """
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []
    all_features = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            trajectory = batch['trajectory'].to(device)
            positions = batch['positions'].to(device)
            velocities = batch['velocities'].to(device)
            labels = batch['label'].to(device)
            images = batch['images']
            if isinstance(images, torch.Tensor) and images.numel() == 0:
                images = None

            risk_pred, features, _ = model(
                trajectory_sequence=trajectory,
                positions=positions,
                velocities=velocities,
                images=images
            )

            all_preds.extend(risk_pred.squeeze().cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_features.append(features)

    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    return all_preds, all_labels, all_features


def compute_safety_metrics(predictions, labels, thresholds=None):
    # Convert continuous labels to binary (high risk = 1)
    labels_binary = (labels >= 0.5).astype(int)
    """
    Compute safety-critical metrics for collision prediction.
    """
    if thresholds is None:
        thresholds = np.linspace(0.1, 0.9, 9)

    results = []

    for threshold in thresholds:
        pred_binary = (predictions >= threshold).astype(int)

        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(labels_binary, pred_binary).ravel()

        # Safety metrics
        collision_detection_rate = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall
        false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0  # False positive rate
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0

        results.append({
            'threshold': threshold,
            'collision_detection_rate': collision_detection_rate,
            'false_alarm_rate': false_alarm_rate,
            'precision': precision,
            'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn
        })

    return pd.DataFrame(results)


def plot_evaluation_results(predictions, labels, safety_metrics, save_dir):
    """
    Create evaluation plots and save results.
    """
    os.makedirs(save_dir, exist_ok=True)

    # ROC Curve
    plt.figure(figsize=(10, 8))

    # ROC AUC
    labels_binary_plot = (labels >= 0.5).astype(int)
    fpr, tpr, _ = roc_curve(labels_binary_plot, predictions)
    roc_auc = auc(fpr, tpr)

    plt.subplot(2, 2, 1)
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True)

    # Precision-Recall Curve
    plt.subplot(2, 2, 2)
    precision, recall, _ = precision_recall_curve(labels_binary_plot, predictions)
    pr_auc = auc(recall, precision)

    plt.plot(recall, precision, label=f'PR curve (AUC = {pr_auc:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True)

    # Safety Metrics
    plt.subplot(2, 2, 3)
    plt.plot(safety_metrics['threshold'], safety_metrics['collision_detection_rate'],
             label='Collision Detection Rate', marker='o')
    plt.plot(safety_metrics['threshold'], safety_metrics['false_alarm_rate'],
             label='False Alarm Rate', marker='s')
    plt.xlabel('Threshold')
    plt.ylabel('Rate')
    plt.title('Safety Metrics vs Threshold')
    plt.legend()
    plt.grid(True)

    # Prediction Distribution
    plt.subplot(2, 2, 4)
    plt.hist(predictions[labels == 0], alpha=0.7, label='No Collision', bins=50)
    plt.hist(predictions[labels == 1], alpha=0.7, label='Collision', bins=50)
    plt.xlabel('Predicted Risk')
    plt.ylabel('Count')
    plt.title('Prediction Distribution')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'multimodal_evaluation.png'), dpi=300, bbox_inches='tight')
    plt.show()

    # Save metrics to CSV
    safety_metrics.to_csv(os.path.join(save_dir, 'safety_metrics.csv'), index=False)

    # Save summary statistics
    summary = {
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'mean_collision_detection_rate': safety_metrics['collision_detection_rate'].mean(),
        'mean_false_alarm_rate': safety_metrics['false_alarm_rate'].mean(),
        'optimal_threshold': safety_metrics.loc[
            safety_metrics['collision_detection_rate'].idxmax(), 'threshold'
        ]
    }

    pd.DataFrame([summary]).to_csv(os.path.join(save_dir, 'evaluation_summary.csv'), index=False)

    return summary


def compare_with_baselines(predictions, labels, baseline_results, save_dir):
    """
    Compare multimodal model with baseline methods.
    """
    # Load baseline results
    baselines = {}
    for baseline_name, baseline_file in baseline_results.items():
        if os.path.exists(baseline_file):
            baseline_data = pd.read_csv(baseline_file)
            baselines[baseline_name] = baseline_data

    # Create comparison plot
    plt.figure(figsize=(12, 8))

    # ROC comparison
    plt.subplot(2, 2, 1)
    labels_binary_plot = (labels >= 0.5).astype(int)
    fpr, tpr, _ = roc_curve(labels_binary_plot, predictions)
    multimodal_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'Multimodal (AUC = {multimodal_auc:.3f})', linewidth=2)

    for name, data in baselines.items():
        if 'fpr' in data.columns and 'tpr' in data.columns:
            plt.plot(data['fpr'], data['tpr'],
                    label=f'{name} (AUC = {auc(data["fpr"], data["tpr"]):.3f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Comparison')
    plt.legend()
    plt.grid(True)

    # Safety metrics comparison
    plt.subplot(2, 2, 2)
    cdr_multimodal = safety_metrics['collision_detection_rate'].mean()
    far_multimodal = safety_metrics['false_alarm_rate'].mean()

    plt.scatter(far_multimodal, cdr_multimodal, s=100, label='Multimodal', marker='*')

    for name, data in baselines.items():
        if 'false_alarm_rate' in data.columns and 'collision_detection_rate' in data.columns:
            plt.scatter(data['false_alarm_rate'].mean(), data['collision_detection_rate'].mean(),
                       label=name, alpha=0.7)

    plt.xlabel('False Alarm Rate')
    plt.ylabel('Collision Detection Rate')
    plt.title('Safety Metrics Comparison')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'baseline_comparison.png'), dpi=300, bbox_inches='tight')
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Evaluate Multimodal Collision Predictor')
    parser.add_argument('--config', type=str, default='configs/multimodal_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, default='results/models/multimodal/best_multimodal_model.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--test_data', type=str, default='data/synthetic',
                       help='Directory containing test data')
    parser.add_argument('--output_dir', type=str, default='results/evaluation/multimodal',
                       help='Directory to save evaluation results')
    parser.add_argument('--baseline_results', type=str, nargs='*',
                       help='Paths to baseline result CSV files')
    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Create test dataset
    test_dataset = MultimodalSatelliteDataset(
        args.test_data, mode='test',
        sequence_length=config['sequence_length'],
        num_satellites=config['num_satellites']
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers']
    )

    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(args.checkpoint, config)

    # Evaluate model
    predictions, labels, features = evaluate_multimodal_model(
        model, test_loader, device, config
    )

    # Compute safety metrics
    safety_metrics = compute_safety_metrics(predictions, labels)

    # Create evaluation plots and save results
    summary = plot_evaluation_results(predictions, labels, safety_metrics, args.output_dir)

    # Compare with baselines if provided
    if args.baseline_results:
        baseline_dict = {}
        for i, baseline_path in enumerate(args.baseline_results):
            name = f'Baseline_{i+1}'
            baseline_dict[name] = baseline_path

        compare_with_baselines(predictions, labels, baseline_dict, args.output_dir)

    # Print summary
    print("\n" + "="*50)
    print("MULTIMODAL MODEL EVALUATION SUMMARY")
    print("="*50)
    print(f"ROC AUC: {summary['roc_auc']:.4f}")
    print(f"PR AUC: {summary['pr_auc']:.4f}")
    print(f"Mean Collision Detection Rate: {summary['mean_collision_detection_rate']:.4f}")
    print(f"Mean False Alarm Rate: {summary['mean_false_alarm_rate']:.4f}")
    print(f"Optimal Threshold: {summary['optimal_threshold']:.2f}")
    print("="*50)

    print(f"\nResults saved to: {args.output_dir}")


if __name__ == "__main__":
    main()