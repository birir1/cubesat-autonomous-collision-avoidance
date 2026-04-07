"""
Evaluation Script for Collision Risk Assessment Models

Evaluates trained models on collision risk prediction tasks.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (roc_curve, auc, precision_recall_curve, confusion_matrix,
                           classification_report, roc_auc_score, average_precision_score)
from sklearn.calibration import calibration_curve

from .dataset_builder import CollisionRiskDatasetBuilder
from .feature_engineering import CollisionRiskFeatureEngineer
from .models.fusion_model import CollisionRiskFusionModel
from .models.static_baseline import StaticCollisionRiskModel
from .models.transformer_risk import TransformerRiskModel

class CollisionRiskEvaluator:
    """
    Evaluator for collision risk assessment models.
    """

    def __init__(self, config: Dict):
        """
        Initialize evaluator.

        Args:
            config: Evaluation configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Using device: {self.device}")

        # Initialize components
        self.dataset_builder = CollisionRiskDatasetBuilder(config.get('dataset_config', {}))
        self.feature_engineer = CollisionRiskFeatureEngineer(config.get('feature_config', {}))

    def load_model(self, model_path: str, model_type: str) -> nn.Module:
        """
        Load trained model from checkpoint.

        Args:
            model_path: Path to model checkpoint
            model_type: Type of model

        Returns:
            Loaded model
        """
        # Initialize model
        if model_type == 'fusion':
            model = CollisionRiskFusionModel(self.config.get('model_config', {}))
        elif model_type == 'static':
            model = StaticCollisionRiskModel(self.config.get('model_config', {}))
        elif model_type == 'transformer':
            model = TransformerRiskModel(self.config.get('model_config', {}))
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()

        self.logger.info(f"Loaded {model_type} model from {model_path}")
        return model

    def evaluate_model(self, model: nn.Module, test_loader: DataLoader,
                      model_type: str) -> Dict[str, Any]:
        """
        Evaluate model on test data.

        Args:
            model: Trained model
            test_loader: Test data loader
            model_type: Type of model

        Returns:
            Dictionary of evaluation metrics
        """
        self.logger.info("Evaluating model...")

        all_predictions = []
        all_targets = []
        all_probabilities = []

        with torch.no_grad():
            for batch in test_loader:
                # Move to device
                trajectories = batch['trajectory'].to(self.device)
                targets = batch['collision_risk'].to(self.device)

                # Extract features if needed
                if hasattr(model, 'feature_engineering') and model.feature_engineering:
                    features = self.feature_engineer.extract_batch_features(
                        trajectories[:, 0], trajectories[:, 1]
                    ).to(self.device)
                    inputs = features
                else:
                    inputs = trajectories

                # Forward pass
                outputs = model(inputs)
                probabilities = torch.sigmoid(outputs).cpu().numpy().flatten()
                predictions = (probabilities > 0.5).astype(int)

                all_predictions.extend(predictions)
                all_probabilities.extend(probabilities)
                all_targets.extend(targets.cpu().numpy())

        # Convert to numpy arrays
        predictions = np.array(all_predictions)
        probabilities = np.array(all_probabilities)
        targets = np.array(all_targets)

        # Calculate comprehensive metrics
        metrics = self._calculate_comprehensive_metrics(targets, predictions, probabilities)

        self.logger.info(f"Evaluation completed. AUC: {metrics['auc']:.4f}")
        return metrics

    def _calculate_comprehensive_metrics(self, targets: np.ndarray,
                                       predictions: np.ndarray,
                                       probabilities: np.ndarray) -> Dict[str, Any]:
        """Calculate comprehensive evaluation metrics."""
        metrics = {}

        # Basic classification metrics
        metrics['accuracy'] = np.mean(predictions == targets)
        metrics['precision'] = np.sum((predictions == 1) & (targets == 1)) / np.sum(predictions == 1) if np.sum(predictions == 1) > 0 else 0
        metrics['recall'] = np.sum((predictions == 1) & (targets == 1)) / np.sum(targets == 1) if np.sum(targets == 1) > 0 else 0
        metrics['f1_score'] = 2 * metrics['precision'] * metrics['recall'] / (metrics['precision'] + metrics['recall']) if (metrics['precision'] + metrics['recall']) > 0 else 0

        # AUC metrics
        if len(np.unique(targets)) > 1:
            metrics['auc'] = roc_auc_score(targets, probabilities)
            metrics['average_precision'] = average_precision_score(targets, probabilities)
        else:
            metrics['auc'] = 0.5
            metrics['average_precision'] = 0.0

        # Confusion matrix
        cm = confusion_matrix(targets, predictions)
        metrics['confusion_matrix'] = cm.tolist()
        metrics['tn'], metrics['fp'], metrics['fn'], metrics['tp'] = cm.ravel()

        # Safety-critical metrics
        metrics['false_negative_rate'] = metrics['fn'] / (metrics['fn'] + metrics['tp']) if (metrics['fn'] + metrics['tp']) > 0 else 0
        metrics['miss_rate'] = metrics['false_negative_rate']  # Alias for clarity

        # Risk assessment metrics
        metrics['collision_detection_rate'] = metrics['recall']  # True positive rate for collision cases
        metrics['false_alarm_rate'] = metrics['fp'] / (metrics['fp'] + metrics['tn']) if (metrics['fp'] + metrics['tn']) > 0 else 0

        # Calibration metrics
        prob_true, prob_pred = calibration_curve(targets, probabilities, n_bins=10)
        metrics['calibration_curve'] = {
            'prob_true': prob_true.tolist(),
            'prob_pred': prob_pred.tolist()
        }

        # Expected Calibration Error (ECE)
        ece = np.mean(np.abs(prob_true - prob_pred))
        metrics['ece'] = ece

        # Brier score
        brier_score = np.mean((probabilities - targets) ** 2)
        metrics['brier_score'] = brier_score

        return metrics

    def evaluate_multiple_models(self, model_paths: Dict[str, str],
                               test_loader: DataLoader) -> Dict[str, Dict]:
        """
        Evaluate multiple models and compare them.

        Args:
            model_paths: Dictionary of model_name -> model_path
            test_loader: Test data loader

        Returns:
            Dictionary of model_name -> metrics
        """
        results = {}

        for model_name, model_path in model_paths.items():
            self.logger.info(f"Evaluating {model_name}...")

            # Extract model type from path or name
            model_type = self._infer_model_type(model_name)

            # Load and evaluate model
            model = self.load_model(model_path, model_type)
            metrics = self.evaluate_model(model, test_loader, model_type)

            results[model_name] = metrics

        return results

    def _infer_model_type(self, model_name: str) -> str:
        """Infer model type from model name."""
        name_lower = model_name.lower()
        if 'fusion' in name_lower:
            return 'fusion'
        elif 'static' in name_lower:
            return 'static'
        elif 'transformer' in name_lower:
            return 'transformer'
        else:
            return 'fusion'  # Default

    def plot_evaluation_results(self, metrics: Dict[str, Any], save_path: str = './plots'):
        """
        Plot comprehensive evaluation results.

        Args:
            metrics: Evaluation metrics
            save_path: Path to save plots
        """
        Path(save_path).mkdir(parents=True, exist_ok=True)

        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Collision Risk Model Evaluation', fontsize=16)

        # Confusion Matrix
        cm = np.array(metrics['confusion_matrix'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
        axes[0, 0].set_title('Confusion Matrix')
        axes[0, 0].set_xlabel('Predicted')
        axes[0, 0].set_ylabel('Actual')

        # ROC Curve (placeholder - would need targets and probabilities)
        axes[0, 1].set_title('ROC Curve')
        axes[0, 1].text(0.5, 0.5, f'AUC: {metrics["auc"]:.4f}',
                       ha='center', va='center', transform=axes[0, 1].transAxes)

        # Precision-Recall Curve (placeholder)
        axes[0, 2].set_title('Precision-Recall Curve')
        axes[0, 2].text(0.5, 0.5, f'AP: {metrics["average_precision"]:.4f}',
                       ha='center', va='center', transform=axes[0, 2].transAxes)

        # Calibration Curve
        if 'calibration_curve' in metrics:
            cal_data = metrics['calibration_curve']
            axes[1, 0].plot(cal_data['prob_pred'], cal_data['prob_true'], 's-')
            axes[1, 0].plot([0, 1], [0, 1], 'k--')
            axes[1, 0].set_title('Calibration Curve')
            axes[1, 0].set_xlabel('Predicted Probability')
            axes[1, 0].set_ylabel('True Probability')
            axes[1, 0].grid(True)

        # Metrics Summary
        metrics_text = ".4f"
        axes[1, 1].text(0.1, 0.9, f'Accuracy: {metrics["accuracy"]:.4f}', fontsize=10)
        axes[1, 1].text(0.1, 0.8, f'Precision: {metrics["precision"]:.4f}', fontsize=10)
        axes[1, 1].text(0.1, 0.7, f'Recall: {metrics["recall"]:.4f}', fontsize=10)
        axes[1, 1].text(0.1, 0.6, f'F1-Score: {metrics["f1_score"]:.4f}', fontsize=10)
        axes[1, 1].text(0.1, 0.5, f'AUC: {metrics["auc"]:.4f}', fontsize=10)
        axes[1, 1].text(0.1, 0.4, f'ECE: {metrics["ece"]:.4f}', fontsize=10)
        axes[1, 1].text(0.1, 0.3, f'Brier: {metrics["brier_score"]:.4f}', fontsize=10)
        axes[1, 1].set_title('Performance Metrics')
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].axis('off')

        # Safety Metrics
        axes[1, 2].text(0.1, 0.8, f'False Negative Rate: {metrics["false_negative_rate"]:.4f}', fontsize=10)
        axes[1, 2].text(0.1, 0.7, f'Collision Detection Rate: {metrics["collision_detection_rate"]:.4f}', fontsize=10)
        axes[1, 2].text(0.1, 0.6, f'False Alarm Rate: {metrics["false_alarm_rate"]:.4f}', fontsize=10)
        axes[1, 2].set_title('Safety-Critical Metrics')
        axes[1, 2].set_xlim(0, 1)
        axes[1, 2].set_ylim(0, 1)
        axes[1, 2].axis('off')

        plt.tight_layout()
        plt.savefig(f"{save_path}/evaluation_results.png", dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"Evaluation plots saved to {save_path}/evaluation_results.png")

    def save_evaluation_report(self, metrics: Dict[str, Any], save_path: str):
        """
        Save detailed evaluation report.

        Args:
            metrics: Evaluation metrics
            save_path: Path to save report
        """
        report = {
            'evaluation_timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'config': self.config
        }

        with open(save_path, 'w') as f:
            json.dump(report, f, indent=2)

        self.logger.info(f"Evaluation report saved to {save_path}")

    def compare_models(self, model_results: Dict[str, Dict], save_path: str = './comparison'):
        """
        Compare multiple models and generate comparison report.

        Args:
            model_results: Dictionary of model_name -> metrics
            save_path: Path to save comparison
        """
        Path(save_path).mkdir(parents=True, exist_ok=True)

        # Create comparison dataframe
        comparison_data = []
        for model_name, metrics in model_results.items():
            row = {
                'Model': model_name,
                'AUC': metrics['auc'],
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1_score'],
                'False Negative Rate': metrics['false_negative_rate'],
                'ECE': metrics['ece'],
                'Brier Score': metrics['brier_score']
            }
            comparison_data.append(row)

        df = pd.DataFrame(comparison_data)
        df.to_csv(f"{save_path}/model_comparison.csv", index=False)

        # Create comparison plot
        metrics_to_plot = ['AUC', 'Accuracy', 'F1-Score', 'False Negative Rate']
        df_plot = df.set_index('Model')[metrics_to_plot]

        plt.figure(figsize=(12, 6))
        df_plot.plot(kind='bar', ax=plt.gca())
        plt.title('Model Comparison')
        plt.ylabel('Score')
        plt.xticks(rotation=45)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(f"{save_path}/model_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"Model comparison saved to {save_path}")


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate collision risk assessment model')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--model_type', type=str, default='fusion',
                       choices=['fusion', 'static', 'transformer'],
                       help='Type of model to evaluate')
    parser.add_argument('--config', type=str, default='./configs/collision_risk_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--save_path', type=str, default='./results/evaluation',
                       help='Path to save evaluation results')
    parser.add_argument('--compare_models', action='store_true',
                       help='Compare multiple models')

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Load configuration
    import yaml
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    config['batch_size'] = args.batch_size

    # Create dataset
    builder = CollisionRiskDatasetBuilder(config)
    dataset = builder.generate_synthetic_dataset(n_samples=2000)

    # Create test loader (use all data for evaluation)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # Initialize evaluator
    evaluator = CollisionRiskEvaluator(config)

    if args.compare_models:
        # Compare multiple models
        model_paths = {
            'Fusion': args.model_path,
            # Add other models here
        }
        results = evaluator.evaluate_multiple_models(model_paths, test_loader)
        evaluator.compare_models(results, args.save_path)
    else:
        # Evaluate single model
        model = evaluator.load_model(args.model_path, args.model_type)
        metrics = evaluator.evaluate_model(model, test_loader, args.model_type)

        # Save results
        evaluator.plot_evaluation_results(metrics, args.save_path)
        evaluator.save_evaluation_report(metrics, f"{args.save_path}/evaluation_report.json")

        # Print summary
        print("Evaluation Results:")
        print(f"AUC: {metrics['auc']:.4f}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"False Negative Rate: {metrics['false_negative_rate']:.4f}")
        print(f"Collision Detection Rate: {metrics['collision_detection_rate']:.4f}")

    print(f"Evaluation completed. Results saved to {args.save_path}")


if __name__ == "__main__":
    main()