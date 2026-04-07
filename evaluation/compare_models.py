"""
Model comparison utilities for evaluating multiple collision avoidance models.
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Any
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc,
    confusion_matrix, classification_report
)

from .metrics import safety_metrics, calibration_metrics


class ModelComparison:
    """
    Comprehensive model comparison framework for collision avoidance models.
    """

    def __init__(self, results_dir: str = "results/model_comparison"):
        """
        Initialize model comparison framework.

        Args:
            results_dir: Directory to save comparison results
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.models = {}
        self.results = {}

    def add_model(self, name: str, model: torch.nn.Module,
                  predictions: np.ndarray, labels: np.ndarray,
                  metadata: Dict[str, Any] = None):
        """
        Add a model to the comparison.

        Args:
            name: Model name/identifier
            model: PyTorch model instance
            predictions: Model predictions (probabilities)
            labels: True labels
            metadata: Additional model metadata
        """
        self.models[name] = {
            'model': model,
            'predictions': predictions,
            'labels': labels,
            'metadata': metadata or {}
        }

    def compute_metrics(self, thresholds: List[float] = None) -> pd.DataFrame:
        """
        Compute comprehensive metrics for all models.

        Args:
            thresholds: Risk thresholds for safety metrics

        Returns:
            DataFrame with metrics for each model
        """
        if thresholds is None:
            thresholds = [0.01, 0.05, 0.1, 0.2, 0.5]

        metrics_data = []

        for name, model_data in self.models.items():
            preds = model_data['predictions']
            labels = model_data['labels']

            # Basic metrics
            auc_roc = roc_auc_score(labels, preds)
            precision, recall, _ = precision_recall_curve(labels, preds)
            auc_pr = auc(recall, precision)

            # Safety metrics at different thresholds
            safety_results = {}
            for threshold in thresholds:
                safety = safety_metrics(labels, preds, threshold)
                safety_results.update({
                    f'cdr_{threshold}': safety['collision_detection_rate'],
                    f'far_{threshold}': safety['false_alarm_rate'],
                    f'precision_{threshold}': safety['precision'],
                    f'recall_{threshold}': safety['recall']
                })

            # Calibration metrics
            calib = calibration_metrics(labels, preds)
            calib_results = {
                'ece': calib['ece'],
                'mce': calib['mce'],
                'brier_score': calib['brier_score']
            }

            # Model size and complexity
            model = model_data['model']
            param_count = sum(p.numel() for p in model.parameters())

            # Combine all metrics
            model_metrics = {
                'model': name,
                'auc_roc': auc_roc,
                'auc_pr': auc_pr,
                'parameters': param_count,
                **safety_results,
                **calib_results
            }

            metrics_data.append(model_metrics)

        self.results['metrics'] = pd.DataFrame(metrics_data)
        return self.results['metrics']

    def statistical_comparison(self) -> Dict[str, Any]:
        """
        Perform statistical comparison between models.

        Returns:
            Dictionary with statistical test results
        """
        if 'metrics' not in self.results:
            self.compute_metrics()

        df = self.results['metrics']

        # Pairwise comparisons
        comparisons = {}
        metrics_to_compare = ['auc_roc', 'auc_pr', 'cdr_0.1', 'far_0.1']

        for metric in metrics_to_compare:
            if metric in df.columns:
                values = df[metric].values
                best_idx = np.argmax(values)
                best_model = df.iloc[best_idx]['model']
                best_value = values[best_idx]

                # Simple ranking
                rankings = np.argsort(values)[::-1]
                ranked_models = df.iloc[rankings]['model'].tolist()

                comparisons[metric] = {
                    'best_model': best_model,
                    'best_value': best_value,
                    'ranking': ranked_models,
                    'improvement_over_baseline': best_value - values.min()
                }

        self.results['statistical_comparison'] = comparisons
        return comparisons

    def plot_comparison(self, save_plots: bool = True):
        """
        Generate comparison plots.

        Args:
            save_plots: Whether to save plots to disk
        """
        if 'metrics' not in self.results:
            self.compute_metrics()

        df = self.results['metrics']
        plots_dir = self.results_dir / "plots"
        plots_dir.mkdir(exist_ok=True)

        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)

        # 1. AUC Comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # AUC-ROC
        sns.barplot(data=df, x='model', y='auc_roc', ax=axes[0,0])
        axes[0,0].set_title('AUC-ROC Comparison')
        axes[0,0].tick_params(axis='x', rotation=45)

        # AUC-PR
        sns.barplot(data=df, x='model', y='auc_pr', ax=axes[0,1])
        axes[0,1].set_title('AUC-PR Comparison')
        axes[0,1].tick_params(axis='x', rotation=45)

        # Safety Metrics
        safety_df = df.melt(id_vars=['model'],
                           value_vars=[f'cdr_{t}' for t in [0.01, 0.05, 0.1]],
                           var_name='threshold', value_name='value')
        sns.lineplot(data=safety_df, x='threshold', y='value', hue='model',
                    marker='o', ax=axes[1,0])
        axes[1,0].set_title('Collision Detection Rate vs Threshold')

        # False Alarm Rate
        far_df = df.melt(id_vars=['model'],
                        value_vars=[f'far_{t}' for t in [0.01, 0.05, 0.1]],
                        var_name='threshold', value_name='value')
        sns.lineplot(data=far_df, x='threshold', y='value', hue='model',
                    marker='o', ax=axes[1,1])
        axes[1,1].set_title('False Alarm Rate vs Threshold')

        plt.tight_layout()

        if save_plots:
            plt.savefig(plots_dir / "model_comparison_overview.png", dpi=300, bbox_inches='tight')
        plt.show()

        # 2. Calibration Plot
        fig, ax = plt.subplots(figsize=(10, 8))

        for name, model_data in self.models.items():
            preds = model_data['predictions']
            labels = model_data['labels']

            # Compute calibration curve
            from sklearn.calibration import calibration_curve
            prob_true, prob_pred = calibration_curve(labels, preds, n_bins=10)

            ax.plot(prob_pred, prob_true, marker='o', label=name, linewidth=2)

        ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
        ax.set_xlabel('Predicted Probability')
        ax.set_ylabel('Actual Probability')
        ax.set_title('Calibration Curves')
        ax.legend()
        ax.grid(True, alpha=0.3)

        if save_plots:
            plt.savefig(plots_dir / "calibration_comparison.png", dpi=300, bbox_inches='tight')
        plt.show()

    def generate_report(self) -> str:
        """
        Generate comprehensive comparison report.

        Returns:
            Formatted report string
        """
        if 'metrics' not in self.results:
            self.compute_metrics()

        if 'statistical_comparison' not in self.results:
            self.statistical_comparison()

        df = self.results['metrics']
        stats = self.results['statistical_comparison']

        report = []
        report.append("# Model Comparison Report")
        report.append("=" * 50)
        report.append("")

        # Summary statistics
        report.append("## Summary Statistics")
        report.append(f"Number of models compared: {len(df)}")
        report.append(f"Best AUC-ROC: {df['auc_roc'].max():.4f} ({df.loc[df['auc_roc'].idxmax(), 'model']})")
        report.append(f"Best AUC-PR: {df['auc_pr'].max():.4f} ({df.loc[df['auc_pr'].idxmax(), 'model']})")
        report.append("")

        # Detailed metrics table
        report.append("## Detailed Metrics")
        report.append("")
        report.append(df.to_markdown(index=False))
        report.append("")

        # Statistical comparison
        report.append("## Statistical Comparison")
        for metric, comparison in stats.items():
            report.append(f"### {metric.upper()}")
            report.append(f"- Best Model: {comparison['best_model']}")
            report.append(".4f")
            report.append(f"- Ranking: {' > '.join(comparison['ranking'])}")
            report.append(".4f")
            report.append("")

        return "\n".join(report)

    def save_results(self):
        """Save all results to disk."""
        # Save metrics
        if 'metrics' in self.results:
            self.results['metrics'].to_csv(
                self.results_dir / "metrics.csv", index=False
            )

        # Save statistical comparison
        if 'statistical_comparison' in self.results:
            with open(self.results_dir / "statistical_comparison.json", 'w') as f:
                json.dump(self.results['statistical_comparison'], f, indent=2)

        # Save report
        report = self.generate_report()
        with open(self.results_dir / "comparison_report.md", 'w') as f:
            f.write(report)

        print(f"Results saved to {self.results_dir}")


def compare_models_from_predictions(model_predictions: Dict[str, Dict[str, np.ndarray]],
                                   save_dir: str = "results/model_comparison") -> ModelComparison:
    """
    Convenience function to compare models from prediction dictionaries.

    Args:
        model_predictions: Dict[model_name -> {'predictions': array, 'labels': array, 'model': nn.Module}]
        save_dir: Directory to save results

    Returns:
        ModelComparison instance with results
    """
    comparator = ModelComparison(save_dir)

    for name, data in model_predictions.items():
        comparator.add_model(
            name=name,
            model=data.get('model'),
            predictions=data['predictions'],
            labels=data['labels'],
            metadata=data.get('metadata', {})
        )

    # Run full comparison
    comparator.compute_metrics()
    comparator.statistical_comparison()
    comparator.plot_comparison()
    comparator.save_results()

    return comparator