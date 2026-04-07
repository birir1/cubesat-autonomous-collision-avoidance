"""
Machine learning baseline models for collision risk assessment.

Implements traditional ML methods like Random Forest, Gradient Boosting,
and Feed-forward Neural Networks for comparison with deep learning approaches.
"""

import numpy as np
from typing import Optional, Dict, Any
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report


class MLBaseline:
    """
    Machine learning baseline models for collision risk assessment.

    Supports multiple traditional ML algorithms for comparison.
    """

    def __init__(self, model_type: str = 'random_forest', **kwargs):
        """
        Initialize ML baseline.

        Args:
            model_type: Type of ML model ('random_forest', 'gradient_boosting', 'mlp')
            **kwargs: Model-specific parameters
        """
        self.model_type = model_type
        self.scaler = StandardScaler()

        if model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', 10),
                random_state=42
            )
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=kwargs.get('n_estimators', 100),
                learning_rate=kwargs.get('learning_rate', 0.1),
                max_depth=kwargs.get('max_depth', 3),
                random_state=42
            )
        elif model_type == 'mlp':
            self.model = MLPClassifier(
                hidden_layer_sizes=kwargs.get('hidden_layer_sizes', (100, 50)),
                activation=kwargs.get('activation', 'relu'),
                learning_rate_init=kwargs.get('learning_rate', 0.001),
                max_iter=kwargs.get('max_iter', 1000),
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'MLBaseline':
        """
        Fit the ML model.

        Args:
            X: Feature matrix
            y: Target labels

        Returns:
            Self for chaining
        """
        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Fit model
        self.model.fit(X_scaled, y)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Feature matrix

        Returns:
            Predicted probabilities
        """
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predict class labels.

        Args:
            X: Feature matrix
            threshold: Classification threshold

        Returns:
            Predicted labels
        """
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance.

        Args:
            X: Feature matrix
            y: Target labels

        Returns:
            Dictionary of evaluation metrics
        """
        y_pred_proba = self.predict_proba(X)
        y_pred = self.predict(X)

        # Calculate metrics
        auc = roc_auc_score(y, y_pred_proba)

        # Classification report
        report = classification_report(y, y_pred, output_dict=True)

        return {
            'auc': auc,
            'accuracy': report['accuracy'],
            'precision': report['weighted avg']['precision'],
            'recall': report['weighted avg']['recall'],
            'f1_score': report['weighted avg']['f1-score']
        }

    def get_feature_importance(self) -> Optional[np.ndarray]:
        """
        Get feature importance scores.

        Returns:
            Feature importance array or None if not available
        """
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            # For linear models
            return np.abs(self.model.coef_[0])
        else:
            return None

    def get_model_name(self) -> str:
        """Get descriptive model name."""
        return f"{self.model_type.replace('_', ' ').title()}"


class EnsembleMLBaseline:
    """
    Ensemble of multiple ML baselines for improved performance.
    """

    def __init__(self, models: Optional[Dict[str, MLBaseline]] = None):
        """
        Initialize ensemble baseline.

        Args:
            models: Dictionary of model_name -> MLBaseline instances
        """
        if models is None:
            models = {
                'rf': MLBaseline('random_forest'),
                'gb': MLBaseline('gradient_boosting'),
                'mlp': MLBaseline('mlp')
            }

        self.models = models
        self.fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'EnsembleMLBaseline':
        """
        Fit all models in the ensemble.

        Args:
            X: Feature matrix
            y: Target labels

        Returns:
            Self for chaining
        """
        for name, model in self.models.items():
            print(f"Fitting {name}...")
            model.fit(X, y)

        self.fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities using ensemble averaging.

        Args:
            X: Feature matrix

        Returns:
            Average predicted probabilities
        """
        if not self.fitted:
            raise ValueError("Models must be fitted before prediction")

        probas = []
        for model in self.models.values():
            probas.append(model.predict_proba(X))

        # Average predictions
        return np.mean(probas, axis=0)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predict labels using ensemble.

        Args:
            X: Feature matrix
            threshold: Classification threshold

        Returns:
            Predicted labels
        """
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate ensemble performance.

        Args:
            X: Feature matrix
            y: Target labels

        Returns:
            Dictionary of evaluation metrics
        """
        y_pred_proba = self.predict_proba(X)
        y_pred = self.predict(X)

        auc = roc_auc_score(y, y_pred_proba)
        report = classification_report(y, y_pred, output_dict=True)

        return {
            'auc': auc,
            'accuracy': report['accuracy'],
            'precision': report['weighted avg']['precision'],
            'recall': report['weighted avg']['recall'],
            'f1_score': report['weighted avg']['f1-score']
        }

    def get_model_name(self) -> str:
        """Get descriptive model name."""
        return "Ensemble ML Baseline"


def create_feature_matrix(states1: np.ndarray, states2: np.ndarray) -> np.ndarray:
    """
    Create feature matrix from satellite state pairs.

    Args:
        states1: Primary satellite states (N x 6)
        states2: Secondary satellite states (N x 6)

    Returns:
        Feature matrix (N x M) where M is number of features
    """
    # Relative position and velocity
    rel_pos = states2[:, :3] - states1[:, :3]
    rel_vel = states2[:, 3:] - states1[:, 3:]

    # Derived features
    distance = np.linalg.norm(rel_pos, axis=1, keepdims=True)
    speed = np.linalg.norm(rel_vel, axis=1, keepdims=True)
    rel_speed = np.linalg.norm(rel_vel, axis=1, keepdims=True)

    # Combine features
    features = np.concatenate([
        rel_pos,      # relative position (x, y, z)
        rel_vel,      # relative velocity (vx, vy, vz)
        distance,     # Euclidean distance
        speed,        # relative speed magnitude
        rel_speed     # same as speed (for compatibility)
    ], axis=1)

    return features