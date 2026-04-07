# models/baselines/random_forest.py

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler


class RandomForestBaseline:
    """
    Random Forest baseline for collision risk estimation.

    Designed for orbital conjunction features:
    - relative position (x, y, z)
    - relative velocity (vx, vy, vz)
    - distance
    - speed
    - time-to-closest-approach (TCA)
    - miss distance

    Outputs:
        Continuous risk score ∈ [0, 1]
    """

    def __init__(
        self,
        n_estimators=500,
        max_depth=25,
        min_samples_split=4,
        min_samples_leaf=2,
        use_scaling=True,
        random_state=42
    ):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            n_jobs=-1,
            random_state=random_state
        )

        self.use_scaling = use_scaling
        self.scaler = StandardScaler() if use_scaling else None
        self.is_trained = False

    def _validate_input(self, X):
        """
        Ensure correct shape and numeric stability
        """
        X = np.asarray(X, dtype=np.float64)

        if X.ndim == 1:
            X = X.reshape(1, -1)

        # Replace NaNs/Infs (important for real orbital data)
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

        return X

    def train(self, X, y):
        """
        Train on real orbital dataset
        """
        X = self._validate_input(X)
        y = np.asarray(y, dtype=np.float64)

        # Clip targets to valid probability range
        y = np.clip(y, 0.0, 1.0)

        if self.use_scaling:
            X = self.scaler.fit_transform(X)

        self.model.fit(X, y)
        self.is_trained = True

    def predict(self, X):
        """
        Predict collision risk
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction.")

        X = self._validate_input(X)

        if self.use_scaling:
            X = self.scaler.transform(X)

        preds = self.model.predict(X)

        # Ensure valid probability range
        preds = np.clip(preds, 0.0, 1.0)

        return preds

    def predict_with_uncertainty(self, X):
        """
        Estimate uncertainty using tree variance (useful baseline)
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction.")

        X = self._validate_input(X)

        if self.use_scaling:
            X = self.scaler.transform(X)

        # Collect predictions from all trees
        tree_preds = np.array([tree.predict(X) for tree in self.model.estimators_])

        mean_pred = np.mean(tree_preds, axis=0)
        std_pred = np.std(tree_preds, axis=0)

        mean_pred = np.clip(mean_pred, 0.0, 1.0)

        return mean_pred, std_pred