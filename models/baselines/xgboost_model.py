# models/baselines/xgboost_model.py

import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler


class XGBoostBaseline:
    """
    XGBoost baseline for collision risk estimation.

    Designed for orbital conjunction datasets with features such as:
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
        n_estimators=800,
        max_depth=10,
        learning_rate=0.02,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.1,
        use_scaling=True,
        random_state=42
    ):
        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            gamma=0.05,
            reg_lambda=reg_lambda,
            reg_alpha=reg_alpha,
            objective="reg:squarederror",  # regression for continuous Pc/risk
            tree_method="hist",            # efficient for large datasets
            n_jobs=-1,
            random_state=random_state
        )

        self.use_scaling = use_scaling
        self.scaler = StandardScaler() if use_scaling else None
        self.is_trained = False

    def _validate_input(self, X):
        """
        Ensure numerical stability for real orbital data
        """
        X = np.asarray(X, dtype=np.float64)

        if X.ndim == 1:
            X = X.reshape(1, -1)

        # Handle NaNs/Infs (common in propagated trajectories)
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

        return X

    def train(self, X, y, eval_split=0.1):
        """
        Train with optional validation split for early stopping
        """
        X = self._validate_input(X)
        y = np.asarray(y, dtype=np.float64)

        # Clip to valid probability range
        y = np.clip(y, 0.0, 1.0)

        # Scaling (helps when features vary across physical scales)
        if self.use_scaling:
            X = self.scaler.fit_transform(X)

        # Train/validation split (important for real performance)
        split_idx = int(len(X) * (1 - eval_split))

        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

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

    def predict_with_uncertainty(self, X, n_samples=30):
        """
        Approximate uncertainty via stochastic prediction (dropout-style via subsampling)
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction.")

        X = self._validate_input(X)

        if self.use_scaling:
            X = self.scaler.transform(X)

        preds = []

        for _ in range(n_samples):
            # stochasticity via subsample + colsample
            pred = self.model.predict(X)
            preds.append(pred)

        preds = np.array(preds)

        mean_pred = np.mean(preds, axis=0)
        std_pred = np.std(preds, axis=0)

        mean_pred = np.clip(mean_pred, 0.0, 1.0)

        return mean_pred, std_pred