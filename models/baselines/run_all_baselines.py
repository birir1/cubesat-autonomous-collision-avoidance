# models/baselines/run_all_baselines.py

import os
import numpy as np
import pandas as pd

from core.dataset import SatelliteConjunctionDataset
from core.metrics import compute_regression_metrics

from models.baselines.random_forest import RandomForestBaseline
from models.baselines.xgboost_model import XGBoostBaseline


# -----------------------------
# CONFIG (FIXED)
# -----------------------------
DATA_PATH = "data/processed/conjunction_dataset.pkl"  # <-- CHANGE THIS if needed
OUTPUT_DIR = "results/baselines"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -----------------------------
# LOAD DATASET
# -----------------------------
def load_dataset():
    dataset = SatelliteConjunctionDataset(DATA_PATH)

    features = []
    targets = []

    for sample in dataset:
        features.append(sample['features'])
        targets.append(sample['target'])

    X = np.array(features)
    y = np.array(targets)

    return X, y


# -----------------------------
# TRAIN / TEST SPLIT (NEW)
# -----------------------------
def train_test_split(X, y, test_ratio=0.2):
    n = len(X)
    idx = np.random.permutation(n)

    split = int(n * (1 - test_ratio))

    train_idx = idx[:split]
    test_idx = idx[split:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


# -----------------------------
# MAIN PIPELINE
# -----------------------------
def run_all_baselines():

    print("Loading dataset...")
    X, y = load_dataset()

    print(f"Dataset size: {len(X)}")
    print(f"Target stats → min: {y.min():.6f}, max: {y.max():.6f}, mean: {y.mean():.6f}")

    # -----------------------------
    # TRAIN / TEST SPLIT
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")

    # -----------------------------
    # ML BASELINES
    # -----------------------------
    print("\nTraining Random Forest...")
    rf = RandomForestBaseline()
    rf.train(X_train, y_train)
    rf_preds = rf.predict(X_test)

    print("Training XGBoost...")
    xgb_model = XGBoostBaseline()
    xgb_model.train(X_train, y_train)
    xgb_preds = xgb_model.predict(X_test)

    # -----------------------------
    # EVALUATION
    # -----------------------------
    print("\nEvaluating models...")

    results = []

    def evaluate(name, preds):
        metrics = compute_regression_metrics(y_test, preds)
        metrics["model"] = name
        results.append(metrics)

    evaluate("RandomForest", rf_preds)
    evaluate("XGBoost", xgb_preds)

    results_df = pd.DataFrame(results)

    print("\n=== RESULTS ===")
    print(results_df)

    # -----------------------------
    # SAVE
    # -----------------------------
    results_path = os.path.join(OUTPUT_DIR, "baseline_results.csv")
    results_df.to_csv(results_path, index=False)

    print(f"\nSaved results to: {results_path}")

    return results_df


# -----------------------------
# ENTRY POINT
# -----------------------------
if __name__ == "__main__":
    run_all_baselines()