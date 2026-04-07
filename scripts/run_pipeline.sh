#!/usr/bin/env bash
set -euo pipefail

# =========================================================
# ENV SETUP
# =========================================================
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "======================================="
echo " CubeSat Collision Risk Pipeline (v5)"
echo "======================================="

if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    echo "Using virtual environment: ${VIRTUAL_ENV}"
fi

export PYTHONHASHSEED=42
export PYTHONPATH="$ROOT_DIR:$PYTHONPATH"
DEBUG=${DEBUG:-0}

# =========================================================
# STEP 1: TRAINING
# =========================================================
echo ""
echo "---------------------------------------"
echo "Step 1: Training Models"
echo "---------------------------------------"

if [[ "$DEBUG" == "1" ]]; then
    python3 -m scripts.train_all --config configs/training.yaml --debug
else
    python3 -m scripts.train_all --config configs/training.yaml
fi

echo "✅ Training completed"

# =========================================================
# STEP 2: EVALUATION
# =========================================================
echo ""
echo "---------------------------------------"
echo "Step 2: Evaluating Models"
echo "---------------------------------------"

if [[ "$DEBUG" == "1" ]]; then
    python3 -m evaluation.evaluate_all --config configs/evaluation.yaml --debug
else
    python3 -m evaluation.evaluate_all --config configs/evaluation.yaml
fi

echo "✅ Evaluation completed"

# =========================================================
# STEP 3: GENERATE FIGURES AND TABLES
# =========================================================
echo ""
echo "---------------------------------------"
echo "Step 3: Generating Figures and Tables"
echo "---------------------------------------"

python3 -m scripts.plot_metrics
echo "✅ Figures and tables generated successfully"

# =========================================================
# STEP 4: QUICK METRICS CHECK
# =========================================================
echo ""
echo "---------------------------------------"
echo "Step 4: Quick Metrics Summary"
echo "---------------------------------------"

METRICS_FILE="results/metrics/gnn_collision_metrics.csv"

if [[ -f "$METRICS_FILE" ]]; then
    echo "📊 GNN Metrics:"
    cat "$METRICS_FILE"
else
    echo "⚠️ GNN metrics file not found: $METRICS_FILE"
fi

# =========================================================
# STEP 5: SANITY CHECK (CRITICAL)
# =========================================================
echo ""
echo "---------------------------------------"
echo "Step 5: Sanity Check (Prediction Collapse)"
echo "---------------------------------------"

python3 - <<EOF
import pandas as pd
import os

metrics_path = "results/metrics/gnn_collision_metrics.csv"

if os.path.exists(metrics_path):
    df = pd.read_csv(metrics_path)

    if "metric" in df.columns:
        print("Metrics file format OK")

    try:
        far = float(df[df["metric"] == "false_alarm_rate"]["value"].values[0])
        recall = float(df[df["metric"] == "recall"]["value"].values[0])

        print(f"False Alarm Rate: {far:.4f}")
        print(f"Recall: {recall:.4f}")

        if far > 0.95:
            print("🚨 WARNING: Predicting ALL positives")
        elif recall < 0.1:
            print("🚨 WARNING: Predicting ALL negatives")
        else:
            print("✅ Model behavior looks reasonable")

    except Exception:
        print("⚠️ Could not parse metrics values")

else:
    print("⚠️ Metrics file not available")
EOF

# =========================================================
# DONE
# =========================================================
echo ""
echo "======================================="
echo " Pipeline completed successfully 🚀"
echo "======================================="