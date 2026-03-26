"""
Full Autonomous CubeSat Collision Avoidance Pipeline

Pipeline:
Dataset → Vision Detection → Tracking → State Estimation
→ Trajectory Prediction → Collision Risk → RL Maneuver

This script integrates all modules of the research system.
"""

import os
import torch
import numpy as np

from utils.data_loader import load_space_images
from utils.feature_engineering import build_state_vectors

from phases.phase3_trajectory_prediction.models.lstm_model import LSTMTrajectoryPredictor
from phases.phase3_trajectory_prediction.models.transformer_model import TransformerTrajectoryPredictor

from phases.phase4_collision_risk_estimation.models.gnn_model import GNNCollisionModel

from phases.phase5_maneuver_planning_rl.environment.orbital_env import OrbitalCollisionEnv
from phases.phase5_maneuver_planning_rl.models.ppo_agent import PPO


# ------------------------------------------------
# CONFIG
# ------------------------------------------------

MODEL_DIR = "results/saved_models"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------
# LOAD MODELS
# ------------------------------------------------

def load_models():

    print("Loading trained models...")

    lstm_model = LSTMTrajectoryPredictor().to(DEVICE)
    transformer_model = TransformerTrajectoryPredictor().to(DEVICE)

    gnn_model = GNNCollisionModel().to(DEVICE)

    ppo_model = PPO(state_dim=24, action_dim=2)

    # optional loading if trained
    try:
        lstm_model.load_state_dict(
            torch.load(os.path.join(MODEL_DIR, "lstm_model.pth"))
        )
    except:
        pass

    try:
        transformer_model.load_state_dict(
            torch.load(os.path.join(MODEL_DIR, "transformer_model.pth"))
        )
    except:
        pass

    try:
        gnn_model.load_state_dict(
            torch.load(os.path.join(MODEL_DIR, "gnn_collision_model.pth"))
        )
    except:
        pass

    try:
        ppo_model.load(
            os.path.join(MODEL_DIR, "ppo_cubesat_final.pth")
        )
    except:
        pass

    print("Models loaded")

    return lstm_model, transformer_model, gnn_model, ppo_model


# ------------------------------------------------
# TRAJECTORY PREDICTION
# ------------------------------------------------

def predict_trajectory(state_vectors, lstm_model, transformer_model):

    state_tensor = torch.FloatTensor(state_vectors).to(DEVICE)

    with torch.no_grad():

        lstm_pred = lstm_model(state_tensor)

        transformer_pred = transformer_model(state_tensor)

    prediction = (lstm_pred + transformer_pred) / 2

    return prediction.cpu().numpy()


# ------------------------------------------------
# COLLISION RISK ESTIMATION
# ------------------------------------------------

def estimate_collision(predicted_trajectories, gnn_model):

    trajectory_tensor = torch.FloatTensor(predicted_trajectories).to(DEVICE)

    with torch.no_grad():

        risk_scores = gnn_model(trajectory_tensor)

    return risk_scores.cpu().numpy()


# ------------------------------------------------
# MANEUVER PLANNING
# ------------------------------------------------

def plan_maneuver(state, ppo_agent):

    action = ppo_agent.select_action(state)

    return action


# ------------------------------------------------
# MAIN PIPELINE
# ------------------------------------------------

def run_pipeline():

    print("Starting CubeSat Autonomous Collision Avoidance Pipeline")

    lstm_model, transformer_model, gnn_model, ppo_agent = load_models()

    env = OrbitalCollisionEnv()

    state = env.reset()

    # ------------------------------------------------
    # STEP 1: DATA INPUT
    # ------------------------------------------------

    images = load_space_images("data/raw/space_images")

    # ------------------------------------------------
    # STEP 2: FEATURE EXTRACTION
    # ------------------------------------------------

    state_vectors = build_state_vectors(images)

    # ------------------------------------------------
    # STEP 3: TRAJECTORY PREDICTION
    # ------------------------------------------------

    predicted_trajectories = predict_trajectory(
        state_vectors,
        lstm_model,
        transformer_model
    )

    print("Trajectory prediction complete")

    # ------------------------------------------------
    # STEP 4: COLLISION RISK
    # ------------------------------------------------

    collision_risk = estimate_collision(
        predicted_trajectories,
        gnn_model
    )

    print("Collision risk scores:", collision_risk)

    # ------------------------------------------------
    # STEP 5: MANEUVER DECISION
    # ------------------------------------------------

    maneuver = plan_maneuver(state, ppo_agent)

    print("Recommended maneuver:", maneuver)

    # ------------------------------------------------
    # STEP 6: APPLY ACTION
    # ------------------------------------------------

    next_state, reward, terminated, truncated, info = env.step(maneuver)

    print("Environment response:", reward, info)

    print("Pipeline complete")


# ------------------------------------------------

if __name__ == "__main__":

    run_pipeline()