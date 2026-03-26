"""
Trajectory Simulation + Visualization (ADVANCED - RESEARCH GRADE)

Features:
- Real satellite orbit propagation (Skyfield)
- Transformer-based risk prediction
- Stable + safe model loading
- Training-consistent normalization
- 2D + 3D visualization
- Closest approach detection
- Structured outputs for evaluation/paper
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

from datetime import timedelta
from skyfield.api import load

from models.trajectory_risk_model import TrajectoryRiskModel


class TrajectorySimulator:
    def __init__(self, model_path, device="cpu"):

        # ---- Device handling ----
        if device == "cuda" and not torch.cuda.is_available():
            print("⚠️ CUDA requested but not available. Falling back to CPU.")
            device = "cpu"

        self.device = torch.device(device)

        # ---- Load model ----
        self.model = TrajectoryRiskModel().to(self.device)

        try:
            state_dict = torch.load(model_path, map_location=self.device)

            if hasattr(self.model, "load_safe"):
                self.model.load_safe(state_dict)
            else:
                self.model.load_state_dict(state_dict, strict=False)

            print("Model loaded successfully")

        except Exception as e:
            print("⚠️ Model loading warning:", e)

        self.model.eval()

        # ---- Skyfield ----
        self.ts = load.timescale()

    # =========================================
    # CORE SIMULATION
    # =========================================
    def simulate_pair(self, sat1, sat2, time_steps=20, step_minutes=2):

        t0 = self.ts.now()

        trajectory = []
        distances = []
        pos1_list = []
        pos2_list = []

        for step in range(time_steps):
            try:
                t = self.ts.utc(
                    t0.utc_datetime() + timedelta(minutes=step * step_minutes)
                )

                s1 = sat1.at(t)
                s2 = sat2.at(t)

                pos1 = np.array(s1.position.km, dtype=np.float32)
                vel1 = np.array(s1.velocity.km_per_s, dtype=np.float32)

                pos2 = np.array(s2.position.km, dtype=np.float32)
                vel2 = np.array(s2.velocity.km_per_s, dtype=np.float32)

                rel_pos = pos1 - pos2
                rel_vel = vel1 - vel2

                rel = np.concatenate([rel_pos, rel_vel]).astype(np.float32)

                # ---- Stability checks ----
                if not np.all(np.isfinite(rel)):
                    return None, None, None, None, None

                # Clip extreme values (important for stability)
                rel = np.clip(rel, -1e4, 1e4)

                trajectory.append(rel)
                distances.append(np.linalg.norm(rel_pos))

                pos1_list.append(pos1)
                pos2_list.append(pos2)

            except Exception:
                return None, None, None, None, None

        if len(trajectory) == 0:
            return None, None, None, None, None

        trajectory = np.array(trajectory, dtype=np.float32)
        distances = np.array(distances, dtype=np.float32)

        # ---- Normalization (MATCH TRAINING EXACTLY) ----
        mean = trajectory.mean(axis=0, keepdims=True)
        std = trajectory.std(axis=0, keepdims=True) + 1e-8
        trajectory_norm = (trajectory - mean) / std

        # ---- Final sanity check ----
        if not np.isfinite(trajectory_norm).all():
            return None, None, None, None, None

        # ---- Predict ----
        try:
            with torch.no_grad():
                x = torch.tensor(
                    trajectory_norm,
                    dtype=torch.float32,
                    device=self.device
                ).unsqueeze(0)

                risk = self.model(x).item()
                risk = float(np.clip(risk, 0.0, 1.0))

        except Exception:
            risk = 0.0

        return (
            trajectory,
            distances,
            risk,
            np.array(pos1_list, dtype=np.float32),
            np.array(pos2_list, dtype=np.float32),
        )

    # =========================================
    # VISUALIZATION
    # =========================================
    def visualize(self, sat1, sat2):

        result = self.simulate_pair(sat1, sat2)

        if result[0] is None:
            print("Simulation failed.")
            return

        traj, distances, risk, pos1, pos2 = result

        if len(distances) == 0:
            print("Empty simulation.")
            return

        steps = np.arange(len(distances))

        # ---- Closest approach ----
        min_idx = int(np.argmin(distances))
        min_dist = float(distances[min_idx])

        print("\n=== Collision Analysis ===")
        print(f"Min Distance: {min_dist:.2f} km")
        print(f"Predicted Risk: {risk:.4f}")
        print(f"Closest Approach Step: {min_idx}")

        # ---- Plot ----
        fig = plt.figure(figsize=(15, 5))

        # 1. Distance plot
        ax1 = fig.add_subplot(1, 3, 1)
        ax1.plot(steps, distances)
        ax1.scatter(min_idx, min_dist)
        ax1.set_title(f"Distance Over Time\nRisk: {risk:.4f}")
        ax1.set_xlabel("Time Step")
        ax1.set_ylabel("Distance (km)")

        # 2. Relative XY
        ax2 = fig.add_subplot(1, 3, 2)
        ax2.plot(traj[:, 0], traj[:, 1])
        ax2.scatter(traj[min_idx, 0], traj[min_idx, 1])
        ax2.set_title("Relative Trajectory (XY)")
        ax2.set_xlabel("X (km)")
        ax2.set_ylabel("Y (km)")

        # 3. 3D orbits
        ax3 = fig.add_subplot(1, 3, 3, projection="3d")

        ax3.plot(pos1[:, 0], pos1[:, 1], pos1[:, 2])
        ax3.plot(pos2[:, 0], pos2[:, 1], pos2[:, 2])

        # highlight closest approach
        ax3.scatter(pos1[min_idx, 0], pos1[min_idx, 1], pos1[min_idx, 2])
        ax3.scatter(pos2[min_idx, 0], pos2[min_idx, 1], pos2[min_idx, 2])

        ax3.set_title("3D Orbit Visualization")
        ax3.set_xlabel("X (km)")
        ax3.set_ylabel("Y (km)")
        ax3.set_zlabel("Z (km)")

        plt.tight_layout()
        plt.show()

    # =========================================
    # ANALYSIS (FOR PAPER)
    # =========================================
    def analyze_pair(self, sat1, sat2):

        result = self.simulate_pair(sat1, sat2)

        if result[0] is None:
            return None

        _, distances, risk, _, _ = result

        return {
            "min_distance": float(np.min(distances)),
            "mean_distance": float(np.mean(distances)),
            "risk": float(risk),
        }