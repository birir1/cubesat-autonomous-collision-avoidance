"""
Collision Engine using REAL orbital data + trained ML model

Uses:
- Skyfield for precise orbit propagation
- PyTorch model for collision risk scoring
- Physics-aware filtering (distance gating)
"""

import torch
import numpy as np
from skyfield.api import load

from models.collision_risk_model import CollisionRiskModel


class CollisionEngine:
    def __init__(self, model_path):
        # Load trained model
        self.model = CollisionRiskModel()
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model.eval()

        # Skyfield time system
        self.ts = load.timescale()

        # Feature normalization (important!)
        self.eps = 1e-8

    def get_state(self, sat):
        """
        Get position + velocity using Skyfield
        Returns: np.array [x, y, z, vx, vy, vz]
        """

        try:
            t = self.ts.now()
            geocentric = sat.at(t)

            pos = np.array(geocentric.position.km)
            vel = np.array(geocentric.velocity.km_per_s)

            return np.concatenate([pos, vel])

        except Exception:
            return None  # skip bad satellites

    def normalize(self, x):
        """
        Normalize features to stabilize ML predictions
        """
        return (x - np.mean(x)) / (np.std(x) + self.eps)

    def compute_risk(self, sat1, sat2):
        """
        Compute collision risk between two satellites
        """

        s1 = self.get_state(sat1)
        s2 = self.get_state(sat2)

        if s1 is None or s2 is None:
            return 0.0

        # Relative state
        relative_state = np.abs(s1 - s2)

        # Normalize
        relative_state = self.normalize(relative_state)

        # Convert to tensor with batch dimension
        x = torch.tensor(relative_state, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            risk = self.model(x).item()

        return risk

    def detect_high_risk_pairs(
        self,
        sats,
        threshold=0.001,     # ✅ FIXED (realistic threshold)
        distance_filter=50,  # km pre-filter
        max_pairs=10000
    ):
        """
        Detect high-risk satellite pairs using ML + physics filter

        Args:
            sats: list of satellites
            threshold: ML risk threshold
            distance_filter: only check satellites within this distance (km)
            max_pairs: computation cap

        Returns:
            list of risky pairs
        """

        risky_pairs = []
        count = 0

        for i in range(len(sats)):
            s1 = self.get_state(sats[i])
            if s1 is None:
                continue

            for j in range(i + 1, len(sats)):

                if count >= max_pairs:
                    return risky_pairs

                s2 = self.get_state(sats[j])
                if s2 is None:
                    continue

                # ✅ PHYSICS FILTER (huge performance + realism boost)
                distance = np.linalg.norm(s1[:3] - s2[:3])

                if distance > distance_filter:
                    continue  # too far → ignore

                # ML risk
                risk = self.compute_risk(sats[i], sats[j])

                if risk > threshold:
                    risky_pairs.append({
                        "sat1": sats[i].name,
                        "sat2": sats[j].name,
                        "risk": float(risk),
                        "distance_km": float(distance)
                    })

                count += 1

        return risky_pairs