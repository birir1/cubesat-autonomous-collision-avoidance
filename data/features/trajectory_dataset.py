"""
Trajectory Dataset Builder (RESEARCH-GRADE VERSION)

Upgrades:
- Longer temporal window (better for Transformers)
- Temporal-aware risk computation
- Balanced synthetic vs real sampling
- Hard sample emphasis
- Improved realism (noise injection)
"""

import numpy as np
from skyfield.api import load
from datetime import timedelta
import random


# ============================================
# NORMALIZATION
# ============================================

def normalize_trajectory(traj):
    mean = np.mean(traj, axis=0, keepdims=True)
    std = np.std(traj, axis=0, keepdims=True) + 1e-8
    return (traj - mean) / std


# ============================================
# RISK FUNCTION (TEMPORAL-AWARE FIX)
# ============================================

def compute_temporal_risk(distances, rel_speeds):
    """
    Uses full trajectory instead of single point
    """

    min_dist = np.min(distances)
    mean_dist = np.mean(distances)

    mean_speed = np.mean(rel_speeds)

    risk_d = np.exp(-min_dist / 20.0)
    risk_d_avg = np.exp(-mean_dist / 50.0)

    risk_v = min(mean_speed / 10.0, 1.0)

    risk = 0.6 * risk_d + 0.2 * risk_d_avg + 0.2 * risk_v

    return float(np.clip(risk, 0.0, 1.0))


# ============================================
# SYNTHETIC CLOSE ENCOUNTER
# ============================================

def inject_close_encounter(pos1, vel1):

    distance = np.random.choice(
        [
            np.random.uniform(0.1, 2.0),
            np.random.uniform(2.0, 10.0),
            np.random.uniform(10.0, 50.0),
        ],
        p=[0.3, 0.4, 0.3]  # less bias toward extreme
    )

    direction = np.random.normal(size=3)
    direction /= np.linalg.norm(direction)

    pos2 = pos1 + direction * distance

    rel_vel = np.random.normal(0, 1.0, size=3)
    vel2 = vel1 + rel_vel

    return pos2, vel2


# ============================================
# MAIN BUILDER
# ============================================

def build_trajectory_dataset(
    sats,
    num_samples=5000,
    time_steps=20,        # 🔥 CRITICAL UPGRADE
    step_minutes=5
):

    ts = load.timescale()
    X, y = [], []

    print("Building trajectory dataset...")

    valid_sats = [s for s in sats if hasattr(s, "at")]
    n_sats = len(valid_sats)
    print(f"Valid satellites: {n_sats}")

    counts = {"high": 0, "medium": 0, "low": 0}

    i = 0
    attempts = 0
    max_attempts = num_samples * 20

    while i < num_samples and attempts < max_attempts:
        attempts += 1

        try:
            sat1 = random.choice(valid_sats)
            t0 = ts.now()
            trajectory = []

            use_synthetic = random.random() < 0.5  # 🔥 better balance

            for step in range(time_steps):

                t = ts.utc(t0.utc_datetime() + timedelta(minutes=step * step_minutes))
                s1 = sat1.at(t)

                pos1 = np.array(s1.position.km)
                vel1 = np.array(s1.velocity.km_per_s)

                if use_synthetic:
                    pos2, vel2 = inject_close_encounter(pos1, vel1)
                else:
                    sat2 = random.choice(valid_sats)
                    s2 = sat2.at(t)
                    pos2 = np.array(s2.position.km)
                    vel2 = np.array(s2.velocity.km_per_s)

                rel_pos = pos1 - pos2
                rel_vel = vel1 - vel2

                rel = np.concatenate([rel_pos, rel_vel])

                if not np.all(np.isfinite(rel)):
                    trajectory = None
                    break

                # realism noise
                rel += np.random.normal(0, 0.01, size=6)

                rel = np.clip(rel, -1e4, 1e4)

                trajectory.append(rel)

            if trajectory is None:
                continue

            trajectory = np.array(trajectory)

            distances = np.linalg.norm(trajectory[:, :3], axis=1)
            rel_speeds = np.linalg.norm(trajectory[:, 3:], axis=1)

            risk = compute_temporal_risk(distances, rel_speeds)

            # ============================================
            # BALANCING (IMPROVED)
            # ============================================

            if risk > 0.5:
                counts["high"] += 1
                keep = True

            elif risk > 0.2:
                counts["medium"] += 1
                keep = True

            else:
                counts["low"] += 1
                keep = random.random() < 0.4

            # 🔥 HARD SAMPLE BOOST
            if 0.2 < risk < 0.5:
                keep = True

            if not keep:
                continue

            trajectory = normalize_trajectory(trajectory)

            if not np.all(np.isfinite(trajectory)):
                continue

            X.append(trajectory)
            y.append(risk)
            i += 1

            if i % 1000 == 0:
                print(f"Built {i} samples")

        except Exception:
            continue

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    print(f"\nFinal dataset: {X.shape}")

    if len(y) > 0:
        print(
            f"Risk distribution → min: {y.min():.4f}, "
            f"mean: {y.mean():.4f}, max: {y.max():.4f}"
        )

    print("\nSampling stats:")
    print(counts)

    if counts["high"] < 100:
        print("\n WARNING: Too few high-risk samples!")

    return X, y