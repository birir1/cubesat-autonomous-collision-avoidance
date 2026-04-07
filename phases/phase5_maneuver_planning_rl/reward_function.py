"""
Reward function definitions for maneuver planning with reinforcement learning.
"""

from typing import Any, Dict

import numpy as np


class ManeuverReward:
    """Base reward for maneuver planning."""

    def __init__(self, config: Dict[str, Any]):
        self.collision_penalty = config.get('collision_penalty', -1000.0)
        self.success_reward = config.get('success_reward', 100.0)
        self.distance_reward_weight = config.get('distance_reward_weight', 0.1)
        self.fuel_penalty_weight = config.get('fuel_penalty_weight', 0.05)
        self.safe_distance = config.get('safe_distance', 1000.0)
        self.warning_distance = config.get('warning_distance', 500.0)

    def compute_reward(self, state: np.ndarray, action: np.ndarray, next_state: np.ndarray, done: bool, info: Dict[str, Any]) -> float:
        distance = float(info.get('next_distance', np.linalg.norm(next_state[:3] - state[:3])))
        fuel = float(info.get('fuel_used', np.linalg.norm(action)))
        reward = 0.0

        if info.get('collision', False) or distance < self.warning_distance:
            reward += self.collision_penalty

        if distance >= self.safe_distance and not done:
            reward += min(distance / self.safe_distance, 1.0) * self.distance_reward_weight * 100.0

        reward -= fuel * self.fuel_penalty_weight

        if done and distance >= self.safe_distance:
            reward += self.success_reward

        return float(reward)

    def get_reward_breakdown(self, state: np.ndarray, action: np.ndarray, next_state: np.ndarray, done: bool, info: Dict[str, Any]) -> Dict[str, float]:
        return {
            'collision_penalty': float(self.collision_penalty if info.get('collision', False) or float(info.get('next_distance', 1e6)) < self.warning_distance else 0.0),
            'distance_reward': float(max(0.0, min(float(info.get('next_distance', 0.0)) / self.safe_distance, 1.0) * self.distance_reward_weight * 100.0)),
            'fuel_penalty': float(np.linalg.norm(action) * self.fuel_penalty_weight),
            'success_reward': float(self.success_reward if done and float(info.get('next_distance', 0.0)) >= self.safe_distance else 0.0)
        }


def create_reward_function(config: Dict[str, Any]) -> ManeuverReward:
    """Create the reward function object for RL training."""
    return ManeuverReward(config)
