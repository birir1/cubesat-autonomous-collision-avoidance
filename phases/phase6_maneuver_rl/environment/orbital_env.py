"""Orbital collision environment for maneuver RL benchmarks."""

import numpy as np


class OrbitalCollisionEnv:
    """Minimal environment for benchmarking maneuver planning agents."""

    def __init__(self, num_objects: int = 5, max_steps: int = 100):
        self.num_objects = num_objects
        self.max_steps = max_steps
        self.state_dim = 24
        self.action_dim = 2
        self.reset()

    def reset(self):
        self.step_count = 0
        self.current_state = np.zeros(self.state_dim, dtype=np.float32)
        self.current_info = {
            'collision': False,
            'min_distance': float('inf'),
            'collision_risk': 0.0
        }
        return self.current_state, self.current_info

    def step(self, action):
        self.step_count += 1
        self.current_state = np.zeros(self.state_dim, dtype=np.float32)
        self.current_info = {
            'collision': False,
            'min_distance': float(10000.0 - self.step_count * 10.0),
            'collision_risk': 0.0
        }
        terminated = self.step_count >= self.max_steps
        truncated = False
        reward = -1.0
        return self.current_state, reward, terminated, truncated, self.current_info

    def render(self):
        pass

    def close(self):
        pass
