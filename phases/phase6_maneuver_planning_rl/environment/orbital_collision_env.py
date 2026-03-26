"""
Orbital Collision Avoidance Environment

Reinforcement Learning environment for training agents to perform
collision avoidance maneuvers for CubeSats.

State:
[x, y, vx, vy] for the controlled CubeSat + nearby objects

Action:
thruster adjustments (delta-v)

Reward:
+ safe distance
- collision risk
- fuel consumption
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


# ----------------------------------------------------
# ENVIRONMENT
# ----------------------------------------------------

class OrbitalCollisionEnv(gym.Env):

    def __init__(
        self,
        num_objects=5,
        safe_distance=15.0,
        max_steps=200
    ):

        super().__init__()

        self.num_objects = num_objects
        self.safe_distance = safe_distance
        self.max_steps = max_steps

        self.dt = 1.0

        # state: cubeSat + other objects
        self.state_dim = (num_objects + 1) * 4

        self.action_space = spaces.Box(
            low=-0.5,
            high=0.5,
            shape=(2,),
            dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.state_dim,),
            dtype=np.float32
        )

        self.reset()

    # ------------------------------------------------

    def reset(self):

        self.step_count = 0

        # CubeSat initial state
        self.cubesat = np.array([
            np.random.uniform(-50, 50),
            np.random.uniform(-50, 50),
            np.random.uniform(-1, 1),
            np.random.uniform(-1, 1)
        ])

        # Other objects (debris/satellites)
        self.objects = []

        for _ in range(self.num_objects):

            obj = np.array([
                np.random.uniform(-50, 50),
                np.random.uniform(-50, 50),
                np.random.uniform(-1, 1),
                np.random.uniform(-1, 1)
            ])

            self.objects.append(obj)

        return self._get_state(), {}

    # ------------------------------------------------

    def step(self, action):

        self.step_count += 1

        # apply maneuver
        self.cubesat[2] += action[0]
        self.cubesat[3] += action[1]

        # update cubeSat position
        self.cubesat[0] += self.cubesat[2] * self.dt
        self.cubesat[1] += self.cubesat[3] * self.dt

        # update other objects
        for obj in self.objects:

            obj[0] += obj[2] * self.dt
            obj[1] += obj[3] * self.dt

        reward = self._compute_reward(action)

        terminated = self._check_collision()
        truncated = self.step_count >= self.max_steps

        return self._get_state(), reward, terminated, truncated, {}

    # ------------------------------------------------

    def _get_state(self):

        state = list(self.cubesat)

        for obj in self.objects:
            state.extend(obj)

        return np.array(state, dtype=np.float32)

    # ------------------------------------------------

    def _compute_reward(self, action):

        reward = 0

        # fuel penalty
        fuel_penalty = np.linalg.norm(action)
        reward -= 0.1 * fuel_penalty

        # distance reward
        for obj in self.objects:

            dist = np.linalg.norm(
                self.cubesat[:2] - obj[:2]
            )

            if dist < self.safe_distance:

                reward -= 5

            else:

                reward += 0.1

        return reward

    # ------------------------------------------------

    def _check_collision(self):

        for obj in self.objects:

            dist = np.linalg.norm(
                self.cubesat[:2] - obj[:2]
            )

            if dist < 2:

                return True

        return False