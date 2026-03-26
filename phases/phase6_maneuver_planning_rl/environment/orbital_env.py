"""
Orbital Collision Avoidance Reinforcement Learning Environment

This environment simulates simplified orbital dynamics for CubeSat
collision avoidance research.

State Representation
--------------------
For each object:
[x, y, vx, vy]

Objects include:
- Controlled CubeSat
- Nearby satellites / debris

Action
------
Continuous thruster maneuver:
[delta_vx, delta_vy]

Reward Objectives
-----------------
+ Maintain safe distance from objects
+ Minimize fuel usage
+ Avoid collisions

Episode Termination
-------------------
- Collision occurs
- Maximum simulation steps reached
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class OrbitalCollisionEnv(gym.Env):

    metadata = {"render_modes": []}

    def __init__(
        self,
        num_objects=5,
        safe_distance=20.0,
        collision_distance=2.0,
        max_steps=300,
        dt=1.0,
        max_delta_v=0.3,
        world_size=200
    ):
        super(OrbitalCollisionEnv, self).__init__()

        self.num_objects = num_objects
        self.safe_distance = safe_distance
        self.collision_distance = collision_distance
        self.max_steps = max_steps
        self.dt = dt
        self.max_delta_v = max_delta_v
        self.world_size = world_size

        self.state_dim = (num_objects + 1) * 4

        # ------------------------------------------------
        # ACTION SPACE
        # ------------------------------------------------
        self.action_space = spaces.Box(
            low=-max_delta_v,
            high=max_delta_v,
            shape=(2,),
            dtype=np.float32
        )

        # ------------------------------------------------
        # OBSERVATION SPACE
        # ------------------------------------------------
        obs_limit = world_size * 5

        self.observation_space = spaces.Box(
            low=-obs_limit,
            high=obs_limit,
            shape=(self.state_dim,),
            dtype=np.float32
        )

        self.seed()
        self.reset()

    # ------------------------------------------------
    # RANDOM SEED
    # ------------------------------------------------

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    # ------------------------------------------------
    # RESET ENVIRONMENT
    # ------------------------------------------------

    def reset(self, seed=None, options=None):

        if seed is not None:
            self.seed(seed)

        self.step_count = 0

        # Controlled CubeSat
        self.cubesat = np.array([
            self.np_random.uniform(-50, 50),
            self.np_random.uniform(-50, 50),
            self.np_random.uniform(-1, 1),
            self.np_random.uniform(-1, 1)
        ], dtype=np.float32)

        # Nearby objects
        self.objects = []

        for _ in range(self.num_objects):

            obj = np.array([
                self.np_random.uniform(-100, 100),
                self.np_random.uniform(-100, 100),
                self.np_random.uniform(-1, 1),
                self.np_random.uniform(-1, 1)
            ], dtype=np.float32)

            self.objects.append(obj)

        return self._get_state(), {}

    # ------------------------------------------------
    # STEP FUNCTION
    # ------------------------------------------------

    def step(self, action):

        self.step_count += 1

        # Convert action safely to numpy
        action = np.array(action, dtype=np.float32).flatten()

        # Clip thrust
        action = np.clip(action, -self.max_delta_v, self.max_delta_v)

        # ------------------------------------------------
        # APPLY MANEUVER (delta-v)
        # ------------------------------------------------
        self.cubesat[2] += float(action[0])
        self.cubesat[3] += float(action[1])

        # ------------------------------------------------
        # UPDATE CUBESAT POSITION
        # ------------------------------------------------
        self.cubesat[0] += self.cubesat[2] * self.dt
        self.cubesat[1] += self.cubesat[3] * self.dt

        # ------------------------------------------------
        # UPDATE OTHER OBJECTS
        # ------------------------------------------------
        for obj in self.objects:
            obj[0] += obj[2] * self.dt
            obj[1] += obj[3] * self.dt

        # ------------------------------------------------
        # BOUNDARY LIMITS
        # ------------------------------------------------
        self.cubesat[:2] = np.clip(
            self.cubesat[:2],
            -self.world_size,
            self.world_size
        )

        # ------------------------------------------------
        # REWARD
        # ------------------------------------------------
        reward, min_distance = self._compute_reward(action)

        collision = min_distance < self.collision_distance

        terminated = bool(collision)
        truncated = self.step_count >= self.max_steps

        info = {
            "min_distance": float(min_distance),
            "collision": bool(collision),
            "step": int(self.step_count)
        }

        return self._get_state(), float(reward), terminated, truncated, info

    # ------------------------------------------------
    # STATE VECTOR
    # ------------------------------------------------

    def _get_state(self):

        state = list(self.cubesat)

        for obj in self.objects:
            state.extend(obj)

        return np.array(state, dtype=np.float32)

    # ------------------------------------------------
    # REWARD FUNCTION
    # ------------------------------------------------

    def _compute_reward(self, action):

        reward = 0.0

        # Fuel penalty
        fuel_cost = np.linalg.norm(action)
        reward -= 0.5 * fuel_cost

        min_distance = float("inf")

        for obj in self.objects:

            dist = np.linalg.norm(self.cubesat[:2] - obj[:2])

            min_distance = min(min_distance, dist)

            if dist < self.collision_distance:

                reward -= 100

            elif dist < self.safe_distance:

                reward -= 5

            else:

                reward += 0.2

        return reward, min_distance

    # ------------------------------------------------
    # OPTIONAL RENDER
    # ------------------------------------------------

    def render(self):

        print("\n--- Environment State ---")
        print("CubeSat:", self.cubesat)

        for i, obj in enumerate(self.objects):
            print(f"Object {i}:", obj)

    # ------------------------------------------------
    # CLOSE
    # ------------------------------------------------

    def close(self):
        pass