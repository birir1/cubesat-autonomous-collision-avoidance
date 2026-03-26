"""Benchmark scripts for comparing models"""

import os
import torch
import numpy as np
from tqdm import trange

from phases.phase6_maneuver_planning_rl.environment.orbital_env import OrbitalCollisionEnv
from phases.phase6_maneuver_planning_rl.models.maddpg_agent import MADDPG


def benchmark_maddpg(model_type='standard', episodes=50):
    """Run quickly and report average performance of MADDPG model types."""
    env = OrbitalCollisionEnv(num_objects=5, max_steps=100)

    maddpg = MADDPG(
        num_agents=3,
        state_dim=24,
        action_dim=2,
        model_type=model_type,
        device=torch.device('cpu')
    )

    metrics = {
        'rewards': [],
        'collisions': [],
        'min_distances': []
    }

    for _ in trange(episodes, desc=f"Benchmarking {model_type}"):
        state, info = env.reset()
        episode_reward = 0.0
        collision = False

        for _ in range(env.max_steps):
            actions = []
            for agent_id in range(3):
                s_tensor = torch.FloatTensor(state).unsqueeze(0)
                a = maddpg.agents[agent_id].actor(s_tensor).detach().numpy()[0]
                actions.append(np.clip(a, -1, 1))

            next_state, reward, terminated, truncated, info = env.step(np.array(actions))
            episode_reward += float(reward)
            if info.get('collision', False):
                collision = True
            state = next_state

            if terminated or truncated:
                break

        metrics['rewards'].append(episode_reward)
        metrics['collisions'].append(1 if collision else 0)
        metrics['min_distances'].append(info.get('min_distance', np.nan))

    env.close()

    return {
        'avg_reward': np.mean(metrics['rewards']),
        'collision_rate': np.mean(metrics['collisions']),
        'avg_min_distance': np.nanmean(metrics['min_distances'])
    }


if __name__ == '__main__':
    import torch
    results_std = benchmark_maddpg('standard', episodes=10)
    results_light = benchmark_maddpg('lightweight', episodes=10)

    print('Standard MADDPG:', results_std)
    print('Lightweight MADDPG:', results_light)

