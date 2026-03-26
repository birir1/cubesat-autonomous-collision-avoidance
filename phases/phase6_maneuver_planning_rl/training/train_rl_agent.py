"""
Training Script for CubeSat Collision Avoidance RL Agents

This script trains:
1) PPO agent (single satellite control)
2) MADDPG agents (multi-satellite coordination)

Environment:
OrbitalCollisionEnv

Outputs:
results/saved_models/
results/metrics/
"""

import os
import numpy as np
import torch
from tqdm import tqdm

from phases.phase6_maneuver_planning_rl.environment.orbital_env import OrbitalCollisionEnv
from phases.phase6_maneuver_planning_rl.models.ppo_agent import PPO
from phases.phase6_maneuver_planning_rl.models.maddpg_agent import MADDPG


# ------------------------------------------------
# CONFIGURATION
# ------------------------------------------------

MAX_EPISODES = 5000
MAX_STEPS = 300
SAVE_INTERVAL = 500

RESULTS_DIR = "results"
MODEL_DIR = os.path.join(RESULTS_DIR, "saved_models")
METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)


# ------------------------------------------------
# PPO TRAINING
# ------------------------------------------------

def train_ppo():

    print("\nStarting PPO Training")

    env = OrbitalCollisionEnv()

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = PPO(state_dim, action_dim)

    reward_history = []

    episode_bar = tqdm(range(MAX_EPISODES), desc="PPO Training")

    for episode in episode_bar:

        state, _ = env.reset()
        state = np.array(state, dtype=np.float32)

        episode_reward = 0

        for step in range(MAX_STEPS):

            action = agent.select_action(state)

            next_state, reward, terminated, truncated, info = env.step(action)

            done = terminated or truncated

            next_state = np.array(next_state, dtype=np.float32)

            agent.buffer.rewards.append(reward)
            agent.buffer.dones.append(done)

            state = next_state
            episode_reward += reward

            if done:
                break

        agent.update()

        reward_history.append(episode_reward)

        avg_reward = np.mean(reward_history[-100:])

        episode_bar.set_postfix({
            "episode_reward": round(episode_reward, 2),
            "avg100": round(avg_reward, 2)
        })

        if episode % SAVE_INTERVAL == 0 and episode > 0:
            model_path = os.path.join(MODEL_DIR, f"ppo_cubesat_ep{episode}.pth")
            agent.save(model_path)
            print("Model saved:", model_path)

    final_model = os.path.join(MODEL_DIR, "ppo_cubesat_final.pth")
    agent.save(final_model)

    print("\nPPO Training Finished")
    print("Final model:", final_model)


# ------------------------------------------------
# MADDPG TRAINING
# ------------------------------------------------

def train_maddpg():

    print("\nStarting MADDPG Training")

    num_agents = 3

    envs = [OrbitalCollisionEnv() for _ in range(num_agents)]

    state_dim = envs[0].observation_space.shape[0]
    action_dim = envs[0].action_space.shape[0]

    maddpg = MADDPG(
        num_agents=num_agents,
        state_dim=state_dim,
        action_dim=action_dim
    )

    reward_history = []

    episode_bar = tqdm(range(MAX_EPISODES), desc="MADDPG Training")

    for episode in episode_bar:

        states = []

        for env in envs:
            s, _ = env.reset()
            states.append(np.array(s, dtype=np.float32))

        episode_rewards = np.zeros(num_agents)

        for step in range(MAX_STEPS):

            actions = maddpg.select_actions(states)

            next_states = []
            rewards = []
            dones = []

            for i, env in enumerate(envs):

                ns, r, terminated, truncated, info = env.step(actions[i])

                done = terminated or truncated

                ns = np.array(ns, dtype=np.float32)

                next_states.append(ns)
                rewards.append(r)
                dones.append(done)

                episode_rewards[i] += r

            maddpg.replay_buffer.push(
                states,
                actions,
                rewards,
                next_states,
                dones
            )

            states = next_states

            maddpg.update()

            if any(dones):
                break

        avg_reward = np.mean(episode_rewards)
        reward_history.append(avg_reward)

        avg_last = np.mean(reward_history[-100:])

        episode_bar.set_postfix({
            "avg_reward": round(avg_reward, 2),
            "avg100": round(avg_last, 2)
        })

    print("\nMADDPG Training Finished")


# ------------------------------------------------
# MAIN
# ------------------------------------------------

if __name__ == "__main__":

    print("\nCubeSat Collision Avoidance RL Training")

    train_ppo()

    train_maddpg()

    print("\nTraining Complete")