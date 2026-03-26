"""
MADDPG Training Script for Multi-Agent Collision Avoidance
"""

import os
import sys
import yaml
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import logging

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from phases.phase6_maneuver_planning_rl.environment.orbital_env import OrbitalCollisionEnv
from phases.phase6_maneuver_planning_rl.models.maddpg_agent import MADDPG, ReplayBuffer
from utils.logger import setup_logger
from utils.experiment_tracker import ExperimentTracker

logger = setup_logger(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ================================================
# SAFE SAVE FUNCTION (FIXES YOUR ERROR)
# ================================================
def safe_save_model(maddpg, path):
    """
    Saves MADDPG model safely even if .save() is not implemented
    """
    try:
        # Try native save (if implemented)
        maddpg.save(path)
    except AttributeError:
        # Fallback manual checkpoint
        checkpoint = {}

        if hasattr(maddpg, 'agents'):
            checkpoint['actors'] = []
            checkpoint['critics'] = []

            for agent in maddpg.agents:
                if hasattr(agent, 'actor'):
                    checkpoint['actors'].append(agent.actor.state_dict())
                if hasattr(agent, 'critic'):
                    checkpoint['critics'].append(agent.critic.state_dict())

                if hasattr(agent, 'target_actor'):
                    checkpoint.setdefault('target_actors', []).append(agent.target_actor.state_dict())
                if hasattr(agent, 'target_critic'):
                    checkpoint.setdefault('target_critics', []).append(agent.target_critic.state_dict())

                if hasattr(agent, 'actor_optimizer'):
                    checkpoint.setdefault('actor_optimizers', []).append(agent.actor_optimizer.state_dict())
                if hasattr(agent, 'critic_optimizer'):
                    checkpoint.setdefault('critic_optimizers', []).append(agent.critic_optimizer.state_dict())

        torch.save(checkpoint, path)


# ================================================
# CONFIG
# ================================================
def load_config(config_path="configs/maddpg_config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


# ================================================
# TRAINING LOOP
# ================================================
def train_maddpg(config=None, verbose=True):

    if config is None:
        config = load_config()

    # Directories
    model_dir = config['checkpointing']['checkpoint_dir']
    metrics_dir = config['training']['metrics_dir']
    log_dir = config['training']['tensorboard_log_dir']

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Environment
    env_config = config['environment']
    env = OrbitalCollisionEnv(
        num_objects=env_config['num_objects'],
        safe_distance=env_config['safe_distance'],
        collision_distance=env_config['collision_distance'],
        max_steps=env_config['max_steps'],
        dt=env_config['dt'],
        max_delta_v=env_config['max_delta_v'],
        world_size=env_config['world_size']
    )

    num_agents = env_config['num_agents']
    state_dim = config['actor_network']['state_dim']
    action_dim = config['actor_network']['action_dim']

    maddpg = MADDPG(
        num_agents=num_agents,
        state_dim=state_dim,
        action_dim=action_dim,
        model_type=config.get('model_type', 'standard'),
        device=DEVICE
    )

    replay_buffer = ReplayBuffer(buffer_size=config['training']['buffer_size'])

    tracker = ExperimentTracker(
        log_dir=log_dir,
        config=config,
        log_name="maddpg_training"
    )

    max_episodes = config['training']['max_episodes']
    batch_size = config['training']['batch_size']
    save_interval = config['checkpointing']['save_interval']
    warmup_episodes = config['training']['warmup_episodes']

    # Noise
    noise_scale = config['exploration']['initial_noise_scale']
    final_noise = config['exploration']['final_noise_scale']
    noise_decay = max(1, config['exploration']['noise_decay_episodes'])

    metrics = {
        'episode': [],
        'reward': [],
        'collision_rate': [],
        'min_distance': [],
        'episode_length': [],
        'actor_loss': [],
        'critic_loss': []
    }

    best_reward = float('-inf')
    collision_count = 0

    if verbose:
        logger.info(f"Starting MADDPG training on {DEVICE}")
        logger.info(f"Environment: {env_config.get('name', 'OrbitalCollisionEnv')} with {num_agents} agents")
        logger.info(f"Training for {max_episodes} episodes")

    # ================================================
    # TRAINING LOOP
    # ================================================
    for episode in tqdm(range(max_episodes), disable=not verbose, desc="MADDPG Training"):

        state, _ = env.reset()
        episode_reward = 0
        episode_collision = False
        min_distance = float('inf')

        # Noise decay (safe)
        decay_ratio = min(1.0, episode / noise_decay)
        current_noise = noise_scale - (noise_scale - final_noise) * decay_ratio

        for step in range(env_config['max_steps']):

            actions = []
            for agent_id in range(num_agents):
                agent_state = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    action = maddpg.agents[agent_id].actor(agent_state).cpu().numpy()

                noise = np.random.normal(0, current_noise, size=action.shape)
                action = np.clip(action + noise, -1, 1)

                actions.append(action[0])

            actions_array = np.array(actions)

            next_state, reward, terminated, truncated, info = env.step(actions_array)

            episode_reward += reward
            min_distance = min(min_distance, info.get('min_distance', float('inf')))

            if info.get('collision', False):
                episode_collision = True
                collision_count += 1

            replay_buffer.push(
                state=state,
                action=actions_array,
                reward=reward,
                next_state=next_state,
                done=terminated or truncated
            )

            state = next_state

            if terminated or truncated:
                break

        # ================================================
        # TRAINING UPDATE
        # ================================================
        actor_loss, critic_loss = 0, 0

        if episode > warmup_episodes and len(replay_buffer) > batch_size:
            for _ in range(env_config['max_steps']):
                losses = maddpg.update(batch_size)

                if isinstance(losses, tuple):
                    actor_loss += float(losses[0])
                    critic_loss += float(losses[1])

        collision_rate = collision_count / (episode + 1)

        metrics['episode'].append(episode)
        metrics['reward'].append(episode_reward)
        metrics['collision_rate'].append(collision_rate)
        metrics['min_distance'].append(min_distance if min_distance != float('inf') else -1)
        metrics['episode_length'].append(step + 1)
        metrics['actor_loss'].append(actor_loss)
        metrics['critic_loss'].append(critic_loss)

        tracker.log({
            'episode': episode,
            'reward': episode_reward,
            'collision': episode_collision,
            'min_distance': min_distance,
            'noise_scale': current_noise,
            'actor_loss': actor_loss,
            'critic_loss': critic_loss,
            'buffer_size': len(replay_buffer)
        })

        # ================================================
        # CHECKPOINTING (FIXED)
        # ================================================
        if (episode + 1) % save_interval == 0 or episode_reward > best_reward:

            save_path = os.path.join(model_dir, f"maddpg_cubesat_ep{episode}.pth")
            safe_save_model(maddpg, save_path)

            if episode_reward > best_reward:
                best_reward = episode_reward
                best_path = os.path.join(model_dir, "maddpg_cubesat_best.pth")
                safe_save_model(maddpg, best_path)

        if episode == max_episodes - 1:
            final_path = os.path.join(model_dir, "maddpg_cubesat_final.pth")
            safe_save_model(maddpg, final_path)

    # ================================================
    # SAVE METRICS
    # ================================================
    df_metrics = pd.DataFrame(metrics)
    metrics_path = os.path.join(metrics_dir, "maddpg_training_metrics.csv")
    df_metrics.to_csv(metrics_path, index=False)

    if verbose:
        logger.info(f"Metrics saved to {metrics_path}")
        logger.info(f"Final collision rate: {collision_rate:.2%}")
        logger.info(f"Avg reward (last 100): {np.mean(metrics['reward'][-100:]):.2f}")

    env.close()
    tracker.close()

    return {
        'metrics': metrics,
        'model': maddpg,
        'config': config,
        'device': DEVICE
    }


# ================================================
# MAIN
# ================================================
if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/maddpg_config.yaml')
    parser.add_argument('--no-verbose', action='store_true')

    args = parser.parse_args()

    config = load_config(args.config)

    train_maddpg(config=config, verbose=not args.no_verbose)

    print("\n" + "="*50)
    print("MADDPG Training Complete")
    print("="*50)