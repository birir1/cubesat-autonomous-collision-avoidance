"""
Training Script for Maneuver RL Agents

Trains reinforcement learning agents for autonomous collision avoidance maneuvers.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

from .agents.ppo import PPOAgent
from .agents.sac_agent import SACAgent
from .agents.ddpg_agent import DDPGAgent
from .environment.collision_environment import CollisionEnvironment
from .reward import ManeuverReward
from .evaluate import ManeuverEvaluator

class ManeuverRLTrainer:
    """
    Trainer for maneuver RL agents.
    """

    def __init__(self, config: Dict):
        """
        Initialize trainer.

        Args:
            config: Training configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Using device: {self.device}")

        # Initialize components
        self.environment = CollisionEnvironment(config.get('env_config', {}))
        self.reward_function = ManeuverReward(config.get('reward_config', {}))

        # Initialize agent
        self.agent = self._initialize_agent()
        self.agent.to(self.device)

        # Initialize metrics tracking
        self.metrics_history = {
            'episode_rewards': [],
            'episode_lengths': [],
            'collision_rates': [],
            'maneuver_success_rates': [],
            'fuel_efficiency': []
        }

    def _initialize_agent(self):
        """Initialize the RL agent based on configuration."""
        agent_type = self.config.get('agent_type', 'ppo')

        if agent_type.lower() == 'ppo':
            agent = PPOAgent(self.config.get('agent_config', {}))
        elif agent_type.lower() == 'sac':
            agent = SACAgent(self.config.get('agent_config', {}))
        elif agent_type.lower() == 'ddpg':
            agent = DDPGAgent(self.config.get('agent_config', {}))
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")

        self.logger.info(f"Initialized {agent_type.upper()} agent")
        return agent

    def train_episode(self) -> Dict[str, float]:
        """Train for one episode."""
        state = self.environment.reset()
        episode_reward = 0
        episode_length = 0
        done = False

        episode_states = []
        episode_actions = []
        episode_rewards = []
        episode_log_probs = []

        while not done and episode_length < self.config.get('max_episode_length', 1000):
            # Select action
            action, log_prob = self.agent.select_action(state)

            # Execute action
            next_state, reward, done, info = self.environment.step(action)

            # Modify reward
            modified_reward = self.reward_function.compute_reward(
                state, action, next_state, done, info
            )

            # Store transition
            episode_states.append(state)
            episode_actions.append(action)
            episode_rewards.append(modified_reward)
            episode_log_probs.append(log_prob)

            # Update state
            state = next_state
            episode_reward += modified_reward
            episode_length += 1

        # Update agent
        if hasattr(self.agent, 'update'):
            metrics = self.agent.update(episode_states, episode_actions,
                                      episode_rewards, episode_log_probs)
        else:
            metrics = {}

        # Episode metrics
        episode_metrics = {
            'episode_reward': episode_reward,
            'episode_length': episode_length,
            'collision': info.get('collision', False),
            'maneuver_success': info.get('maneuver_success', False),
            'fuel_used': info.get('fuel_used', 0.0)
        }

        episode_metrics.update(metrics)
        return episode_metrics

    def train(self, num_episodes: int, save_path: str = './results/maneuver_rl',
              eval_frequency: int = 100):
        """
        Train the agent.

        Args:
            num_episodes: Number of episodes to train
            save_path: Path to save results
            eval_frequency: How often to evaluate
        """
        Path(save_path).mkdir(parents=True, exist_ok=True)
        best_reward = float('-inf')

        self.logger.info(f"Starting training for {num_episodes} episodes")

        for episode in range(num_episodes):
            # Train episode
            episode_metrics = self.train_episode()

            # Log episode metrics
            self.logger.info(f"Episode {episode+1}/{num_episodes}")
            self.logger.info(f"Reward: {episode_metrics['episode_reward']:.2f}, "
                           f"Length: {episode_metrics['episode_length']}")

            # Store metrics
            self.metrics_history['episode_rewards'].append(episode_metrics['episode_reward'])
            self.metrics_history['episode_lengths'].append(episode_metrics['episode_length'])
            self.metrics_history['collision_rates'].append(episode_metrics['collision'])
            self.metrics_history['maneuver_success_rates'].append(episode_metrics['maneuver_success'])
            self.metrics_history['fuel_efficiency'].append(episode_metrics.get('fuel_used', 0))

            # Evaluate periodically
            if (episode + 1) % eval_frequency == 0:
                eval_metrics = self.evaluate(num_eval_episodes=10)
                self.logger.info(f"Evaluation - Avg Reward: {eval_metrics['avg_reward']:.2f}")

                # Save best model
                if eval_metrics['avg_reward'] > best_reward:
                    best_reward = eval_metrics['avg_reward']
                    self.save_checkpoint(episode, eval_metrics, f"{save_path}/best_model.pth")

            # Save latest model
            if (episode + 1) % 1000 == 0:
                self.save_checkpoint(episode, episode_metrics, f"{save_path}/model_episode_{episode+1}.pth")

        # Save final metrics
        self.save_metrics(f"{save_path}/training_metrics.json")

        # Plot training history
        self.plot_training_history(save_path)

        self.logger.info("Training completed")

    def evaluate(self, num_eval_episodes: int = 100) -> Dict[str, float]:
        """Evaluate the agent."""
        self.agent.eval()

        episode_rewards = []
        collision_count = 0
        success_count = 0
        total_fuel = 0

        for _ in range(num_eval_episodes):
            state = self.environment.reset()
            episode_reward = 0
            done = False
            episode_length = 0

            while not done and episode_length < self.config.get('max_episode_length', 1000):
                # Select action (deterministic for evaluation)
                action, _ = self.agent.select_action(state, deterministic=True)

                # Execute action
                next_state, reward, done, info = self.environment.step(action)

                episode_reward += reward
                state = next_state
                episode_length += 1

            episode_rewards.append(episode_reward)
            if info.get('collision', False):
                collision_count += 1
            if info.get('maneuver_success', False):
                success_count += 1
            total_fuel += info.get('fuel_used', 0)

        eval_metrics = {
            'avg_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'collision_rate': collision_count / num_eval_episodes,
            'success_rate': success_count / num_eval_episodes,
            'avg_fuel': total_fuel / num_eval_episodes
        }

        return eval_metrics

    def save_checkpoint(self, episode: int, metrics: Dict, path: str):
        """Save model checkpoint."""
        checkpoint = {
            'episode': episode,
            'agent_state_dict': self.agent.state_dict(),
            'metrics': metrics,
            'config': self.config
        }
        torch.save(checkpoint, path)
        self.logger.info(f"Checkpoint saved to {path}")

    def save_metrics(self, path: str):
        """Save training metrics."""
        with open(path, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
        self.logger.info(f"Metrics saved to {path}")

    def plot_training_history(self, save_path: str = './plots'):
        """Plot training history."""
        Path(save_path).mkdir(parents=True, exist_ok=True)

        # Plot episode rewards
        plt.figure(figsize=(15, 10))

        plt.subplot(2, 3, 1)
        plt.plot(self.metrics_history['episode_rewards'])
        plt.title('Episode Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Reward')

        # Plot episode lengths
        plt.subplot(2, 3, 2)
        plt.plot(self.metrics_history['episode_lengths'])
        plt.title('Episode Lengths')
        plt.xlabel('Episode')
        plt.ylabel('Length')

        # Plot collision rates
        plt.subplot(2, 3, 3)
        plt.plot(self.metrics_history['collision_rates'])
        plt.title('Collision Rate')
        plt.xlabel('Episode')
        plt.ylabel('Rate')

        # Plot success rates
        plt.subplot(2, 3, 4)
        plt.plot(self.metrics_history['maneuver_success_rates'])
        plt.title('Maneuver Success Rate')
        plt.xlabel('Episode')
        plt.ylabel('Rate')

        # Plot fuel efficiency
        plt.subplot(2, 3, 5)
        plt.plot(self.metrics_history['fuel_efficiency'])
        plt.title('Fuel Usage')
        plt.xlabel('Episode')
        plt.ylabel('Fuel')

        # Plot moving averages
        plt.subplot(2, 3, 6)
        window = 100
        if len(self.metrics_history['episode_rewards']) >= window:
            rewards_ma = np.convolve(self.metrics_history['episode_rewards'],
                                    np.ones(window)/window, mode='valid')
            plt.plot(rewards_ma)
        plt.title(f'Rewards (Moving Average, window={window})')
        plt.xlabel('Episode')
        plt.ylabel('Reward')

        plt.tight_layout()
        plt.savefig(f"{save_path}/training_history.png", dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"Training plots saved to {save_path}/training_history.png")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train maneuver RL agent')
    parser.add_argument('--agent_type', type=str, default='ppo',
                       choices=['ppo', 'sac', 'ddpg'],
                       help='Type of RL agent')
    parser.add_argument('--num_episodes', type=int, default=10000,
                       help='Number of training episodes')
    parser.add_argument('--config', type=str, default='./configs/maneuver_rl_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--save_path', type=str, default='./results/maneuver_rl',
                       help='Path to save results')
    parser.add_argument('--eval_frequency', type=int, default=100,
                       help='Evaluation frequency')

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Load configuration
    import yaml
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Override config with command line args
    config['agent_type'] = args.agent_type

    # Initialize trainer
    trainer = ManeuverRLTrainer(config)

    # Train agent
    trainer.train(args.num_episodes, args.save_path, args.eval_frequency)

    print(f"Training completed. Results saved to {args.save_path}")


if __name__ == "__main__":
    main()