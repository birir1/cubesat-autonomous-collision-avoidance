"""
Evaluation Script for Maneuver RL Agents

Evaluates trained RL agents on collision avoidance maneuver tasks.
"""

import torch
import torch.nn as nn
import numpy as np
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc

from .agents.ppo import PPOAgent
from .agents.sac_agent import SACAgent
from .agents.ddpg_agent import DDPGAgent
from .environment.collision_environment import CollisionEnvironment
from .reward import ManeuverReward

class ManeuverEvaluator:
    """
    Evaluator for maneuver RL agents.
    """

    def __init__(self, config: Dict):
        """
        Initialize evaluator.

        Args:
            config: Evaluation configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Using device: {self.device}")

        # Initialize environment and reward
        self.environment = CollisionEnvironment(config.get('env_config', {}))
        self.reward_function = ManeuverReward(config.get('reward_config', {}))

    def load_agent(self, model_path: str, agent_type: str) -> nn.Module:
        """
        Load trained agent from checkpoint.

        Args:
            model_path: Path to model checkpoint
            agent_type: Type of agent

        Returns:
            Loaded agent
        """
        # Initialize agent
        if agent_type == 'ppo':
            agent = PPOAgent(self.config.get('agent_config', {}))
        elif agent_type == 'sac':
            agent = SACAgent(self.config.get('agent_config', {}))
        elif agent_type == 'ddpg':
            agent = DDPGAgent(self.config.get('agent_config', {}))
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        agent.load_state_dict(checkpoint['agent_state_dict'])
        agent.to(self.device)
        agent.eval()

        self.logger.info(f"Loaded {agent_type} agent from {model_path}")
        return agent

    def evaluate_agent(self, agent: nn.Module, num_episodes: int = 100,
                      render: bool = False) -> Dict[str, Any]:
        """
        Evaluate agent on test episodes.

        Args:
            agent: Trained agent
            num_episodes: Number of evaluation episodes
            render: Whether to render episodes

        Returns:
            Evaluation metrics
        """
        self.logger.info(f"Evaluating agent on {num_episodes} episodes...")

        episode_rewards = []
        episode_lengths = []
        collisions = []
        maneuver_successes = []
        fuel_usage = []
        min_distances = []

        for episode in range(num_episodes):
            episode_reward, episode_length, info = self._evaluate_episode(
                agent, render=render and episode < 5  # Render first 5 episodes
            )

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            collisions.append(info.get('collision', False))
            maneuver_successes.append(info.get('maneuver_success', False))
            fuel_usage.append(info.get('fuel_used', 0.0))
            min_distances.append(info.get('min_distance', float('inf')))

        # Calculate comprehensive metrics
        metrics = self._calculate_metrics(
            episode_rewards, episode_lengths, collisions,
            maneuver_successes, fuel_usage, min_distances
        )

        self.logger.info(f"Evaluation completed. Avg reward: {metrics['avg_reward']:.2f}")
        return metrics

    def _evaluate_episode(self, agent: nn.Module, render: bool = False) -> Tuple[float, int, Dict]:
        """Evaluate single episode."""
        state = self.environment.reset()
        episode_reward = 0
        episode_length = 0
        done = False

        while not done and episode_length < self.config.get('max_episode_length', 1000):
            # Select action (deterministic for evaluation)
            action, _ = agent.select_action(state, deterministic=True)

            # Execute action
            next_state, reward, done, info = self.environment.step(action)

            # Modify reward for evaluation
            modified_reward = self.reward_function.compute_reward(
                state, action, next_state, done, info
            )

            episode_reward += modified_reward
            state = next_state
            episode_length += 1

            if render:
                self.environment.render()

        return episode_reward, episode_length, info

    def _calculate_metrics(self, rewards: List[float], lengths: List[int],
                          collisions: List[bool], successes: List[bool],
                          fuel_usage: List[float], min_distances: List[float]) -> Dict[str, Any]:
        """Calculate comprehensive evaluation metrics."""
        metrics = {}

        # Basic statistics
        metrics['avg_reward'] = np.mean(rewards)
        metrics['std_reward'] = np.std(rewards)
        metrics['min_reward'] = np.min(rewards)
        metrics['max_reward'] = np.max(rewards)

        metrics['avg_episode_length'] = np.mean(lengths)
        metrics['std_episode_length'] = np.std(lengths)

        # Safety metrics
        metrics['collision_rate'] = np.mean(collisions)
        metrics['num_collisions'] = np.sum(collisions)

        # Performance metrics
        metrics['maneuver_success_rate'] = np.mean(successes)
        metrics['num_successes'] = np.sum(successes)

        # Efficiency metrics
        metrics['avg_fuel_usage'] = np.mean(fuel_usage)
        metrics['std_fuel_usage'] = np.std(fuel_usage)
        metrics['total_fuel_usage'] = np.sum(fuel_usage)

        # Distance metrics
        valid_distances = [d for d in min_distances if d != float('inf')]
        if valid_distances:
            metrics['avg_min_distance'] = np.mean(valid_distances)
            metrics['std_min_distance'] = np.std(valid_distances)
            metrics['min_min_distance'] = np.min(valid_distances)
        else:
            metrics['avg_min_distance'] = float('inf')
            metrics['std_min_distance'] = 0.0
            metrics['min_min_distance'] = float('inf')

        # Risk assessment
        metrics['high_risk_episodes'] = np.sum(np.array(min_distances) < 1000)  # < 1km
        metrics['safe_episodes'] = np.sum(np.array(min_distances) >= 1000)

        return metrics

    def compare_agents(self, agent_results: Dict[str, Dict], save_path: str = './comparison'):
        """
        Compare multiple agents.

        Args:
            agent_results: Dictionary of agent_name -> metrics
            save_path: Path to save comparison
        """
        Path(save_path).mkdir(parents=True, exist_ok=True)

        # Create comparison dataframe
        comparison_data = []
        for agent_name, metrics in agent_results.items():
            row = {
                'Agent': agent_name,
                'Avg Reward': metrics['avg_reward'],
                'Collision Rate': metrics['collision_rate'],
                'Success Rate': metrics['maneuver_success_rate'],
                'Avg Fuel': metrics['avg_fuel_usage'],
                'Avg Min Distance': metrics['avg_min_distance']
            }
            comparison_data.append(row)

        import pandas as pd
        df = pd.DataFrame(comparison_data)
        df.to_csv(f"{save_path}/agent_comparison.csv", index=False)

        # Create comparison plot
        metrics_to_plot = ['Avg Reward', 'Collision Rate', 'Success Rate', 'Avg Fuel']
        df_plot = df.set_index('Agent')[metrics_to_plot]

        plt.figure(figsize=(12, 6))
        df_plot.plot(kind='bar', ax=plt.gca())
        plt.title('Agent Comparison')
        plt.ylabel('Value')
        plt.xticks(rotation=45)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(f"{save_path}/agent_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"Agent comparison saved to {save_path}")

    def plot_evaluation_results(self, metrics: Dict[str, Any], save_path: str = './plots'):
        """
        Plot detailed evaluation results.

        Args:
            metrics: Evaluation metrics
            save_path: Path to save plots
        """
        Path(save_path).mkdir(parents=True, exist_ok=True)

        # Create comprehensive evaluation plot
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Maneuver RL Agent Evaluation', fontsize=16)

        # Reward distribution
        axes[0, 0].hist(metrics.get('rewards', []), bins=20, alpha=0.7)
        axes[0, 0].axvline(metrics['avg_reward'], color='red', linestyle='--', label=f'Avg: {metrics["avg_reward"]:.2f}')
        axes[0, 0].set_title('Reward Distribution')
        axes[0, 0].set_xlabel('Reward')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend()

        # Episode lengths
        axes[0, 1].hist(metrics.get('lengths', []), bins=20, alpha=0.7)
        axes[0, 1].axvline(metrics['avg_episode_length'], color='red', linestyle='--',
                          label=f'Avg: {metrics["avg_episode_length"]:.1f}')
        axes[0, 1].set_title('Episode Length Distribution')
        axes[0, 1].set_xlabel('Length')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].legend()

        # Safety metrics
        safety_labels = ['Safe', 'Collision']
        safety_values = [metrics['num_successes'], metrics['num_collisions']]
        axes[0, 2].bar(safety_labels, safety_values, color=['green', 'red'])
        axes[0, 2].set_title('Safety Outcomes')
        axes[0, 2].set_ylabel('Count')

        # Fuel usage
        axes[1, 0].hist(metrics.get('fuel_usage', []), bins=20, alpha=0.7)
        axes[1, 0].axvline(metrics['avg_fuel_usage'], color='red', linestyle='--',
                          label=f'Avg: {metrics["avg_fuel_usage"]:.2f}')
        axes[1, 0].set_title('Fuel Usage Distribution')
        axes[1, 0].set_xlabel('Fuel Used')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()

        # Minimum distance
        valid_distances = [d for d in metrics.get('min_distances', []) if d != float('inf')]
        if valid_distances:
            axes[1, 1].hist(valid_distances, bins=20, alpha=0.7)
            axes[1, 1].axvline(metrics['avg_min_distance'], color='red', linestyle='--',
                              label=f'Avg: {metrics["avg_min_distance"]:.1f}')
            axes[1, 1].set_title('Minimum Distance Distribution')
            axes[1, 1].set_xlabel('Min Distance (km)')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].legend()

        # Performance summary
        summary_text = ".2f"
        axes[1, 2].text(0.1, 0.9, f'Average Reward: {metrics["avg_reward"]:.2f}', fontsize=10)
        axes[1, 2].text(0.1, 0.8, f'Collision Rate: {metrics["collision_rate"]:.3f}', fontsize=10)
        axes[1, 2].text(0.1, 0.7, f'Success Rate: {metrics["maneuver_success_rate"]:.3f}', fontsize=10)
        axes[1, 2].text(0.1, 0.6, f'Avg Fuel Usage: {metrics["avg_fuel_usage"]:.2f}', fontsize=10)
        axes[1, 2].text(0.1, 0.5, f'Avg Min Distance: {metrics["avg_min_distance"]:.1f} km', fontsize=10)
        axes[1, 2].set_title('Performance Summary')
        axes[1, 2].set_xlim(0, 1)
        axes[1, 2].set_ylim(0, 1)
        axes[1, 2].axis('off')

        plt.tight_layout()
        plt.savefig(f"{save_path}/evaluation_results.png", dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"Evaluation plots saved to {save_path}/evaluation_results.png")

    def save_evaluation_report(self, metrics: Dict[str, Any], save_path: str):
        """
        Save detailed evaluation report.

        Args:
            metrics: Evaluation metrics
            save_path: Path to save report
        """
        report = {
            'evaluation_timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'config': self.config
        }

        with open(save_path, 'w') as f:
            json.dump(report, f, indent=2)

        self.logger.info(f"Evaluation report saved to {save_path}")

    def stress_test(self, agent: nn.Module, num_episodes: int = 1000,
                   difficulty_levels: List[str] = None) -> Dict[str, Dict]:
        """
        Perform stress testing with different difficulty levels.

        Args:
            agent: Trained agent
            num_episodes: Number of episodes per difficulty level
            difficulty_levels: List of difficulty levels to test

        Returns:
            Results for each difficulty level
        """
        if difficulty_levels is None:
            difficulty_levels = ['easy', 'medium', 'hard', 'extreme']

        results = {}

        for difficulty in difficulty_levels:
            self.logger.info(f"Stress testing at {difficulty} difficulty...")

            # Modify environment config for difficulty
            env_config = self.config.get('env_config', {}).copy()
            env_config['difficulty'] = difficulty

            # Create environment with modified config
            test_env = CollisionEnvironment(env_config)

            # Temporarily replace environment
            original_env = self.environment
            self.environment = test_env

            # Evaluate at this difficulty
            metrics = self.evaluate_agent(agent, num_episodes)

            results[difficulty] = metrics

            # Restore original environment
            self.environment = original_env

        return results


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate maneuver RL agent')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--agent_type', type=str, default='ppo',
                       choices=['ppo', 'sac', 'ddpg'],
                       help='Type of RL agent')
    parser.add_argument('--num_episodes', type=int, default=100,
                       help='Number of evaluation episodes')
    parser.add_argument('--config', type=str, default='./configs/maneuver_rl_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--save_path', type=str, default='./results/evaluation',
                       help='Path to save evaluation results')
    parser.add_argument('--render', action='store_true',
                       help='Render evaluation episodes')
    parser.add_argument('--stress_test', action='store_true',
                       help='Perform stress testing')

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Load configuration
    import yaml
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Initialize evaluator
    evaluator = ManeuverEvaluator(config)

    # Load agent
    agent = evaluator.load_agent(args.model_path, args.agent_type)

    if args.stress_test:
        # Perform stress testing
        stress_results = evaluator.stress_test(agent, num_episodes=args.num_episodes // 4)
        evaluator.compare_agents(stress_results, f"{args.save_path}/stress_test")

        # Save stress test results
        with open(f"{args.save_path}/stress_test_results.json", 'w') as f:
            json.dump(stress_results, f, indent=2)
    else:
        # Standard evaluation
        metrics = evaluator.evaluate_agent(agent, args.num_episodes, args.render)

        # Save results
        evaluator.plot_evaluation_results(metrics, args.save_path)
        evaluator.save_evaluation_report(metrics, f"{args.save_path}/evaluation_report.json")

        # Print summary
        print("Evaluation Results:")
        print(f"Average Reward: {metrics['avg_reward']:.2f} ± {metrics['std_reward']:.2f}")
        print(f"Collision Rate: {metrics['collision_rate']:.3f}")
        print(f"Maneuver Success Rate: {metrics['maneuver_success_rate']:.3f}")
        print(f"Average Fuel Usage: {metrics['avg_fuel_usage']:.2f}")
        print(f"Average Min Distance: {metrics['avg_min_distance']:.1f} km")

    print(f"Evaluation completed. Results saved to {args.save_path}")


if __name__ == "__main__":
    main()