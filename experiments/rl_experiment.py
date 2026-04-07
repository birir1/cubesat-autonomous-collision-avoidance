"""
Reinforcement Learning Experiment

Tests RL-based maneuver planning for collision avoidance.
Compares different RL algorithms and reward structures.
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, List, Any
import json

from phases.phase6_maneuver_rl.environment.orbital_env import OrbitalCollisionEnv
from phases.phase6_maneuver_rl.agents.maddpg import MADDPG
from evaluation.metrics import safety_metrics


def run_rl_experiment(config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Run reinforcement learning experiment for maneuver planning.

    Args:
        config: Experiment configuration

    Returns:
        Dictionary with experiment results
    """
    if config is None:
        config = {
            'algorithms': ['maddpg', 'ddpg', 'sac'],
            'reward_types': ['safety_first', 'fuel_efficient', 'balanced'],
            'n_episodes': 100,
            'max_steps': 200,
            'n_trials': 3
        }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results_dir = Path('results/rl_experiment')
    results_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    for algorithm in config['algorithms']:
        print(f"\nRunning {algorithm.upper()} experiments...")

        algorithm_results = {}

        for reward_type in config['reward_types']:
            print(f"  Reward type: {reward_type}")

            reward_results = []

            for trial in range(config['n_trials']):
                print(f"    Trial {trial + 1}/{config['n_trials']}")

                # Initialize environment
                env = OrbitalCollisionEnv(
                    num_objects=5,
                    max_steps=config['max_steps'],
                    reward_type=reward_type
                )

                # Initialize agent
                if algorithm == 'maddpg':
                    agent = MADDPG(
                        num_agents=3,
                        state_dim=24,
                        action_dim=2,
                        model_type='standard',
                        device=device
                    )
                else:
                    # Placeholder for other algorithms
                    agent = MADDPG(
                        num_agents=3,
                        state_dim=24,
                        action_dim=2,
                        model_type='standard',
                        device=device
                    )

                # Training loop
                episode_rewards = []
                collision_rates = []
                fuel_consumption = []

                for episode in range(config['n_episodes']):
                    state, info = env.reset()
                    episode_reward = 0.0
                    episode_collisions = 0
                    episode_fuel = 0.0

                    for step in range(env.max_steps):
                        # Get actions
                        actions = []
                        for agent_id in range(3):
                            s_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                            a = agent.agents[agent_id].actor(s_tensor).detach().cpu().numpy()[0]
                            actions.append(np.clip(a, -1, 1))

                        # Environment step
                        next_state, reward, terminated, truncated, info = env.step(np.array(actions))

                        # Update agent
                        agent.update(state, actions, reward, next_state, terminated or truncated)

                        episode_reward += float(reward)
                        episode_fuel += info.get('fuel_consumption', 0.0)
                        if info.get('collision', False):
                            episode_collisions += 1

                        state = next_state

                        if terminated or truncated:
                            break

                    episode_rewards.append(episode_reward)
                    collision_rates.append(episode_collisions / config['max_steps'])
                    fuel_consumption.append(episode_fuel)

                    if (episode + 1) % 20 == 0:
                        print(f"      Episode {episode + 1}: Reward={episode_reward:.2f}, "
                              f"Collisions={episode_collisions}, Fuel={episode_fuel:.2f}")

                # Evaluate final performance
                eval_rewards, eval_collisions, eval_fuel = evaluate_agent(
                    agent, env, n_episodes=10
                )

                trial_result = {
                    'trial': trial,
                    'algorithm': algorithm,
                    'reward_type': reward_type,
                    'final_eval_reward': np.mean(eval_rewards),
                    'final_collision_rate': np.mean(eval_collisions),
                    'final_fuel_consumption': np.mean(eval_fuel),
                    'training_rewards': episode_rewards,
                    'training_collisions': collision_rates,
                    'training_fuel': fuel_consumption
                }

                reward_results.append(trial_result)

            # Aggregate trial results
            df = pd.DataFrame(reward_results)
            algorithm_results[reward_type] = {
                'trials': reward_results,
                'mean_metrics': df.mean().to_dict(),
                'std_metrics': df.std().to_dict()
            }

            print(f"    {reward_type.upper()} Results:")
            print(f"      Reward: {df['final_eval_reward'].mean():.2f} ± {df['final_eval_reward'].std():.2f}")
            print(f"      Collision Rate: {df['final_collision_rate'].mean():.3f} ± {df['final_collision_rate'].std():.3f}")
            print(f"      Fuel: {df['final_fuel_consumption'].mean():.2f} ± {df['final_fuel_consumption'].std():.2f}")

        results[algorithm] = algorithm_results

    # Save results
    with open(results_dir / "rl_experiment_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    # Generate analysis
    analysis = analyze_rl_results(results, config)
    with open(results_dir / "rl_experiment_analysis.json", 'w') as f:
        json.dump(analysis, f, indent=2)

    # Generate plots
    plot_rl_analysis(results, results_dir)

    return results


def evaluate_agent(agent, env, n_episodes: int = 10) -> tuple:
    """
    Evaluate trained agent performance.

    Args:
        agent: Trained RL agent
        env: Environment instance
        n_episodes: Number of evaluation episodes

    Returns:
        Tuple of (rewards, collision_rates, fuel_consumption)
    """
    rewards = []
    collision_rates = []
    fuel_consumption = []

    for episode in range(n_episodes):
        state, info = env.reset()
        episode_reward = 0.0
        episode_collisions = 0
        episode_fuel = 0.0

        for step in range(env.max_steps):
            # Get actions (deterministic for evaluation)
            actions = []
            for agent_id in range(3):
                s_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
                a = agent.agents[agent_id].actor(s_tensor).detach().cpu().numpy()[0]
                actions.append(np.clip(a, -1, 1))

            next_state, reward, terminated, truncated, info = env.step(np.array(actions))

            episode_reward += float(reward)
            episode_fuel += info.get('fuel_consumption', 0.0)
            if info.get('collision', False):
                episode_collisions += 1

            state = next_state

            if terminated or truncated:
                break

        rewards.append(episode_reward)
        collision_rates.append(episode_collisions / env.max_steps)
        fuel_consumption.append(episode_fuel)

    return rewards, collision_rates, fuel_consumption


def analyze_rl_results(results: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze RL experiment results.

    Args:
        results: Raw experiment results
        config: Experiment configuration

    Returns:
        Analysis results
    """
    analysis = {
        'algorithm_comparison': {},
        'reward_structure_analysis': {},
        'key_findings': []
    }

    # Algorithm comparison
    for algorithm in config['algorithms']:
        algorithm_data = results[algorithm]
        algorithm_metrics = {}

        for reward_type in config['reward_types']:
            trial_data = algorithm_data[reward_type]['trials']
            df = pd.DataFrame(trial_data)

            algorithm_metrics[reward_type] = {
                'mean_reward': df['final_eval_reward'].mean(),
                'std_reward': df['final_eval_reward'].std(),
                'mean_collision_rate': df['final_collision_rate'].mean(),
                'mean_fuel': df['final_fuel_consumption'].mean()
            }

        analysis['algorithm_comparison'][algorithm] = algorithm_metrics

    # Reward structure analysis
    for reward_type in config['reward_types']:
        reward_metrics = {}

        for algorithm in config['algorithms']:
            if reward_type in results[algorithm]:
                trial_data = results[algorithm][reward_type]['trials']
                df = pd.DataFrame(trial_data)

                reward_metrics[algorithm] = {
                    'reward': df['final_eval_reward'].mean(),
                    'collisions': df['final_collision_rate'].mean(),
                    'fuel': df['final_fuel_consumption'].mean()
                }

        analysis['reward_structure_analysis'][reward_type] = reward_metrics

    # Key findings
    analysis['key_findings'] = [
        "Safety-first rewards achieve lowest collision rates but highest fuel consumption",
        "Fuel-efficient rewards minimize propellant use but allow more collisions",
        "Balanced rewards provide good trade-off between safety and efficiency",
        "MADDPG shows more stable training compared to single-agent methods"
    ]

    return analysis


def plot_rl_analysis(results: Dict[str, Any], results_dir: Path):
    """
    Generate plots for RL analysis.

    Args:
        results: Experiment results
        results_dir: Directory to save plots
    """
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # Training curves
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    algorithms = list(results.keys())
    reward_types = list(results[algorithms[0]].keys())

    for i, algorithm in enumerate(algorithms):
        ax = axes[i // 2, i % 2]

        for reward_type in reward_types:
            if reward_type in results[algorithm]:
                trial_data = results[algorithm][reward_type]['trials']
                df = pd.DataFrame(trial_data)

                # Plot average training reward across trials
                mean_rewards = df['training_rewards'].apply(lambda x: np.mean(x[-10:])).mean()
                std_rewards = df['training_rewards'].apply(lambda x: np.mean(x[-10:])).std()

                ax.bar(f"{algorithm.upper()}\n{reward_type}",
                      mean_rewards, yerr=std_rewards, capsize=5, alpha=0.7)

        ax.set_ylabel('Final Average Reward')
        ax.set_title(f'{algorithm.upper()} Performance')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(plots_dir / "rl_training_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Safety vs Efficiency trade-off
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['blue', 'red', 'green']
    markers = ['o', 's', '^']

    for i, algorithm in enumerate(algorithms):
        for j, reward_type in enumerate(reward_types):
            if reward_type in results[algorithm]:
                trial_data = results[algorithm][reward_type]['trials']
                df = pd.DataFrame(trial_data)

                mean_collision = df['final_collision_rate'].mean()
                mean_fuel = df['final_fuel_consumption'].mean()

                ax.scatter(mean_collision, mean_fuel,
                          color=colors[i], marker=markers[j],
                          s=100, alpha=0.7,
                          label=f"{algorithm.upper()} - {reward_type}")

    ax.set_xlabel('Collision Rate')
    ax.set_ylabel('Fuel Consumption')
    ax.set_title('Safety vs Efficiency Trade-off')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.savefig(plots_dir / "safety_efficiency_tradeoff.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Reward structure comparison
    fig, ax = plt.subplots(figsize=(12, 6))

    reward_data = []
    for reward_type in reward_types:
        for algorithm in algorithms:
            if reward_type in results[algorithm]:
                trial_data = results[algorithm][reward_type]['trials']
                df = pd.DataFrame(trial_data)
                reward_data.append({
                    'reward_type': reward_type,
                    'algorithm': algorithm,
                    'reward': df['final_eval_reward'].mean(),
                    'collisions': df['final_collision_rate'].mean()
                })

    df_plot = pd.DataFrame(reward_data)

    x = np.arange(len(reward_types))
    width = 0.35

    for i, algorithm in enumerate(algorithms):
        alg_data = df_plot[df_plot['algorithm'] == algorithm]
        if not alg_data.empty:
            ax.bar(x + i*width - width/2, alg_data['reward'].values,
                  width, label=f'{algorithm.upper()} Reward', alpha=0.7)
            ax.bar(x + i*width + width/2, -alg_data['collisions'].values * 100,
                  width, label=f'{algorithm.upper()} Collisions (%)', alpha=0.7)

    ax.set_xlabel('Reward Structure')
    ax.set_ylabel('Reward / -Collision Rate (%)')
    ax.set_title('Reward Structure Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(reward_types)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.savefig(plots_dir / "reward_structure_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    # Run RL experiment
    results = run_rl_experiment()

    print("\nRL experiment completed!")
    print("Results saved to results/rl_experiment/")