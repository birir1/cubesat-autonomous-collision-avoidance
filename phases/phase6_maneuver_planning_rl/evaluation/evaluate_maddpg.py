"""
Evaluation Script for MADDPG Agents

Tests trained MADDPG model and computes collision avoidance metrics.

Metrics:
  - Collision rate
  - Successful avoidance rate
  - Fuel efficiency (total delta-v)
  - Minimum distance to obstacles
  - Episode length statistics
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

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from phases.phase6_maneuver_planning_rl.environment.orbital_env import OrbitalCollisionEnv
from phases.phase6_maneuver_planning_rl.models.maddpg_agent import MADDPG
from utils.logger import setup_logger

logger = setup_logger(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ================================================
# EVALUATION FUNCTIONS
# ================================================

class MADDPGEvaluator:
    """Evaluator for MADDPG agents"""
    
    def __init__(self, model_path, config_path="configs/maddpg_config.yaml"):
        """
        Initialize evaluator
        
        Args:
            model_path (str): Path to saved MADDPG model
            config_path (str): Path to configuration file
        """
        
        self.model_path = model_path
        self.config = self._load_config(config_path)
        self.device = DEVICE
        
        # Load model
        self.maddpg = self._load_model()
        
        # Initialize environment
        env_config = self.config['environment']
        self.env = OrbitalCollisionEnv(
            num_objects=env_config['num_objects'],
            safe_distance=env_config['safe_distance'],
            collision_distance=env_config['collision_distance'],
            max_steps=env_config['max_steps'],
            dt=env_config['dt'],
            max_delta_v=env_config['max_delta_v'],
            world_size=env_config['world_size']
        )
        
        self.num_agents = env_config['num_agents']
        
    def _load_config(self, config_path):
        """Load configuration"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _load_model(self):
        """Load trained MADDPG model"""
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        state_dim = self.config['actor_network']['state_dim']
        action_dim = self.config['actor_network']['action_dim']
        hidden_dim = self.config['actor_network']['hidden_dim']
        
        maddpg = MADDPG(
            num_agents=self.config['environment']['num_agents'],
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            device=self.device
        )
        
        maddpg.load(self.model_path)
        return maddpg
    
    def evaluate(self, num_episodes=100, verbose=True):
        """
        Evaluate MADDPG performance
        
        Args:
            num_episodes (int): Number of test episodes
            verbose (bool): Print progress
            
        Returns:
            dict: Evaluation metrics
        """
        
        metrics = {
            'episode': [],
            'reward': [],
            'collision': [],
            'min_distance': [],
            'fuel_cost': [],
            'episode_length': [],
            'success': []
        }
        
        total_collisions = 0
        total_success = 0
        
        iterator = tqdm(range(num_episodes), disable=not verbose, desc="Evaluating MADDPG")
        
        for episode in iterator:
            
            state = self.env.reset()
            episode_reward = 0
            episode_collision = False
            min_distance = float('inf')
            fuel_cost = 0
            step_count = 0
            
            # Run episode
            for step in range(self.config['environment']['max_steps']):
                
                actions = []
                
                # Get deterministic actions (no exploration noise)
                with torch.no_grad():
                    for agent_id in range(self.num_agents):
                        agent_state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                        action = self.maddpg.actors[agent_id](agent_state).cpu().numpy()
                        actions.append(np.clip(action[0], -1, 1))
                
                # Execute actions
                actions_array = np.array(actions)
                next_state, reward, done, info = self.env.step(actions_array)
                
                # Track metrics
                episode_reward += np.sum(reward)
                min_distance = min(min_distance, info.get('min_distance', float('inf')))
                fuel_cost += np.sum(np.abs(actions_array))  # Approximate fuel cost
                
                if info.get('collision', False):
                    episode_collision = True
                    total_collisions += 1
                
                state = next_state
                step_count = step + 1
                
                if done:
                    break
            
            # Check success
            success = not episode_collision and min_distance > self.config['environment']['safe_distance']
            if success:
                total_success += 1
            
            # Store metrics
            metrics['episode'].append(episode)
            metrics['reward'].append(episode_reward)
            metrics['collision'].append(episode_collision)
            metrics['min_distance'].append(min_distance if min_distance != float('inf') else -1)
            metrics['fuel_cost'].append(fuel_cost)
            metrics['episode_length'].append(step_count)
            metrics['success'].append(success)
        
        # Compute statistics
        stats = {
            'num_episodes': num_episodes,
            'collision_rate': total_collisions / num_episodes,
            'success_rate': total_success / num_episodes,
            'avg_reward': np.mean(metrics['reward']),
            'std_reward': np.std(metrics['reward']),
            'avg_min_distance': np.mean([d for d in metrics['min_distance'] if d > 0]),
            'avg_fuel_cost': np.mean(metrics['fuel_cost']),
            'avg_episode_length': np.mean(metrics['episode_length']),
            'min_distance_worst': np.min([d for d in metrics['min_distance'] if d > 0]),
            'max_episode_length': np.max(metrics['episode_length']),
            'min_episode_length': np.min(metrics['episode_length'])
        }
        
        if verbose:
            logger.info("\n" + "="*60)
            logger.info("MADDPG EVALUATION RESULTS")
            logger.info("="*60)
            logger.info(f"Episodes: {num_episodes}")
            logger.info(f"Collision Rate: {stats['collision_rate']:.2%}")
            logger.info(f"Success Rate: {stats['success_rate']:.2%}")
            logger.info(f"Avg Reward: {stats['avg_reward']:.2f} ± {stats['std_reward']:.2f}")
            logger.info(f"Avg Min Distance: {stats['avg_min_distance']:.2f}m")
            logger.info(f"Avg Fuel Cost: {stats['avg_fuel_cost']:.2f}")
            logger.info(f"Avg Episode Length: {stats['avg_episode_length']:.1f} steps")
            logger.info("="*60 + "\n")
        
        return {
            'metrics': pd.DataFrame(metrics),
            'stats': stats
        }


def evaluate_model(model_path, num_episodes=100, output_dir="results/reports"):
    """
    Quick evaluation function
    
    Args:
        model_path (str): Path to model
        num_episodes (int): Number of episodes
        output_dir (str): Output directory for results
    """
    
    evaluator = MADDPGEvaluator(model_path)
    results = evaluator.evaluate(num_episodes=num_episodes, verbose=True)
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    results['metrics'].to_csv(
        os.path.join(output_dir, "maddpg_evaluation_metrics.csv"),
        index=False
    )
    
    return results


# ================================================
# MAIN
# ================================================

if __name__ == "__main__":
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate MADDPG model")
    parser.add_argument('--model', type=str, required=True,
                       help='Path to saved MADDPG model')
    parser.add_argument('--episodes', type=int, default=100,
                       help='Number of evaluation episodes')
    parser.add_argument('--output', type=str, default='results/reports',
                       help='Output directory')
    
    args = parser.parse_args()
    
    results = evaluate_model(
        model_path=args.model,
        num_episodes=args.episodes,
        output_dir=args.output
    )
    
    print("\nEvaluation complete. Results saved.")
