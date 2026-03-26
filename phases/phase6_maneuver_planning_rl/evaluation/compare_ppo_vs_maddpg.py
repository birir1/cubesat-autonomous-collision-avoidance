"""
Compare PPO vs MADDPG Performance

Evaluates both single-agent (PPO) and multi-agent (MADDPG) models
and compares their collision avoidance effectiveness.
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import logging

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from phases.phase6_maneuver_planning_rl.environment.orbital_env import OrbitalCollisionEnv
from phases.phase6_maneuver_planning_rl.evaluation.evaluate_maddpg import MADDPGEvaluator
from utils.logger import setup_logger

logger = setup_logger(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ================================================
# COMPARISON CLASS
# ================================================

class ModelComparator:
    """Compare multiple RL models"""
    
    def __init__(self, config_path="configs/maddpg_config.yaml"):
        """Initialize comparator"""
        import yaml
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.device = DEVICE
    
    def compare_models(self, model_paths_dict, num_episodes=100, verbose=True):
        """
        Compare multiple models
        
        Args:
            model_paths_dict (dict): {model_name: model_path}
            num_episodes (int): Test episodes
            verbose (bool): Print progress
            
        Returns:
            dict: Comparison results
        """
        
        results = {}
        
        for model_name, model_path in model_paths_dict.items():
            
            logger.info(f"\nEvaluating {model_name}...")
            
            if "maddpg" in model_name.lower():
                evaluator = MADDPGEvaluator(model_path, config_path="configs/maddpg_config.yaml")
            else:
                # PPO evaluation would go here
                logger.warning(f"PPO evaluation not yet implemented for {model_name}")
                continue
            
            eval_results = evaluator.evaluate(num_episodes=num_episodes, verbose=verbose)
            results[model_name] = eval_results
        
        # Create comparison dataframe
        comparison_df = self._create_comparison_df(results)
        
        return {
            'results': results,
            'comparison': comparison_df
        }
    
    def _create_comparison_df(self, results):
        """Create comparison dataframe"""
        
        comparison_data = []
        
        for model_name, eval_result in results.items():
            stats = eval_result['stats']
            
            comparison_data.append({
                'Model': model_name,
                'Collision Rate': f"{stats['collision_rate']:.2%}",
                'Success Rate': f"{stats['success_rate']:.2%}",
                'Avg Reward': f"{stats['avg_reward']:.2f}",
                'Avg Min Distance': f"{stats['avg_min_distance']:.2f}m",
                'Avg Fuel Cost': f"{stats['avg_fuel_cost']:.2f}",
                'Avg Episode Length': f"{stats['avg_episode_length']:.1f}"
            })
        
        return pd.DataFrame(comparison_data)
    
    def plot_comparison(self, results, output_dir="results/figures"):
        """Plot comparison results"""
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract metrics
        model_names = list(results.keys())
        collision_rates = [results[m]['stats']['collision_rate'] for m in model_names]
        success_rates = [results[m]['stats']['success_rate'] for m in model_names]
        avg_rewards = [results[m]['stats']['avg_reward'] for m in model_names]
        fuel_costs = [results[m]['stats']['avg_fuel_cost'] for m in model_names]
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('PPO vs MADDPG Performance Comparison', fontsize=16, fontweight='bold')
        
        # Collision Rate
        axes[0, 0].bar(model_names, collision_rates, color=['#FF6B6B', '#4ECDC4'])
        axes[0, 0].set_ylabel('Collision Rate')
        axes[0, 0].set_title('Collision Rate (Lower is Better)')
        axes[0, 0].set_ylim([0, 1])
        for i, v in enumerate(collision_rates):
            axes[0, 0].text(i, v + 0.02, f'{v:.2%}', ha='center')
        
        # Success Rate
        axes[0, 1].bar(model_names, success_rates, color=['#FF6B6B', '#4ECDC4'])
        axes[0, 1].set_ylabel('Success Rate')
        axes[0, 1].set_title('Success Rate (Higher is Better)')
        axes[0, 1].set_ylim([0, 1])
        for i, v in enumerate(success_rates):
            axes[0, 1].text(i, v + 0.02, f'{v:.2%}', ha='center')
        
        # Average Reward
        axes[1, 0].bar(model_names, avg_rewards, color=['#FF6B6B', '#4ECDC4'])
        axes[1, 0].set_ylabel('Average Reward')
        axes[1, 0].set_title('Average Reward (Higher is Better)')
        for i, v in enumerate(avg_rewards):
            axes[1, 0].text(i, v + 5, f'{v:.2f}', ha='center')
        
        # Fuel Cost
        axes[1, 1].bar(model_names, fuel_costs, color=['#FF6B6B', '#4ECDC4'])
        axes[1, 1].set_ylabel('Fuel Cost (Δv)')
        axes[1, 1].set_title('Average Fuel Cost (Lower is Better)')
        for i, v in enumerate(fuel_costs):
            axes[1, 1].text(i, v + 0.1, f'{v:.2f}', ha='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'model_comparison.png'), dpi=300)
        logger.info(f"Comparison plot saved to {output_dir}/model_comparison.png")
        
        return fig


# ================================================
# MAIN
# ================================================

def main():
    """Main comparison function"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare PPO vs MADDPG models")
    parser.add_argument('--ppo-model', type=str, default='results/saved_models/ppo_cubesat_final.pth',
                       help='Path to PPO model')
    parser.add_argument('--maddpg-model', type=str, default='results/saved_models/maddpg_cubesat_final.pth',
                       help='Path to MADDPG model')
    parser.add_argument('--episodes', type=int, default=100,
                       help='Number of evaluation episodes')
    parser.add_argument('--output', type=str, default='results/reports',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Models to compare
    model_paths = {
        'MADDPG': args.maddpg_model,
    }
    
    # Run comparison
    comparator = ModelComparator()
    results = comparator.compare_models(
        model_paths,
        num_episodes=args.episodes,
        verbose=True
    )
    
    # Print comparison table
    logger.info("\n" + "="*80)
    logger.info("MODEL COMPARISON RESULTS")
    logger.info("="*80)
    print(results['comparison'].to_string(index=False))
    logger.info("="*80 + "\n")
    
    # Plot results
    comparator.plot_comparison(results['results'], output_dir=args.output)
    
    # Save comparison table
    os.makedirs(args.output, exist_ok=True)
    results['comparison'].to_csv(
        os.path.join(args.output, 'ppo_vs_maddpg_comparison.csv'),
        index=False
    )
    
    logger.info(f"Comparison results saved to {args.output}/")


if __name__ == "__main__":
    main()
