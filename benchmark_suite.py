"""
Benchmark Suite for RL Models

Compare performance across different scenarios and metrics.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm
import logging

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from phases.phase6_maneuver_planning_rl.environment.orbital_env import OrbitalCollisionEnv
from phases.phase6_maneuver_planning_rl.models.maddpg_agent import MADDPG
from utils.logger import setup_logger
from utils.rl_metrics import CollisionMetrics, FuelMetrics, SafetyMetrics

logger = setup_logger(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ================================================
# BENCHMARK SCENARIOS
# ================================================

class BenchmarkScenario:
    """Base benchmark scenario"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    def setup_environment(self) -> OrbitalCollisionEnv:
        """Setup environment for scenario"""
        raise NotImplementedError


class HeadOnApproachScenario(BenchmarkScenario):
    """Single debris approaching head-on"""
    
    def __init__(self):
        super().__init__("head_on", "Single debris approaching CubeSat head-on")
    
    def setup_environment(self):
        env = OrbitalCollisionEnv(
            num_objects=2,
            safe_distance=20.0,
            max_steps=200,
            world_size=200
        )
        return env


class MultiObjectScenario(BenchmarkScenario):
    """Multiple objects in close proximity"""
    
    def __init__(self):
        super().__init__("multi_object", "Multiple objects in congested orbit")
    
    def setup_environment(self):
        env = OrbitalCollisionEnv(
            num_objects=8,
            safe_distance=20.0,
            max_steps=300,
            world_size=300
        )
        return env


class DebrisFieldScenario(BenchmarkScenario):
    """CubeSat navigating debris field"""
    
    def __init__(self):
        super().__init__("debris_field", "CubeSat navigating dispersed debris field")
    
    def setup_environment(self):
        env = OrbitalCollisionEnv(
            num_objects=15,
            safe_distance=15.0,
            max_steps=400,
            world_size=500
        )
        return env


class CoordinatedEvasionScenario(BenchmarkScenario):
    """Multiple CubeSats coordinating"""
    
    def __init__(self):
        super().__init__("coordinated", "Multiple satellites coordinating avoidance")
    
    def setup_environment(self):
        env = OrbitalCollisionEnv(
            num_objects=10,
            safe_distance=25.0,
            max_steps=350,
            world_size=400
        )
        return env


# ================================================
# BENCHMARK SUITE
# ================================================

class BenchmarkSuite:
    """Comprehensive benchmark suite"""
    
    def __init__(self, model_path: str = None):
        """Initialize benchmark"""
        self.model_path = model_path
        self.scenarios = [
            HeadOnApproachScenario(),
            MultiObjectScenario(),
            DebrisFieldScenario(),
            CoordinatedEvasionScenario()
        ]
        self.results = {}
    
    def run_scenario(self, scenario: BenchmarkScenario, num_episodes: int = 50, 
                    verbose: bool = True) -> Dict:
        """
        Run scenario benchmark
        
        Args:
            scenario: Scenario to run
            num_episodes: Number of test episodes
            verbose: Print progress
            
        Returns:
            Scenario results
        """
        
        env = scenario.setup_environment()
        
        results = {
            'scenario': scenario.name,
            'episodes': [],
            'collisions': [],
            'success': [],
            'min_distances': [],
            'fuel_costs': [],
            'lengths': [],
            'rewards': []
        }
        
        iterator = tqdm(range(num_episodes), disable=not verbose, desc=f"Benchmarking {scenario.name}")
        
        for episode in iterator:
            state = env.reset()
            episode_reward = 0
            collision = False
            min_distance = float('inf')
            fuel_cost = 0
            length = 0
            
            for step in range(env.max_steps):
                
                # Simple baseline: maintain distance maneuver
                # (In reality would use MADDPG here)
                action = np.random.randn(env.max_objects if hasattr(env, 'max_objects') else 5, 2)
                action = np.clip(action * 0.1, -1, 1)
                
                next_state, reward, done, info = env.step(action)
                
                episode_reward += np.sum(reward)
                min_distance = min(min_distance, info.get('min_distance', float('inf')))
                fuel_cost += np.sum(np.abs(action))
                collision = collision or info.get('collision', False)
                length = step + 1
                
                state = next_state
                
                if done:
                    break
            
            # Store results
            success = not collision and min_distance > scenario.setup_environment().safe_distance
            
            results['episodes'].append(episode)
            results['collisions'].append(collision)
            results['success'].append(success)
            results['min_distances'].append(min_distance)
            results['fuel_costs'].append(fuel_cost)
            results['lengths'].append(length)
            results['rewards'].append(episode_reward)
        
        env.close()
        
        self.results[scenario.name] = results
        return results
    
    def run_all_scenarios(self, num_episodes: int = 50, verbose: bool = True):
        """Run all benchmark scenarios"""
        
        logger.info("\n" + "="*60)
        logger.info("Running Comprehensive Benchmark Suite")
        logger.info("="*60 + "\n")
        
        for scenario in self.scenarios:
            logger.info(f"Scenario: {scenario.name} - {scenario.description}")
            self.run_scenario(scenario, num_episodes, verbose)
            logger.info("")
    
    def print_results_table(self):
        """Print results as table"""
        
        summary_data = []
        
        for scenario_name, results in self.results.items():
            
            collisions = np.array(results['collisions'])
            success = np.array(results['success'])
            min_dists = np.array(results['min_distances'])
            fuel = np.array(results['fuel_costs'])
            lengths = np.array(results['lengths'])
            rewards = np.array(results['rewards'])
            
            summary_data.append({
                'Scenario': scenario_name.upper(),
                'Episodes': len(results['episodes']),
                'Collision Rate': f"{np.mean(collisions):.1%}",
                'Success Rate': f"{np.mean(success):.1%}",
                'Avg Min Distance': f"{np.mean(min_dists):.1f}m",
                'Avg Fuel Cost': f"{np.mean(fuel):.2f}",
                'Avg Episode Length': f"{np.mean(lengths):.0f} steps",
                'Avg Reward': f"{np.mean(rewards):.2f}"
            })
        
        df = pd.DataFrame(summary_data)
        
        print("\n" + "="*100)
        print("BENCHMARK RESULTS SUMMARY")
        print("="*100)
        print(df.to_string(index=False))
        print("="*100 + "\n")
        
        return df
    
    def save_results(self, output_dir: str = "results/reports"):
        """Save benchmark results"""
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save detailed results
        for scenario_name, results in self.results.items():
            df = pd.DataFrame(results)
            df.to_csv(
                os.path.join(output_dir, f"benchmark_{scenario_name}.csv"),
                index=False
            )
        
        logger.info(f"Benchmark results saved to {output_dir}/")


# ================================================
# MAIN
# ================================================

def main():
    """Run benchmark suite"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Benchmark Suite")
    parser.add_argument('--episodes', type=int, default=50,
                       help='Episodes per scenario')
    parser.add_argument('--output', type=str, default='results/reports',
                       help='Output directory')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to trained model')
    
    args = parser.parse_args()
    
    # Create suite
    suite = BenchmarkSuite(model_path=args.model)
    
    # Run all scenarios
    suite.run_all_scenarios(num_episodes=args.episodes, verbose=True)
    
    # Print results
    suite.print_results_table()
    
    # Save results
    suite.save_results(args.output)


if __name__ == "__main__":
    main()
