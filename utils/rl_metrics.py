"""
RL Metrics Module

Compute and track metrics for reinforcement learning agents:
- Collision rates
- Fuel efficiency
- Safety margins
- Success rates
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


# ================================================
# COLLISION METRICS
# ================================================

class CollisionMetrics:
    """Compute collision-related metrics"""
    
    @staticmethod
    def collision_rate(collisions: np.ndarray) -> float:
        """
        Calculate collision rate
        
        Args:
            collisions: Binary array [episode_count]
            
        Returns:
            float: Collision rate [0, 1]
        """
        return np.mean(collisions)
    
    @staticmethod
    def success_rate(collisions: np.ndarray, safe_distances: np.ndarray, 
                     safe_threshold: float = 20.0) -> float:
        """
        Calculate success rate (no collision AND maintained safe distance)
        
        Args:
            collisions: Binary array [episode_count]
            safe_distances: Min distances per episode [episode_count]
            safe_threshold: Safe distance threshold
            
        Returns:
            float: Success rate [0, 1]
        """
        no_collision = ~collisions
        maintained_distance = safe_distances >= safe_threshold
        return np.mean(no_collision & maintained_distance)
    
    @staticmethod
    def close_call_rate(safe_distances: np.ndarray, thresholds: Tuple[float, float] = (5.0, 20.0)) -> float:
        """
        Calculate rate of "close calls" (between thresholds)
        
        Args:
            safe_distances: Min distances per episode
            thresholds: (lower, upper) distance thresholds
            
        Returns:
            float: Close call rate
        """
        lower, upper = thresholds
        close_calls = (safe_distances >= lower) & (safe_distances < upper)
        return np.mean(close_calls)


# ================================================
# FUEL EFFICIENCY METRICS
# ================================================

class FuelMetrics:
    """Compute fuel/energy-related metrics"""
    
    @staticmethod
    def total_delta_v(actions: np.ndarray) -> float:
        """
        Calculate total delta-v (fuel cost)
        
        Args:
            actions: Array of velocity changes [episode_steps, action_dim]
            
        Returns:
            float: Total delta-v
        """
        return np.sum(np.abs(actions))
    
    @staticmethod
    def avg_delta_v_per_step(actions: np.ndarray) -> float:
        """
        Calculate average delta-v per step
        
        Args:
            actions: Array of velocity changes
            
        Returns:
            float: Average delta-v per step
        """
        return np.mean(np.linalg.norm(actions, axis=1))
    
    @staticmethod
    def delta_v_efficiency(total_delta_v: float, distance_traveled: float) -> float:
        """
        Calculate efficiency ratio (distance / fuel)
        
        Args:
            total_delta_v: Total velocity change used
            distance_traveled: Distance covered in orbit
            
        Returns:
            float: Efficiency ratio
        """
        if total_delta_v == 0:
            return float('inf')
        return distance_traveled / total_delta_v


# ================================================
# SAFETY METRICS
# ================================================

class SafetyMetrics:
    """Compute safety-related metrics"""
    
    @staticmethod
    def minimum_safety_margin(min_distances: np.ndarray, safe_distance: float) -> float:
        """
        Calculate minimum safety margin across episodes
        
        Args:
            min_distances: Min distances per episode
            safe_distance: Nominal safe distance
            
        Returns:
            float: Worst-case margin
        """
        worst_case = np.min(min_distances)
        return worst_case - safe_distance
    
    @staticmethod
    def safety_margin_distribution(min_distances: np.ndarray, safe_distance: float) -> Dict:
        """
        Calculate safety margin statistics
        
        Args:
            min_distances: Min distances per episode
            safe_distance: Nominal safe distance
            
        Returns:
            dict: Statistics including mean, std, percentiles
        """
        margins = min_distances - safe_distance
        
        return {
            'mean_margin': np.mean(margins),
            'std_margin': np.std(margins),
            'min_margin': np.min(margins),
            'max_margin': np.max(margins),
            'p5_margin': np.percentile(margins, 5),
            'p25_margin': np.percentile(margins, 25),
            'p75_margin': np.percentile(margins, 75),
            'p95_margin': np.percentile(margins, 95)
        }
    
    @staticmethod
    def critical_encounters(min_distances: np.ndarray, threshold: float = 5.0) -> float:
        """
        Calculate rate of critical encounters (closer than threshold)
        
        Args:
            min_distances: Min distances per episode
            threshold: Distance threshold for "critical"
            
        Returns:
            float: Critical encounter rate
        """
        return np.mean(min_distances < threshold)


# ================================================
# EPISODE METRICS
# ================================================

class EpisodeMetrics:
    """Compute per-episode metrics"""
    
    @staticmethod
    def episode_length_stats(lengths: np.ndarray) -> Dict:
        """
        Calculate episode length statistics
        
        Args:
            lengths: Episode lengths
            
        Returns:
            dict: Statistics
        """
        return {
            'mean_length': np.mean(lengths),
            'std_length': np.std(lengths),
            'min_length': np.min(lengths),
            'max_length': np.max(lengths),
            'median_length': np.median(lengths)
        }
    
    @staticmethod
    def cumulative_reward(rewards: np.ndarray) -> np.ndarray:
        """
        Calculate cumulative rewards
        
        Args:
            rewards: Rewards per step
            
        Returns:
            np.ndarray: Cumulative rewards
        """
        return np.cumsum(rewards)
    
    @staticmethod
    def reward_statistics(episode_rewards: np.ndarray) -> Dict:
        """
        Calculate reward statistics
        
        Args:
            episode_rewards: Total reward per episode
            
        Returns:
            dict: Statistics
        """
        return {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'min_reward': np.min(episode_rewards),
            'max_reward': np.max(episode_rewards),
            'median_reward': np.median(episode_rewards),
            'trend': np.mean(episode_rewards[-100:]) - np.mean(episode_rewards[:100])
        }


# ================================================
# MULTI-AGENT METRICS
# ================================================

class MultiAgentMetrics:
    """Compute multi-agent specific metrics"""
    
    @staticmethod
    def coordination_efficiency(agent_actions: List[np.ndarray]) -> float:
        """
        Calculate coordination efficiency (how well agents move together)
        
        Args:
            agent_actions: List of action arrays per agent
            
        Returns:
            float: Coordination score [0, 1]
        """
        if len(agent_actions) < 2:
            return 1.0
        
        # Compute pairwise action similarity
        similarities = []
        for i in range(len(agent_actions)):
            for j in range(i+1, len(agent_actions)):
                # Cosine similarity
                a1 = agent_actions[i].flatten()
                a2 = agent_actions[j].flatten()
                sim = np.dot(a1, a2) / (np.linalg.norm(a1) * np.linalg.norm(a2) + 1e-8)
                similarities.append(sim)
        
        return np.mean(similarities) if similarities else 1.0
    
    @staticmethod
    def collision_avoidance_redundancy(agent_separations: List[float], safe_distance: float) -> float:
        """
        Calculate redundancy in collision avoidance (multiple agents avoiding same target)
        
        Args:
            agent_separations: Separation distances from each agent
            safe_distance: Safe distance threshold
            
        Returns:
            float: Redundancy factor
        """
        safe_count = sum(1 for sep in agent_separations if sep > safe_distance)
        return safe_count / len(agent_separations) if agent_separations else 0.0


# ================================================
# AGGREGATE METRICS COMPUTATION
# ================================================

def compute_all_metrics(episodes_data: List[Dict], config: Dict) -> Dict:
    """
    Compute all metrics from episode data
    
    Args:
        episodes_data: List of episode records
        config: Configuration dict with thresholds
        
    Returns:
        dict: Comprehensive metrics
    """
    
    df = pd.DataFrame(episodes_data)
    
    safe_distance = config.get('safe_distance', 20.0)
    collision_distance = config.get('collision_distance', 2.0)
    
    metrics = {
        'collision': CollisionMetrics.collision_rate(df['collision'].values),
        'success': CollisionMetrics.success_rate(
            df['collision'].values,
            df['min_distance'].values,
            safe_distance
        ),
        'close_call_rate': CollisionMetrics.close_call_rate(df['min_distance'].values),
        'critical_encounters': SafetyMetrics.critical_encounters(
            df['min_distance'].values,
            collision_distance
        ),
        'safety_margins': SafetyMetrics.safety_margin_distribution(
            df['min_distance'].values,
            safe_distance
        ),
        'fuel': {
            'avg_delta_v': np.mean([np.sum(np.abs(a)) for a in df['actions']]) if 'actions' in df else 0
        },
        'episode': EpisodeMetrics.episode_length_stats(df['length'].values),
        'rewards': EpisodeMetrics.reward_statistics(df['reward'].values)
    }
    
    return metrics


if __name__ == "__main__":
    # Example usage
    print("RL Metrics Module")
