"""
Reward Function for Maneuver RL

Defines reward functions for collision avoidance maneuver learning.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging

class ManeuverReward:
    """
    Reward function for maneuver planning in collision avoidance.
    """

    def __init__(self, config: Dict):
        """
        Initialize reward function.

        Args:
            config: Reward configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Reward weights
        self.collision_penalty = config.get('collision_penalty', -1000.0)
        self.success_reward = config.get('success_reward', 100.0)
        self.distance_reward_weight = config.get('distance_reward_weight', 1.0)
        self.fuel_penalty_weight = config.get('fuel_penalty_weight', 0.1)
        self.time_penalty_weight = config.get('time_penalty_weight', 0.01)

        # Distance thresholds
        self.safe_distance = config.get('safe_distance', 1000.0)  # meters
        self.warning_distance = config.get('warning_distance', 5000.0)  # meters

        # Fuel parameters
        self.max_fuel = config.get('max_fuel', 100.0)
        self.fuel_efficiency_target = config.get('fuel_efficiency_target', 0.8)

    def compute_reward(self, state: np.ndarray, action: np.ndarray,
                      next_state: np.ndarray, done: bool, info: Dict) -> float:
        """
        Compute reward for state-action-next_state transition.

        Args:
            state: Current state
            action: Action taken
            next_state: Next state
            done: Whether episode is done
            info: Additional information

        Returns:
            Reward value
        """
        reward = 0.0

        # Collision penalty
        if info.get('collision', False):
            reward += self.collision_penalty
            return reward  # Early return for collision

        # Success reward
        if info.get('maneuver_success', False) and done:
            reward += self.success_reward

        # Distance-based reward
        distance_reward = self._compute_distance_reward(state, next_state, info)
        reward += distance_reward

        # Fuel efficiency penalty
        fuel_penalty = self._compute_fuel_penalty(action, info)
        reward += fuel_penalty

        # Time penalty (encourage faster solutions)
        if not done:
            reward += self.time_penalty_weight

        # Action smoothness penalty (discourage erratic maneuvers)
        smoothness_penalty = self._compute_smoothness_penalty(action, info)
        reward += smoothness_penalty

        return reward

    def _compute_distance_reward(self, state: np.ndarray, next_state: np.ndarray,
                                info: Dict) -> float:
        """Compute reward based on inter-satellite distance."""
        current_distance = info.get('current_distance', np.linalg.norm(state[:3] - state[6:9]))
        next_distance = info.get('next_distance', np.linalg.norm(next_state[:3] - next_state[6:9]))

        # Reward for increasing distance (moving away from collision)
        distance_change = next_distance - current_distance

        # Scale reward based on distance ranges
        if next_distance >= self.safe_distance:
            # Safe zone - small positive reward for maintaining safety
            reward = 0.1
        elif next_distance >= self.warning_distance:
            # Warning zone - reward for moving away, penalty for moving closer
            reward = self.distance_reward_weight * distance_change / 1000.0  # Scale to km
        else:
            # Danger zone - strong reward/penalty
            reward = 2.0 * self.distance_reward_weight * distance_change / 1000.0

        return reward

    def _compute_fuel_penalty(self, action: np.ndarray, info: Dict) -> float:
        """Compute penalty for fuel usage."""
        fuel_used = info.get('fuel_used', 0.0)
        fuel_remaining = info.get('fuel_remaining', self.max_fuel)

        # Penalty increases as fuel is consumed
        fuel_ratio = fuel_used / self.max_fuel
        penalty = -self.fuel_penalty_weight * fuel_ratio

        # Extra penalty for low fuel
        if fuel_remaining / self.max_fuel < 0.2:  # Less than 20% fuel
            penalty -= 0.5

        return penalty

    def _compute_smoothness_penalty(self, action: np.ndarray, info: Dict) -> float:
        """Compute penalty for action smoothness."""
        # Penalize large, sudden changes in thrust
        thrust_magnitude = np.linalg.norm(action)

        # Penalize high thrust (encourage efficient maneuvers)
        smoothness_penalty = -0.01 * thrust_magnitude

        # Check for previous action to penalize changes
        prev_action = info.get('previous_action')
        if prev_action is not None:
            action_change = np.linalg.norm(action - prev_action)
            smoothness_penalty -= 0.05 * action_change

        return smoothness_penalty

    def get_reward_breakdown(self, state: np.ndarray, action: np.ndarray,
                           next_state: np.ndarray, done: bool, info: Dict) -> Dict[str, float]:
        """
        Get detailed breakdown of reward components.

        Args:
            state: Current state
            action: Action taken
            next_state: Next state
            done: Whether episode is done
            info: Additional information

        Returns:
            Dictionary of reward components
        """
        breakdown = {}

        # Individual components
        breakdown['collision_penalty'] = self.collision_penalty if info.get('collision', False) else 0.0
        breakdown['success_reward'] = self.success_reward if (info.get('maneuver_success', False) and done) else 0.0
        breakdown['distance_reward'] = self._compute_distance_reward(state, next_state, info)
        breakdown['fuel_penalty'] = self._compute_fuel_penalty(action, info)
        breakdown['time_penalty'] = self.time_penalty_weight if not done else 0.0
        breakdown['smoothness_penalty'] = self._compute_smoothness_penalty(action, info)

        # Total reward
        breakdown['total_reward'] = sum(breakdown.values())

        return breakdown


class ShapedManeuverReward(ManeuverReward):
    """
    Reward function with potential-based shaping for better learning.
    """

    def __init__(self, config: Dict):
        super(ShapedManeuverReward, self).__init__(config)

        # Shaping parameters
        self.shaping_weight = config.get('shaping_weight', 0.1)
        self.potential_function = config.get('potential_function', 'distance')

    def compute_reward(self, state: np.ndarray, action: np.ndarray,
                      next_state: np.ndarray, done: bool, info: Dict) -> float:
        """
        Compute shaped reward.
        """
        # Base reward
        base_reward = super().compute_reward(state, action, next_state, done, info)

        # Potential-based shaping
        potential_current = self._compute_potential(state, info)
        potential_next = self._compute_potential(next_state, info)

        shaping_reward = self.shaping_weight * (potential_next - potential_current)

        return base_reward + shaping_reward

    def _compute_potential(self, state: np.ndarray, info: Dict) -> float:
        """Compute potential function value."""
        distance = info.get('current_distance', np.linalg.norm(state[:3] - state[6:9]))

        if self.potential_function == 'distance':
            # Higher potential for safer distances
            return -1.0 / (distance + 1.0)  # Negative to encourage larger distances
        elif self.potential_function == 'log_distance':
            # Logarithmic potential
            return -np.log(distance + 1.0)
        elif self.potential_function == 'exp_distance':
            # Exponential potential
            return np.exp(-distance / 1000.0)
        else:
            return 0.0


class SparseManeuverReward(ManeuverReward):
    """
    Sparse reward function that only gives rewards at episode end.
    """

    def __init__(self, config: Dict):
        super(SparseManeuverReward, self).__init__(config)

        # Sparse reward parameters
        self.sparse_success_reward = config.get('sparse_success_reward', 1000.0)
        self.sparse_collision_penalty = config.get('sparse_collision_penalty', -1000.0)
        self.sparse_fuel_penalty_weight = config.get('sparse_fuel_penalty_weight', 1.0)

    def compute_reward(self, state: np.ndarray, action: np.ndarray,
                      next_state: np.ndarray, done: bool, info: Dict) -> float:
        """
        Compute sparse reward.
        """
        if not done:
            return 0.0  # No reward until episode ends

        # Terminal rewards
        reward = 0.0

        if info.get('collision', False):
            reward += self.sparse_collision_penalty
        elif info.get('maneuver_success', False):
            reward += self.sparse_success_reward

            # Bonus for fuel efficiency
            fuel_used = info.get('fuel_used', 0.0)
            fuel_efficiency = 1.0 - (fuel_used / self.max_fuel)
            if fuel_efficiency >= self.fuel_efficiency_target:
                reward += 100.0

        return reward


class CuriosityDrivenReward(ManeuverReward):
    """
    Reward function with curiosity-driven exploration.
    """

    def __init__(self, config: Dict):
        super(CuriosityDrivenReward, self).__init__(config)

        # Curiosity parameters
        self.curiosity_weight = config.get('curiosity_weight', 0.1)
        self.state_visit_counts = {}  # Simple state visitation count

    def compute_reward(self, state: np.ndarray, action: np.ndarray,
                      next_state: np.ndarray, done: bool, info: Dict) -> float:
        """
        Compute reward with curiosity bonus.
        """
        # Base reward
        base_reward = super().compute_reward(state, action, next_state, done, info)

        # Curiosity bonus based on state novelty
        state_key = self._discretize_state(next_state)
        visit_count = self.state_visit_counts.get(state_key, 0)
        self.state_visit_counts[state_key] = visit_count + 1

        # Curiosity bonus decreases with visit count
        curiosity_bonus = self.curiosity_weight / (1.0 + visit_count)

        return base_reward + curiosity_bonus

    def _discretize_state(self, state: np.ndarray, bins: int = 10) -> Tuple:
        """Discretize continuous state for counting."""
        # Simple discretization of position components
        discretized = []
        for i in range(min(6, len(state))):  # First 6 components (position + velocity)
            bin_idx = min(bins - 1, max(0, int((state[i] + 10000) / 20000 * bins)))  # Assume range [-10000, 10000]
            discretized.append(bin_idx)

        return tuple(discretized)


# Registry of reward functions
REWARD_FUNCTIONS = {
    'standard': ManeuverReward,
    'shaped': ShapedManeuverReward,
    'sparse': SparseManeuverReward,
    'curiosity': CuriosityDrivenReward
}


def create_reward_function(reward_type: str, config: Dict) -> ManeuverReward:
    """
    Factory function to create reward function.

    Args:
        reward_type: Type of reward function
        config: Configuration dictionary

    Returns:
        Reward function instance
    """
    if reward_type not in REWARD_FUNCTIONS:
        raise ValueError(f"Unknown reward type: {reward_type}")

    return REWARD_FUNCTIONS[reward_type](config)


if __name__ == "__main__":
    # Example usage
    config = {
        'collision_penalty': -1000.0,
        'success_reward': 100.0,
        'distance_reward_weight': 1.0,
        'fuel_penalty_weight': 0.1,
        'safe_distance': 1000.0,
        'warning_distance': 5000.0
    }

    reward_fn = ManeuverReward(config)

    # Example state transition
    state = np.array([0, 0, 0, 0, 0, 0, 1000, 0, 0])  # 1km separation
    action = np.array([0.1, 0.0, 0.0])  # Small thrust
    next_state = np.array([10, 0, 0, 0.1, 0, 0, 1010, 0, 0])  # Moved apart
    info = {'current_distance': 1000.0, 'next_distance': 1014.0, 'fuel_used': 0.1}

    reward = reward_fn.compute_reward(state, action, next_state, False, info)
    breakdown = reward_fn.get_reward_breakdown(state, action, next_state, False, info)

    print(f"Total reward: {reward}")
    print("Reward breakdown:")
    for component, value in breakdown.items():
        print(f"  {component}: {value}")

    # Test different reward types
    shaped_reward = ShapedManeuverReward(config)
    shaped_value = shaped_reward.compute_reward(state, action, next_state, False, info)
    print(f"Shaped reward: {shaped_value}")

    sparse_reward = SparseManeuverReward(config)
    sparse_value = sparse_reward.compute_reward(state, action, next_state, True,
                                               {**info, 'maneuver_success': True})
    print(f"Sparse reward (success): {sparse_value}")