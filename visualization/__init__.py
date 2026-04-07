"""
Visualization Module

Provides comprehensive visualization tools for the CubeSat Autonomous
Collision Avoidance framework including:

- Collision probability plots and heatmaps
- 3D orbital trajectory visualization
- RL policy behavior analysis
- Training progress monitoring
- Multi-agent trajectory plotting
- MADDPG-specific visualizations
"""

from .plot_collision_probability import (
    plot_collision_probability_heatmap,
    plot_probability_distribution,
    plot_risk_vs_distance,
    plot_conjunction_timeline,
    create_risk_assessment_dashboard
)

from .plot_orbit_3d import (
    plot_earth_3d,
    plot_satellite_orbits,
    plot_conjunction_scenario,
    create_orbit_animation,
    plot_maneuver_trajectories
)

from .plot_rl_policy_behavior import (
    plot_action_distribution,
    plot_value_function,
    plot_policy_heatmap,
    plot_learning_curves,
    plot_maneuver_success_rate,
    plot_action_trajectories,
    create_policy_animation,
    plot_q_function_heatmap
)

from .plot_multi_agent_trajectories import *
from .plot_maddpg_training import *

__all__ = [
    # Collision probability visualization
    'plot_collision_probability_heatmap',
    'plot_probability_distribution',
    'plot_risk_vs_distance',
    'plot_conjunction_timeline',
    'create_risk_assessment_dashboard',

    # 3D orbital visualization
    'plot_earth_3d',
    'plot_satellite_orbits',
    'plot_conjunction_scenario',
    'create_orbit_animation',
    'plot_maneuver_trajectories',

    # RL policy visualization
    'plot_action_distribution',
    'plot_value_function',
    'plot_policy_heatmap',
    'plot_learning_curves',
    'plot_maneuver_success_rate',
    'plot_action_trajectories',
    'create_policy_animation',
    'plot_q_function_heatmap',
]