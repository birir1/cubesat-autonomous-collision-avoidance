"""
Multi-Agent Trajectory Visualization

Plot satellite trajectories and collision avoidance maneuvers.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

sns.set_style("whitegrid")


# ================================================
# TRAJECTORY PLOTTING
# ================================================

def plot_2d_trajectories(episode_data: dict, safe_distance: float = 20.0, 
                         collision_distance: float = 2.0, 
                         output_path: str = "trajectory_2d.png"):
    """
    Plot 2D orbital trajectories
    
    Args:
        episode_data: Dict with 'positions' [timesteps, num_agents, 2]
        safe_distance: Safe separation distance
        collision_distance: Collision threshold
        output_path: Output file path
    """
    
    positions = episode_data['positions']
    num_agents = positions.shape[1]
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    colors = plt.cm.tab10(np.linspace(0, 1, num_agents))
    
    # Plot trajectories
    for agent_id in range(num_agents):
        trajectory = positions[:, agent_id, :]
        ax.plot(trajectory[:, 0], trajectory[:, 1], label=f'Agent {agent_id}',
               color=colors[agent_id], linewidth=2, alpha=0.7)
        
        # Mark start and end
        ax.scatter(trajectory[0, 0], trajectory[0, 1], marker='o', s=100,
                  color=colors[agent_id], edgecolors='black', linewidths=2, zorder=5)
        ax.scatter(trajectory[-1, 0], trajectory[-1, 1], marker='*', s=300,
                  color=colors[agent_id], edgecolors='black', linewidths=1, zorder=5)
    
    # Add collision regions
    if 'obstacles' in episode_data:
        for obs in episode_data['obstacles']:
            circle = patches.Circle(obs, collision_distance, fill=False, 
                                   edgecolor='red', linewidth=2, linestyle='--', label='Collision Zone')
            ax.add_patch(circle)
            circle_safe = patches.Circle(obs, safe_distance, fill=False,
                                        edgecolor='orange', linewidth=1, linestyle=':', label='Safe Zone')
            ax.add_patch(circle_safe)
    
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_title('Multi-Agent Orbital Trajectories (2D)')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"2D trajectory plot saved to {output_path}")
    plt.close()


def plot_3d_trajectories(episode_data: dict, output_path: str = "trajectory_3d.png"):
    """
    Plot 3D orbital trajectories with velocity vectors
    
    Args:
        episode_data: Dict with 'positions' and 'velocities'
        output_path: Output file path
    """
    
    positions = episode_data['positions']
    num_agents = positions.shape[1]
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    colors = plt.cm.tab10(np.linspace(0, 1, num_agents))
    
    # Plot 3D trajectories (using x, y, and time as z)
    for agent_id in range(num_agents):
        trajectory = positions[:, agent_id, :]
        time_axis = np.arange(len(trajectory))
        
        ax.plot(trajectory[:, 0], trajectory[:, 1], time_axis,
               label=f'Agent {agent_id}', color=colors[agent_id], linewidth=2)
        
        # Mark key points
        ax.scatter(trajectory[0, 0], trajectory[0, 1], time_axis[0],
                  marker='o', s=100, color=colors[agent_id])
    
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_zlabel('Time (steps)')
    ax.set_title('Multi-Agent Orbital Trajectories (3D)')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"3D trajectory plot saved to {output_path}")
    plt.close()


def plot_inter_agent_distances(episode_data: dict, output_path: str = "distances.png"):
    """
    Plot pairwise distances between agents over time
    
    Args:
        episode_data: Dict with 'positions'
        output_path: Output file path
    """
    
    positions = episode_data['positions']
    num_agents = positions.shape[1]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Compute pairwise distances
    for i in range(num_agents):
        for j in range(i+1, num_agents):
            distances = np.linalg.norm(
                positions[:, i, :] - positions[:, j, :],
                axis=1
            )
            ax.plot(distances, label=f'Agent {i}-{j}', linewidth=2)
    
    # Add safe distance threshold
    ax.axhline(y=20, color='orange', linestyle='--', linewidth=2, label='Safe Distance')
    ax.axhline(y=2, color='red', linestyle='--', linewidth=2, label='Collision Distance')
    
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Distance (m)')
    ax.set_title('Inter-Agent Distances Over Time')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Distance plot saved to {output_path}")
    plt.close()


def plot_action_sequences(episode_data: dict, output_path: str = "actions.png"):
    """
    Plot action sequences for each agent
    
    Args:
        episode_data: Dict with 'actions' [timesteps, num_agents, action_dim]
        output_path: Output file path
    """
    
    actions = episode_data['actions']
    num_agents = actions.shape[1]
    action_dim = actions.shape[2]
    
    fig, axes = plt.subplots(num_agents, 1, figsize=(12, 3*num_agents))
    if num_agents == 1:
        axes = [axes]
    
    for agent_id in range(num_agents):
        agent_actions = actions[:, agent_id, :]
        
        for action_dim_id in range(action_dim):
            axes[agent_id].plot(agent_actions[:, action_dim_id],
                              label=f'Action {action_dim_id}', linewidth=1.5)
        
        axes[agent_id].set_ylabel(f'Agent {agent_id}')
        axes[agent_id].set_xlabel('Time Step')
        axes[agent_id].legend(loc='best', fontsize=8)
        axes[agent_id].grid(True, alpha=0.3)
    
    axes[0].set_title('Control Actions Over Time')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Action sequence plot saved to {output_path}")
    plt.close()


def plot_collision_avoidance_heatmap(episode_data: dict, grid_size: int = 50,
                                     output_path: str = "heatmap.png"):
    """
    Plot heatmap of collision avoidance (density of maneuvers)
    
    Args:
        episode_data: Dict with 'positions'
        grid_size: Resolution of heatmap
        output_path: Output file path
    """
    
    positions = episode_data['positions']
    
    # Create grid
    x_min, x_max = positions[:, :, 0].min(), positions[:, :, 0].max()
    y_min, y_max = positions[:, :, 1].min(), positions[:, :, 1].max()
    
    heatmap = np.zeros((grid_size, grid_size))
    
    # Fill heatmap with position density
    for step in range(len(positions)):
        for agent in range(positions.shape[1]):
            x, y = positions[step, agent, :]
            i = int((x - x_min) / (x_max - x_min + 1e-8) * (grid_size - 1))
            j = int((y - y_min) / (y_max - y_min + 1e-8) * (grid_size - 1))
            
            if 0 <= i < grid_size and 0 <= j < grid_size:
                heatmap[j, i] += 1
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(heatmap, cmap='YlOrRd', origin='lower',
                  extent=[x_min, x_max, y_min, y_max])
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_title('Collision Avoidance Activity Heatmap')
    plt.colorbar(im, ax=ax, label='Agent Visits')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Heatmap saved to {output_path}")
    plt.close()


# ================================================
# BATCH VISUALIZATION
# ================================================

def visualize_episode(episode_data: dict, output_dir: str = "results/figures/episodes"):
    """
    Create all visualizations for an episode
    
    Args:
        episode_data: Complete episode data dict
        output_dir: Output directory
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    episode_num = episode_data.get('episode', 0)
    episode_dir = os.path.join(output_dir, f"episode_{episode_num:05d}")
    os.makedirs(episode_dir, exist_ok=True)
    
    # Plot all visualizations
    plot_2d_trajectories(episode_data, 
                        output_path=os.path.join(episode_dir, "trajectory_2d.png"))
    
    if len(episode_data['positions'].shape) > 2:
        plot_3d_trajectories(episode_data,
                            output_path=os.path.join(episode_dir, "trajectory_3d.png"))
    
    plot_inter_agent_distances(episode_data,
                              output_path=os.path.join(episode_dir, "distances.png"))
    
    if 'actions' in episode_data:
        plot_action_sequences(episode_data,
                             output_path=os.path.join(episode_dir, "actions.png"))
    
    plot_collision_avoidance_heatmap(episode_data,
                                    output_path=os.path.join(episode_dir, "heatmap.png"))
    
    print(f"Episode {episode_num} visualizations saved to {episode_dir}")


# ================================================
# MAIN
# ================================================

if __name__ == "__main__":
    print("Multi-Agent Trajectory Visualization Module")
