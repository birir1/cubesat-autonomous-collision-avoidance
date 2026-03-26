"""
MADDPG Training Visualization

Comprehensive visualization suite for MADDPG training including:
- Training curves and metrics
- Trajectory videos and animations
- Collision risk heatmaps
- Performance comparison tables
- Multi-agent behavior analysis
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
from pathlib import Path
from matplotlib.patches import Circle
from matplotlib.colors import LinearSegmentedColormap
import cv2
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10


# ================================================
# PLOTTING FUNCTIONS
# ================================================

def plot_training_curves(metrics_csv: str, output_dir: str = "results/figures"):
    """
    Plot training curves from metrics CSV

    Args:
        metrics_csv: Path to metrics CSV file
        output_dir: Output directory for plots
    """

    # Load metrics
    df = pd.read_csv(metrics_csv)

    os.makedirs(output_dir, exist_ok=True)

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('MADDPG Training Progress', fontsize=16, fontweight='bold')

    # 1. Episode Reward
    axes[0, 0].plot(df['episode'], df['reward'], alpha=0.6, linewidth=0.8)
    axes[0, 0].plot(df['episode'], df['reward'].rolling(window=50).mean(),
                    linewidth=2, label='50-episode MA', color='orange')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].set_title('Episode Reward')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Collision Rate
    axes[0, 1].plot(df['episode'], df['collision_rate'], alpha=0.6, linewidth=0.8)
    axes[0, 1].plot(df['episode'], df['collision_rate'].rolling(window=50).mean(),
                    linewidth=2, label='50-episode MA', color='orange')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Collision Rate')
    axes[0, 1].set_title('Collision Rate Over Time')
    axes[0, 1].set_ylim([0, 1])
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Minimum Distance
    axes[0, 2].scatter(df['episode'], df['min_distance'], alpha=0.3, s=10)
    axes[0, 2].axhline(y=20, color='r', linestyle='--', label='Safe Distance')
    axes[0, 2].axhline(y=2, color='darkred', linestyle='--', label='Collision Distance')
    axes[0, 2].set_xlabel('Episode')
    axes[0, 2].set_ylabel('Minimum Distance (m)')
    axes[0, 2].set_title('Closest Approach to Obstacles')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # 4. Episode Length
    axes[1, 0].scatter(df['episode'], df['episode_length'], alpha=0.4, s=10)
    axes[1, 0].plot(df['episode'], df['episode_length'].rolling(window=50).mean(),
                    linewidth=2, label='50-episode MA', color='orange')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Episode Length (steps)')
    axes[1, 0].set_title('Episode Duration')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 5. Actor Loss
    if 'actor_loss' in df.columns and df['actor_loss'].sum() > 0:
        axes[1, 1].semilogy(df['episode'], df['actor_loss'] + 1e-8, alpha=0.6, linewidth=0.8)
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Actor Loss (log scale)')
        axes[1, 1].set_title('Actor Network Loss')
        axes[1, 1].grid(True, alpha=0.3)

    # 6. Critic Loss
    if 'critic_loss' in df.columns and df['critic_loss'].sum() > 0:
        axes[1, 2].semilogy(df['episode'], df['critic_loss'] + 1e-8, alpha=0.6, linewidth=0.8)
        axes[1, 2].set_xlabel('Episode')
        axes[1, 2].set_ylabel('Critic Loss (log scale)')
        axes[1, 2].set_title('Critic Network Loss')
        axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'maddpg_training_curves.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Training curves saved to {plot_path}")
    plt.close()


def plot_learning_metrics_summary(metrics_csv: str, output_dir: str = "results/figures"):
    """Plot summary statistics"""

    df = pd.read_csv(metrics_csv)
    os.makedirs(output_dir, exist_ok=True)

    # Calculate statistics by episode bins
    bins = np.linspace(0, len(df), 11)  # 10 bins
    df['bin'] = pd.cut(df['episode'], bins=bins)

    grouped = df.groupby('bin').agg({
        'reward': ['mean', 'std'],
        'collision_rate': 'mean',
        'min_distance': 'mean',
        'episode_length': 'mean'
    })

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('MADDPG Training Summary Statistics', fontsize=14, fontweight='bold')

    # Reward with error bars
    x = range(len(grouped))
    axes[0, 0].errorbar(x, grouped['reward']['mean'], yerr=grouped['reward']['std'],
                       marker='o', capsize=5)
    axes[0, 0].set_xlabel('Training Phase')
    axes[0, 0].set_ylabel('Average Reward')
    axes[0, 0].set_title('Reward Progress')
    axes[0, 0].grid(True, alpha=0.3)

    # Collision rate trend
    axes[0, 1].plot(x, grouped['collision_rate']['mean'], marker='o', linewidth=2)
    axes[0, 1].fill_between(x, 0, grouped['collision_rate']['mean'], alpha=0.3)
    axes[0, 1].set_xlabel('Training Phase')
    axes[0, 1].set_ylabel('Collision Rate')
    axes[0, 1].set_ylim([0, 1])
    axes[0, 1].set_title('Collision Rate Trend')
    axes[0, 1].grid(True, alpha=0.3)

    # Min distance improvement
    axes[1, 0].plot(x, grouped['min_distance']['mean'], marker='s', linewidth=2, color='green')
    axes[1, 0].axhline(y=20, color='r', linestyle='--', label='Safe Distance')
    axes[1, 0].set_xlabel('Training Phase')
    axes[1, 0].set_ylabel('Avg Min Distance (m)')
    axes[1, 0].set_title('Safety Margin Improvement')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Episode length
    axes[1, 1].plot(x, grouped['episode_length']['mean'], marker='^', linewidth=2, color='purple')
    axes[1, 1].set_xlabel('Training Phase')
    axes[1, 1].set_ylabel('Avg Episode Length')
    axes[1, 1].set_title('Episode Duration Trend')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'maddpg_summary_statistics.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Summary statistics saved to {plot_path}")
    plt.close()


# ================================================
# VIDEO AND ANIMATION FUNCTIONS
# ================================================

def create_trajectory_video(trajectory_data: Dict, output_dir: str = "results/videos",
                           video_name: str = "maddpg_trajectory.mp4", fps: int = 10):
    """
    Create video animation of multi-agent trajectories

    Args:
        trajectory_data: Dictionary containing trajectory information
        output_dir: Output directory for video
        video_name: Name of output video file
        fps: Frames per second for video
    """

    os.makedirs(output_dir, exist_ok=True)

    # Extract trajectory data
    positions = trajectory_data.get('positions', [])  # List of [num_agents, 3] arrays
    obstacles = trajectory_data.get('obstacles', [])  # List of [num_obstacles, 3] arrays
    collision_risks = trajectory_data.get('collision_risks', [])  # List of risk values
    episode_length = len(positions)

    if episode_length == 0:
        print("No trajectory data available")
        return

    # Setup video writer
    video_path = os.path.join(output_dir, video_name)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Create figure for animation
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Set axis limits based on data
    all_positions = np.concatenate(positions, axis=0)
    x_min, x_max = all_positions[:, 0].min(), all_positions[:, 0].max()
    y_min, y_max = all_positions[:, 1].min(), all_positions[:, 1].max()
    z_min, z_max = all_positions[:, 2].min(), all_positions[:, 2].max()

    margin = 50
    ax.set_xlim([x_min - margin, x_max + margin])
    ax.set_ylim([y_min - margin, y_max + margin])
    ax.set_zlim([z_min - margin, z_max + margin])

    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_zlabel('Z (km)')
    ax.set_title('Multi-Agent CubeSat Trajectories')

    # Initialize plot elements
    agent_scatter = None
    obstacle_scatter = None
    trajectory_lines = []

    def animate(frame):
        nonlocal agent_scatter, obstacle_scatter, trajectory_lines

        ax.clear()
        ax.set_xlim([x_min - margin, x_max + margin])
        ax.set_ylim([y_min - margin, y_max + margin])
        ax.set_zlim([z_min - margin, z_max + margin])
        ax.set_xlabel('X (km)')
        ax.set_ylabel('Y (km)')
        ax.set_zlabel('Z (km)')

        # Plot current positions
        current_pos = positions[frame]
        num_agents = current_pos.shape[0]

        # Color agents by collision risk
        risk_colors = plt.cm.RdYlGn_r(np.clip(collision_risks[frame] if frame < len(collision_risks) else [0]*num_agents, 0, 1))

        agent_scatter = ax.scatter(current_pos[:, 0], current_pos[:, 1], current_pos[:, 2],
                                 c=risk_colors, s=100, alpha=0.8, edgecolors='black', linewidth=2)

        # Plot obstacles
        if frame < len(obstacles) and len(obstacles[frame]) > 0:
            obs_pos = obstacles[frame]
            obstacle_scatter = ax.scatter(obs_pos[:, 0], obs_pos[:, 1], obs_pos[:, 2],
                                        c='red', s=50, alpha=0.6, marker='x')

        # Plot trajectory trails
        for i in range(num_agents):
            if frame > 0:
                trail_positions = np.array([pos[i] for pos in positions[:frame+1]])
                line, = ax.plot(trail_positions[:, 0], trail_positions[:, 1], trail_positions[:, 2],
                              alpha=0.3, linewidth=1, color=risk_colors[i])
                trajectory_lines.append(line)

        ax.set_title(f'Multi-Agent Trajectories - Step {frame+1}/{episode_length}')
        return [agent_scatter, obstacle_scatter] + trajectory_lines

    # Create animation
    anim = animation.FuncAnimation(fig, animate, frames=episode_length,
                                 interval=1000/fps, blit=False)

    # Save animation
    anim.save(video_path, writer='ffmpeg', fps=fps, dpi=100)
    plt.close()

    print(f"Trajectory video saved to {video_path}")


def plot_collision_risk_heatmap(trajectory_data: Dict, output_dir: str = "results/figures"):
    """
    Create heatmap of collision risks over time and space

    Args:
        trajectory_data: Dictionary containing trajectory and risk information
        output_dir: Output directory for plots
    """

    os.makedirs(output_dir, exist_ok=True)

    collision_risks = trajectory_data.get('collision_risks', [])
    positions = trajectory_data.get('positions', [])

    if not collision_risks or not positions:
        print("No collision risk data available")
        return

    # Create time-space risk matrix
    num_steps = len(collision_risks)
    num_agents = len(collision_risks[0]) if collision_risks else 0

    if num_agents == 0:
        return

    risk_matrix = np.array(collision_risks)

    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(risk_matrix.T, aspect='auto', cmap='RdYlGn_r', origin='lower')

    ax.set_xlabel('Time Step')
    ax.set_ylabel('Agent ID')
    ax.set_title('Collision Risk Heatmap Over Time')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Collision Risk')

    # Add risk level annotations
    risk_levels = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    colors = ['green', 'yellowgreen', 'yellow', 'orange', 'red', 'darkred']
    for level, color in zip(risk_levels, colors):
        ax.axhline(y=level * num_agents, color=color, linestyle='--', alpha=0.5, linewidth=1)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'collision_risk_heatmap.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Collision risk heatmap saved to {plot_path}")
    plt.close()


def plot_agent_trajectories_2d(trajectory_data: Dict, output_dir: str = "results/figures"):
    """
    Create 2D trajectory plots for each agent

    Args:
        trajectory_data: Dictionary containing trajectory information
        output_dir: Output directory for plots
    """

    os.makedirs(output_dir, exist_ok=True)

    positions = trajectory_data.get('positions', [])
    obstacles = trajectory_data.get('obstacles', [])

    if not positions:
        print("No trajectory data available")
        return

    num_agents = positions[0].shape[0] if positions else 0
    num_steps = len(positions)

    # Create subplots for each agent
    fig, axes = plt.subplots(2, (num_agents + 1) // 2, figsize=(15, 10))
    if num_agents == 1:
        axes = [axes]
    axes = axes.flatten()

    fig.suptitle('Agent Trajectories in XY Plane', fontsize=16, fontweight='bold')

    for agent_id in range(min(num_agents, len(axes))):
        ax = axes[agent_id]

        # Extract trajectory for this agent
        agent_positions = np.array([pos[agent_id] for pos in positions])

        # Plot trajectory
        ax.plot(agent_positions[:, 0], agent_positions[:, 1],
               linewidth=2, alpha=0.8, label=f'Agent {agent_id+1}')

        # Plot start and end points
        ax.scatter(agent_positions[0, 0], agent_positions[0, 1],
                  c='green', s=100, marker='o', label='Start', zorder=5)
        ax.scatter(agent_positions[-1, 0], agent_positions[-1, 1],
                  c='red', s=100, marker='x', label='End', zorder=5)

        # Plot obstacles (static for simplicity)
        if obstacles and len(obstacles) > 0:
            obs_pos = obstacles[0]  # Use first timestep obstacles
            ax.scatter(obs_pos[:, 0], obs_pos[:, 1], c='red', s=50,
                      alpha=0.6, marker='x', label='Obstacles')

        ax.set_xlabel('X Position (km)')
        ax.set_ylabel('Y Position (km)')
        ax.set_title(f'Agent {agent_id+1} Trajectory')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axis('equal')

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'agent_trajectories_2d.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"2D trajectory plots saved to {plot_path}")
    plt.close()


# ================================================
# PERFORMANCE ANALYSIS FUNCTIONS
# ================================================

def create_performance_table(metrics_files: List[str], model_names: List[str],
                           output_dir: str = "results/tables"):
    """
    Create performance comparison table for different models

    Args:
        metrics_files: List of paths to metrics CSV files
        model_names: List of model names corresponding to metrics files
        output_dir: Output directory for table
    """

    os.makedirs(output_dir, exist_ok=True)

    performance_data = []

    for metrics_file, model_name in zip(metrics_files, model_names):
        if not os.path.exists(metrics_file):
            print(f"Warning: Metrics file {metrics_file} not found")
            continue

        df = pd.read_csv(metrics_file)

        # Calculate final performance metrics (last 100 episodes)
        final_episodes = df.tail(100)

        metrics = {
            'Model': model_name,
            'Final Reward': f"{final_episodes['reward'].mean():.2f} ± {final_episodes['reward'].std():.2f}",
            'Min Reward': f"{final_episodes['reward'].min():.2f}",
            'Max Reward': f"{final_episodes['reward'].max():.2f}",
            'Collision Rate': f"{final_episodes['collision_rate'].mean():.3f}",
            'Min Distance': f"{final_episodes['min_distance'].mean():.2f} m",
            'Episode Length': f"{final_episodes['episode_length'].mean():.1f} steps",
            'Training Episodes': len(df)
        }

        performance_data.append(metrics)

    # Create DataFrame and save as CSV
    results_df = pd.DataFrame(performance_data)
    table_path = os.path.join(output_dir, 'model_performance_comparison.csv')
    results_df.to_csv(table_path, index=False)

    # Also create a formatted table image
    fig, ax = plt.subplots(figsize=(12, len(performance_data) * 0.8))
    ax.axis('off')

    # Create table
    table_data = results_df.values
    column_labels = results_df.columns

    table = ax.table(cellText=table_data, colLabels=column_labels,
                    loc='center', cellLoc='center')

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    # Style the table
    for (i, j), cell in table.get_celld().items():
        if i == 0:  # Header row
            cell.set_fontsize(12)
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#e6e6e6')
        else:  # Data rows
            if j == 0:  # Model names
                cell.set_fontsize(11)
                cell.set_text_props(weight='bold')

    plt.title('Model Performance Comparison', fontsize=14, fontweight='bold', pad=20)
    table_path_png = os.path.join(output_dir, 'model_performance_comparison.png')
    plt.savefig(table_path_png, dpi=300, bbox_inches='tight')
    print(f"Performance comparison table saved to {table_path}")
    print(f"Performance comparison image saved to {table_path_png}")
    plt.close()

    return results_df


def plot_model_comparison(metrics_files: List[str], model_names: List[str],
                         output_dir: str = "results/figures"):
    """
    Create comparison plots for different models

    Args:
        metrics_files: List of paths to metrics CSV files
        model_names: List of model names
        output_dir: Output directory for plots
    """

    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')

    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']

    for i, (metrics_file, model_name) in enumerate(zip(metrics_files, model_names)):
        if not os.path.exists(metrics_file):
            continue

        df = pd.read_csv(metrics_file)
        color = colors[i % len(colors)]

        # Reward comparison
        axes[0, 0].plot(df['episode'], df['reward'].rolling(window=50).mean(),
                       label=model_name, color=color, linewidth=2)

        # Collision rate comparison
        axes[0, 1].plot(df['episode'], df['collision_rate'].rolling(window=50).mean(),
                       label=model_name, color=color, linewidth=2)

        # Min distance comparison
        axes[1, 0].plot(df['episode'], df['min_distance'].rolling(window=50).mean(),
                       label=model_name, color=color, linewidth=2)

        # Episode length comparison
        axes[1, 1].plot(df['episode'], df['episode_length'].rolling(window=50).mean(),
                       label=model_name, color=color, linewidth=2)

    # Configure plots
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Average Reward')
    axes[0, 0].set_title('Reward Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Collision Rate')
    axes[0, 1].set_title('Collision Rate Comparison')
    axes[0, 1].set_ylim([0, 1])
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Min Distance (m)')
    axes[1, 0].set_title('Safety Margin Comparison')
    axes[1, 0].axhline(y=20, color='r', linestyle='--', alpha=0.7, label='Safe Distance')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Episode Length (steps)')
    axes[1, 1].set_title('Episode Duration Comparison')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'model_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Model comparison plot saved to {plot_path}")
    plt.close()


# ================================================
# UTILITY FUNCTIONS
# ================================================

def save_trajectory_snapshot(trajectory_data: Dict, timestep: int,
                           output_dir: str = "results/figures"):
    """
    Save a single snapshot of trajectories at a specific timestep

    Args:
        trajectory_data: Dictionary containing trajectory information
        timestep: Timestep to capture
        output_dir: Output directory for image
    """

    os.makedirs(output_dir, exist_ok=True)

    positions = trajectory_data.get('positions', [])
    obstacles = trajectory_data.get('obstacles', [])
    collision_risks = trajectory_data.get('collision_risks', [])

    if timestep >= len(positions):
        print(f"Timestep {timestep} out of range")
        return

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot current positions
    current_pos = positions[timestep]
    num_agents = current_pos.shape[0]

    # Color by collision risk
    risks = collision_risks[timestep] if timestep < len(collision_risks) else [0] * num_agents
    risk_colors = plt.cm.RdYlGn_r(np.clip(risks, 0, 1))

    ax.scatter(current_pos[:, 0], current_pos[:, 1], current_pos[:, 2],
              c=risk_colors, s=100, alpha=0.8, edgecolors='black', linewidth=2)

    # Plot obstacles
    if timestep < len(obstacles) and len(obstacles[timestep]) > 0:
        obs_pos = obstacles[timestep]
        ax.scatter(obs_pos[:, 0], obs_pos[:, 1], obs_pos[:, 2],
                  c='red', s=50, alpha=0.6, marker='x')

    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_zlabel('Z (km)')
    ax.set_title(f'Trajectory Snapshot - Timestep {timestep}')

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap='RdYlGn_r', norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Collision Risk')

    plt.tight_layout()
    snapshot_path = os.path.join(output_dir, f'trajectory_snapshot_t{timestep:03d}.png')
    plt.savefig(snapshot_path, dpi=300, bbox_inches='tight')
    print(f"Trajectory snapshot saved to {snapshot_path}")
    plt.close()


# ================================================
# MAIN
# ================================================

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="MADDPG Training Visualization Suite")
    parser.add_argument('--metrics', type=str, required=True,
                       help='Path to metrics CSV file')
    parser.add_argument('--output', type=str, default='results/figures',
                       help='Output directory')
    parser.add_argument('--mode', type=str, default='all',
                       choices=['curves', 'summary', 'video', 'heatmap', 'trajectories', 'table', 'all'],
                       help='Visualization mode')
    parser.add_argument('--trajectory-data', type=str,
                       help='Path to trajectory data pickle file (for video/heatmap modes)')
    parser.add_argument('--compare-models', nargs='+',
                       help='List of metrics CSV files for model comparison')
    parser.add_argument('--model-names', nargs='+',
                       help='List of model names for comparison')

    args = parser.parse_args()

    print(f"MADDPG Visualization Suite - Mode: {args.mode}")

    if args.mode in ['curves', 'all']:
        print(f"Plotting training curves from {args.metrics}")
        plot_training_curves(args.metrics, args.output)

    if args.mode in ['summary', 'all']:
        print(f"Plotting summary statistics from {args.metrics}")
        plot_learning_metrics_summary(args.metrics, args.output)

    if args.mode in ['video', 'all'] and args.trajectory_data:
        print(f"Creating trajectory video from {args.trajectory_data}")
        # Load trajectory data (assuming pickle format)
        import pickle
        with open(args.trajectory_data, 'rb') as f:
            trajectory_data = pickle.load(f)
        create_trajectory_video(trajectory_data, os.path.join(args.output, '../videos'))

    if args.mode in ['heatmap', 'all'] and args.trajectory_data:
        print(f"Creating collision risk heatmap from {args.trajectory_data}")
        import pickle
        with open(args.trajectory_data, 'rb') as f:
            trajectory_data = pickle.load(f)
        plot_collision_risk_heatmap(trajectory_data, args.output)

    if args.mode in ['trajectories', 'all'] and args.trajectory_data:
        print(f"Plotting 2D trajectories from {args.trajectory_data}")
        import pickle
        with open(args.trajectory_data, 'rb') as f:
            trajectory_data = pickle.load(f)
        plot_agent_trajectories_2d(trajectory_data, args.output)

    if args.mode == 'table' and args.compare_models and args.model_names:
        print("Creating performance comparison table")
        create_performance_table(args.compare_models, args.model_names,
                               os.path.join(args.output, '../tables'))

    if args.mode == 'compare' and args.compare_models and args.model_names:
        print("Creating model comparison plots")
        plot_model_comparison(args.compare_models, args.model_names, args.output)

    print("Visualization complete!")


def plot_learning_metrics_summary(metrics_csv: str, output_dir: str = "results/figures"):
    """Plot summary statistics"""
    
    df = pd.read_csv(metrics_csv)
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate statistics by episode bins
    bins = np.linspace(0, len(df), 11)  # 10 bins
    df['bin'] = pd.cut(df['episode'], bins=bins)
    
    grouped = df.groupby('bin').agg({
        'reward': ['mean', 'std'],
        'collision_rate': 'mean',
        'min_distance': 'mean',
        'episode_length': 'mean'
    })
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('MADDPG Training Summary Statistics', fontsize=14, fontweight='bold')
    
    # Reward with error bars
    x = range(len(grouped))
    axes[0, 0].errorbar(x, grouped['reward']['mean'], yerr=grouped['reward']['std'],
                       marker='o', capsize=5)
    axes[0, 0].set_xlabel('Training Phase')
    axes[0, 0].set_ylabel('Average Reward')
    axes[0, 0].set_title('Reward Progress')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Collision rate trend
    axes[0, 1].plot(x, grouped['collision_rate']['mean'], marker='o', linewidth=2)
    axes[0, 1].fill_between(x, 0, grouped['collision_rate']['mean'], alpha=0.3)
    axes[0, 1].set_xlabel('Training Phase')
    axes[0, 1].set_ylabel('Collision Rate')
    axes[0, 1].set_ylim([0, 1])
    axes[0, 1].set_title('Collision Rate Trend')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Min distance improvement
    axes[1, 0].plot(x, grouped['min_distance']['mean'], marker='s', linewidth=2, color='green')
    axes[1, 0].axhline(y=20, color='r', linestyle='--', label='Safe Distance')
    axes[1, 0].set_xlabel('Training Phase')
    axes[1, 0].set_ylabel('Avg Min Distance (m)')
    axes[1, 0].set_title('Safety Margin Improvement')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Episode length
    axes[1, 1].plot(x, grouped['episode_length']['mean'], marker='^', linewidth=2, color='purple')
    axes[1, 1].set_xlabel('Training Phase')
    axes[1, 1].set_ylabel('Avg Episode Length')
    axes[1, 1].set_title('Episode Duration Trend')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'maddpg_summary_statistics.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Summary statistics saved to {plot_path}")
    plt.close()


# ================================================
# MAIN
# ================================================

# if __name__ == "__main__":
    
#     import argparse
    
#     parser = argparse.ArgumentParser(description="Plot MADDPG training curves")
#     parser.add_argument('--metrics', type=str, required=True,
#                        help='Path to metrics CSV file')
#     parser.add_argument('--output', type=str, default='results/figures',
#                        help='Output directory')
    
#     args = parser.parse_args()
    
#     print(f"Plotting training curves from {args.metrics}")
#     plot_training_curves(args.metrics, args.output)
#     plot_learning_metrics_summary(args.metrics, args.output)
#     print("Done!")
