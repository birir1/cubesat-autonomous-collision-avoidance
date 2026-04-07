"""
3D Orbital Visualization

Creates interactive 3D plots of satellite orbits, conjunction scenarios,
and collision avoidance maneuvers.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib.patches import Circle
import warnings
warnings.filterwarnings('ignore')

def plot_earth_3d(ax=None, alpha=0.3):
    """
    Add Earth sphere to 3D plot.

    Args:
        ax: Matplotlib 3D axes
        alpha: Transparency of Earth sphere
    """
    if ax is None:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

    # Earth parameters
    earth_radius = 6371  # km

    # Create sphere
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = earth_radius * np.outer(np.cos(u), np.sin(v))
    y = earth_radius * np.outer(np.sin(u), np.sin(v))
    z = earth_radius * np.outer(np.ones(np.size(u)), np.cos(v))

    # Plot Earth
    ax.plot_surface(x, y, z, color='lightblue', alpha=alpha)

    return ax

def plot_satellite_orbits(positions_list, labels=None, colors=None,
                         title="Satellite Orbits", show_earth=True):
    """
    Plot 3D satellite orbits.

    Args:
        positions_list: List of position arrays, each of shape (N, 3)
        labels: List of labels for each orbit
        colors: List of colors for each orbit
        title: Plot title
        show_earth: Whether to show Earth sphere
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    if show_earth:
        ax = plot_earth_3d(ax, alpha=0.2)

    # Default colors if not provided
    if colors is None:
        colors = plt.cm.tab10(np.linspace(0, 1, len(positions_list)))

    # Plot orbits
    for i, positions in enumerate(positions_list):
        color = colors[i] if isinstance(colors, (list, np.ndarray)) else colors
        label = labels[i] if labels else f'Satellite {i+1}'

        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2],
               color=color, linewidth=2, label=label, alpha=0.8)

        # Plot current position
        ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2],
                  color=color, s=100, marker='o', edgecolors='black')

    # Set labels and title
    ax.set_xlabel('X (km)', fontsize=12)
    ax.set_ylabel('Y (km)', fontsize=12)
    ax.set_zlabel('Z (km)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Set equal aspect ratio
    max_range = np.max([np.ptp(positions.flatten()) for positions in positions_list])
    ax.set_xlim([-max_range/2, max_range/2])
    ax.set_ylim([-max_range/2, max_range/2])
    ax.set_zlim([-max_range/2, max_range/2])

    # Add legend
    if labels:
        ax.legend()

    plt.tight_layout()
    return fig, ax

def plot_conjunction_scenario(primary_positions, secondary_positions,
                           conjunction_point=None, title="Conjunction Scenario"):
    """
    Plot a conjunction scenario with primary and secondary satellites.

    Args:
        primary_positions: Primary satellite trajectory (N, 3)
        secondary_positions: Secondary satellite trajectory (N, 3)
        conjunction_point: TCA position (3,) or None
        title: Plot title
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Add Earth
    ax = plot_earth_3d(ax, alpha=0.1)

    # Plot orbits
    ax.plot(primary_positions[:, 0], primary_positions[:, 1], primary_positions[:, 2],
           color='blue', linewidth=2, label='Primary Satellite', alpha=0.8)

    ax.plot(secondary_positions[:, 0], secondary_positions[:, 1], secondary_positions[:, 2],
           color='red', linewidth=2, label='Secondary Satellite', alpha=0.8)

    # Plot current positions
    ax.scatter(primary_positions[-1, 0], primary_positions[-1, 1], primary_positions[-1, 2],
              color='blue', s=150, marker='o', edgecolors='black', label='Primary Current')

    ax.scatter(secondary_positions[-1, 0], secondary_positions[-1, 1], secondary_positions[-1, 2],
              color='red', s=150, marker='^', edgecolors='black', label='Secondary Current')

    # Highlight conjunction point
    if conjunction_point is not None:
        ax.scatter(conjunction_point[0], conjunction_point[1], conjunction_point[2],
                  color='yellow', s=200, marker='*', edgecolors='black',
                  linewidth=2, label='TCA', zorder=10)

        # Add conjunction warning sphere
        warning_radius = 10  # km
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        x = conjunction_point[0] + warning_radius * np.outer(np.cos(u), np.sin(v))
        y = conjunction_point[1] + warning_radius * np.outer(np.sin(u), np.sin(v))
        z = conjunction_point[2] + warning_radius * np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x, y, z, color='red', alpha=0.1)

    # Set labels and title
    ax.set_xlabel('X (km)', fontsize=12)
    ax.set_ylabel('Y (km)', fontsize=12)
    ax.set_zlabel('Z (km)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Set equal aspect ratio
    all_positions = np.vstack([primary_positions, secondary_positions])
    max_range = np.ptp(all_positions, axis=0).max()
    center = np.mean(all_positions, axis=0)
    ax.set_xlim([center[0] - max_range/2, center[0] + max_range/2])
    ax.set_ylim([center[1] - max_range/2, center[1] + max_range/2])
    ax.set_zlim([center[2] - max_range/2, center[2] + max_range/2])

    ax.legend()
    plt.tight_layout()
    return fig, ax

def create_orbit_animation(positions_list, labels=None, interval=100,
                          title="Satellite Orbit Animation"):
    """
    Create animated 3D plot of satellite orbits.

    Args:
        positions_list: List of position arrays
        labels: List of satellite labels
        interval: Animation interval in milliseconds
        title: Animation title
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Add Earth
    ax = plot_earth_3d(ax, alpha=0.1)

    # Initialize lines and points
    lines = []
    points = []
    colors = plt.cm.tab10(np.linspace(0, 1, len(positions_list)))

    for i, positions in enumerate(positions_list):
        color = colors[i]
        label = labels[i] if labels else f'Satellite {i+1}'

        line, = ax.plot([], [], [], color=color, linewidth=2, label=label, alpha=0.8)
        point, = ax.scatter([], [], [], color=color, s=150, marker='o', edgecolors='black')

        lines.append(line)
        points.append(point)

    # Set labels and title
    ax.set_xlabel('X (km)', fontsize=12)
    ax.set_ylabel('Y (km)', fontsize=12)
    ax.set_zlabel('Z (km)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Set axis limits
    all_positions = np.vstack(positions_list)
    max_range = np.ptp(all_positions, axis=0).max()
    center = np.mean(all_positions, axis=0)
    ax.set_xlim([center[0] - max_range/2, center[0] + max_range/2])
    ax.set_ylim([center[1] - max_range/2, center[1] + max_range/2])
    ax.set_zlim([center[2] - max_range/2, center[2] + max_range/2])

    ax.legend()

    def animate(frame):
        for i, positions in enumerate(positions_list):
            # Update trajectory
            lines[i].set_data(positions[:frame+1, 0], positions[:frame+1, 1])
            lines[i].set_3d_properties(positions[:frame+1, 2])

            # Update current position
            points[i].set_offsets(positions[frame, :2])
            points[i].set_3d_properties(positions[frame, 2])

        return lines + points

    # Create animation
    n_frames = min(len(pos) for pos in positions_list)
    anim = animation.FuncAnimation(fig, animate, frames=n_frames,
                                 interval=interval, blit=True)

    plt.tight_layout()
    return fig, anim

def plot_maneuver_trajectories(original_trajectory, maneuver_trajectory,
                              title="Collision Avoidance Maneuver"):
    """
    Plot original and maneuver trajectories for collision avoidance.

    Args:
        original_trajectory: Original satellite trajectory (N, 3)
        maneuver_trajectory: Trajectory after maneuver (N, 3)
        title: Plot title
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Add Earth
    ax = plot_earth_3d(ax, alpha=0.1)

    # Plot trajectories
    ax.plot(original_trajectory[:, 0], original_trajectory[:, 1], original_trajectory[:, 2],
           color='red', linewidth=2, linestyle='--', label='Original Trajectory', alpha=0.7)

    ax.plot(maneuver_trajectory[:, 0], maneuver_trajectory[:, 1], maneuver_trajectory[:, 2],
           color='green', linewidth=3, label='Maneuver Trajectory', alpha=0.9)

    # Plot start and end points
    ax.scatter(original_trajectory[0, 0], original_trajectory[0, 1], original_trajectory[0, 2],
              color='red', s=200, marker='s', edgecolors='black', label='Start')

    ax.scatter(maneuver_trajectory[-1, 0], maneuver_trajectory[-1, 1], maneuver_trajectory[-1, 2],
              color='green', s=200, marker='^', edgecolors='black', label='End (Safe)')

    # Highlight maneuver initiation
    maneuver_start_idx = len(original_trajectory) // 3  # Assume maneuver starts 1/3 through
    ax.scatter(original_trajectory[maneuver_start_idx, 0],
              original_trajectory[maneuver_start_idx, 1],
              original_trajectory[maneuver_start_idx, 2],
              color='orange', s=150, marker='*', edgecolors='black',
              label='Maneuver Start', zorder=10)

    # Set labels and title
    ax.set_xlabel('X (km)', fontsize=12)
    ax.set_ylabel('Y (km)', fontsize=12)
    ax.set_zlabel('Z (km)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Set equal aspect ratio
    all_positions = np.vstack([original_trajectory, maneuver_trajectory])
    max_range = np.ptp(all_positions, axis=0).max()
    center = np.mean(all_positions, axis=0)
    ax.set_xlim([center[0] - max_range/2, center[0] + max_range/2])
    ax.set_ylim([center[1] - max_range/2, center[1] + max_range/2])
    ax.set_zlim([center[2] - max_range/2, center[2] + max_range/2])

    ax.legend()
    plt.tight_layout()
    return fig, ax

if __name__ == "__main__":
    # Example usage with synthetic orbital data
    np.random.seed(42)

    # Generate synthetic orbital trajectories
    n_points = 200
    time = np.linspace(0, 2*np.pi, n_points)

    # Satellite 1: Low Earth Orbit
    r1 = 6671  # km (LEO altitude ~400km)
    positions1 = np.column_stack([
        r1 * np.cos(time),
        r1 * np.sin(time),
        200 * np.sin(time * 2)  # Some inclination
    ])

    # Satellite 2: Slightly different orbit
    r2 = 6721  # km (slightly higher)
    phase_offset = np.pi / 6
    positions2 = np.column_stack([
        r2 * np.cos(time + phase_offset),
        r2 * np.sin(time + phase_offset),
        150 * np.sin(time * 2 + phase_offset)
    ])

    # Conjunction scenario
    conjunction_point = np.array([6500, 100, 50])

    # Maneuver trajectories
    original_traj = positions1[:100]
    maneuver_traj = original_traj.copy()
    # Add small maneuver deviation
    maneuver_traj[30:, 0] += np.linspace(0, 200, 70)
    maneuver_traj[30:, 1] += np.linspace(0, -150, 70)

    # Create plots
    plot_satellite_orbits([positions1, positions2],
                         labels=['Satellite A', 'Satellite B'],
                         title="LEO Satellite Orbits")
    plt.savefig('satellite_orbits_3d.png', dpi=300, bbox_inches='tight')
    plt.close()

    plot_conjunction_scenario(positions1, positions2, conjunction_point,
                            title="Close Conjunction Scenario")
    plt.savefig('conjunction_scenario_3d.png', dpi=300, bbox_inches='tight')
    plt.close()

    plot_maneuver_trajectories(original_traj, maneuver_traj,
                              title="Collision Avoidance Maneuver")
    plt.savefig('maneuver_trajectories_3d.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Create animation (save as GIF if needed)
    fig, anim = create_orbit_animation([positions1, positions2],
                                     labels=['Satellite A', 'Satellite B'],
                                     title="Satellite Orbit Animation")
    # anim.save('orbit_animation.gif', writer='pillow', fps=10)
    plt.close()

    print("3D orbital plots saved to visualization/outputs/")
