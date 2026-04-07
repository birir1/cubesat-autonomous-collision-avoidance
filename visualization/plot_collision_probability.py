"""
Collision Probability Visualization

Plots collision probability distributions and risk heatmaps for
satellite conjunction scenarios.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Circle
import warnings
warnings.filterwarnings('ignore')

def plot_collision_probability_heatmap(positions, probabilities, title="Collision Probability Heatmap"):
    """
    Plot collision probability as a heatmap over satellite positions.

    Args:
        positions: Array of shape (N, 2) with x,y positions
        probabilities: Array of shape (N,) with collision probabilities
        title: Plot title
    """
    plt.figure(figsize=(10, 8))

    # Create scatter plot with probability coloring
    scatter = plt.scatter(positions[:, 0], positions[:, 1],
                         c=probabilities, cmap='RdYlGn_r',
                         s=50, alpha=0.7, edgecolors='black', linewidth=0.5)

    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Collision Probability', fontsize=12)

    plt.xlabel('X Position (km)', fontsize=12)
    plt.ylabel('Y Position (km)', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)

    # Add warning zone circles
    for prob_threshold in [0.1, 0.01, 0.001]:
        mask = probabilities >= prob_threshold
        if np.any(mask):
            center = np.mean(positions[mask], axis=0)
            radius = np.max(np.linalg.norm(positions[mask] - center, axis=1))
            circle = Circle(center, radius, fill=False, color='red',
                          linestyle='--', alpha=0.5, linewidth=2,
                          label=f'P ≥ {prob_threshold}')
            plt.gca().add_patch(circle)

    plt.legend()
    plt.tight_layout()
    return plt.gcf()

def plot_probability_distribution(probabilities, title="Collision Probability Distribution"):
    """
    Plot histogram and KDE of collision probabilities.

    Args:
        probabilities: Array of collision probabilities
        title: Plot title
    """
    plt.figure(figsize=(12, 6))

    # Histogram
    plt.subplot(1, 2, 1)
    plt.hist(probabilities, bins=50, alpha=0.7, color='blue', edgecolor='black')
    plt.xlabel('Collision Probability', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Probability Histogram', fontsize=14, fontweight='bold')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)

    # KDE plot
    plt.subplot(1, 2, 2)
    sns.kdeplot(probabilities, fill=True, alpha=0.7, color='green')
    plt.xlabel('Collision Probability', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title('Probability Density', fontsize=14, fontweight='bold')
    plt.xscale('log')
    plt.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    return plt.gcf()

def plot_risk_vs_distance(distances, probabilities, title="Risk vs Distance"):
    """
    Plot collision probability vs separation distance.

    Args:
        distances: Array of separation distances
        probabilities: Array of collision probabilities
        title: Plot title
    """
    plt.figure(figsize=(10, 6))

    plt.scatter(distances, probabilities, alpha=0.6, color='purple', s=30)

    # Add trend line
    z = np.polyfit(np.log(distances), np.log(probabilities), 1)
    p = np.poly1d(z)
    plt.plot(distances, np.exp(p(np.log(distances))),
             color='red', linewidth=2, label=f'Power law fit: {z[0]:.2f}')

    plt.xlabel('Separation Distance (km)', fontsize=12)
    plt.ylabel('Collision Probability', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    return plt.gcf()

def plot_conjunction_timeline(times, probabilities, conjunction_id=None,
                            title="Conjunction Timeline"):
    """
    Plot collision probability over time for a conjunction event.

    Args:
        times: Array of time points (relative to TCA)
        probabilities: Array of collision probabilities over time
        conjunction_id: Optional conjunction identifier
        title: Plot title
    """
    plt.figure(figsize=(12, 6))

    plt.plot(times, probabilities, 'b-', linewidth=2, marker='o', markersize=4)

    # Highlight TCA (time of closest approach)
    tca_idx = np.argmin(np.abs(times))
    plt.axvline(x=times[tca_idx], color='red', linestyle='--', alpha=0.7,
                label=f'TCA: t={times[tca_idx]:.1f}h')

    plt.scatter(times[tca_idx], probabilities[tca_idx],
               color='red', s=100, zorder=5, label=f'Max P: {probabilities[tca_idx]:.2e}')

    plt.xlabel('Time from TCA (hours)', fontsize=12)
    plt.ylabel('Collision Probability', fontsize=12)
    if conjunction_id:
        plt.title(f'{title} - Conjunction {conjunction_id}', fontsize=14, fontweight='bold')
    else:
        plt.title(title, fontsize=14, fontweight='bold')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    return plt.gcf()

def create_risk_assessment_dashboard(positions, probabilities, distances, times=None):
    """
    Create a comprehensive risk assessment dashboard.

    Args:
        positions: Satellite positions
        probabilities: Collision probabilities
        distances: Separation distances
        times: Optional time points for timeline
    """
    fig = plt.figure(figsize=(16, 12))

    # Heatmap
    plt.subplot(2, 2, 1)
    scatter = plt.scatter(positions[:, 0], positions[:, 1],
                         c=probabilities, cmap='RdYlGn_r',
                         s=30, alpha=0.7)
    plt.colorbar(scatter)
    plt.xlabel('X Position (km)')
    plt.ylabel('Y Position (km)')
    plt.title('Collision Risk Heatmap')
    plt.grid(True, alpha=0.3)

    # Probability distribution
    plt.subplot(2, 2, 2)
    plt.hist(probabilities, bins=30, alpha=0.7, color='blue', edgecolor='black')
    plt.xlabel('Collision Probability')
    plt.ylabel('Frequency')
    plt.title('Probability Distribution')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)

    # Risk vs Distance
    plt.subplot(2, 2, 3)
    plt.scatter(distances, probabilities, alpha=0.6, color='purple', s=20)
    plt.xlabel('Separation Distance (km)')
    plt.ylabel('Collision Probability')
    plt.title('Risk vs Distance')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)

    # Timeline (if times provided)
    plt.subplot(2, 2, 4)
    if times is not None:
        plt.plot(times, probabilities, 'b-', linewidth=2)
        plt.xlabel('Time (hours)')
        plt.ylabel('Collision Probability')
        plt.title('Probability Timeline')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'No timeline data available',
                transform=plt.gca().transAxes, ha='center', va='center',
                fontsize=12)
        plt.title('Probability Timeline (N/A)')

    plt.suptitle('Collision Risk Assessment Dashboard', fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    # Example usage with synthetic data
    np.random.seed(42)
    n_points = 1000

    # Generate synthetic data
    positions = np.random.normal(0, 100, (n_points, 2))
    distances = np.linalg.norm(positions, axis=1)
    probabilities = 1e-6 * np.exp(-distances / 50) * np.random.uniform(0.1, 2, n_points)
    times = np.linspace(-24, 24, 100)
    time_probs = 1e-6 * np.exp(-np.abs(times) / 6) * np.random.uniform(0.5, 1.5, len(times))

    # Create plots
    plot_collision_probability_heatmap(positions, probabilities)
    plt.savefig('collision_probability_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

    plot_probability_distribution(probabilities)
    plt.savefig('probability_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

    plot_risk_vs_distance(distances, probabilities)
    plt.savefig('risk_vs_distance.png', dpi=300, bbox_inches='tight')
    plt.close()

    plot_conjunction_timeline(times, time_probs, conjunction_id="SAT-001-SAT-002")
    plt.savefig('conjunction_timeline.png', dpi=300, bbox_inches='tight')
    plt.close()

    create_risk_assessment_dashboard(positions, probabilities, distances, times)
    plt.savefig('risk_assessment_dashboard.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("Collision probability plots saved to visualization/outputs/")
