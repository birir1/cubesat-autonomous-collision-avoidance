# plot_rl_policy_behavior.py
"""
RL Policy Behavior Visualization

Visualizes reinforcement learning policy behavior, action distributions,
value functions, and learning curves for collision avoidance maneuvers.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Circle, Rectangle
import matplotlib.animation as animation
import warnings
warnings.filterwarnings('ignore')

def plot_action_distribution(actions, title="Action Distribution"):
    """
    Plot distribution of actions taken by RL agent.

    Args:
        actions: Array of actions (N, action_dim)
        title: Plot title
    """
    if actions.ndim == 1:
        actions = actions.reshape(-1, 1)

    n_actions = actions.shape[1]
    fig, axes = plt.subplots(1, n_actions, figsize=(5*n_actions, 5))

    if n_actions == 1:
        axes = [axes]

    for i in range(n_actions):
        ax = axes[i]
        action_data = actions[:, i]

        # Histogram
        ax.hist(action_data, bins=30, alpha=0.7, color='blue', edgecolor='black')
        ax.set_xlabel(f'Action {i+1} Value', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(f'Action {i+1} Distribution', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Add statistics
        mean_val = np.mean(action_data)
        std_val = np.std(action_data)
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2,
                  label=f'Mean: {mean_val:.3f}')
        ax.axvline(mean_val + std_val, color='orange', linestyle=':', linewidth=2,
                  label=f'+1σ: {mean_val+std_val:.3f}')
        ax.axvline(mean_val - std_val, color='orange', linestyle=':', linewidth=2,
                  label=f'-1σ: {mean_val-std_val:.3f}')
        ax.legend()

    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig

def plot_value_function(states, values, title="Value Function"):
    """
    Plot value function over state space.

    Args:
        states: State array (N, state_dim) - typically position/distance
        values: Value estimates (N,)
        title: Plot title
    """
    plt.figure(figsize=(10, 8))

    if states.shape[1] >= 2:
        # 2D state space
        scatter = plt.scatter(states[:, 0], states[:, 1], c=values,
                             cmap='viridis', s=50, alpha=0.7, edgecolors='black')
        plt.colorbar(scatter, label='Value')
        plt.xlabel('State Dimension 1', fontsize=12)
        plt.ylabel('State Dimension 2', fontsize=12)
    else:
        # 1D state space
        plt.scatter(states[:, 0], values, alpha=0.6, color='blue', s=30)
        plt.xlabel('State', fontsize=12)
        plt.ylabel('Value', fontsize=12)

    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return plt.gcf()

def plot_policy_heatmap(states, actions, title="Policy Heatmap"):
    """
    Plot policy as heatmap over state space.

    Args:
        states: State array (N, state_dim)
        actions: Action array (N, action_dim)
        title: Plot title
    """
    if states.shape[1] < 2 or actions.shape[1] < 1:
        print("Need at least 2D states and 1D actions for heatmap")
        return None

    fig, axes = plt.subplots(1, actions.shape[1], figsize=(6*actions.shape[1], 5))

    if actions.shape[1] == 1:
        axes = [axes]

    for i in range(actions.shape[1]):
        ax = axes[i]

        # Create 2D histogram
        h = ax.hist2d(states[:, 0], states[:, 1], weights=actions[:, i],
                     bins=20, cmap='RdYlBu_r', alpha=0.8)

        plt.colorbar(h[3], ax=ax, label=f'Action {i+1}')
        ax.set_xlabel('State Dimension 1', fontsize=12)
        ax.set_ylabel('State Dimension 2', fontsize=12)
        ax.set_title(f'Action {i+1} Policy', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig

def plot_learning_curves(rewards, losses=None, title="RL Learning Curves"):
    """
    Plot training rewards and losses over episodes.

    Args:
        rewards: Array of episode rewards
        losses: Optional array of training losses
        title: Plot title
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8)) if losses is not None else plt.subplots(1, 1, figsize=(12, 6))

    if losses is not None:
        ax1, ax2 = axes
    else:
        ax1 = axes if not isinstance(axes, np.ndarray) else axes[0]

    # Rewards plot
    ax1.plot(rewards, 'b-', linewidth=2, alpha=0.7, label='Episode Reward')
    ax1.plot(np.convolve(rewards, np.ones(100)/100, mode='valid'),
            'r-', linewidth=3, label='Moving Average (100 eps)')

    ax1.set_xlabel('Episode', fontsize=12)
    ax1.set_ylabel('Reward', fontsize=12)
    ax1.set_title('Training Rewards', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Losses plot
    if losses is not None:
        ax2.plot(losses, 'g-', linewidth=2, alpha=0.7, label='Training Loss')
        ax2.plot(np.convolve(losses, np.ones(100)/100, mode='valid'),
                'orange', linewidth=3, label='Moving Average (100 steps)')

        ax2.set_xlabel('Training Step', fontsize=12)
        ax2.set_ylabel('Loss', fontsize=12)
        ax2.set_title('Training Losses', fontsize=14, fontweight='bold')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig

def plot_maneuver_success_rate(episodes, successes, title="Maneuver Success Rate"):
    """
    Plot success rate of collision avoidance maneuvers over training.

    Args:
        episodes: Episode numbers
        successes: Success indicators (0/1)
        title: Plot title
    """
    plt.figure(figsize=(10, 6))

    # Calculate rolling success rate
    window_size = 50
    success_rate = np.convolve(successes, np.ones(window_size)/window_size, mode='valid')
    valid_episodes = episodes[window_size-1:]

    plt.plot(valid_episodes, success_rate, 'g-', linewidth=3, label='Success Rate')
    plt.axhline(y=0.95, color='red', linestyle='--', alpha=0.7,
               label='Target Success Rate (95%)')

    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Success Rate', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylim([0, 1.1])
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    return plt.gcf()

def plot_action_trajectories(trajectories, actions, title="Action Trajectories"):
    """
    Plot action trajectories over time for different episodes.

    Args:
        trajectories: List of state trajectories
        actions: List of action sequences
        title: Plot title
    """
    n_episodes = min(5, len(trajectories))  # Show up to 5 episodes

    fig, axes = plt.subplots(n_episodes, 1, figsize=(12, 4*n_episodes))

    if n_episodes == 1:
        axes = [axes]

    for i in range(n_episodes):
        ax = axes[i]

        traj = trajectories[i]
        act = actions[i]

        # Plot actions over time
        if act.ndim > 1:
            for j in range(act.shape[1]):
                ax.plot(act[:, j], label=f'Action {j+1}', linewidth=2)
        else:
            ax.plot(act, label='Action', linewidth=2, color='blue')

        ax.set_xlabel('Time Step', fontsize=12)
        ax.set_ylabel('Action Value', fontsize=12)
        ax.set_title(f'Episode {i+1} Actions', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig

def create_policy_animation(states, actions, values, interval=200,
                           title="Policy Learning Animation"):
    """
    Create animation showing policy evolution during training.

    Args:
        states: State trajectories over time (T, N, state_dim)
        actions: Action trajectories over time (T, N, action_dim)
        values: Value estimates over time (T, N)
        interval: Animation interval
        title: Animation title
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Initialize plots
    scatter1 = axes[0].scatter([], [], c=[], cmap='viridis', s=50, alpha=0.7)
    scatter2 = axes[1].scatter([], [], c=[], cmap='RdYlBu_r', s=50, alpha=0.7)
    line3, = axes[2].plot([], [], 'b-', linewidth=2)

    # Set up axes
    for ax, title_text in zip(axes, ['Value Function', 'Policy', 'Action Trajectory']):
        ax.set_xlabel('State Dim 1', fontsize=12)
        ax.set_ylabel('State Dim 2' if ax != axes[2] else 'Action', fontsize=12)
        ax.set_title(title_text, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

    axes[2].set_xlabel('Time Step', fontsize=12)

    def animate(frame):
        current_states = states[frame]
        current_actions = actions[frame]
        current_values = values[frame]

        # Update value function
        scatter1.set_offsets(current_states[:, :2])
        scatter1.set_array(current_values)

        # Update policy
        scatter2.set_offsets(current_states[:, :2])
        if current_actions.ndim > 1:
            scatter2.set_array(current_actions[:, 0])  # First action dimension
        else:
            scatter2.set_array(current_actions)

        # Update action trajectory
        axes[2].clear()
        axes[2].plot(current_actions, 'b-', linewidth=2)
        axes[2].set_xlabel('Time Step', fontsize=12)
        axes[2].set_ylabel('Action', fontsize=12)
        axes[2].set_title('Action Trajectory', fontsize=14, fontweight='bold')
        axes[2].grid(True, alpha=0.3)

        return scatter1, scatter2, line3

    anim = animation.FuncAnimation(fig, animate, frames=len(states),
                                 interval=interval, blit=False)

    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig, anim

def plot_q_function_heatmap(states, q_values, title="Q-Function Heatmap"):
    """
    Plot Q-function values as heatmap over state-action space.

    Args:
        states: State array (N, state_dim)
        q_values: Q-values (N, n_actions)
        title: Plot title
    """
    if states.shape[1] < 2:
        print("Need at least 2D states for Q-function heatmap")
        return None

    n_actions = q_values.shape[1]
    fig, axes = plt.subplots(1, n_actions, figsize=(6*n_actions, 5))

    if n_actions == 1:
        axes = [axes]

    for i in range(n_actions):
        ax = axes[i]

        h = ax.hist2d(states[:, 0], states[:, 1], weights=q_values[:, i],
                     bins=20, cmap='plasma', alpha=0.8)

        plt.colorbar(h[3], ax=ax, label=f'Q-Value Action {i+1}')
        ax.set_xlabel('State Dimension 1', fontsize=12)
        ax.set_ylabel('State Dimension 2', fontsize=12)
        ax.set_title(f'Q(a={i+1})', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    # Example usage with synthetic RL data
    np.random.seed(42)

    # Generate synthetic training data
    n_episodes = 1000
    n_steps = 50

    # Episode rewards (improving over time)
    episode_rewards = np.random.normal(0, 1, n_episodes)
    episode_rewards = episode_rewards + np.linspace(-2, 2, n_episodes)  # Learning trend
    episode_rewards = np.cumsum(episode_rewards) / np.arange(1, n_episodes+1)  # Moving average

    # Training losses
    training_losses = np.random.exponential(1, n_episodes * 10)
    training_losses = training_losses * np.exp(-np.linspace(0, 3, len(training_losses)))  # Decreasing

    # Success rates
    successes = np.random.binomial(1, np.linspace(0.3, 0.95, n_episodes))

    # State-action data
    n_states = 500
    states = np.random.normal(0, 10, (n_states, 2))  # 2D state space
    actions = np.random.normal(0, 2, (n_states, 2))  # 2D action space
    values = np.sum(states**2, axis=1) * -0.1 + np.random.normal(0, 0.1, n_states)  # Value function
    q_values = np.random.normal(0, 1, (n_states, 3))  # 3 actions

    # Action trajectories
    trajectories = [np.random.normal(0, 5, (n_steps, 2)) for _ in range(3)]
    action_seqs = [np.random.normal(0, 1, (n_steps, 2)) for _ in range(3)]

    # Create plots
    plot_action_distribution(actions, title="RL Action Distributions")
    plt.savefig('rl_action_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

    plot_value_function(states, values, title="Learned Value Function")
    plt.savefig('rl_value_function.png', dpi=300, bbox_inches='tight')
    plt.close()

    plot_policy_heatmap(states, actions, title="RL Policy Visualization")
    plt.savefig('rl_policy_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

    plot_learning_curves(episode_rewards, training_losses, title="RL Training Progress")
    plt.savefig('rl_learning_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

    plot_maneuver_success_rate(np.arange(n_episodes), successes,
                              title="Collision Avoidance Success Rate")
    plt.savefig('rl_success_rate.png', dpi=300, bbox_inches='tight')
    plt.close()

    plot_action_trajectories(trajectories, action_seqs, title="Sample Action Trajectories")
    plt.savefig('rl_action_trajectories.png', dpi=300, bbox_inches='tight')
    plt.close()

    plot_q_function_heatmap(states, q_values, title="Q-Function Landscape")
    plt.savefig('rl_q_function.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("RL policy behavior plots saved to visualization/outputs/")
