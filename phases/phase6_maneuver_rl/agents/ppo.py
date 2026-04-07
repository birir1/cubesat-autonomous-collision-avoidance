"""
Proximal Policy Optimization (PPO) Agent for Maneuver Planning

Implements PPO algorithm for learning collision avoidance maneuvers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
from collections import deque

class ActorNetwork(nn.Module):
    """
    Actor network for PPO policy.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        """
        Initialize actor network.

        Args:
            state_dim: State dimension
            action_dim: Action dimension
            hidden_dim: Hidden layer dimension
        """
        super(ActorNetwork, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

        # Initialize output layer for better exploration
        nn.init.xavier_uniform_(self.network[-1].weight, gain=0.01)
        nn.init.constant_(self.network[-1].bias, 0.0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            state: Input state

        Returns:
            Action logits
        """
        return self.network(state)


class CriticNetwork(nn.Module):
    """
    Critic network for PPO value function.
    """

    def __init__(self, state_dim: int, hidden_dim: int = 256):
        """
        Initialize critic network.

        Args:
            state_dim: State dimension
            hidden_dim: Hidden layer dimension
        """
        super(CriticNetwork, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            state: Input state

        Returns:
            Value estimate
        """
        return self.network(state)


class PPOAgent(nn.Module):
    """
    PPO agent for maneuver planning.
    """

    def __init__(self, config: Dict):
        """
        Initialize PPO agent.

        Args:
            config: Agent configuration
        """
        super(PPOAgent, self).__init__()

        self.config = config
        self.logger = logging.getLogger(__name__)

        # Dimensions
        self.state_dim = config.get('state_dim', 12)  # 6 for each satellite
        self.action_dim = config.get('action_dim', 3)  # 3D thrust
        self.hidden_dim = config.get('hidden_dim', 256)

        # Networks
        self.actor = ActorNetwork(self.state_dim, self.action_dim, self.hidden_dim)
        self.critic = CriticNetwork(self.state_dim, self.hidden_dim)

        # Old networks for importance sampling
        self.actor_old = ActorNetwork(self.state_dim, self.action_dim, self.hidden_dim)
        self.critic_old = CriticNetwork(self.state_dim, self.hidden_dim)

        # Copy parameters to old networks
        self._update_old_networks()

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.get('actor_lr', 3e-4))
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.get('critic_lr', 1e-3))

        # PPO parameters
        self.clip_param = config.get('clip_param', 0.2)
        self.value_loss_coef = config.get('value_loss_coef', 0.5)
        self.entropy_coef = config.get('entropy_coef', 0.01)
        self.max_grad_norm = config.get('max_grad_norm', 0.5)

        # Action scaling
        self.action_scale = config.get('action_scale', 1.0)
        self.action_bias = config.get('action_bias', 0.0)

        # Training buffers
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

    def _update_old_networks(self):
        """Update old networks with current network parameters."""
        self.actor_old.load_state_dict(self.actor.state_dict())
        self.critic_old.load_state_dict(self.critic.state_dict())

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, float]:
        """
        Select action given state.

        Args:
            state: Current state
            deterministic: Whether to select deterministically

        Returns:
            Tuple of (action, log_probability)
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        with torch.no_grad():
            action_logits = self.actor_old(state_tensor)
            value = self.critic_old(state_tensor)

        # Create distribution
        action_std = torch.ones_like(action_logits) * 0.1  # Fixed std for simplicity
        dist = Normal(action_logits, action_std)

        if deterministic:
            action = action_logits.squeeze(0).numpy()
        else:
            action = dist.sample().squeeze(0).numpy()

        log_prob = dist.log_prob(torch.FloatTensor(action)).sum().item()

        # Scale action
        action = action * self.action_scale + self.action_bias

        return action, log_prob

    def store_transition(self, state: np.ndarray, action: np.ndarray,
                        log_prob: float, reward: float, value: float, done: bool):
        """
        Store transition in buffer.

        Args:
            state: State
            action: Action
            log_prob: Log probability of action
            reward: Reward
            value: Value estimate
            done: Whether episode ended
        """
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def compute_gae(self, rewards: List[float], values: List[float], dones: List[bool],
                   gamma: float = 0.99, gae_lambda: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Generalized Advantage Estimation (GAE).

        Args:
            rewards: Episode rewards
            values: Value estimates
            dones: Done flags
            gamma: Discount factor
            gae_lambda: GAE lambda

        Returns:
            Tuple of (advantages, returns)
        """
        advantages = np.zeros(len(rewards))
        returns = np.zeros(len(rewards))

        gae = 0
        next_value = 0

        for i in reversed(range(len(rewards))):
            if dones[i]:
                next_value = 0

            delta = rewards[i] + gamma * next_value - values[i]
            gae = delta + gamma * gae_lambda * gae
            advantages[i] = gae
            returns[i] = gae + values[i]
            next_value = values[i]

        return advantages, returns

    def update(self, states: List[np.ndarray], actions: List[np.ndarray],
              rewards: List[float], log_probs: List[float]) -> Dict[str, float]:
        """
        Update policy and value networks.

        Args:
            states: Batch of states
            actions: Batch of actions
            rewards: Batch of rewards
            log_probs: Batch of log probabilities

        Returns:
            Dictionary of loss metrics
        """
        # Convert to tensors
        states_tensor = torch.FloatTensor(np.array(states))
        actions_tensor = torch.FloatTensor(np.array(actions))
        old_log_probs_tensor = torch.FloatTensor(log_probs)

        # Get current value estimates
        values_tensor = self.critic(states_tensor).squeeze()

        # Compute advantages and returns
        advantages, returns = self.compute_gae(rewards, values_tensor.detach().numpy(),
                                             [False] * len(rewards))  # Assume no dones for simplicity

        advantages_tensor = torch.FloatTensor(advantages)
        returns_tensor = torch.FloatTensor(returns)

        # Normalize advantages
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)

        # PPO update
        for _ in range(self.config.get('ppo_epochs', 10)):
            # Get current policy outputs
            action_logits = self.actor(states_tensor)
            action_std = torch.ones_like(action_logits) * 0.1
            dist = Normal(action_logits, action_std)

            new_log_probs = dist.log_prob(actions_tensor).sum(dim=1)
            entropy = dist.entropy().mean()

            # Importance sampling ratio
            ratios = torch.exp(new_log_probs - old_log_probs_tensor)

            # Clipped surrogate objective
            surr1 = ratios * advantages_tensor
            surr2 = torch.clamp(ratios, 1 - self.clip_param, 1 + self.clip_param) * advantages_tensor
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            new_values = self.critic(states_tensor).squeeze()
            value_loss = F.mse_loss(new_values, returns_tensor)

            # Total loss
            total_loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy

            # Update actor
            self.actor_optimizer.zero_grad()
            policy_loss.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()

            # Update critic
            self.critic_optimizer.zero_grad()
            value_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.critic_optimizer.step()

        # Update old networks
        self._update_old_networks()

        # Clear buffers
        self.clear_buffers()

        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item(),
            'total_loss': total_loss.item()
        }

    def clear_buffers(self):
        """Clear experience buffers."""
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

    def save_checkpoint(self, path: str):
        """Save agent checkpoint."""
        checkpoint = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'config': self.config
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str):
        """Load agent checkpoint."""
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self._update_old_networks()


class RecurrentPPOAgent(PPOAgent):
    """
    PPO agent with recurrent policy for handling temporal dependencies.
    """

    def __init__(self, config: Dict):
        super(RecurrentPPOAgent, self).__init__(config)

        # Recurrent layers
        self.actor_lstm = nn.LSTM(self.hidden_dim, self.hidden_dim, batch_first=True)
        self.critic_lstm = nn.LSTM(self.hidden_dim, self.hidden_dim, batch_first=True)

        # Update networks to include LSTM
        self.actor = nn.Sequential(
            nn.Linear(self.state_dim, self.hidden_dim),
            nn.ReLU(),
            # LSTM will be applied separately
            nn.Linear(self.hidden_dim, self.action_dim)
        )

        self.critic = nn.Sequential(
            nn.Linear(self.state_dim, self.hidden_dim),
            nn.ReLU(),
            # LSTM will be applied separately
            nn.Linear(self.hidden_dim, 1)
        )

    def select_action(self, state: np.ndarray, hidden_state: Optional[Tuple] = None,
                     deterministic: bool = False) -> Tuple[np.ndarray, float, Tuple]:
        """
        Select action with recurrent state.

        Args:
            state: Current state
            hidden_state: Previous hidden state
            deterministic: Whether to select deterministically

        Returns:
            Tuple of (action, log_probability, new_hidden_state)
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0)  # [1, 1, state_dim]

        with torch.no_grad():
            # Actor forward
            x = self.actor[0](state_tensor)
            if hidden_state is not None:
                x, new_hidden = self.actor_lstm(x, hidden_state[0])
            else:
                x, new_hidden = self.actor_lstm(x)
            action_logits = self.actor[2](x.squeeze(0))

            # Critic forward
            x_critic = self.critic[0](state_tensor)
            if hidden_state is not None:
                x_critic, _ = self.critic_lstm(x_critic, hidden_state[1])
            else:
                x_critic, _ = self.critic_lstm(x_critic)
            value = self.critic[2](x_critic.squeeze(0))

        # Create distribution
        action_std = torch.ones_like(action_logits) * 0.1
        dist = Normal(action_logits, action_std)

        if deterministic:
            action = action_logits.squeeze(0).numpy()
        else:
            action = dist.sample().squeeze(0).numpy()

        log_prob = dist.log_prob(torch.FloatTensor(action)).sum().item()

        # Scale action
        action = action * self.action_scale + self.action_bias

        return action, log_prob, (new_hidden, new_hidden)


if __name__ == "__main__":
    # Example usage
    config = {
        'state_dim': 12,
        'action_dim': 3,
        'hidden_dim': 256,
        'actor_lr': 3e-4,
        'critic_lr': 1e-3,
        'clip_param': 0.2,
        'value_loss_coef': 0.5,
        'entropy_coef': 0.01
    }

    agent = PPOAgent(config)

    # Example state
    state = np.random.randn(12)
    action, log_prob = agent.select_action(state)

    print(f"Selected action: {action}")
    print(f"Action shape: {action.shape}")
    print(f"Log probability: {log_prob}")

    # Test recurrent agent
    recurrent_agent = RecurrentPPOAgent(config)
    action, log_prob, hidden = recurrent_agent.select_action(state)

    print(f"Recurrent action: {action}")
    print(f"Hidden state shapes: {[h.shape for h in hidden[0]]}")