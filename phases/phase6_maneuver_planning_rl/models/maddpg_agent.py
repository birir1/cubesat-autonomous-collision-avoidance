"""
MADDPG Multi-Agent Reinforcement Learning

Used for coordinated CubeSat collision avoidance.

Features
--------
• Multiple satellite agents
• Centralized critic
• Decentralized actors
• Replay buffer
• Target networks
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------
# REPLAY BUFFER
# ------------------------------------------------

class ReplayBuffer:

    def __init__(self, buffer_size=100000):

        self.buffer = deque(maxlen=buffer_size)

    def push(self, state, action, reward, next_state, done):

        self.buffer.append(
            (state, action, reward, next_state, done)
        )

    def sample(self, batch_size):

        batch = random.sample(self.buffer, batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )

    def __len__(self):
        return len(self.buffer)


# ------------------------------------------------
# ACTOR NETWORK
# ------------------------------------------------

class Actor(nn.Module):

    def __init__(self, state_dim, action_dim):

        super().__init__()

        hidden = 256

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
            nn.Tanh()
        )

    def forward(self, state):

        return self.net(state)


# ------------------------------------------------
# CRITIC NETWORK
# ------------------------------------------------

class Critic(nn.Module):

    def __init__(self, total_state_dim, total_action_dim):

        super().__init__()

        hidden = 256

        self.net = nn.Sequential(

            nn.Linear(total_state_dim + total_action_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)

        )

    def forward(self, states, actions):

        x = torch.cat([states, actions], dim=1)

        return self.net(x)


# ------------------------------------------------
# LIGHTWEIGHT NETWORKS (New)
# ------------------------------------------------

class LightweightActor(nn.Module):

    def __init__(self, state_dim, action_dim, hidden=64):

        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
            nn.Tanh()
        )

    def forward(self, state):
        return self.net(state)


class LightweightCritic(nn.Module):

    def __init__(self, total_state_dim, total_action_dim, hidden=64):

        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(total_state_dim + total_action_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, states, actions):
        x = torch.cat([states, actions], dim=1)
        return self.net(x)


# ------------------------------------------------
# TRANSFORMER NETWORKS (New)
# ------------------------------------------------

class TransformerActor(nn.Module):

    def __init__(self, state_dim, action_dim, nhead=4, num_layers=2, dim_feedforward=128):

        super().__init__()

        self.input_proj = nn.Linear(state_dim, dim_feedforward)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_feedforward,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_proj = nn.Linear(dim_feedforward, action_dim)

    def forward(self, state):

        # Add sequence dimension for transformer
        if state.dim() == 2:
            state = state.unsqueeze(1)  # (batch, 1, state_dim)

        x = self.input_proj(state)
        x = self.transformer(x)

        # Take the first (and only) sequence element
        x = x[:, 0, :]

        action = self.output_proj(x)
        return torch.tanh(action)


class TransformerCritic(nn.Module):

    def __init__(self, total_state_dim, total_action_dim, nhead=4, num_layers=2, dim_feedforward=128):

        super().__init__()

        input_dim = total_state_dim + total_action_dim
        self.input_proj = nn.Linear(input_dim, dim_feedforward)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_feedforward,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_proj = nn.Linear(dim_feedforward, 1)

    def forward(self, states, actions):

        x = torch.cat([states, actions], dim=1)

        # Add sequence dimension for transformer
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, input_dim)

        x = self.input_proj(x)
        x = self.transformer(x)

        # Take the first (and only) sequence element
        x = x[:, 0, :]

        return self.output_proj(x)


# ------------------------------------------------
# AGENT
# ------------------------------------------------

class MADDPGAgent:

    def __init__(
        self,
        state_dim,
        action_dim,
        total_state_dim,
        total_action_dim,
        lr_actor=1e-4,
        lr_critic=1e-3,
        device=None,
        model_type='standard'
    ):

        self.device = device if device is not None else DEVICE

        if model_type == 'lightweight':
            self.actor = LightweightActor(state_dim, action_dim).to(self.device)
            self.actor_target = LightweightActor(state_dim, action_dim).to(self.device)

            self.critic = LightweightCritic(total_state_dim, total_action_dim).to(self.device)
            self.critic_target = LightweightCritic(total_state_dim, total_action_dim).to(self.device)
        elif model_type == 'transformer':
            self.actor = TransformerActor(state_dim, action_dim).to(self.device)
            self.actor_target = TransformerActor(state_dim, action_dim).to(self.device)

            self.critic = TransformerCritic(total_state_dim, total_action_dim).to(self.device)
            self.critic_target = TransformerCritic(total_state_dim, total_action_dim).to(self.device)
        else:
            self.actor = Actor(state_dim, action_dim).to(self.device)
            self.actor_target = Actor(state_dim, action_dim).to(self.device)

            self.critic = Critic(total_state_dim, total_action_dim).to(self.device)
            self.critic_target = Critic(total_state_dim, total_action_dim).to(self.device)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

    def act(self, state):

        state = torch.FloatTensor(state).to(self.device)

        action = self.actor(state)

        return action.detach().cpu().numpy()


# ------------------------------------------------
# MADDPG SYSTEM
# ------------------------------------------------

class MADDPG:

    def __init__(
        self,
        num_agents,
        state_dim,
        action_dim,
        gamma=0.95,
        tau=0.01,
        device=None,
        model_type='standard'
    ):

        self.num_agents = num_agents
        self.gamma = gamma
        self.tau = tau
        self.device = device if device is not None else DEVICE
        self.model_type = model_type

        total_state_dim = state_dim * num_agents
        total_action_dim = action_dim * num_agents

        self.agents = [

            MADDPGAgent(
                state_dim,
                action_dim,
                total_state_dim,
                total_action_dim,
                device=self.device,
                model_type=self.model_type
            )

            for _ in range(num_agents)

        ]

        self.replay_buffer = ReplayBuffer()

    # ------------------------------------------------

    def select_actions(self, states):

        actions = []

        for i, agent in enumerate(self.agents):

            action = agent.act(states[i])

            actions.append(action)

        return actions

    # ------------------------------------------------

    def update(self, batch_size=256):

        if len(self.replay_buffer) < batch_size:
            return 0.0, 0.0

        states, actions, rewards, next_states, dones = \
            self.replay_buffer.sample(batch_size)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        total_states = states.view(batch_size, -1)
        total_actions = actions.view(batch_size, -1)

        next_actions = []

        for i, agent in enumerate(self.agents):

            next_action = agent.actor_target(next_states[:, i])

            next_actions.append(next_action)

        next_actions = torch.cat(next_actions, dim=1)

        total_next_states = next_states.view(batch_size, -1)

        actor_loss_total = 0.0
        critic_loss_total = 0.0

        # update each agent
        for i, agent in enumerate(self.agents):

            # Critic update
            q_target = agent.critic_target(
                total_next_states,
                next_actions
            )

            y = rewards[:, i].unsqueeze(1) + \
                self.gamma * q_target * (1 - dones[:, i].unsqueeze(1))

            q = agent.critic(
                total_states,
                total_actions
            )

            critic_loss = nn.MSELoss()(q, y.detach())

            agent.critic_optimizer.zero_grad()
            critic_loss.backward()
            agent.critic_optimizer.step()

            critic_loss_total += critic_loss.item()

            # Actor update
            current_actions = []

            for j, other_agent in enumerate(self.agents):

                if j == i:
                    current_actions.append(
                        other_agent.actor(states[:, j])
                    )
                else:
                    current_actions.append(
                        actions[:, j]
                    )

            current_actions = torch.cat(current_actions, dim=1)

            actor_loss = -agent.critic(
                total_states,
                current_actions
            ).mean()

            agent.actor_optimizer.zero_grad()
            actor_loss.backward()
            agent.actor_optimizer.step()

            actor_loss_total += actor_loss.item()

            # Soft update targets
            self._soft_update(agent.actor, agent.actor_target)
            self._soft_update(agent.critic, agent.critic_target)

        return actor_loss_total / self.num_agents, critic_loss_total / self.num_agents

    # ------------------------------------------------

    def _soft_update(self, net, target_net):

        for target_param, param in zip(
            target_net.parameters(),
            net.parameters()
        ):

            target_param.data.copy_(
                self.tau * param.data +
                (1 - self.tau) * target_param.data
            )


def create_maddpg(num_agents, state_dim, action_dim, model_type='standard', **kwargs):
    """Factory for standard vs lightweight MADDPG."""
    return MADDPG(
        num_agents=num_agents,
        state_dim=state_dim,
        action_dim=action_dim,
        model_type=model_type,
        **kwargs
    )

    # ------------------------------------------------

    def save(self, path):
        """Save MADDPG model weights."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        checkpoint = {
            'num_agents': self.num_agents,
            'gamma': self.gamma,
            'tau': self.tau,
            'agents': [
                {
                    'actor': agent.actor.state_dict(),
                    'critic': agent.critic.state_dict(),
                    'actor_target': agent.actor_target.state_dict(),
                    'critic_target': agent.critic_target.state_dict()
                }
                for agent in self.agents
            ]
        }
        torch.save(checkpoint, path)

    def load(self, path):
        """Load MADDPG model weights."""
        checkpoint = torch.load(path, map_location=self.device)
        self.num_agents = checkpoint['num_agents']
        self.gamma = checkpoint['gamma']
        self.tau = checkpoint['tau']

        for agent, agent_ckpt in zip(self.agents, checkpoint['agents']):
            agent.actor.load_state_dict(agent_ckpt['actor'])
            agent.critic.load_state_dict(agent_ckpt['critic'])
            agent.actor_target.load_state_dict(agent_ckpt['actor_target'])
            agent.critic_target.load_state_dict(agent_ckpt['critic_target'])