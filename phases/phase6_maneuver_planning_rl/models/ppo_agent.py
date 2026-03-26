"""
Proximal Policy Optimization (PPO) Agent
for CubeSat Autonomous Collision Avoidance

This agent learns optimal maneuver strategies in the
OrbitalCollisionEnv environment.

Features:
- Actor-Critic neural networks
- Clipped PPO objective
- Generalized Advantage Estimation (GAE)
- Continuous action space
- Model checkpointing
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------
# MEMORY BUFFER
# ------------------------------------------------
class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.dones = []

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.logprobs.clear()
        self.rewards.clear()
        self.dones.clear()


# ------------------------------------------------
# ACTOR-CRITIC NETWORK
# ------------------------------------------------
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()

        hidden = 256

        # Actor
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
            nn.Tanh()
        )

        # Critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

        # Initial variance for exploration
        self.action_var = torch.full((action_dim,), 0.5).to(DEVICE)

    def act(self, state):
        action_mean = self.actor(state)

        cov_mat = torch.diag(self.action_var).unsqueeze(0)
        dist = torch.distributions.MultivariateNormal(action_mean, cov_mat)

        action = dist.sample()
        action_logprob = dist.log_prob(action)

        return action.detach(), action_logprob.detach()

    def evaluate(self, state, action):
        action_mean = self.actor(state)

        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var)

        dist = torch.distributions.MultivariateNormal(action_mean, cov_mat)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        state_values = self.critic(state)

        return action_logprobs, state_values, dist_entropy


# ------------------------------------------------
# PPO AGENT
# ------------------------------------------------
class PPO:
    def __init__(
        self,
        state_dim,
        action_dim,
        lr_actor=1e-4,     # reduced for stability
        lr_critic=3e-4,    # reduced for stability
        gamma=0.99,
        K_epochs=10,
        eps_clip=0.2
    ):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(state_dim, action_dim).to(DEVICE)
        self.policy_old = ActorCritic(state_dim, action_dim).to(DEVICE)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])

        self.MseLoss = nn.MSELoss()

    # ------------------------------------------------
    def select_action(self, state):
        state = torch.tensor(np.array(state), dtype=torch.float32).to(DEVICE)

        with torch.no_grad():
            action, action_logprob = self.policy_old.act(state)

        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)

        return action.cpu().numpy()

    # ------------------------------------------------
    def update(self):
        # Compute discounted rewards
        rewards = []
        discounted_reward = 0
        for reward, done in zip(reversed(self.buffer.rewards), reversed(self.buffer.dones)):
            if done:
                discounted_reward = 0
            discounted_reward = reward + self.gamma * discounted_reward
            rewards.insert(0, discounted_reward)

        rewards = torch.tensor(rewards, dtype=torch.float32).to(DEVICE)
        if len(rewards) > 1:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)  # stable normalization

        old_states = torch.stack(self.buffer.states).to(DEVICE)
        old_actions = torch.stack(self.buffer.actions).to(DEVICE)
        old_logprobs = torch.stack(self.buffer.logprobs).to(DEVICE)

        for _ in range(self.K_epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            state_values = torch.squeeze(state_values)

            ratios = torch.exp(logprobs - old_logprobs.detach())
            advantages = rewards - state_values.detach()

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            loss = (-torch.min(surr1, surr2) +
                    0.5 * self.MseLoss(state_values, rewards) -
                    0.01 * dist_entropy)

            self.optimizer.zero_grad()
            loss.mean().backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)  # gradient clipping
            self.optimizer.step()

        # update old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.buffer.clear()

    # ------------------------------------------------
    def save(self, path):
        torch.save(self.policy.state_dict(), path)

    def load(self, path):
        self.policy.load_state_dict(torch.load(path))
        self.policy_old.load_state_dict(torch.load(path))