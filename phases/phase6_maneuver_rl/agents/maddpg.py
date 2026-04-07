"""MADDPG model implementation for multi-agent collision avoidance."""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List


class MADDPGActor(nn.Module):
    """Simple actor network for MADDPG agents."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super(MADDPGActor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class MADDPG:
    """Multi-agent DDPG wrapper used for benchmark scenarios."""

    def __init__(self, num_agents: int = 3, state_dim: int = 24, action_dim: int = 2,
                 hidden_dim: int = 128, device: torch.device = torch.device('cpu')):
        self.device = device
        self.num_agents = num_agents
        self.agents = [self._build_agent(state_dim, action_dim, hidden_dim) for _ in range(num_agents)]

    def _build_agent(self, state_dim: int, action_dim: int, hidden_dim: int):
        agent = type('Agent', (), {})()
        agent.actor = MADDPGActor(state_dim, action_dim, hidden_dim).to(self.device)
        agent.actor.eval()
        agent.optimizer = optim.Adam(agent.actor.parameters(), lr=1e-3)
        return agent

    def act(self, observations: List[np.ndarray]) -> List[np.ndarray]:
        actions = []
        with torch.no_grad():
            for idx, obs in enumerate(observations):
                tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                action = self.agents[idx].actor(tensor).cpu().numpy()[0]
                actions.append(action)
        return actions
