"""
dqn.py - compact DQN for the discrete-action T1DRiskEnv (research simulation only).

Standard DQN: MLP Q-network, uniform replay buffer, periodic hard target-network copy,
Huber loss, gradient clipping. `double=True` (default) uses Double-DQN targets
(action chosen by the online net, valued by the target net); `double=False` is vanilla DQN.
Terminal flag for bootstrapping = environment TERMINATION only. Time-limit truncation is
NOT treated as terminal (we keep bootstrapping through it).
"""
from __future__ import annotations

import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from baselines import Policy


class QNet(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, seed: int = 0):
        self.cap, self.n, self.i = capacity, 0, 0
        self.s = np.zeros((capacity, obs_dim), np.float32)
        self.s2 = np.zeros((capacity, obs_dim), np.float32)
        self.a = np.zeros(capacity, np.int64)
        self.r = np.zeros(capacity, np.float32)
        self.d = np.zeros(capacity, np.float32)
        self.rng = np.random.default_rng(seed)

    def add(self, s, a, r, s2, d):
        i = self.i
        self.s[i], self.a[i], self.r[i], self.s2[i], self.d[i] = s, a, r, s2, d
        self.i = (i + 1) % self.cap
        self.n = min(self.n + 1, self.cap)

    def sample(self, batch: int):
        idx = self.rng.integers(0, self.n, size=batch)
        return self.s[idx], self.a[idx], self.r[idx], self.s2[idx], self.d[idx]

    def __len__(self):
        return self.n


class DQNAgent:
    def __init__(self, obs_dim, n_actions, lr=1e-3, gamma=0.99, batch_size=64, buffer_size=100_000,
                 target_update=500, double=True, hidden=128, grad_clip=10.0, seed=0):
        # plain ints: gym's action_space.n is a numpy integer, which weights_only checkpoint loading rejects
        self.obs_dim, self.n_actions, self.hidden = int(obs_dim), int(n_actions), int(hidden)
        self.gamma, self.batch_size, self.target_update = gamma, batch_size, target_update
        self.double, self.grad_clip = double, grad_clip
        self.q = QNet(obs_dim, n_actions, hidden)
        self.q_target = QNet(obs_dim, n_actions, hidden)
        self.q_target.load_state_dict(self.q.state_dict())
        self.q_target.eval()
        self.opt = torch.optim.Adam(self.q.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size, obs_dim, seed)
        self.rng = np.random.default_rng(seed + 1)
        self.updates = 0

    @torch.no_grad()
    def q_values(self, obs) -> np.ndarray:
        return self.q(torch.as_tensor(np.asarray(obs), dtype=torch.float32).unsqueeze(0))[0].numpy()

    def act(self, obs, eps: float = 0.0) -> int:
        if self.rng.random() < eps:
            return int(self.rng.integers(self.n_actions))
        return int(np.argmax(self.q_values(obs)))

    def update(self):
        if len(self.buffer) < self.batch_size:
            return None
        s, a, r, s2, d = (torch.as_tensor(x) for x in self.buffer.sample(self.batch_size))
        q = self.q(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            if self.double:
                a2 = self.q(s2).argmax(1, keepdim=True)
                q2 = self.q_target(s2).gather(1, a2).squeeze(1)
            else:
                q2 = self.q_target(s2).max(1).values
            target = r + self.gamma * (1.0 - d) * q2
        loss = F.smooth_l1_loss(q, target)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), self.grad_clip)
        self.opt.step()
        self.updates += 1
        if self.updates % self.target_update == 0:
            self.q_target.load_state_dict(self.q.state_dict())
        return float(loss.item())

    def save(self, path, env_config: dict | None = None):
        torch.save({
            "state_dict": self.q.state_dict(),
            "obs_dim": self.obs_dim, "n_actions": self.n_actions, "hidden": self.hidden,
            "env_config_json": json.dumps(env_config or {}),
        }, path)

    @classmethod
    def load(cls, path):
        ck = torch.load(path, map_location="cpu", weights_only=True)
        agent = cls(ck["obs_dim"], ck["n_actions"], hidden=ck["hidden"], buffer_size=1)
        agent.q.load_state_dict(ck["state_dict"])
        agent.q_target.load_state_dict(ck["state_dict"])
        return agent, json.loads(ck["env_config_json"])


class DQNPolicy(Policy):
    """Greedy (epsilon = 0) policy wrapper for evaluation."""

    def __init__(self, agent: DQNAgent, name: str = "dqn"):
        self.agent, self.name = agent, name

    def act(self, obs, env):
        return self.agent.act(obs, 0.0)
