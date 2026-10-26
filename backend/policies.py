"""Policy catalogue: baselines from baselines.py plus DQN checkpoints from runs/.

Policy ids:
  "random" | "fixed_basal" | "cgm_rule" | "announced_bb"   -> baselines.make_baselines()
  "ckpt:<run_name>/<file>.pt"                              -> dqn.DQNAgent.load(runs/<run_name>/<file>.pt)
"""
from __future__ import annotations

import os

from baselines import make_baselines

BASELINE_IDS = ("random", "fixed_basal", "cgm_rule", "announced_bb")
RUNS_DIR = os.environ.get("GG_RUNS_DIR", "runs")


def is_checkpoint_id(policy_id: str) -> bool:
    return policy_id.startswith("ckpt:")


def checkpoint_path(policy_id: str) -> str:
    rel = policy_id[len("ckpt:"):]
    run, fname = rel.split("/", 1)
    if "/" in fname or ".." in run or ".." in fname or not fname.endswith(".pt"):
        raise ValueError(f"bad checkpoint id {policy_id!r}")
    return os.path.join(RUNS_DIR, run, fname)


def load_policy(policy_id: str, seed: int = 0):
    """Return (policy, env_config_from_checkpoint_or_None)."""
    if policy_id in BASELINE_IDS:
        for p in make_baselines(seed=seed):
            if p.name == policy_id:
                return p, None
    if is_checkpoint_id(policy_id):
        from dqn import DQNAgent, DQNPolicy  # torch only when needed
        path = checkpoint_path(policy_id)
        if not os.path.exists(path):
            raise FileNotFoundError(f"checkpoint not found: {path}")
        agent, cfg = DQNAgent.load(path)
        return DQNPolicy(agent, policy_id, mask_actions=bool(cfg.get("mask_actions", False))), cfg
    raise ValueError(f"unknown policy id {policy_id!r}")
