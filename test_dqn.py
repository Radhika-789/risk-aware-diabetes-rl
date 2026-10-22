"""
test_dqn.py - tests for dqn.py and train_dqn.py. Needs torch. No pytest needed.
Run:  python test_dqn.py        (about 1 minute)
"""
import os
import subprocess
import sys
import tempfile
import time
import traceback

import numpy as np
import torch

from dqn import DQNAgent, DQNPolicy, ReplayBuffer
from rollout import run_episode
from t1d_env import T1DRiskEnv, TEST_PATIENTS

RESULTS = []


def test(fn):
    t0 = time.time()
    try:
        fn()
        RESULTS.append((fn.__name__, True))
        print(f"PASS  {fn.__name__}  ({time.time() - t0:.1f}s)")
    except Exception as e:  # noqa: BLE001
        RESULTS.append((fn.__name__, False))
        print(f"FAIL  {fn.__name__}: {e!r}")
        traceback.print_exc()
    return fn


@test
def test_replay_buffer_wraps_and_samples():
    b = ReplayBuffer(5, 3)
    for i in range(8):
        b.add(np.full(3, i), i % 2, float(i), np.full(3, i + 1), 0.0)
    assert len(b) == 5
    s, a, r, s2, d = b.sample(4)
    assert s.shape == (4, 3) and a.shape == (4,) and set(r.tolist()) <= {3.0, 4.0, 5.0, 6.0, 7.0}


@test
def test_dqn_learns_trivial_bandit():
    """gamma=0 and all transitions terminal: Q(s,a) must converge to the immediate reward (action 2 = best)."""
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    agent = DQNAgent(obs_dim=4, n_actions=4, lr=3e-3, gamma=0.0, batch_size=32, buffer_size=5000, seed=0)
    for _ in range(2000):
        s = rng.normal(size=4).astype(np.float32)
        a = int(rng.integers(4))
        agent.buffer.add(s, a, 1.0 if a == 2 else 0.0, s, 1.0)
    losses = [agent.update() for _ in range(600)]
    assert np.isfinite(losses).all() and np.mean(losses[-50:]) < np.mean(losses[:50])
    picks = [agent.act(rng.normal(size=4).astype(np.float32), 0.0) for _ in range(50)]
    assert all(p == 2 for p in picks), picks


@test
def test_double_and_vanilla_both_update():
    for double in (True, False):
        agent = DQNAgent(5, 3, batch_size=8, buffer_size=100, double=double, target_update=2)
        for i in range(50):
            agent.buffer.add(np.random.rand(5), i % 3, 1.0, np.random.rand(5), float(i % 7 == 0))
        losses = [agent.update() for _ in range(6)]
        assert all(np.isfinite(l) for l in losses)


@test
def test_save_load_roundtrip():
    agent = DQNAgent(11, 16, seed=3)
    obs = np.linspace(0, 1, 11).astype(np.float32)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "m.pt")
        agent.save(path, {"history_len": 8, "announce_meals": False})
        agent2, cfg = DQNAgent.load(path)
    assert np.allclose(agent.q_values(obs), agent2.q_values(obs)) and cfg["history_len"] == 8


@test
def test_dqn_policy_runs_in_env():
    env = T1DRiskEnv(patients=TEST_PATIENTS, episode_days=0.1)
    agent = DQNAgent(env.obs_dim, env.action_space.n, seed=0)
    m = run_episode(env, DQNPolicy(agent, "dqn_untrained"), seed=4, patient="adult#010")
    assert 0 <= m["tir_70_180"] <= 100 and np.isfinite(m["episode_return"])


@test
def test_train_script_end_to_end_tiny():
    with tempfile.TemporaryDirectory() as d:
        cmd = [sys.executable, "-W", "ignore", "train_dqn.py", "--episodes", "3", "--episode-days", "0.1",
               "--eval-every", "2", "--learning-starts", "5", "--batch-size", "4", "--out-dir", d,
               "--run-name", "tiny"]
        res = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(os.path.abspath(__file__)))
        assert res.returncode == 0, res.stderr[-1500:]
        run = os.path.join(d, "tiny")
        for f in ("train_log.csv", "val_log.csv", "best.pt", "final.pt", "config.json"):
            assert os.path.exists(os.path.join(run, f)), f
        agent, cfg = DQNAgent.load(os.path.join(run, "best.pt"))
        assert cfg["reward_mode"] == "risk" and agent.obs_dim == 11


if __name__ == "__main__":
    n_fail = sum(1 for _, ok in RESULTS if not ok)
    print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} tests passed")
    sys.exit(1 if n_fail else 0)
