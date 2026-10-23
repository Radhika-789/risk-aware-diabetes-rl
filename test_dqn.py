"""
test_dqn.py - tests for dqn.py and train_dqn.py. Needs torch. No pytest needed.
Run:  python test_dqn.py        (about 1 minute)
"""
import csv
import os
import subprocess
import sys
import tempfile
import time
import traceback

import numpy as np
import torch

from dqn import DQNAgent, DQNPolicy, ReplayBuffer
from evaluate import resolve_labels
from rollout import run_episode
from t1d_env import T1DRiskEnv, TEST_PATIENTS
from train_dqn import epsilon_at

HERE = os.path.dirname(os.path.abspath(__file__))

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



# ----------------------------------------------------------------------------------------------
# Phase 1 audit tests
# ----------------------------------------------------------------------------------------------
def _constant_net(net, bias):
    """Make a Q-network output a fixed vector (weights zeroed, last-layer bias = `bias`)."""
    last = net.net[-1]
    last.weight.data.zero_()
    last.bias.data.copy_(torch.tensor(bias, dtype=torch.float32))


@test
def test_double_dqn_target_is_exact():
    def loss_for(double, done):
        agent = DQNAgent(3, 3, gamma=1.0, batch_size=1, buffer_size=4, double=double, seed=0)
        _constant_net(agent.q, [0.0, 1.0, 0.0])          # online net prefers action 1
        _constant_net(agent.q_target, [5.0, 2.0, 0.0])   # target net: action 0 -> 5 (its max), action 1 -> 2
        agent.buffer.add(np.zeros(3, np.float32), 0, 0.0, np.zeros(3, np.float32), done)
        return agent.update()
    assert abs(loss_for(True, 0.0) - 1.5) < 1e-6    # Double: target = q_target[argmax q_online] = 2 -> Huber(0-2) = 1.5
    assert abs(loss_for(False, 0.0) - 4.5) < 1e-6   # vanilla: target = max q_target = 5 -> Huber(0-5) = 4.5
    assert abs(loss_for(True, 1.0)) < 1e-6          # terminal: no bootstrap, target = reward = 0 -> loss 0


@test
def test_target_network_hard_copy_schedule():
    agent = DQNAgent(4, 3, lr=1e-2, gamma=0.9, batch_size=8, buffer_size=100, target_update=3, seed=0)
    rng = np.random.default_rng(0)
    for _ in range(50):
        agent.buffer.add(rng.normal(size=4).astype(np.float32), int(rng.integers(3)), float(rng.normal()),
                         rng.normal(size=4).astype(np.float32), 0.0)

    def same():
        return all(torch.equal(a, b) for a, b in zip(agent.q.parameters(), agent.q_target.parameters()))
    assert same()
    flags = []
    for _ in range(6):
        agent.update()
        flags.append(same())
    assert flags == [False, False, True, False, False, True], flags


@test
def test_epsilon_schedule():
    vals = [epsilon_at(e, 1.0, 0.05, 10) for e in range(20)]
    assert vals[0] == 1.0 and abs(vals[5] - 0.525) < 1e-12 and abs(vals[10] - 0.05) < 1e-12
    assert vals[15] == 0.05 == vals[19]
    assert all(a >= b for a, b in zip(vals, vals[1:]))


@test
def test_checkpoint_labels_are_unique_or_rejected():
    assert resolve_labels(["runs/a/best.pt", "runs/a/final.pt", "runs/b/best.pt"], [], ["random"]) == \
        ["a_best", "a_final", "b_best"]
    for ck, labels in ((["p1.pt", "p2.pt"], ["x", "x"]), (["p1.pt"], ["random"]), (["p1.pt", "p2.pt"], ["only_one"])):
        try:
            resolve_labels(ck, labels, ["random"])
            raise AssertionError(f"should have been rejected: {labels}")
        except SystemExit:
            pass


def _train(out_dir, run_name, extra):
    cmd = [sys.executable, "-W", "ignore", "train_dqn.py", "--episodes", "3", "--episode-days", "0.1",
           "--eval-every", "2", "--batch-size", "4", "--out-dir", out_dir, "--run-name", run_name] + extra
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=HERE)
    assert res.returncode == 0, res.stderr[-1500:]
    return res


def _rows(path):
    return list(csv.DictReader(open(path)))


@test
def test_learning_starts_reproducibility_and_evaluate_end_to_end():
    with tempfile.TemporaryDirectory() as d:
        # (1) learning-starts above the number of available steps -> warning, no updates, empty mean_loss
        r_no = _train(d, "no_updates", ["--learning-starts", "100000"])
        assert "WARNING" in r_no.stdout
        assert all(r["mean_loss"] == "" for r in _rows(os.path.join(d, "no_updates", "train_log.csv")))
        # (2) with a small threshold updates run (finite numeric loss) and the run is reproducible from its seed
        _train(d, "a", ["--learning-starts", "5"])
        _train(d, "b", ["--learning-starts", "5"])
        ra, rb = _rows(os.path.join(d, "a", "train_log.csv")), _rows(os.path.join(d, "b", "train_log.csv"))
        assert all(r["mean_loss"] not in ("", "nan") and np.isfinite(float(r["mean_loss"])) for r in ra)
        drop = lambda rows: [{k: v for k, v in r.items() if k != "elapsed_s"} for r in rows]  # noqa: E731
        assert drop(ra) == drop(rb), "same seed must reproduce the training log exactly"
        for name in ("best.pt", "final.pt"):
            wa = DQNAgent.load(os.path.join(d, "a", name))[0].q.state_dict()
            wb = DQNAgent.load(os.path.join(d, "b", name))[0].q.state_dict()
            assert all(torch.equal(wa[k], wb[k]) for k in wa), f"{name} weights differ between identical runs"
        # (3) evaluate both checkpoints WITHOUT --labels: distinct default labels, 6 episodes each
        out = os.path.join(d, "eval")
        cmd = [sys.executable, "-W", "ignore", "evaluate.py", "--checkpoints",
               os.path.join(d, "a", "best.pt"), os.path.join(d, "a", "final.pt"),
               "--split", "val", "--episodes-per-patient", "1", "--episode-days", "0.1", "--out", out]
        res = subprocess.run(cmd, capture_output=True, text=True, cwd=HERE)
        assert res.returncode == 0, res.stderr[-1500:]
        summary = {r["policy"]: int(r["n_episodes"]) for r in _rows(os.path.join(out, "summary.csv"))}
        assert summary == {"random": 6, "fixed_basal": 6, "cgm_rule": 6, "announced_bb": 6, "a_best": 6, "a_final": 6}, summary
        assert "survival_fraction" in _rows(os.path.join(out, "episodes.csv"))[0]
        assert os.path.exists(os.path.join(out, "summary.md"))


if __name__ == "__main__":
    n_fail = sum(1 for _, ok in RESULTS if not ok)
    print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} tests passed")
    sys.exit(1 if n_fail else 0)