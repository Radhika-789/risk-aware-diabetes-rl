"""
train_dqn.py - train a DQN on the TRAIN virtual patients, select the checkpoint on VALIDATION patients.
Research simulation only. The test patients are never touched here (see evaluate.py).

Quick pipeline check (~2-3 min):   python train_dqn.py --episodes 12 --eval-every 6 --run-name quick
Real run (budget ~40 min/300 ep):  python train_dqn.py --episodes 300 --seed 0 --run-name dqn_risk_s0
Outputs in runs/<run-name>/ : train_log.csv, val_log.csv, best.pt, final.pt, config.json
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time

import numpy as np
import torch

from dqn import DQNAgent, DQNPolicy
from rollout import evaluate_policy, metrics_from_log
from t1d_env import RewardConfig, ShieldConfig, T1DRiskEnv, TRAIN_PATIENTS, VAL_PATIENTS

VAL_SEED_BASE = 10_000_000   # fixed validation episodes, identical at every evaluation


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-name", default="dqn_risk_s0")
    p.add_argument("--out-dir", default="runs")
    #p.add_argument("--reward-mode", choices=["risk", "tir"], default="risk")
    p.add_argument("--reward-mode", choices=["risk", "tir", "risk_tails"], default="risk")  #from here
    p.add_argument("--very-high-extra", type=float, default=None)
    p.add_argument("--very-high-slope", type=float, default=None)
    p.add_argument("--severe-extra-tails", type=float, default=None)
    p.add_argument("--select-by", choices=["return", "clinical"], default="return") #to here
    p.add_argument("--episode-days", type=float, default=1.0)
    p.add_argument("--action-repeat", type=int, default=5)
    p.add_argument("--history-len", type=int, default=8)
    p.add_argument("--announce-meals", action="store_true")
    p.add_argument("--no-shield", action="store_true")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--buffer-size", type=int, default=100_000)
    p.add_argument("--target-update", type=int, default=500, help="gradient steps between target copies")
    p.add_argument("--no-double", action="store_true", help="vanilla DQN targets instead of Double-DQN")
    p.add_argument("--learning-starts", type=int, default=1000, help="env steps before updates begin")
    p.add_argument("--eps-start", type=float, default=1.0)
    p.add_argument("--eps-end", type=float, default=0.05)
    p.add_argument("--eps-decay-frac", type=float, default=0.5, help="fraction of episodes for linear decay")
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--val-episodes", type=int, default=1, help="episodes per validation patient")
    p.add_argument("--threads", type=int, default=1)
    return p.parse_args()


def env_config(args):
    #return dict(episode_days=args.episode_days, action_repeat=args.action_repeat,
    #            history_len=args.history_len, announce_meals=args.announce_meals,
    #            reward_mode=args.reward_mode, shield=not args.no_shield)
    overrides = {"very_high_extra": args.very_high_extra, "very_high_slope": args.very_high_slope,
                 "severe_extra_tails": args.severe_extra_tails}
    return dict(episode_days=args.episode_days, action_repeat=args.action_repeat,
                history_len=args.history_len, announce_meals=args.announce_meals,
                reward_mode=args.reward_mode, shield=not args.no_shield,
                reward_params={k: v for k, v in overrides.items() if v is not None})

def make_env(patients, cfg):
    return T1DRiskEnv(
        patients=patients, episode_days=cfg["episode_days"], action_repeat=cfg["action_repeat"],
        history_len=cfg["history_len"], announce_meals=cfg["announce_meals"],
        #reward_cfg=RewardConfig(mode=cfg["reward_mode"]), shield_cfg=ShieldConfig(enabled=cfg["shield"]))
        reward_cfg=RewardConfig(mode=cfg["reward_mode"], **cfg.get("reward_params", {})), shield_cfg=ShieldConfig(enabled=cfg["shield"]))


def epsilon_at(ep, eps_start, eps_end, decay_episodes):
    """Linear decay from eps_start to eps_end over `decay_episodes` episodes, then constant."""
    return max(eps_end, eps_start - (eps_start - eps_end) * ep / decay_episodes)


def mean_of(rows, key):
    return float(np.mean([r[key] for r in rows]))

def clinical_score(rows):   #from here
    return (mean_of(rows, "tir_70_180")
            - 3.0 * max(0.0, mean_of(rows, "tbr_lt70") - 4.0)
            - 5.0 * max(0.0, mean_of(rows, "tbr_lt54") - 1.0)
            - 2.0 * max(0.0, mean_of(rows, "tar_gt250") - 5.0)
            - 100.0 * mean_of(rows, "terminated_early"))   #to here
  
def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)

    cfg = env_config(args)
    run_dir = os.path.join(args.out_dir, args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump({"args": vars(args), "env_config": cfg, "train_patients": TRAIN_PATIENTS,
                   "val_patients": VAL_PATIENTS}, f, indent=2)

    env, val_env = make_env(TRAIN_PATIENTS, cfg), make_env(VAL_PATIENTS, cfg)
    agent = DQNAgent(env.obs_dim, env.action_space.n, lr=args.lr, gamma=args.gamma,
                     batch_size=args.batch_size, buffer_size=args.buffer_size,
                     target_update=args.target_update, double=not args.no_double, seed=args.seed)

    max_steps = args.episodes * env.max_decisions
    if max_steps < args.learning_starts:
        print(f"WARNING: at most {max_steps} environment steps can run but --learning-starts is {args.learning_starts}. "
              "No gradient update will happen, so the network stays untrained (loss prints as nan). "
              "Fine as a pipeline check; not a real training run.", flush=True)

    train_f = open(os.path.join(run_dir, "train_log.csv"), "w", newline="")
    val_f = open(os.path.join(run_dir, "val_log.csv"), "w", newline="")
    train_w = csv.writer(train_f)
    val_w = csv.writer(val_f)
    train_w.writerow(["episode", "env_steps", "patient", "epsilon", "return", "tir", "tbr_lt70", "tbr_lt54",
                      "tar_gt180", "mean_bg", "terminated", "mean_loss", "shield_suspends", "elapsed_s"])
    val_w.writerow(["episode", "val_return", "val_tir", "val_tbr_lt70", "val_tbr_lt54", "val_terminated", "is_best"])

    best_val, total_steps, t0 = -np.inf, 0, time.time()
    decay_eps = max(1, int(args.eps_decay_frac * args.episodes))
    for ep in range(args.episodes):
        eps = epsilon_at(ep, args.eps_start, args.eps_end, decay_eps)
        obs, _ = env.reset(seed=args.seed * 1_000_000 + ep)      # fresh patient / meals / noise each episode
        done, ret, losses = False, 0.0, []
        while not done:
            a = agent.act(obs, eps)
            obs2, r, term, trunc, _ = env.step(a)
            agent.buffer.add(obs, a, r, obs2, float(term))         # bootstrap through time-limit truncation
            obs, ret, done = obs2, ret + r, term or trunc
            total_steps += 1
            if total_steps >= args.learning_starts:
                loss = agent.update()
                if loss is not None:
                    losses.append(loss)
        log = env.episode_log()
        m = metrics_from_log(env, log)
        train_w.writerow([ep + 1, total_steps, log["patient"], round(eps, 3), round(ret, 3),
                          round(m["tir_70_180"], 2), round(m["tbr_lt70"], 2), round(m["tbr_lt54"], 2),
                          round(m["tar_gt180"], 2), round(m["mean_bg"], 1), int(log["terminated"]),
                          round(float(np.mean(losses)), 5) if losses else "", log["shield_suspend_decisions"],
                          round(time.time() - t0, 1)])
        train_f.flush()
        print(f"ep {ep + 1:4d}/{args.episodes} | {log['patient']:15s} | eps {eps:.2f} | return {ret:8.1f} | "
              f"TIR {m['tir_70_180']:5.1f}% | <70 {m['tbr_lt70']:5.1f}% | loss "
              f"{(np.mean(losses) if losses else float('nan')):.4f} | {time.time() - t0:6.0f}s", flush=True)

        if (ep + 1) % args.eval_every == 0 or ep + 1 == args.episodes:
            rows = evaluate_policy(val_env, DQNPolicy(agent), VAL_PATIENTS, args.val_episodes, VAL_SEED_BASE)
            v_ret = mean_of(rows, "episode_return")
            v_sel = clinical_score(rows) if args.select_by == "clinical" else v_ret
            is_best = v_ret > best_val
            if is_best:
                best_val = v_ret
                agent.save(os.path.join(run_dir, "best.pt"), cfg)
            val_w.writerow([ep + 1, round(v_ret, 3), round(mean_of(rows, "tir_70_180"), 2),
                            round(mean_of(rows, "tbr_lt70"), 2), round(mean_of(rows, "tbr_lt54"), 2),
                            round(mean_of(rows, "terminated_early"), 3), int(is_best)])
            val_f.flush()
            print(f"   VAL @ep {ep + 1}: return {v_ret:.1f} | TIR {mean_of(rows, 'tir_70_180'):.1f}% | "
                  #f"<70 {mean_of(rows, 'tbr_lt70'):.1f}% {'(new best)' if is_best else ''}", flush=True
                                    f"<70 {mean_of(rows, 'tbr_lt70'):.1f}% | <54 {mean_of(rows, 'tbr_lt54'):.1f}% | "
                                    f">250 {mean_of(rows, 'tar_gt250'):.1f}% {'(new best)' if is_best else ''}", flush=True)

    agent.save(os.path.join(run_dir, "final.pt"), cfg)
    train_f.close()
    val_f.close()
    print(f"done in {time.time() - t0:.0f}s. Checkpoints: {run_dir}/best.pt (val-selected), {run_dir}/final.pt")


if __name__ == "__main__":
    main()