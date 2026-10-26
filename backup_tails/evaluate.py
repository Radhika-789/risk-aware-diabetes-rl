"""
evaluate.py - compare baselines and DQN checkpoints on HELD-OUT virtual patients (research simulation only).

Every policy runs the IDENTICAL episodes (same patient, meals, sensor noise, initial state).
Works without torch if you pass no checkpoints (baselines only).

  python evaluate.py --out results_baselines --episodes-per-patient 2
  python evaluate.py --checkpoints runs/dqn_risk_s0/best.pt runs/dqn_risk_s1/best.pt --labels dqn_s0 dqn_s1 --plot

Outputs in <out>/: episodes.csv (one row per policy x episode), summary.csv, summary.md, traces.png (--plot)
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from baselines import make_baselines
from rollout import evaluate_policy, run_episode
from t1d_env import RewardConfig, ShieldConfig, T1DRiskEnv, TEST_PATIENTS, TRAIN_PATIENTS, VAL_PATIENTS

SPLITS = {"test": TEST_PATIENTS, "val": VAL_PATIENTS, "train": TRAIN_PATIENTS}
KEY_METRICS = ["tir_70_180", "tbr_lt70", "tbr_lt54", "tar_gt180", "tar_gt250", "mean_bg", "cv_bg", "lbgi", "hbgi",
               "terminated_early", "survival_fraction", "total_insulin_u", "episode_return"]
DEFAULT_CFG = dict(episode_days=1.0, action_repeat=5, history_len=8, announce_meals=False,
                   reward_mode="risk", shield=True)


def make_env(patients, cfg):
    return T1DRiskEnv(
        patients=patients, episode_days=cfg["episode_days"], action_repeat=cfg["action_repeat"],
        history_len=cfg["history_len"], announce_meals=cfg["announce_meals"],
        #reward_cfg=RewardConfig(mode=cfg["reward_mode"]), shield_cfg=ShieldConfig(enabled=cfg["shield"]))
                reward_cfg=RewardConfig(mode=cfg["reward_mode"], **cfg.get("reward_params", {})), shield_cfg=ShieldConfig(enabled=cfg["shield"]))


def resolve_labels(checkpoints, labels, reserved):
    """Unique display names for checkpoints. Names key every per-policy summary row, so duplicates (or reuse of a
    baseline name) would silently merge different policies into one row."""
    if labels:
        if len(labels) != len(checkpoints):
            raise SystemExit("--labels must match --checkpoints (one label per checkpoint)")
        out = list(labels)
    else:
        out = []
        for c in checkpoints:
            run = os.path.basename(os.path.dirname(os.path.normpath(c))) or "dqn"
            out.append(f"{run}_{os.path.splitext(os.path.basename(c))[0]}")
    bad = sorted({l for l in out if out.count(l) > 1} | (set(out) & set(reserved)))
    if bad:
        raise SystemExit(f"checkpoint labels must be unique and must not reuse baseline names; problem: {bad}")
    return out


def boot_ci(x, n=1000, seed=0):
    x = np.asarray(x, dtype=float)
    if x.size < 2:
        return float(x.mean()), float(x.mean())
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n, x.size), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pol, g in df.groupby("policy", sort=False):
        row = {"policy": pol, "n_episodes": len(g)}
        for k in KEY_METRICS:
            lo, hi = boot_ci(g[k].values)
            row[f"{k}_mean"], row[f"{k}_sd"], row[f"{k}_ci_lo"], row[f"{k}_ci_hi"] = g[k].mean(), g[k].std(), lo, hi
        rows.append(row)
    return pd.DataFrame(rows)


def md_table(summary: pd.DataFrame, df: pd.DataFrame, split: str) -> str:
    def cell(r, k, d=1):
        return f"{r[k + '_mean']:.{d}f} [{r[k + '_ci_lo']:.{d}f}, {r[k + '_ci_hi']:.{d}f}]"
    lines = [f"### Overall ({split} patients; mean [95% bootstrap CI over episodes])", "",
             "| Policy | n | TIR 70-180 % | Time <70 % | Time <54 % | Time >180 % | Time >250% | Mean BG | LBGI | HBGI | Terminated | Return |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in summary.iterrows():
        lines.append(f"| {r['policy']} | {int(r['n_episodes'])} | {cell(r, 'tir_70_180')} | {cell(r, 'tbr_lt70')} | "
                     f"{cell(r, 'tbr_lt54')} | {cell(r, 'tar_gt180')} | {cell(r, 'tar_gt250')} | {cell(r, 'mean_bg', 0)} | {cell(r, 'lbgi', 2)} | "
                     f"{cell(r, 'hbgi', 2)} | {r['terminated_early_mean']:.2f} | {cell(r, 'episode_return', 0)} |")
    df = df.assign(group=df["patient"].str.split("#").str[0])
    lines += ["", "### By age group (mean TIR % / mean time <70 %)", "",
              "| Policy | adolescent | adult | child |", "|---|---|---|---|"]
    for pol, g in df.groupby("policy", sort=False):
        cells = []
        for grp in ("adolescent", "adult", "child"):
            s = g[g.group == grp]
            cells.append(f"{s.tir_70_180.mean():.1f} / {s.tbr_lt70.mean():.1f}" if len(s) else "-")
        lines.append(f"| {pol} | " + " | ".join(cells) + " |")
    lines += ["", "Research simulation only. Simulated results do not imply clinical performance."]
    return "\n".join(lines)


def plot_traces(env, policies, patient, seed, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for pol in policies:
        _, log = run_episode(env, pol, seed, patient, keep_log=True)
        ax.plot(np.arange(len(log["bg"])) * 3 / 60.0, log["bg"], label=pol.name, lw=1.4)
    ax.axhspan(70, 180, color="green", alpha=0.08)
    ax.axhline(70, color="red", ls=":", lw=0.8)
    ax.set_xlabel("hours")
    ax.set_ylabel("true BG (mg/dL)")
    ax.set_title(f"{patient}, seed {seed} (identical meals/noise for all policies)")
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="*", default=[])
    ap.add_argument("--labels", nargs="*", default=[])
    ap.add_argument("--split", choices=list(SPLITS), default="test")
    ap.add_argument("--episodes-per-patient", type=int, default=2)
    ap.add_argument("--seed-base", type=int, default=20_000_000)
    ap.add_argument("--episode-days", type=float, default=None)
    ap.add_argument("--no-shield", action="store_true", help="override: evaluate with the safety shield OFF")
    ap.add_argument("--out", default="results")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    policies, cfg = list(make_baselines(seed=0)), dict(DEFAULT_CFG)
    if args.checkpoints:
        from dqn import DQNAgent, DQNPolicy      # torch imported only when needed
        labels = resolve_labels(args.checkpoints, args.labels, [p.name for p in policies])
        for i, (ck, lab) in enumerate(zip(args.checkpoints, labels)):
            agent, ck_cfg = DQNAgent.load(ck)
            if i == 0:
                cfg.update(ck_cfg)
            elif ck_cfg and any(ck_cfg.get(k, cfg[k]) != cfg[k] for k in ("history_len", "announce_meals", "action_repeat")):
                raise SystemExit(f"checkpoint {ck} has a different observation/decision-interval config than the first one")
            policies.append(DQNPolicy(agent, lab))
    if args.episode_days:
        cfg["episode_days"] = args.episode_days
    if args.no_shield:
        cfg["shield"] = False
    print("eval env config:", json.dumps(cfg))

    env = make_env(SPLITS[args.split], cfg)
    for pol in policies:
        agent = getattr(pol, "agent", None)
        if agent is not None and (agent.obs_dim != env.obs_dim or agent.n_actions != env.action_space.n):
            raise SystemExit(f"checkpoint '{pol.name}' expects obs_dim={agent.obs_dim}, n_actions={agent.n_actions}, but the "
                             f"evaluation environment has obs_dim={env.obs_dim}, n_actions={env.action_space.n}")
    rows = []
    for pol in policies:
        res = evaluate_policy(env, pol, SPLITS[args.split], args.episodes_per_patient, args.seed_base)
        for r in res:
            rows.append({"policy": pol.name, **r})
        print(f"  {pol.name:14s} "
    f"TIR {np.mean([r['tir_70_180'] for r in res]):5.1f}% | "
    f"<70 {np.mean([r['tbr_lt70'] for r in res]):5.1f}% | "
    f"<54 {np.mean([r['tbr_lt54'] for r in res]):5.1f}% | "
    f">180 {np.mean([r['tar_gt180'] for r in res]):5.1f}% | "
    f">250 {np.mean([r['tar_gt250'] for r in res]):5.1f}% | "
    f"terminated {np.mean([r['terminated_early'] for r in res]):.2f}")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out, "episodes.csv"), index=False)
    summary = summarize(df)
    summary.to_csv(os.path.join(args.out, "summary.csv"), index=False)
    with open(os.path.join(args.out, "summary.md"), "w") as f:
        f.write(md_table(summary, df, args.split))
    if args.plot:
        plot_traces(env, policies, SPLITS[args.split][0], args.seed_base, os.path.join(args.out, "traces.png"))
    print(f"wrote {args.out}/episodes.csv, summary.csv, summary.md" + (", traces.png" if args.plot else ""))


if __name__ == "__main__":
    main()