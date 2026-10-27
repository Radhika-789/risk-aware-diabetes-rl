"""
graphs.py - the 10 report figures from one training run and one evaluation directory.

    python graphs.py --run runs/dqn_risk_s0_300 --eval runs/eval_300_test_both --out figures
    python graphs.py --run runs/dqn_risk_s0_300 --eval runs/eval_300_test_both --dqn-label dqn_risk_s0_300_best

Reads only files written by train_dqn.py (train_log.csv, val_log.csv) and evaluate.py (summary.csv, episodes.csv).
Writes figures/01_... to figures/10_...png plus figures/summary_table.md. Research simulation only.
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------- style
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight",
    "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.labelsize": 10,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
    "legend.frameon": False, "legend.fontsize": 9,
})
C = {"in": "#2a9d8f", "low": "#e9a23b", "sev_low": "#d7263d", "high": "#f28e2b", "very_high": "#a23b72",
     "line": "#1f4e79", "ma": "#e76f51", "val": "#264653", "best": "#e76f51", "grey": "#8d99ae"}
POLICY_ORDER = ["random", "fixed_basal", "cgm_rule", "announced_bb"]   # baselines first, DQN checkpoints after
NICE = {"random": "Random", "fixed_basal": "Fixed basal", "cgm_rule": "CGM rule", "announced_bb": "Announced BB\n(privileged)"}


def nice(p: str) -> str:
    if p in NICE:
        return NICE[p]
    # checkpoint labels like dqn_risk_s0_300_best -> "DQN\ndqn_risk_s0_300 best"; wrap long names on two lines
    parts = p.split("_")
    if len(parts) > 2:
        mid = len(parts) // 2
        return "_".join(parts[:mid]) + "\n" + "_".join(parts[mid:])
    return p


def moving_avg(x, w):
    x = pd.Series(x, dtype=float)
    return x.rolling(w, min_periods=max(1, w // 3)).mean()


def order_policies(policies, dqn_label):
    base = [p for p in POLICY_ORDER if p in policies]
    dqn = [p for p in policies if p not in POLICY_ORDER]
    if dqn_label and dqn_label in dqn:          # the headline checkpoint goes last so it is the right-most bar
        dqn = [p for p in dqn if p != dqn_label] + [dqn_label]
    return base + dqn


def bar_colors(policies, dqn_label):
    return [C["line"] if (p == dqn_label or (dqn_label is None and p not in POLICY_ORDER)) else C["grey"] for p in policies]


def save(fig, out, name):
    path = os.path.join(out, name)
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path)


# ---------------------------------------------------------------- 1-4: training
def fig_return(train, out, w):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(train.episode, train["return"], color=C["line"], lw=0.8, alpha=0.35, label="Episode return")
    ax.plot(train.episode, moving_avg(train["return"], w), color=C["ma"], lw=2.2, label=f"{w}-episode moving average")
    ax.axhline(0, color="k", lw=0.6, alpha=0.5)
    ax.set(xlabel="Training episode", ylabel="Episode return (sum of per-interval reward)",
           title="1. Training episode return")
    eps_end = train.loc[train.epsilon <= train.epsilon.min() + 1e-9, "episode"].min()
    if pd.notna(eps_end):
        ax.axvline(eps_end, color=C["grey"], ls="--", lw=1)
        ax.text(eps_end, ax.get_ylim()[1], f" ε reaches {train.epsilon.min():.2f}", va="top", fontsize=8, color=C["grey"])
    ax.legend(loc="lower right")
    fig.text(0.01, -0.03, "Each episode is a different training patient and meal schedule, hence the spread.", fontsize=8, color="#555")
    save(fig, out, "01_training_return.png")


def fig_train_tir(train, out, w):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(train.episode, moving_avg(train.tir, w), color=C["in"], lw=2.2, label="TIR 70–180 mg/dL")
    ax.plot(train.episode, moving_avg(train.tbr_lt70, w), color=C["low"], lw=2.0, label="TBR < 70 mg/dL")
    ax.plot(train.episode, moving_avg(train.tar_gt180, w), color=C["high"], lw=2.0, label="TAR > 180 mg/dL")
    ax.scatter(train.episode, train.tir, s=5, color=C["in"], alpha=0.2)
    ax.set(xlabel="Training episode", ylabel="% of episode time", ylim=(0, 100),
           title=f"2. Glycaemic metrics during training ({w}-episode moving average)")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.14))
    save(fig, out, "02_training_tir.png")


def fig_val(val, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    best = val[val.is_best == 1].iloc[-1] if (val.is_best == 1).any() else None
    ax = axes[0]
    ax.plot(val.episode, val.val_return, "-o", color=C["val"], ms=4, lw=1.8, label="Validation return")
    if best is not None:
        ax.scatter([best.episode], [best.val_return], s=120, color=C["best"], zorder=5, label=f"best.pt (ep {int(best.episode)})")
    ax.axhline(0, color="k", lw=0.6, alpha=0.5)
    ax.set(xlabel="Training episode", ylabel="Mean return on validation episodes", title="3a. Validation return (selection criterion)")
    ax.legend()
    ax = axes[1]
    ax.plot(val.episode, val.val_tir, "-o", color=C["in"], ms=4, lw=1.8, label="TIR 70–180")
    ax.plot(val.episode, val.val_tbr_lt70, "-s", color=C["low"], ms=4, lw=1.8, label="TBR < 70")
    ax.plot(val.episode, val.val_tbr_lt54, "-^", color=C["sev_low"], ms=4, lw=1.8, label="TBR < 54")
    if best is not None:
        ax.axvline(best.episode, color=C["best"], ls="--", lw=1.2)
    ax.set(xlabel="Training episode", ylabel="% of episode time", ylim=(0, 100), title="3b. Validation glycaemic metrics")
    ax.legend()
    fig.text(0.01, -0.03, "Fixed validation patients and seeds at every evaluation (identical episodes), so points are comparable.", fontsize=8, color="#555")
    save(fig, out, "03_validation.png")


def fig_loss(train, out, w):
    d = train.dropna(subset=["mean_loss"])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(d.episode, d.mean_loss, color=C["line"], lw=0.8, alpha=0.4, label="Per-episode mean Huber loss")
    ax.plot(d.episode, moving_avg(d.mean_loss, w), color=C["ma"], lw=2.2, label=f"{w}-episode moving average")
    ax.set(xlabel="Training episode", ylabel="Huber loss", title="4. DQN training loss")
    ax.legend()
    fig.text(0.01, -0.06, "DQN loss tracks the magnitude of Q-values, which grows as returns improve and the target network\n"
                          "moves; it is not expected to decrease monotonically. Checkpoint selection uses validation return.",
             fontsize=8, color="#555")
    save(fig, out, "04_training_loss.png")


# ---------------------------------------------------------------- 5-10: evaluation
def bar_metric(summary, episodes, key, title, ylabel, out, name, dqn_label, color, better="higher", ref=None):
    pols = order_policies(list(summary.policy), dqn_label)
    s = summary.set_index("policy").loc[pols]
    mean, lo, hi = s[f"{key}_mean"].values, s[f"{key}_ci_lo"].values, s[f"{key}_ci_hi"].values
    fig, ax = plt.subplots(figsize=(max(8, 1.4 * len(pols)), 4.2))
    x = np.arange(len(pols))
    cols = bar_colors(pols, dqn_label)
    cols = [color if c == C["line"] else c for c in cols]
    ax.bar(x, mean, color=cols, width=0.62, zorder=2)
    ax.errorbar(x, mean, yerr=[mean - lo, hi - mean], fmt="none", ecolor="k", elinewidth=1, capsize=4, zorder=3)
    # per-episode points
    for i, p in enumerate(pols):
        v = episodes.loc[episodes.policy == p, key].values
        ax.scatter(np.full(v.size, x[i]) + np.random.default_rng(0).uniform(-0.18, 0.18, v.size), v,
                   s=10, color="k", alpha=0.35, zorder=4)
    for i, m in enumerate(mean):
        ax.text(x[i], hi[i] + 0.8, f"{m:.1f}", ha="center", va="bottom", fontsize=8.5)
    if ref is not None:
        ax.axhline(ref[0], color=C["grey"], ls="--", lw=1)
        ax.text(len(pols) - 0.5, ref[0], f" {ref[1]}", va="bottom", ha="right", fontsize=8, color="#555")
    ax.set_xticks(x, [nice(p) for p in pols], fontsize=8.5)
    ax.set(ylabel=ylabel, title=title)
    ax.set_ylim(0, max(hi.max() * 1.18, 5))
    ax.grid(axis="x", visible=False)
    n = int(s.n_episodes.iloc[0])
    fig.text(0.01, -0.03, f"Bars: mean over {n} held-out test episodes per policy; whiskers: 95% bootstrap CI; dots: individual episodes. "
                          f"{'Higher' if better == 'higher' else 'Lower'} is better.", fontsize=8, color="#555")
    save(fig, out, name)


def fig_distribution(summary, out, dqn_label):
    pols = order_policies(list(summary.policy), dqn_label)
    s = summary.set_index("policy").loc[pols]
    sev_low, low, inr = s.tbr_lt54_mean, s.tbr_lt70_mean - s.tbr_lt54_mean, s.tir_70_180_mean
    high, very_high = s.tar_gt180_mean - s.tar_gt250_mean, s.tar_gt250_mean
    fig, ax = plt.subplots(figsize=(9, 4.4))
    y = np.arange(len(pols))
    left = np.zeros(len(pols))
    for vals, col, lab in [(sev_low, C["sev_low"], "< 54 (severe low)"), (low, C["low"], "54–70 (low)"),
                           (inr, C["in"], "70–180 (in range)"), (high, C["high"], "180–250 (high)"),
                           (very_high, C["very_high"], "> 250 (very high)")]:
        ax.barh(y, vals.values, left=left, color=col, height=0.62, label=lab, zorder=2)
        for i, v in enumerate(vals.values):
            if v >= 6:
                ax.text(left[i] + v / 2, y[i], f"{v:.0f}%", ha="center", va="center", fontsize=8,
                        color="white", fontweight="bold")
        left += vals.values
    ax.set_yticks(y, [nice(p).replace("\n", " ") for p in pols])
    ax.invert_yaxis()
    ax.set(xlim=(0, 100), xlabel="% of episode time (mean over held-out test episodes)",
           title="10. Glucose range distribution per policy")
    ax.grid(axis="y", visible=False)
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    save(fig, out, "10_range_distribution.png")


def summary_table(summary, out, dqn_label):
    pols = order_policies(list(summary.policy), dqn_label)
    s = summary.set_index("policy").loc[pols]
    rows = ["| Policy | n | TIR 70–180 % | TBR <70 % | TBR <54 % | TAR >180 % | TAR >250 % | Mean BG | LBGI | HBGI | Terminated |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]
    for p, r in s.iterrows():
        c = lambda k, d=1: f"{r[k + '_mean']:.{d}f} [{r[k + '_ci_lo']:.{d}f}, {r[k + '_ci_hi']:.{d}f}]"
        rows.append(f"| {p} | {int(r.n_episodes)} | {c('tir_70_180')} | {c('tbr_lt70')} | {c('tbr_lt54')} | {c('tar_gt180')} | "
                    f"{c('tar_gt250')} | {c('mean_bg', 0)} | {c('lbgi', 2)} | {c('hbgi', 2)} | {r.terminated_early_mean:.2f} |")
    rows.append("\nmean [95% bootstrap CI over episodes]. Research simulation only; simulated results do not imply clinical performance.")
    with open(os.path.join(out, "summary_table.md"), "w") as f:
        f.write("\n".join(rows))
    print("wrote", os.path.join(out, "summary_table.md"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="runs/<name> containing train_log.csv and val_log.csv")
    ap.add_argument("--eval", required=True, help="directory with evaluate.py output (summary.csv, episodes.csv)")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--window", type=int, default=15, help="moving-average window in episodes")
    ap.add_argument("--dqn-label", default=None, help="policy label of the headline DQN checkpoint in summary.csv")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    train = pd.read_csv(os.path.join(args.run, "train_log.csv"))
    val = pd.read_csv(os.path.join(args.run, "val_log.csv"))
    summary = pd.read_csv(os.path.join(args.eval, "summary.csv"))
    episodes = pd.read_csv(os.path.join(args.eval, "episodes.csv"))
    dqn_label = args.dqn_label
    if dqn_label is None:
        dqn = [p for p in summary.policy if p not in POLICY_ORDER]
        bests = [p for p in dqn if "best" in p]
        dqn_label = bests[-1] if bests else (dqn[-1] if dqn else None)
    print("headline DQN policy:", dqn_label)

    fig_return(train, args.out, args.window)
    fig_train_tir(train, args.out, args.window)
    fig_val(val, args.out)
    fig_loss(train, args.out, args.window)
    bar_metric(summary, episodes, "tir_70_180", "5. Time in range 70–180 mg/dL", "TIR (%)", args.out,
               "05_tir_comparison.png", dqn_label, C["in"], "higher", ref=(70, "consensus target ≥ 70%"))
    bar_metric(summary, episodes, "tar_gt180", "6. Time above range > 180 mg/dL", "TAR > 180 (%)", args.out,
               "06_tar_comparison.png", dqn_label, C["high"], "lower", ref=(25, "consensus target < 25%"))
    bar_metric(summary, episodes, "tbr_lt70", "7. Time below range < 70 mg/dL", "TBR < 70 (%)", args.out,
               "07_tbr_comparison.png", dqn_label, C["low"], "lower", ref=(4, "consensus target < 4%"))
    bar_metric(summary, episodes, "tbr_lt54", "8. Severe hypoglycaemia: time < 54 mg/dL", "TBR < 54 (%)", args.out,
               "08_severe_hypo_comparison.png", dqn_label, C["sev_low"], "lower", ref=(1, "consensus target < 1%"))
    bar_metric(summary, episodes, "tar_gt250", "9. Severe hyperglycaemia: time > 250 mg/dL", "TAR > 250 (%)", args.out,
               "09_severe_hyper_comparison.png", dqn_label, C["very_high"], "lower", ref=(5, "consensus target < 5%"))
    fig_distribution(summary, args.out, dqn_label)
    summary_table(summary, args.out, dqn_label)


if __name__ == "__main__":
    main()