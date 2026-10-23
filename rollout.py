"""rollout.py - run policies in T1DRiskEnv and collect per-episode metrics (research simulation only)."""
from __future__ import annotations

import numpy as np

from metrics import episode_metrics


def metrics_from_log(env, log):
    """Episode metrics from the full-resolution log.

    If the simulator ended the episode early (true BG < 10 or > 600), the remaining time of the episode is
    counted at the terminal BG (which is out of range), so a policy cannot look better by crashing early.
    `survival_fraction` = fraction of the episode that was actually simulated (1.0 if it did not terminate).
    Insulin and carbohydrate totals are NOT padded."""
    bg = log["bg"]
    n_total = env.max_decisions * env.action_repeat
    recorded = len(bg)
    if log["terminated"] and recorded < n_total:
        bg = np.concatenate([bg, np.full(n_total - recorded, bg[-1])])
    m = episode_metrics(bg, log["insulin_u"], log["meal_g"], terminated=log["terminated"])
    m["survival_fraction"] = recorded / n_total
    return m


def run_episode(env, policy, seed, patient=None, keep_log=False):
    """One episode. (seed, patient) fully determines meals, sensor noise and initial state,
    so different policies can be compared on IDENTICAL episodes."""
    obs, _ = env.reset(seed=seed, options={"patient": patient} if patient else None)
    policy.reset()
    total, done = 0.0, False
    while not done:
        a = policy.act(obs, env)
        if isinstance(a, tuple):
            obs, r, term, trunc, _ = env.step_units(*a)
        else:
            obs, r, term, trunc, _ = env.step(int(a))
        total += r
        done = term or trunc
    log = env.episode_log()
    m = metrics_from_log(env, log)
    m.update(
        episode_return=float(total),
        patient=log["patient"],
        seed=int(seed),
        shield_suspend_decisions=log["shield_suspend_decisions"],
        shield_bolus_blocks=log["shield_bolus_blocks"],
    )
    return (m, log) if keep_log else m


def evaluate_policy(env, policy, patients, episodes_per_patient, base_seed):
    """Same seeds for every policy: seed = base_seed + 1000 * patient_index + k."""
    rows = []
    for pi, patient in enumerate(patients):
        for k in range(episodes_per_patient):
            rows.append(run_episode(env, policy, base_seed + 1000 * pi + k, patient))
    return rows