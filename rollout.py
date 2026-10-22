"""rollout.py - run policies in T1DRiskEnv and collect per-episode metrics (research simulation only)."""
from __future__ import annotations

from metrics import episode_metrics


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
    m = episode_metrics(log["bg"], log["insulin_u"], log["meal_g"], terminated=log["terminated"])
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
