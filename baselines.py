"""
baselines.py - non-learning policies for comparison with the DQN (research simulation only).

All policies act through the SAME wrapper (same dose caps and shield) as the DQN.
A policy has .name, .reset(), and .act(obs, env) which returns either a discrete action index
or a tuple (basal_mult, bolus_units) for the continuous-dose reference baseline.

  random          : uniformly random discrete action (sanity floor)
  fixed_basal     : nominal basal, never boluses (shows what happens when meals are not covered)
  cgm_rule        : simple hand-written CGM-only rule over the same discrete actions (UNTUNED heuristic,
                    NOT a clinical algorithm); sees only the observation vector
  announced_bb    : REFERENCE using privileged information: meal grams announced after the interval,
                    patient carb ratio / correction factor (Quest.csv), BBController-style formula.
                    Not a fair POMDP competitor; it shows what meal knowledge buys.
"""
from __future__ import annotations

import numpy as np

from t1d_env import decode_obs


class Policy:
    name = "policy"

    def reset(self):
        pass

    def act(self, obs, env):
        raise NotImplementedError


class RandomPolicy(Policy):
    name = "random"

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def act(self, obs, env):
        return int(self.rng.integers(env.action_space.n))


class FixedBasalPolicy(Policy):
    name = "fixed_basal"

    def act(self, obs, env):
        return env.nearest_action(1.0, 0.0)


class CGMRulePolicy(Policy):
    name = "cgm_rule"

    def act(self, obs, env):
        d = decode_obs(obs, env.history_len)
        cgm, trend, iob = d["cgm"][-1], d["cgm"][-1] - d["cgm"][-2], d["iob_norm"]   # trend: per interval
        if cgm < 100 or (cgm < 130 and trend < -10):
            return env.nearest_action(0.0, 0.0)            # suspend basal, no bolus
        if cgm > 250 and iob < 1.0 and trend > -10:
            return env.nearest_action(1.0, 0.06)
        if cgm > 180 and iob < 0.8 and trend > 0:
            return env.nearest_action(1.0, 0.03)
        return env.nearest_action(1.0, 0.0)


class AnnouncedMealBBPolicy(Policy):
    name = "announced_bb"

    def __init__(self, target=140.0):
        self.target = target

    def act(self, obs, env):
        p, meal = env.pinfo, env.last_meal_g
        bolus = 0.0
        if meal > 0:
            bolus = meal / p["CR"]
            if env.cgm_last > 150:
                bolus += (env.cgm_last - self.target) / p["CF"]
        return (1.0, max(bolus, 0.0))


def make_baselines(seed=0):
    return [RandomPolicy(seed), FixedBasalPolicy(), CGMRulePolicy(), AnnouncedMealBBPolicy()]
