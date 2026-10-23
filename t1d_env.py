"""
t1d_env.py - Gymnasium POMDP environment around SimGlucose's CORE simulator.

RESEARCH SIMULATION ONLY. Nothing here is a real-world insulin dosing system or advice.

Uses only SimGlucose API that was read from source and run (see PROJECT_PROGRESS.md):
  T1DPatient.withName / CGMSensor.withName / InsulinPump.withName / RandomScenario /
  T1DSimEnv.reset() / T1DSimEnv.step(Action(basal, bolus)) -> Step(observation, reward, done, info)
  info keys used: 'bg' (true BG), 'meal'.  sim attrs used: BG_hist, CGM_hist, insulin_hist, CHO_hist, time.

DESIGN SUMMARY
  Decision interval : `action_repeat` sensor samples (default 5 x 3 min = 15 min). Action is held.
  Observation (float32, partial/noisy):
      [last K CGM readings / 400]  (noisy Dexcom CGM, one per decision interval)
      [estimated IOB / (0.1 * TDI)] (our own estimate, NOT the simulator's hidden insulin state)
      [sin, cos of time of day]
      [grams eaten in last interval / 100]   (ONLY if announce_meals=True; default False)
  Hidden from the agent: true BG, patient identity/parameters, meal schedule, simulator ODE state.
  Actions (Discrete): index = basal_idx * n_bolus + bolus_idx
      basal multiplier  in (0, 0.5, 1.0, 1.5) x the patient's nominal basal rate
      bolus (units)     in (0, 0.03, 0.06, 0.12) x the patient's total daily insulin (TDI)
  Safety layer ("shield", configurable): suspend basal+bolus if CGM < 80; block a bolus that
      would push estimated IOB above 0.20 x TDI. Hard caps on any commanded dose always apply.
  Reward (per decision interval, from TRUE BG, averaged over the interval's sensor samples):
      in range 70-180: +1; hyper: mild linear penalty; hypo: strong penalty, extra below 54;
      -terminal_penalty if the simulator ends the episode (BG<10 or >600).
  Episode ends: terminated (simulator rule) or truncated at episode_days.
"""
from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

import simglucose
from simglucose.actuator.pump import InsulinPump
from simglucose.controller.base import Action
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.sensor.cgm import CGMSensor
from simglucose.simulation.env import T1DSimEnv
from simglucose.simulation.scenario_gen import RandomScenario

SAMPLE_TIME_MIN = 3.0   # Dexcom sample_time in sensor_params.csv (asserted at reset)
CGM_SCALE = 400.0
MIN_PER_DAY = 1440.0

# ----------------------------------------------------------------------------------------
# Patient split (group-stratified). Train / validation (model selection) / held-out test.
# ----------------------------------------------------------------------------------------
GROUPS = ("adolescent", "adult", "child")


def _names(group, ids):
    return [f"{group}#{i:03d}" for i in ids]


TRAIN_PATIENTS = [n for g in GROUPS for n in _names(g, range(1, 7))]   # 18 patients
VAL_PATIENTS = [n for g in GROUPS for n in _names(g, (7, 8))]          # 6 patients
TEST_PATIENTS = [n for g in GROUPS for n in _names(g, (9, 10))]        # 6 held-out patients


@lru_cache(maxsize=1)
def _patient_table() -> pd.DataFrame:
    params_dir = os.path.join(os.path.dirname(simglucose.__file__), "params")
    vp = pd.read_csv(os.path.join(params_dir, "vpatient_params.csv"))
    quest = pd.read_csv(os.path.join(params_dir, "Quest.csv"))  # Name, CR, CF, Age, TDI
    table = quest.merge(vp[["Name", "BW", "u2ss"]], on="Name").set_index("Name")
    table["basal_U_per_min"] = table["u2ss"] * table["BW"] / 6000.0   # same formula as BBController
    return table


def patient_info(name: str) -> dict:
    row = _patient_table().loc[name]
    return {k: float(row[k]) for k in ("CR", "CF", "TDI", "BW", "Age", "basal_U_per_min")}


# ----------------------------------------------------------------------------------------
# Configs
# ----------------------------------------------------------------------------------------
@dataclass(frozen=True)
class RewardConfig:
    mode: str = "risk"            # "risk" (asymmetric, hypo-averse) or "tir" (in-range indicator only)
    low: float = 70.0
    high: float = 180.0
    severe_low: float = 54.0
    in_range: float = 1.0
    hyper_base: float = 0.5       # penalty just above `high`
    hyper_slope: float = 0.01     # extra penalty per mg/dL above `high`
    hyper_cap: float = 3.0
    hypo_base: float = 2.0        # penalty just below `low`
    hypo_slope: float = 0.1       # extra penalty per mg/dL below `low`
    hypo_cap: float = 10.0
    severe_extra: float = 5.0     # additional penalty below `severe_low`
    terminal_penalty: float = 50.0


@dataclass(frozen=True)
class ActionConfig:
    basal_mults: tuple = (0.0, 0.5, 1.0, 1.5)
    bolus_fracs_of_tdi: tuple = (0.0, 0.03, 0.06, 0.12)


@dataclass(frozen=True)
class ShieldConfig:
    enabled: bool = True
    suspend_below_cgm: float = 80.0
    iob_cap_frac_of_tdi: float = 0.20
    # Hard caps ALWAYS applied (even if the shield is disabled). The simulator's own pump limit is
    # 30 U/min (= 90 U per 3-min step), which is absurdly large, so we bound doses ourselves.
    hard_max_basal_mult: float = 3.0
    hard_max_bolus_frac_of_tdi: float = 0.25


def reward_bg(bg: float, cfg: RewardConfig) -> float:
    """Reward for one true-BG sample (mg/dL)."""
    in_rng = cfg.low <= bg <= cfg.high
    if cfg.mode == "tir":
        return cfg.in_range if in_rng else 0.0
    if cfg.mode != "risk":
        raise ValueError(f"unknown reward mode {cfg.mode!r}")
    if in_rng:
        return cfg.in_range
    if bg > cfg.high:
        return -min(cfg.hyper_base + cfg.hyper_slope * (bg - cfg.high), cfg.hyper_cap)
    pen = min(cfg.hypo_base + cfg.hypo_slope * (cfg.low - bg), cfg.hypo_cap)
    if bg < cfg.severe_low:
        pen += cfg.severe_extra
    return -pen


def apply_shield(basal_mult, bolus_u, cgm, iob_u, tdi, cfg: ShieldConfig):
    """Pure function. Returns (basal_mult, bolus_u, suspended, bolus_blocked). Uses CGM only."""
    if not cfg.enabled:
        return basal_mult, bolus_u, False, False
    if cgm < cfg.suspend_below_cgm:
        return 0.0, 0.0, True, bolus_u > 0
    if bolus_u > 0 and iob_u + bolus_u > cfg.iob_cap_frac_of_tdi * tdi:
        return basal_mult, 0.0, False, True
    return basal_mult, bolus_u, False, False


def decode_obs(obs: np.ndarray, history_len: int) -> dict:
    """Helper for rule-based baselines: unpack the observation vector."""
    return {
        "cgm": np.asarray(obs[:history_len], dtype=np.float64) * CGM_SCALE,
        "iob_norm": float(obs[history_len]),
    }


# ----------------------------------------------------------------------------------------
# Environment
# ----------------------------------------------------------------------------------------
class T1DRiskEnv(gym.Env):
    metadata = {"render_modes": []}
    SENSOR_HARDWARE = "Dexcom"
    PUMP_HARDWARE = "Insulet"

    def __init__(
        self,
        patients=None,
        episode_days: float = 1.0,
        action_repeat: int = 5,
        history_len: int = 8,
        announce_meals: bool = False,
        random_start_hour: bool = False,
        dia_min: float = 300.0,
        reward_cfg: RewardConfig = RewardConfig(),
        action_cfg: ActionConfig = ActionConfig(),
        shield_cfg: ShieldConfig = ShieldConfig(),
    ):
        super().__init__()
        self.patients = list(patients) if patients else list(TRAIN_PATIENTS)
        for p in self.patients:
            patient_info(p)  # fail early on bad names
        self.episode_days = episode_days
        self.action_repeat = int(action_repeat)
        self.history_len = int(history_len)
        self.announce_meals = bool(announce_meals)
        self.random_start_hour = bool(random_start_hour)
        self.dia_min = float(dia_min)
        self.rcfg, self.acfg, self.scfg = reward_cfg, action_cfg, shield_cfg

        self.n_basal = len(self.acfg.basal_mults)
        self.n_bolus = len(self.acfg.bolus_fracs_of_tdi)
        self.action_space = spaces.Discrete(self.n_basal * self.n_bolus)
        self.obs_dim = self.history_len + 1 + 2 + int(self.announce_meals)
        self.observation_space = spaces.Box(-5.0, 10.0, shape=(self.obs_dim,), dtype=np.float32)
        self.max_decisions = int(round(episode_days * MIN_PER_DAY / (SAMPLE_TIME_MIN * self.action_repeat)))

        self.sim = None
        self._over = True

    # ---------------- helpers ----------------
    def nearest_action(self, basal_mult: float, bolus_frac_of_tdi: float) -> int:
        ib = int(np.argmin(np.abs(np.asarray(self.acfg.basal_mults) - basal_mult)))
        ik = int(np.argmin(np.abs(np.asarray(self.acfg.bolus_fracs_of_tdi) - bolus_frac_of_tdi)))
        return ib * self.n_bolus + ik

    def decode_action(self, a: int):
        """index -> (basal multiplier, bolus units) for the CURRENT patient."""
        ib, ik = divmod(int(a), self.n_bolus)
        return self.acfg.basal_mults[ib], self.acfg.bolus_fracs_of_tdi[ik] * self.pinfo["TDI"]

    def _compute_iob(self) -> float:
        """Simplified IOB estimate: net insulin above nominal basal, linear decay over DIA.
        Net units delivered in sensor step j weigh (1 - age/DIA), age = minutes since that step.
        Negative values are possible (basal suspended). NOT the simulator's hidden insulin state."""
        st = SAMPLE_TIME_MIN
        delivered = np.asarray(self.sim.insulin_hist, dtype=np.float64) * st   # U per sensor step
        if delivered.size == 0:
            return 0.0
        net = delivered - self.pinfo["basal_U_per_min"] * st
        age = (delivered.size - 1 - np.arange(delivered.size)) * st
        return float(np.sum(net * np.clip(1.0 - age / self.dia_min, 0.0, 1.0)))

    def _obs(self) -> np.ndarray:
        cgm = np.asarray(self._cgm_hist, dtype=np.float32) / CGM_SCALE
        iob_n = np.clip(self.iob / (0.1 * self.pinfo["TDI"]), -3.0, 5.0)
        t = self.sim.time
        h = t.hour + t.minute / 60.0
        feats = [*cgm, iob_n, np.sin(2 * np.pi * h / 24.0), np.cos(2 * np.pi * h / 24.0)]
        if self.announce_meals:
            feats.append(self.last_meal_g / 100.0)
        return np.asarray(feats, dtype=np.float32)

    # ---------------- gymnasium API ----------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        # Fixed draw order so (seed, patient) fully determines the episode, however the patient is chosen.
        idx = int(self.np_random.integers(0, len(self.patients)))
        s_pat, s_sen, s_meal = (int(x) for x in self.np_random.integers(0, 2**31 - 1, size=3))
        hour = int(self.np_random.integers(0, 24)) if self.random_start_hour else 0
        requested = options.get("patient")
        if requested is not None and requested not in self.patients:
            # Leakage guard: an environment built for one split must not run patients from another split.
            raise ValueError(f"patient {requested!r} is not in this environment's patient list")
        self.patient_name = str(requested or self.patients[idx])
        self.pinfo = patient_info(self.patient_name)

        patient = T1DPatient.withName(self.patient_name, random_init_bg=True, seed=s_pat)
        sensor = CGMSensor.withName(self.SENSOR_HARDWARE, seed=s_sen)
        pump = InsulinPump.withName(self.PUMP_HARDWARE)
        scenario = RandomScenario(start_time=datetime(2018, 1, 1, hour, 0, 0), seed=s_meal)
        self.sim = T1DSimEnv(patient, sensor, pump, scenario)
        first = self.sim.reset()
        assert abs(self.sim.sample_time - SAMPLE_TIME_MIN) < 1e-9, "unexpected sensor sample time"

        self.cgm_last = float(first.observation.CGM)
        self._cgm_hist = deque([self.cgm_last] * self.history_len, maxlen=self.history_len)
        self.iob = 0.0
        self.last_meal_g = 0.0
        self._decisions = 0
        self._over = False
        self._terminated = False
        self.n_suspend = 0
        self.n_bolus_blocked = 0
        info = {"patient": self.patient_name, "bg": float(first.info["bg"]), "cgm": self.cgm_last,
                "iob": 0.0, "time": self.sim.time}
        return self._obs(), info

    def step(self, action):
        a = int(action)
        if not 0 <= a < self.action_space.n:
            raise ValueError(f"action {a} out of range")
        mult, units = self.decode_action(a)
        return self._advance(mult, units, a)

    def step_units(self, basal_mult: float, bolus_units: float):
        """Continuous-dose entry point (used only by the announced-meal reference baseline)."""
        return self._advance(basal_mult, bolus_units, -1)

    def _advance(self, basal_mult, bolus_u, action_idx):
        if self.sim is None or self._over:
            raise RuntimeError("call reset() before step(); the episode is over")
        sc, st = self.scfg, SAMPLE_TIME_MIN
        basal_mult = float(np.clip(basal_mult, 0.0, sc.hard_max_basal_mult))
        bolus_u = float(np.clip(bolus_u, 0.0, sc.hard_max_bolus_frac_of_tdi * self.pinfo["TDI"]))
        req_basal, req_bolus = basal_mult, bolus_u
        basal_mult, bolus_u, suspended, blocked = apply_shield(
            basal_mult, bolus_u, self.cgm_last, self.iob, self.pinfo["TDI"], sc)
        self.n_suspend += int(suspended)
        self.n_bolus_blocked += int(blocked)

        basal_rate = basal_mult * self.pinfo["basal_U_per_min"]        # U/min
        bgs, meal_g, terminated, step = [], 0.0, False, None
        for i in range(self.action_repeat):
            # SimGlucose treats basal and bolus as RATES (U/min) over the 3-minute step, so a bolus of
            # B units is commanded as B / 3 U/min. Bolus only in the first sensor step of the interval.
            bolus_rate = bolus_u / st if i == 0 else 0.0
            step = self.sim.step(Action(basal=basal_rate, bolus=bolus_rate))
            bgs.append(float(step.info["bg"]))
            meal_g += float(step.info["meal"]) * st                    # info['meal'] = grams / sample_time
            if step.done:
                terminated = True
                break

        reward = float(np.mean([reward_bg(b, self.rcfg) for b in bgs]))
        if terminated:
            reward -= self.rcfg.terminal_penalty
        self.cgm_last = float(step.observation.CGM)
        self._cgm_hist.append(self.cgm_last)
        self.last_meal_g = meal_g
        self.iob = self._compute_iob()
        self._decisions += 1
        truncated = (not terminated) and self._decisions >= self.max_decisions
        self._over = terminated or truncated
        self._terminated = terminated
        info = {
            "patient": self.patient_name, "time": self.sim.time, "bg": bgs[-1],
            "bg_mean": float(np.mean(bgs)), "cgm": self.cgm_last, "iob": self.iob,
            "meal_g": meal_g, "action": action_idx,
            "basal_mult": basal_mult, "bolus_units": bolus_u,
            "req_basal_mult": req_basal, "req_bolus_units": req_bolus,
            "shield_suspend": suspended, "shield_bolus_blocked": blocked,
            "decisions": self._decisions, "terminated_by_sim": terminated,
        }
        return self._obs(), reward, terminated, truncated, info

    def episode_log(self) -> dict:
        """Full-resolution (3-min) record of the episode just run, for metrics and plots."""
        s, st = self.sim, SAMPLE_TIME_MIN
        return {
            "patient": self.patient_name,
            "bg": np.asarray(s.BG_hist[1:], dtype=np.float64),          # true BG (index 0 = initial state)
            "cgm": np.asarray(s.CGM_hist[1:], dtype=np.float64),
            "insulin_u": np.asarray(s.insulin_hist, dtype=np.float64) * st,   # U delivered per sensor step
            "meal_g": np.asarray(s.CHO_hist, dtype=np.float64) * st,
            "terminated": bool(self._terminated),
            "shield_suspend_decisions": self.n_suspend,
            "shield_bolus_blocks": self.n_bolus_blocked,
        }