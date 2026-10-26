"""SimSession - one live T1DRiskEnv episode driven step by step by a policy.

The simulator advances ONLY inside step(); nothing is extrapolated. A background runner thread calls step()
at a bounded rate while status == "running". Branch rollouts deep-copy the environment at the current state and
run alternative policies forward: those are counterfactual futures, never the live trace.

RESEARCH SIMULATION ONLY.
"""
from __future__ import annotations

import copy
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any

import numpy as np

from metrics import episode_metrics
from rollout import metrics_from_log
from t1d_env import (SAMPLE_TIME_MIN, ActionConfig, RewardConfig, ShieldConfig, T1DRiskEnv, TEST_PATIENTS,
                     TRAIN_PATIENTS, VAL_PATIENTS, patient_info)

from .policies import load_policy

SPLIT_OF = {**{p: "train" for p in TRAIN_PATIENTS}, **{p: "val" for p in VAL_PATIENTS},
            **{p: "test" for p in TEST_PATIENTS}}

# Fields a checkpoint fixes (they define obs_dim / decision interval). Everything else is editable.
CHECKPOINT_LOCKED = ("history_len", "action_repeat", "announce_meals", "obs_v2", "actions_v2", "mask_actions")

DEFAULT_SETTINGS: dict[str, Any] = dict(
    patient="adult#009", seed=1, episode_days=1.0, action_repeat=5, history_len=8, announce_meals=False,
    obs_v2=False, actions_v2=False, mask_actions=False,
    reward_mode="risk", shield=True, suspend_below_cgm=80.0, iob_cap_frac_of_tdi=0.20,
)


def build_env(s: dict) -> T1DRiskEnv:
    return T1DRiskEnv(
        patients=[s["patient"]], episode_days=float(s["episode_days"]), action_repeat=int(s["action_repeat"]),
        history_len=int(s["history_len"]), announce_meals=bool(s["announce_meals"]), obs_v2=bool(s["obs_v2"]),
        reward_cfg=RewardConfig(mode=s["reward_mode"]),
        action_cfg=ActionConfig.v2() if s["actions_v2"] else ActionConfig(),
        shield_cfg=ShieldConfig(enabled=bool(s["shield"]), suspend_below_cgm=float(s["suspend_below_cgm"]),
                                iob_cap_frac_of_tdi=float(s["iob_cap_frac_of_tdi"])),
    )


def merge_settings(user: dict, ckpt_cfg: dict | None) -> tuple[dict, list[str]]:
    """Apply checkpoint-locked fields on top of the user's settings. Returns (settings, locked_field_names)."""
    s = {**DEFAULT_SETTINGS, **{k: v for k, v in (user or {}).items() if k in DEFAULT_SETTINGS}}
    locked = []
    if ckpt_cfg:
        for k in CHECKPOINT_LOCKED:
            if k in ckpt_cfg:
                s[k] = ckpt_cfg[k]
                locked.append(k)
    return s, locked


def trace_from_sim(env: T1DRiskEnv, start_sample: int = 0) -> dict:
    """3-minute resolution record straight from the simulator histories. Index 0 = initial state."""
    sim = env.sim
    bg = np.asarray(sim.BG_hist, dtype=float)
    cgm = np.asarray(sim.CGM_hist, dtype=float)
    ins = np.concatenate([[0.0], np.asarray(sim.insulin_hist, dtype=float) * SAMPLE_TIME_MIN])
    cho = np.concatenate([[0.0], np.asarray(sim.CHO_hist, dtype=float) * SAMPLE_TIME_MIN])
    n = len(bg)
    t_min = np.arange(n) * SAMPLE_TIME_MIN
    sl = slice(start_sample, n)
    return {
        "sample_min": float(SAMPLE_TIME_MIN),
        "t_min": t_min[sl].round(1).tolist(),
        "cgm": cgm[sl].round(2).tolist(),
        "bg_sim_only": bg[sl].round(2).tolist(),
        "insulin_u": ins[sl].round(4).tolist(),
        "meal_g": cho[sl].round(2).tolist(),
    }


def partial_metrics(env: T1DRiskEnv) -> dict:
    bg = np.asarray(env.sim.BG_hist[1:], dtype=float)
    if bg.size == 0:
        return {}
    m = episode_metrics(bg, np.asarray(env.sim.insulin_hist) * SAMPLE_TIME_MIN,
                        np.asarray(env.sim.CHO_hist) * SAMPLE_TIME_MIN, terminated=env._terminated)
    m["scope"] = "episode_so_far"
    return m


def _policy_step(env: T1DRiskEnv, policy, obs):
    a = policy.act(obs, env)
    if isinstance(a, tuple):
        return env.step_units(*a), -1
    return env.step(int(a)), int(a)


@dataclass
class Decision:
    i: int
    t_min: float
    sim_time: str
    cgm: float
    bg_sim_only: float
    iob_u: float
    action: int
    req_basal_mult: float
    req_bolus_u: float
    basal_mult: float
    bolus_u: float
    shield_suspend: bool
    shield_bolus_blocked: bool
    reward: float
    meal_g: float
    q_values: list[float] | None = None
    wall_ms: float = 0.0


class SimSession:
    def __init__(self, policy_id: str, settings: dict | None):
        self.id = uuid.uuid4().hex[:10]
        self.lock = threading.RLock()
        self.policy_id = policy_id
        self.policy, ckpt_cfg = load_policy(policy_id, seed=int((settings or {}).get("seed", 0)))
        self.checkpoint_env_config = ckpt_cfg
        self.settings, self.locked_fields = merge_settings(settings or {}, ckpt_cfg)
        self.env = build_env(self.settings)
        self.created_at = time.time()
        self.version = 0
        self.error: str | None = None
        self.status = "idle"
        self.decisions: list[Decision] = []
        self.events: list[dict] = []
        self._runner: threading.Thread | None = None
        self._run_flag = threading.Event()
        self.speed = 2.0  # decisions per second while running
        self.reset(seed=self.settings["seed"])

    # ------------------------------------------------------------------ lifecycle
    def reset(self, seed: int | None = None):
        with self.lock:
            self.pause(wait=False)
            if seed is not None:
                self.settings["seed"] = int(seed)
            self.obs, info = self.env.reset(seed=int(self.settings["seed"]),
                                            options={"patient": self.settings["patient"]})
            self.policy.reset()
            self.start_time = info["time"]
            self.decisions = []
            self.events = []
            self.status = "idle"
            self.error = None
            self.version += 1

    @property
    def over(self) -> bool:
        return bool(self.env._over)

    def step(self, *, only_if_running: bool = False) -> Decision | None:
        """Advance exactly one decision interval (action_repeat x 3 min) of real SimGlucose steps."""
        with self.lock:
            if only_if_running and not self._run_flag.is_set():
                return None
            if self.over:
                self._finish()
                return None
            t0 = time.perf_counter()
            qv = None
            agent = getattr(self.policy, "agent", None)
            if agent is not None:
                qv = [round(float(x), 4) for x in agent.q_values(self.obs)]
            try:
                (obs2, r, term, trunc, info), a = _policy_step(self.env, self.policy, self.obs)
            except Exception as e:  # noqa: BLE001
                self.status, self.error = "error", f"{type(e).__name__}: {e}"
                self._run_flag.clear()
                self.version += 1
                raise
            self.obs = obs2
            t = info["time"]
            d = Decision(
                i=len(self.decisions) + 1, t_min=round(info["decisions"] * self.env.action_repeat * SAMPLE_TIME_MIN, 1),
                sim_time=t.isoformat(), cgm=round(float(info["cgm"]), 2), bg_sim_only=round(float(info["bg"]), 2),
                iob_u=round(float(info["iob"]), 4), action=a,
                req_basal_mult=float(info["req_basal_mult"]), req_bolus_u=round(float(info["req_bolus_units"]), 4),
                basal_mult=float(info["basal_mult"]), bolus_u=round(float(info["bolus_units"]), 4),
                shield_suspend=bool(info["shield_suspend"]), shield_bolus_blocked=bool(info["shield_bolus_blocked"]),
                reward=round(float(r), 4), meal_g=round(float(info["meal_g"]), 2), q_values=qv,
                wall_ms=round((time.perf_counter() - t0) * 1000, 1),
            )
            self.decisions.append(d)
            if d.shield_suspend:
                self.events.append({"i": d.i, "t_min": d.t_min, "kind": "suspend",
                                    "detail": f"CGM {d.cgm:.0f} < {self.settings['suspend_below_cgm']:.0f}: basal and bolus suspended"})
            if d.shield_bolus_blocked:
                self.events.append({"i": d.i, "t_min": d.t_min, "kind": "bolus_blocked",
                                    "detail": f"requested bolus {d.req_bolus_u:.2f} U blocked (IOB cap)"})
            if d.meal_g > 0:
                self.events.append({"i": d.i, "t_min": d.t_min, "kind": "meal", "detail": f"{d.meal_g:.0f} g CHO"})
            if term or trunc:
                self._finish()
            elif self.status == "idle":
                self.status = "paused"
            self.version += 1
            return d

    def _finish(self):
        self._run_flag.clear()
        self.status = "terminated" if self.env._terminated else "completed"
        if self.status == "terminated":
            self.events.append({"i": len(self.decisions), "t_min": self.decisions[-1].t_min if self.decisions else 0,
                                "kind": "terminated", "detail": "simulator ended the episode (true BG out of bounds)"})

    def run(self, speed: float | None = None):
        with self.lock:
            if self.over or self.status == "error":
                return
            if speed:
                self.speed = max(0.1, min(float(speed), 50.0))
            if self._run_flag.is_set():
                return
            self._run_flag.set()
            self.status = "running"
            self.version += 1
            self._runner = threading.Thread(target=self._run_loop, daemon=True)
            self._runner.start()

    def _run_loop(self):
        while self._run_flag.is_set():
            t0 = time.perf_counter()
            try:
                d = self.step(only_if_running=True)
            except Exception:  # noqa: BLE001
                return
            if d is None or self.over:
                return
            delay = 1.0 / self.speed - (time.perf_counter() - t0)
            if delay > 0:
                time.sleep(delay)
        with self.lock:
            if self.status == "running":
                self.status = "paused"
                self.version += 1

    def pause(self, wait: bool = True):
        self._run_flag.clear()
        if wait and self._runner is not None and self._runner.is_alive() and threading.current_thread() is not self._runner:
            self._runner.join(timeout=5)
        with self.lock:
            if self.status == "running":
                self.status = "paused"
                self.version += 1

    # ------------------------------------------------------------------ branch rollouts (counterfactual futures)
    def branch(self, policy_ids: list[str]) -> dict:
        """Deep-copy the live environment at its current state and run each policy to the end of the episode.
        Returns per-policy future traces starting at the branch point. Nothing here touches the live session."""
        with self.lock:
            if self.env.sim is None:
                raise RuntimeError("session has no episode")
            branch_sample = len(self.env.sim.BG_hist) - 1
            branch_decision = len(self.decisions)
            snapshots = {pid: copy.deepcopy(self.env) for pid in policy_ids}
            obs0 = np.array(self.obs, copy=True)
        out = {"kind": "counterfactual_branch", "branch_decision": branch_decision,
               "branch_t_min": round(branch_sample * SAMPLE_TIME_MIN, 1), "rollouts": []}
        for pid in policy_ids:
            env = snapshots[pid]
            pol, ckpt_cfg = load_policy(pid, seed=int(self.settings["seed"]))
            if ckpt_cfg and any(ckpt_cfg.get(k, self.settings[k]) != self.settings[k] for k in CHECKPOINT_LOCKED):
                out["rollouts"].append({"policy": pid, "error": "checkpoint observation config differs from the live session"})
                continue
            obs, total, done = obs0, 0.0, env._over
            if hasattr(pol, "reset"):
                pol.reset()
            decs = []
            while not done:
                (obs, r, term, trunc, info), a = _policy_step(env, pol, obs)
                total += r
                done = term or trunc
                decs.append({"t_min": round(info["decisions"] * env.action_repeat * SAMPLE_TIME_MIN, 1), "action": a,
                             "basal_mult": info["basal_mult"], "bolus_u": round(float(info["bolus_units"]), 4),
                             "shield_suspend": bool(info["shield_suspend"]),
                             "shield_bolus_blocked": bool(info["shield_bolus_blocked"])})
            log = env.episode_log()
            full = metrics_from_log(env, log)
            seg_bg = np.asarray(env.sim.BG_hist[branch_sample + 1:], dtype=float)
            seg = episode_metrics(seg_bg, terminated=log["terminated"]) if seg_bg.size else {}
            out["rollouts"].append({
                "policy": pid, "future_return": round(total, 3), "decisions": decs,
                "trace": trace_from_sim(env, start_sample=branch_sample),
                "segment_metrics": seg, "full_episode_metrics": full,
            })
        return out

    # ------------------------------------------------------------------ serialization
    def state(self, include_trace: bool = True) -> dict:
        with self.lock:
            env = self.env
            last = self.decisions[-1] if self.decisions else None
            pinfo = patient_info(self.settings["patient"])
            d = {
                "id": self.id, "status": self.status, "error": self.error, "version": self.version,
                "policy": self.policy_id, "settings": self.settings, "locked_fields": self.locked_fields,
                "patient_split": SPLIT_OF.get(self.settings["patient"], "unknown"),
                "patient_info": pinfo,
                "speed": self.speed,
                "clock": {
                    "start": self.start_time.isoformat(), "now": env.sim.time.isoformat(),
                    "elapsed_min": round(len(env.sim.BG_hist[1:]) * SAMPLE_TIME_MIN, 1),
                    "total_min": round(env.max_decisions * env.action_repeat * SAMPLE_TIME_MIN, 1),
                    "decision": len(self.decisions), "max_decisions": env.max_decisions,
                    "decision_min": env.action_repeat * SAMPLE_TIME_MIN,
                },
                "current": {
                    "cgm": round(float(env.cgm_last), 2),
                    "bg_sim_only": round(float(env.sim.BG_hist[-1]), 2),
                    "iob_u": round(float(env.iob), 4),
                    "iob_cap_u": round(float(self.settings["iob_cap_frac_of_tdi"] * pinfo["TDI"]), 3),
                },
                "last_decision": asdict(last) if last else None,
                "shield_counts": {"suspends": env.n_suspend, "bolus_blocks": env.n_bolus_blocked},
                "events": self.events[-200:],
                "metrics_so_far": partial_metrics(env),
                "action_space": {"n": int(env.action_space.n), "basal_mults": list(env.acfg.basal_mults),
                                 "bolus_fracs_of_tdi": list(env.acfg.bolus_fracs_of_tdi),
                                 "tdi_u": pinfo["TDI"], "basal_u_per_min": pinfo["basal_U_per_min"]},
                "decisions": [asdict(x) for x in self.decisions],
            }
            if include_trace:
                d["trace"] = trace_from_sim(env)
            return d
