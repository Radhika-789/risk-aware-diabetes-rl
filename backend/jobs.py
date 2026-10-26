"""Background jobs: paired full-episode rollouts and evaluate.py runs. Results are read back from real outputs."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import traceback
import uuid

import numpy as np

from rollout import metrics_from_log, run_episode
from t1d_env import SAMPLE_TIME_MIN

from .policies import checkpoint_path, is_checkpoint_id
from .session import CHECKPOINT_LOCKED, build_env, merge_settings, trace_from_sim
from .policies import load_policy


class Job:
    def __init__(self, kind: str, params: dict):
        self.id = uuid.uuid4().hex[:10]
        self.kind, self.params = kind, params
        self.status = "queued"
        self.progress = 0.0
        self.message = ""
        self.result = None
        self.error = None
        self.created = time.time()
        self.finished = None

    def to_dict(self, with_result=True):
        d = {"id": self.id, "kind": self.kind, "params": self.params, "status": self.status,
             "progress": self.progress, "message": self.message, "error": self.error,
             "created": self.created, "finished": self.finished}
        if with_result:
            d["result"] = self.result
        return d


JOBS: dict[str, Job] = {}
_lock = threading.Lock()


def _start(job: Job, target):
    JOBS[job.id] = job

    def wrap():
        job.status = "running"
        try:
            job.result = target(job)
            job.status = "done"
            job.progress = 1.0
        except Exception as e:  # noqa: BLE001
            job.status = "error"
            job.error = f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}"
        job.finished = time.time()

    threading.Thread(target=wrap, daemon=True).start()
    return job


# ---------------------------------------------------------------- paired rollouts (same patient/seed/meals/noise)
def paired_rollouts(policy_ids: list[str], settings: dict) -> Job:
    def target(job: Job):
        out = {"kind": "counterfactual_full_episode", "rollouts": [], "settings": None}
        base_cfg = None
        for k, pid in enumerate(policy_ids):
            job.message = f"rolling out {pid}"
            pol, ckpt_cfg = load_policy(pid, seed=int(settings.get("seed", 0)))
            s, locked = merge_settings(settings, ckpt_cfg)
            if base_cfg is None:
                base_cfg = s
                out["settings"] = s
                out["locked_fields"] = locked
            elif any(s[kk] != base_cfg[kk] for kk in CHECKPOINT_LOCKED):
                out["rollouts"].append({"policy": pid, "error": "observation config differs from first policy"})
                continue
            env = build_env(base_cfg)
            m, log = run_episode(env, pol, int(base_cfg["seed"]), base_cfg["patient"], keep_log=True)
            out["rollouts"].append({"policy": pid, "metrics": m, "trace": trace_from_sim(env),
                                    "shield": {"suspends": log["shield_suspend_decisions"],
                                               "bolus_blocks": log["shield_bolus_blocks"]}})
            job.progress = (k + 1) / len(policy_ids)
        return out

    return _start(Job("paired_rollouts", {"policies": policy_ids, "settings": settings}), target)


# ---------------------------------------------------------------- evaluate.py as a subprocess
def run_evaluation(checkpoint_ids: list[str], split: str, episodes_per_patient: int, out_name: str,
                   episode_days: float | None = None, no_shield: bool = False) -> Job:
    if split not in ("test", "val", "train"):
        raise ValueError("split must be test|val|train")
    if not out_name.startswith("results") or "/" in out_name or ".." in out_name:
        raise ValueError("out_name must look like results_<something>")

    def target(job: Job):
        cmd = [sys.executable, "-W", "ignore", "evaluate.py", "--split", split,
               "--episodes-per-patient", str(int(episodes_per_patient)), "--out", out_name]
        labels = []
        if checkpoint_ids:
            paths = [checkpoint_path(c) for c in checkpoint_ids if is_checkpoint_id(c)]
            labels = [c[len("ckpt:"):].replace("/", "_").replace(".pt", "") for c in checkpoint_ids]
            cmd += ["--checkpoints", *paths, "--labels", *labels]
        if episode_days:
            cmd += ["--episode-days", str(float(episode_days))]
        if no_shield:
            cmd += ["--no-shield"]
        job.message = " ".join(cmd)
        os.makedirs(out_name, exist_ok=True)
        with open(os.path.join(out_name, "meta.json"), "w") as f:
            json.dump({"split": split, "checkpoints": checkpoint_ids, "labels": labels,
                       "episodes_per_patient": episodes_per_patient, "episode_days": episode_days,
                       "shield": not no_shield, "started": time.time(), "command": cmd}, f, indent=2)
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        lines = []
        for line in proc.stdout:
            lines.append(line.rstrip())
            job.message = line.strip()[:200]
        rc = proc.wait()
        if rc != 0:
            raise RuntimeError("evaluate.py failed:\n" + "\n".join(lines[-20:]))
        with open(os.path.join(out_name, "meta.json")) as f:
            meta = json.load(f)
        meta["finished"] = time.time()
        with open(os.path.join(out_name, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        return {"out": out_name, "log": lines[-40:]}

    return _start(Job("evaluation", {"checkpoints": checkpoint_ids, "split": split,
                                     "episodes_per_patient": episodes_per_patient, "out": out_name}), target)
