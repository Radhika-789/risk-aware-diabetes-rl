"""GlucoGuard research API (FastAPI). Run from the repo root:

    uvicorn backend.app:app --reload --port 8000

RESEARCH SIMULATION ONLY. Not a medical device. Nothing here is insulin dosing advice.
"""
from __future__ import annotations

import os
import threading
import time
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from t1d_env import (ActionConfig, RewardConfig, ShieldConfig, TEST_PATIENTS, TRAIN_PATIENTS, VAL_PATIENTS,
                     patient_info)

from . import jobs, registry
from .policies import BASELINE_IDS
from .session import CHECKPOINT_LOCKED, DEFAULT_SETTINGS, SimSession

DISCLAIMER = ("Research simulation only (SimGlucose virtual patients). Not a medical device, not clinically "
              "validated, and never insulin dosing advice.")

app = FastAPI(title="GlucoGuard research API", version="0.1", description=DISCLAIMER)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

SESSIONS: dict[str, SimSession] = {}
_sessions_lock = threading.Lock()
MAX_SESSIONS = 8


def _get(sid: str) -> SimSession:
    s = SESSIONS.get(sid)
    if s is None:
        raise HTTPException(404, f"session {sid} not found")
    return s


# ------------------------------------------------------------------ catalogue
@app.get("/api/meta")
def meta():
    rc, ac, sc = RewardConfig(), ActionConfig(), ShieldConfig()
    return {
        "disclaimer": DISCLAIMER,
        "patients": {"train": TRAIN_PATIENTS, "val": VAL_PATIENTS, "test": TEST_PATIENTS},
        "baselines": list(BASELINE_IDS),
        "default_settings": DEFAULT_SETTINGS,
        "checkpoint_locked_fields": list(CHECKPOINT_LOCKED),
        "action_space": {"basal_mults": list(ac.basal_mults), "bolus_fracs_of_tdi": list(ac.bolus_fracs_of_tdi)},
        "reward": rc.__dict__,
        "shield_defaults": sc.__dict__,
        "thresholds": {"severe_low": 54, "low": 70, "high": 180, "very_high": 250},
        "sample_min": 3.0,
    }


@app.get("/api/patient")
def patient(name: str):
    try:
        info = patient_info(name)
    except KeyError:
        raise HTTPException(404, "unknown patient")
    split = "train" if name in TRAIN_PATIENTS else "val" if name in VAL_PATIENTS else "test" if name in TEST_PATIENTS else "unknown"
    return {"name": name, "split": split, **info}


@app.get("/api/runs")
def runs():
    return {"runs_dir": registry.RUNS_DIR, "runs": registry.list_runs()}


@app.get("/api/runs/{name}/logs")
def run_logs(name: str):
    try:
        return registry.run_logs(name)
    except FileNotFoundError:
        raise HTTPException(404, "run not found")


@app.get("/api/results")
def results():
    return {"results": registry.list_results()}


# ------------------------------------------------------------------ live sessions
class CreateSession(BaseModel):
    policy: str = Field(default="cgm_rule")
    settings: dict[str, Any] = Field(default_factory=dict)


@app.get("/api/sessions")
def list_sessions():
    return [s.state(include_trace=False) | {"decisions": None} for s in SESSIONS.values()]


@app.post("/api/sessions")
def create_session(body: CreateSession):
    with _sessions_lock:
        if len(SESSIONS) >= MAX_SESSIONS:
            oldest = min(SESSIONS.values(), key=lambda s: s.created_at)
            oldest.pause()
            del SESSIONS[oldest.id]
        try:
            s = SimSession(body.policy, body.settings)
        except (ValueError, FileNotFoundError, KeyError) as e:
            raise HTTPException(400, str(e))
        SESSIONS[s.id] = s
    return s.state()


@app.get("/api/sessions/{sid}")
def get_session(sid: str, trace: bool = True):
    return _get(sid).state(include_trace=trace)


@app.post("/api/sessions/{sid}/step")
def step(sid: str, n: int = 1):
    s = _get(sid)
    if s.status == "running":
        raise HTTPException(409, "pause the session before manual stepping")
    try:
        for _ in range(max(1, min(n, 50))):
            if s.step() is None:
                break
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"simulator step failed: {e}")
    return s.state()


class RunBody(BaseModel):
    speed: float | None = None


@app.post("/api/sessions/{sid}/run")
def run(sid: str, body: RunBody | None = None):
    s = _get(sid)
    s.run(body.speed if body else None)
    return s.state(include_trace=False) | {"decisions": None}


@app.post("/api/sessions/{sid}/pause")
def pause(sid: str):
    s = _get(sid)
    s.pause()
    return s.state(include_trace=False) | {"decisions": None}


class ResetBody(BaseModel):
    seed: int | None = None


@app.post("/api/sessions/{sid}/reset")
def reset(sid: str, body: ResetBody | None = None):
    s = _get(sid)
    s.reset(body.seed if body else None)
    return s.state()


@app.delete("/api/sessions/{sid}")
def delete(sid: str):
    s = _get(sid)
    s.pause()
    del SESSIONS[sid]
    return {"ok": True}


class BranchBody(BaseModel):
    policies: list[str]


@app.post("/api/sessions/{sid}/branch")
def branch(sid: str, body: BranchBody):
    s = _get(sid)
    if s.status == "running":
        raise HTTPException(409, "pause the session before branching")
    if not body.policies or len(body.policies) > 4:
        raise HTTPException(400, "1-4 policies")
    try:
        return s.branch(body.policies)
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(400, str(e))


# ------------------------------------------------------------------ jobs
class RolloutBody(BaseModel):
    policies: list[str]
    settings: dict[str, Any] = Field(default_factory=dict)


@app.post("/api/rollouts")
def rollouts(body: RolloutBody):
    if not body.policies or len(body.policies) > 4:
        raise HTTPException(400, "1-4 policies")
    return jobs.paired_rollouts(body.policies, body.settings).to_dict(with_result=False)


class EvalBody(BaseModel):
    checkpoints: list[str] = Field(default_factory=list)
    split: str = "test"
    episodes_per_patient: int = 1
    out: str = "results_ui"
    episode_days: float | None = None
    no_shield: bool = False


@app.post("/api/evaluations")
def evaluations(body: EvalBody):
    try:
        j = jobs.run_evaluation(body.checkpoints, body.split, body.episodes_per_patient, body.out,
                                body.episode_days, body.no_shield)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return j.to_dict(with_result=False)


@app.get("/api/jobs")
def list_jobs():
    return [j.to_dict(with_result=False) for j in sorted(jobs.JOBS.values(), key=lambda j: -j.created)]


@app.get("/api/jobs/{jid}")
def get_job(jid: str):
    j = jobs.JOBS.get(jid)
    if j is None:
        raise HTTPException(404, "job not found")
    return j.to_dict()


@app.get("/api/health")
def health():
    return {"ok": True, "time": time.time(), "sessions": len(SESSIONS)}


# Serve the built frontend if present (frontend/dist) so one process can host everything.
_dist = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend", "dist")
if os.path.isdir(_dist):
    from fastapi.responses import FileResponse

    app.mount("/assets", StaticFiles(directory=os.path.join(_dist, "assets")), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        # SPA fallback: unknown non-API paths serve index.html so client-side routes deep-link correctly.
        full = os.path.join(_dist, path)
        if path and os.path.isfile(full):
            return FileResponse(full)
        return FileResponse(os.path.join(_dist, "index.html"))
