"""API tests for the GlucoGuard backend. Run from the repo root:  python -m pytest backend/tests -q"""
import time

import pytest
from fastapi.testclient import TestClient

from backend.app import app

client = TestClient(app)
SHORT = {"patient": "adult#009", "seed": 3, "episode_days": 0.125}   # 12 decisions = 3 simulated hours


def test_meta_and_patient():
    m = client.get("/api/meta").json()
    assert "Research simulation only" in m["disclaimer"]
    assert len(m["patients"]["test"]) == 6 and m["baselines"] == ["random", "fixed_basal", "cgm_rule", "announced_bb"]
    p = client.get("/api/patient", params={"name": "child#010"}).json()
    assert p["split"] == "test" and p["TDI"] > 0
    assert client.get("/api/patient", params={"name": "nobody"}).status_code == 404


def test_session_step_pause_resume_reset():
    r = client.post("/api/sessions", json={"policy": "cgm_rule", "settings": SHORT})
    assert r.status_code == 200, r.text
    s = r.json()
    sid = s["id"]
    assert s["status"] == "idle" and s["clock"]["decision"] == 0 and s["clock"]["max_decisions"] == 12
    assert len(s["trace"]["cgm"]) == 1                      # only the initial sample exists before any step

    s = client.post(f"/api/sessions/{sid}/step").json()
    assert s["status"] == "paused" and s["clock"]["decision"] == 1
    assert len(s["trace"]["cgm"]) == 1 + 5                  # one decision = 5 real sensor samples
    d = s["last_decision"]
    assert d["t_min"] == 15.0 and "bg_sim_only" in d and "req_basal_mult" in d and "basal_mult" in d

    s = client.post(f"/api/sessions/{sid}/step?n=3").json()
    assert s["clock"]["decision"] == 4 and s["clock"]["elapsed_min"] == 60.0

    client.post(f"/api/sessions/{sid}/run", json={"speed": 50})
    assert client.post(f"/api/sessions/{sid}/step").status_code == 409   # no manual steps while running
    time.sleep(0.15)
    s = client.post(f"/api/sessions/{sid}/pause").json()
    assert s["status"] in ("paused", "completed")
    n = s["clock"]["decision"]
    time.sleep(0.2)
    assert client.get(f"/api/sessions/{sid}").json()["clock"]["decision"] == n   # really paused

    client.post(f"/api/sessions/{sid}/run", json={"speed": 50})
    for _ in range(100):
        st = client.get(f"/api/sessions/{sid}?trace=false").json()
        if st["status"] in ("completed", "terminated"):
            break
        time.sleep(0.05)
    assert st["status"] == "completed" and st["clock"]["decision"] == 12
    assert st["metrics_so_far"]["n_samples"] == 60
    assert client.post(f"/api/sessions/{sid}/run").json()["status"] == "completed"   # cannot run past the end

    s = client.post(f"/api/sessions/{sid}/reset", json={"seed": 4}).json()
    assert s["status"] == "idle" and s["clock"]["decision"] == 0 and s["settings"]["seed"] == 4
    assert client.delete(f"/api/sessions/{sid}").json()["ok"]
    assert client.get(f"/api/sessions/{sid}").status_code == 404


def test_determinism_same_seed():
    a = client.post("/api/sessions", json={"policy": "fixed_basal", "settings": SHORT}).json()["id"]
    b = client.post("/api/sessions", json={"policy": "fixed_basal", "settings": SHORT}).json()["id"]
    ta = client.post(f"/api/sessions/{a}/step?n=6").json()["trace"]["bg_sim_only"]
    tb = client.post(f"/api/sessions/{b}/step?n=6").json()["trace"]["bg_sim_only"]
    assert ta == tb


def test_branch_is_counterfactual_and_does_not_touch_live_session():
    sid = client.post("/api/sessions", json={"policy": "cgm_rule", "settings": SHORT}).json()["id"]
    client.post(f"/api/sessions/{sid}/step?n=4")
    before = client.get(f"/api/sessions/{sid}").json()
    br = client.post(f"/api/sessions/{sid}/branch", json={"policies": ["fixed_basal", "cgm_rule"]}).json()
    assert br["kind"] == "counterfactual_branch" and br["branch_decision"] == 4 and br["branch_t_min"] == 60.0
    assert [r["policy"] for r in br["rollouts"]] == ["fixed_basal", "cgm_rule"]
    fut = br["rollouts"][0]["trace"]
    assert fut["t_min"][0] == 60.0 and len(fut["t_min"]) == 1 + 8 * 5
    assert br["rollouts"][0]["full_episode_metrics"]["n_samples"] == 60
    after = client.get(f"/api/sessions/{sid}").json()
    assert after["clock"]["decision"] == 4 and after["trace"] == before["trace"]   # live state untouched


def test_bad_policy_and_checkpoint_ids():
    assert client.post("/api/sessions", json={"policy": "nope"}).status_code == 400
    assert client.post("/api/sessions", json={"policy": "ckpt:missing/best.pt"}).status_code == 400
    assert client.post("/api/sessions", json={"policy": "ckpt:../x/best.pt"}).status_code == 400


def test_paired_rollout_job():
    j = client.post("/api/rollouts", json={"policies": ["fixed_basal", "cgm_rule"], "settings": SHORT}).json()
    for _ in range(200):
        j = client.get(f"/api/jobs/{j['id']}").json()
        if j["status"] in ("done", "error"):
            break
        time.sleep(0.1)
    assert j["status"] == "done", j.get("error")
    r = j["result"]
    assert r["kind"] == "counterfactual_full_episode" and len(r["rollouts"]) == 2
    assert r["rollouts"][0]["trace"]["meal_g"] == r["rollouts"][1]["trace"]["meal_g"]   # identical meals
    assert "tir_70_180" in r["rollouts"][0]["metrics"]


def test_registry_endpoints_do_not_crash_without_runs():
    assert client.get("/api/runs").status_code == 200
    assert client.get("/api/results").status_code == 200
    assert client.get("/api/runs/nothing/logs").status_code == 404


@pytest.mark.skipif(not __import__("os").path.exists("runs/demo_20ep/best.pt"), reason="demo checkpoint not trained")
def test_checkpoint_session_locks_obs_fields():
    s = client.post("/api/sessions", json={"policy": "ckpt:demo_20ep/best.pt",
                                           "settings": {**SHORT, "history_len": 3}}).json()
    assert s["settings"]["history_len"] == 8 and "history_len" in s["locked_fields"]
    s = client.post(f"/api/sessions/{s['id']}/step").json()
    assert len(s["last_decision"]["q_values"]) == 16
