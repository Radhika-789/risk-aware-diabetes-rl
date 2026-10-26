"""Read-only view over training runs (runs/<name>/) and evaluation outputs (results*/ dirs).

Everything reported here is read from files written by train_dqn.py / evaluate.py. Nothing is synthesized.
"""
from __future__ import annotations

import csv
import glob
import json
import os
from functools import lru_cache

from .policies import RUNS_DIR

RESULTS_GLOB = os.environ.get("GG_RESULTS_GLOB", "results*")


def _read_csv(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        rows = []
        for r in csv.DictReader(f):
            out = {}
            for k, v in r.items():
                if v is None or v == "":
                    out[k] = None
                    continue
                try:
                    out[k] = float(v) if ("." in v or "e" in v.lower()) else int(v)
                except ValueError:
                    out[k] = v
            rows.append(out)
        return rows


@lru_cache(maxsize=64)
def _checkpoint_meta(path: str, mtime: float) -> dict:
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=True)
    n_params = int(sum(v.numel() for v in ck["state_dict"].values()))
    return {"obs_dim": int(ck["obs_dim"]), "n_actions": int(ck["n_actions"]), "hidden": int(ck["hidden"]),
            "n_params": n_params, "env_config": json.loads(ck.get("env_config_json", "{}") or "{}"),
            "layers": [k for k in ck["state_dict"].keys() if k.endswith("weight")]}


def checkpoint_meta(path: str) -> dict:
    st = os.stat(path)
    meta = dict(_checkpoint_meta(path, st.st_mtime))
    meta.update(size_bytes=st.st_size, modified=st.st_mtime)
    return meta


def list_runs() -> list[dict]:
    runs = []
    for cfg_path in sorted(glob.glob(os.path.join(RUNS_DIR, "*", "config.json"))):
        run_dir = os.path.dirname(cfg_path)
        name = os.path.basename(run_dir)
        try:
            with open(cfg_path) as f:
                config = json.load(f)
        except Exception:  # noqa: BLE001
            config = {}
        train = _read_csv(os.path.join(run_dir, "train_log.csv"))
        val = _read_csv(os.path.join(run_dir, "val_log.csv"))
        ckpts = []
        for fname in ("best.pt", "final.pt"):
            p = os.path.join(run_dir, fname)
            if os.path.exists(p):
                ckpts.append({"id": f"ckpt:{name}/{fname}", "file": fname, "path": p, **checkpoint_meta(p)})
        best_rows = [r for r in val if r.get("is_best") == 1]
        selected = best_rows[-1] if best_rows else None
        runs.append({
            "name": name, "dir": run_dir, "config": config,
            "episodes_planned": (config.get("args") or {}).get("episodes"),
            "episodes_done": len(train),
            "complete": bool(train) and (config.get("args") or {}).get("episodes") == len(train),
            "last_train": train[-1] if train else None,
            "n_val_evals": len(val),
            "selected_checkpoint": selected,   # the val_log row that produced best.pt (max val_return so far)
            "checkpoints": ckpts,
            "demo": name.startswith("demo"),
        })
    return runs


def run_logs(name: str) -> dict:
    run_dir = os.path.join(RUNS_DIR, name)
    if ".." in name or not os.path.isdir(run_dir):
        raise FileNotFoundError(name)
    return {"name": name, "train": _read_csv(os.path.join(run_dir, "train_log.csv")),
            "val": _read_csv(os.path.join(run_dir, "val_log.csv"))}


def list_results() -> list[dict]:
    """Every directory matching results*/ containing summary.csv (written by evaluate.py)."""
    out = []
    for d in sorted(glob.glob(RESULTS_GLOB)):
        s = os.path.join(d, "summary.csv")
        if not os.path.isdir(d) or not os.path.exists(s):
            continue
        meta_path = os.path.join(d, "meta.json")
        meta = {}
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
        summary = _read_csv(s)
        episodes = _read_csv(os.path.join(d, "episodes.csv"))
        split = meta.get("split")
        if not split:
            # evaluate.py does not record the split in CSV; infer it from the patients that were run.
            from t1d_env import TEST_PATIENTS, TRAIN_PATIENTS, VAL_PATIENTS
            pats = {e.get("patient") for e in episodes}
            for nm, lst in (("test", TEST_PATIENTS), ("val", VAL_PATIENTS), ("train", TRAIN_PATIENTS)):
                if pats and pats <= set(lst):
                    split = nm
                    break
            split = split or "unknown"
        out.append({"name": os.path.basename(d), "dir": d, "split": split, "meta": meta,
                    "policies": [r["policy"] for r in summary], "summary": summary, "episodes": episodes,
                    "modified": os.path.getmtime(s)})
    return out
