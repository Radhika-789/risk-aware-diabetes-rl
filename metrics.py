"""
metrics.py - glycemic metrics for one episode of TRUE blood glucose (mg/dL, 3-min samples).
Research simulation only.

LBGI / HBGI use the Kovatchev risk formula, f(BG) = 1.509*(ln(BG)^1.084 - 5.381),
risk = 10*f^2, LBGI = mean over samples of risk where f<0, HBGI likewise for f>0.
test_wrapper.py cross-checks this against simglucose.analysis.risk.risk_index.
"""
from __future__ import annotations

import numpy as np

LOW, HIGH, SEVERE_LOW, VERY_HIGH = 70.0, 180.0, 54.0, 250.0


def kovatchev_f(bg):
    bg = np.clip(np.asarray(bg, dtype=np.float64), 1.0, None)
    return 1.509 * (np.log(bg) ** 1.084 - 5.381)


def lbgi_hbgi(bg):
    f = kovatchev_f(bg)
    return float(np.mean(10.0 * np.minimum(f, 0.0) ** 2)), float(np.mean(10.0 * np.maximum(f, 0.0) ** 2))


def episode_metrics(bg, insulin_u=None, meal_g=None, terminated=False) -> dict:
    bg = np.asarray(bg, dtype=np.float64)
    lbgi, hbgi = lbgi_hbgi(bg)
    m = {
        "tir_70_180": 100.0 * float(np.mean((bg >= LOW) & (bg <= HIGH))),
        "tbr_lt70": 100.0 * float(np.mean(bg < LOW)),
        "tbr_lt54": 100.0 * float(np.mean(bg < SEVERE_LOW)),
        "tar_gt180": 100.0 * float(np.mean(bg > HIGH)),
        "tar_gt250": 100.0 * float(np.mean(bg > VERY_HIGH)),
        "mean_bg": float(bg.mean()),
        "sd_bg": float(bg.std()),
        "cv_bg": 100.0 * float(bg.std() / bg.mean()),
        "min_bg": float(bg.min()),
        "max_bg": float(bg.max()),
        "lbgi": lbgi,
        "hbgi": hbgi,
        "terminated_early": float(bool(terminated)),
        "n_samples": int(bg.size),
    }
    if insulin_u is not None:
        m["total_insulin_u"] = float(np.sum(insulin_u))
    if meal_g is not None:
        m["total_carbs_g"] = float(np.sum(meal_g))
    return m
