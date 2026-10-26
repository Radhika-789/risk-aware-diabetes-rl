"""
Phase 0 smoke test for SimGlucose (research simulation only; NOT medical software).

Uses the CORE simulator API (simglucose.simulation.env.T1DSimEnv), not the bundled
gym wrapper, because the bundled wrapper only controls basal insulin (bolus is
hard-coded to 0) and depends on the legacy `gym==0.9.4` package.

What it checks:
  1. Imports + versions
  2. Build env (patient + CGM sensor + pump + meal scenario), reset, print reset output
  3. Step with the built-in BBController (basal-bolus baseline); print a few steps
  4. Run one simulated day, print summary + wall-clock speed
  5. Reproducibility: same seeds -> same trace? Does env.reset() replay the same episode?

Run:  python smoke_test_simglucose.py
"""
import sys
import time
from datetime import datetime
from importlib.metadata import version

import numpy as np

from simglucose.actuator.pump import InsulinPump
from simglucose.controller.base import Action
from simglucose.controller.basal_bolus_ctrller import BBController
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.sensor.cgm import CGMSensor
from simglucose.simulation.env import T1DSimEnv
from simglucose.simulation.scenario_gen import RandomScenario

PATIENT = "adolescent#001"
SENSOR = "Dexcom"      # sample_time = 3 min (from sensor_params.csv)
PUMP = "Insulet"       # inc 0.05, max_basal/max_bolus = 30 U/min (pump_params.csv)
START = datetime(2018, 1, 1, 0, 0, 0)


def make_env(patient_name=PATIENT, seed=0):
    """Build one simulator instance. Separate seeds for patient / sensor / meals."""
    patient = T1DPatient.withName(patient_name, random_init_bg=False, seed=seed)
    sensor = CGMSensor.withName(SENSOR, seed=seed + 1)
    pump = InsulinPump.withName(PUMP)
    scenario = RandomScenario(start_time=START, seed=seed + 2)
    return T1DSimEnv(patient, sensor, pump, scenario)


def run_bb(env, n_steps, verbose_first=0, verbose_meals=0):
    """Run the basal-bolus controller for n_steps; return CGM trace + history."""
    ctrl = BBController()
    step = env.reset()  # namedtuple: observation, reward, done, info(dict)
    obs, reward, done, info = step.observation, step.reward, step.done, step.info
    cgm_trace = [obs.CGM]
    meals_shown = 0
    for k in range(n_steps):
        action = ctrl.policy(obs, reward, done, **info)  # Action(basal, bolus) in U/min
        step = env.step(action)
        obs, reward, done, info = step.observation, step.reward, step.done, step.info
        cgm_trace.append(obs.CGM)
        show = k < verbose_first
        if info["meal"] > 0 and meals_shown < verbose_meals:
            show, meals_shown = True, meals_shown + 1
        if show:
            print(f"  step {k:3d} | {info['time']} | CGM={obs.CGM:7.2f} | BG={info['bg']:7.2f} "
                  f"| meal={info['meal']:5.2f} g/min | basal={action.basal:.4f} "
                  f"bolus={action.bolus:.4f} U/min | reward={reward:8.4f} | done={done}")
        if done:
            print(f"  episode terminated early at step {k} (BG<10 or BG>600 per env code)")
            break
    return np.array(cgm_trace), env


def main():
    print("=== 1. Versions ===")
    print("python     :", sys.version.split()[0])
    for pkg in ("simglucose", "numpy", "scipy", "pandas", "gymnasium", "gym"):
        print(f"{pkg:11s}:", version(pkg))
    try:
        import torch
        print("torch      :", torch.__version__)
    except ImportError:
        print("torch      : not installed (fine for this test)")

    print("\n=== 2. Build env + reset ===")
    env = make_env(seed=0)
    first = env.reset()
    print("reset().observation :", first.observation)
    print("reset().reward/done :", first.reward, first.done)
    print("reset().info keys   :", sorted(first.info.keys()))
    print("sample_time (min)   :", env.sample_time)

    print("\n=== 3. First steps + first meal steps (BBController baseline) ===")
    run_bb(make_env(seed=0), n_steps=480, verbose_first=4, verbose_meals=3)

    print("\n=== 4. One simulated day (480 steps x 3 min) ===")
    t0 = time.time()
    cgm, e = run_bb(make_env(seed=0), n_steps=480)
    dt = time.time() - t0
    bg = np.array(e.BG_hist)
    print(f"steps run          : {len(cgm) - 1}")
    print(f"wall-clock         : {dt:.2f} s  (~{(len(cgm) - 1) / dt:.0f} agent-steps/s)")
    print(f"CGM min/mean/max   : {cgm.min():.1f} / {cgm.mean():.1f} / {cgm.max():.1f} mg/dL")
    print(f"true BG time 70-180: {100 * np.mean((bg >= 70) & (bg <= 180)):.1f} %")
    print(f"true BG time <70   : {100 * np.mean(bg < 70):.1f} %")
    print(f"total meals (g)    : {sum(e.CHO_hist) * e.sample_time:.0f}")

    print("\n=== 5. Reproducibility checks ===")
    a, _ = run_bb(make_env(seed=0), n_steps=480)
    b, _ = run_bb(make_env(seed=0), n_steps=480)
    c, _ = run_bb(make_env(seed=7), n_steps=480)
    print("same seeds, new env -> identical CGM trace :", np.allclose(a, b))
    print("different seed       -> different CGM trace:", not np.allclose(a, c))
    env2 = make_env(seed=0)
    r1, _ = run_bb(env2, n_steps=480)
    r2, _ = run_bb(env2, n_steps=480)   # run_bb calls env.reset() again on the SAME object
    print("env.reset() on same object replays same episode:", np.allclose(r1, r2))

    print("\nSMOKE TEST FINISHED (no exceptions).")


if __name__ == "__main__":
    main()
