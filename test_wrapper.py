"""
test_wrapper.py - tests for t1d_env / metrics / baselines / rollout. No torch, no pytest needed.
Run:  python test_wrapper.py        (takes roughly 1-2 minutes)
Prints PASS/FAIL per test and exits with code 1 if any test fails.
"""
import sys
import time
import traceback
from types import SimpleNamespace

import numpy as np

import metrics
from baselines import Policy, make_baselines
from rollout import metrics_from_log, run_episode
from t1d_env import (RewardConfig, ShieldConfig, T1DRiskEnv, TEST_PATIENTS, TRAIN_PATIENTS, VAL_PATIENTS,
                     apply_shield, decode_obs, patient_info, reward_bg, SAMPLE_TIME_MIN)

RESULTS = []


def test(fn):
    t0 = time.time()
    try:
        fn()
        RESULTS.append((fn.__name__, True, ""))
        print(f"PASS  {fn.__name__}  ({time.time() - t0:.1f}s)")
    except Exception as e:  # noqa: BLE001
        RESULTS.append((fn.__name__, False, repr(e)))
        print(f"FAIL  {fn.__name__}: {e!r}")
        traceback.print_exc()
    return fn


NO_SHIELD = ShieldConfig(enabled=False)


def short_env(**kw):
    kw.setdefault("episode_days", 0.25)   # 24 decisions = 6 simulated hours
    return T1DRiskEnv(patients=TEST_PATIENTS, **kw)


@test
def test_patient_split_disjoint():
    assert len(TRAIN_PATIENTS) == 18 and len(VAL_PATIENTS) == 6 and len(TEST_PATIENTS) == 6
    assert not (set(TRAIN_PATIENTS) & set(VAL_PATIENTS)) and not (set(TRAIN_PATIENTS) & set(TEST_PATIENTS))
    assert not (set(VAL_PATIENTS) & set(TEST_PATIENTS))
    for p in TRAIN_PATIENTS + VAL_PATIENTS + TEST_PATIENTS:
        info = patient_info(p)
        assert info["TDI"] > 0 and info["basal_U_per_min"] > 0


@test
def test_reset_and_obs_space():
    env = short_env()
    obs, info = env.reset(seed=0)
    assert obs.shape == (env.obs_dim,) and obs.dtype == np.float32
    assert env.observation_space.contains(obs)
    assert env.action_space.n == 16
    assert info["patient"] in TEST_PATIENTS


@test
def test_step_contract_and_time_limit():
    env = short_env()
    env.reset(seed=0)
    n, term, trunc = 0, False, False
    while not (term or trunc):
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        assert env.observation_space.contains(obs) and np.isfinite(r)
        n += 1
    assert trunc and not term and n == env.max_decisions == 24, (n, term, trunc)
    try:
        env.step(0)
        raise AssertionError("step after episode end should raise")
    except RuntimeError:
        pass


@test
def test_determinism_same_seed_and_different_seed():
    def trace(seed):
        env = short_env()
        obs, _ = env.reset(seed=seed, options={"patient": "adult#009"})
        out = [obs.copy()]
        for _ in range(12):
            obs, r, *_ = env.step(env.nearest_action(1.0, 0.0))
            out.append(np.append(obs, r))
        return out
    a, b, c = trace(5), trace(5), trace(6)
    assert all(np.array_equal(x, y) for x, y in zip(a, b)), "same seed must reproduce exactly"
    assert not all(np.array_equal(x, z) for x, z in zip(a, c)), "different seed must differ"


@test
def test_gymnasium_env_checker():
    from gymnasium.utils.env_checker import check_env
    check_env(short_env(), skip_render_check=True)


@test
def test_action_units_conversion():
    """bolus index -> bolus units, delivered as units/3 U/min over the first 3-min step."""
    env = short_env(shield_cfg=NO_SHIELD)
    env.reset(seed=1, options={"patient": "adolescent#009"})
    tdi, basal = env.pinfo["TDI"], env.pinfo["basal_U_per_min"]
    a = env.nearest_action(1.0, 0.06)
    mult, units = env.decode_action(a)
    assert mult == 1.0 and abs(units - 0.06 * tdi) < 1e-9
    env.step(a)
    delivered_first_step = env.sim.insulin_hist[0] * SAMPLE_TIME_MIN
    expected = basal * SAMPLE_TIME_MIN + units
    assert abs(delivered_first_step - expected) < 1e-3, (delivered_first_step, expected)
    delivered_second_step = env.sim.insulin_hist[1] * SAMPLE_TIME_MIN
    assert abs(delivered_second_step - basal * SAMPLE_TIME_MIN) < 1e-3   # bolus only in first step


@test
def test_iob_estimate_matches_linear_decay():
    env = short_env(shield_cfg=NO_SHIELD, episode_days=0.5)
    env.reset(seed=1, options={"patient": "adolescent#009"})
    _, units = env.decode_action(env.nearest_action(1.0, 0.06))
    env.step(env.nearest_action(1.0, 0.06))
    # bolus delivered in the 1st of 5 sensor steps; age of that step at decision end = 4 steps = 12 min
    expected = units * (1 - 4 * SAMPLE_TIME_MIN / env.dia_min)
    assert abs(env.iob - expected) / expected < 0.01, (env.iob, expected)
    prev = env.iob
    for _ in range(6):
        env.step(env.nearest_action(1.0, 0.0))
        assert env.iob < prev + 1e-9, "IOB must decay with nominal basal and no new bolus"
        prev = env.iob
    expected_later = units * max(0.0, 1 - (4 + 6 * 5) * SAMPLE_TIME_MIN / env.dia_min)
    assert abs(env.iob - expected_later) / units < 0.02, (env.iob, expected_later)


@test
def test_shield_logic():
    cfg = ShieldConfig()
    tdi = 40.0
    assert apply_shield(1.5, 2.0, 70.0, 0.0, tdi, cfg) == (0.0, 0.0, True, True)     # low CGM -> suspend
    assert apply_shield(1.0, 2.0, 150.0, 7.0, tdi, cfg) == (1.0, 0.0, False, True)   # 7+2 > 0.2*40=8
    assert apply_shield(1.0, 2.0, 150.0, 5.0, tdi, cfg) == (1.0, 2.0, False, False)  # allowed
    assert apply_shield(1.0, 2.0, 50.0, 0.0, tdi, ShieldConfig(enabled=False)) == (1.0, 2.0, False, False)


@test
def test_hard_caps_apply_even_without_shield():
    env = short_env(shield_cfg=NO_SHIELD)
    env.reset(seed=2, options={"patient": "child#009"})
    _, _, _, _, info = env.step_units(50.0, 1e6)
    assert info["basal_mult"] == env.scfg.hard_max_basal_mult
    assert info["bolus_units"] <= env.scfg.hard_max_bolus_frac_of_tdi * env.pinfo["TDI"] + 1e-9


@test
def test_reward_shape():
    r = RewardConfig()
    assert reward_bg(120, r) == 1.0 and reward_bg(70, r) == 1.0 and reward_bg(180, r) == 1.0
    assert reward_bg(45, r) < reward_bg(60, r) < reward_bg(69, r) < 0          # worse as BG drops
    assert reward_bg(300, r) < reward_bg(200, r) < 0
    assert reward_bg(45, r) < reward_bg(205, r)                                 # 25 below << 25 above
    assert reward_bg(50, r) < reward_bg(55, r) - 1.0                            # severe-low extra penalty kicks in
    t = RewardConfig(mode="tir")
    assert reward_bg(120, t) == 1.0 and reward_bg(50, t) == 0.0 and reward_bg(300, t) == 0.0


@test
def test_terminal_penalty_and_termination():
    env = short_env(shield_cfg=NO_SHIELD, episode_days=1.0)
    env.reset(seed=3, options={"patient": "adult#009"})
    term, steps, last_r = False, 0, 0.0
    while steps < env.max_decisions and not term:     # max insulin: should crash BG below 10
        _, last_r, term, trunc, _ = env.step(env.nearest_action(1.5, 0.12))
        steps += 1
        if trunc:
            break
    assert term, "sustained maximum dosing should drive the simulator to its BG<10 termination"
    assert last_r < -env.rcfg.terminal_penalty + 5


@test
def test_meal_announcement_flag_changes_obs_dim():
    env = short_env(announce_meals=True)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (short_env().obs_dim + 1,)
    d = decode_obs(obs, env.history_len)
    assert d["cgm"].shape == (env.history_len,)


@test
def test_metrics_basic_and_risk_crosscheck():
    bg = np.array([60, 65, 100, 120, 150, 200, 260, 300], dtype=float)
    m = metrics.episode_metrics(bg)
    assert abs(m["tir_70_180"] - 37.5) < 1e-9 and abs(m["tbr_lt70"] - 25.0) < 1e-9
    assert abs(m["tbr_lt54"]) < 1e-9 and abs(m["tar_gt180"] - 37.5) < 1e-9 and abs(m["tar_gt250"] - 25.0) < 1e-9
    from simglucose.analysis.risk import risk_index
    for v in (45.0, 80.0, 112.5, 180.0, 300.0):
        lbgi_ref, hbgi_ref, _ = risk_index([v], 1)
        lbgi, hbgi = metrics.lbgi_hbgi([v])
        assert abs(lbgi - lbgi_ref) < 1e-6 and abs(hbgi - hbgi_ref) < 1e-6, (v, lbgi, lbgi_ref, hbgi, hbgi_ref)


@test
def test_baselines_run_and_paired_episodes_identical_inputs():
    env = short_env()
    seen = {}
    for pol in make_baselines(seed=0):
        m = run_episode(env, pol, seed=11, patient="adolescent#010")
        assert 0 <= m["tir_70_180"] <= 100 and np.isfinite(m["episode_return"])
        seen[pol.name] = m
    # identical (seed, patient) -> identical meals regardless of policy
    carbs = {round(v["total_carbs_g"], 6) for v in seen.values()}
    assert len(carbs) == 1, carbs
    assert set(seen) == {"random", "fixed_basal", "cgm_rule", "announced_bb"}



# ----------------------------------------------------------------------------------------------
# Phase 1 audit tests
# ----------------------------------------------------------------------------------------------
@test
def test_patient_override_guard_and_train_env_isolation():
    train_env = T1DRiskEnv(patients=TRAIN_PATIENTS)
    seen = {train_env.reset(seed=i)[1]["patient"] for i in range(60)}
    assert seen <= set(TRAIN_PATIENTS) and not (seen & set(VAL_PATIENTS + TEST_PATIENTS))
    train_env.reset(seed=0, options={"patient": TRAIN_PATIENTS[0]})          # allowed
    for outside in (VAL_PATIENTS[0], TEST_PATIENTS[0]):
        try:
            train_env.reset(seed=0, options={"patient": outside})
            raise AssertionError("override with a patient outside the env's list must raise")
        except ValueError:
            pass


@test
def test_cgm_history_alignment_and_reward_integration():
    """Observation history = CGM at decision boundaries (initial fill = the reset reading the agent saw);
    step CGM equals the simulator's last recorded CGM; reward = mean per-sample reward of the TRUE BG."""
    env = short_env(episode_days=0.5, shield_cfg=NO_SHIELD)
    obs, info = env.reset(seed=3, options={"patient": "adult#009"})
    first_cgm = info["cgm"]
    assert np.allclose(obs[:8] * 400, first_cgm, atol=1e-3), "history must start filled with the reset reading"
    for d in range(1, 9):
        obs, r, term, trunc, info = env.step(env.nearest_action(1.0, 0.03 if d == 2 else 0.0))
        assert not term
        for j, k in enumerate(range(d - 7, d + 1)):      # entry j = decision boundary k (k < 1: initial fill)
            expected = env.sim.CGM_hist[5 * k] if k >= 1 else first_cgm
            assert abs(obs[j] * 400 - expected) < 1e-3, (d, j)
        assert abs(env.sim.CGM_hist[-1] - env.cgm_last) < 1e-9
        assert abs(obs[7] * 400 - env.cgm_last) < 1e-3
        expected_r = np.mean([reward_bg(b, env.rcfg) for b in env.sim.BG_hist[-5:]])
        assert abs(r - expected_r) < 1e-9


@test
def test_all_actions_map_to_the_documented_table():
    env = short_env()
    env.reset(seed=0, options={"patient": "adolescent#010"})
    tdi = env.pinfo["TDI"]
    for a in range(env.action_space.n):
        ib, ik = divmod(a, 4)
        mult, units = env.decode_action(a)
        assert mult == env.acfg.basal_mults[ib] and abs(units - env.acfg.bolus_fracs_of_tdi[ik] * tdi) < 1e-12
        assert env.nearest_action(mult, env.acfg.bolus_fracs_of_tdi[ik]) == a


@test
def test_initial_bg_in_range_for_all_patients():
    env = T1DRiskEnv(patients=TRAIN_PATIENTS + VAL_PATIENTS + TEST_PATIENTS)
    for p in TRAIN_PATIENTS + VAL_PATIENTS + TEST_PATIENTS:
        _, info = env.reset(seed=7, options={"patient": p})
        assert 70 <= info["bg"] <= 180 and 70 <= info["cgm"] <= 180, (p, info["bg"], info["cgm"])


@test
def test_metrics_from_log_counts_remaining_time_after_termination():
    env = SimpleNamespace(max_decisions=24, action_repeat=5)             # 120 samples per episode
    bg = np.append(np.full(59, 120.0), 9.0)                              # 60 samples, then crash below 10
    log = {"bg": bg, "insulin_u": np.zeros(60), "meal_g": np.zeros(60), "terminated": True}
    m = metrics_from_log(env, log)
    assert m["n_samples"] == 120 and abs(m["survival_fraction"] - 0.5) < 1e-12
    assert abs(m["tir_70_180"] - 100 * 59 / 120) < 1e-9, m["tir_70_180"]
    assert abs(m["tbr_lt54"] - 100 * 61 / 120) < 1e-9, m["tbr_lt54"]
    ok = metrics_from_log(env, {"bg": np.full(120, 120.0), "insulin_u": np.zeros(120),
                                "meal_g": np.zeros(120), "terminated": False})
    assert ok["survival_fraction"] == 1.0 and ok["tir_70_180"] == 100.0


@test
def test_crashed_episode_cannot_report_tir_above_survival_fraction():
    class MaxInsulin(Policy):
        name = "max_insulin"

        def act(self, obs, env):
            return env.nearest_action(1.5, 0.12)
    env = T1DRiskEnv(patients=TEST_PATIENTS, episode_days=1.0, shield_cfg=NO_SHIELD)
    m = run_episode(env, MaxInsulin(), seed=3, patient="adult#009")
    assert m["terminated_early"] == 1.0 and m["survival_fraction"] < 1.0
    assert m["n_samples"] == env.max_decisions * env.action_repeat
    assert m["tir_70_180"] <= 100 * m["survival_fraction"] + 1e-9, (m["tir_70_180"], m["survival_fraction"])


if __name__ == "__main__":
    n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
    print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} tests passed")
    sys.exit(1 if n_fail else 0)