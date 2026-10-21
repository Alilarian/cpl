"""Reward-hacking metrics on synthetic curves."""
import csv
import os

import numpy as np

from research.reward_learning.hacking import hacking_metrics, slope_ci

REFS = dict(T_rand=10.0, T_oracle=160.0, L_rand=10.0, L_oracle=160.0, oracle_success=1.0, random_success=0.0)


def _write_log(d, steps, true, learned, success):
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "log.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["step", "eval/reward", "eval/learned_reward", "eval/success",
                                          "eval/learned_reward_std", "eval/length"])
        w.writeheader()
        for s, t, l, su in zip(steps, true, learned, success):
            w.writerow({"step": s, "eval/reward": t, "eval/learned_reward": l, "eval/success": su,
                        "eval/learned_reward_std": 25.0, "eval/length": 250})


def test_oracle_path_has_zero_gap_and_no_flag(tmp_path):
    steps = np.arange(0, 1_000_001, 10_000)
    true = np.linspace(10, 160, len(steps))
    _write_log(str(tmp_path), steps, true, true, np.linspace(0, 1, len(steps)))
    m = hacking_metrics(str(tmp_path), REFS, val_disagreement=0.1)
    assert np.allclose(m["curves"]["G"], 0.0)
    assert not m["flag"]["hacking"]
    assert np.isclose(m["disagreement_ratio"]["final"], 25.0 / 250 / 0.1)


def test_learned_rising_true_flat_is_flagged(tmp_path):
    steps = np.arange(0, 1_000_001, 10_000)
    rng = np.random.RandomState(0)
    learned = np.linspace(10, 400, len(steps)) + rng.randn(len(steps))
    true = 40 + rng.randn(len(steps))  # stuck far below the oracle
    _write_log(str(tmp_path), steps, true, learned, np.full(len(steps), 0.1))
    m = hacking_metrics(str(tmp_path), REFS, None)
    assert m["flag"]["hacking"] and m["final"]["G"] > 1.0
    assert m["second_half"]["L_slope_ci"][0] > 0


def test_good_learned_reward_is_not_flagged(tmp_path):
    steps = np.arange(0, 1_000_001, 10_000)
    rng = np.random.RandomState(1)
    true = np.minimum(160, np.linspace(10, 300, len(steps))) + rng.randn(len(steps))
    learned = true + rng.randn(len(steps))
    _write_log(str(tmp_path), steps, true, learned, np.minimum(1, np.linspace(0, 2, len(steps))))
    m = hacking_metrics(str(tmp_path), REFS, None)
    assert not m["flag"]["hacking"] and abs(m["final"]["G"]) < 0.1


def test_normalization_is_invariant_to_the_affine_reward_map(tmp_path):
    steps = np.arange(0, 1_000_001, 10_000)
    rng = np.random.RandomState(2)
    learned_std = rng.randn(len(steps)).cumsum()  # learned return in standardized units
    true = 50 + rng.randn(len(steps))
    a, b, T = 0.3, 0.6, 250
    _write_log(str(tmp_path / "x"), steps, true, learned_std, np.zeros(len(steps)))
    _write_log(str(tmp_path / "y"), steps, true, a * learned_std + b * T, np.zeros(len(steps)))
    refs_x = dict(REFS, L_rand=-3.0, L_oracle=4.0)
    refs_y = dict(REFS, L_rand=a * -3.0 + b * T, L_oracle=a * 4.0 + b * T)
    gx = hacking_metrics(str(tmp_path / "x"), refs_x, None)["curves"]["G"]
    gy = hacking_metrics(str(tmp_path / "y"), refs_y, None)["curves"]["G"]
    assert np.allclose(gx, gy)


def test_slope_bootstrap_is_reproducible():
    x = np.arange(50) * 10_000.0
    y = 0.5 * x / 1e6 + np.random.RandomState(0).randn(50) * 0.01
    s1, ci1 = slope_ci(x, y, 500, 7)
    s2, ci2 = slope_ci(x, y, 500, 7)
    assert s1 == s2 and ci1 == ci2
    assert ci1[0] <= s1 <= ci1[1] and abs(s1 - 0.5) < 0.1
