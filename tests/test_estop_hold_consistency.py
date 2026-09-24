"""
Regression + real-data checks for the holding-model E-stop generator
(scripts/generate_estop_hold_labels.py, scripts/estop_hold_common.py).

Part A (always runs, no CHPC/oracle/MetaWorld needed): exercises the pure,
environment-free building blocks the spec calls out as correctness gates --
the Sum-estimator backward pass (Section 6), the hold-controller's t=0
fallback (Section 3), and the deterministic first-crossing stopping rule
(Section 8) -- against hand-computed / synthetic expectations.

Part B (CHPC only; auto-skipped elsewhere): runs the real evaluate_segment
search against a real pool + oracle and checks the spec's Section 14
validation table: restoration fidelity, Delta_tau > threshold, every earlier
evaluated t satisfies Delta_t <= threshold, and the full-segment score-
difference identity (S(prefix+hold) - S(full_orig) == gamma^tau * Delta_tau,
i.e. the common prefix telescopes out exactly).

Run everywhere:  pytest tests/test_estop_hold_consistency.py -v
On CHPC, point at a real pool / oracle if the defaults don't match:
  ESTOP_HOLD_POOL_PATH=/scratch/.../demo_pool/mw_drawer-open-v2/pool.npz \\
  ESTOP_HOLD_ORACLE_DIR=runs/runs/chpc/oracle_sac_seeds/mw_drawer-open-v2/seed-1 \\
  pytest tests/test_estop_hold_consistency.py -v
"""

import os

import numpy as np
import pytest
import torch

import scripts.estop_hold_common as ehc
from scripts.tune_estop_hold_threshold import first_stop, threshold_for_rate

# ---------------------------------------------------------------------------
# Synthetic env + oracle standing in for MetaWorld + the frozen PIQL oracle,
# so evaluate_segment's real control flow (restore -> hold rollout -> MCMC
# value queries -> backward continuation pass -> stopping rule) can be
# exercised end-to-end with no CHPC/MetaWorld dependency at all. Dynamics are
# trivial by design: state' = state + action[:-1] (last action dim is an
# inert "gripper" component, mirroring MetaWorld's action layout), so a hold
# action (zero displacement, spec Section 3) provably freezes the state.
# ---------------------------------------------------------------------------


class _FakeEnv:
    def __init__(self):
        self._state = None

    def set_state(self, state):
        self._state = np.asarray(state, dtype=np.float32).copy()

    def get_obs(self):
        return self._state.copy()

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        self._state = self._state + action[:-1]
        reward = -float(np.square(self._state).sum())
        return self._state.copy(), reward, False, {}


class _FakeDist:
    def __init__(self, lead_shape, act_dim):
        self._lead_shape = lead_shape
        self._act_dim = act_dim

    def sample(self):
        return torch.zeros(*self._lead_shape, self._act_dim)


class _FakeNetwork:
    def encoder(self, obs):
        return obs

    def actor(self, obs):
        return _FakeDist(obs.shape[:-1], act_dim=4)

    def critic(self, obs, action):
        value = -(obs ** 2).sum(dim=-1)
        return value.unsqueeze(0)  # fake ensemble dim of size 1


class _FakeOracle:
    def __init__(self):
        self.network = _FakeNetwork()


def _build_toy_segment(start, step, L):
    """
    Deterministic segment matching _FakeEnv's dynamics exactly: state moves by
    `step` every timestep starting from `start`. Returns (obs, action, reward,
    state) arrays shaped like one pool.npz row.
    """
    obs_dim = len(start)
    obs = np.empty((L, obs_dim), dtype=np.float32)
    action = np.empty((L, obs_dim + 1), dtype=np.float32)
    reward = np.empty((L,), dtype=np.float32)
    state = np.empty((L, obs_dim), dtype=np.float32)

    s = np.array(start, dtype=np.float32)
    for t in range(L):
        obs[t] = s
        state[t] = s
        action[t] = np.array(list(step) + [0.3], dtype=np.float32)  # arbitrary inert gripper command
        s = s + np.array(step, dtype=np.float32)
        reward[t] = -float(np.square(s).sum())

    return obs, action, reward, state


def test_evaluate_segment_stops_immediately_when_holding_is_always_better():
    """
    Segment starts at the origin (reward-maximizing point under -||s||^2) and
    moves steadily away. Holding at s_0 keeps reward at 0 forever; continuing
    only gets worse. tau=0 should fire (spec Section 8: "Can the human stop
    immediately? Yes") with a large, strictly positive gap.
    """
    obs, action, reward, state = _build_toy_segment(start=(0.0, 0.0, 0.0), step=(0.5, 0.0, 0.0), L=10)
    env, oracle = _FakeEnv(), _FakeOracle()

    result = ehc.evaluate_segment(
        env, oracle, obs, action, reward, state,
        gamma=0.9, mcmc_samples=4, device="cpu", min_horizon=2, threshold=0.5, stop_early=True,
    )

    assert result["stopped"] is True
    assert result["stop_index"] == 0
    assert result["oracle_gap"] > 0.5
    assert result["horizon"] == 10
    assert result["positive"]["obs"].shape == (10, 3)
    assert result["negative"]["obs"].shape == (10, 3)
    # The hold suffix must actually be frozen at s_0 -- not a repeated action
    # label masquerading as one (spec Section 3: "physically simulated").
    np.testing.assert_allclose(result["positive"]["obs"], 0.0, atol=1e-6)
    np.testing.assert_allclose(result["positive"]["reward"], 0.0, atol=1e-6)
    # The original suffix must be the real recorded continuation, unmodified.
    np.testing.assert_allclose(result["negative"]["obs"], obs[0:10], atol=1e-6)


def test_evaluate_segment_censors_when_continuing_is_always_better():
    """
    Segment starts away from the origin and steadily returns to it (reward
    improving every step). Holding anywhere just freezes at a worse reward
    than letting the recorded trajectory continue -- no candidate should ever
    cross a nonnegative threshold, so the segment must be censored.
    """
    obs, action, reward, state = _build_toy_segment(start=(5.0, 0.0, 0.0), step=(-0.5, 0.0, 0.0), L=10)
    env, oracle = _FakeEnv(), _FakeOracle()

    result = ehc.evaluate_segment(
        env, oracle, obs, action, reward, state,
        gamma=0.9, mcmc_samples=4, device="cpu", min_horizon=2, threshold=0.0, stop_early=True,
    )

    assert result["stopped"] is False
    assert result["stop_index"] is None
    assert all(d is None or d <= 0.0 for _, d in result["gaps"])


def test_estop_hold_buffer_roundtrip_and_mask_zeroes_padding(tmp_path):
    """
    Closes the loop between generate_estop_hold_labels.py's on-disk format and
    the training-side masking logic: write a synthetic estop_hold_labels.npz
    with per-pair `horizon` < T (i.e. real padding), load it through
    EstopHoldBuffer, and confirm a horizon-derived mask (computed exactly as
    EstopHoldCPL / EstopHoldPIQL / RewardEstopHoldCPL do) makes padded garbage
    contribute nothing to a summed per-step score -- regardless of what junk
    value the padding actually holds.
    """
    import gym

    from research.algs.scoring import score_segments
    from research.datasets import EstopHoldBuffer

    M, T, obs_dim, act_dim = 4, 6, 3, 2
    horizon = np.array([3, 6, 4, 1], dtype=np.int32)

    obs = np.random.default_rng(0).normal(size=(M, 2, T, obs_dim)).astype(np.float32)
    # Scaled well within [-1, 1] so EstopHoldBuffer's action_eps clipping is a
    # no-op here -- this test is about masking, not clipping.
    action = (0.1 * np.random.default_rng(1).normal(size=(M, 2, T, act_dim))).astype(np.float32)
    reward = np.random.default_rng(2).normal(size=(M, 2, T)).astype(np.float32)
    for m in range(M):
        h = horizon[m]
        obs[m, :, h:] = 999.0    # deliberately huge "garbage" padding
        action[m, :, h:] = 999.0
        reward[m, :, h:] = 999.0

    path = str(tmp_path / "estop_hold_labels.npz")
    ehc.save_npz(
        path,
        obs=obs, action=action, reward=reward,
        horizon=horizon,
        stop_index=(T - horizon).astype(np.int32),
        oracle_gap=np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        pool_index=np.arange(M, dtype=np.int32),
        checkpoint_step=np.zeros(M, dtype=np.int64),
        threshold=np.array([0.5], dtype=np.float32),
    )

    obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
    act_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32)
    dataset = EstopHoldBuffer(obs_space, act_space, path=path, batch_size=M)

    batch = next(iter(dataset))
    assert batch["obs"].shape == (M, 2, T, obs_dim)
    # EstopHoldBuffer shuffles row order (np.random.permutation) -- identify
    # each returned row by its (unique, round-tripped) oracle_gap rather than
    # assuming positional order is preserved.
    gap_to_orig = {1.0: 0, 2.0: 1, 3.0: 2, 4.0: 3}
    order = [gap_to_orig[round(float(g), 3)] for g in batch["oracle_gap"]]
    np.testing.assert_array_equal(batch["horizon"], horizon[order])

    obs_t = torch.from_numpy(batch["obs"])
    action_t = torch.from_numpy(batch["action"])
    horizon_t = torch.from_numpy(batch["horizon"].astype(np.int64))

    time = torch.arange(T)
    mask = (time.unsqueeze(0) < horizon_t.unsqueeze(1)).float().unsqueeze(1)  # (M, 1, T)

    def _sum_scorer(o, a):
        # Deliberately sensitive to the 999.0 padding sentinel, so any leak
        # through the mask would blow up the assertion below.
        return o.sum(dim=-1) + a.sum(dim=-1)

    masked_score = score_segments(obs_t, action_t, _sum_scorer, discount=1.0, mask=mask)  # (M, 2)
    unmasked_score = score_segments(obs_t, action_t, _sum_scorer, discount=1.0, mask=None)

    for row, m in enumerate(order):
        h = int(horizon[m])
        if h < T:
            # Masked score must equal summing only the first h real steps by hand.
            expected0 = float(obs[m, 0, :h].sum() + action[m, 0, :h].sum())
            expected1 = float(obs[m, 1, :h].sum() + action[m, 1, :h].sum())
            assert masked_score[row, 0].item() == pytest.approx(expected0, abs=1e-3)
            assert masked_score[row, 1].item() == pytest.approx(expected1, abs=1e-3)
            # The unmasked score, by contrast, must be contaminated by the 999.0 padding.
            assert unmasked_score[row, 0].item() != pytest.approx(expected0, abs=1e-3)
        else:
            assert masked_score[row, 0].item() == pytest.approx(unmasked_score[row, 0].item(), abs=1e-3)

# ---------------------------------------------------------------------------
# Part A: pure logic, no oracle/CHPC needed
# ---------------------------------------------------------------------------


def test_continuation_scores_matches_manual_backward_pass():
    """S(C_t) = B_t - V(s_t), B_L = V(s_L), B_t = r_t + gamma*B_{t+1} (spec Section 6)."""
    rewards = np.array([1.0, 0.5, -0.2, 2.0], dtype=np.float64)
    values = np.array([0.1, 0.3, -0.1, 0.4, 0.0], dtype=np.float64)  # L+1 = 5 values
    gamma = 0.9

    scores = ehc.continuation_scores(rewards, values, gamma)

    # Manual per-t computation from the raw (unrolled) definition:
    #   S(C_t) = sum_{k=t}^{L-1} gamma^(k-t) r_k + gamma^(L-t) V(s_L) - V(s_t)
    L = len(rewards)
    for t in range(L):
        expected = sum(gamma ** (k - t) * rewards[k] for k in range(t, L))
        expected += gamma ** (L - t) * values[L]
        expected -= values[t]
        assert scores[t] == pytest.approx(expected, abs=1e-9), f"mismatch at t={t}"


def test_continuation_scores_requires_L_plus_1_values():
    with pytest.raises(AssertionError):
        ehc.continuation_scores(np.zeros(4), np.zeros(4), 0.99)  # missing the +1 endpoint value


def test_make_hold_action_zeros_displacement_keeps_gripper():
    action = ehc.make_hold_action(gripper_command=0.73, act_dim=4)
    assert action.shape == (4,)
    np.testing.assert_allclose(action[:3], 0.0)
    assert action[3] == pytest.approx(0.73)


def test_gripper_command_at_uses_previous_action_except_at_t0():
    action_i = np.array([[0, 0, 0, 0.1], [0, 0, 0, 0.5], [0, 0, 0, -0.9]], dtype=np.float32)
    # t=0: no prior action inside the segment -> falls back to action[0]'s gripper command.
    assert ehc.gripper_command_at(action_i, 0) == pytest.approx(0.1)
    # t>0: the action that was actually applied to reach s_t is action[t-1].
    assert ehc.gripper_command_at(action_i, 1) == pytest.approx(0.1)
    assert ehc.gripper_command_at(action_i, 2) == pytest.approx(0.5)


def test_max_gap_ignores_none_entries():
    assert ehc.max_gap([(0, 0.5), (1, None), (2, 1.5), (3, -0.3)]) == pytest.approx(1.5)
    assert ehc.max_gap([(0, None)]) == float("-inf")
    assert ehc.max_gap([]) == float("-inf")


def test_first_stop_is_the_first_strict_crossing_in_chronological_order():
    """
    Spec Section 8: tau = min{t : Delta_t > c}, evaluated in chronological
    order, strictly >. A later, larger gap must not be picked over an earlier
    crossing, and a gap exactly equal to the threshold must not count.
    """
    times = [0, 1, 2, 3, 4]
    gaps = [-2.0, 0.4, 1.0, 1.0, 5.0]  # candidate at t=2 crosses first at c=0.9
    tau = first_stop(times, gaps, threshold=0.9)
    assert tau == 2

    # Exactly-equal-to-threshold must NOT trigger a stop (spec: "Strictly Delta_t > c").
    tau_eq = first_stop(times, gaps, threshold=1.0)
    assert tau_eq == 4  # only the last entry (5.0) strictly exceeds 1.0

    # No crossing at all -> censored.
    assert first_stop(times, gaps, threshold=10.0) is None


def test_threshold_for_rate_never_goes_negative_and_respects_p_max():
    """
    Spec: "with a nonnegative threshold c >= 0, segments with M_i <= 0 cannot
    produce a stop" -- so the achievable rate is capped at p_max = P(M_i > 0),
    and threshold_for_rate must never return a negative c to force a higher rate.
    """
    max_gaps = np.array([-2, -0.5, 0, 0.2, 0.6, 1, 1.8, 3, 5, 8], dtype=np.float64)
    p_max = float(np.mean(max_gaps > 0.0))

    # Even asking for a rate above p_max must not go negative.
    c, actual_rate = threshold_for_rate(max_gaps, target_rate=1.0)
    assert c >= 0.0
    assert actual_rate <= p_max + 1e-9

    # A rate comfortably within p_max should roughly match the requested fraction.
    c2, actual_rate2 = threshold_for_rate(max_gaps, target_rate=0.2)
    assert c2 >= 0.0
    assert actual_rate2 == pytest.approx(0.2, abs=0.15)


# ---------------------------------------------------------------------------
# Part B: real pool + real oracle (CHPC only)
# ---------------------------------------------------------------------------

DEFAULT_POOL_PATH = os.environ.get(
    "ESTOP_HOLD_POOL_PATH",
    "/scratch/general/vast/u1472210/mw_de_pool/mw_drawer-open-v2/pool.npz",
)
DEFAULT_ORACLE_DIR = os.environ.get(
    "ESTOP_HOLD_ORACLE_DIR",
    "runs/runs/chpc/oracle_sac_seeds/mw_drawer-open-v2/seed-1",
)

_CHPC_DATA_AVAILABLE = os.path.exists(DEFAULT_POOL_PATH) and os.path.isdir(DEFAULT_ORACLE_DIR)

_SKIP_REASON = (
    "Needs a real pool.npz + oracle checkpoint (CHPC only). "
    f"Looked for POOL_PATH={DEFAULT_POOL_PATH!r} ORACLE_DIR={DEFAULT_ORACLE_DIR!r} -- "
    "override with ESTOP_HOLD_POOL_PATH / ESTOP_HOLD_ORACLE_DIR env vars."
)


@pytest.mark.skipif(not _CHPC_DATA_AVAILABLE, reason=_SKIP_REASON)
def test_restoration_reproduces_recorded_transitions():
    """
    Spec Section 2: "Before generating feedback, verify restoration: restore a
    snapshot, execute the recorded next action, and check that the resulting
    observation and reward match the dataset within numerical tolerance."
    """
    oracle, env = ehc.load_model(DEFAULT_ORACLE_DIR, os.path.join(DEFAULT_ORACLE_DIR, "best_model.pt"), "cpu")
    with open(DEFAULT_POOL_PATH, "rb") as f:
        pool = np.load(f)
        pool_obs, pool_action, pool_reward, pool_state = (
            pool["obs"], pool["action"], pool["reward"], pool["state"],
        )

    N, T, _ = pool_obs.shape
    rng = np.random.default_rng(0)
    for i in rng.choice(N, size=min(5, N), replace=False):
        for t in rng.choice(T - 1, size=min(3, T - 1), replace=False):
            obs = ehc.restore_state(env, pool_state[i, t])
            np.testing.assert_allclose(obs, pool_obs[i, t], atol=1e-4, rtol=1e-4)
            next_obs, reward, _done = ehc.env_step(env, pool_action[i, t])
            np.testing.assert_allclose(next_obs, pool_obs[i, t + 1], atol=1e-3, rtol=1e-3)
            assert reward == pytest.approx(float(pool_reward[i, t]), abs=1e-3)


@pytest.mark.skipif(not _CHPC_DATA_AVAILABLE, reason=_SKIP_REASON)
def test_evaluate_segment_invariants_on_real_data():
    """
    Spec Section 14's core gates for every emitted pair:
      - Delta_tau > threshold
      - every earlier evaluated t satisfies Delta_t <= threshold
      - both suffixes share the same start state and the same length (horizon)
      - full-segment score difference equals gamma^tau * Delta_tau (prefix cancellation)
    """
    oracle, env = ehc.load_model(DEFAULT_ORACLE_DIR, os.path.join(DEFAULT_ORACLE_DIR, "best_model.pt"), "cpu")
    with open(DEFAULT_POOL_PATH, "rb") as f:
        pool = np.load(f)
        pool_obs, pool_action, pool_reward, pool_state = (
            pool["obs"], pool["action"], pool["reward"], pool["state"],
        )

    N, T, _ = pool_obs.shape
    threshold = 0.5
    gamma = 0.99
    n_checked_stops = 0

    rng = np.random.default_rng(1)
    for i in rng.choice(N, size=min(10, N), replace=False):
        result = ehc.evaluate_segment(
            env, oracle, pool_obs[i], pool_action[i], pool_reward[i], pool_state[i],
            gamma, mcmc_samples=16, device="cpu", min_horizon=5, threshold=threshold, stop_early=True,
        )

        gaps = result["gaps"]
        if not result["stopped"]:
            assert all(d is None or d <= threshold for _, d in gaps)
            continue

        n_checked_stops += 1
        tau = result["stop_index"]
        delta_tau = result["oracle_gap"]
        assert delta_tau > threshold

        earlier = [d for t, d in gaps if t < tau]
        assert all(d is None or d <= threshold for d in earlier)

        horizon = result["horizon"]
        assert horizon == T - tau
        assert result["positive"]["obs"].shape[0] == horizon
        assert result["negative"]["obs"].shape[0] == horizon
        np.testing.assert_allclose(result["positive"]["obs"][0], pool_obs[i, tau], atol=1e-4, rtol=1e-4)
        np.testing.assert_allclose(result["negative"]["obs"][0], pool_obs[i, tau], atol=1e-6, rtol=1e-6)

        # Full-segment score-difference identity: the common prefix [0, tau)
        # telescopes out exactly, leaving gamma^tau * Delta_tau (see this
        # file's module docstring for the derivation).
        final_obs, _, _ = ehc.reconstruct_final_obs(env, pool_state[i, T - 1], pool_action[i, T - 1])
        all_obs = np.concatenate([pool_obs[i], final_obs[None]], axis=0)
        values = ehc.oracle_state_values(all_obs, oracle, mcmc_samples=16, device="cpu")
        cont_scores = ehc.continuation_scores(pool_reward[i], values, gamma)
        s_full_orig = float(cont_scores[0])
        s_prefix_plus_hold = (
            float(sum(gamma ** k * pool_reward[i, k] for k in range(tau)))
            + gamma ** tau * (delta_tau + float(cont_scores[tau]) + float(values[tau]))
            - float(values[0])
        )
        assert (s_prefix_plus_hold - s_full_orig) == pytest.approx(gamma ** tau * delta_tau, abs=1e-3)

    assert n_checked_stops > 0, "none of the sampled segments produced a stop at threshold=0.5 -- lower it and retry"
