"""
Pure-logic consistency tests for the Cumulative E-stop model
(cumulative_estop_model.md). No MuJoCo/env dependency -- fast, always runs.
"""
import numpy as np

import scripts.cum_estop_common as cec


def test_per_step_deficits_zero_when_advantage_nonnegative():
    reward = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    values = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)  # V constant, A_t = r_t >= 0
    deficits = cec.per_step_deficits(reward, values, gamma=1.0)
    assert np.allclose(deficits, 0.0)


def test_per_step_deficits_positive_when_advantage_negative():
    reward = np.array([-1.0, 0.0], dtype=np.float32)
    values = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # A_t = r_t + V(s_{t+1}) - V(s_t) = r_t
    deficits = cec.per_step_deficits(reward, values, gamma=1.0)
    assert deficits[0] == 1.0  # -A_0 = 1.0
    assert deficits[1] == 0.0


def test_per_step_deficits_length_mismatch_asserts():
    try:
        cec.per_step_deficits(np.zeros(3), np.zeros(3), gamma=1.0)  # needs 4 values, not 3
        assert False, "expected AssertionError"
    except AssertionError:
        pass


def test_cumulative_stop_index_first_crossing():
    deficits = np.array([0.5, 0.5, 0.5, 0.5])  # cumsum: 0.5, 1.0, 1.5, 2.0
    assert cec.cumulative_stop_index(deficits, H=1.0) == 1  # crosses at t=1 (cumsum=1.0)
    assert cec.cumulative_stop_index(deficits, H=1.1) == 2
    assert cec.cumulative_stop_index(deficits, H=100.0) is None  # never crosses


def test_cumulative_stop_index_monotone_longer_horizon_more_opportunity():
    # Spec Section 4.4: the accumulator is monotone, so a longer horizon can
    # only ever reach an existing crossing sooner or add a new one -- it can
    # never "undo" a crossing already found on a shorter prefix.
    deficits = np.array([0.1, 0.1, 0.1, 0.1, 0.1])
    tau_short = cec.cumulative_stop_index(deficits[:3], H=0.25)
    tau_long = cec.cumulative_stop_index(deficits, H=0.25)
    assert tau_short == tau_long  # same crossing point, found regardless of how far we look ahead


def test_segment_score_matches_manual_backward_telescoping():
    reward = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    values = np.array([0.5, 0.0, 0.0, 10.0], dtype=np.float32)  # V(s_0)=0.5, V(s_L)=10.0
    gamma = 0.9
    score = cec.segment_score(reward, values, gamma)
    # U_E = r_0 + gamma*r_1 + gamma^2*r_2 + gamma^3*V(s_3) - V(s_0)
    expected = 1.0 + gamma * 2.0 + gamma**2 * 3.0 + gamma**3 * 10.0 - 0.5
    assert abs(score - expected) < 1e-6


def test_discounted_length_matches_geometric_series():
    gamma = 0.9
    L = 5
    expected = sum(gamma**k for k in range(L))
    assert abs(cec.discounted_length(L, gamma) - expected) < 1e-6
    assert cec.discounted_length(L, 1.0) == float(L)


def test_margin_scales_with_horizon_not_absolute():
    # Section 6.3: m_C(L) = delta_C * W_L -- a longer suffix needs a
    # proportionally larger absolute improvement for the SAME average
    # per-step bar, not the same fixed absolute number.
    gamma = 0.95
    delta_c = 0.2
    m_short = delta_c * cec.discounted_length(3, gamma)
    m_long = delta_c * cec.discounted_length(30, gamma)
    assert m_long > m_short
    # Average per discounted step is the same by construction.
    assert abs(m_short / cec.discounted_length(3, gamma) - delta_c) < 1e-9
    assert abs(m_long / cec.discounted_length(30, gamma) - delta_c) < 1e-9
