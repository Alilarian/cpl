"""RRC loss: hand-computed values per feedback type, per-act weighting, masking, discounting."""
import math

import numpy as np
import torch

from research.reward_learning.rrc_loss import (choice_correct, per_act_weights, rrc_nll, segment_returns,
                                               window_returns)


def test_pairwise_bradley_terry_matches_hand_value():
    # pref / corr / scalar / cum_estop: K=2, chosen 0
    scores = torch.tensor([[1.0, 0.2]])
    target = torch.tensor([[1.0, 0.0]])
    expected = math.log(1 + math.exp(-(1.0 - 0.2)))  # -log sigmoid(R0 - R1)
    assert torch.allclose(rrc_nll(scores, target), torch.tensor([expected]))


def test_beta_scales_logits():
    scores = torch.tensor([[1.0, 0.2]])
    target = torch.tensor([[1.0, 0.0]])
    expected = math.log(1 + math.exp(-2.0 * 0.8))
    assert torch.allclose(rrc_nll(scores, target, beta=2.0), torch.tensor([expected]))


def test_kway_demo_softmax_matches_hand_value():
    scores = torch.tensor([[2.0, 1.0, 0.0]])
    target = torch.tensor([[1.0, 0.0, 0.0]])
    expected = -2.0 + math.log(math.exp(2) + math.exp(1) + math.exp(0))
    assert torch.allclose(rrc_nll(scores, target), torch.tensor([expected]), atol=1e-6)


def test_soft_target_for_ties():
    scores = torch.tensor([[1.0, 0.5, -1.0]])
    target = torch.tensor([[0.5, 0.5, 0.0]])
    logp = torch.log_softmax(scores, -1)[0]
    expected = -(0.5 * logp[0] + 0.5 * logp[1])
    assert torch.allclose(rrc_nll(scores, target)[0], expected)
    assert choice_correct(scores, target)[0] == 1.0
    assert choice_correct(torch.tensor([[-1.0, 0.5, 1.0]]), target)[0] == 0.0


def test_option_mask_excludes_padding_and_gives_zero_gradient():
    # Credit assignment with a variable number of windows: padded windows must not matter.
    s_full = torch.tensor([[0.3, 1.2, -0.4, 5.0, 7.0]], requires_grad=True)
    mask = torch.tensor([[True, True, True, False, False]])
    target = torch.tensor([[0.0, 1.0, 0.0, 0.0, 0.0]])
    loss = rrc_nll(s_full, target, mask)
    ref = rrc_nll(s_full[:, :3].detach(), target[:, :3])
    assert torch.allclose(loss.detach(), ref)
    loss.sum().backward()
    assert torch.all(s_full.grad[0, 3:] == 0)
    assert torch.isfinite(s_full.grad).all()


def test_segment_returns_discount_and_mask():
    r = torch.tensor([[1.0, 1.0, 1.0]])
    assert torch.allclose(segment_returns(r, 0.5), torch.tensor([1.75]))
    assert torch.allclose(segment_returns(r, 0.5, torch.tensor([[1.0, 1.0, 0.0]])), torch.tensor([1.5]))
    assert torch.allclose(segment_returns(r, 1.0), torch.tensor([3.0]))


def test_window_returns_equal_brute_force():
    torch.manual_seed(0)
    r = torch.randn(4, 64, dtype=torch.float32)
    for gamma in (1.0, 0.9, 0.99):
        k = 20
        brute = torch.stack([(r[:, i : i + k] * gamma ** torch.arange(k)).sum(1) for i in range(64 - k + 1)], 1)
        assert torch.allclose(window_returns(r, k, gamma), brute, atol=1e-4), gamma


def test_cum_estop_shared_prefix_cancels():
    # Two full arms identical up to tau: the score difference only sees the discounted suffix.
    torch.manual_seed(1)
    L, tau, g = 64, 23, 0.99
    prefix = torch.randn(tau)
    a0 = torch.cat([prefix, torch.randn(L - tau)])
    a1 = torch.cat([prefix, torch.randn(L - tau)])
    R = segment_returns(torch.stack([a0, a1]), g)
    w = g ** torch.arange(L, dtype=torch.float32)
    expected = ((a0 - a1)[tau:] * w[tau:]).sum()
    assert torch.allclose(R[0] - R[1], expected, atol=1e-5)


def test_per_act_weights_every_act_counts_once():
    w = per_act_weights(np.array([0, 0, 1]))
    assert np.allclose(w, [0.25, 0.25, 0.5])
    # duplicating an act's structures leaves the weighted loss unchanged
    losses = np.array([1.0, 3.0])
    base = (per_act_weights(np.array([0, 1])) * losses).sum()
    dup = (per_act_weights(np.array([0, 0, 0, 1])) * np.array([1.0, 1.0, 1.0, 3.0])).sum()
    assert np.isclose(base, dup)
    assert np.isclose(per_act_weights(np.arange(7)).sum(), 1.0)


def test_rrc_nll_broadcasts_over_members():
    scores = torch.randn(3, 5, 4)
    target = torch.zeros(5, 4)
    target[:, 0] = 1
    out = rrc_nll(scores, target)
    assert out.shape == (3, 5)
    for m in range(3):
        assert torch.allclose(out[m], rrc_nll(scores[m], target))
    assert choice_correct(scores, target).shape == (3, 5)
