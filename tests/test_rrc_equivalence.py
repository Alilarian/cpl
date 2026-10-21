"""
RRC loss == ARIC's own scoring/loss code with r_theta in place of alpha*log pi, at bias = 1
(exactly what research/algs/reward_cpl.py computes), and choice structures == ARIC's buffers.
"""
import os

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from research.algs.cpl import demo_cross_entropy
from research.algs.scoring import score_segments
from research.reward_learning.ensemble import EnsembleMLP
from research.reward_learning.rrc_loss import rrc_nll, segment_returns, window_returns

D, A = 35, 4


def _scorer():
    net = EnsembleMLP(D, A, ensemble_size=1, hidden_layers=(16, 16), seeds=[0])
    return net, (lambda o, a: net(o, a)[0])  # (N, T) per-step reward of the single member


@pytest.mark.parametrize("K,T,gamma", [(2, 64, 1.0), (9, 64, 1.0), (2, 16, 1.0), (2, 64, 0.99)])
def test_stacked_types_match_reward_cpl(K, T, gamma):
    # pref / corr / scalar (K=2, gamma 1), demo (K=9), cum_estop (K=2, DiscountedCorrCPL gamma 0.99)
    torch.manual_seed(K + T)
    net, scorer = _scorer()
    obs, act = torch.randn(6, K, T, D), torch.rand(6, K, T, A) * 2 - 1
    aric = demo_cross_entropy(score_segments(obs, act, scorer, discount=gamma), bias=1.0)[0]
    ours = rrc_nll(segment_returns(net(obs, act)[0], gamma), F.one_hot(torch.zeros(6, dtype=torch.long), K).float()).mean()
    assert torch.allclose(aric, ours, atol=1e-5)


def test_credit_matches_reward_credit_cpl():
    torch.manual_seed(3)
    net, scorer = _scorer()
    T, k = 64, 20
    C = T - k + 1
    base_o, base_a = torch.randn(5, T, D), torch.rand(5, T, A) * 2 - 1
    win_o = torch.stack([base_o[:, i : i + k] for i in range(C)], 1)  # what PMCreditAssignmentBuffer yields
    win_a = torch.stack([base_a[:, i : i + k] for i in range(C)], 1)
    chosen = torch.tensor([0, 7, 44, 13, 20])
    aric = F.cross_entropy(score_segments(win_o, win_a, scorer, discount=1.0), chosen)  # RewardCreditCPL, bias 1
    ours = rrc_nll(window_returns(net(base_o, base_a)[0], k, 1.0), F.one_hot(chosen, C).float()).mean()
    assert torch.allclose(aric, ours, atol=1e-5)


# ---------------------------------------------------------------------------- real data (local mirrors)

ENV = "mw_button-press-v2"
ROOT = f"datasets/mw_de_labels/{ENV}"
HAVE_DATA = os.path.exists(f"{ROOT}/pool.npz")


@pytest.fixture(scope="module")
def pool():
    from research.reward_learning.data import PoolIndex

    return PoolIndex(f"{ROOT}/pool.npz", expected_episodes=200)


@pytest.mark.skipif(not HAVE_DATA, reason="local label mirrors not present")
@pytest.mark.parametrize("ftype,buffer,fname", [("pref", "CorrBuffer", "pref_labels.npz"), ("corr", "CorrBuffer", "corr_labels.npz"),
                                                 ("demo", "DemoBuffer", "demo_labels_K9.npz"), ("scalar", "CorrBuffer", "scalar_labels.npz")])
def test_choice_structures_equal_aric_buffers(pool, ftype, buffer, fname):
    import gym
    from research import datasets
    from research.reward_learning.data import load_choice_set, load_rrc_config

    cfg = load_rrc_config(f"configs/rrc_sac/{ftype}.yaml", ["configs/rrc_sac/local.yaml"])
    cs = load_choice_set(cfg, ENV, pool)
    space = gym.spaces.Box(-np.inf, np.inf, (D,), np.float32)
    n = cfg["data"]["subsample_n"]
    buf = getattr(datasets, buffer)(space, space, path=f"{ROOT}/{fname}", split="all", subsample_n=min(n, len(cs)), subsample_seed=0)
    sample = np.random.RandomState(0).choice(len(cs), 20, replace=False)
    assert np.array_equal(cs.obs[sample], buf.obs[sample]) and np.array_equal(cs.act[sample], buf.action[sample])
    assert np.array_equal(cs.true_reward[sample], buf.reward[sample])
    assert np.all(cs.target[sample].argmax(1) == 0)  # ARIC pins the preferred / demo arm at index 0
    assert cs.gamma == 1.0  # DemoCPL scores with an undiscounted sum
    # the stored labeler scores rank index 0 first, as ARIC's interpretation assumes
    assert np.all(cs.adv_scores[sample].argmax(1) == 0)


@pytest.mark.skipif(not HAVE_DATA, reason="local label mirrors not present")
def test_credit_choice_structures_equal_stored_windows(pool):
    from research.reward_learning.data import aric_subsample_indices, load_choice_set, load_rrc_config, npz_shape, read_npz_rows

    cfg = load_rrc_config("configs/rrc_sac/credit.yaml", ["configs/rrc_sac/local.yaml"])
    cs = load_choice_set(cfg, ENV, pool)
    path = f"{ROOT}/credit_labels.npz"
    rows_all = aric_subsample_indices(npz_shape(path, "obs")[0], 10000, 0)
    sample = np.sort(np.random.RandomState(0).choice(len(cs), 20, replace=False))
    win = read_npz_rows(path, "obs", rows_all[sample])
    k = cs.window
    for j, i in enumerate(sample):
        for c in (0, 17, 44):
            assert np.array_equal(cs.obs[i, c : c + k], win[j, c])
    with np.load(path) as f:
        chosen = f["chosen_idx"][rows_all[sample]]
    assert np.array_equal(cs.target[sample].argmax(1), chosen)
