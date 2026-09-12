"""
K=2 degenerate check for the reward-model baseline (research/algs/reward_cpl.py).

At K=2 with contrastive_bias=1.0, demo_cross_entropy's K-way softmax reduces
exactly to a standard Bradley-Terry BCE: -log sigmoid(r0 - r1). This exercises
the (B, K, T, ...) -> (B*K, T, ...) -> reward-net -> (B, K) reshape pipeline in
score_segments/RewardCPL at the smallest nontrivial K, catching indexing/reshape
bugs before trusting it at K=7 (demo) or C~31-53 (credit assignment).

This test is self-contained (synthetic obs/action, no real *_labels.npz, no
MetaWorld/mujoco) so it runs anywhere torch is installed. It does not replace
running `scripts/generate_demo_labels.py --n-counterfactuals 1` on a real pool
(which requires the oracle/MetaWorld machinery only available on the cluster)
but validates the same reshape/loss-equivalence claim at the tensor level.

Run with:  pytest tests/test_reward_cpl_k2.py -v
"""

import gym
import numpy as np
import torch

from research.algs.cpl import demo_cross_entropy
from research.algs.reward_cpl import RewardCPL
from research.networks.base import RewardPolicy
from research.networks.mlp import ContinuousMLPCritic


def make_model(obs_dim=8, act_dim=3, seed=0):
    torch.manual_seed(seed)
    obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
    act_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32)
    model = RewardCPL(
        obs_space,
        act_space,
        network_class=RewardPolicy,
        dataset_class=None,
        network_kwargs=dict(
            reward_class=ContinuousMLPCritic,
            reward_kwargs=dict(ensemble_size=1),
            hidden_layers=[32, 32],
        ),
        optim_class=torch.optim.Adam,
        optim_kwargs=dict(lr=3e-4),
        contrastive_bias=1.0,
        discount=1.0,
        device="cpu",
    )
    model.eval()
    return model


def test_k2_matches_hand_written_bradley_terry():
    model = make_model()
    B, K, T, obs_dim, act_dim = 17, 2, 5, 8, 3
    torch.manual_seed(1)
    obs = torch.randn(B, K, T, obs_dim)
    action = torch.rand(B, K, T, act_dim) * 2 - 1  # in [-1, 1]
    batch = {"obs": obs, "action": action, "label": torch.zeros(B)}

    with torch.no_grad():
        loss, accuracy, seg_score = model._get_reward_loss(batch)

    assert seg_score.shape == (B, K)

    # Independently hand-compute Bradley-Terry BCE on the SAME per-segment scores.
    r0, r1 = seg_score[:, 0], seg_score[:, 1]
    hand_loss = -torch.nn.functional.logsigmoid(r0 - r1).mean()
    hand_accuracy = (r0 > r1).float().mean()

    assert torch.allclose(loss, hand_loss, atol=1e-6), (loss.item(), hand_loss.item())
    assert torch.allclose(accuracy, hand_accuracy)

    # And confirm it matches calling demo_cross_entropy directly on the same scores
    # (the function RewardCPL actually calls internally at bias=1.0, K=2).
    demo_loss, demo_accuracy = demo_cross_entropy(seg_score, bias=1.0)
    assert torch.allclose(loss, demo_loss, atol=1e-6)
    assert torch.allclose(accuracy, demo_accuracy)


def test_seg_score_is_permutation_sensitive_but_scorer_is_shared():
    # Sanity: swapping the two arms should swap which one "wins" -- confirms the
    # (B, K, T) reshape isn't silently transposing or aliasing arms.
    model = make_model(seed=2)
    B, K, T, obs_dim, act_dim = 9, 2, 4, 8, 3
    torch.manual_seed(3)
    obs = torch.randn(B, K, T, obs_dim)
    action = torch.rand(B, K, T, act_dim) * 2 - 1
    batch = {"obs": obs, "action": action, "label": torch.zeros(B)}
    batch_swapped = {"obs": obs.flip(1), "action": action.flip(1), "label": torch.zeros(B)}

    with torch.no_grad():
        _, _, seg_score = model._get_reward_loss(batch)
        _, _, seg_score_swapped = model._get_reward_loss(batch_swapped)

    assert torch.allclose(seg_score[:, 0], seg_score_swapped[:, 1], atol=1e-6)
    assert torch.allclose(seg_score[:, 1], seg_score_swapped[:, 0], atol=1e-6)


if __name__ == "__main__":
    test_k2_matches_hand_written_bradley_terry()
    test_seg_score_is_permutation_sensitive_but_scorer_is_shared()
    print("All K=2 degenerate checks passed.")
