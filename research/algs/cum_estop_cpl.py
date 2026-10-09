from typing import Dict

import torch

from .cpl import DemoCPL, demo_cross_entropy
from .scoring import score_segments


class DiscountedCorrCPL(DemoCPL):
    """
    CPL for the Cumulative E-stop model (cumulative_estop_model.md), trained
    on the pre-mixed, pre-shuffled 50/50 flat file built by
    scripts/build_cum_estop_mix.py -- a single CorrBuffer-shaped
    (obs, action, reward) K=2 dataset where each row is independently either
    a stop_correction pair or a no_stop_demo pair (not distinguished at
    training time; both get the exact same loss).

    The only behavioral difference from plain DemoCPL: scores with a real
    discount applied from t=0 of the full stored trajectory (DemoCPL/CPL's
    base `_get_demo_loss` is a flat, undiscounted sum). For a
    stop_correction row this makes its shared prefix (up to stop_index,
    identical on both arms) cancel out of the comparison with exactly a
    gamma**tau scale on the diverging suffix (spec Section 9.2) -- with no
    masking or suffix-slicing needed in code, since both arms are always
    full, equal-length T-step trajectories. For a no_stop_demo row there is
    no shared prefix to cancel, but the same global discount is still the
    correct scoring convention per the spec's own J_theta definition
    (Section 9.1) -- applied uniformly, not conditionally, since rows aren't
    tagged by branch at training time.

    Args:
        discount : gamma applied as gamma**t from t=0 (default 0.99).
                   DemoCPL/CPL default to an undiscounted flat sum
                   (discount=1.0 behavior); this is the one difference from
                   the generic base class.
        contrastive_bias, bc_steps, bc_coeff, bc_pool_path, bc_pool_batch_size :
                   same meaning as DemoCPL.
    """

    def __init__(self, *args, discount: float = 0.99, **kwargs):
        super().__init__(*args, **kwargs)
        assert 0.0 < discount <= 1.0, "discount must be in (0, 1]"
        self.discount = discount

    def _policy_scorer(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        obs_enc = self.network.encoder(obs)
        dist = self.network.actor(obs_enc)
        if isinstance(dist, torch.distributions.Distribution):
            lp = dist.log_prob(action)
        else:
            assert dist.shape == action.shape
            lp = -torch.square(dist - action).sum(dim=-1)
        return self.alpha * lp

    def _get_demo_loss(self, batch: Dict):
        obs, action = batch["obs"], batch["action"]  # (B, K, T, ...)

        seg_adv = score_segments(obs, action, self._policy_scorer, discount=self.discount)  # (B, K)

        B, K, T, _ = obs.shape
        obs_enc = self.network.encoder(obs.reshape(B * K, T, -1))
        dist = self.network.actor(obs_enc)
        if isinstance(dist, torch.distributions.Distribution):
            lp = dist.log_prob(action.reshape(B * K, T, -1))
        else:
            lp = -torch.square(dist - action.reshape(B * K, T, -1)).sum(dim=-1)
        lp = lp.reshape(B, K, T)
        bc_loss = -lp[:, 0, :].mean()  # imitate the preferred arm (index 0) only

        demo_loss, accuracy = demo_cross_entropy(seg_adv, bias=self.contrastive_bias)
        return demo_loss, bc_loss, accuracy
