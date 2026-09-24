from typing import Callable, Optional

import torch


def score_segments(
    seg_obs: torch.Tensor,
    seg_act: torch.Tensor,
    scorer: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    discount: float = 1.0,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Shared (B, K, T, ...) -> (B, K) segment scoring step used by both ARIC's
    policy-score CPL losses and the reward-model baseline. The only thing
    that differs between the two arms is what `scorer` computes per step:
    ARIC uses alpha * log pi(a|s), the reward baseline uses r_theta(s, a).

    Args:
        seg_obs: (B, K, T, *obs_shape)
        seg_act: (B, K, T, act_dim)
        scorer:  callable (N, T, *obs_shape) x (N, T, act_dim) -> (N, T) per-step score
        discount: gamma applied as gamma ** t before summing over T (1.0 = flat sum),
                  discounted from t=0 regardless of any masking (i.e. from the start
                  of whatever segment/suffix is passed in, per ARIC E-stop spec Section 11:
                  "discount from the suffix start").
        mask:    optional (B, K, T) or (B, 1, T) bool/float mask, 1 for real (unpadded)
                 timesteps and 0 for padding beyond a per-sample horizon (e.g. the
                 holding-model E-stop's variable-length suffixes, EstopHoldBuffer).
                 Broadcasts across K when given as (B, 1, T). None = no masking
                 (all T steps are real), the original unmasked behaviour.

    Returns:
        (B, K) discounted, masked segment score (summed over valid timesteps only)
    """
    B, K, T = seg_obs.shape[0], seg_obs.shape[1], seg_obs.shape[2]
    assert seg_act.shape[0] == B and seg_act.shape[1] == K and seg_act.shape[2] == T, (
        "obs/action segment shape mismatch: " + str(seg_obs.shape) + " vs " + str(seg_act.shape)
    )
    obs_flat = seg_obs.reshape(B * K, T, *seg_obs.shape[3:])
    act_flat = seg_act.reshape(B * K, T, *seg_act.shape[3:])
    per_step = scorer(obs_flat, act_flat)
    assert per_step.shape == (B * K, T), "scorer must return (N, T), got " + str(per_step.shape)
    per_step = per_step.reshape(B, K, T)
    if mask is not None:
        per_step = per_step * mask.to(per_step.dtype)
    if discount == 1.0:
        return per_step.sum(dim=-1)
    discounts = discount ** torch.arange(T, device=per_step.device, dtype=per_step.dtype)
    return (per_step * discounts).sum(dim=-1)
