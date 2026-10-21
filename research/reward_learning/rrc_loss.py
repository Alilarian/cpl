"""
RRC likelihood for one choice structure:

    R(c)  = sum_t gamma^t r(s_t, a_t)        over option c's grounded segment,
                                               indexed from that segment's own start
    loss  = - sum_c target_c * log softmax_c(beta * R(c))

with target one-hot on ARIC's chosen option c* (uniform over tied options when
ARIC's interpretation yields ties), invalid options masked to -inf, and each
act's structures averaged before averaging over acts (per_act_weights).
"""

from typing import Optional

import numpy as np
import torch


def discount_weights(length: int, gamma: float, device=None, dtype=torch.float32) -> torch.Tensor:
    return gamma ** torch.arange(length, device=device, dtype=dtype)


def segment_returns(r: torch.Tensor, gamma: float, step_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """r: (..., L) per-step rewards -> (...,) discounted sum from each segment's t=0."""
    if step_mask is not None:
        r = r * step_mask.to(r.dtype)
    if gamma == 1.0:
        return r.sum(dim=-1)
    return (r * discount_weights(r.shape[-1], gamma, r.device, r.dtype)).sum(dim=-1)


def window_returns(r: torch.Tensor, k: int, gamma: float) -> torch.Tensor:
    """
    All length-k windows of one per-step reward sequence, from one forward pass:
        R_i = sum_{t=0}^{k-1} gamma^t r_{i+t},   i = 0 .. T-k
    via the discounted prefix sum P_j = sum_{s<j} gamma^s r_s:  R_i = (P_{i+k} - P_i) / gamma^i.
    r: (..., T) -> (..., T-k+1)
    """
    T = r.shape[-1]
    assert 0 < k <= T
    if gamma == 1.0:
        P = torch.nn.functional.pad(torch.cumsum(r, dim=-1), (1, 0))
        return P[..., k:] - P[..., : T - k + 1]
    # Accumulate in float64: dividing by gamma^i amplifies rounding error for large i.
    w = discount_weights(T, gamma, r.device, torch.float64)
    P = torch.nn.functional.pad(torch.cumsum(r.double() * w, dim=-1), (1, 0))
    out = (P[..., k:] - P[..., : T - k + 1]) / w[: T - k + 1]
    return out.to(r.dtype)


def masked_log_softmax(logits: torch.Tensor, option_mask: Optional[torch.Tensor]) -> torch.Tensor:
    if option_mask is not None:
        logits = logits.masked_fill(~option_mask.bool(), float("-inf"))
    return torch.log_softmax(logits, dim=-1)


def rrc_nll(
    scores: torch.Tensor,
    target: torch.Tensor,
    option_mask: Optional[torch.Tensor] = None,
    beta: float = 1.0,
) -> torch.Tensor:
    """
    scores: (..., K) option returns R(c); target: (..., K) soft target summing to 1 over
    valid options; option_mask: (..., K) bool, False for padded options.
    Returns (...,) per-structure negative log-likelihood.
    """
    logp = masked_log_softmax(beta * scores, option_mask)
    # 0 * -inf must contribute 0 for masked options, not NaN.
    contrib = torch.where(target > 0, target * logp, torch.zeros_like(logp))
    return -contrib.sum(dim=-1)


def choice_correct(scores: torch.Tensor, target: torch.Tensor, option_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """1 if the top-scoring valid option is (one of) the chosen option(s), else 0. (...,)"""
    if option_mask is not None:
        scores = scores.masked_fill(~option_mask.bool(), float("-inf"))
    pred = scores.argmax(dim=-1, keepdim=True)
    chosen = target >= target.max(dim=-1, keepdim=True).values - 1e-12
    chosen = (chosen & (target > 0)).expand_as(scores)  # target may be shared across a leading member dim
    return chosen.gather(-1, pred).squeeze(-1).to(scores.dtype)


def per_act_weights(act_ids: np.ndarray) -> np.ndarray:
    """
    Row weights w with sum_rows w * loss_row == mean over acts of (mean over that act's rows),
    so every act counts once regardless of how many choice structures it produced.
    """
    act_ids = np.asarray(act_ids)
    _, inverse, counts = np.unique(act_ids, return_inverse=True, return_counts=True)
    return (1.0 / (counts[inverse] * len(counts))).astype(np.float32)
