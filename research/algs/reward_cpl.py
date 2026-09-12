"""
Reward-model baseline for the CPL feedback-type comparison ("Stage-3 scorer swap").

Everywhere ARIC (CPL/DemoCPL/CreditAssignmentCPL/EstopCPL, see cpl.py) computes a
per-arm score as alpha * sum_t log pi(a_t | s_t) and feeds it into a K-way
contrastive loss, these classes compute the same per-arm score as
sum_t gamma**t * r_theta(s_t, a_t) instead, using the identical loss functions
(demo_cross_entropy / biased_bce_with_* / the e-stop softplus) imported
unmodified from cpl.py. No label generator, filter, or buffer class is touched.

contrastive_bias defaults to 1.0 here (not DemoCPL's 0.5 / CreditAssignmentCPL's
biased default): the bias term is a conservative regularizer that only makes
sense because pi is a normalized distribution (CPL Prop. 2) -- an unnormalized
r_theta can trivially rescale its own output to game a biased loss instead.
Regularize r_theta with l2_coeff if needed and report bias=1.0 as deliberate.
"""

from typing import Any, Dict

import torch
import torch.nn.functional as F

from .base import Algorithm
from .cpl import biased_bce_with_logits, biased_bce_with_scores, demo_cross_entropy
from .scoring import score_segments


class RewardCPLBase(Algorithm):
    def __init__(
        self,
        *args,
        contrastive_bias: float = 1.0,
        discount: float = 1.0,
        l2_coeff: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        assert "reward" in self.network.CONTAINERS, "RewardCPL* requires a network with a 'reward' container."
        assert contrastive_bias > 0.0 and contrastive_bias <= 1.0
        self.contrastive_bias = contrastive_bias
        self.discount = discount
        self.l2_coeff = l2_coeff

    def setup_optimizers(self) -> None:
        self.optim["reward"] = self.optim_class(self.network.reward.parameters(), **self.optim_kwargs)

    def _scorer(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        # network.reward returns (E, N, T); collapse the ensemble dim like piql.py does.
        return self.network.reward(obs, act).mean(dim=0)

    def _score_segments(self, seg_obs: torch.Tensor, seg_act: torch.Tensor) -> torch.Tensor:
        return score_segments(seg_obs, seg_act, self._scorer, discount=self.discount)

    def _l2(self, seg_obs: torch.Tensor, seg_act: torch.Tensor) -> torch.Tensor:
        # Mean squared PER-STEP reward (not the summed segment score, so it doesn't scale with T).
        B, K, T = seg_obs.shape[0], seg_obs.shape[1], seg_obs.shape[2]
        per_step = self._scorer(
            seg_obs.reshape(B * K, T, *seg_obs.shape[3:]),
            seg_act.reshape(B * K, T, *seg_act.shape[3:]),
        )
        return (per_step**2).mean()

    @staticmethod
    def _traj_shift_metrics(seg_score: torch.Tensor) -> Dict[str, float]:
        """
        Per-trajectory mean/variance of r_theta, to make the shift-identifiability
        argument visible: scalar/credit-assignment losses are invariant to adding a
        per-trajectory constant to r_theta (unlike ARIC's normalized log pi), so a
        large traj_r_var for those types (relative to pairwise/demo) is expected.
        """
        with torch.no_grad():
            per_traj_mean = seg_score.mean(dim=1)
            return dict(
                traj_r_mean=per_traj_mean.mean().item(),
                traj_r_var=per_traj_mean.var(unbiased=False).item(),
            )

    def _get_train_action(self, *args, **kwargs):
        raise NotImplementedError("RewardCPL* has no actor and is never rolled out in an env.")


class RewardCPL(RewardCPLBase):
    """
    Reward-baseline analogue of DemoCPL / the base CPL pairwise loss.
    Covers pairwise, demonstrative, corrective, scalar, and sequential e-stop
    feedback -- all of which share either:
      (a) CorrBuffer/DemoBuffer's (B, K, T, ...) format with the preferred/demo
          arm pinned at index 0 (K=2 for corr/scalar/seq_estop, K=7 for demo), or
      (b) FeedbackBuffer/PMFeedbackBuffer's obs_1/obs_2/label pairwise format.
    """

    def setup_datasets(self, env, total_steps: int) -> None:
        super().setup_datasets(env, total_steps)
        # One-time diagnostic: fraction of demo/corr/scalar/seq_estop choice sets where
        # the pinned index-0 arm disagrees with the oracle rl_sum ranking. This is the
        # irreducible loss floor any scorer (ARIC or reward-baseline) can hit on this
        # loss -- report it alongside validation cross-entropy (cf. CPL Fig. 8).
        dataset = getattr(self, "dataset", None)
        reward = getattr(dataset, "reward", None)
        if reward is not None and reward.ndim == 3:
            oracle_sum = reward.sum(axis=-1)  # (N, K)
            pin_is_argmax = oracle_sum[:, 0] == oracle_sum.max(axis=1)
            disagreement_frac = 1.0 - float(pin_is_argmax.mean())
            self._pin_disagreement_frac = disagreement_frac
            print(f"[RewardCPL] pinned-index-0 disagreement with oracle ranking: {disagreement_frac:.4f}")
        else:
            self._pin_disagreement_frac = None

    def _get_reward_loss(self, batch: Dict[str, Any]):
        if isinstance(batch, dict) and "obs_1" in batch:
            # FeedbackBuffer (comparison/rank mode) or PMFeedbackBuffer.
            seg_obs = torch.stack((batch["obs_1"], batch["obs_2"]), dim=1)
            seg_act = torch.stack((batch["action_1"], batch["action_2"]), dim=1)
            seg_score = self._score_segments(seg_obs, seg_act)
            s1, s2 = seg_score[:, 0], seg_score[:, 1]
            loss, accuracy = biased_bce_with_logits(s1, s2, batch["label"].float(), bias=self.contrastive_bias)
        elif isinstance(batch, dict) and "score" in batch:
            # FeedbackBuffer (score mode).
            seg_obs, seg_act = batch["obs"].unsqueeze(1), batch["action"].unsqueeze(1)
            seg_score = self._score_segments(seg_obs, seg_act).squeeze(1)
            loss, accuracy = biased_bce_with_scores(seg_score, batch["score"].float(), bias=self.contrastive_bias)
        else:
            # CorrBuffer / DemoBuffer: (B, K, T, ...), preferred/demo arm pinned at index 0.
            seg_obs, seg_act = batch["obs"], batch["action"]
            seg_score = self._score_segments(seg_obs, seg_act)
            loss, accuracy = demo_cross_entropy(seg_score, bias=self.contrastive_bias)

        total_loss = loss + self.l2_coeff * self._l2(seg_obs, seg_act)
        return total_loss, accuracy, seg_score

    def train_step(self, batch: Dict, step: int, total_steps: int) -> Dict:
        loss, accuracy, _ = self._get_reward_loss(batch)
        self.optim["reward"].zero_grad(set_to_none=True)
        loss.backward()
        self.optim["reward"].step()
        metrics = dict(reward_loss=loss.item(), accuracy=accuracy.item())
        if step == 0 and getattr(self, "_pin_disagreement_frac", None) is not None:
            metrics["pin_disagreement_frac"] = self._pin_disagreement_frac
        return metrics

    def validation_step(self, batch: Any) -> Dict:
        with torch.no_grad():
            loss, accuracy, seg_score = self._get_reward_loss(batch)
        metrics = dict(reward_loss=loss.item(), accuracy=accuracy.item())
        metrics.update(self._traj_shift_metrics(seg_score))
        return metrics


class RewardCreditCPL(RewardCPLBase):
    """
    Reward-baseline analogue of CreditAssignmentCPL. Batches from
    PMCreditAssignmentBuffer: obs/action (B, C, T, ...), label (B,) long = chosen window.
    """

    def _get_reward_loss(self, batch: Dict[str, Any]):
        seg_obs, seg_act = batch["obs"], batch["action"]
        seg_score = self._score_segments(seg_obs, seg_act)  # (B, C)
        chosen = batch["label"].long()

        bias_mask = torch.full_like(seg_score, self.contrastive_bias)
        bias_mask[torch.arange(len(chosen), device=seg_score.device), chosen] = 1.0
        loss = F.cross_entropy(seg_score * bias_mask, chosen)
        loss = loss + self.l2_coeff * self._l2(seg_obs, seg_act)

        with torch.no_grad():
            accuracy = (seg_score.argmax(dim=1) == chosen).float().mean()
        return loss, accuracy, seg_score

    def train_step(self, batch: Dict, step: int, total_steps: int) -> Dict:
        loss, accuracy, _ = self._get_reward_loss(batch)
        self.optim["reward"].zero_grad(set_to_none=True)
        loss.backward()
        self.optim["reward"].step()
        return dict(reward_loss=loss.item(), accuracy=accuracy.item())

    def validation_step(self, batch: Any) -> Dict:
        with torch.no_grad():
            loss, accuracy, seg_score = self._get_reward_loss(batch)
        metrics = dict(reward_loss=loss.item(), accuracy=accuracy.item())
        metrics.update(self._traj_shift_metrics(seg_score))
        return metrics


class RewardEstopCPL(RewardCPLBase):
    """
    Reward-baseline analogue of EstopCPL. Batches from EstopBuffer:
    obs/action (B, 2, T, ...) -- index 0 = halt prefix (padded tail repeats the
    last (s_tau, a_tau)), index 1 = full trajectory. Both sides are length T and
    scored unmasked -- an additive shift in r_theta cancels in the logit
    difference exactly as it does for EstopCPL's alpha * log pi.

    contrastive_bias is meaningless here (forced to 1.0, matching EstopCPL);
    beta_prime plays that role. discount is a real hyperparameter (unlike
    RewardCPL/RewardCreditCPL where it must stay 1.0) -- match EstopCPL's config
    value exactly so the comparison stays discount-matched.
    """

    def __init__(self, *args, beta_prime: float = 1.0, **kwargs):
        super().__init__(*args, contrastive_bias=1.0, **kwargs)
        self.beta_prime = beta_prime

    def _get_reward_loss(self, batch: Dict[str, Any]):
        seg_obs, seg_act = batch["obs"], batch["action"]
        seg_score = self._score_segments(seg_obs, seg_act)  # (B, 2)
        logit = self.beta_prime * (seg_score[:, 0] - seg_score[:, 1])
        loss = F.softplus(-logit).mean()
        loss = loss + self.l2_coeff * self._l2(seg_obs, seg_act)

        with torch.no_grad():
            accuracy = (logit > 0).float().mean()
        return loss, accuracy, seg_score

    def train_step(self, batch: Dict, step: int, total_steps: int) -> Dict:
        loss, accuracy, _ = self._get_reward_loss(batch)
        self.optim["reward"].zero_grad(set_to_none=True)
        loss.backward()
        self.optim["reward"].step()
        return dict(reward_loss=loss.item(), accuracy=accuracy.item())

    def validation_step(self, batch: Any) -> Dict:
        with torch.no_grad():
            loss, accuracy, seg_score = self._get_reward_loss(batch)
        metrics = dict(reward_loss=loss.item(), accuracy=accuracy.item())
        metrics.update(self._traj_shift_metrics(seg_score))
        return metrics
