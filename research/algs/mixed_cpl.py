import itertools
from typing import Any, Dict, Optional

import torch

from research.utils import utils

from .cpl import CPL, biased_choice_cross_entropy


class MixedCPL(CPL):
    """
    CPL trained jointly on an arbitrary mix of feedback types.

    Every feedback type in this codebase already reduces to the same "choice
    set" tensor contract (see research/datasets/mixed_buffer.py's docstring):

        obs    : (B, K, T, obs_dim)
        action : (B, K, T, act_dim)
        label  : (B,)  long, index of the preferred/chosen candidate

    demo/pref/corr/scalar always set label=0; credit_assignment sets a
    variable per-row chosen index. Because the contract is shared, one batch
    can contain several *named* sub-batches -- one per feedback type -- each
    with its own K, and each scored by the exact same
    `biased_choice_cross_entropy` used by DemoCPL/CreditAssignmentCPL. Adding
    a new feedback type to a mix is therefore a config change (another
    dataset component + weight), not new code.

    Expects batches shaped like:
        {"credit_assignment": {"obs": ..., "action": ..., "label": ...},
         "demo":              {"obs": ..., "action": ..., "label": ...}}
    as produced by research.datasets.mixed_buffer.MixedFeedbackBuffer.

    Per-component losses are combined as a fixed-weight sum (not a
    data-count-weighted average): `component_weights` is a free hyperparameter
    per component name, independent of how large that component's dataset/
    batch slice is. This lets a component's influence on the gradient be
    tuned separately from its data share -- e.g. starting a secondary
    feedback type at component_weights=0.1 to match a 10%-of-rows mix, then
    sweeping it if that turns out too weak a signal. Any component name not
    present in `component_weights` defaults to weight 1.0 (so the majority/
    "base" component of a mix usually doesn't need to be listed explicitly).

    Args:
        contrastive_bias  : shared bias applied to every component's loss
                             (default 0.75)
        component_weights : Dict[str, float], loss weight per component name
                             (default {} -> every component weight 1.0)
        bc_steps, bc_coeff, alpha : same meaning as in CPL/DemoCPL/
                             CreditAssignmentCPL; bc_loss is combined across
                             components the same way as the main loss.
    """

    def __init__(
        self,
        *args,
        contrastive_bias: float = 0.75,
        component_weights: Optional[Dict[str, float]] = None,
        bc_steps: int = 0,
        bc_coeff: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(
            *args,
            contrastive_bias=contrastive_bias,
            bc_steps=bc_steps,
            bc_coeff=bc_coeff,
            **kwargs,
        )
        self.component_weights = {} if component_weights is None else dict(component_weights)

    def _get_component_loss(self, sub_batch: Dict):
        obs, action = sub_batch["obs"], sub_batch["action"]
        chosen = sub_batch["label"].long()
        B, K, T, _ = obs.shape

        lp = self._log_prob(obs.reshape(B * K, T, -1), action.reshape(B * K, T, -1))
        lp = lp.reshape(B, K, T)

        bc_loss = -lp[torch.arange(B, device=lp.device), chosen, :].mean()

        seg_adv = self.alpha * lp.sum(dim=-1)  # (B, K)
        loss, accuracy = biased_choice_cross_entropy(seg_adv, chosen, bias=self.contrastive_bias)
        return loss, bc_loss, accuracy

    def _get_mixed_loss(self, batch: Dict[str, Dict]):
        total_loss = 0.0
        total_bc_loss = 0.0
        logs = {}
        for name, sub_batch in batch.items():
            loss_i, bc_loss_i, accuracy_i = self._get_component_loss(sub_batch)
            weight = self.component_weights.get(name, 1.0)
            total_loss = total_loss + weight * loss_i
            total_bc_loss = total_bc_loss + weight * bc_loss_i
            logs[f"{name}_loss"] = loss_i.item()
            logs[f"{name}_accuracy"] = accuracy_i.item()
        return total_loss, total_bc_loss, logs

    def train_step(self, batch: Dict[str, Dict], step: int, total_steps: int) -> Dict:
        mixed_loss, bc_loss, logs = self._get_mixed_loss(batch)

        if step < self.bc_steps:
            loss = bc_loss
        else:
            loss = mixed_loss + self.bc_coeff * bc_loss

        self.optim["actor"].zero_grad()
        loss.backward()
        self.optim["actor"].step()

        if step == self.bc_steps - 1:  # Switch to optimizing the mixed loss here.
            del self.optim["actor"]
            params = itertools.chain(self.network.actor.parameters(), self.network.encoder.parameters())
            groups = utils.create_optim_groups(params, self.optim_kwargs)
            self.optim["actor"] = self.optim_class(groups)
            self.setup_schedulers(do_nothing=False)  # actually start the schedulers.

        return dict(loss=loss.item(), mixed_loss=mixed_loss.item(), bc_loss=bc_loss.item(), **logs)

    def validation_step(self, batch: Any) -> Dict:
        with torch.no_grad():
            mixed_loss, bc_loss, logs = self._get_mixed_loss(batch)
        return dict(mixed_loss=mixed_loss.item(), bc_loss=bc_loss.item(), **logs)
