import itertools
from typing import Any, Dict, Optional, Type

import torch
import torch.nn.functional as F

from research.networks.base import ActorCriticValueRewardPolicy

from .cpl import demo_cross_entropy
from .off_policy_algorithm import OffPolicyAlgorithm


def iql_loss(pred, target, expectile=0.5):
    err = target - pred
    weight = torch.abs(expectile - (err < 0).float())
    return weight * torch.square(err)


class PIQL(OffPolicyAlgorithm):
    def __init__(
        self,
        *args,
        tau: float = 0.005,
        target_freq: int = 1,
        expectile: Optional[float] = None,
        beta: float = 1,
        clip_score: float = 100.0,
        reward_steps: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        assert isinstance(self.network, ActorCriticValueRewardPolicy)
        self.tau = tau
        self.target_freq = target_freq
        self.expectile = expectile
        self.beta = beta
        self.clip_score = clip_score
        self.reward_steps = reward_steps

        self.reward_criterion = torch.nn.BCEWithLogitsLoss(reduction="none")

    def setup_network(self, network_class: Type[torch.nn.Module], network_kwargs: Dict) -> None:
        self.network = network_class(
            self.processor.observation_space, self.processor.action_space, **network_kwargs
        ).to(self.device)
        self.target_network = network_class(
            self.processor.observation_space, self.processor.action_space, **network_kwargs
        ).to(self.device)
        self.target_network.load_state_dict(self.network.state_dict())
        for param in self.target_network.parameters():
            param.requires_grad = False

    def setup_optimizers(self) -> None:
        # Default optimizer initialization
        network_keys = ("actor", "critic", "value", "reward")
        default_kwargs = {k: v for k, v in self.optim_kwargs.items() if k not in network_keys}
        assert all([isinstance(self.optim_kwargs.get(k, dict()), dict) for k in network_keys])

        # Update the encoder with the actor. This does better for weighted imitation policy objectives.
        actor_kwargs = default_kwargs.copy()
        actor_kwargs.update(self.optim_kwargs.get("actor", dict()))
        actor_params = itertools.chain(self.network.actor.parameters(), self.network.encoder.parameters())
        self.optim["actor"] = self.optim_class(actor_params, **actor_kwargs)

        critic_kwargs = default_kwargs.copy()
        critic_kwargs.update(self.optim_kwargs.get("critic", dict()))
        self.optim["critic"] = self.optim_class(self.network.critic.parameters(), **critic_kwargs)

        value_kwargs = default_kwargs.copy()
        value_kwargs.update(self.optim_kwargs.get("value", dict()))
        self.optim["value"] = self.optim_class(self.network.value.parameters(), **value_kwargs)

        reward_kwargs = default_kwargs.copy()
        reward_kwargs.update(self.optim_kwargs.get("reward", dict()))
        self.optim["reward"] = self.optim_class(self.network.reward.parameters(), **reward_kwargs)

    def _get_reward_batch(self, batch: Dict):
        """
        Unpacks a comparison batch into (obs, action, discount), each shaped
        (N, S+1, ...), where N packs together every arm/candidate this batch
        carries (2B for pairwise: obs_1/obs_2 concatenated). Subclasses override
        this to support K-way comparison shapes (demo/credit/e-stop/...); the
        rest of train_step is agnostic to what N actually is.
        """
        obs = torch.cat([batch["obs_1"], batch["obs_2"]], dim=0)  # (2B, S+1)
        action = torch.cat([batch["action_1"], batch["action_2"]], dim=0)  # (2B, S+1)
        discount = torch.cat((batch["discount_1"], batch["discount_2"]), dim=0)  # (2B, S+1)
        return obs, action, discount

    def _get_reward_loss_and_accuracy(self, reward: torch.Tensor, batch: Dict):
        """
        reward: (E, N, S+1) raw reward-net predictions over the exact (obs, action)
        _get_reward_batch just returned (same N-ordering). Returns (loss, accuracy)
        for the reward net's own gradient step. Base PIQL: pairwise Bradley-Terry
        BCE over the whole-segment summed return, unchanged from the original.
        """
        r1, r2 = torch.chunk(reward.sum(dim=-1), 2, dim=1)  # Should return two (E, B)
        logits = r2 - r1
        labels = batch["label"].float().unsqueeze(0).expand_as(logits)
        assert labels.shape == logits.shape
        loss = self.reward_criterion(logits, labels).mean()
        with torch.no_grad():
            accuracy = ((r2 > r1) == torch.round(labels)).float().mean()
        return loss, accuracy

    def train_step(self, batch: Dict, step: int, total_steps: int) -> Dict:
        assert isinstance(batch, dict), "P-IQL requires a feedback/comparison batch."
        obs, action, discount = self._get_reward_batch(batch)

        if step < self.reward_steps:
            self.network.reward.train()
            reward = self.network.reward(obs, action)
            reward_loss, reward_accuracy = self._get_reward_loss_and_accuracy(reward, batch)

            self.optim["reward"].zero_grad(set_to_none=True)
            reward_loss.backward()
            self.optim["reward"].step()

            reward = reward.detach().mean(dim=0)
        else:
            with torch.no_grad():
                reward = self.network.reward(obs, action).mean(dim=0)

        # Encode everything
        obs = self.network.encoder(obs)
        next_obs = obs[:, 1:].detach()
        obs = obs[:, :-1]
        action = action[:, :-1]
        discount = discount[:, :-1]
        reward = reward[:, :-1]

        with torch.no_grad():
            target_q = self.target_network.critic(obs, action)
            target_q = torch.min(target_q, dim=0)[0]
        vs = self.network.value(obs.detach())
        v_loss = iql_loss(vs, target_q.unsqueeze(0).expand_as(vs), self.expectile).mean()

        self.optim["value"].zero_grad(set_to_none=True)
        v_loss.backward()
        self.optim["value"].step()

        # Next, update the actor. We detach and use the old value, v for computational efficiency
        # and use the target_q value though the JAX IQL recomputes both
        # Pytorch IQL versions have not.
        with torch.no_grad():
            adv = target_q - torch.mean(vs, dim=0)  # min trick is not used on value.
            exp_adv = torch.exp(adv / self.beta)
            if self.clip_score is not None:
                exp_adv = torch.clamp(exp_adv, max=self.clip_score)

        dist = self.network.actor(obs)  # Use encoder gradients for the actor.
        if isinstance(dist, torch.distributions.Distribution):
            bc_loss = -dist.log_prob(action)
        elif torch.is_tensor(dist):
            assert dist.shape == action.shape
            bc_loss = torch.nn.functional.mse_loss(dist, action, reduction="none").sum(dim=-1)
        else:
            raise ValueError("Invalid policy output provided")
        actor_loss = (exp_adv * bc_loss).mean()

        self.optim["actor"].zero_grad(set_to_none=True)
        actor_loss.backward()
        self.optim["actor"].step()

        # Next, Finally update the critic
        with torch.no_grad():
            next_vs = self.network.value(next_obs)
            next_v = torch.mean(next_vs, dim=0, keepdim=True)  # Min trick is not used on value.
            target = reward + discount * next_v  # use the predicted reward.
        qs = self.network.critic(obs.detach(), action)
        q_loss = torch.nn.functional.mse_loss(qs, target.expand_as(qs), reduction="none").mean()

        self.optim["critic"].zero_grad(set_to_none=True)
        q_loss.backward()
        self.optim["critic"].step()

        metrics = dict(
            q_loss=q_loss.item(),
            v_loss=v_loss.item(),
            actor_loss=actor_loss.item(),
            v=vs.mean().item(),
            q=qs.mean().item(),
            adv=adv.mean().item(),
            reward=reward.mean().item(),
        )

        if step < self.reward_steps:
            metrics["reward_loss"] = reward_loss.item()
            metrics["reward_accuracy"] = reward_accuracy.item()

        # Update the networks. These are done in a stack to support different grad options for the encoder.
        if step % self.target_freq == 0:
            with torch.no_grad():
                # Only run on the critic and encoder, those are the only weights we update.
                for param, target_param in zip(
                    self.network.critic.parameters(), self.target_network.critic.parameters()
                ):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return metrics

    def validation_step(self, batch: Dict) -> Dict:
        # Compute the loss
        if isinstance(batch, (tuple, list)) and "label" in batch[1]:
            feedback_batch = batch[1]
        elif isinstance(batch, dict):
            feedback_batch = batch
        else:
            return dict()
        with torch.no_grad():
            obs, action, _ = self._get_reward_batch(feedback_batch)
            reward = self.network.reward(obs, action)
            reward_loss, reward_accuracy = self._get_reward_loss_and_accuracy(reward, feedback_batch)
        return dict(
            reward_loss=reward_loss.item(), reward_accuracy=reward_accuracy.item(), reward=reward.mean().item()
        )

    def _get_train_action(self, obs: Any, step: int, total_steps: int):
        batch = dict(obs=obs)
        with torch.no_grad():
            action = self.predict(batch, is_batched=False, sample=True)
        return action


class DemoPIQL(PIQL):
    """
    P-IQL generalized to demo/correction/scalar/seq-e-stop/pairwise-shaped K-way
    comparison data: DemoBuffer ((B,K,T,...), K=7, expert pinned at index 0) or
    CorrBuffer ((B,2,T,...), preferred arm pinned at index 0 -- corr/scalar/
    seq_estop/pref all share this K=2 layout). Only the reward loss changes,
    from PIQL's hardcoded 2-arm Bradley-Terry BCE to the K-way demo_cross_entropy
    (identical to unbiased Bradley-Terry BCE at K=2, contrastive_bias=1.0 -- see
    tests/test_reward_cpl_k2.py). Everything downstream of the reward prediction
    (value/actor/critic loss, target updates) is untouched from PIQL, just now
    running over all K arms' transitions instead of always exactly 2.
    """

    def __init__(self, *args, contrastive_bias: float = 1.0, transition_discount: float = 0.99, **kwargs):
        super().__init__(*args, **kwargs)
        assert contrastive_bias > 0.0 and contrastive_bias <= 1.0
        self.contrastive_bias = contrastive_bias
        # DemoBuffer/CorrBuffer carry no per-sample discount field (unlike FeedbackBuffer's
        # discount_1/discount_2) -- match PIQL's flat dataset-level discount convention directly.
        self.transition_discount = transition_discount

    def _get_reward_batch(self, batch: Dict):
        B, K, T, _ = batch["obs"].shape
        obs = batch["obs"].reshape(B * K, T, -1)
        action = batch["action"].reshape(B * K, T, -1)
        discount = self.transition_discount * torch.ones(B * K, T, device=obs.device, dtype=obs.dtype)
        return obs, action, discount

    def _get_reward_loss_and_accuracy(self, reward: torch.Tensor, batch: Dict):
        B, K, T, _ = batch["obs"].shape
        seg_adv = reward.mean(dim=0).sum(dim=-1).reshape(B, K)
        return demo_cross_entropy(seg_adv, bias=self.contrastive_bias)


class CreditAssignmentPIQL(PIQL):
    """
    P-IQL generalized to credit-assignment data: PMCreditAssignmentBuffer
    ((B, C, T, ...), C ~ 31-53 candidate windows, chosen_idx label). Reward loss
    is the same biased cross-entropy CreditAssignmentCPL/RewardCreditCPL use.
    """

    def __init__(self, *args, contrastive_bias: float = 1.0, transition_discount: float = 0.99, **kwargs):
        super().__init__(*args, **kwargs)
        assert contrastive_bias > 0.0 and contrastive_bias <= 1.0
        self.contrastive_bias = contrastive_bias
        self.transition_discount = transition_discount

    def _get_reward_batch(self, batch: Dict):
        B, C, T, _ = batch["obs"].shape
        obs = batch["obs"].reshape(B * C, T, -1)
        action = batch["action"].reshape(B * C, T, -1)
        discount = self.transition_discount * torch.ones(B * C, T, device=obs.device, dtype=obs.dtype)
        return obs, action, discount

    def _get_reward_loss_and_accuracy(self, reward: torch.Tensor, batch: Dict):
        B, C, T, _ = batch["obs"].shape
        seg_adv = reward.mean(dim=0).sum(dim=-1).reshape(B, C)
        chosen = batch["label"].long()
        bias_mask = torch.full_like(seg_adv, self.contrastive_bias)
        bias_mask[torch.arange(B, device=seg_adv.device), chosen] = 1.0
        loss = F.cross_entropy(seg_adv * bias_mask, chosen)
        with torch.no_grad():
            accuracy = (seg_adv.argmax(dim=1) == chosen).float().mean()
        return loss, accuracy


class EstopHoldPIQL(PIQL):
    """
    P-IQL generalized to holding-model e-stop data: EstopHoldBuffer
    ((B,2,T,...), index 0 = hold suffix H_tau (preferred), index 1 = original
    suffix C_tau, both stored ZERO-padded to T with a shared real length
    batch["horizon"]).

    Base PIQL.train_step doesn't just feed _get_reward_batch's (obs, action)
    to the reward net -- it also treats each row as a literal length-T
    transition SEQUENCE for value/critic/actor training via the obs[:, :-1] /
    obs[:, 1:] shift (see PIQL.train_step). Feeding the zero-padded tail
    through unchanged would inject synthetic "teleport to the zero vector"
    transitions into value/critic/actor training. _get_reward_batch therefore
    rewrites the padded tail to repeat each row's last REAL (s, a) before
    anything touches the network -- including the reward net's own forward
    pass, which otherwise would be queried on out-of-distribution zero
    vectors. The reward loss itself is unaffected: _get_reward_loss_and_accuracy
    masks by batch["horizon"] regardless of how the tail was filled in.

    reward_discount (the reward loss's own J_pi discount, matching
    EstopHoldCPL's `discount`) and transition_discount (the flat per-
    transition discount fed to the IQL Bellman target, matching every other
    PIQL variant's dataset-level convention) are two distinct concepts --
    do not conflate them.
    """

    def __init__(
        self,
        *args,
        reward_discount: float = 0.99,
        beta_prime: float = 1.0,
        transition_discount: float = 0.99,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        assert reward_discount > 0.0 and reward_discount <= 1.0
        self.reward_discount = reward_discount
        self.beta_prime = beta_prime
        self.transition_discount = transition_discount

    def _get_reward_batch(self, batch: Dict):
        B, _, T, _ = batch["obs"].shape
        obs = batch["obs"].reshape(B * 2, T, -1).clone()
        action = batch["action"].reshape(B * 2, T, -1).clone()

        horizon = batch["horizon"].unsqueeze(1).expand(-1, 2).reshape(-1)  # (B*2,)
        time = torch.arange(T, device=obs.device)
        pad_mask = time.unsqueeze(0) >= horizon.unsqueeze(1)  # (B*2, T), True beyond real length

        row = torch.arange(B * 2, device=obs.device)
        last_idx = (horizon - 1).clamp(min=0)
        last_obs = obs[row, last_idx].unsqueeze(1).expand(-1, T, -1)
        last_act = action[row, last_idx].unsqueeze(1).expand(-1, T, -1)
        obs[pad_mask] = last_obs[pad_mask]
        action[pad_mask] = last_act[pad_mask]

        discount = self.transition_discount * torch.ones(B * 2, T, device=obs.device, dtype=obs.dtype)
        return obs, action, discount

    def _get_reward_loss_and_accuracy(self, reward: torch.Tensor, batch: Dict):
        B, _, T, _ = batch["obs"].shape
        per_step = reward.mean(dim=0).reshape(B, 2, T)  # (B, 2, T)

        time = torch.arange(T, device=per_step.device)
        mask = (time.unsqueeze(0) < batch["horizon"].unsqueeze(1)).to(dtype=per_step.dtype)  # (B, T)
        discounts = self.reward_discount ** torch.arange(T, device=per_step.device, dtype=per_step.dtype)

        J = (per_step * mask.unsqueeze(1) * discounts).sum(dim=-1)  # (B, 2)
        logit = self.beta_prime * (J[:, 0] - J[:, 1])
        loss = F.softplus(-logit).mean()
        with torch.no_grad():
            accuracy = (logit > 0).float().mean()
        return loss, accuracy
