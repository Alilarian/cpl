"""
RewardFn adapter that scores PPO rollouts with a frozen reward net exported from
research/ (scripts/export_reward_checkpoint.py), mirroring the existing
CustomReward class in train_RL_agent.py but backed by research's ContinuousMLPCritic
architecture instead of a LightningNetwork.

Implements three requirements from the reward-CPL baseline spec, all here rather
than in exp_manager.py's core PPO loop:
  - reward clamped to [0, 1] after normalization (CPL found unbounded reward
    changes the implicit preference over episode length),
  - beta_KL * (log pi - log mu) subtracted when a BC-pretrained reference policy
    mu and a live PPO policy reference are both attached (see
    train_baselines/train_baselines/kl_reward_callback.py, which sets
    self.current_policy on each PPO step -- RewardVecEnvWrapper's reward_fn
    protocol only passes (old_obs, action, new_obs, done), not the live policy).
"""

from typing import Optional

import numpy as np
import torch

from multi_type_feedback.research_reward_net import load_research_reward_net
from multi_type_feedback.utils import RewardFn


class ResearchRewardFn(RewardFn):
    def __init__(
        self,
        reward_checkpoint_path: str,
        mu_policy=None,
        kl_coeff: float = 0.0,
        clip_range=(0.0, 1.0),
        device: str = "cpu",
    ):
        super().__init__()
        self.device = device
        checkpoint = torch.load(reward_checkpoint_path, map_location=device, weights_only=False)
        self.net = load_research_reward_net(checkpoint).to(device)
        self.mean = checkpoint["reward_mean"]
        self.std = checkpoint["reward_std"]
        self.clip_range = clip_range
        self.mu_policy = mu_policy
        self.kl_coeff = kl_coeff
        # Set externally, once per PPO step, by KLRewardSyncCallback -- this is the
        # only way to get the live training policy into a reward_fn, since
        # RewardVecEnvWrapper.step_wait() only forwards (obs, action, next_obs, done).
        self.current_policy = None

    def __call__(
        self, state: np.ndarray, actions: np.ndarray, next_state: np.ndarray, _done: np.ndarray
    ) -> np.ndarray:
        obs_t = torch.as_tensor(state, device=self.device, dtype=torch.float32)
        act_t = torch.as_tensor(actions, device=self.device, dtype=torch.float32)
        with torch.no_grad():
            raw_reward = self.net(obs_t, act_t)
            reward = (raw_reward - self.mean) / self.std
            reward = torch.clamp(reward, self.clip_range[0], self.clip_range[1])

            if self.kl_coeff > 0.0 and self.mu_policy is not None and self.current_policy is not None:
                policy_obs, _ = self.current_policy.obs_to_tensor(state)
                mu_obs, _ = self.mu_policy.obs_to_tensor(state)
                logp_pi = self.current_policy.get_distribution(policy_obs).log_prob(act_t)
                logp_mu = self.mu_policy.get_distribution(mu_obs).log_prob(act_t)
                reward = reward - self.kl_coeff * (logp_pi - logp_mu)

        return reward.cpu().numpy()
