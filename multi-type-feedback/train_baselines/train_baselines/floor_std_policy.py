"""
SB3 ActorCriticPolicy variant that floors (and optionally caps) the Gaussian
policy's action std, e.g. std in [0.5, 1.5] as CPL uses for its PPO+learned-reward
baseline. Without a floor, PPO on a learned/imperfect reward can collapse the
policy's std toward zero, exploiting whatever direction the reward model is most
confidently (and often wrongly) rewarding.

This is the one place in the PPO bridge that requires a genuine custom SB3 policy
class -- the reward clamp, KL-to-reference term, and BC-init all live in the
reward function (research_reward_fn.py) and a callback (kl_reward_callback.py)
instead, so exp_manager.py's core PPO loop is never touched.
"""

import math

import torch as th
from stable_baselines3.common.distributions import DiagGaussianDistribution
from stable_baselines3.common.policies import ActorCriticPolicy


class FloorStdActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, *args, std_floor: float = 0.5, std_ceil: float = 1.5, **kwargs):
        self._std_floor = std_floor
        self._std_ceil = std_ceil
        super().__init__(*args, **kwargs)

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor):
        mean_actions = self.action_net(latent_pi)
        if isinstance(self.action_dist, DiagGaussianDistribution):
            log_std = th.clamp(self.log_std, min=math.log(self._std_floor), max=math.log(self._std_ceil))
            return self.action_dist.proba_distribution(mean_actions, log_std)
        return super()._get_action_dist_from_latent(latent_pi)
