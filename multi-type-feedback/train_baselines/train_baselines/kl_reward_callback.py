"""
SB3 callback that hands the live, currently-training PPO policy to a
ResearchRewardFn each step, so it can compute beta_KL * (log pi - log mu) inside
the reward function. RewardVecEnvWrapper.step_wait() only calls
reward_fn(old_obs, actions, new_obs, dones) -- it never passes the policy -- so
this is the only way to get a live policy reference into the reward function
without modifying exp_manager.py's core PPO loop or imitation's RewardVecEnvWrapper.

Usage (in a driving script, after exp_manager.setup_experiment()):
    exp_manager.callbacks.append(KLRewardSyncCallback(reward_fn))
    exp_manager.learn(model)
"""

from stable_baselines3.common.callbacks import BaseCallback


class KLRewardSyncCallback(BaseCallback):
    def __init__(self, reward_fn, verbose: int = 0):
        super().__init__(verbose)
        self.reward_fn = reward_fn

    def _on_step(self) -> bool:
        self.reward_fn.current_policy = self.model.policy
        return True
