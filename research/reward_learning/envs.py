"""
Env wrappers that put a learned reward into the oracle SAC loop.

- LearnedRewardWrapper (training env only): reward := a * r_hat(s_t, a_t) + b, where
  r_hat is the standardized ensemble reward on the RAW observation before the step and
  the clipped action, and (a, b) match the true reward's mean/std on the feedback data
  (research SAC does not normalize rewards). The true reward is kept in
  info["true_reward"]. AsyncEnv's worker drops `info`, so per-episode true/learned
  returns and success are also appended to a CSV from inside the env process.
- ShadowRewardWrapper (eval env): leaves the TRUE reward untouched and adds
  info["learned_reward"] / info["learned_reward_std"]; eval_policy sums scalar info keys
  per episode, so eval/learned_reward is logged for free. It also buffers eval
  episodes so the driver can save trajectory snapshots.

The reward model runs as a plain numpy forward pass: AsyncEnv forks its worker after
the parent has initialized torch/CUDA, and torch CPU ops in a forked child are not
fork-safe (OpenMP), whereas numpy matmuls are.
"""

import csv
import os
from typing import Dict, Optional, Tuple

import gym
import numpy as np

TRAIN_EPISODE_FIELDS = ["episode", "env_step", "length", "true_return", "learned_return", "success",
                        "mean_member_std"]


class NumpyRewardModel:
    """numpy re-implementation of RewardEnsemble.reward_and_std (verified equal in tests)."""

    def __init__(self, ckpt_path: str):
        import torch  # only to read the file; no torch compute happens in this class

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        arch, sd = ckpt["arch"], ckpt["state_dict"]
        n_layers = len(arch["hidden_layers"]) + 1
        self.W = [sd[f"layers.{i}.weight"].numpy().astype(np.float64) for i in range(n_layers)]
        self.b = [sd[f"layers.{i}.bias"].numpy().astype(np.float64) for i in range(n_layers)]
        self.mu = ckpt["mu"].numpy().astype(np.float64)
        self.sigma = ckpt["sigma"].numpy().astype(np.float64)
        self.output = arch["output"]
        self.slope = arch["negative_slope"]
        self.obs_dim, self.act_dim = arch["obs_dim"], arch["act_dim"]
        self.extra = ckpt.get("extra", {})

    def members(self, obs: np.ndarray, act: np.ndarray) -> np.ndarray:
        """obs (N, obs_dim), act (N, act_dim) -> (E, N) raw member rewards."""
        x = np.concatenate([np.asarray(obs, np.float64), np.asarray(act, np.float64)], axis=-1)
        assert x.shape[-1] == self.obs_dim + self.act_dim, x.shape
        h = np.broadcast_to(x, (self.W[0].shape[0],) + x.shape)
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            h = h @ W + b
            if i < len(self.W) - 1:
                h = np.where(h > 0, h, self.slope * h)
        h = h[..., 0]
        return np.tanh(h) if self.output == "tanh" else h

    def reward_and_std(self, obs: np.ndarray, act: np.ndarray):
        z = (self.members(obs, act) - self.mu[:, None]) / self.sigma[:, None]
        return z.mean(axis=0), z.std(axis=0), z


def _split_step(result):
    if len(result) == 5:
        obs, reward, terminated, truncated, info = result
        return obs, reward, terminated or truncated, info, result
    obs, reward, done, info = result
    return obs, reward, done, info, result


def _rebuild_step(result, obs, reward, info):
    if len(result) == 5:
        return obs, reward, result[2], result[3], info
    return obs, reward, result[2], info


def _reset_obs(out):
    return out[0] if isinstance(out, tuple) else out


class _RewardWrapperBase(gym.Wrapper):
    def __init__(self, env: gym.Env, mode: str, reward_ckpt: Optional[str] = None, affine: Tuple[float, float] = (1.0, 0.0)):
        super().__init__(env)
        assert mode in ("learned", "true"), mode
        self.mode = mode
        self.model = NumpyRewardModel(reward_ckpt) if mode == "learned" else None
        self.a, self.b = float(affine[0]), float(affine[1])
        self._obs = None
        self._low, self._high = self.env.action_space.low, self.env.action_space.high

    @property
    def _max_episode_steps(self):
        return self.env._max_episode_steps

    def _learned(self, obs, action, true_reward) -> Tuple[float, float, Optional[np.ndarray]]:
        if self.mode == "true":
            return float(true_reward), 0.0, None
        a = np.clip(np.asarray(action, np.float32), self._low, self._high)
        r, std, z = self.model.reward_and_std(np.asarray(obs, np.float32)[None], a[None])
        return self.a * float(r[0]) + self.b, float(std[0]), z[:, 0]


class LearnedRewardWrapper(_RewardWrapperBase):
    def __init__(self, env, mode, reward_ckpt=None, affine=(1.0, 0.0), episode_log: Optional[str] = None):
        super().__init__(env, mode, reward_ckpt, affine)
        self.episode_log = episode_log
        self._episode = 0
        self._env_step = 0
        if episode_log is not None and os.path.exists(episode_log):
            # Resuming: continue the step / episode counters from the last logged episode.
            with open(episode_log) as f:
                rows = list(csv.DictReader(f))
            if rows:
                self._episode = int(rows[-1]["episode"]) + 1
                self._env_step = int(rows[-1]["env_step"])
        self._reset_acc()

    def _reset_acc(self):
        self._len, self._true, self._learn, self._succ, self._std = 0, 0.0, 0.0, 0.0, 0.0

    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)
        self._obs = _reset_obs(out)
        self._reset_acc()
        return out

    def step(self, action):
        result = self.env.step(action)
        next_obs, true_r, done, info, raw = _split_step(result)
        learned_r, std, _ = self._learned(self._obs, action, true_r)
        info = dict(info)
        info["true_reward"] = float(true_r)
        info["learned_reward"] = learned_r
        self._len += 1
        self._env_step += 1
        self._true += float(true_r)
        self._learn += learned_r
        self._std += std
        self._succ = max(self._succ, float(info.get("success", 0.0)))
        if done:
            self._log_episode()
        self._obs = next_obs
        return _rebuild_step(raw, next_obs, learned_r, info)

    def _log_episode(self):
        if self.episode_log is None:
            return
        new = not os.path.exists(self.episode_log)
        with open(self.episode_log, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=TRAIN_EPISODE_FIELDS)
            if new:
                w.writeheader()
            w.writerow(dict(episode=self._episode, env_step=self._env_step, length=self._len,
                            true_return=self._true, learned_return=self._learn, success=self._succ,
                            mean_member_std=self._std / max(self._len, 1)))
        self._episode += 1


class ShadowRewardWrapper(_RewardWrapperBase):
    def __init__(self, env, mode, reward_ckpt=None, affine=(1.0, 0.0)):
        super().__init__(env, mode, reward_ckpt, affine)
        self._episodes = []
        self._cur = None

    def _finish_episode(self):
        if self._cur is not None and len(self._cur["act"]) > 0:
            self._episodes.append({k: np.asarray(v) for k, v in self._cur.items()})
        self._cur = None

    def reset(self, **kwargs):
        self._finish_episode()
        out = self.env.reset(**kwargs)
        self._obs = _reset_obs(out)
        self._cur = dict(obs=[], act=[], true_reward=[], learned_reward=[], member_reward=[], success=[])
        return out

    def step(self, action):
        result = self.env.step(action)
        next_obs, true_r, done, info, raw = _split_step(result)
        learned_r, std, z = self._learned(self._obs, action, true_r)
        info = dict(info)
        info["learned_reward"] = learned_r
        info["learned_reward_std"] = std
        if self._cur is not None:
            self._cur["obs"].append(np.asarray(self._obs, np.float32))
            self._cur["act"].append(np.clip(np.asarray(action, np.float32), self._low, self._high))
            self._cur["true_reward"].append(float(true_r))
            self._cur["learned_reward"].append(learned_r)
            self._cur["member_reward"].append(np.zeros(1) if z is None else z)
            self._cur["success"].append(float(info.get("success", 0.0)))
        self._obs = next_obs
        return _rebuild_step(raw, next_obs, true_r, info)

    def pop_episodes(self):
        self._finish_episode()
        eps, self._episodes = self._episodes, []
        return eps


def save_eval_episodes(episodes, path: str) -> None:
    """Stack equal-length eval episodes (MetaWorld: always the full horizon) into one npz."""
    if not episodes:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out = {k: np.stack([e[k] for e in episodes]) for k in episodes[0]}
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        np.savez(f, **out)
    os.replace(tmp, path)


def make_env(env_name: str, env_kwargs: Dict, wrapper, wrapper_kwargs: Dict, role: str, mode: str,
             reward_ckpt: Optional[str], affine: Tuple[float, float], episode_log: Optional[str] = None):
    """Module-level (picklable) env factory: exactly Config.get_env(...) + one reward wrapper."""
    from research.utils.config import get_env

    env = get_env(env=env_name, env_kwargs=env_kwargs, wrapper=wrapper, wrapper_kwargs=wrapper_kwargs)
    if role == "train":
        return LearnedRewardWrapper(env, mode, reward_ckpt, affine, episode_log)
    assert role == "eval", role
    return ShadowRewardWrapper(env, mode, reward_ckpt, affine)
