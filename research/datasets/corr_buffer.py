import math
from typing import Optional

import gym
import numpy as np
import torch

from research.utils import utils


class CorrBuffer(torch.utils.data.IterableDataset):
    """
    Dataset for corrective feedback.

    Loads corr_labels.npz produced by generate_corr_labels.py.
    Each sample is a K=2 pair:
        index 0: expert correction (best by oracle rl_sum)
        index 1: original pool segment (the agent's own trajectory)

    Both trajectories start from the same initial state s_0, so the expert
    trajectory provides a causally-linked correction: "from your state, do this."

    Returns batches:
        obs    : (B, 2, T, obs_dim)
        action : (B, 2, T, act_dim)
        reward : (B, 2, T)
        label  : (B,)  always 0 — the expert correction is always at index 0

    Args:
        observation_space: gym observation space (used for type checking only)
        action_space:      gym action space (used for type checking only)
        path:              path to corr_labels.npz
        batch_size:        number of pairs per batch
        capacity:          if set, only load the first N samples
                           (useful since pairs are sorted by improvement magnitude,
                            so capacity selects the most corrective examples)
        action_eps:        clips actions to [-1+eps, 1-eps] (default 1e-5)
        segment_length:    if set, randomly crop each trajectory to this length
        reward_scale:      scalar multiplier applied to rewards
        reward_shift:      scalar offset applied to rewards after scaling
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        path: Optional[str] = None,
        batch_size: int = 32,
        capacity: Optional[int] = None,
        action_eps: float = 1e-5,
        segment_length: Optional[int] = None,
        reward_scale: float = 1.0,
        reward_shift: float = 0.0,
    ):
        assert path is not None, "Must provide path to corr_labels.npz"

        with open(path, "rb") as f:
            raw = np.load(f)
            obs    = raw["obs"]        # (N, 2, T, obs_dim)
            action = raw["action"]     # (N, 2, T, act_dim)
            reward = raw["reward"]     # (N, 2, T)

        N = obs.shape[0]
        if capacity is not None and capacity < N:
            obs    = obs[:capacity]
            action = action[:capacity]
            reward = reward[:capacity]

        obs    = obs.astype(np.float32)
        action = action.astype(np.float32)
        reward = reward.astype(np.float32)

        lim    = 1 - action_eps
        action = np.clip(action, -lim, lim)
        reward = reward_scale * reward + reward_shift

        self.obs            = obs
        self.action         = action
        self.reward         = reward
        self.batch_size     = batch_size
        self.segment_length = segment_length

    def __len__(self):
        return self.obs.shape[0]

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        num_workers = worker_info.num_workers if worker_info is not None else 1
        worker_id   = worker_info.id          if worker_info is not None else 0

        N          = len(self)
        chunk_size = N // num_workers
        my_inds    = np.arange(chunk_size * worker_id, chunk_size * (worker_id + 1))
        idxs       = np.random.permutation(my_inds)

        T_full = self.obs.shape[2]

        for i in range(math.ceil(len(idxs) / self.batch_size)):
            batch_inds = idxs[i * self.batch_size : (i + 1) * self.batch_size]

            if self.segment_length is not None:
                start = np.random.randint(0, T_full - self.segment_length + 1)
                end   = start + self.segment_length
            else:
                start, end = 0, T_full

            obs    = self.obs[batch_inds, :, start:end]     # (B, 2, t, obs_dim)
            action = self.action[batch_inds, :, start:end]  # (B, 2, t, act_dim)
            reward = self.reward[batch_inds, :, start:end]  # (B, 2, t)

            # Label is always 0: the expert correction is always at index 0
            label = np.zeros(len(batch_inds), dtype=np.float32)

            yield {
                "obs":    obs,
                "action": action,
                "reward": reward,
                "label":  label,
            }
