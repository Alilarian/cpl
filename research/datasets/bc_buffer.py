import math
from typing import Optional

import gym
import numpy as np
import torch


class BCBuffer(torch.utils.data.IterableDataset):
    """
    Dataset for BC warmup from a scored trajectory pool.

    Loads bc_pool.npz produced by score_pool_for_bc.py.  Each sample is a
    single segment (no K axis) — the top-X% of pool segments by oracle rl_sum.
    Segments are already sorted descending by rl_sum in the file.

    Returns batches:
        obs    : (B, T, obs_dim)
        action : (B, T, act_dim)
        reward : (B, T)

    No label is returned — BC training uses (obs, action) pairs directly.
    The trainer is responsible for computing the imitation loss (e.g. MSE or NLL
    over all T timesteps of each segment).

    Args:
        observation_space: gym observation space (type checking only)
        action_space:      gym action space (type checking only)
        path:              path to bc_pool.npz
        batch_size:        segments per batch
        capacity:          if set, only load the first N samples (highest rl_sum)
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
        batch_size: int = 64,
        capacity: Optional[int] = None,
        action_eps: float = 1e-5,
        segment_length: Optional[int] = None,
        reward_scale: float = 1.0,
        reward_shift: float = 0.0,
    ):
        assert path is not None, "Must provide path to bc_pool.npz"

        with open(path, "rb") as f:
            raw    = np.load(f)
            obs    = raw["obs"]     # (N, T, obs_dim)
            action = raw["action"]  # (N, T, act_dim)
            reward = raw["reward"]  # (N, T)

        N = obs.shape[0]
        if capacity is not None and capacity < N:
            # Segments are sorted descending by rl_sum — capacity selects the best
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

        T_full = self.obs.shape[1]

        for i in range(math.ceil(len(idxs) / self.batch_size)):
            batch_inds = idxs[i * self.batch_size : (i + 1) * self.batch_size]

            if self.segment_length is not None:
                start = np.random.randint(0, T_full - self.segment_length + 1)
                end   = start + self.segment_length
            else:
                start, end = 0, T_full

            yield {
                "obs":    self.obs[batch_inds, start:end],     # (B, t, obs_dim)
                "action": self.action[batch_inds, start:end],  # (B, t, act_dim)
                "reward": self.reward[batch_inds, start:end],  # (B, t)
            }
