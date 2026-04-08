import math
from typing import Optional

import gym
import numpy as np
import torch

from research.utils import utils


class DemoBuffer(torch.utils.data.IterableDataset):
    """
    Dataset for demonstrative feedback (Phase 5).

    Loads demo_labels_K<K>.npz produced by generate_demo_labels.py.
    Each sample is a K-way choice set where index 0 is always the demonstration
    (best trajectory by oracle rl_sum) and indices 1..K-1 are counterfactuals
    sorted descending by quality.

    Returns batches:
        obs    : (B, K, T, obs_dim)
        action : (B, K, T, act_dim)
        reward : (B, K, T)
        label  : (B,)  always 0 — the demonstration is always at index 0

    Args:
        observation_space: gym observation space (used for type checking only)
        action_space:      gym action space (used for type checking only)
        path:              path to demo_labels_K*.npz
        batch_size:        number of choice sets per batch
        capacity:          if set, only load the first N samples
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
        assert path is not None, "Must provide path to demo_labels_K*.npz"

        with open(path, "rb") as f:
            raw = np.load(f)
            obs    = raw["obs"]     # (N, K, T, obs_dim)
            action = raw["action"]  # (N, K, T, act_dim)
            reward = raw["reward"]  # (N, K, T)

        N = obs.shape[0]
        if capacity is not None and capacity < N:
            obs    = obs[:capacity]
            action = action[:capacity]
            reward = reward[:capacity]

        # Preprocessing — match FeedbackBuffer conventions
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

            obs    = self.obs[batch_inds, :, start:end]     # (B, K, t, obs_dim)
            action = self.action[batch_inds, :, start:end]  # (B, K, t, act_dim)
            reward = self.reward[batch_inds, :, start:end]  # (B, K, t)

            # Label is always 0: the demonstration is always at index 0
            label = np.zeros(len(batch_inds), dtype=np.float32)

            yield {
                "obs":    obs,
                "action": action,
                "reward": reward,
                "label":  label,
            }
