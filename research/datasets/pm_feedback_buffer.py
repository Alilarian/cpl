"""
Pairwise preference feedback buffer for PointMass.

Reads the (N, 2, T, obs_dim) format produced by
``generate_pm_feedback.py --type pref``.

Compatible with the existing CPL algorithm: each yielded batch contains
``obs_1 / obs_2 / action_1 / action_2 / label`` exactly as expected by
``CPL._get_cpl_loss``.
"""

import math

import gym
import numpy as np
import torch


class PMFeedbackBuffer(torch.utils.data.IterableDataset):
    """
    Parameters
    ----------
    path        : path to pref_labels.npz (from generate_pm_feedback.py --type pref)
    batch_size  : pairs per yielded batch
    capacity    : max number of pairs to load (None = all)
    action_eps  : clip actions to [-1+eps, 1-eps] for numerical safety
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        path: str,
        batch_size: int = 64,
        capacity: int = None,
        action_eps: float = 1e-5,
    ):
        data = np.load(path)
        obs    = data["obs"].astype(np.float32)          # (N, 2, T, obs_dim)
        action = data["action"].astype(np.float32)       # (N, 2, T, act_dim)
        scores = data["adv_scores"].astype(np.float32)   # (N, 2)

        N = obs.shape[0]
        if capacity is not None and N > capacity:
            obs    = obs[:capacity]
            action = action[:capacity]
            scores = scores[:capacity]
            N = capacity

        lim = 1.0 - action_eps
        action = np.clip(action, -lim, lim)

        # Label: 1.0 if segment-1 is better, 0.0 if segment-0 is better, 0.5 if tied
        hard   = (scores[:, 1] > scores[:, 0]).astype(np.float32)
        soft   = 0.5 * (scores[:, 0] == scores[:, 1]).astype(np.float32)
        labels = hard + soft  # (N,)

        self.obs_0  = obs[:, 0]     # (N, T, obs_dim)
        self.obs_1  = obs[:, 1]
        self.act_0  = action[:, 0]  # (N, T, act_dim)
        self.act_1  = action[:, 1]
        self.labels = labels        # (N,)
        self.N = N
        self.batch_size = batch_size

    def __len__(self) -> int:
        return self.N

    def __iter__(self):
        worker_info  = torch.utils.data.get_worker_info()
        num_workers  = worker_info.num_workers if worker_info is not None else 1
        worker_id    = worker_info.id          if worker_info is not None else 0

        chunk   = self.N // num_workers
        my_inds = np.arange(chunk * worker_id, chunk * (worker_id + 1))
        idxs    = np.random.permutation(my_inds)

        for i in range(math.ceil(len(idxs) / self.batch_size)):
            b = idxs[i * self.batch_size : (i + 1) * self.batch_size]
            yield {
                "obs_1":    self.obs_0[b],    # (B, T, obs_dim)
                "obs_2":    self.obs_1[b],
                "action_1": self.act_0[b],    # (B, T, act_dim)
                "action_2": self.act_1[b],
                "label":    self.labels[b],   # (B,)
            }
