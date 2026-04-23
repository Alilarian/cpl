import math
from typing import Optional

import gym
import numpy as np
import torch


class EstopBuffer(torch.utils.data.IterableDataset):
    """
    Dataset for (non-sequential) E-stop feedback.

    Loads estop_labels.npz produced by generate_estop_labels.py.
    Each sample is a pair:
        index 0: halt prefix  σ_{0:τ}  (preferred — oracle stopped here)
        index 1: full trajectory σ_{0:T}  (non-preferred — allowed to continue)

    Both segments are stored at full length T.  The halt prefix has obs[τ]
    and action[τ] repeated for steps τ+1..T-1 and reward zeroed after τ.
    stop_time τ is returned in the batch so EstopCPL can mask the discounted
    sum J_π(σ_{0:τ}) to only the real steps k = 0..τ.

    Returns batches:
        obs       : (B, 2, T, obs_dim)
        action    : (B, 2, T, act_dim)
        reward    : (B, 2, T)
        stop_time : (B,)  int32 — τ for each pair

    Args:
        observation_space : gym observation space (used for type-checking only)
        action_space      : gym action space (used for type-checking only)
        path              : path to estop_labels.npz
        batch_size        : number of pairs per batch
        capacity          : if set, only load the first N samples
        action_eps        : clips actions to [-1+eps, 1-eps] (default 1e-5)
        reward_scale      : scalar multiplier applied to rewards
        reward_shift      : scalar offset applied to rewards after scaling
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        path: Optional[str] = None,
        batch_size: int = 32,
        capacity: Optional[int] = None,
        action_eps: float = 1e-5,
        reward_scale: float = 1.0,
        reward_shift: float = 0.0,
    ):
        assert path is not None, "Must provide path to estop_labels.npz"

        with open(path, "rb") as f:
            raw = np.load(f)
            obs       = raw["obs"]        # (N, 2, T, obs_dim)
            action    = raw["action"]     # (N, 2, T, act_dim)
            reward    = raw["reward"]     # (N, 2, T)
            stop_time = raw["stop_time"]  # (N,)  int32

        N = obs.shape[0]
        if capacity is not None and capacity < N:
            obs       = obs[:capacity]
            action    = action[:capacity]
            reward    = reward[:capacity]
            stop_time = stop_time[:capacity]

        obs    = obs.astype(np.float32)
        action = action.astype(np.float32)
        reward = reward.astype(np.float32)

        lim    = 1 - action_eps
        action = np.clip(action, -lim, lim)
        reward = reward_scale * reward + reward_shift

        self.obs        = obs
        self.action     = action
        self.reward     = reward
        self.stop_time  = stop_time          # kept as int32
        self.batch_size = batch_size

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

        for i in range(math.ceil(len(idxs) / self.batch_size)):
            batch_inds = idxs[i * self.batch_size : (i + 1) * self.batch_size]

            yield {
                "obs":       self.obs[batch_inds],        # (B, 2, T, obs_dim)
                "action":    self.action[batch_inds],     # (B, 2, T, act_dim)
                "reward":    self.reward[batch_inds],     # (B, 2, T)
                "stop_time": self.stop_time[batch_inds],  # (B,)
            }
