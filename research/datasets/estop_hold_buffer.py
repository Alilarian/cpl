import math
from typing import Optional

import gym
import numpy as np
import torch


class EstopHoldBuffer(torch.utils.data.IterableDataset):
    """
    Dataset for holding-model E-stop feedback (Model A of the ARIC E-stop spec).

    Loads estop_hold_labels.npz produced by generate_estop_hold_labels.py.
    Each sample is a pair of suffixes, both of the same real length horizon[m]:
        index 0: hold suffix H_tau     (preferred -- a physically simulated
                                        holding rollout that the frozen oracle
                                        scored as strictly better by > threshold)
        index 1: original suffix C_tau (non-preferred -- the recorded continuation)

    Both sides here are stored ZERO-padded beyond their real length, not
    repeat-padded -- there is no "halt and hold forever" fiction to exploit,
    since both suffixes are already equal-length real rollouts. Downstream losses
    MUST use the `horizon` field to build a mask and sum only over real
    timesteps k = 0..horizon-1 (see research/algs/scoring.py::score_segments's
    `mask` argument, and research/algs/cpl.py::EstopHoldCPL).

    Returns batches:
        obs        : (B, 2, T, obs_dim)
        action     : (B, 2, T, act_dim)
        reward     : (B, 2, T)
        horizon    : (B,)  int32 -- real (unpadded) length, shared by both arms of a pair
        stop_index : (B,)  int32 -- tau
        oracle_gap : (B,)  float32 -- Delta_tau

    Args:
        observation_space : gym observation space (used for type-checking only)
        action_space      : gym action space (used for type-checking only)
        path              : path to estop_hold_labels.npz
        batch_size        : number of pairs per batch
        capacity          : if set, only load the first N samples
        action_eps        : clips actions to [-1+eps, 1-eps] (default 1e-5)
        reward_scale      : scalar multiplier applied to rewards
        reward_shift      : scalar offset applied to rewards after scaling
        split             : "all" (default, preserves existing behaviour), "train", or "val"
        val_frac          : fraction of pairs held out for "val" when split != "all"
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
        split: str = "all",
        val_frac: float = 0.1,
    ):
        assert path is not None, "Must provide path to estop_hold_labels.npz"
        assert split in ("all", "train", "val")

        with open(path, "rb") as f:
            raw = np.load(f)
            obs = raw["obs"]                # (N, 2, T, obs_dim)
            action = raw["action"]          # (N, 2, T, act_dim)
            reward = raw["reward"]          # (N, 2, T)
            horizon = raw["horizon"]        # (N,)  int32
            stop_index = raw["stop_index"]  # (N,)  int32
            oracle_gap = raw["oracle_gap"]  # (N,)  float32

        if split != "all":
            N_full = obs.shape[0]
            perm = np.random.RandomState(0).permutation(N_full)
            n_val = int(N_full * val_frac)
            split_idx = perm[n_val:] if split == "train" else perm[:n_val]
            split_idx = np.sort(split_idx)
            obs, action, reward = obs[split_idx], action[split_idx], reward[split_idx]
            horizon, stop_index, oracle_gap = horizon[split_idx], stop_index[split_idx], oracle_gap[split_idx]

        N = obs.shape[0]
        if capacity is not None and capacity < N:
            obs, action, reward = obs[:capacity], action[:capacity], reward[:capacity]
            horizon, stop_index, oracle_gap = horizon[:capacity], stop_index[:capacity], oracle_gap[:capacity]

        obs = obs.astype(np.float32)
        action = action.astype(np.float32)
        reward = reward.astype(np.float32)

        lim = 1 - action_eps
        action = np.clip(action, -lim, lim)
        reward = reward_scale * reward + reward_shift

        self.obs = obs
        self.action = action
        self.reward = reward
        self.horizon = horizon.astype(np.int32)
        self.stop_index = stop_index.astype(np.int32)
        self.oracle_gap = oracle_gap.astype(np.float32)
        self.batch_size = batch_size

    def __len__(self):
        return self.obs.shape[0]

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        num_workers = worker_info.num_workers if worker_info is not None else 1
        worker_id = worker_info.id if worker_info is not None else 0

        N = len(self)
        chunk_size = N // num_workers
        my_inds = np.arange(chunk_size * worker_id, chunk_size * (worker_id + 1))
        idxs = np.random.permutation(my_inds)

        for i in range(math.ceil(len(idxs) / self.batch_size)):
            batch_inds = idxs[i * self.batch_size : (i + 1) * self.batch_size]

            yield {
                "obs": self.obs[batch_inds],                # (B, 2, T, obs_dim)
                "action": self.action[batch_inds],           # (B, 2, T, act_dim)
                "reward": self.reward[batch_inds],            # (B, 2, T)
                "horizon": self.horizon[batch_inds],          # (B,)
                "stop_index": self.stop_index[batch_inds],   # (B,)
                "oracle_gap": self.oracle_gap[batch_inds],   # (B,)
            }
