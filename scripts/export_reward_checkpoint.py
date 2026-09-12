"""
Exports a Phase-1 RewardCPL*/RewardPolicy checkpoint (research/algs/reward_cpl.py)
as a portable, framework-independent .pt file: raw reward-net weights + architecture
metadata + reward normalization constants -- no dependency on the `research` package
is needed to reconstruct or load the result.

This is the bridge artifact multi-type-feedback's PPO baseline consumes (see
multi-type-feedback/multi_type_feedback/research_reward_net.py and
research_reward_fn.py). research/ and multi-type-feedback/ are kept in separate
Python environments on the cluster, so weights -- not a live cross-import -- are
what crosses that boundary.

Usage:
  python scripts/export_reward_checkpoint.py \
      --checkpoint runs/reward_demo/best_model.pt \
      --obs-dim 35 --act-dim 4 \
      --hidden-layers 512 512 --dropout 0.25 \
      --transition-data datasets/mw/data/mw_drawer-open-v2_ep2500_n0.3 \
      --out runs/reward_demo/portable_reward.pt
"""

import argparse

import gym
import numpy as np
import torch

REWARD_CHECKPOINT_PREFIX = "reward."  # RewardPolicy.CONTAINERS = ["reward"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Phase-1 RewardCPL* checkpoint (best_model.pt).")
    parser.add_argument("--obs-dim", type=int, required=True)
    parser.add_argument("--act-dim", type=int, required=True)
    parser.add_argument("--hidden-layers", type=int, nargs="+", default=[512, 512])
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument(
        "--transition-data",
        default=None,
        help="Directory of transition .npz episodes (the same pool used for Phase-2 IQL) to "
        "compute reward normalization over. If omitted, normalization is left at "
        "mean=0/std=1 and must be supplied separately.",
    )
    parser.add_argument("--norm-samples", type=int, default=20000)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state_dict = checkpoint["network"]
    reward_state_dict = {
        (k[len(REWARD_CHECKPOINT_PREFIX) :] if k.startswith(REWARD_CHECKPOINT_PREFIX) else k): v
        for k, v in state_dict.items()
    }

    mean, std = 0.0, 1.0
    if args.transition_data is not None:
        from research.datasets.replay_buffer.buffer import ReplayBuffer
        from research.networks.mlp import ContinuousMLPCritic

        obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(args.obs_dim,), dtype=np.float32)
        act_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(args.act_dim,), dtype=np.float32)
        net = ContinuousMLPCritic(
            obs_space, act_space, ensemble_size=1, hidden_layers=args.hidden_layers, dropout=args.dropout
        )
        net.load_state_dict(reward_state_dict)
        net.eval()

        buffer = ReplayBuffer(
            obs_space,
            act_space,
            path=args.transition_data,
            sample_fn="sample_qlearning",
            sample_kwargs=dict(discount=0.99, nstep=1, batch_size=1024, sample_by_timesteps=True),
            capacity=None,
            distributed=False,
        )
        samples, n_seen = [], 0
        for batch in buffer:
            obs = torch.as_tensor(batch["obs"], dtype=torch.float32)
            action = torch.as_tensor(batch["action"], dtype=torch.float32)
            with torch.no_grad():
                r = net(obs, action).mean(dim=0)
            samples.append(r.numpy())
            n_seen += len(r)
            if n_seen >= args.norm_samples:
                break
        samples = np.concatenate(samples)
        mean, std = float(samples.mean()), float(samples.std() + 1e-8)
        print(f"Computed reward normalization over {len(samples)} transitions: mean={mean:.4f} std={std:.4f}")
    else:
        print("No --transition-data supplied; exporting with mean=0.0, std=1.0 (unnormalized).")

    torch.save(
        {
            "state_dict": reward_state_dict,
            "arch": dict(obs_dim=args.obs_dim, act_dim=args.act_dim, hidden_layers=args.hidden_layers, dropout=args.dropout),
            "reward_mean": mean,
            "reward_std": std,
        },
        args.out,
    )
    print("Exported portable reward checkpoint to", args.out)


if __name__ == "__main__":
    main()
