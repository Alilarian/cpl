"""
Benchmark oracle SAC checkpoints to identify the best model(s) to use as experts
for demonstrative feedback generation.

For each run directory (one per seed), evaluates the specified checkpoint by:
  1. Rolling out num_ep full episodes
  2. Computing advantage-based scores identical to add_pref_labels.py:
       rl_lp      = sum_t log pi(a_t | s_t)
       rl_dis_dir = sum_t gamma^t (Q(s_t,a_t) - V(s_t))
       rl_sum     = sum_t r_t + V(s_T) - V(s_0)
       rl_dis_sum = sum_t gamma^t r_t + gamma^T V(s_T) - V(s_0)
  3. Averaging across episodes to produce a scalar score per run

Output: a CSV ranking all evaluated runs by each metric, and a summary of
        which checkpoint(s) are recommended as oracle experts.

Usage (single seed):
    python scripts/benchmark_oracles.py \
        --run-dirs runs/runs/chpc/oracle_sac_all/mw_drawer-open-v2 \
        --num-ep 20

Usage (multiple seeds, pick top 3):
    python scripts/benchmark_oracles.py \
        --run-dirs runs/seed0/mw_drawer runs/seed1/mw_drawer runs/seed2/mw_drawer \
        --num-ep 20 \
        --top-n 3
"""

import argparse
import csv
import math
import os

import numpy as np
import torch

from research.utils.config import Config


def compute_episode_advantages(obs_seq, action_seq, reward_seq, model, discount, mcmc_samples, device):
    """
    Compute all advantage metrics for a single episode using the same logic as
    add_pref_labels.py.

    Args:
        obs_seq:    np.ndarray (T, obs_dim)
        action_seq: np.ndarray (T, act_dim)
        reward_seq: np.ndarray (T,)
        model:      loaded CPL model with network.encoder, network.actor, network.critic
        discount:   float, gamma
        mcmc_samples: int, M samples for V(s) estimation
        device:     torch.device

    Returns:
        dict of scalar advantage metrics for this episode
    """
    T = obs_seq.shape[0]

    # Add batch dimension: (1, T, dim)
    obs = torch.from_numpy(obs_seq).float().unsqueeze(0).to(device)
    action = torch.from_numpy(action_seq).float().unsqueeze(0).to(device)
    reward = torch.from_numpy(reward_seq).float().unsqueeze(0).to(device)

    metrics = {}

    with torch.no_grad():
        obs_enc = model.network.encoder(obs)                          # (1, T, D)
        dist = model.network.actor(obs_enc)                           # (1, T, D_a)

        if isinstance(dist, torch.distributions.Distribution):
            lp = dist.log_prob(torch.clamp(action, min=-0.999, max=0.999))  # (1, T)
            # sum over timesteps → scalar per episode
            metrics["rl_lp"] = lp.sum(dim=-1).squeeze(0).cpu().item()
        else:
            # deterministic actor: use negative MSE as proxy
            lp = -torch.square(dist - action).sum(dim=-1)
            metrics["rl_lp"] = lp.sum(dim=-1).squeeze(0).cpu().item()

        if hasattr(model.network, "critic"):
            q = model.network.critic(obs_enc, action).mean(dim=0)    # (1, T)

            # Estimate V(s) = E_{a'~pi}[Q(s,a')] via MCMC
            obs_exp = obs_enc.unsqueeze(2).expand(-1, -1, mcmc_samples, -1)  # (1, T, M, D)
            dist_exp = model.network.actor(obs_exp)
            sampled_actions = dist_exp.sample()                               # (1, T, M, D_a)
            v = model.network.critic(obs_exp, sampled_actions).mean(dim=0)   # (1, T, M)
            v = v.mean(dim=2)                                                 # (1, T)

            # discount vector: [1, gamma, gamma^2, ..., gamma^(T-1)]
            disc = torch.pow(
                discount * torch.ones(T, device=device),
                torch.arange(T, device=device).float(),
            ).unsqueeze(0)  # (1, T)

            adv = q - v  # (1, T)

            metrics["rl_dir"]     = adv.sum(dim=-1).squeeze(0).cpu().item()
            metrics["rl_dis_dir"] = (disc * adv).sum(dim=-1).squeeze(0).cpu().item()
            metrics["rl_sum"]     = (
                reward[:, :-1].sum(dim=-1) + v[:, -1] - v[:, 0]
            ).squeeze(0).cpu().item()
            metrics["rl_dis_sum"] = (
                (disc * reward)[:, :-1].sum(dim=-1)
                + (discount ** T) * v[:, -1]
                - v[:, 0]
            ).squeeze(0).cpu().item()

    return metrics


def rollout_and_score(model, env, num_ep, discount, mcmc_samples, device):
    """
    Roll out the model for num_ep episodes and return average advantage metrics
    and average reward.
    """
    all_metrics = []
    all_rewards = []

    for ep in range(num_ep):
        obs = env.reset()
        if isinstance(obs, tuple):  # gym >= 0.26
            obs = obs[0]

        obs_list, action_list, reward_list = [], [], []
        done = False

        while not done:
            obs_list.append(obs)
            with torch.no_grad():
                action = model.predict(dict(obs=obs), sample=False)
            action = np.clip(action, env.action_space.low, env.action_space.high)

            result = env.step(action)
            if len(result) == 5:  # gym >= 0.26: obs, reward, terminated, truncated, info
                next_obs, reward, terminated, truncated, _ = result
                done = terminated or truncated
            else:
                next_obs, reward, done, _ = result

            action_list.append(action)
            reward_list.append(reward)
            obs = next_obs

        obs_seq    = np.stack(obs_list,    axis=0).astype(np.float32)   # (T, obs_dim)
        action_seq = np.stack(action_list, axis=0).astype(np.float32)   # (T, act_dim)
        reward_seq = np.array(reward_list, dtype=np.float32)            # (T,)

        ep_metrics = compute_episode_advantages(
            obs_seq, action_seq, reward_seq, model, discount, mcmc_samples, device
        )
        all_metrics.append(ep_metrics)
        all_rewards.append(reward_seq.sum())

        if (ep + 1) % 5 == 0:
            print(f"  Episode {ep+1}/{num_ep} done, reward={reward_seq.sum():.2f}")

    # Average across episodes
    avg_metrics = {k: float(np.mean([m[k] for m in all_metrics if k in m]))
                   for k in all_metrics[0]}
    avg_metrics["reward"] = float(np.mean(all_rewards))
    avg_metrics["reward_std"] = float(np.std(all_rewards))
    return avg_metrics


def load_model(run_dir, checkpoint_name, device):
    checkpoint_path = os.path.join(run_dir, checkpoint_name)
    assert os.path.isfile(checkpoint_path), f"Checkpoint not found: {checkpoint_path}"

    config = Config.load(run_dir)
    config["checkpoint"] = None
    config = config.parse()

    env_fn = config.get_train_env_fn() or config.get_eval_env_fn()
    env = env_fn()

    model = config.get_model(
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
    )
    model.load(checkpoint_path)
    model.eval()
    return model, env


def main():
    parser = argparse.ArgumentParser(description="Benchmark oracle SAC checkpoints.")
    parser.add_argument(
        "--run-dirs",
        nargs="+",
        required=True,
        help="One or more run directories (one per seed). Each must contain config.yaml.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="best_model.pt",
        help="Checkpoint filename to evaluate within each run dir (default: best_model.pt).",
    )
    parser.add_argument(
        "--num-ep",
        type=int,
        default=20,
        help="Number of evaluation episodes per run (default: 20).",
    )
    parser.add_argument(
        "--discount",
        type=float,
        default=0.99,
        help="Discount factor gamma (default: 0.99).",
    )
    parser.add_argument(
        "--mcmc-samples",
        type=int,
        default=64,
        help="Number of MCMC samples for V(s) estimation (default: 64).",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=None,
        help="Print top-N runs by rl_lp score. If None, print all.",
    )
    parser.add_argument(
        "--rank-by",
        type=str,
        default="rl_sum",
        choices=["rl_lp", "rl_dis_dir", "rl_sum", "rl_dis_sum", "reward"],
        help="Metric to rank runs by (default: rl_sum).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save results CSV. If None, saves to first run-dir's parent as oracle_benchmark.csv.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: 'cpu', 'cuda', or 'auto' (default: auto).",
    )
    args = parser.parse_args()

    # Resolve device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Using device: {device}")

    results = []

    for run_dir in args.run_dirs:
        run_dir = os.path.normpath(run_dir)
        print(f"\n{'='*60}")
        print(f"Evaluating: {run_dir}  [{args.checkpoint}]")
        print(f"{'='*60}")

        model, env = load_model(run_dir, args.checkpoint, device)

        avg_metrics = rollout_and_score(
            model, env, args.num_ep, args.discount, args.mcmc_samples, device
        )

        row = {
            "run_dir":    run_dir,
            "checkpoint": args.checkpoint,
            **avg_metrics,
        }
        results.append(row)

        print(f"  Results for {os.path.basename(run_dir)}:")
        for k, v in avg_metrics.items():
            print(f"    {k:20s}: {v:.4f}")

    # Rank by chosen metric
    results.sort(key=lambda r: r.get(args.rank_by, float("-inf")), reverse=True)

    # Determine output path
    if args.output is None:
        parent = os.path.dirname(os.path.normpath(args.run_dirs[0]))
        output_path = os.path.join(parent, "oracle_benchmark.csv")
    else:
        output_path = args.output

    # Write CSV
    if results:
        fieldnames = list(results[0].keys())
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults saved to: {output_path}")

    # Print ranking summary
    top_n = args.top_n if args.top_n is not None else len(results)
    print(f"\n{'='*60}")
    print(f"RANKING (by {args.rank_by}, top {top_n} of {len(results)}):")
    print(f"{'='*60}")
    for rank, row in enumerate(results[:top_n], 1):
        print(
            f"  #{rank}  {os.path.basename(row['run_dir']):30s}"
            f"  {args.rank_by}={row.get(args.rank_by, 'N/A'):>10.4f}"
            f"  reward={row['reward']:>8.2f}"
        )

    print(f"\nRECOMMENDED ORACLE(s) — top {top_n}:")
    for row in results[:top_n]:
        print(f"  {os.path.join(row['run_dir'], row['checkpoint'])}")


if __name__ == "__main__":
    main()
