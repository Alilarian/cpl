"""
Compute summary statistics for the trajectory pool (Phase 3) and
demonstrative feedback dataset (Phase 4).

Outputs a per-environment statistics table to stdout and saves a CSV.

Statistics reported
-------------------
Pool (pool.npz):
  - N segments, segment length T, obs/action/state dims
  - Segment reward (undiscounted sum): mean, std, min, max
  - Segment reward (discounted sum, gamma=0.99): mean, std, min, max
  - Checkpoint step coverage: n_unique, min, max

Demo labels (demo_labels_K*.npz):
  - N kept, K (choice set size), skip rate vs pool
  - Per-position mean undiscounted reward: demo (pos 0), mid, worst (pos K-1)
  - rl_sum per position: mean and std  (quality gradient across ranking)
  - Demo position (pos 0) rl_sum: mean, std, min, max
  - Demo advantage gap (rl_sum[0] - rl_sum[K-1]): mean, std, min, max
  - Choice set action diversity: mean pairwise L2 distance across K trajectories
    (measures counterfactual coverage / spread)
  - Checkpoint step coverage of kept segments: n_unique, min, max

Usage (on CHPC):
    python scripts/dataset_stats.py \\
        --pool-dir /scratch/general/vast/u1472210/demo_pool \\
        --labels-dir /scratch/general/vast/u1472210/demo_labels \\
        --output results/dataset_stats.csv

Usage (local, if data synced):
    python scripts/dataset_stats.py \\
        --pool-dir datasets/demo_pool \\
        --labels-dir datasets/demo_labels
"""

import argparse
import csv
import os

import numpy as np


ENVS_ALL = [
    "mw_bin-picking-v2",
    "mw_button-press-v2",
    "mw_door-open-v2",
    "mw_drawer-open-v2",
    "mw_plate-slide-v2",
]

DISCOUNT = 0.99


def discounted_sum(reward, gamma):
    """reward: (N, T) → (N,) discounted return."""
    T = reward.shape[1]
    disc = gamma ** np.arange(T, dtype=np.float64)
    return (reward * disc[None, :]).sum(axis=1)


def pairwise_action_diversity(action):
    """
    action: (K, T, act_dim)
    Returns mean pairwise L2 distance between all K trajectories.
    """
    K = action.shape[0]
    if K < 2:
        return 0.0
    flat = action.reshape(K, -1)
    dists = []
    for i in range(K):
        for j in range(i + 1, K):
            dists.append(np.linalg.norm(flat[i] - flat[j]))
    return float(np.mean(dists))


def stats_dict(arr, prefix):
    """Return mean/std/min/max for a 1-D array with a key prefix."""
    return {
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_std":  float(np.std(arr)),
        f"{prefix}_min":  float(np.min(arr)),
        f"{prefix}_max":  float(np.max(arr)),
    }


def pool_stats(pool_path):
    print(f"  Loading pool: {pool_path}")
    d = np.load(pool_path)
    obs    = d["obs"]             # (N, T, obs_dim)
    action = d["action"]          # (N, T, act_dim)
    reward = d["reward"]          # (N, T)
    state  = d["state"]           # (N, T, state_dim)
    ckpt   = d["checkpoint_step"] # (N,)

    N, T, obs_dim = obs.shape
    act_dim       = action.shape[2]
    state_dim     = state.shape[2]

    seg_reward      = reward.sum(axis=1)
    seg_disc_reward = discounted_sum(reward, DISCOUNT)

    row = {
        "pool_N":         N,
        "pool_T":         T,
        "pool_obs_dim":   obs_dim,
        "pool_act_dim":   act_dim,
        "pool_state_dim": state_dim,
    }
    row.update(stats_dict(seg_reward,      "pool_reward"))
    row.update(stats_dict(seg_disc_reward, "pool_disc_reward"))

    unique_ckpts = np.unique(ckpt)
    row["pool_ckpt_n_unique"] = len(unique_ckpts)
    row["pool_ckpt_min"]      = int(unique_ckpts.min())
    row["pool_ckpt_max"]      = int(unique_ckpts.max())

    return row, N


def demo_stats(labels_path, pool_N):
    print(f"  Loading demo labels: {labels_path}")
    d = np.load(labels_path)
    obs        = d["obs"]             # (N, K, T, obs_dim)
    action     = d["action"]          # (N, K, T, act_dim)
    reward     = d["reward"]          # (N, K, T)
    adv_scores = d["adv_scores"]      # (N, K) rl_sum descending
    ckpt       = d["checkpoint_step"] # (N,)

    N, K, T, _ = obs.shape

    skip_rate   = 1.0 - N / pool_N if pool_N > 0 else float("nan")
    pos_reward  = reward.sum(axis=2)   # (N, K) undiscounted reward per candidate
    adv_gap     = adv_scores[:, 0] - adv_scores[:, -1]   # (N,)
    mid_pos     = K // 2

    # Choice set action diversity — subsample up to 500 for speed
    n_sample  = min(N, 500)
    idx       = np.random.choice(N, n_sample, replace=False)
    diversity = float(np.mean([pairwise_action_diversity(action[i]) for i in idx]))

    unique_ckpts = np.unique(ckpt)

    row = {
        "demo_N":         N,
        "demo_K":         K,
        "demo_skip_rate": round(skip_rate, 4),
    }

    # Per-position mean undiscounted reward: demo, mid, worst
    row["demo_pos0_reward_mean"]           = float(pos_reward[:, 0].mean())
    row[f"demo_pos{mid_pos}_reward_mean"]  = float(pos_reward[:, mid_pos].mean())
    row[f"demo_pos{K-1}_reward_mean"]      = float(pos_reward[:, K - 1].mean())

    # rl_sum per position: mean and std (quality gradient)
    for k in range(K):
        row[f"demo_rl_sum_pos{k}_mean"] = float(adv_scores[:, k].mean())
        row[f"demo_rl_sum_pos{k}_std"]  = float(adv_scores[:, k].std())

    # Demo (pos 0) rl_sum full stats
    row.update(stats_dict(adv_scores[:, 0], "demo_rl_sum_pos0"))

    # Advantage gap stats
    row.update(stats_dict(adv_gap, "demo_adv_gap"))

    # Discounted reward for demo position
    demo_disc_reward = discounted_sum(reward[:, 0, :], DISCOUNT)
    row.update(stats_dict(demo_disc_reward, "demo_disc_reward_pos0"))

    # Action diversity
    row["demo_action_diversity_mean"] = diversity

    # Checkpoint coverage of kept segments
    row["demo_ckpt_n_unique"] = len(unique_ckpts)
    row["demo_ckpt_min"]      = int(unique_ckpts.min())
    row["demo_ckpt_max"]      = int(unique_ckpts.max())

    return row


def find_labels_file(labels_dir, env_name):
    """Find demo_labels_K*.npz for the given env."""
    env_dir = os.path.join(labels_dir, env_name)
    if not os.path.isdir(env_dir):
        return None
    for fname in sorted(os.listdir(env_dir)):
        if fname.startswith("demo_labels_K") and fname.endswith(".npz"):
            return os.path.join(env_dir, fname)
    return None


def print_table(rows, env_names):
    keys = [
        ("pool_N",                       "Pool N"),
        ("pool_T",                       "Pool T"),
        ("pool_reward_mean",             "Pool Rew μ"),
        ("pool_reward_std",              "Pool Rew σ"),
        ("pool_disc_reward_mean",        "Pool DiscRew μ"),
        ("pool_disc_reward_std",         "Pool DiscRew σ"),
        ("pool_ckpt_n_unique",           "Pool #Ckpts"),
        ("demo_N",                       "Demo N"),
        ("demo_K",                       "Demo K"),
        ("demo_skip_rate",               "Skip Rate"),
        ("demo_pos0_reward_mean",        "Demo Rew μ"),
        ("demo_rl_sum_pos0_mean",        "rl_sum[0] μ"),
        ("demo_rl_sum_pos0_std",         "rl_sum[0] σ"),
        ("demo_adv_gap_mean",            "AdvGap μ"),
        ("demo_adv_gap_std",             "AdvGap σ"),
        ("demo_adv_gap_min",             "AdvGap min"),
        ("demo_adv_gap_max",             "AdvGap max"),
        ("demo_disc_reward_pos0_mean",   "Demo DiscRew μ"),
        ("demo_action_diversity_mean",   "Diversity μ"),
        ("demo_ckpt_n_unique",           "Demo #Ckpts"),
    ]

    col_w = 14
    env_w = 24

    header = f"{'Env':<{env_w}}" + "".join(f"{label:>{col_w}}" for _, label in keys)
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))

    for env_name, row in zip(env_names, rows):
        if row is None:
            print(f"{env_name:<{env_w}}  (missing)")
            continue
        line = f"{env_name:<{env_w}}"
        for key, _ in keys:
            val = row.get(key, "N/A")
            if isinstance(val, float):
                line += f"{val:>{col_w}.2f}"
            else:
                line += f"{str(val):>{col_w}}"
        print(line)

    print("=" * len(header) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Dataset statistics for pool and demo labels.")
    parser.add_argument(
        "--pool-dir", type=str,
        default="/scratch/general/vast/u1472210/demo_pool",
        help="Root dir containing <env>/pool.npz files.",
    )
    parser.add_argument(
        "--labels-dir", type=str,
        default="/scratch/general/vast/u1472210/demo_labels",
        help="Root dir containing <env>/demo_labels_K*.npz files.",
    )
    parser.add_argument("--envs", nargs="+", default=ENVS_ALL,
                        help="Environments to process (default: all 5).")
    parser.add_argument("--output", type=str, default="results/dataset_stats.csv",
                        help="Output CSV path (default: results/dataset_stats.csv).")
    args = parser.parse_args()

    all_rows  = []
    env_names = []

    for env_name in args.envs:
        print(f"\n{'='*55}")
        print(f"Environment: {env_name}")

        row      = {"env": env_name}
        has_data = False
        pool_N   = 0

        pool_path = os.path.join(args.pool_dir, env_name, "pool.npz")
        if os.path.exists(pool_path):
            p_row, pool_N = pool_stats(pool_path)
            row.update(p_row)
            has_data = True
        else:
            print(f"  [SKIP] pool.npz not found: {pool_path}")

        labels_path = find_labels_file(args.labels_dir, env_name)
        if labels_path:
            row.update(demo_stats(labels_path, pool_N))
            has_data = True
        else:
            print(f"  [SKIP] demo_labels_K*.npz not found in: "
                  f"{os.path.join(args.labels_dir, env_name)}")

        all_rows.append(row if has_data else None)
        env_names.append(env_name)

    print_table(all_rows, env_names)

    valid_rows = [r for r in all_rows if r is not None]
    if valid_rows:
        out_dir = os.path.dirname(args.output)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        # Collect all field names across envs
        fieldnames = ["env"]
        for r in valid_rows:
            for k in r:
                if k not in fieldnames:
                    fieldnames.append(k)
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(valid_rows)
        print(f"Saved → {args.output}")
    else:
        print("No data found — nothing saved.")


if __name__ == "__main__":
    main()
