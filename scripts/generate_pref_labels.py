"""
Generate pairwise preference labels directly from a trajectory pool.

This implements the comparative feedback generation from multi-type-feedback
(generate_feedback.py / get_preference_pairs) adapted to CPL's pipeline.

Each pool segment is scored with oracle rl_sum = Σ r_t + V(s_T) - V(s_0).
Then pairs are formed by random sampling: pairs where the oracle score gap
exceeds a threshold are kept, sorted by gap magnitude, and the top n_pairs
are saved.

No expert rollouts are needed — segments already exist in the pool.
This is much faster than generate_demo_labels or generate_corr_labels because
the only GPU work is a single batch scoring pass over the pool.

Relationship to multi-type-feedback
------------------------------------
multi-type-feedback scores with raw discounted return and uses tolerance=std/10.
We use oracle rl_sum (same metric as the rest of the CPL pipeline) with the
same std/10 default threshold.

Output format is identical to corr_labels.npz (K=2), so CorrBuffer can be
used directly to load pref_labels.npz for training.

Output (--output-dir/<env>/pref_labels.npz):
    obs             : (N, 2, T, obs_dim)   [0]=better segment, [1]=worse segment
    action          : (N, 2, T, act_dim)
    reward          : (N, 2, T)
    adv_scores      : (N, 2)               [better_rl_sum, worse_rl_sum]
    gap             : (N,)                 adv_scores[:,0] - adv_scores[:,1] > 0
    checkpoint_step : (N, 2)              source checkpoint for each segment

Usage:
    python scripts/generate_pref_labels.py \\
        --pool-path  /scratch/general/vast/u1472210/demo_pool/mw_drawer-open-v2/pool.npz \\
        --run-dir    /scratch/general/vast/u1472210/oracle_sac_all/mw_drawer-open-v2 \\
        --n-pairs    20000 \\
        --output-dir /scratch/general/vast/u1472210/pref_labels

Usage (all envs, loop):
    for ENV in mw_bin-picking-v2 mw_button-press-v2 mw_door-open-v2 \\
               mw_drawer-open-v2 mw_plate-slide-v2; do
        python scripts/generate_pref_labels.py \\
            --pool-path  /scratch/general/vast/u1472210/demo_pool/$ENV/pool.npz \\
            --run-dir    /scratch/general/vast/u1472210/oracle_sac_all/$ENV \\
            --output-dir /scratch/general/vast/u1472210/pref_labels
    done
"""

import argparse
import io
import os

import numpy as np
import torch

from research.utils.config import Config


# ---------------------------------------------------------------------------
# Utilities (shared with other generate_* scripts)
# ---------------------------------------------------------------------------

def save_npz(path, **arrays):
    """Atomically save a compressed npz (write to .tmp, then rename)."""
    tmp = path + ".tmp"
    with io.BytesIO() as buf:
        np.savez_compressed(buf, **arrays)
        buf.seek(0)
        with open(tmp, "wb") as f:
            f.write(buf.read())
    os.replace(tmp, path)


def load_model(run_dir, checkpoint_path, device):
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
    return model


def score_pool_batch(obs_b, action_b, reward_b, oracle, discount, mcmc_samples, device):
    """
    Score a batch of B segments with oracle rl_sum.

    Args:
        obs_b    : (B, T, obs_dim)  numpy
        action_b : (B, T, act_dim)  numpy
        reward_b : (B, T)           numpy

    Returns:
        rl_sum : (B,)  numpy  — Σ r_t + V(s_T) - V(s_0)
    """
    obs_t    = torch.from_numpy(obs_b).float().to(device)
    action_t = torch.from_numpy(action_b).float().to(device)
    reward_t = torch.from_numpy(reward_b).float().to(device)

    T = obs_b.shape[1]

    with torch.no_grad():
        obs_enc = oracle.network.encoder(obs_t)

        obs_exp   = obs_enc.unsqueeze(2).expand(-1, -1, mcmc_samples, -1)
        sampled_a = oracle.network.actor(obs_exp).sample()
        v = oracle.network.critic(obs_exp, sampled_a).mean(dim=0)
        v = v.mean(dim=2)   # (B, T)

        rl_sum = (
            reward_t[:, :-1].sum(dim=-1) + v[:, -1] - v[:, 0]
        ).cpu().numpy()

    return rl_sum


# ---------------------------------------------------------------------------
# Pair sampling
# ---------------------------------------------------------------------------

def sample_pairs(N, n_candidates, rng):
    """
    Sample n_candidates random index pairs (i, j) with i != j from [0, N).

    Returns:
        idx_a : (n_candidates,)
        idx_b : (n_candidates,)
    """
    idx_a = rng.integers(0, N, size=n_candidates)
    idx_b = rng.integers(0, N, size=n_candidates)
    # Avoid self-pairs
    same = idx_a == idx_b
    while same.any():
        idx_b[same] = rng.integers(0, N, size=same.sum())
        same = idx_a == idx_b
    return idx_a, idx_b


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate pairwise preference labels from a scored trajectory pool."
    )
    parser.add_argument(
        "--pool-path", type=str, required=True,
        help="Path to pool.npz from build_trajectory_pool.py",
    )
    parser.add_argument(
        "--run-dir", type=str, required=True,
        help="Oracle SAC run dir (config.yaml + best_model.pt)",
    )
    parser.add_argument(
        "--oracle-checkpoint", type=str, default="best_model.pt",
        help="Oracle checkpoint filename (default: best_model.pt)",
    )
    parser.add_argument(
        "--n-pairs", type=int, default=20000,
        help="Number of preference pairs to generate (default: 20000). "
             "Matches pool size for 1:1 ratio with the pool.",
    )
    parser.add_argument(
        "--min-gap-fraction", type=float, default=0.0,
        help="Pairs with |rl_sum_a - rl_sum_b| < min_gap_fraction × std(rl_sum) "
             "are discarded as ambiguous (default: 0.0 — keep all pairs regardless "
             "of gap size, identical to the corrective labels convention).",
    )
    parser.add_argument(
        "--oversampling-factor", type=float, default=3.0,
        help="Sample this many candidate pairs before filtering "
             "(default: 3.0 — ensures enough pairs survive the gap filter).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=256,
        help="Segments scored per GPU forward pass (default: 256). Reduce if OOM.",
    )
    parser.add_argument(
        "--mcmc-samples", type=int, default=64,
        help="MCMC samples for V(s) estimation (default: 64)",
    )
    parser.add_argument(
        "--discount", type=float, default=0.99,
    )
    parser.add_argument(
        "--sort-by-gap", action="store_true", default=True,
        help="Sort output pairs by gap magnitude descending (most informative first, "
             "default: True — matching multi-type-feedback's improvement-sort step).",
    )
    parser.add_argument(
        "--no-sort", dest="sort_by_gap", action="store_false",
        help="Disable sorting by gap magnitude.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for pair sampling (default: 42)",
    )
    parser.add_argument(
        "--output-dir", type=str,
        default="/scratch/general/vast/u1472210/pref_labels",
    )
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    env_name = os.path.basename(os.path.dirname(args.pool_path))

    print("=" * 60)
    print(f"Environment      : {env_name}")
    print(f"Pool path        : {args.pool_path}")
    print(f"Oracle dir       : {args.run_dir}")
    print(f"Target n_pairs   : {args.n_pairs:,}")
    print(f"Min gap fraction : {args.min_gap_fraction}  (threshold = frac × std(rl_sum))")
    print(f"Oversampling     : {args.oversampling_factor}×")
    print(f"Batch size       : {args.batch_size}")
    print(f"MCMC samples     : {args.mcmc_samples}")
    print(f"Seed             : {args.seed}")
    print(f"Device           : {device}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Guard
    # ------------------------------------------------------------------
    out_dir  = os.path.join(args.output_dir, env_name)
    out_path = os.path.join(out_dir, "pref_labels.npz")
    if os.path.exists(out_path):
        print(f"\nOutput already exists: {out_path}")
        print("Delete it to regenerate.")
        return

    # ------------------------------------------------------------------
    # Load pool
    # ------------------------------------------------------------------
    print(f"\nLoading pool ...")
    with open(args.pool_path, "rb") as f:
        pool = np.load(f)
        pool_obs    = pool["obs"]             # (N, T, obs_dim)
        pool_action = pool["action"]          # (N, T, act_dim)
        pool_reward = pool["reward"]          # (N, T)
        pool_ckpt   = pool["checkpoint_step"] # (N,)

    N, T, obs_dim = pool_obs.shape
    act_dim = pool_action.shape[2]
    print(f"  N={N:,} segments, T={T}, obs_dim={obs_dim}, act_dim={act_dim}")

    # ------------------------------------------------------------------
    # Load oracle
    # ------------------------------------------------------------------
    oracle_ckpt = os.path.join(args.run_dir, args.oracle_checkpoint)
    print(f"\nLoading oracle: {oracle_ckpt}")
    oracle = load_model(args.run_dir, oracle_ckpt, device)
    oracle.eval()

    # ------------------------------------------------------------------
    # Score all pool segments with oracle rl_sum
    # ------------------------------------------------------------------
    print(f"\nScoring all {N:,} segments in batches of {args.batch_size} ...")
    rl_sum_all = np.empty(N, dtype=np.float32)
    n_batches  = (N + args.batch_size - 1) // args.batch_size

    for b in range(n_batches):
        start = b * args.batch_size
        end   = min(start + args.batch_size, N)
        rl_sum_all[start:end] = score_pool_batch(
            pool_obs[start:end],
            pool_action[start:end],
            pool_reward[start:end],
            oracle, args.discount, args.mcmc_samples, device,
        )
        if (b + 1) % 10 == 0 or b == n_batches - 1:
            print(f"  [{b+1:>4}/{n_batches}]  scored {end:>6,}/{N:,}")

    rl_sum_std  = float(np.std(rl_sum_all))
    rl_sum_mean = float(np.mean(rl_sum_all))
    threshold   = args.min_gap_fraction * rl_sum_std

    print(f"\nrl_sum over all {N:,} segments:")
    print(f"  mean={rl_sum_mean:.3f}  std={rl_sum_std:.3f}"
          f"  min={rl_sum_all.min():.3f}  max={rl_sum_all.max():.3f}")
    print(f"Gap threshold: {args.min_gap_fraction} × {rl_sum_std:.3f} = {threshold:.3f}")

    # ------------------------------------------------------------------
    # Sample candidate pairs and filter by gap
    # ------------------------------------------------------------------
    n_candidates = int(args.n_pairs * args.oversampling_factor)
    rng = np.random.default_rng(args.seed)

    print(f"\nSampling {n_candidates:,} candidate pairs ...")
    idx_a, idx_b = sample_pairs(N, n_candidates, rng)

    score_a = rl_sum_all[idx_a]
    score_b = rl_sum_all[idx_b]
    gap = score_a - score_b   # signed: positive means a is better

    # Keep only pairs with |gap| above threshold
    keep = np.abs(gap) >= threshold
    idx_a  = idx_a[keep]
    idx_b  = idx_b[keep]
    gap    = gap[keep]

    n_kept = len(idx_a)
    print(f"  Candidates sampled : {n_candidates:,}")
    print(f"  After gap filter   : {n_kept:,}  "
          f"(filter rate: {1 - n_kept/n_candidates:.1%})")

    if n_kept < args.n_pairs:
        print(f"\nWARNING: only {n_kept:,} pairs survive the gap filter "
              f"(target was {args.n_pairs:,}).")
        print("  Options: increase --oversampling-factor, lower --min-gap-fraction, "
              "or accept fewer pairs.")
        n_out = n_kept
    else:
        n_out = args.n_pairs

    # Sort by |gap| descending (most informative first) then truncate
    order = np.argsort(np.abs(gap))[::-1]
    idx_a = idx_a[order][:n_out]
    idx_b = idx_b[order][:n_out]
    gap   = gap[order][:n_out]

    # ------------------------------------------------------------------
    # Build output arrays: index 0 = better (higher rl_sum)
    # ------------------------------------------------------------------
    # For pairs where gap > 0, a is better → a at index 0
    # For pairs where gap < 0, b is better → b at index 0
    a_is_better = gap >= 0

    better_idx = np.where(a_is_better, idx_a, idx_b)   # (N_out,)
    worse_idx  = np.where(a_is_better, idx_b, idx_a)   # (N_out,)

    print(f"\nBuilding output arrays for {n_out:,} pairs ...")
    out_obs     = np.stack([pool_obs[better_idx],    pool_obs[worse_idx]],    axis=1)  # (N, 2, T, obs)
    out_action  = np.stack([pool_action[better_idx], pool_action[worse_idx]], axis=1)
    out_reward  = np.stack([pool_reward[better_idx], pool_reward[worse_idx]], axis=1)
    out_adv     = np.stack([rl_sum_all[better_idx],  rl_sum_all[worse_idx]],  axis=1)  # (N, 2)
    out_gap     = np.abs(gap[:n_out]).astype(np.float32)
    out_ckpt    = np.stack([pool_ckpt[better_idx],   pool_ckpt[worse_idx]],   axis=1)  # (N, 2)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\nGap statistics (better_rl_sum - worse_rl_sum):")
    print(f"  mean={out_gap.mean():.3f}  std={out_gap.std():.3f}"
          f"  min={out_gap.min():.3f}  max={out_gap.max():.3f}")

    unique_ckpts = np.unique(out_ckpt)
    print(f"Checkpoint coverage: {len(unique_ckpts)} unique steps "
          f"(min={unique_ckpts.min():,}  max={unique_ckpts.max():,})")

    # Sanity check: adv_scores[:,0] should always be >= adv_scores[:,1]
    assert np.all(out_adv[:, 0] >= out_adv[:, 1]), \
        "BUG: index 0 is not always the better trajectory!"

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    os.makedirs(out_dir, exist_ok=True)
    save_npz(
        out_path,
        obs=out_obs,
        action=out_action,
        reward=out_reward,
        adv_scores=out_adv,
        gap=out_gap,
        checkpoint_step=out_ckpt,
    )
    print(f"\nSaved → {out_path}")
    print(f"  obs             : {out_obs.shape}")
    print(f"  action          : {out_action.shape}")
    print(f"  reward          : {out_reward.shape}")
    print(f"  adv_scores      : {out_adv.shape}  ([better_rl_sum, worse_rl_sum])")
    print(f"  gap             : {out_gap.shape}")
    print(f"  checkpoint_step : {out_ckpt.shape}")
    print(f"\nLoad with: CorrBuffer(path='.../{env_name}/pref_labels.npz', ...)")


if __name__ == "__main__":
    main()
