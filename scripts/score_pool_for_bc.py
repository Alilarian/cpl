"""
Score all segments in a trajectory pool with oracle rl_sum and save the
top-X% as a fixed BC warmup dataset.

No rollouts needed — pool segments are scored directly in batches through the
oracle critic (same V-function used in generate_demo_labels.py and
generate_corr_labels.py).

    rl_sum[i] = Σ_t r_t + V(s_T) - V(s_0)

All N pool segments are scored, then sorted descending by rl_sum.  The top
--top-pct percent are saved as bc_pool.npz.  Because no rollouts are needed,
this is much faster than generating demo/corr labels — the full pool can be
scored in a single GPU pass of ~80 batches.

Output (--output-dir/<env>/bc_pool.npz):
    obs             : (N_bc, T, obs_dim)   top-pct segments by rl_sum
    action          : (N_bc, T, act_dim)
    reward          : (N_bc, T)
    rl_sum          : (N_bc,)              oracle quality score, descending
    checkpoint_step : (N_bc,)             source checkpoint for each segment

Usage (single env):
    python scripts/score_pool_for_bc.py \\
        --pool-path  /scratch/general/vast/u1472210/demo_pool/mw_drawer-open-v2/pool.npz \\
        --run-dir    /scratch/general/vast/u1472210/oracle_sac_all/mw_drawer-open-v2 \\
        --top-pct    20 \\
        --output-dir /scratch/general/vast/u1472210/bc_pool

Usage (all envs, loop):
    for ENV in mw_bin-picking-v2 mw_button-press-v2 mw_door-open-v2 \\
               mw_drawer-open-v2 mw_plate-slide-v2; do
        python scripts/score_pool_for_bc.py \\
            --pool-path  /scratch/general/vast/u1472210/demo_pool/$ENV/pool.npz \\
            --run-dir    /scratch/general/vast/u1472210/oracle_sac_all/$ENV \\
            --top-pct    20 \\
            --output-dir /scratch/general/vast/u1472210/bc_pool
    done
"""

import argparse
import io
import os

import numpy as np
import torch

from research.utils.config import Config


# ---------------------------------------------------------------------------
# Model loading  (identical to generate_corr_labels.py)
# ---------------------------------------------------------------------------

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


def save_npz(path, **arrays):
    """Atomically save a compressed npz (write to .tmp, then rename)."""
    tmp = path + ".tmp"
    with io.BytesIO() as buf:
        np.savez_compressed(buf, **arrays)
        buf.seek(0)
        with open(tmp, "wb") as f:
            f.write(buf.read())
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Batch scoring
# ---------------------------------------------------------------------------

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
    B, T, _ = obs_b.shape
    obs_t    = torch.from_numpy(obs_b).float().to(device)     # (B, T, obs_dim)
    action_t = torch.from_numpy(action_b).float().to(device)  # (B, T, act_dim)
    reward_t = torch.from_numpy(reward_b).float().to(device)  # (B, T)

    with torch.no_grad():
        obs_enc = oracle.network.encoder(obs_t)   # (B, T, enc_dim)

        # MCMC estimate of V(s) at every timestep
        # Expand for mcmc_samples: (B, T, mcmc_samples, enc_dim)
        obs_exp   = obs_enc.unsqueeze(2).expand(-1, -1, mcmc_samples, -1)
        sampled_a = oracle.network.actor(obs_exp).sample()
        v = oracle.network.critic(obs_exp, sampled_a).mean(dim=0)  # (B, T, mcmc_samples)
        v = v.mean(dim=2)                                           # (B, T)

        # rl_sum = Σ_{t=0}^{T-2} r_t + V(s_{T-1}) - V(s_0)
        rl_sum = (
            reward_t[:, :-1].sum(dim=-1) + v[:, -1] - v[:, 0]
        ).cpu().numpy()   # (B,)

    return rl_sum


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Score pool segments with oracle rl_sum and save top-X% for BC."
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
        "--top-pct", type=float, default=20.0,
        help="Percentage of pool segments to keep, selected by highest rl_sum "
             "(default: 20.0 → top 20%%, i.e. 4000 of 20000 segments/env)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=256,
        help="Segments scored per GPU forward pass (default: 256). "
             "Reduce if OOM.",
    )
    parser.add_argument(
        "--mcmc-samples", type=int, default=64,
        help="MCMC samples for V(s) estimation (default: 64)",
    )
    parser.add_argument(
        "--discount", type=float, default=0.99,
        help="Discount factor used in rl_sum (default: 0.99; informational only "
             "— the undiscounted version is the primary sort key here)",
    )
    parser.add_argument(
        "--output-dir", type=str,
        default="/scratch/general/vast/u1472210/bc_pool",
        help="Root output directory.  bc_pool.npz is written to <output-dir>/<env>/.",
    )
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    assert 0 < args.top_pct <= 100, "--top-pct must be in (0, 100]"

    # ------------------------------------------------------------------
    # Derive env name from pool path
    # ------------------------------------------------------------------
    # pool-path is expected to be  <somewhere>/<env>/pool.npz
    env_name = os.path.basename(os.path.dirname(args.pool_path))

    print("=" * 60)
    print(f"Environment  : {env_name}")
    print(f"Pool path    : {args.pool_path}")
    print(f"Oracle dir   : {args.run_dir}")
    print(f"Top %%        : {args.top_pct:.1f}")
    print(f"Batch size   : {args.batch_size}")
    print(f"MCMC samples : {args.mcmc_samples}")
    print(f"Device       : {device}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Guard — skip if already done
    # ------------------------------------------------------------------
    out_dir  = os.path.join(args.output_dir, env_name)
    out_path = os.path.join(out_dir, "bc_pool.npz")
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
    # Score all segments in batches
    # ------------------------------------------------------------------
    print(f"\nScoring {N:,} segments in batches of {args.batch_size} ...")
    all_rl_sum = np.empty(N, dtype=np.float32)
    n_batches  = (N + args.batch_size - 1) // args.batch_size

    for b in range(n_batches):
        start = b * args.batch_size
        end   = min(start + args.batch_size, N)

        obs_b    = pool_obs[start:end]
        action_b = pool_action[start:end]
        reward_b = pool_reward[start:end]

        rl_sum_b = score_pool_batch(
            obs_b, action_b, reward_b,
            oracle, args.discount, args.mcmc_samples, device,
        )
        all_rl_sum[start:end] = rl_sum_b

        if (b + 1) % 10 == 0 or b == n_batches - 1:
            print(f"  [{b+1:>4}/{n_batches}]  scored {end:>6,}/{N:,} segments")

    print(f"\nrl_sum — all {N:,} segments:")
    print(f"  mean={all_rl_sum.mean():.3f}  std={all_rl_sum.std():.3f}"
          f"  min={all_rl_sum.min():.3f}  max={all_rl_sum.max():.3f}")

    # ------------------------------------------------------------------
    # Select top-X% by rl_sum
    # ------------------------------------------------------------------
    n_keep = max(1, int(round(N * args.top_pct / 100.0)))
    # argsort descending; keep top n_keep
    order     = np.argsort(all_rl_sum)[::-1]
    top_inds  = order[:n_keep]

    # Keep sorted order (descending rl_sum) in the output
    bc_obs    = pool_obs[top_inds]
    bc_action = pool_action[top_inds]
    bc_reward = pool_reward[top_inds]
    bc_rl_sum = all_rl_sum[top_inds]
    bc_ckpt   = pool_ckpt[top_inds]

    threshold = bc_rl_sum[-1]   # lowest rl_sum kept
    print(f"\nSelected top {args.top_pct:.1f}%: {n_keep:,} / {N:,} segments")
    print(f"  rl_sum threshold (lowest kept) : {threshold:.3f}")
    print(f"  rl_sum of kept — "
          f"mean={bc_rl_sum.mean():.3f}  std={bc_rl_sum.std():.3f}"
          f"  min={bc_rl_sum.min():.3f}  max={bc_rl_sum.max():.3f}")

    unique_ckpts = np.unique(bc_ckpt)
    print(f"  Checkpoint coverage: {len(unique_ckpts)} unique steps "
          f"(min={unique_ckpts.min():,}  max={unique_ckpts.max():,})")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    os.makedirs(out_dir, exist_ok=True)
    save_npz(
        out_path,
        obs=bc_obs,
        action=bc_action,
        reward=bc_reward,
        rl_sum=bc_rl_sum,
        checkpoint_step=bc_ckpt,
    )
    print(f"\nSaved → {out_path}")
    print(f"  obs    : {bc_obs.shape}")
    print(f"  action : {bc_action.shape}")
    print(f"  reward : {bc_reward.shape}")
    print(f"  rl_sum : {bc_rl_sum.shape}")


if __name__ == "__main__":
    main()
