"""
Generate credit assignment feedback labels for MetaWorld environments.

For each pool trajectory, all contiguous subsegments of length k are scored
with oracle rl_sum.  The oracle selects the highest-scoring window as the
"credited" subsegment (argmax).  This is the MetaWorld analogue of the PM
credit_assignment type in generate_pm_feedback.py.

Efficient implementation: V(s_t) is computed once for every obs in the pool
via batched MCMC, then all C = T-k+1 window scores per trajectory are
computed analytically from prefix reward sums and the precomputed V array.
This costs the same as scoring the full pool once, regardless of C.

Window score formula (consistent with score_pool_batch / generate_pref_labels):
    window_score[n, i] = Σ_{t=i}^{i+k-2} r[n,t] + V[n, i+k-1] - V[n, i]

Output (--output-dir/<env>/credit_labels.npz):
    obs         : (N, C, k, obs_dim)  all candidate windows
    action      : (N, C, k, act_dim)
    adv_scores  : (N, C)              oracle rl_sum per candidate
    chosen_idx  : (N,) int32          argmax window per trajectory
    checkpoint_step : (N,)

Dataset : PMCreditAssignmentBuffer
Alg     : CreditAssignmentCPL  (bc_steps=0 for no warmup)

Usage:
    python scripts/generate_credit_labels.py \\
        --pool-path  /scratch/.../demo_pool/mw_drawer-open-v2/pool.npz \\
        --run-dir    /scratch/.../oracle_sac_all/mw_drawer-open-v2 \\
        --subsegment-len 20 \\
        --n-examples 300000 \\
        --output-dir /scratch/.../credit_labels

Parameters:
    --subsegment-len k : length of each candidate window (default: 20)
    --n-examples       : cap on output examples (default: no cap)
"""

import argparse
import io
import os

import numpy as np
import torch

from research.utils.config import Config


# ---------------------------------------------------------------------------
# Utilities
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


def compute_v_batch(obs_b, oracle, mcmc_samples, device):
    """
    Compute V(s_t) at every timestep for a batch of B segments.

    Args:
        obs_b : (B, T, obs_dim)  numpy

    Returns:
        v : (B, T)  numpy — V(obs_b[:, t, :]) for each t
    """
    obs_t = torch.from_numpy(obs_b).float().to(device)  # (B, T, obs_dim)

    with torch.no_grad():
        obs_enc   = oracle.network.encoder(obs_t)                          # (B, T, D)
        obs_exp   = obs_enc.unsqueeze(2).expand(-1, -1, mcmc_samples, -1) # (B, T, M, D)
        sampled_a = oracle.network.actor(obs_exp).sample()                 # (B, T, M, act_dim)
        q         = oracle.network.critic(obs_exp, sampled_a).mean(dim=0) # (B, T, M)
        v         = q.mean(dim=2)                                          # (B, T)

    return v.cpu().numpy()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate credit assignment labels for a MetaWorld environment."
    )
    parser.add_argument("--pool-path", type=str, required=True,
                        help="Path to pool.npz from build_trajectory_pool.py")
    parser.add_argument("--run-dir", type=str, required=True,
                        help="Oracle SAC run dir (config.yaml + checkpoint)")
    parser.add_argument("--oracle-checkpoint", type=str, default="best_model.pt",
                        help="Oracle checkpoint filename (default: best_model.pt)")
    parser.add_argument("--subsegment-len", type=int, default=20,
                        help="Window length k (default: 20); C = T-k+1 candidates per traj")
    parser.add_argument("--n-examples", type=int, default=None,
                        help="Cap on output examples (pool trajectories; default: all)")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Segments per GPU forward pass for V computation (default: 256)")
    parser.add_argument("--mcmc-samples", type=int, default=64,
                        help="MCMC samples for V(s) estimation (default: 64)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for subsampling if --n-examples < N (default: 42)")
    parser.add_argument("--output-dir", type=str,
                        default="/scratch/general/vast/u1472210/credit_labels")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    env_name = os.path.basename(os.path.dirname(args.pool_path))
    out_dir  = os.path.join(args.output_dir, env_name)
    out_path = os.path.join(out_dir, "credit_labels.npz")

    if os.path.exists(out_path):
        print(f"\nOutput already exists: {out_path}")
        print("Delete it to regenerate.")
        return

    print("=" * 65)
    print(f"Environment      : {env_name}")
    print(f"Pool path        : {args.pool_path}")
    print(f"Oracle dir       : {args.run_dir}")
    print(f"Subsegment len k : {args.subsegment_len}")
    print(f"N-examples cap   : {args.n_examples if args.n_examples else 'no cap'}")
    print(f"Batch size       : {args.batch_size}")
    print(f"MCMC samples     : {args.mcmc_samples}")
    print(f"Device           : {device}")
    print("=" * 65)

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
    k = args.subsegment_len
    C = T - k + 1

    print(f"  N={N:,} segments  T={T}  obs_dim={obs_dim}  act_dim={act_dim}")
    print(f"  k={k}  C={C} candidates per trajectory")

    if k >= T:
        print(f"ERROR: --subsegment-len {k} must be < T={T}")
        return
    if C < 2:
        print(f"ERROR: only {C} window — need C >= 2 (reduce k or use longer pool segments)")
        return

    # Subsample if requested (random, seed-controlled)
    if args.n_examples is not None and N > args.n_examples:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(N, size=args.n_examples, replace=False)
        idx.sort()
        pool_obs    = pool_obs[idx]
        pool_action = pool_action[idx]
        pool_reward = pool_reward[idx]
        pool_ckpt   = pool_ckpt[idx]
        N = args.n_examples
        print(f"  Subsampled to {N:,} examples (seed={args.seed})")

    # ------------------------------------------------------------------
    # Load oracle
    # ------------------------------------------------------------------
    oracle_ckpt = os.path.join(args.run_dir, args.oracle_checkpoint)
    print(f"\nLoading oracle: {oracle_ckpt}")
    oracle = load_model(args.run_dir, oracle_ckpt, device)

    # ------------------------------------------------------------------
    # Compute V(s_t) for all N×T observations (batched)
    #
    # V_all[n, t] = V(pool_obs[n, t, :]) — value at timestep t of traj n.
    # Requires the same compute as scoring the full pool once.
    # ------------------------------------------------------------------
    print(f"\nComputing V at all {N*T:,} pool observations (batched by segment) ...")
    V_all = np.empty((N, T), dtype=np.float32)
    n_batches = (N + args.batch_size - 1) // args.batch_size

    for b in range(n_batches):
        s = b * args.batch_size
        e = min(s + args.batch_size, N)
        V_all[s:e] = compute_v_batch(pool_obs[s:e], oracle, args.mcmc_samples, device)
        if (b + 1) % 20 == 0 or b == n_batches - 1:
            print(f"  [{b+1:>4}/{n_batches}]  segments {e:>7,}/{N:,}")

    print(f"  V stats: mean={V_all.mean():.3f}  std={V_all.std():.3f}"
          f"  min={V_all.min():.3f}  max={V_all.max():.3f}")

    # ------------------------------------------------------------------
    # Compute window scores analytically
    #
    # window_score[n, i] = Σ_{t=i}^{i+k-2} r[n,t] + V[n, i+k-1] - V[n, i]
    #   (consistent with score_pool_batch which drops the last reward)
    #
    # Using prefix sums: reward_prefix[n, t] = Σ_{s=0}^{t-1} r[n,s]
    # ------------------------------------------------------------------
    print(f"\nComputing window scores for all {N:,} × {C} = {N*C:,} windows ...")

    reward_prefix = np.zeros((N, T + 1), dtype=np.float64)
    reward_prefix[:, 1:] = np.cumsum(pool_reward.astype(np.float64), axis=1)

    # window_score[n, i] = (prefix[n, i+k-1] - prefix[n, i]) + V[n, i+k-1] - V[n, i]
    win_scores = np.empty((N, C), dtype=np.float32)
    for i in range(C):
        win_scores[:, i] = (
            (reward_prefix[:, i + k - 1] - reward_prefix[:, i]).astype(np.float32)
            + V_all[:, i + k - 1] - V_all[:, i]
        )

    # Oracle: argmax window per trajectory
    chosen_idx = np.argmax(win_scores, axis=1).astype(np.int32)  # (N,)

    # ------------------------------------------------------------------
    # Build observation and action windows
    # ------------------------------------------------------------------
    print(f"\nBuilding {N:,} × {C} × {k} observation windows ...")
    all_win_obs = np.stack(
        [pool_obs[:, i:i + k] for i in range(C)], axis=1
    ).astype(np.float32)                                          # (N, C, k, obs_dim)
    all_win_act = np.stack(
        [pool_action[:, i:i + k] for i in range(C)], axis=1
    ).astype(np.float32)                                          # (N, C, k, act_dim)

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------
    chosen_scores = win_scores[np.arange(N), chosen_idx]         # (N,)
    mean_scores   = win_scores.mean(axis=1)
    min_scores    = win_scores.min(axis=1)
    gap_vs_mean   = chosen_scores - mean_scores
    gap_vs_min    = chosen_scores - min_scores
    pct           = [5, 25, 50, 75, 95]

    print(f"\n{'─'*60}")
    print(f"Credit assignment statistics")
    print(f"{'─'*60}")
    print(f"  Examples      : {N:,}  (1 per pool trajectory)")
    print(f"  Candidates/ex : C={C}  (T={T}, k={k})")

    sv = np.percentile(win_scores.ravel(), pct)
    print(f"\n  Window rl_sum (all {N*C:,} windows):")
    print(f"    mean={win_scores.mean():.3f}  std={win_scores.std():.3f}  "
          f"min={win_scores.min():.3f}  max={win_scores.max():.3f}")
    print(f"    {'  '.join(f'p{p}={v:.3f}' for p, v in zip(pct, sv))}")

    cv = np.percentile(chosen_scores, pct)
    print(f"\n  Chosen window rl_sum:")
    print(f"    mean={chosen_scores.mean():.3f}  std={chosen_scores.std():.3f}")
    print(f"    {'  '.join(f'p{p}={v:.3f}' for p, v in zip(pct, cv))}")

    gm = np.percentile(gap_vs_mean, pct)
    gi = np.percentile(gap_vs_min,  pct)
    print(f"\n  Gap (chosen vs mean / min):")
    print(f"    vs_mean: mean={gap_vs_mean.mean():.3f}  "
          f"{'  '.join(f'p{p}={v:.3f}' for p, v in zip(pct, gm))}")
    print(f"    vs_min:  mean={gap_vs_min.mean():.3f}   "
          f"{'  '.join(f'p{p}={v:.3f}' for p, v in zip(pct, gi))}")

    pp = np.percentile(chosen_idx, pct)
    print(f"\n  Chosen window position (0=earliest, {C-1}=latest):")
    print(f"    mean={chosen_idx.mean():.1f}  std={chosen_idx.std():.1f}")
    print(f"    {'  '.join(f'p{p}={v:.0f}' for p, v in zip(pct, pp))}")
    print(f"{'─'*60}")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    os.makedirs(out_dir, exist_ok=True)
    save_npz(
        out_path,
        obs=all_win_obs,
        action=all_win_act,
        adv_scores=win_scores,
        chosen_idx=chosen_idx,
        checkpoint_step=pool_ckpt.astype(np.int64),
    )

    print(f"\nSaved → {out_path}")
    print(f"  obs         : {all_win_obs.shape}  (N, C, k, obs_dim)")
    print(f"  action      : {all_win_act.shape}")
    print(f"  adv_scores  : {win_scores.shape}  rl_sum per candidate window")
    print(f"  chosen_idx  : {chosen_idx.shape}  oracle argmax (0-based)")
    print(f"\nTrain with:")
    print(f"  dataset: PMCreditAssignmentBuffer")
    print(f"  alg:     CreditAssignmentCPL")
    print(f"  alg_kwargs:")
    print(f"    bc_steps: 0")


if __name__ == "__main__":
    main()
