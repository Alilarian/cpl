"""
Generate scalar feedback labels for MetaWorld environments.

Per-trajectory temporal-subsegment scalar feedback → local hard preferences.
Every pool trajectory is processed independently — a preference NEVER compares
subsegments from two different trajectories (traj(A) == traj(B) always).

Pipeline per trajectory
------------------------
  1. Split the T-step trajectory into K overlapping temporal subsegments of
     length h, starting at every multiple of sub_stride
     (starts = arange(0, T-h+1, sub_stride)).
  2. Score every subsegment with oracle rl_sum = Σ r_t + V(s_T) - V(s_0),
     where V(s) ≈ mean_m Q(s, a_m), a_m ~ π(·|s) (the SAC oracle critic).
  3. Normalize ALL subsegment scores GLOBALLY (1st/99th percentile over the
     whole dataset, not per trajectory) to f ∈ [-1, 1].
  4. Optionally add Gaussian noise (noise_std; default 0.0 → exact oracle).
  5. Slide a comparison window of W consecutive subsegments (stride
     cmp_stride) over the K subsegments. Within a window, candidate pairs
     (i, j) are kept only if min_lag <= j-i <= max_lag (drops neighboring/
     too-distant subsegments); at most max_pairs_per_window are sampled.
  6. Each sampled pair becomes a hard preference: f_i vs f_j, discarding only
     exact ties (|f_i - f_j| <= scalar_delta).

Output format is K=2 compatible with CorrBuffer (preferred at index 0,
label always 0), matching the layout of pref_labels.npz so that DemoCPL
with contrastive_bias=0.5 and bc_steps=0 can train directly.

Output (--output-dir/<env>/scalar_labels.npz):
    obs             : (M, 2, h, obs_dim)   [0]=preferred, [1]=non-preferred
    action          : (M, 2, h, act_dim)
    reward          : (M, 2, h)
    adv_scores      : (M, 2)               [f_preferred, f_non-preferred]
    checkpoint_step : (M, 2)
    traj_id         : (M,)                 pool row each pair was drawn from
    start_idx       : (M, 2)               [start_preferred, start_non_preferred]

Usage:
    python scripts/generate_scalar_labels.py \\
        --pool-path  /scratch/.../demo_pool/mw_drawer-open-v2/pool.npz \\
        --run-dir    /scratch/.../oracle_sac_seeds/mw_drawer-open-v2 \\
        --segment-len 16 --sub-stride 12 --window-size 5 --cmp-stride 5 \\
        --min-lag 2 --max-lag 4 --max-pairs-per-window 4 \\
        --noise-std 0.0 --scalar-delta 0.0 \\
        --output-dir /scratch/.../scalar_labels

Parameters:
    --segment-len h        : subsegment length in steps (default: 16)
    --sub-stride           : stride between subsegment starts (default: 12)
    --window-size W        : subsegments per comparison window (default: 5)
    --cmp-stride           : stride between comparison windows (default: 5)
    --min-lag / --max-lag  : allowed subsegment index gap j-i (default: 2 / 4)
    --max-pairs-per-window : cap on sampled pairs per window (default: 4)
    --noise-std            : Gaussian noise std on normalized f (default: 0.0)
    --scalar-delta         : tie threshold δ; discard |f_i-f_j|<=δ (default: 0.0)
    --n-pairs              : cap on output pairs via random subsample (default: no cap)
"""

import argparse
import io
import os

import numpy as np
import torch

from research.utils.config import Config


# ---------------------------------------------------------------------------
# Shared utilities (mirrors generate_pref_labels.py)
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


def score_pool_batch(obs_b, action_b, reward_b, oracle, mcmc_samples, device):
    """
    Score a batch of B segments with oracle rl_sum = Σ r_t + V(s_T) - V(s_0).

    Args:
        obs_b    : (B, T, obs_dim)  numpy
        action_b : (B, T, act_dim)  numpy  (unused; kept for API compatibility)
        reward_b : (B, T)           numpy

    Returns:
        rl_sum : (B,)  numpy
    """
    obs_t    = torch.from_numpy(obs_b).float().to(device)
    reward_t = torch.from_numpy(reward_b).float().to(device)

    with torch.no_grad():
        obs_enc   = oracle.network.encoder(obs_t)                                 # (B, T, D)
        obs_exp   = obs_enc.unsqueeze(2).expand(-1, -1, mcmc_samples, -1)        # (B, T, M, D)
        sampled_a = oracle.network.actor(obs_exp).sample()                        # (B, T, M, act_dim)
        v         = oracle.network.critic(obs_exp, sampled_a).mean(dim=0)        # (B, T, M)
        v         = v.mean(dim=2)                                                 # (B, T)

        rl_sum = (reward_t[:, :-1].sum(dim=-1) + v[:, -1] - v[:, 0]).cpu().numpy()

    return rl_sum  # (B,)


def score_all_subsegments(seg_obs, seg_act, seg_rew, oracle, mcmc_samples, device, batch_size):
    """
    Score every (trajectory, subsegment) with oracle rl_sum, batched for GPU memory.

    seg_obs/seg_act/seg_rew : (N, K, h, dim) / (N, K, h)
    returns: (N, K) float32
    """
    N, K, h = seg_rew.shape
    flat_obs = seg_obs.reshape(N * K, h, seg_obs.shape[-1])
    flat_act = seg_act.reshape(N * K, h, seg_act.shape[-1])
    flat_rew = seg_rew.reshape(N * K, h)

    n_total   = N * K
    n_batches = (n_total + batch_size - 1) // batch_size
    flat_scores = np.empty(n_total, dtype=np.float32)

    for b in range(n_batches):
        s = b * batch_size
        e = min(s + batch_size, n_total)
        flat_scores[s:e] = score_pool_batch(
            flat_obs[s:e], flat_act[s:e], flat_rew[s:e],
            oracle, mcmc_samples, device,
        )
        if (b + 1) % 50 == 0 or b == n_batches - 1:
            print(f"  [{b+1:>4}/{n_batches}]  scored {e:>9,}/{n_total:,}")

    return flat_scores.reshape(N, K)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate scalar feedback labels for a MetaWorld environment."
    )
    parser.add_argument("--pool-path", type=str, required=True,
                        help="Path to pool.npz from build_trajectory_pool.py")
    parser.add_argument("--run-dir", type=str, required=True,
                        help="Oracle SAC run dir (config.yaml + checkpoint)")
    parser.add_argument("--oracle-checkpoint", type=str, default="best_model.pt",
                        help="Oracle checkpoint filename (default: best_model.pt)")

    # Temporal subsegments
    parser.add_argument("--segment-len", type=int, default=16,
                        help="Subsegment length h in steps (default: 16)")
    parser.add_argument("--sub-stride", type=int, default=12,
                        help="Stride between subsegment start offsets (default: 12); "
                             "starts = arange(0, T-h+1, sub_stride)")

    # Local comparison
    parser.add_argument("--window-size", type=int, default=5,
                        help="Subsegments K per comparison window W (default: 5)")
    parser.add_argument("--cmp-stride", type=int, default=5,
                        help="Stride between comparison windows over the K "
                             "subsegments of a trajectory (default: 5)")
    parser.add_argument("--min-lag", type=int, default=2,
                        help="Minimum subsegment index gap j-i for a candidate "
                             "pair (default: 2; drops neighboring subsegments)")
    parser.add_argument("--max-lag", type=int, default=4,
                        help="Maximum subsegment index gap j-i for a candidate "
                             "pair (default: 4)")
    parser.add_argument("--max-pairs-per-window", type=int, default=4,
                        help="Cap on candidate pairs sampled per comparison "
                             "window, per trajectory (default: 4)")

    # Scalar feedback
    parser.add_argument("--noise-std", type=float, default=0.0,
                        help="Gaussian noise std added to the normalized "
                             "oracle scalar (default: 0.0, exact oracle score)")
    parser.add_argument("--scalar-delta", type=float, default=0.0,
                        help="Indifference threshold δ: discard a pair only "
                             "when |f_i-f_j| <= δ (default: 0.0, exact ties only)")

    parser.add_argument("--n-pairs", type=int, default=None,
                        help="Cap on output pairs; random subsample if exceeded (default: no cap)")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Segments per GPU forward pass (default: 256)")
    parser.add_argument("--mcmc-samples", type=int, default=64,
                        help="MCMC samples for V(s) estimation (default: 64)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str,
                        default="/scratch/general/vast/u1472210/scalar_labels")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    env_name = os.path.basename(os.path.dirname(args.pool_path))
    out_dir  = os.path.join(args.output_dir, env_name)
    out_path = os.path.join(out_dir, "scalar_labels.npz")

    if os.path.exists(out_path):
        print(f"\nOutput already exists: {out_path}")
        print("Delete it to regenerate.")
        return

    print("=" * 65)
    print(f"Environment          : {env_name}")
    print(f"Pool path            : {args.pool_path}")
    print(f"Oracle dir           : {args.run_dir}")
    print(f"segment_len (h)      : {args.segment_len}")
    print(f"sub_stride           : {args.sub_stride}")
    print(f"window_size (W)      : {args.window_size}")
    print(f"cmp_stride           : {args.cmp_stride}")
    print(f"min_lag / max_lag    : {args.min_lag} / {args.max_lag}")
    print(f"max_pairs_per_window : {args.max_pairs_per_window}")
    print(f"noise_std            : {args.noise_std}")
    print(f"scalar_delta         : {args.scalar_delta}")
    print(f"N-pairs cap          : {args.n_pairs if args.n_pairs else 'no cap'}")
    print(f"Batch size           : {args.batch_size}")
    print(f"MCMC samples         : {args.mcmc_samples}")
    print(f"Device               : {device}")
    print("=" * 65)

    rng = np.random.default_rng(args.seed)

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
    h = args.segment_len
    print(f"  N={N:,} trajectories  T={T}  obs_dim={obs_dim}  act_dim={act_dim}")

    if h > T:
        print(f"ERROR: --segment-len {h} > pool T={T}")
        return

    # ------------------------------------------------------------------
    # Step 1: temporal subsegments (same start offsets for every trajectory)
    # ------------------------------------------------------------------
    starts = np.arange(0, T - h + 1, args.sub_stride)
    K = len(starts)
    W = args.window_size
    if W > K:
        print(f"ERROR: --window-size {W} exceeds K={K} subsegments "
              f"(T={T}, h={h}, sub_stride={args.sub_stride})")
        return

    print(f"\nTemporal subsegments: T={T}  h={h}  sub_stride={args.sub_stride}  "
          f"→ K={K}  starts={starts.tolist()}")

    seg_obs = np.stack([pool_obs[:, s:s + h]    for s in starts], axis=1)  # (N, K, h, obs_dim)
    seg_act = np.stack([pool_action[:, s:s + h] for s in starts], axis=1)  # (N, K, h, act_dim)
    seg_rew = np.stack([pool_reward[:, s:s + h] for s in starts], axis=1)  # (N, K, h)

    # ------------------------------------------------------------------
    # Load oracle
    # ------------------------------------------------------------------
    oracle_ckpt = os.path.join(args.run_dir, args.oracle_checkpoint)
    print(f"\nLoading oracle: {oracle_ckpt}")
    oracle = load_model(args.run_dir, oracle_ckpt, device)

    # ------------------------------------------------------------------
    # Step 2-3: score every subsegment, normalize GLOBALLY to [-1, 1]
    # ------------------------------------------------------------------
    print(f"\nScoring {N * K:,} subsegments with oracle rl_sum (h={h}) ...")
    raw_scores = score_all_subsegments(
        seg_obs, seg_act, seg_rew, oracle, args.mcmc_samples, device, args.batch_size,
    )  # (N, K)

    print(f"\nrl_sum: mean={raw_scores.mean():.3f}  std={raw_scores.std():.3f}"
          f"  p1={np.percentile(raw_scores, 1):.3f}  p99={np.percentile(raw_scores, 99):.3f}")

    p1, p99 = np.percentile(raw_scores, [1, 99])
    denom = p99 - p1
    if denom < 1e-8:
        print("  WARNING: oracle scores nearly constant; scalar values will be ~0.")
        f_oracle = np.zeros((N, K), dtype=np.float32)
    else:
        f_oracle = np.clip((raw_scores - p1) / denom * 2.0 - 1.0, -1.0, 1.0).astype(np.float32)

    # ------------------------------------------------------------------
    # Step 4: optional noise (default 0.0 → f == f_oracle exactly)
    # ------------------------------------------------------------------
    if args.noise_std > 0:
        noise = rng.normal(0.0, args.noise_std, size=(N, K)).astype(np.float32)
        f = np.clip(f_oracle + noise, -1.0, 1.0)
    else:
        f = f_oracle
    print(f"  Normalized scalar f (noise_std={args.noise_std}): "
          f"mean={f.mean():.3f}  std={f.std():.3f}")

    # ------------------------------------------------------------------
    # Steps 5-6: comparison windows → lag-filtered candidate pairs → sample
    #            → hard preference.  The candidate structure only depends on
    #            K, W, cmp_stride, min_lag, max_lag — identical for every
    #            trajectory — so this is fully vectorized over N.
    # ------------------------------------------------------------------
    win_offsets = list(range(0, K - W + 1, args.cmp_stride))
    if not win_offsets:
        print(f"ERROR: --window-size {W} exceeds K={K} subsegments (cmp_stride={args.cmp_stride})")
        return

    local_candidates = [(i, j) for i in range(W) for j in range(i + 1, W)
                         if args.min_lag <= (j - i) <= args.max_lag]
    if not local_candidates:
        print(f"ERROR: no candidate pairs satisfy min_lag={args.min_lag} <= j-i <= "
              f"max_lag={args.max_lag} for window_size={W}")
        return
    n_cand   = len(local_candidates)
    n_sample = min(args.max_pairs_per_window, n_cand)

    print(f"\nComparison windows/trajectory : {len(win_offsets)}  (W={W}, cmp_stride={args.cmp_stride})")
    print(f"Eligible candidate pairs/window: {n_cand}  (min_lag={args.min_lag}, max_lag={args.max_lag})")
    print(f"Sampled pairs/window          : {n_sample}  (max_pairs_per_window={args.max_pairs_per_window})")
    print(f"δ = {args.scalar_delta}  (pairs with |f_i-f_j| <= δ are discarded as ties)")

    out_obs, out_act, out_rew, out_adv = [], [], [], []
    out_traj_id, out_start_idx, out_ckpt = [], [], []
    n_ties_total = n_sampled_total = 0
    traj_idx_all = np.arange(N)

    for w_off in win_offsets:
        cand_i = np.array([w_off + i for i, j in local_candidates])   # (n_cand,)
        cand_j = np.array([w_off + j for i, j in local_candidates])

        # Independent per-trajectory sample of n_sample of n_cand candidates,
        # without replacement, via argsort of per-row random keys.
        keys  = rng.random((N, n_cand))
        order = np.argsort(keys, axis=1)[:, :n_sample]      # (N, n_sample)

        sel_i = cand_i[order]        # (N, n_sample)  subsegment index i (into K)
        sel_j = cand_j[order]        # (N, n_sample)  subsegment index j (into K)

        f_i = f[traj_idx_all[:, None], sel_i]   # (N, n_sample)
        f_j = f[traj_idx_all[:, None], sel_j]

        diff     = f_i - f_j
        tie_mask = np.abs(diff) <= args.scalar_delta
        n_ties_total    += int(tie_mask.sum())
        n_sampled_total += diff.size

        pref_is_i = diff > 0
        pref_idx  = np.where(pref_is_i, sel_i, sel_j)      # (N, n_sample)
        nonp_idx  = np.where(pref_is_i, sel_j, sel_i)
        f_pref    = np.where(pref_is_i, f_i, f_j)
        f_nonp    = np.where(pref_is_i, f_j, f_i)

        rows, cols = np.where(~tie_mask)
        if len(rows) == 0:
            continue

        pi = pref_idx[rows, cols]     # subsegment index within K (preferred)
        ni = nonp_idx[rows, cols]     # subsegment index within K (non-preferred)

        out_obs.append(np.stack([seg_obs[rows, pi], seg_obs[rows, ni]], axis=1))
        out_act.append(np.stack([seg_act[rows, pi], seg_act[rows, ni]], axis=1))
        out_rew.append(np.stack([seg_rew[rows, pi], seg_rew[rows, ni]], axis=1))
        out_adv.append(np.stack([f_pref[rows, cols], f_nonp[rows, cols]], axis=1))
        out_traj_id.append(rows.astype(np.int64))
        out_start_idx.append(np.stack([starts[pi], starts[ni]], axis=1).astype(np.int64))
        out_ckpt.append(np.stack([pool_ckpt[rows], pool_ckpt[rows]], axis=1).astype(np.int64))

    M = sum(a.shape[0] for a in out_obs)
    if M == 0:
        print("\nERROR: 0 pairs generated (every candidate was an exact tie).")
        print("  Options: lower --scalar-delta or check the oracle scoring.")
        return

    obs_out   = np.concatenate(out_obs,       axis=0).astype(np.float32)  # (M, 2, h, obs_dim)
    act_out   = np.concatenate(out_act,       axis=0).astype(np.float32)
    rew_out   = np.concatenate(out_rew,       axis=0).astype(np.float32)
    adv_out   = np.concatenate(out_adv,       axis=0).astype(np.float32)  # (M, 2)
    traj_out  = np.concatenate(out_traj_id,   axis=0)                     # (M,)
    start_out = np.concatenate(out_start_idx, axis=0)                     # (M, 2)
    ckpt_out  = np.concatenate(out_ckpt,      axis=0)                     # (M, 2)

    # ------------------------------------------------------------------
    # Optional cap: random subsample (uniform over trajectories/pairs)
    # ------------------------------------------------------------------
    if args.n_pairs is not None and M > args.n_pairs:
        keep      = rng.permutation(M)[:args.n_pairs]
        obs_out   = obs_out[keep]
        act_out   = act_out[keep]
        rew_out   = rew_out[keep]
        adv_out   = adv_out[keep]
        traj_out  = traj_out[keep]
        start_out = start_out[keep]
        ckpt_out  = ckpt_out[keep]
        M = args.n_pairs
        print(f"\nCapped to --n-pairs={args.n_pairs} (random subsample)")

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------
    gaps = adv_out[:, 0] - adv_out[:, 1]  # always > 0
    n_unique_traj = len(np.unique(traj_out))
    pct = [5, 25, 50, 75, 95]

    print(f"\n{'─'*60}")
    print(f"Scalar feedback statistics")
    print(f"{'─'*60}")
    print(f"  Trajectories total     : {N:,}")
    print(f"  Trajectories contributing pairs : {n_unique_traj:,}")
    print(f"  Subsegments/trajectory : K={K}  (h={h}, sub_stride={args.sub_stride})")
    print(f"  Candidates/window      : {n_cand}  →  sampled {n_sample}")
    print(f"  Ties discarded (|Δf|<={args.scalar_delta}): {n_ties_total:,}"
          f"  ({100*n_ties_total/max(1,n_sampled_total):.1f}% of sampled candidates)")
    print(f"  Pairs kept             : {M:,}  (avg {M/max(1,N):.2f} per trajectory)")

    f_vals = np.percentile(f_oracle, pct)
    print(f"\n  Normalized scalar f (all subsegments, before pairing):")
    print(f"    mean={f_oracle.mean():.3f}  std={f_oracle.std():.3f}")
    print(f"    {'  '.join(f'p{p}={v:.3f}' for p, v in zip(pct, f_vals))}")

    gv = np.percentile(gaps, pct)
    print(f"\n  Scalar gap Δf (preferred − non-preferred f):")
    print(f"    mean={gaps.mean():.3f}  std={gaps.std():.3f}  "
          f"min={gaps.min():.3f}  max={gaps.max():.3f}")
    print(f"    {'  '.join(f'p{p}={v:.3f}' for p, v in zip(pct, gv))}")
    print(f"{'─'*60}")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    os.makedirs(out_dir, exist_ok=True)
    save_npz(
        out_path,
        obs=obs_out,
        action=act_out,
        reward=rew_out,
        adv_scores=adv_out,
        checkpoint_step=ckpt_out,
        traj_id=traj_out,
        start_idx=start_out,
    )

    print(f"\nSaved → {out_path}")
    print(f"  obs             : {obs_out.shape}  ([0]=preferred, [1]=non-preferred  h={h})")
    print(f"  action          : {act_out.shape}")
    print(f"  reward          : {rew_out.shape}")
    print(f"  adv_scores      : {adv_out.shape}  (scalar values ∈ [-1, 1])")
    print(f"  checkpoint_step : {ckpt_out.shape}")
    print(f"  traj_id         : {traj_out.shape}  start_idx: {start_out.shape}")
    print(f"\nTrain with:")
    print(f"  dataset: CorrBuffer")
    print(f"  dataset_kwargs:")
    print(f"    path: .../scalar_labels/{env_name}/scalar_labels.npz")
    print(f"  alg: DemoCPL")
    print(f"  alg_kwargs:")
    print(f"    bc_steps: 0  contrastive_bias: 0.5")


if __name__ == "__main__":
    main()
