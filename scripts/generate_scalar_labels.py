"""
Generate scalar feedback labels for MetaWorld environments.

Each pool segment is scored with oracle rl_sum (Σ r_t + V(s_T) - V(s_0))
over the first --segment-len steps, normalized to [-1, 1], then Gaussian
noise is added to simulate a human evaluator.  A sliding window of W
adjacent pool segments defines one evaluation batch; within each window
every pair (A, B) with |f_A - f_B| > --scalar-delta is extracted.

Output format is K=2 compatible with CorrBuffer (preferred at index 0,
label always 0), matching the layout of pref_labels.npz so that DemoCPL
with contrastive_bias=0.5 and bc_steps=0 can train directly.

Output (--output-dir/<env>/scalar_labels.npz):
    obs             : (M, 2, h, obs_dim)   [0]=preferred, [1]=non-preferred
    action          : (M, 2, h, act_dim)
    reward          : (M, 2, h)
    adv_scores      : (M, 2)               [f_preferred, f_non-preferred] (noisy scalar)
    checkpoint_step : (M, 2)

Usage:
    python scripts/generate_scalar_labels.py \\
        --pool-path  /scratch/.../demo_pool/mw_drawer-open-v2/pool.npz \\
        --run-dir    /scratch/.../oracle_sac_all/mw_drawer-open-v2 \\
        --segment-len 20 \\
        --window-size 10 \\
        --noise-std 0.1 \\
        --n-pairs 300000 \\
        --output-dir /scratch/.../scalar_labels

Parameters:
    --segment-len h  : steps per scalar segment scored by oracle (default: 20)
    --window-size W  : sliding window width over pool indices (default: 10)
    --noise-std      : Gaussian noise std added to normalized oracle score (default: 0.1)
    --scalar-delta   : indifference threshold δ; skip pairs with |f_A-f_B| < δ (default: 0.0)
    --n-pairs        : cap on output pairs; extraction stops early when reached (default: no cap)
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
    parser.add_argument("--segment-len", type=int, default=20,
                        help="Steps h used for oracle scoring (first h steps of each segment, default: 20)")
    parser.add_argument("--window-size", type=int, default=10,
                        help="Sliding window width W over pool indices (default: 10)")
    parser.add_argument("--noise-std", type=float, default=0.1,
                        help="Gaussian noise std added to oracle scalar value (default: 0.1)")
    parser.add_argument("--scalar-delta", type=float, default=0.0,
                        help="Indifference threshold δ; skip pairs |f_A-f_B| < δ (default: 0.0)")
    parser.add_argument("--n-pairs", type=int, default=None,
                        help="Cap on output pairs (default: no cap)")
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
    print(f"Environment      : {env_name}")
    print(f"Pool path        : {args.pool_path}")
    print(f"Oracle dir       : {args.run_dir}")
    print(f"Segment len h    : {args.segment_len}")
    print(f"Window size W    : {args.window_size}")
    print(f"Noise std        : {args.noise_std}")
    print(f"Scalar delta δ   : {args.scalar_delta}")
    print(f"N-pairs cap      : {args.n_pairs if args.n_pairs else 'no cap'}")
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
    h = args.segment_len
    W = args.window_size
    print(f"  N={N:,} segments  T={T}  obs_dim={obs_dim}  act_dim={act_dim}")

    if h > T:
        print(f"ERROR: --segment-len {h} > pool T={T}")
        return
    if W > N:
        print(f"ERROR: --window-size {W} > pool N={N}")
        return

    # ------------------------------------------------------------------
    # Load oracle
    # ------------------------------------------------------------------
    oracle_ckpt = os.path.join(args.run_dir, args.oracle_checkpoint)
    print(f"\nLoading oracle: {oracle_ckpt}")
    oracle = load_model(args.run_dir, oracle_ckpt, device)

    # ------------------------------------------------------------------
    # Score first h steps of each pool segment with oracle rl_sum
    # ------------------------------------------------------------------
    seg_obs = pool_obs[:, :h]     # (N, h, obs_dim)
    seg_act = pool_action[:, :h]  # (N, h, act_dim)
    seg_rew = pool_reward[:, :h]  # (N, h)

    print(f"\nScoring {N:,} segments with oracle rl_sum (h={h}) ...")
    rl_sum_all = np.empty(N, dtype=np.float32)
    n_batches = (N + args.batch_size - 1) // args.batch_size

    for b in range(n_batches):
        s = b * args.batch_size
        e = min(s + args.batch_size, N)
        rl_sum_all[s:e] = score_pool_batch(
            seg_obs[s:e], seg_act[s:e], seg_rew[s:e],
            oracle, args.mcmc_samples, device,
        )
        if (b + 1) % 50 == 0 or b == n_batches - 1:
            print(f"  [{b+1:>4}/{n_batches}]  scored {e:>7,}/{N:,}")

    print(f"\nrl_sum: mean={rl_sum_all.mean():.3f}  std={rl_sum_all.std():.3f}"
          f"  p1={np.percentile(rl_sum_all, 1):.3f}  p99={np.percentile(rl_sum_all, 99):.3f}")

    # ------------------------------------------------------------------
    # Normalize to [-1, 1] via 1st–99th percentile
    # ------------------------------------------------------------------
    p1, p99 = np.percentile(rl_sum_all, [1, 99])
    denom = p99 - p1
    if denom < 1e-8:
        print("  WARNING: oracle scores nearly constant; scalar values will be ~0.")
        scalar_oracle = np.zeros(N, dtype=np.float32)
    else:
        scalar_oracle = np.clip(
            (rl_sum_all - p1) / denom * 2.0 - 1.0, -1.0, 1.0
        ).astype(np.float32)

    rng = np.random.default_rng(args.seed)
    noise = rng.normal(0.0, args.noise_std, size=N).astype(np.float32)
    scalar_noisy = np.clip(scalar_oracle + noise, -1.0, 1.0)
    print(f"  Noisy scalar: mean={scalar_noisy.mean():.3f}  std={scalar_noisy.std():.3f}")

    # ------------------------------------------------------------------
    # Sliding window pair extraction
    # ------------------------------------------------------------------
    n_windows = N - W + 1
    print(f"\nExtracting pairs  (W={W}  δ={args.scalar_delta}  n_windows={n_windows:,}) ...")
    print(f"  Up to {n_windows * W * (W-1) // 2:,} candidate pairs")

    out_obs  = []
    out_act  = []
    out_rew  = []
    out_adv  = []
    out_ckpt = []
    n_candidates = n_skipped = 0

    for i in range(n_windows):
        win_f = scalar_noisy[i: i + W]

        for a in range(W):
            for b_idx in range(a + 1, W):
                fa, fb = float(win_f[a]), float(win_f[b_idx])
                n_candidates += 1
                diff = fa - fb
                if abs(diff) < args.scalar_delta:
                    n_skipped += 1
                    continue

                if diff > 0:
                    pi, ni = i + a, i + b_idx
                    f_pref, f_npref = fa, fb
                else:
                    pi, ni = i + b_idx, i + a
                    f_pref, f_npref = fb, fa

                out_obs.append(np.stack([seg_obs[pi], seg_obs[ni]], axis=0))   # (2, h, obs_dim)
                out_act.append(np.stack([seg_act[pi], seg_act[ni]], axis=0))
                out_rew.append(np.stack([seg_rew[pi], seg_rew[ni]], axis=0))
                out_adv.append([f_pref, f_npref])
                out_ckpt.append([int(pool_ckpt[pi]), int(pool_ckpt[ni])])

        if args.n_pairs is not None and len(out_obs) >= args.n_pairs:
            print(f"  Reached n_pairs={args.n_pairs}, stopping early.")
            break

        if (i + 1) % 10000 == 0 or i == n_windows - 1:
            print(f"  [{i+1:>7}/{n_windows:>7}]  pairs={len(out_obs):>9,}")

    if args.n_pairs is not None:
        out_obs  = out_obs[:args.n_pairs]
        out_act  = out_act[:args.n_pairs]
        out_rew  = out_rew[:args.n_pairs]
        out_adv  = out_adv[:args.n_pairs]
        out_ckpt = out_ckpt[:args.n_pairs]

    M = len(out_obs)
    if M == 0:
        print("\nERROR: 0 pairs generated. Lower --scalar-delta or increase --window-size.")
        return

    obs_out  = np.stack(out_obs,  axis=0).astype(np.float32)  # (M, 2, h, obs_dim)
    act_out  = np.stack(out_act,  axis=0).astype(np.float32)
    rew_out  = np.stack(out_rew,  axis=0).astype(np.float32)
    adv_out  = np.array(out_adv,  dtype=np.float32)            # (M, 2)
    ckpt_out = np.array(out_ckpt, dtype=np.int64)              # (M, 2)

    gaps = adv_out[:, 0] - adv_out[:, 1]  # always > 0
    pct  = [5, 25, 50, 75, 95]

    print(f"\n{'─'*60}")
    print(f"Scalar feedback statistics")
    print(f"{'─'*60}")
    print(f"  Pool segments     : {N:,}  (h={h} steps each)")
    print(f"  Windows processed : {min(n_windows, i+1):,} / {n_windows:,}  (W={W})")
    print(f"  Candidates        : {n_candidates:,}")
    print(f"  Skipped (|Δf|<δ) : {n_skipped:,}  ({100*n_skipped/max(1,n_candidates):.1f}%)")
    print(f"  Pairs kept        : {M:,}")

    gv = np.percentile(gaps, pct)
    print(f"\n  Scalar gap Δf (preferred − non-preferred):")
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
    )

    print(f"\nSaved → {out_path}")
    print(f"  obs             : {obs_out.shape}  ([0]=preferred, [1]=non-preferred  h={h})")
    print(f"  action          : {act_out.shape}")
    print(f"  reward          : {rew_out.shape}")
    print(f"  adv_scores      : {adv_out.shape}  (noisy scalar values ∈ [-1, 1])")
    print(f"  checkpoint_step : {ckpt_out.shape}")
    print(f"\nTrain with:")
    print(f"  dataset: CorrBuffer")
    print(f"  dataset_kwargs:")
    print(f"    path: .../scalar_labels/{env_name}/scalar_labels.npz")
    print(f"  alg: DemoCPL")
    print(f"  alg_kwargs:")
    print(f"    bc_steps: 0  contrastive_bias: 0.5")


if __name__ == "__main__":
    main()
