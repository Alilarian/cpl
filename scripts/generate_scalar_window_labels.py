"""
Generate noiseless window-choice scalar labels for MetaWorld environments.

Human model: the human watches one pool segment and gives a rating in [-1, 1]
every h steps, so a T-step segment gets K non-overlapping ratings. Consecutive
ratings are grouped into windows of W that slide by --window-stride ratings, so
neighbouring windows overlap. Each window is one label: the highest-rated
subsegment is the chosen option, the other W-1 are the alternatives. For every
distinct pool segment, --windows-per-segment of its windows are drawn at random.

Pipeline per segment
--------------------
  1. Subsegment c covers steps [c*h, (c+1)*h). Its score telescopes across the
     grid (no transition is dropped):
         A_c = sum_{t=c*h}^{(c+1)*h-1} r_t + V(s_{(c+1)*h}) - V(s_{c*h})
     This needs s_{K*h}, so K = (T-1) // h  (h=10, T=64: K=6, steps 60-63 unrated).
  2. Normalize GLOBALLY (1st/99th percentile over all subsegments of all
     segments) to the rating f_c in [-1, 1], as in generate_scalar_labels.py.
  3. Windows start every --window-stride ratings: [0,1,2], [1,2,3], ... for W=3.
  4. Draw --windows-per-segment windows uniformly without replacement; the
     label of a window is argmax f over its W subsegments.

No noise anywhere:
  * V(s) = mean_q Q_q(s, tanh(mu(s))) at the oracle's deterministic action,
    so there is no action sampling.
  * Ratings are not perturbed.
  * A window whose top rating is tied (two ratings clipped to the same bound)
    has no unique chosen option and is never drawn. Segments with fewer than
    --windows-per-segment untied windows are dropped.

Exact-duplicate pool rows (the pool samples windows with replacement) are
labelled once.

Output (--output-dir/<env>/scalar_window_labels.npz), one row per label. Rows
are grouped by segment (a segment's windows are adjacent, in time order) and
segments are in random order, so capacity = n * windows_per_segment keeps n
whole segments. subsample_n draws rows independently and splits segments.
    obs             : (M, W, h, obs_dim)  the W subsegments of the window, in time order
    action          : (M, W, h, act_dim)
    reward          : (M, W, h)
    chosen_idx      : (M,)     index of the highest-rated subsegment in the window
    adv_scores      : (M, W)   ratings f in [-1, 1]
    raw_scores      : (M, W)   unnormalized A_c
    traj_id         : (M,)     pool row of the segment
    window_idx      : (M,)     which window of the segment (0 .. n_windows-1)
    start_idx       : (M, W)   subsegment start steps within the segment
    checkpoint_step : (M,)

Same layout as credit labels, so it trains with
    dataset: PMCreditAssignmentBuffer    alg: CreditAssignmentCPL

Usage:
    python scripts/generate_scalar_window_labels.py \\
        --pool-path  datasets/mw_de_labels/mw_drawer-open-v2/pool.npz \\
        --run-dir    datasets/mw_de_labels/mw_drawer-open-v2/oracle \\
        --output-dir datasets/mw_de_labels
"""

import argparse
import os

import numpy as np
import torch

from generate_scalar_labels import load_model, save_npz


def deterministic_values(obs, oracle, device, batch_size):
    """V(s) = mean over the twin critics of Q(s, tanh(mu(s))).  obs: (B, D) -> (B,)"""
    out = np.empty(len(obs), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, len(obs), batch_size):
            o   = torch.from_numpy(obs[s:s + batch_size]).float().to(device)
            enc = oracle.network.encoder(o)
            act = oracle.network.actor(enc).base_dist.loc                 # tanh(mu), no sampling
            out[s:s + batch_size] = oracle.network.critic(enc, act).mean(dim=0).cpu().numpy()
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Generate noiseless window-choice scalar labels for a MetaWorld environment."
    )
    parser.add_argument("--pool-path", type=str, required=True,
                        help="Path to pool.npz from build_trajectory_pool.py")
    parser.add_argument("--run-dir", type=str, required=True,
                        help="Oracle SAC run dir (config.yaml + checkpoint)")
    parser.add_argument("--oracle-checkpoint", type=str, default="best_model.pt",
                        help="Oracle checkpoint filename (default: best_model.pt)")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Writes <output-dir>/<env>/scalar_window_labels.npz")

    parser.add_argument("--rating-period", type=int, default=10,
                        help="Steps per rating h; subsegments do not overlap (default: 10)")
    parser.add_argument("--window-size", type=int, default=3,
                        help="Ratings per window W (default: 3)")
    parser.add_argument("--window-stride", type=int, default=1,
                        help="Ratings between consecutive window starts; < W means "
                             "neighbouring windows overlap (default: 1)")
    parser.add_argument("--windows-per-segment", type=int, default=2,
                        help="Windows drawn at random from each segment (default: 2)")

    parser.add_argument("--batch-size", type=int, default=4096,
                        help="States per oracle forward pass (default: 4096)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for the window draw and segment order (default: 42)")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    env_name = os.path.basename(os.path.dirname(args.pool_path))
    out_dir  = os.path.join(args.output_dir, env_name)
    out_path = os.path.join(out_dir, "scalar_window_labels.npz")
    if os.path.exists(out_path):
        print(f"\nOutput already exists: {out_path}")
        print("Delete it to regenerate.")
        return

    h, W, n_per = args.rating_period, args.window_size, args.windows_per_segment
    rng = np.random.default_rng(args.seed)

    print("=" * 65)
    print(f"Environment          : {env_name}")
    print(f"Pool path            : {args.pool_path}")
    print(f"Oracle               : {os.path.join(args.run_dir, args.oracle_checkpoint)}")
    print(f"rating_period (h)    : {h}")
    print(f"window_size (W)      : {W}")
    print(f"window_stride        : {args.window_stride}")
    print(f"windows_per_segment  : {n_per}")
    print(f"V(s)                 : Q(s, tanh(mu(s))), deterministic")
    print(f"Device               : {device}")
    print("=" * 65)

    # ------------------------------------------------------------------
    # Load pool, drop exact-duplicate segments
    # ------------------------------------------------------------------
    with open(args.pool_path, "rb") as f:
        pool = np.load(f)
        pool_obs    = pool["obs"]             # (N, T, obs_dim)
        pool_action = pool["action"]          # (N, T, act_dim)
        pool_reward = pool["reward"]          # (N, T)
        pool_ckpt   = pool["checkpoint_step"] # (N,)
    N, T, obs_dim = pool_obs.shape

    flat = np.ascontiguousarray(pool_obs.reshape(N, -1))
    keys = flat.view(np.dtype((np.void, flat.dtype.itemsize * flat.shape[1]))).ravel()
    _, first = np.unique(keys, return_index=True)
    rows = np.sort(first)                     # pool row of each distinct segment
    Nd = len(rows)
    print(f"\nPool: N={N:,} rows, {Nd:,} distinct segments  T={T}")

    K = (T - 1) // h
    win_starts = np.arange(0, K - W + 1, args.window_stride)
    n_win = len(win_starts)
    if n_win < n_per:
        print(f"ERROR: {n_win} windows per segment (K={K}, W={W}, stride={args.window_stride}) "
              f"< --windows-per-segment {n_per}")
        return
    members = win_starts[:, None] + np.arange(W)[None]          # (n_win, W) rating indices
    starts  = np.arange(K) * h
    print(f"Ratings/segment K={K} at steps {starts.tolist()} (steps {K * h}-{T - 1} unrated)")
    print(f"Windows/segment {n_win}: {[m.tolist() for m in members]}")

    # ------------------------------------------------------------------
    # Score every subsegment: V only at the K+1 grid boundaries
    # ------------------------------------------------------------------
    oracle = load_model(args.run_dir, os.path.join(args.run_dir, args.oracle_checkpoint), device)
    bnd = np.arange(K + 1) * h                                   # (K+1,)
    V = deterministic_values(pool_obs[rows][:, bnd].reshape(-1, obs_dim), oracle, device,
                             args.batch_size).reshape(Nd, K + 1)
    rew = pool_reward[rows]
    A = np.stack([rew[:, c * h:(c + 1) * h].sum(1) + V[:, c + 1] - V[:, c] for c in range(K)], 1)

    p1, p99 = np.percentile(A, [1, 99])
    f = np.clip((A - p1) / (p99 - p1) * 2.0 - 1.0, -1.0, 1.0).astype(np.float32)
    print(f"\nraw A: mean={A.mean():.3f}  p1={p1:.3f}  p99={p99:.3f}  "
          f"| ratings clipped to ±1: {100 * (np.abs(f) == 1).mean():.1f}%")

    # ------------------------------------------------------------------
    # Draw windows (untied only), build rows grouped by segment
    # ------------------------------------------------------------------
    F = f[:, members]                                            # (Nd, n_win, W)
    tied = (F == F.max(-1, keepdims=True)).sum(-1) > 1           # (Nd, n_win)
    draw_keys = rng.random((Nd, n_win))
    draw_keys[tied] = np.inf
    picked = np.sort(np.argsort(draw_keys, axis=1)[:, :n_per], axis=1)   # (Nd, n_per)
    ok = (~tied).sum(1) >= n_per
    print(f"Windows with a tied top rating: {int(tied.sum()):,} of {tied.size:,}; "
          f"segments dropped (< {n_per} untied windows): {int((~ok).sum()):,}")

    seg = rng.permutation(np.where(ok)[0])                       # segments in random order
    seg_r = np.repeat(seg, n_per)                                # (M,)
    win_r = picked[seg].ravel()                                  # (M,)
    sub   = members[win_r]                                       # (M, W) rating indices
    M = len(seg_r)

    t_idx = starts[sub][..., None] + np.arange(h)                # (M, W, h)
    prow  = rows[seg_r]
    obs_out    = pool_obs[prow[:, None, None], t_idx].astype(np.float32)
    act_out    = pool_action[prow[:, None, None], t_idx].astype(np.float32)
    rew_out    = pool_reward[prow[:, None, None], t_idx].astype(np.float32)
    adv_out    = f[seg_r[:, None], sub]
    raw_out    = A[seg_r[:, None], sub].astype(np.float32)
    chosen_out = adv_out.argmax(1).astype(np.int64)

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------
    srt = np.sort(adv_out, 1)
    gap = srt[:, -1] - srt[:, -2]
    pct = [5, 25, 50, 75, 95]
    print(f"\n{'─' * 60}")
    print(f"  Segments used          : {len(seg):,} of {Nd:,} distinct")
    print(f"  Labels                 : {M:,}  ({n_per} per segment)")
    print(f"  Chosen position        : "
          f"{'  '.join(f'{100 * p:.0f}%' for p in np.bincount(chosen_out, minlength=W) / M)}")
    print(f"  Gap chosen − runner-up : "
          f"{'  '.join(f'p{p}={v:.3f}' for p, v in zip(pct, np.percentile(gap, pct)))}")
    print(f"  Gap < 0.01             : {100 * (gap < 0.01).mean():.1f}% of labels")
    print(f"{'─' * 60}")

    os.makedirs(out_dir, exist_ok=True)
    save_npz(
        out_path,
        obs=obs_out,
        action=act_out,
        reward=rew_out,
        chosen_idx=chosen_out,
        adv_scores=adv_out,
        raw_scores=raw_out,
        traj_id=prow.astype(np.int64),
        window_idx=win_r.astype(np.int64),
        start_idx=starts[sub].astype(np.int64),
        checkpoint_step=pool_ckpt[prow].astype(np.int64),
    )
    print(f"\nSaved → {out_path}")
    print(f"  obs        : {obs_out.shape}  (W={W} subsegments of h={h} steps)")
    print(f"  chosen_idx : {chosen_out.shape}")
    print(f"\nTrain with:")
    print(f"  dataset: PMCreditAssignmentBuffer")
    print(f"  dataset_kwargs:")
    print(f"    path: {out_path}")
    print(f"    capacity: <budget>   # prefix; a multiple of {n_per} keeps whole segments")
    print(f"  alg: CreditAssignmentCPL")


if __name__ == "__main__":
    main()
