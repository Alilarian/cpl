"""
Simulate all human feedback types for the PointMass navigation environment.

Uses a precomputed VI advantage table (advantage_vi.npz) instead of a live
SAC critic, so no GPU or MuJoCo is needed.

Feedback types  (--type)
------------------------
  pref              Pairwise preference — score pool pairs with rl_sum, keep better
  corr              Corrective         — pool segment vs expert rollout from same s_0
  demo              Demonstrative      — K-way counterfactuals sorted by rl_sum
  scalar            Scalar feedback    — each trajectory is split into K temporal
                                         subsegments (length h, stride sub_stride),
                                         each scored with oracle rl_sum, normalized
                                         globally to [-1,1]; within a comparison
                                         window of W subsegments, candidate pairs
                                         with min_lag <= |i-j| <= max_lag are formed
                                         and a random subset is converted into hard
                                         preferences. Never compares across trajectories.
  credit_assignment Credit assignment  — oracle selects best length-k subsegment
                                         from each reference trajectory (T=64);
                                         multi-way cross-entropy loss over C candidates

Scoring metric (same as MetaWorld generate_*_labels.py)
--------------------------------------------------------
  rl_sum(σ) = Σ_t r_t  +  V*(s_T) − V*(s_0)

  V*(s) is obtained by bilinear interpolation in the VI value table.

Output schemas (identical to MetaWorld equivalents)
----------------------------------------------------
  pref / corr :  obs (N,2,T,2)  action (N,2,T,2)  reward (N,2,T)
                 adv_scores (N,2)  [gap (N,) for pref]
  demo        :  obs (N,K,T,2)  action (N,K,T,2)  reward (N,K,T)
                 adv_scores (N,K)
  scalar      :  obs (M,2,segment_len,2)  action (M,2,segment_len,2)
                 reward (M,2,segment_len)  adv_scores (M,2)
                 checkpoint_step (M,2)  traj_id (M,)  start_idx (M,2)
                 Same core layout as pref — PMFeedbackBuffer loads it directly.
                 traj_id/start_idx are provenance-only (verify no cross-traj pairs).
  credit_assn :  obs (N,C,k,2)  action (N,C,k,2)  adv_scores (N,C)
                 chosen_idx (N,)  checkpoint_step (N,)
                 C = T-k+1 candidate windows per trajectory.
                 PMCreditAssignmentBuffer loads it directly.

Usage
-----
# Pairwise preference
python scripts/generate_pm_feedback.py \\
    --type pref \\
    --pool datasets/pm/pool.npz \\
    --advantage-npz runs/pm_sac_oracle/advantage_vi_N100.npz \\
    --out datasets/pm/pref_labels.npz

# Corrective
python scripts/generate_pm_feedback.py \\
    --type corr \\
    --pool datasets/pm/pool.npz \\
    --advantage-npz runs/pm_sac_oracle/advantage_vi_N100.npz \\
    --out datasets/pm/corr_labels.npz

# Demonstrative (K=5)
python scripts/generate_pm_feedback.py \\
    --type demo \\
    --pool datasets/pm/pool.npz \\
    --advantage-npz runs/pm_sac_oracle/advantage_vi_N100.npz \\
    --out datasets/pm/demo_labels.npz \\
    --n-counterfactuals 4

# Scalar feedback
python scripts/generate_pm_feedback.py \\
    --type scalar \\
    --pool datasets/pm/pool.npz \\
    --advantage-npz runs/pm_sac_oracle/advantage_vi_N100.npz \\
    --out datasets/pm/scalar_labels.npz \\
    --segment-len 16 --sub-stride 12 --window-size 5 --cmp-stride 5 \\
    --min-lag 2 --max-lag 4 --max-pairs-per-window 4 \\
    --noise-std 0.0 --scalar-delta 0.0

# Credit assignment
python scripts/generate_pm_feedback.py \\
    --type credit_assignment \\
    --pool datasets/pm/pool.npz \\
    --advantage-npz runs/pm_sac_oracle/advantage_vi_N100.npz \\
    --out datasets/pm/credit_assignment_labels.npz \\
    --subsegment-len 12
"""

import argparse
import importlib.util
import io
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_module(dotted_name, rel_path):
    path = os.path.join(_ROOT, rel_path)
    spec = importlib.util.spec_from_file_location(dotted_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[dotted_name] = mod
    spec.loader.exec_module(mod)
    return mod


_pm  = _load_module("research.envs.pointmass",     "research/envs/pointmass.py")
_avi = _load_module("research.utils.advantage_vi", "research/utils/advantage_vi.py")

PointMassGymEnv = _pm.PointMassGymEnv
AdvantageVI     = _avi.AdvantageVI


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def save_npz(path, **arrays):
    """Atomic compressed save: write to .tmp then rename."""
    tmp = path + ".tmp"
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with io.BytesIO() as buf:
        np.savez_compressed(buf, **arrays)
        buf.seek(0)
        with open(tmp, "wb") as f:
            f.write(buf.read())
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# VI-based scoring primitives
# ---------------------------------------------------------------------------

def vi_values(avi: AdvantageVI, obs: np.ndarray) -> np.ndarray:
    """
    Bilinear interpolation of V*(s) for an arbitrary-shape obs array.

    obs : (*batch, 2)  — positions in [0,1]²
    returns: (*batch,) float64
    """
    orig_shape = obs.shape[:-1]
    flat = obs.reshape(-1, 2).astype(np.float64)
    cell = float(avi.xs[1] - avi.xs[0])

    fx = flat[:, 0] / cell - 0.5
    fy = flat[:, 1] / cell - 0.5
    i0 = np.clip(np.floor(fx).astype(int), 0, avi.N - 2)
    j0 = np.clip(np.floor(fy).astype(int), 0, avi.N - 2)
    wx = np.clip(fx - i0, 0.0, 1.0)
    wy = np.clip(fy - j0, 0.0, 1.0)
    i1, j1 = i0 + 1, j0 + 1

    v = (avi.V[i0, j0] * (1 - wx) * (1 - wy)
         + avi.V[i1, j0] * wx       * (1 - wy)
         + avi.V[i0, j1] * (1 - wx) * wy
         + avi.V[i1, j1] * wx       * wy)

    return v.reshape(orig_shape)


def score_rl_sum(avi: AdvantageVI, obs: np.ndarray, reward: np.ndarray) -> np.ndarray:
    """
    rl_sum(σ) = Σ_t r_t  +  V*(s_T) − V*(s_0)

    obs    : (N, T, 2)
    reward : (N, T)
    returns: (N,) float64
    """
    V = vi_values(avi, obs)                     # (N, T)
    return reward[:, :-1].sum(axis=1) + V[:, -1] - V[:, 0]


# ---------------------------------------------------------------------------
# Expert rollout
# ---------------------------------------------------------------------------

def rollout_from_state(env: PointMassGymEnv, policy_fn, s0: np.ndarray,
                       T: int) -> tuple:
    """
    Restore env to s0, roll out policy_fn for T steps.

    If the episode ends early (goal or trap reached), the remaining steps are
    padded by repeating the final obs/action and zeroing reward — consistent
    with the halt-prefix construction in e-stop labeling.

    Returns (obs, action, reward)  each shape (T, dim).
    """
    env.set_state(s0)
    obs = env._pos.astype(np.float32).copy()

    ep_obs, ep_act, ep_rew = [], [], []
    done = False
    for _ in range(T):
        if done:
            # Pad: repeat last obs/action, zero reward
            ep_obs.append(ep_obs[-1].copy())
            ep_act.append(ep_act[-1].copy())
            ep_rew.append(0.0)
        else:
            action = np.array(policy_fn(obs), dtype=np.float32)
            obs, reward, done, _ = env.step(action)
            ep_obs.append(obs.copy())
            ep_act.append(action.copy())
            ep_rew.append(float(reward))

    return (np.array(ep_obs,  dtype=np.float32),
            np.array(ep_act,  dtype=np.float32),
            np.array(ep_rew,  dtype=np.float32))


# ---------------------------------------------------------------------------
# Feedback type: pairwise preference
# ---------------------------------------------------------------------------

def sample_and_filter_pairs(scores: np.ndarray, N: int, n_candidates: int,
                            min_adv_gap: float, rng: np.random.Generator):
    """
    Sample n_candidates distinct-index (a, b) pairs with replacement and keep
    only those whose |score gap| >= min_adv_gap.

    Shared by generate_pref (scripts/generate_pm_feedback.py) and the webapp
    trial-bank builder (webapp/backend/trial_bank.py) so both draw candidate
    pairs the same way.

    Returns
    -------
    idx_a, idx_b : kept candidate indices (arrays, same length)
    raw_gap      : scores[idx_a] - scores[idx_b] for kept pairs (signed)
    """
    idx_a = rng.integers(0, N, size=n_candidates)
    idx_b = rng.integers(0, N, size=n_candidates)
    same  = idx_a == idx_b
    while same.any():
        idx_b[same] = rng.integers(0, N, size=int(same.sum()))
        same = idx_a == idx_b

    raw_gap = scores[idx_a] - scores[idx_b]                         # signed (N_cands,)
    keep    = np.abs(raw_gap) >= min_adv_gap
    return idx_a[keep], idx_b[keep], raw_gap[keep]


def generate_pref(pool, avi, args, rng):
    """
    Score all N pool segments, sample a fixed candidate pool of pairs, filter
    by |gap|, sort by gap descending, take top n_pairs.

    Subset guarantee
    ----------------
    When --n-candidates is set to a fixed value (independent of --n-pairs),
    every budget level uses the same rng calls → same candidate pairs →
    same filter mask → same sorted order.  Budget B1 < B2 then gets
    sorted_pairs[:B1] which is a literal prefix of sorted_pairs[:B2].
    """
    obs    = pool["obs"]              # (N, T, 2)
    action = pool["action"]           # (N, T, 2)
    reward = pool["reward"]           # (N, T)
    ckpt   = pool["checkpoint_step"]  # (N,)
    N      = obs.shape[0]

    # ── Step 1: score all segments ──────────────────────────────────────
    print(f"\nScoring {N} pool segments …")
    scores = score_rl_sum(avi, obs, reward)                          # (N,)
    print(f"  rl_sum: mean={scores.mean():.3f}  std={scores.std():.3f}"
          f"  min={scores.min():.3f}  max={scores.max():.3f}")

    # ── Step 2: sample candidate pairs (fixed pool → subset guarantee) ──
    if args.n_candidates is not None:
        n_cands = args.n_candidates
    else:
        n_cands = int(args.n_pairs * args.oversampling_factor) if args.n_pairs else N * 3

    print(f"\nSampling {n_cands:,} candidate pairs (N={N}, with replacement) …")

    # ── Step 3: filter by |gap| ─────────────────────────────────────────
    idx_a, idx_b, raw_gap = sample_and_filter_pairs(
        scores, N, n_cands, args.min_adv_gap, rng)
    n_kept = len(idx_a)

    print(f"  After gap filter (|gap| ≥ {args.min_adv_gap}): "
          f"{n_kept:,} / {n_cands:,}  ({100 * n_kept / n_cands:.1f}%)")

    if n_kept == 0:
        print("  ERROR: 0 pairs kept. Lower --min-adv-gap or raise --n-candidates.")
        return

    if args.n_pairs is not None and n_kept < args.n_pairs:
        print(f"  WARNING: only {n_kept:,} pairs survive the filter "
              f"(target {args.n_pairs:,}). Raise --n-candidates or lower --min-adv-gap.")

    # ── Step 4: order filtered pairs ────────────────────────────────────
    if args.sort_by_gap:
        # Sort by |gap| descending: top-n = most extreme comparisons
        order = np.argsort(-np.abs(raw_gap))
        print(f"  Ordering: sorted by gap descending (most informative first)")
    else:
        # Random shuffle: nested subsets are representative, not extreme
        order = rng.permutation(n_kept)
        print(f"  Ordering: random shuffle (nested representative subsets)")
    idx_a   = idx_a[order]
    idx_b   = idx_b[order]
    raw_gap = raw_gap[order]

    # ── Step 5: take top n_pairs (prefix of sorted/shuffled list) ────────
    if args.n_pairs is not None:
        idx_a   = idx_a[:args.n_pairs]
        idx_b   = idx_b[:args.n_pairs]
        raw_gap = raw_gap[:args.n_pairs]

    n_out = len(idx_a)

    # ── Step 6: build arrays — index 0 = better (higher rl_sum) ─────────
    a_is_better = raw_gap >= 0
    better_idx  = np.where(a_is_better, idx_a, idx_b)
    worse_idx   = np.where(a_is_better, idx_b, idx_a)

    obs_out  = np.stack([obs[better_idx],    obs[worse_idx]],    axis=1).astype(np.float32)
    act_out  = np.stack([action[better_idx], action[worse_idx]], axis=1).astype(np.float32)
    rew_out  = np.stack([reward[better_idx], reward[worse_idx]], axis=1).astype(np.float32)
    adv_out  = np.stack([scores[better_idx], scores[worse_idx]], axis=1).astype(np.float32)
    gap_out  = np.abs(raw_gap).astype(np.float32)
    ckpt_out = np.stack([ckpt[better_idx],   ckpt[worse_idx]],   axis=1).astype(np.int64)

    pct = [5, 25, 50, 75, 95]
    gv  = np.percentile(gap_out, pct)
    print(f"\n{'─'*60}")
    print(f"Pref statistics")
    print(f"{'─'*60}")
    print(f"  Pool segments    : {N}")
    print(f"  Candidates       : {n_cands:,}")
    print(f"  After gap filter : {n_kept:,}")
    print(f"  Output pairs     : {n_out:,}")
    print(f"\n  |gap| (better − worse rl_sum):")
    print(f"    mean={gap_out.mean():.3f}  std={gap_out.std():.3f}  "
          f"min={gap_out.min():.3f}  max={gap_out.max():.3f}")
    print(f"    {'  '.join(f'p{p}={v:.3f}' for p, v in zip(pct, gv))}")
    print(f"{'─'*60}")

    save_npz(args.out,
             obs=obs_out, action=act_out, reward=rew_out,
             adv_scores=adv_out, gap=gap_out, checkpoint_step=ckpt_out,
             n_choice_structures=np.int64(n_out))

    print(f"\nSaved → {args.out}")
    print(f"  obs              : {obs_out.shape}  ([0]=better, [1]=worse)")
    print(f"  gap              : mean={gap_out.mean():.3f}  median={np.median(gap_out):.3f}")
    print(f"  n_choice_structures: {n_out}")
    print(f"\nTrain with:  dataset: PrefBuffer")

# ---------------------------------------------------------------------------
# Feedback type: corrective
# ---------------------------------------------------------------------------

def generate_corr(pool, avi, env, args, rng):
    """
    For each pool segment: roll out the expert from s_0, score both with
    rl_sum, keep pair if |gap| >= min_adv_gap (index 0 = better trajectory).
    """
    obs    = pool["obs"]
    action = pool["action"]
    reward = pool["reward"]
    state  = pool["state"]
    ckpt   = pool["checkpoint_step"]
    N, T   = obs.shape[:2]

    expert = env.make_expert_policy()

    print(f"\nGenerating corrective pairs for {N} segments …")

    out_obs, out_act, out_rew, out_adv, out_impr, out_ckpt = [], [], [], [], [], []
    n_kept = n_dropped = 0

    for i in range(N):
        s0 = state[i, 0]                                # [x, y, step] at t=0
        ex_obs, ex_act, ex_rew = rollout_from_state(env, expert, s0, T)

        # Score expert and original together: (2, T, 2) and (2, T)
        both_obs = np.stack([ex_obs, obs[i]],       axis=0)   # (2, T, 2)
        both_rew = np.stack([ex_rew, reward[i]],    axis=0)   # (2, T)
        scores   = score_rl_sum(avi, both_obs, both_rew)      # (2,)

        improvement = abs(float(scores[0] - scores[1]))
        if improvement < args.min_adv_gap:
            n_dropped += 1
            continue

        # Index 0 = better (higher rl_sum)
        if scores[0] >= scores[1]:
            pref_obs, pref_act, pref_rew = ex_obs, ex_act, ex_rew
            nonpref_obs, nonpref_act, nonpref_rew = obs[i], action[i], reward[i]
        else:
            pref_obs, pref_act, pref_rew = obs[i], action[i], reward[i]
            nonpref_obs, nonpref_act, nonpref_rew = ex_obs, ex_act, ex_rew
            scores = scores[[1, 0]]

        out_obs.append(np.stack([pref_obs,  nonpref_obs],  axis=0))
        out_act.append(np.stack([pref_act,  nonpref_act],  axis=0))
        out_rew.append(np.stack([pref_rew,  nonpref_rew],  axis=0))
        out_adv.append([float(scores[0]), float(scores[1])])
        out_impr.append(improvement)
        out_ckpt.append(int(ckpt[i]))
        n_kept += 1

        if (i + 1) % 500 == 0 or i == N - 1:
            print(f"  [{i+1:>5}/{N}]  kept={n_kept}  dropped={n_dropped}")

        if args.n_pairs is not None and n_kept >= args.n_pairs:
            print(f"  Reached n_pairs={args.n_pairs}, stopping early.")
            break

    print(f"  Pairs kept={n_kept}  dropped={n_dropped}")
    if n_kept == 0:
        print("  ERROR: 0 pairs kept. Lower --min-adv-gap.")
        return

    obs_out  = np.stack(out_obs,  axis=0).astype(np.float32)
    act_out  = np.stack(out_act,  axis=0).astype(np.float32)
    rew_out  = np.stack(out_rew,  axis=0).astype(np.float32)
    adv_out  = np.array(out_adv,  dtype=np.float32)
    impr_out = np.array(out_impr, dtype=np.float32)
    ckpt_out = np.array(out_ckpt, dtype=np.int64)

    # Sort by improvement magnitude (most informative pairs first)
    order = np.argsort(-impr_out)
    save_npz(args.out,
             obs=obs_out[order], action=act_out[order], reward=rew_out[order],
             adv_scores=adv_out[order], improvement=impr_out[order],
             checkpoint_step=ckpt_out[order],
             n_choice_structures=np.int64(obs_out.shape[0]))

    print(f"\nSaved → {args.out}")
    print(f"  obs              : {obs_out.shape}  ([0]=better, [1]=worse)")
    print(f"  improvement      : mean={impr_out.mean():.3f}  median={np.median(impr_out):.3f}")
    print(f"  n_choice_structures: {obs_out.shape[0]}")
    print(f"\nTrain with:  dataset: CorrBuffer")


# ---------------------------------------------------------------------------
# Feedback type: demonstrative (K-way)
# ---------------------------------------------------------------------------

def generate_demo(pool, avi, env, args, rng):
    """
    Build K-way choice sets:
      index 0  : expert rollout from s_0              (best)
      index 1..(K-2) : noisy expert at varying eps   (counterfactuals)
      index K-1: original pool segment               (usually worst)

    K = 1 (expert) + n_counterfactuals + 1 (original).
    All K candidates scored by rl_sum, then sorted descending so index 0
    is always the best.
    """
    obs    = pool["obs"]
    action = pool["action"]
    reward = pool["reward"]
    state  = pool["state"]
    ckpt   = pool["checkpoint_step"]
    N, T   = obs.shape[:2]

    K = args.n_counterfactuals + 2         # expert + counterfactuals + original
    # Noise levels spread across (0, 1) for counterfactuals
    eps_values = np.linspace(0.1, 0.9, args.n_counterfactuals)

    expert = env.make_expert_policy()
    noisy_policies = [env.make_noisy_expert_policy(eps=float(e)) for e in eps_values]

    print(f"\nGenerating demo pairs  K={K}  (1 expert + {args.n_counterfactuals} "
          f"noisy + 1 original)")
    print(f"  Noise levels: {eps_values.tolist()}")

    out_obs, out_act, out_rew, out_adv, out_ckpt = [], [], [], [], []
    n_kept = n_dropped = 0

    for i in range(N):
        s0 = state[i, 0]

        # Expert rollout
        ex_obs, ex_act, ex_rew = rollout_from_state(env, expert, s0, T)

        # Noisy-expert counterfactuals
        cf_obs_list, cf_act_list, cf_rew_list = [ex_obs], [ex_act], [ex_rew]
        for pol in noisy_policies:
            c_obs, c_act, c_rew = rollout_from_state(env, pol, s0, T)
            cf_obs_list.append(c_obs)
            cf_act_list.append(c_act)
            cf_rew_list.append(c_rew)

        # Original pool segment
        cf_obs_list.append(obs[i])
        cf_act_list.append(action[i])
        cf_rew_list.append(reward[i])

        # Stack K candidates: (K, T, 2) and (K, T)
        cand_obs = np.stack(cf_obs_list, axis=0)
        cand_rew = np.stack(cf_rew_list, axis=0)
        scores   = score_rl_sum(avi, cand_obs, cand_rew)    # (K,)

        gap = float(scores.max() - scores.min())
        if gap < args.min_adv_gap:
            n_dropped += 1
            continue

        # Sort descending by rl_sum (index 0 = best)
        order = np.argsort(-scores)
        cand_act = np.stack(cf_act_list, axis=0)

        out_obs.append(cand_obs[order])
        out_act.append(cand_act[order])
        out_rew.append(np.stack(cf_rew_list, axis=0)[order])
        out_adv.append(scores[order])
        out_ckpt.append(int(ckpt[i]))
        n_kept += 1

        if (i + 1) % 500 == 0 or i == N - 1:
            print(f"  [{i+1:>5}/{N}]  kept={n_kept}  dropped={n_dropped}")

        if args.n_pairs is not None and n_kept >= args.n_pairs:
            print(f"  Reached n_pairs={args.n_pairs}, stopping early.")
            break

    print(f"  Pairs kept={n_kept}  dropped={n_dropped}")
    if n_kept == 0:
        print("  ERROR: 0 pairs kept. Lower --min-adv-gap.")
        return

    obs_out  = np.stack(out_obs, axis=0).astype(np.float32)
    act_out  = np.stack(out_act, axis=0).astype(np.float32)
    rew_out  = np.stack(out_rew, axis=0).astype(np.float32)
    adv_out  = np.stack(out_adv, axis=0).astype(np.float32)
    ckpt_out = np.array(out_ckpt, dtype=np.int64)

    save_npz(args.out,
             obs=obs_out, action=act_out, reward=rew_out,
             adv_scores=adv_out, checkpoint_step=ckpt_out,
             n_choice_structures=np.int64(obs_out.shape[0]))

    print(f"\nSaved → {args.out}")
    print(f"  obs              : {obs_out.shape}  ([0]=best, [-1]=worst)")
    print(f"  adv_scores       : {adv_out.shape}")
    print(f"  n_choice_structures: {obs_out.shape[0]}")
    print(f"\nTrain with:  dataset: DemoBuffer   (K={K})")


# ---------------------------------------------------------------------------
# Feedback type: scalar (local pairwise preferences from scalar signals)
# ---------------------------------------------------------------------------

def generate_scalar(pool, avi, args, rng):
    """
    Per-trajectory temporal-subsegment scalar feedback → local hard preferences.

    Every pool trajectory is processed independently — a preference NEVER
    compares subsegments from two different trajectories (traj(A) == traj(B)
    always).  Pipeline per trajectory:

      1. Split the T-step trajectory into K overlapping temporal subsegments
         of length h, starting at every multiple of sub_stride
         (starts = arange(0, T-h+1, sub_stride)).
      2. Score every subsegment with oracle rl_sum.
      3. Normalize ALL subsegment scores GLOBALLY (1st/99th percentile over
         the whole dataset, not per trajectory) to f ∈ [-1, 1].
      4. Optionally add Gaussian noise (noise_std; default 0 → exact oracle).
      5. Slide a comparison window of W consecutive subsegments (stride
         cmp_stride) over the K subsegments. Within a window, candidate pairs
         (i, j) are kept only if min_lag <= j-i <= max_lag (drops neighboring/
         too-distant subsegments); at most max_pairs_per_window are sampled.
      6. Each sampled pair becomes a hard preference: f_i vs f_j, discarding
         only exact ties (|f_i - f_j| <= scalar_delta).

    Output schema matches pref labels so PMFeedbackBuffer loads it directly.
    adv_scores stores [f_preferred, f_non_preferred].  traj_id/start_idx are
    stored for provenance so cross-trajectory bugs are easy to catch.
    """
    obs    = pool["obs"]              # (N, T, obs_dim)
    action = pool["action"]           # (N, T, act_dim)
    reward = pool["reward"]           # (N, T)
    ckpt   = pool["checkpoint_step"]  # (N,)
    N, T   = obs.shape[:2]

    h            = args.segment_len            # subsegment length
    sub_stride   = args.sub_stride
    W            = args.window_size            # subsegments per comparison window
    cmp_stride   = args.cmp_stride
    min_lag      = args.min_lag
    max_lag      = args.max_lag
    max_per_win  = args.max_pairs_per_window
    noise_std    = args.noise_std
    delta        = args.scalar_delta

    if h > T:
        raise ValueError(f"--segment-len {h} exceeds pool trajectory length T={T}")

    # ------------------------------------------------------------------
    # Step 1: temporal subsegments (same start offsets for every trajectory)
    # ------------------------------------------------------------------
    starts = np.arange(0, T - h + 1, sub_stride)
    K = len(starts)
    if W > K:
        raise ValueError(f"--window-size {W} exceeds K={K} subsegments "
                          f"(T={T}, h={h}, sub_stride={sub_stride})")

    print(f"\nTemporal subsegments: T={T}  h={h}  sub_stride={sub_stride}  "
          f"→ K={K}  starts={starts.tolist()}")

    seg_obs = np.stack([obs[:, s:s + h]    for s in starts], axis=1)  # (N, K, h, obs_dim)
    seg_act = np.stack([action[:, s:s + h] for s in starts], axis=1)  # (N, K, h, act_dim)
    seg_rew = np.stack([reward[:, s:s + h] for s in starts], axis=1)  # (N, K, h)

    # ------------------------------------------------------------------
    # Step 2-3: score every subsegment, normalize GLOBALLY to [-1, 1]
    # ------------------------------------------------------------------
    print(f"\nScoring {N * K:,} subsegments with oracle rl_sum (h={h}) …")
    flat_obs   = seg_obs.reshape(N * K, h, seg_obs.shape[-1])
    flat_rew   = seg_rew.reshape(N * K, h)
    raw_scores = score_rl_sum(avi, flat_obs, flat_rew).reshape(N, K)   # (N, K)

    print(f"  rl_sum: mean={raw_scores.mean():.3f}  std={raw_scores.std():.3f}  "
          f"p1={np.percentile(raw_scores,1):.3f}  p99={np.percentile(raw_scores,99):.3f}")

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
    if noise_std > 0:
        noise = rng.normal(0.0, noise_std, size=(N, K)).astype(np.float32)
        f = np.clip(f_oracle + noise, -1.0, 1.0)
    else:
        f = f_oracle
    print(f"  Normalized scalar f (noise_std={noise_std}): "
          f"mean={f.mean():.3f}  std={f.std():.3f}")

    # ------------------------------------------------------------------
    # Steps 5-6: comparison windows → lag-filtered candidate pairs → sample
    #            → hard preference.  The candidate structure only depends on
    #            K, W, cmp_stride, min_lag, max_lag — identical for every
    #            trajectory — so this is fully vectorized over N.
    # ------------------------------------------------------------------
    win_offsets = list(range(0, K - W + 1, cmp_stride))
    if not win_offsets:
        raise ValueError(f"--window-size {W} exceeds K={K} subsegments (cmp_stride={cmp_stride})")

    local_candidates = [(i, j) for i in range(W) for j in range(i + 1, W)
                         if min_lag <= (j - i) <= max_lag]
    if not local_candidates:
        raise ValueError(f"No candidate pairs satisfy min_lag={min_lag} <= j-i <= "
                          f"max_lag={max_lag} for window_size={W}")
    n_cand   = len(local_candidates)
    n_sample = min(max_per_win, n_cand)

    print(f"\nComparison windows/trajectory : {len(win_offsets)}  (W={W}, cmp_stride={cmp_stride})")
    print(f"Eligible candidate pairs/window: {n_cand}  (min_lag={min_lag}, max_lag={max_lag})")
    print(f"Sampled pairs/window          : {n_sample}  (max_pairs_per_window={max_per_win})")
    print(f"δ = {delta}  (pairs with |f_i-f_j| <= δ are discarded as ties)")

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
        tie_mask = np.abs(diff) <= delta
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
        out_ckpt.append(np.stack([ckpt[rows], ckpt[rows]], axis=1).astype(np.int64))

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

    # Sanity: every pair must come from a single trajectory.
    assert np.array_equal(ckpt_out[:, 0], ckpt[traj_out]), "cross-trajectory checkpoint mismatch"

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
    gaps = adv_out[:, 0] - adv_out[:, 1]         # always > 0
    n_unique_traj = len(np.unique(traj_out))
    pct = [5, 25, 50, 75, 95]

    print(f"\n{'─'*60}")
    print(f"Scalar feedback statistics")
    print(f"{'─'*60}")
    print(f"  Trajectories total     : {N}")
    print(f"  Trajectories contributing pairs : {n_unique_traj}")
    print(f"  Subsegments/trajectory : K={K}  (h={h}, sub_stride={sub_stride})")
    print(f"  Candidates/window      : {n_cand}  →  sampled {n_sample}")
    print(f"  Ties discarded (|Δf|<={delta}): {n_ties_total}"
          f"  ({100*n_ties_total/max(1,n_sampled_total):.1f}% of sampled candidates)")
    print(f"  Pairs kept             : {M}  (avg {M/max(1,N):.2f} per trajectory)")

    print(f"\n  Normalized scalar f (all subsegments, before pairing):")
    f_vals = np.percentile(f_oracle, pct)
    print(f"    mean={f_oracle.mean():.3f}  std={f_oracle.std():.3f}")
    print(f"    {'  '.join(f'p{p}={v:.3f}' for p, v in zip(pct, f_vals))}")

    gap_vals = np.percentile(gaps, pct)
    print(f"\n  Scalar gap Δf (preferred − non-preferred f):")
    print(f"    mean={gaps.mean():.3f}  std={gaps.std():.3f}  "
          f"min={gaps.min():.3f}  max={gaps.max():.3f}")
    print(f"    {'  '.join(f'p{p}={v:.3f}' for p, v in zip(pct, gap_vals))}")
    print(f"{'─'*60}")

    save_npz(args.out,
             obs=obs_out, action=act_out, reward=rew_out,
             adv_scores=adv_out, checkpoint_step=ckpt_out,
             traj_id=traj_out, start_idx=start_out,
             n_choice_structures=np.int64(N))

    print(f"\nSaved → {args.out}")
    print(f"  obs              : {obs_out.shape}"
          f"  ([0]=preferred, [1]=non-preferred)  h={h}")
    print(f"  adv_scores       : {adv_out.shape}  (scalar values f ∈ [-1,1])")
    print(f"  traj_id          : {traj_out.shape}  start_idx: {start_out.shape}")
    print(f"  n_choice_structures: {N}  (trajectories; pairs={M})")
    print(f"\nTrain with:  dataset: PMFeedbackBuffer")


# ---------------------------------------------------------------------------
# Feedback type: credit assignment (multi-way subsegment selection)
# ---------------------------------------------------------------------------

def generate_credit_assignment(pool, avi, args, rng):
    """
    For each pool trajectory, construct all contiguous windows of length k
    and select the one with the highest rl_sum advantage as the oracle choice.

    C = T - k + 1 candidate windows per trajectory.

    Output schema (loaded by PMCreditAssignmentBuffer):
        obs          : (N, C, k, obs_dim)  all candidate windows
        action       : (N, C, k, act_dim)
        adv_scores   : (N, C)              rl_sum for every candidate
        chosen_idx   : (N,)  int32         argmax of adv_scores per trajectory
        checkpoint_step : (N,)
    """
    obs    = pool["obs"]              # (N, T, obs_dim)
    action = pool["action"]           # (N, T, act_dim)
    reward = pool["reward"]           # (N, T)
    ckpt   = pool["checkpoint_step"]  # (N,)
    N, T   = obs.shape[:2]
    k      = args.subsegment_len
    C      = T - k + 1

    if k >= T:
        raise ValueError(f"--subsegment-len {k} must be < pool trajectory length T={T}")
    if C < 2:
        raise ValueError(f"Only {C} window(s) with k={k}, T={T} — need at least 2 candidates.")

    print(f"\nCredit assignment setup:")
    print(f"  T={T}  k={k}  C={C} candidates per trajectory")
    print(f"  N={N} pool trajectories → up to {N} CA examples")

    # Build all sliding windows (vectorized)
    print(f"\nBuilding all {N}×{C} windows …")
    all_win_obs = np.stack([obs[:, i:i + k] for i in range(C)], axis=1)     # (N, C, k, obs_dim)
    all_win_act = np.stack([action[:, i:i + k] for i in range(C)], axis=1)  # (N, C, k, act_dim)
    all_win_rew = np.stack([reward[:, i:i + k] for i in range(C)], axis=1)  # (N, C, k)

    # Score all windows with rl_sum (batch over N*C)
    print(f"  Scoring {N * C:,} windows with score_rl_sum …")
    flat_obs    = all_win_obs.reshape(N * C, k, obs.shape[2])
    flat_rew    = all_win_rew.reshape(N * C, k)
    flat_scores = score_rl_sum(avi, flat_obs, flat_rew).astype(np.float32)  # (N*C,)
    win_scores  = flat_scores.reshape(N, C)                                  # (N, C)

    # Oracle: deterministic argmax
    chosen_idx = np.argmax(win_scores, axis=1).astype(np.int32)  # (N,)

    # Cap to n_pairs (pool already shuffled; just slice)
    if args.n_pairs is not None and N > args.n_pairs:
        all_win_obs = all_win_obs[:args.n_pairs]
        all_win_act = all_win_act[:args.n_pairs]
        win_scores  = win_scores[:args.n_pairs]
        chosen_idx  = chosen_idx[:args.n_pairs]
        ckpt        = ckpt[:args.n_pairs]
        N = args.n_pairs
        print(f"  Capped to {args.n_pairs} examples")

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------
    chosen_scores     = win_scores[np.arange(N), chosen_idx]   # (N,)
    mean_scores       = win_scores.mean(axis=1)                  # (N,)
    min_scores        = win_scores.min(axis=1)                   # (N,)
    gap_vs_mean       = chosen_scores - mean_scores              # (N,) >= 0
    gap_vs_min        = chosen_scores - min_scores               # (N,) >= 0
    all_flat_scores   = win_scores.ravel()
    pct               = [5, 25, 50, 75, 95]

    print(f"\n{'─'*60}")
    print(f"Credit assignment statistics")
    print(f"{'─'*60}")
    print(f"  Examples         : {N}  (1 per pool trajectory)")
    print(f"  Candidates / ex  : C={C}  (T={T}, k={k})")

    sv = np.percentile(all_flat_scores, pct)
    print(f"\n  Window rl_sum (all {N * C:,} windows):")
    print(f"    mean={all_flat_scores.mean():.3f}  std={all_flat_scores.std():.3f}  "
          f"min={all_flat_scores.min():.3f}  max={all_flat_scores.max():.3f}")
    print(f"    {'  '.join(f'p{p}={v:.3f}' for p, v in zip(pct, sv))}")

    cv = np.percentile(chosen_scores, pct)
    print(f"\n  Chosen window rl_sum:")
    print(f"    mean={chosen_scores.mean():.3f}  std={chosen_scores.std():.3f}")
    print(f"    {'  '.join(f'p{p}={v:.3f}' for p, v in zip(pct, cv))}")

    gm = np.percentile(gap_vs_mean, pct)
    gi = np.percentile(gap_vs_min,  pct)
    print(f"\n  Score gap (chosen vs mean / min):")
    print(f"    vs_mean: mean={gap_vs_mean.mean():.3f}  std={gap_vs_mean.std():.3f}  "
          f"{'  '.join(f'p{p}={v:.3f}' for p, v in zip(pct, gm))}")
    print(f"    vs_min:  mean={gap_vs_min.mean():.3f}   std={gap_vs_min.std():.3f}  "
          f"{'  '.join(f'p{p}={v:.3f}' for p, v in zip(pct, gi))}")

    print(f"\n  Chosen window position (0=earliest, {C - 1}=latest in trajectory):")
    pos_vals = np.percentile(chosen_idx, pct)
    print(f"    mean={chosen_idx.mean():.1f}  std={chosen_idx.std():.1f}  "
          f"min={chosen_idx.min()}  max={chosen_idx.max()}")
    print(f"    {'  '.join(f'p{p}={v:.0f}' for p, v in zip(pct, pos_vals))}")
    # 5-bucket histogram over window positions
    bin_edges = np.linspace(0, C, 6).astype(int)
    cnt_bins  = np.bincount(chosen_idx, minlength=C)
    bar_w = 25
    for bi in range(5):
        lo, hi = bin_edges[bi], bin_edges[bi + 1]
        cnt  = int(cnt_bins[lo:hi].sum())
        frac = cnt / N
        bar  = "█" * int(frac * bar_w)
        print(f"    pos [{lo:2d}-{hi:2d}] [{bar:<{bar_w}}] {cnt:5d}  ({100 * frac:5.1f}%)")
    print(f"{'─'*60}")

    obs_out  = all_win_obs.astype(np.float32)   # (N, C, k, obs_dim)
    act_out  = all_win_act.astype(np.float32)   # (N, C, k, act_dim)
    adv_out  = win_scores                        # (N, C) float32
    idx_out  = chosen_idx                        # (N,) int32
    ckpt_out = ckpt.astype(np.int64)            # (N,)

    save_npz(args.out,
             obs=obs_out, action=act_out,
             adv_scores=adv_out, chosen_idx=idx_out,
             checkpoint_step=ckpt_out,
             n_choice_structures=np.int64(N))

    print(f"\nSaved → {args.out}")
    print(f"  obs              : {obs_out.shape}  (N, C, k, obs_dim)")
    print(f"  action           : {act_out.shape}")
    print(f"  adv_scores       : {adv_out.shape}  rl_sum per candidate window")
    print(f"  chosen_idx       : {idx_out.shape}  oracle-selected window index (argmax)")
    print(f"  n_choice_structures: {N}  (trajectories with credit assignment)")
    print(f"\nTrain with:  dataset: PMCreditAssignmentBuffer")
    print(f"             alg:     PointMassCreditAssignmentCPL")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Simulate human feedback for PointMass using VI advantage."
    )
    parser.add_argument("--type", required=True,
                        choices=["pref", "corr", "demo",
                                 "scalar", "credit_assignment"],
                        help="Feedback type to generate")
    parser.add_argument("--pool",          type=str, required=True,
                        help="Path to pool.npz from generate_pm_pool.py")
    parser.add_argument("--advantage-npz", type=str, required=True,
                        help="Path to advantage_vi_*.npz from compute_advantage_vi.py")
    parser.add_argument("--out",           type=str, required=True,
                        help="Output .npz path")

    # Output size cap
    parser.add_argument("--n-pairs", type=int, default=None,
                        help="Number of pairs/sets to save (default: all). "
                             "pref: top-n by gap from the sorted candidate pool. "
                             "corr/demo: stop early once reached. "
                             "scalar: random subsample after generating all pairs.")
    parser.add_argument("--skip-expert", action="store_true", default=False,
                        help="Remove tier-0 (expert, checkpoint_step==0) segments "
                             "from the pool before generating feedback. Recommended "
                             "for corr so the expert rollout provides "
                             "real correction signal over sub-optimal pool segments.")

    # Common scoring
    parser.add_argument("--gamma",        type=float, default=0.99)
    parser.add_argument("--min-adv-gap",  type=float, default=0.01,
                        help="Min |rl_sum gap| to keep a pair (default: 0.5). "
                             "Used by pref, corr, demo.")

    # Pref-specific candidate pool
    parser.add_argument("--n-candidates", type=int, default=None,
                        help="Fixed number of candidate pairs to sample for pref "
                             "(default: None → n_pairs × oversampling_factor). "
                             "Set to a fixed value across all budget runs to guarantee "
                             "that smaller budgets are literal subsets of larger ones.")
    parser.add_argument("--oversampling-factor", type=float, default=3.0,
                        help="Fallback candidate multiplier when --n-candidates is not "
                             "set: n_candidates = n_pairs × factor (default: 3.0). "
                             "WARNING: this breaks the subset property across budget "
                             "levels since n_candidates then varies with n_pairs.")
    parser.add_argument("--sort-by-gap", action="store_true", default=False,
                        help="Sort filtered candidates by |gap| descending before "
                             "taking the top-n prefix (default: False — random shuffle). "
                             "Shuffle gives representative nested subsets; sort gives "
                             "the most extreme comparisons at every budget level.")

    # Demo-specific
    parser.add_argument("--n-counterfactuals", type=int, default=4,
                        help="Number of noisy-expert counterfactuals in demo "
                             "(total K = n_counterfactuals + 2, default: 4 → K=6)")

    # Scalar-feedback specific (per-trajectory temporal subsegments)
    parser.add_argument("--segment-len",  type=int,   default=16,
                        help="Subsegment length h in steps (default: 16)")
    parser.add_argument("--sub-stride",   type=int,   default=12,
                        help="Stride between subsegment start offsets (default: 12); "
                             "starts = arange(0, T-h+1, sub_stride)")
    parser.add_argument("--window-size",  type=int,   default=5,
                        help="Subsegments K per comparison window W (default: 5)")
    parser.add_argument("--cmp-stride",   type=int,   default=5,
                        help="Stride between comparison windows over the K "
                             "subsegments of a trajectory (default: 5)")
    parser.add_argument("--min-lag",      type=int,   default=2,
                        help="Minimum subsegment index gap j-i for a candidate "
                             "pair (default: 2; drops neighboring subsegments)")
    parser.add_argument("--max-lag",      type=int,   default=4,
                        help="Maximum subsegment index gap j-i for a candidate "
                             "pair (default: 4)")
    parser.add_argument("--max-pairs-per-window", type=int, default=4,
                        help="Cap on candidate pairs sampled per comparison "
                             "window, per trajectory (default: 4)")
    parser.add_argument("--noise-std",    type=float, default=0.0,
                        help="Std of Gaussian noise added to the normalized "
                             "oracle scalar (default: 0.0, exact oracle score)")
    parser.add_argument("--scalar-delta", type=float, default=0.0,
                        help="Indifference threshold δ: discard a pair only "
                             "when |f_i-f_j| <= δ (default: 0.0, exact ties only)")

    # Credit-assignment specific
    parser.add_argument("--subsegment-len", type=int, default=12,
                        help="Length k of each candidate subsegment (default: 12)")

    parser.add_argument("--seed",    type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    print("=" * 65)
    print(f"PointMass feedback generation  type={args.type}")
    print(f"  Pool         : {args.pool}")
    print(f"  Advantage    : {args.advantage_npz}")
    print(f"  Output       : {args.out}")
    print(f"  γ            : {args.gamma}")
    print(f"  Seed         : {args.seed}")
    if args.n_pairs is not None:
        print(f"  n_pairs      : {args.n_pairs}  (cap)")
    if args.type == "scalar":
        print(f"  segment_len (h)      : {args.segment_len}")
        print(f"  sub_stride           : {args.sub_stride}")
        print(f"  window_size (W)      : {args.window_size}")
        print(f"  cmp_stride           : {args.cmp_stride}")
        print(f"  min_lag / max_lag    : {args.min_lag} / {args.max_lag}")
        print(f"  max_pairs_per_window : {args.max_pairs_per_window}")
        print(f"  noise_std            : {args.noise_std}")
        print(f"  scalar_delta         : {args.scalar_delta}")
    if args.type == "credit_assignment":
        print(f"  subsegment_len: {args.subsegment_len}")
    print("=" * 65)

    # Load pool
    print("\nLoading pool …")
    with open(args.pool, "rb") as f:
        pool = dict(np.load(f))
    N, T = pool["obs"].shape[:2]
    print(f"  {N} trajectories  T={T}  obs_dim={pool['obs'].shape[2]}")

    # Optionally drop expert tier (tier 0 = checkpoint_step == 0)
    if args.skip_expert:
        mask = pool["checkpoint_step"] != 0
        n_before = N
        pool = {k: v[mask] for k, v in pool.items()}
        N = pool["obs"].shape[0]
        n_dropped = n_before - N
        print(f"  --skip-expert: removed {n_dropped} tier-0 segments → {N} remaining")

    # Shuffle so --n-pairs samples uniformly across quality tiers
    idx = rng.permutation(N)
    pool = {k: v[idx] for k, v in pool.items()}
    print(f"  Pool shuffled (seed={args.seed})")

    # Load VI advantage table
    print(f"\nLoading advantage table …")
    avi = AdvantageVI.load(args.advantage_npz)
    print(f"  N={avi.N}  K={len(avi.action_grid)}  α={avi.alpha:.4f}")

    # Dispatch
    if args.type == "pref":
        generate_pref(pool, avi, args, rng)
    elif args.type == "corr":
        env = PointMassGymEnv(step_cost=0.01)
        generate_corr(pool, avi, env, args, rng)
    elif args.type == "demo":
        env = PointMassGymEnv(step_cost=0.01)
        generate_demo(pool, avi, env, args, rng)
    elif args.type == "scalar":
        generate_scalar(pool, avi, args, rng)
    elif args.type == "credit_assignment":
        generate_credit_assignment(pool, avi, args, rng)


if __name__ == "__main__":
    main()
