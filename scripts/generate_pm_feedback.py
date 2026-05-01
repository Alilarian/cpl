"""
Simulate all human feedback types for the PointMass navigation environment.

Uses a precomputed VI advantage table (advantage_vi.npz) instead of a live
SAC critic, so no GPU or MuJoCo is needed.

Feedback types  (--type)
------------------------
  pref       Pairwise preference — score pool pairs with rl_sum, keep better
  corr       Corrective         — pool segment vs expert rollout from same s_0
  demo       Demonstrative      — K-way counterfactuals sorted by rl_sum
  estop      E-stop             — halt prefix preferred over full trajectory
  seq_estop  Sequential e-stop  — h-step window pairs around stop time τ

Scoring metric (same as MetaWorld generate_*_labels.py)
--------------------------------------------------------
  rl_sum(σ) = Σ_t r_t  +  V*(s_T) − V*(s_0)

  V*(s) is obtained by bilinear interpolation in the VI value table.

Per-step disadvantage (e-stop types)
-------------------------------------
  Δ_t = max(0,  V*(s_t) − r_t − γ V*(s_{t+1}))

Output schemas (identical to MetaWorld equivalents)
----------------------------------------------------
  pref / corr :  obs (N,2,T,2)  action (N,2,T,2)  reward (N,2,T)
                 adv_scores (N,2)  [gap (N,) for pref]
  demo        :  obs (N,K,T,2)  action (N,K,T,2)  reward (N,K,T)
                 adv_scores (N,K)
  estop       :  obs (M,2,T,2)  action (M,2,T,2)  reward (M,2,T)
                 stop_time (M,)  checkpoint_step (M,)
  seq_estop   :  obs (M,2,h,2)  action (M,2,h,2)  reward (M,2,h)
                 stop_event (M,)  timestep (M,)  traj_idx (M,)
                 checkpoint_step (M,)

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

# E-stop
python scripts/generate_pm_feedback.py \\
    --type estop \\
    --pool datasets/pm/pool.npz \\
    --advantage-npz runs/pm_sac_oracle/advantage_vi_N100.npz \\
    --out datasets/pm/estop_labels.npz \\
    --rho 0.8 --lam 2.0 --kappa 0.3

# Sequential e-stop
python scripts/generate_pm_feedback.py \\
    --type seq_estop \\
    --pool datasets/pm/pool.npz \\
    --advantage-npz runs/pm_sac_oracle/advantage_vi_N100.npz \\
    --out datasets/pm/seq_estop_labels.npz \\
    --rho 0.8 --lam 2.0 --kappa 0.3 --horizon 10
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


def compute_delta(avi: AdvantageVI, obs: np.ndarray, reward: np.ndarray,
                  gamma: float = 0.99) -> np.ndarray:
    """
    Per-step disadvantage  Δ_t = max(0, V*(s_t) − r_t − γ V*(s_{t+1}))

    obs    : (N, T, 2)
    reward : (N, T)
    returns: (N, T) float32  (last column padded with 0)
    """
    V = vi_values(avi, obs)                          # (N, T)
    td = reward[:, :-1] + gamma * V[:, 1:] - V[:, :-1]   # (N, T-1)
    delta_core = np.maximum(0.0, -td).astype(np.float32)
    pad = np.zeros((obs.shape[0], 1), dtype=np.float32)
    return np.concatenate([delta_core, pad], axis=1)       # (N, T)


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
# E-stop simulation (shared by estop and seq_estop)
# ---------------------------------------------------------------------------

def simulate_estop(delta: np.ndarray, rho: float, lam: float, kappa: float,
                   rng: np.random.Generator):
    """
    Run the sequential-stop simulation for a single trajectory.

    H_t = ρ H_{t-1} + Δ_t
    p_t = σ(λ(H_t − κ))
    τ   = first t where Bernoulli(p_t) = 1,  or None (censored)
    """
    T = len(delta)
    H = np.empty(T, dtype=np.float64)
    H[0] = delta[0]
    for t in range(1, T):
        H[t] = rho * H[t - 1] + delta[t]

    x = lam * (H - kappa)
    p = np.where(x >= 0,
                 1.0 / (1.0 + np.exp(-x)),
                 np.exp(x) / (1.0 + np.exp(x)))

    for t in range(T):
        if rng.random() < p[t]:
            return t
    return None


# ---------------------------------------------------------------------------
# Feedback type: pairwise preference
# ---------------------------------------------------------------------------

def generate_pref(pool, avi, args, rng):
    """
    Score all pool segments, randomly pair them, keep pairs where
    |rl_sum_a − rl_sum_b| >= gap_threshold.
    """
    obs    = pool["obs"]              # (N, T, 2)
    action = pool["action"]           # (N, T, 2)
    reward = pool["reward"]           # (N, T)
    ckpt   = pool["checkpoint_step"]  # (N,)
    N      = obs.shape[0]

    print(f"\nScoring {N} pool segments …")
    scores = score_rl_sum(avi, obs, reward)             # (N,)
    print(f"  rl_sum: mean={scores.mean():.3f}  std={scores.std():.3f}"
          f"  min={scores.min():.3f}  max={scores.max():.3f}")

    # Shuffle indices to make random pairs
    idx = rng.permutation(N)
    if N % 2 == 1:
        idx = idx[:-1]                                  # drop last if odd
    a_idx = idx[:N // 2]
    b_idx = idx[N // 2:]

    gap_threshold = args.min_adv_gap
    print(f"  Gap threshold  : {gap_threshold:.4f}")

    out_obs, out_act, out_rew, out_adv, out_gap, out_ckpt = [], [], [], [], [], []

    n_kept = n_dropped = 0
    for ai, bi in zip(a_idx, b_idx):
        gap = float(scores[ai] - scores[bi])
        if abs(gap) < gap_threshold:
            n_dropped += 1
            continue

        # Index 0 = preferred (higher rl_sum)
        if gap >= 0:
            pref_i, nonpref_i = int(ai), int(bi)
        else:
            pref_i, nonpref_i = int(bi), int(ai)
            gap = -gap

        out_obs.append(np.stack([obs[pref_i],    obs[nonpref_i]],    axis=0))
        out_act.append(np.stack([action[pref_i], action[nonpref_i]], axis=0))
        out_rew.append(np.stack([reward[pref_i], reward[nonpref_i]], axis=0))
        out_adv.append([scores[pref_i], scores[nonpref_i]])
        out_gap.append(gap)
        out_ckpt.append([int(ckpt[pref_i]), int(ckpt[nonpref_i])])
        n_kept += 1

    print(f"  Pairs kept={n_kept}  dropped={n_dropped} (gap < threshold)")

    if n_kept == 0:
        print("  ERROR: 0 pairs kept. Lower --min-adv-gap.")
        return

    obs_out  = np.stack(out_obs,  axis=0).astype(np.float32)
    act_out  = np.stack(out_act,  axis=0).astype(np.float32)
    rew_out  = np.stack(out_rew,  axis=0).astype(np.float32)
    adv_out  = np.array(out_adv,  dtype=np.float32)
    gap_out  = np.array(out_gap,  dtype=np.float32)
    ckpt_out = np.array(out_ckpt, dtype=np.int64)

    save_npz(args.out,
             obs=obs_out, action=act_out, reward=rew_out,
             adv_scores=adv_out, gap=gap_out, checkpoint_step=ckpt_out)

    print(f"\nSaved → {args.out}")
    print(f"  obs        : {obs_out.shape}  ([0]=preferred, [1]=non-preferred)")
    print(f"  adv_scores : {adv_out.shape}")
    print(f"  gap        : mean={gap_out.mean():.3f}  median={np.median(gap_out):.3f}")
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
             checkpoint_step=ckpt_out[order])

    print(f"\nSaved → {args.out}")
    print(f"  obs         : {obs_out.shape}  ([0]=better, [1]=worse)")
    print(f"  improvement : mean={impr_out.mean():.3f}  median={np.median(impr_out):.3f}")
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
             adv_scores=adv_out, checkpoint_step=ckpt_out)

    print(f"\nSaved → {args.out}")
    print(f"  obs        : {obs_out.shape}  ([0]=best, [-1]=worst)")
    print(f"  adv_scores : {adv_out.shape}")
    print(f"\nTrain with:  dataset: DemoBuffer   (K={K})")


# ---------------------------------------------------------------------------
# E-stop statistics helpers
# ---------------------------------------------------------------------------

def _tau_stats(tau_arr: np.ndarray, T: int, label: str = "τ") -> None:
    """Print a full percentile table + quartile histogram for stop times."""
    pcts = [5, 10, 25, 50, 75, 90, 95]
    vals = np.percentile(tau_arr, pcts)
    print(f"\n  {label} distribution  (n={len(tau_arr)},  T={T})")
    print(f"    mean={tau_arr.mean():.2f}  std={tau_arr.std():.2f}  "
          f"min={tau_arr.min()}  max={tau_arr.max()}")
    print(f"    {'  '.join(f'p{p}={v:.1f}' for p, v in zip(pcts, vals))}")

    # Quartile histogram: fraction of stops in each quarter of the trajectory
    edges = [0, T // 4, T // 2, 3 * T // 4, T]
    labels_q = ["Q1 (early)", "Q2", "Q3", "Q4 (late)"]
    counts = [int(((tau_arr >= edges[k]) & (tau_arr < edges[k+1])).sum())
              for k in range(4)]
    total = len(tau_arr)
    bar_width = 30
    print(f"\n  Stop-time histogram (trajectory split into quartiles):")
    for lbl, cnt in zip(labels_q, counts):
        frac = cnt / total
        bar  = "█" * int(frac * bar_width)
        print(f"    {lbl:12s} [{bar:<{bar_width}}] {cnt:5d}  ({100*frac:5.1f}%)")


def _delta_stats(delta_all: np.ndarray) -> None:
    """Print Δ_t percentile breakdown and fraction of zero entries."""
    flat = delta_all.ravel()
    nonzero = flat[flat > 0]
    pcts = [25, 50, 75, 90, 95, 99]
    vals = np.percentile(flat, pcts)
    print(f"\n  Δ_t statistics  (shape {delta_all.shape})")
    print(f"    mean={flat.mean():.4f}  std={flat.std():.4f}  "
          f"fraction>0 = {100*len(nonzero)/len(flat):.1f}%")
    print(f"    {'  '.join(f'p{p}={v:.4f}' for p, v in zip(pcts, vals))}")
    if len(nonzero):
        print(f"    non-zero Δ_t:  mean={nonzero.mean():.4f}  "
              f"median={np.median(nonzero):.4f}  p95={np.percentile(nonzero,95):.4f}")


def _reward_comparison_estop(rew_out: np.ndarray, tau_arr: np.ndarray) -> None:
    """
    Compare cumulative reward of halt prefix vs full trajectory.

    rew_out : (M, 2, T)  [0]=halt prefix  [1]=full
    tau_arr : (M,)       stop times
    """
    prefix_total = rew_out[:, 0, :].sum(axis=1)   # Σ r_t for prefix (zero-padded after τ)
    full_total   = rew_out[:, 1, :].sum(axis=1)   # Σ r_t for full trajectory

    # Reward up to τ (same window for fair comparison)
    reward_at_tau = np.array([
        rew_out[i, 1, :int(tau_arr[i]) + 1].sum()
        for i in range(len(tau_arr))
    ])

    print(f"\n  Reward at stop point (cumulative up to τ):")
    print(f"    full traj Σr[0:τ]   : mean={reward_at_tau.mean():.3f}  "
          f"std={reward_at_tau.std():.3f}")
    print(f"    full traj Σr[0:T]   : mean={full_total.mean():.3f}  "
          f"std={full_total.std():.3f}")
    print(f"    halt prefix Σr[0:τ] : mean={prefix_total.mean():.3f}  "
          f"(= Σr[0:τ], reward zeroed after τ)")


# ---------------------------------------------------------------------------
# Feedback type: e-stop (non-sequential)
# ---------------------------------------------------------------------------

def generate_estop(pool, avi, args, rng):
    """
    For each pool trajectory, simulate the e-stop process.  Trajectories where
    the oracle stops produce one pair: halt prefix (preferred) vs full traj.
    Censored trajectories are discarded.
    """
    obs    = pool["obs"]
    action = pool["action"]
    reward = pool["reward"]
    ckpt   = pool["checkpoint_step"]
    N, T   = obs.shape[:2]

    print(f"\nComputing per-step Δ_t for {N} trajectories …")
    delta_all = compute_delta(avi, obs, reward, gamma=args.gamma)     # (N, T)
    print(f"  Δ_t: mean={delta_all.mean():.4f}  std={delta_all.std():.4f}"
          f"  p95={np.percentile(delta_all, 95):.4f}")

    h_ss = float(delta_all.mean()) / (1.0 - args.rho)
    p_ss = 1.0 / (1.0 + np.exp(-args.lam * (h_ss - args.kappa)))
    print(f"  Steady-state H={h_ss:.4f}  → p_stop≈{p_ss:.3f}  (κ={args.kappa})")
    if p_ss < 0.05:
        print("  NOTE: very low stop probability. Lower κ to get more stops.")
    elif p_ss > 0.95:
        print("  NOTE: very high stop probability. Raise κ to allow more trajectories.")

    print(f"\nSimulating e-stop (ρ={args.rho} λ={args.lam} κ={args.kappa}) …")

    out_obs, out_act, out_rew, out_tau, out_ckpt = [], [], [], [], []
    n_stopped = n_censored = 0

    for i in range(N):
        tau = simulate_estop(delta_all[i], args.rho, args.lam, args.kappa, rng)

        if tau is None:
            n_censored += 1
            continue

        n_stopped += 1

        # Halt prefix: repeat state s_τ and zero reward after τ
        prefix_obs = obs[i].copy()
        prefix_obs[tau + 1:] = obs[i, tau]

        prefix_act = action[i].copy()
        prefix_act[tau + 1:] = action[i, tau]

        prefix_rew = reward[i].copy()
        prefix_rew[tau + 1:] = 0.0

        out_obs.append(np.stack([prefix_obs, obs[i]],       axis=0))
        out_act.append(np.stack([prefix_act, action[i]],    axis=0))
        out_rew.append(np.stack([prefix_rew, reward[i]],    axis=0))
        out_tau.append(tau)
        out_ckpt.append(int(ckpt[i]))

        if (i + 1) % 1000 == 0 or i == N - 1:
            print(f"  [{i+1:>5}/{N}]  stopped={n_stopped}  censored={n_censored}")

    M = len(out_obs)
    if M == 0:
        print("\nERROR: 0 pairs generated. Lower κ so more trajectories are stopped.")
        return

    obs_out  = np.stack(out_obs,  axis=0).astype(np.float32)
    act_out  = np.stack(out_act,  axis=0).astype(np.float32)
    rew_out  = np.stack(out_rew,  axis=0).astype(np.float32)
    tau_out  = np.array(out_tau,  dtype=np.int32)
    ckpt_out = np.array(out_ckpt, dtype=np.int64)

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------
    print(f"\n{'─'*60}")
    print(f"E-stop statistics")
    print(f"{'─'*60}")
    print(f"  Trajectories : total={N}  "
          f"stopped={n_stopped} ({100*n_stopped/N:.1f}%)  "
          f"censored={n_censored} ({100*n_censored/N:.1f}%)")

    _delta_stats(delta_all)
    _tau_stats(tau_out, T, label="Stop time τ")
    _reward_comparison_estop(rew_out, tau_out)
    print(f"{'─'*60}")

    save_npz(args.out,
             obs=obs_out, action=act_out, reward=rew_out,
             stop_time=tau_out, checkpoint_step=ckpt_out)

    print(f"\nSaved → {args.out}")
    print(f"  obs       : {obs_out.shape}  ([0]=halt prefix, [1]=full)")
    print(f"  stop_time : {tau_out.shape}  mean τ={tau_out.mean():.1f}")
    print(f"\nTrain with:  dataset: EstopBuffer")


# ---------------------------------------------------------------------------
# Feedback type: sequential e-stop
# ---------------------------------------------------------------------------

def build_seq_pairs(traj_obs, traj_action, traj_reward, tau, h):
    """
    Build (preferred, non-preferred) h-step window pairs for one trajectory.

    Valid decision times: t ∈ [h-1, T-h]
      stop_seg = traj[t-h+1 : t+1]   (last h steps ending at t)
      cont_seg = traj[t     : t+h]    (next h steps starting at t)

    Index 0 = preferred (label = 0, compatible with CorrBuffer):
      t == τ  → stop preferred
      t  < τ  → continue preferred
    Censored (τ=None) → all-continue pairs.
    """
    T = traj_obs.shape[0]
    t_min = h - 1
    t_max = T - h
    if t_max < t_min:
        return []

    t_end = t_max if tau is None else min(tau, t_max)
    if t_end < t_min:
        return []

    pairs = []
    for t in range(t_min, t_end + 1):
        is_stop = (tau is not None) and (t == tau)

        stop_obs = traj_obs[t - h + 1: t + 1]
        stop_act = traj_action[t - h + 1: t + 1]
        stop_rew = traj_reward[t - h + 1: t + 1]
        cont_obs = traj_obs[t: t + h]
        cont_act = traj_action[t: t + h]
        cont_rew = traj_reward[t: t + h]

        if is_stop:
            p_obs, p_act, p_rew = stop_obs, stop_act, stop_rew
            n_obs, n_act, n_rew = cont_obs, cont_act, cont_rew
        else:
            p_obs, p_act, p_rew = cont_obs, cont_act, cont_rew
            n_obs, n_act, n_rew = stop_obs, stop_act, stop_rew

        pairs.append({
            "obs":        np.stack([p_obs, n_obs], axis=0),
            "action":     np.stack([p_act, n_act], axis=0),
            "reward":     np.stack([p_rew, n_rew], axis=0),
            "stop_event": 1.0 if is_stop else 0.0,
            "timestep":   t,
        })
    return pairs


def generate_seq_estop(pool, avi, args, rng):
    obs    = pool["obs"]
    action = pool["action"]
    reward = pool["reward"]
    ckpt   = pool["checkpoint_step"]
    N, T   = obs.shape[:2]
    h      = args.horizon

    print(f"\nComputing per-step Δ_t for {N} trajectories …")
    delta_all = compute_delta(avi, obs, reward, gamma=args.gamma)
    print(f"  Δ_t: mean={delta_all.mean():.4f}  std={delta_all.std():.4f}")

    h_ss = float(delta_all.mean()) / (1.0 - args.rho)
    p_ss = 1.0 / (1.0 + np.exp(-args.lam * (h_ss - args.kappa)))
    print(f"  Steady-state H={h_ss:.4f}  → p_stop≈{p_ss:.3f}  (κ={args.kappa})")

    print(f"\nSimulating seq-estop  (h={h}  ρ={args.rho} λ={args.lam} κ={args.kappa}) …")

    all_obs, all_act, all_rew = [], [], []
    all_stop_event, all_timestep, all_traj_idx, all_ckpt = [], [], [], []
    n_stopped = n_censored = 0

    for i in range(N):
        tau = simulate_estop(delta_all[i], args.rho, args.lam, args.kappa, rng)
        if tau is None:
            n_censored += 1
        else:
            n_stopped += 1

        pairs = build_seq_pairs(obs[i], action[i], reward[i], tau, h)

        for p in pairs:
            all_obs.append(p["obs"])
            all_act.append(p["action"])
            all_rew.append(p["reward"])
            all_stop_event.append(p["stop_event"])
            all_timestep.append(p["timestep"])
            all_traj_idx.append(i)
            all_ckpt.append(int(ckpt[i]))

        if (i + 1) % 1000 == 0 or i == N - 1:
            print(f"  [{i+1:>5}/{N}]  stopped={n_stopped}  censored={n_censored}"
                  f"  pairs_so_far={len(all_obs)}")

    M = len(all_obs)
    if M == 0:
        print("\nERROR: 0 pairs generated.")
        return

    obs_out   = np.stack(all_obs,        axis=0).astype(np.float32)
    act_out   = np.stack(all_act,        axis=0).astype(np.float32)
    rew_out   = np.stack(all_rew,        axis=0).astype(np.float32)
    se_out    = np.array(all_stop_event, dtype=np.float32)
    ts_out    = np.array(all_timestep,   dtype=np.int32)
    ti_out    = np.array(all_traj_idx,   dtype=np.int32)
    ckpt_out  = np.array(all_ckpt,       dtype=np.int64)

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------
    tau_stopped = np.array([
        all_timestep[k]
        for k in range(M) if all_stop_event[k] == 1.0
    ], dtype=np.int32)

    pairs_per_traj = np.bincount(np.array(all_traj_idx, dtype=np.int32), minlength=N)
    # Only count trajectories that produced at least one pair
    active_pairs = pairs_per_traj[pairs_per_traj > 0]

    print(f"\n{'─'*60}")
    print(f"Sequential e-stop statistics")
    print(f"{'─'*60}")
    print(f"  Trajectories : total={N}  "
          f"stopped={n_stopped} ({100*n_stopped/N:.1f}%)  "
          f"censored={n_censored} ({100*n_censored/N:.1f}%)")
    print(f"  Total pairs  : {M}  (avg {M/N:.2f} per traj,  "
          f"avg {active_pairs.mean():.2f} per traj with pairs)")

    _delta_stats(delta_all)

    if len(tau_stopped):
        _tau_stats(tau_stopped, T, label="Actual stop time τ (stopped trajs only)")

    # Decision-time distribution across all pairs
    print(f"\n  Decision-time t distribution  (across all {M} pairs)")
    t_pcts = [5, 25, 50, 75, 95]
    t_vals = np.percentile(ts_out, t_pcts)
    print(f"    mean={ts_out.mean():.2f}  std={ts_out.std():.2f}  "
          f"min={ts_out.min()}  max={ts_out.max()}")
    print(f"    {'  '.join(f'p{p}={v:.1f}' for p, v in zip(t_pcts, t_vals))}")

    # Stop-event vs continue-event reward breakdown
    stop_mask = se_out == 1.0
    cont_mask = ~stop_mask
    print(f"\n  Pair type breakdown:")
    print(f"    stop  events : {stop_mask.sum():5d}  ({100*stop_mask.mean():.1f}%)")
    print(f"    cont  events : {cont_mask.sum():5d}  ({100*cont_mask.mean():.1f}%)")

    if stop_mask.any():
        pref_rew_stop  = rew_out[stop_mask, 0, :].sum(axis=1)  # pref  @ stop events
        npref_rew_stop = rew_out[stop_mask, 1, :].sum(axis=1)  # non-pref @ stop events
        print(f"    stop events — pref (stop_seg) Σr : "
              f"mean={pref_rew_stop.mean():.3f}  std={pref_rew_stop.std():.3f}")
        print(f"    stop events — non-pref (cont_seg) Σr : "
              f"mean={npref_rew_stop.mean():.3f}  std={npref_rew_stop.std():.3f}")

    if cont_mask.any():
        pref_rew_cont  = rew_out[cont_mask, 0, :].sum(axis=1)  # pref  @ cont events
        npref_rew_cont = rew_out[cont_mask, 1, :].sum(axis=1)
        print(f"    cont events — pref (cont_seg) Σr : "
              f"mean={pref_rew_cont.mean():.3f}  std={pref_rew_cont.std():.3f}")
        print(f"    cont events — non-pref (stop_seg) Σr : "
              f"mean={npref_rew_cont.mean():.3f}  std={npref_rew_cont.std():.3f}")

    # Pairs-per-trajectory distribution
    print(f"\n  Pairs per trajectory:")
    pp_pcts = [25, 50, 75, 95]
    pp_vals = np.percentile(active_pairs, pp_pcts)
    print(f"    mean={active_pairs.mean():.2f}  std={active_pairs.std():.2f}  "
          f"min={active_pairs.min()}  max={active_pairs.max()}")
    print(f"    {'  '.join(f'p{p}={v:.1f}' for p, v in zip(pp_pcts, pp_vals))}")
    print(f"{'─'*60}")

    save_npz(args.out,
             obs=obs_out, action=act_out, reward=rew_out,
             stop_event=se_out, timestep=ts_out,
             traj_idx=ti_out, checkpoint_step=ckpt_out)

    print(f"\nSaved → {args.out}")
    print(f"  obs        : {obs_out.shape}  ([0]=preferred, [1]=non-preferred)")
    print(f"  stop_event : {se_out.sum():.0f} stops / {M} pairs"
          f"  ({100*se_out.mean():.1f}% are stop events)")
    print(f"\nTrain with:  dataset: SeqEstopBuffer")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Simulate human feedback for PointMass using VI advantage."
    )
    parser.add_argument("--type", required=True,
                        choices=["pref", "corr", "demo", "estop", "seq_estop"],
                        help="Feedback type to generate")
    parser.add_argument("--pool",          type=str, required=True,
                        help="Path to pool.npz from generate_pm_pool.py")
    parser.add_argument("--advantage-npz", type=str, required=True,
                        help="Path to advantage_vi_*.npz from compute_advantage_vi.py")
    parser.add_argument("--out",           type=str, required=True,
                        help="Output .npz path")

    # Common scoring
    parser.add_argument("--gamma",        type=float, default=0.99)
    parser.add_argument("--min-adv-gap",  type=float, default=0.0,
                        help="Min |rl_sum gap| to keep a pair (default: 0.0)")

    # Demo-specific
    parser.add_argument("--n-counterfactuals", type=int, default=4,
                        help="Number of noisy-expert counterfactuals in demo "
                             "(total K = n_counterfactuals + 2, default: 4 → K=6)")

    # E-stop hyperparameters
    parser.add_argument("--rho",     type=float, default=0.8,
                        help="Memory decay ρ ∈ [0,1] (default: 0.8)")
    parser.add_argument("--lam",     type=float, default=2.0,
                        help="Sigmoid slope λ (default: 2.0)")
    parser.add_argument("--kappa",   type=float, default=0.3,
                        help="Stop threshold κ (default: 0.3 for PointMass)")

    # Seq-estop specific
    parser.add_argument("--horizon", type=int, default=10,
                        help="Window length h for seq-estop pairs (default: 10)")

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
    if args.type in ("estop", "seq_estop"):
        print(f"  ρ={args.rho}  λ={args.lam}  κ={args.kappa}")
    if args.type == "seq_estop":
        print(f"  horizon h    : {args.horizon}")
    print("=" * 65)

    # Load pool
    print("\nLoading pool …")
    with open(args.pool, "rb") as f:
        pool = dict(np.load(f))
    N, T = pool["obs"].shape[:2]
    print(f"  {N} trajectories  T={T}  obs_dim={pool['obs'].shape[2]}")

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
    elif args.type == "estop":
        generate_estop(pool, avi, args, rng)
    elif args.type == "seq_estop":
        generate_seq_estop(pool, avi, args, rng)


if __name__ == "__main__":
    main()
