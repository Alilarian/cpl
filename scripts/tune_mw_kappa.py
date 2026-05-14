"""
Tune the e-stop κ (kappa) hyperparameter for MetaWorld environments.

This is the MetaWorld analogue of tune_pm_kappa.py.  Instead of a
precomputed VI advantage table, V is estimated via MCMC over the trained
SAC actor (same approach as generate_seq_estop_labels.py).

Workflow:
    1. Load pool, optionally subsample --n-sample trajectories for speed.
    2. Compute Δ_t via MCMC for the (sub)sample (the expensive step).
    3. Compute peak_H per trajectory and derive κ candidates as percentiles.
    4. Simulate seq-estop for each κ; report stop_rate, mean_τ, est_pairs.
    5. Recommend the lowest κ satisfying mean_τ ≥ h AND est_pairs ≥ target.

Typical usage pattern:
    - Tune on ~30k segments (fast, ~10 min on GPU).
    - Copy recommended κ into generate_seq_estop_labels.sbatch.
    - Generate 300k pairs from the full 300k pool with that κ.

Usage:
    python scripts/tune_mw_kappa.py \\
        --pool-path  /scratch/.../demo_pool/mw_drawer-open-v2/pool.npz \\
        --run-dir    /scratch/.../oracle_sac_all/mw_drawer-open-v2 \\
        --rho 0.8 --lam 2.0 \\
        --horizon 10 \\
        --n-sample 30000 \\
        --target-pairs 10000

    # All envs (loop):
    for ENV in mw_bin-picking-v2 mw_button-press-v2 mw_door-open-v2 \\
               mw_drawer-open-v2 mw_plate-slide-v2; do
        python scripts/tune_mw_kappa.py \\
            --pool-path /scratch/.../demo_pool/$ENV/pool.npz \\
            --run-dir   /scratch/.../oracle_sac_all/$ENV \\
            --rho 0.8 --lam 2.0 --horizon 10 \\
            --n-sample 30000 --target-pairs 10000
    done
"""

import argparse
import os

import numpy as np
import torch

from research.utils.config import Config


# ---------------------------------------------------------------------------
# Model loading (mirrors generate_seq_estop_labels.py)
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


# ---------------------------------------------------------------------------
# Per-step disadvantage (mirrors generate_seq_estop_labels.py)
# ---------------------------------------------------------------------------

def compute_per_step_disadvantage(obs_b, reward_b, oracle, mcmc_samples, discount, device):
    """
    Δ_t = max(0, V*(s_t) − r_t − γ V*(s_{t+1})) for a batch of B segments.

    Returns:
        delta : (B, T)  numpy, clamped >= 0  (last step padded with 0)
    """
    obs_t    = torch.from_numpy(obs_b).float().to(device)
    reward_t = torch.from_numpy(reward_b).float().to(device)

    with torch.no_grad():
        obs_enc   = oracle.network.encoder(obs_t)
        obs_exp   = obs_enc.unsqueeze(2).expand(-1, -1, mcmc_samples, -1)
        sampled_a = oracle.network.actor(obs_exp).sample()
        q         = oracle.network.critic(obs_exp, sampled_a).mean(dim=0)
        v         = q.mean(dim=2)

        v_curr  = v[:, :-1]
        v_next  = v[:, 1:]
        r_curr  = reward_t[:, :-1]
        td_adv  = r_curr + discount * v_next - v_curr
        delta_core = (-td_adv).clamp(min=0.0)
        pad        = torch.zeros(obs_b.shape[0], 1, device=device, dtype=delta_core.dtype)
        delta      = torch.cat([delta_core, pad], dim=1)

    return delta.cpu().numpy()


# ---------------------------------------------------------------------------
# Kappa sweep utilities (pure numpy — same logic as tune_pm_kappa.py)
# ---------------------------------------------------------------------------

def peak_h_per_traj(delta_all, rho):
    """max H_t = max_t(ρ H_{t-1} + Δ_t) for each trajectory. → (N,)"""
    N, T = delta_all.shape
    peak = np.zeros(N, dtype=np.float32)
    H    = np.zeros(N, dtype=np.float64)
    for t in range(T):
        H = rho * H + delta_all[:, t]
        np.maximum(peak, H, out=peak)
    return peak


def avg_seq_pairs(tau_arr, h, T):
    """Average seq_estop pairs per stopped trajectory at the given τ values."""
    if len(tau_arr) == 0:
        return 0.0
    t_min = h - 1
    depth = np.clip(tau_arr, t_min, T - h) - t_min + 1
    valid = tau_arr >= t_min
    if valid.sum() == 0:
        return 0.0
    return float(depth[valid].mean())


def simulate_seq_estop_single(delta, rho, lam, kappa, rng):
    """Run sequential e-stop for one trajectory. Returns τ or None (censored)."""
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


def simulate_kappa(delta_all, rho, lam, kappa, h, T, rng):
    """Simulate full estop for one κ; return summary metrics."""
    N = delta_all.shape[0]
    taus = []
    for i in range(N):
        tau = simulate_seq_estop_single(delta_all[i], rho, lam, kappa, rng)
        if tau is not None:
            taus.append(tau)

    stop_rate = len(taus) / N
    if not taus:
        return dict(stop_rate=stop_rate, mean_tau=None, median_tau=None,
                    pct_valid=0.0, est_seq_pairs=0)

    taus_arr   = np.array(taus, dtype=np.int32)
    mean_tau   = float(np.mean(taus_arr))
    median_tau = float(np.median(taus_arr))
    valid_mask = taus_arr >= (h - 1)
    pct_valid  = float(valid_mask.mean())
    avg_p      = avg_seq_pairs(taus_arr, h, T)
    est_seq    = int(N * stop_rate * pct_valid * avg_p)

    return dict(
        stop_rate=stop_rate,
        mean_tau=mean_tau,
        median_tau=median_tau,
        pct_valid=pct_valid,
        est_seq_pairs=est_seq,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Tune e-stop κ for MetaWorld by sweeping percentile-based candidates."
    )
    parser.add_argument("--pool-path", type=str, required=True,
                        help="Path to pool.npz from build_trajectory_pool.py")
    parser.add_argument("--run-dir", type=str, required=True,
                        help="Oracle SAC run dir (config.yaml + checkpoint)")
    parser.add_argument("--oracle-checkpoint", type=str, default="best_model.pt")
    parser.add_argument("--rho",     type=float, default=0.8)
    parser.add_argument("--lam",     type=float, default=2.0)
    parser.add_argument("--gamma",   type=float, default=0.99)
    parser.add_argument("--horizon", type=int,   default=10,
                        help="seq_estop window h (default: 10)")
    parser.add_argument("--n-sample", type=int, default=30000,
                        help="Subsample this many segments from the pool for tuning "
                             "(default: 30000). Use None / 0 to use the full pool.")
    parser.add_argument("--batch-size",   type=int, default=256)
    parser.add_argument("--mcmc-samples", type=int, default=64)
    parser.add_argument("--target-pairs", type=int, default=10000,
                        help="Target seq_estop pair count from the subsample "
                             "(default: 10000). Scale up by pool_size/n_sample "
                             "to estimate pairs from full pool.")
    parser.add_argument("--percentiles", type=float, nargs="+",
                        default=[10, 20, 30, 40, 50, 60, 70, 80, 90])
    parser.add_argument("--seed",   type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    env_name = os.path.basename(os.path.dirname(args.pool_path))
    rng = np.random.default_rng(args.seed)
    h   = args.horizon

    print("=" * 65)
    print(f"Environment  : {env_name}")
    print(f"Pool path    : {args.pool_path}")
    print(f"Oracle dir   : {args.run_dir}")
    print(f"ρ={args.rho}  λ={args.lam}  h={h}  γ={args.gamma}")
    print(f"N-sample     : {args.n_sample if args.n_sample else 'all'}")
    print(f"Target pairs : {args.target_pairs:,}  (from subsample)")
    print(f"Device       : {device}")
    print("=" * 65)

    # ------------------------------------------------------------------
    # Load pool (and optionally subsample for speed)
    # ------------------------------------------------------------------
    print("\nLoading pool ...")
    with open(args.pool_path, "rb") as f:
        pool = np.load(f)
        pool_obs    = pool["obs"]     # (N, T, obs_dim)
        pool_reward = pool["reward"]  # (N, T)

    N_full, T = pool_obs.shape[:2]
    print(f"  Full pool: {N_full:,} trajectories  T={T}")

    if T < 2 * h:
        print(f"ERROR: T={T} < 2h={2*h}. Increase pool T or reduce --horizon.")
        return

    if args.n_sample and args.n_sample < N_full:
        idx = rng.choice(N_full, size=args.n_sample, replace=False)
        pool_obs    = pool_obs[idx]
        pool_reward = pool_reward[idx]
        N = args.n_sample
        print(f"  Subsampled to {N:,} for tuning (seed={args.seed})")
    else:
        N = N_full
        print(f"  Using full pool ({N:,} trajectories)")

    scale_factor = N_full / N  # for extrapolating pairs to full pool
    print(f"  Scale factor to full pool: {scale_factor:.1f}×")

    # ------------------------------------------------------------------
    # Load oracle
    # ------------------------------------------------------------------
    oracle_ckpt = os.path.join(args.run_dir, args.oracle_checkpoint)
    print(f"\nLoading oracle: {oracle_ckpt}")
    oracle = load_model(args.run_dir, oracle_ckpt, device)

    # ------------------------------------------------------------------
    # Compute Δ_t for the (sub)sample
    # ------------------------------------------------------------------
    print(f"\nComputing Δ_t for {N:,} trajectories ...")
    delta_all = np.empty((N, T), dtype=np.float32)
    n_batches = (N + args.batch_size - 1) // args.batch_size

    for b in range(n_batches):
        s = b * args.batch_size
        e = min(s + args.batch_size, N)
        delta_all[s:e] = compute_per_step_disadvantage(
            pool_obs[s:e], pool_reward[s:e], oracle,
            args.mcmc_samples, args.gamma, device,
        )
        if (b + 1) % 20 == 0 or b == n_batches - 1:
            print(f"  [{b+1:>4}/{n_batches}]  {e:>7,}/{N:,}")

    print(f"\nΔ_t:  mean={delta_all.mean():.4f}  std={delta_all.std():.4f}"
          f"  p95={np.percentile(delta_all, 95):.4f}")

    # ------------------------------------------------------------------
    # Peak-H per trajectory → κ candidates
    # ------------------------------------------------------------------
    print(f"\nPeak H per trajectory (ρ={args.rho}) ...")
    peak_h = peak_h_per_traj(delta_all, args.rho)
    h_ss   = float(delta_all.mean()) / (1.0 - args.rho)
    print(f"  peak_H: mean={peak_h.mean():.3f}  std={peak_h.std():.3f}"
          f"  p50={np.percentile(peak_h, 50):.3f}  H_ss={h_ss:.3f}")

    kappa_candidates = [
        (p, float(np.percentile(peak_h, p))) for p in args.percentiles
    ]

    # ------------------------------------------------------------------
    # Simulate each κ
    # ------------------------------------------------------------------
    print(f"\nSimulating seq-estop for each κ  (N_sample={N:,}  h={h}) ...")
    print(f"\n{'─'*90}")
    print(f"{'pct':>4}  {'κ':>7}  {'stop%':>6}  "
          f"{'mean_τ':>7}  {'med_τ':>6}  "
          f"{'valid%':>7}  {'est_pairs_sample':>17}  {'est_pairs_full':>14}  {'OK?':>4}")
    print("─" * 90)

    recommendations = []
    for p, kv in kappa_candidates:
        rng_sim = np.random.default_rng(args.seed)
        m = simulate_kappa(delta_all, args.rho, args.lam, kv, h, T, rng_sim)
        est_full = int(m["est_seq_pairs"] * scale_factor)

        ok = (
            m["mean_tau"] is not None
            and m["mean_tau"] >= h
            and m["est_seq_pairs"] >= args.target_pairs
        )
        if ok:
            recommendations.append((p, kv, m, est_full))

        mt  = f"{m['mean_tau']:.1f}"   if m["mean_tau"]   is not None else "  N/A"
        mdt = f"{m['median_tau']:.1f}" if m["median_tau"] is not None else "  N/A"
        print(
            f"  p{p:02d}  κ={kv:6.4f}  "
            f"{100*m['stop_rate']:5.1f}%  "
            f"{mt:>7}  {mdt:>6}  "
            f"{100*m['pct_valid']:6.1f}%  "
            f"{m['est_seq_pairs']:>17,}  "
            f"{est_full:>14,}  "
            f"{'✓' if ok else '✗':>4}"
        )

    print("─" * 90)

    if recommendations:
        best_p, best_kv, best_m, best_est_full = recommendations[0]
        print(f"\n✓ Recommended κ = {best_kv:.4f}  (p{best_p} of peak-H in subsample)")
        print(f"  stop_rate       = {100*best_m['stop_rate']:.1f}%")
        print(f"  mean_τ          = {best_m['mean_tau']:.1f}  (need ≥ h={h})")
        print(f"  pct_valid       = {100*best_m['pct_valid']:.1f}%")
        print(f"  est_pairs_sample= {best_m['est_seq_pairs']:,}  (from {N:,} segs)")
        print(f"  est_pairs_full  = {best_est_full:,}  (scaled to {N_full:,} segs)")
        print(f"\nCopy into generate_seq_estop_labels.sbatch:")
        print(f"  KAPPA=\"{best_kv:.4f}\"")
    else:
        print(f"\n✗ No κ met both criteria (mean_τ ≥ {h} AND est_pairs ≥ {args.target_pairs:,}).")
        print("  Options:")
        print(f"    • Lower --target-pairs (current: {args.target_pairs:,})")
        print(f"    • Lower --horizon (current: {h})")
        print(f"    • Add percentiles < p{min(p for p, _ in kappa_candidates)}")


if __name__ == "__main__":
    main()
