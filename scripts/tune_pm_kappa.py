"""
Tune the e-stop κ (kappa) hyperparameter for PointMass.

Candidate κ values are derived as percentiles of the per-trajectory peak H
(the maximum accumulated-disadvantage value, computed without stopping).
This maps naturally: κ at percentile p  ≈  (100−p)% of trajectories stopped.

For each candidate κ the script simulates the full estop process and reports:
    stop_rate       fraction of trajectories that get stopped
    mean_tau        mean stop time τ (higher = stops deeper in trajectory)
    median_tau      median stop time τ
    pct_valid       fraction of stopped trajs with τ ≥ h−1  (seq_estop validity)
    est_seq_pairs   estimated total seq_estop pairs  (N × stop_rate × pct_valid × avg_depth)

The same κ is recommended for both estop and seq_estop since the stopping
mechanism is identical; seq_estop imposes the stricter constraint (τ ≥ h−1),
so tuning to satisfy it gives meaningful τ for plain estop too.

Usage
-----
python scripts/tune_pm_kappa.py \\
    --pool datasets/pm/pool.npz \\
    --advantage-npz runs/pm_sac_oracle/advantage_vi_N100.npz \\
    --rho 0.8 --lam 2.0 \\
    --horizon 4 \\
    --skip-expert \\
    --target-pairs 10000

The script prints a table and writes the recommended κ to stdout as a one-liner
you can copy into configs and SLURM scripts.
"""

import argparse
import importlib.util
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(dotted, rel):
    if dotted in sys.modules:
        return sys.modules[dotted]
    path = os.path.join(_ROOT, rel)
    spec = importlib.util.spec_from_file_location(dotted, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[dotted] = mod
    spec.loader.exec_module(mod)
    return mod


_pm_fb = _load("generate_pm_feedback", "scripts/generate_pm_feedback.py")
compute_delta   = _pm_fb.compute_delta
simulate_estop  = _pm_fb.simulate_estop
AdvantageVI     = _load("research.utils.advantage_vi",
                        "research/utils/advantage_vi.py").AdvantageVI


# ---------------------------------------------------------------------------
# Peak-H computation (run H recurrence without stopping)
# ---------------------------------------------------------------------------

def peak_h_per_traj(delta_all: np.ndarray, rho: float) -> np.ndarray:
    """
    For each trajectory compute max H_t = max_t(ρ H_{t-1} + Δ_t).
    Shape delta_all: (N, T) → returns (N,).
    """
    N, T = delta_all.shape
    peak = np.zeros(N, dtype=np.float32)
    H = np.zeros(N, dtype=np.float64)
    for t in range(T):
        H = rho * H + delta_all[:, t]
        np.maximum(peak, H, out=peak)
    return peak


# ---------------------------------------------------------------------------
# Per-trajectory avg seq_estop pairs at a given τ
# ---------------------------------------------------------------------------

def avg_seq_pairs(tau_arr: np.ndarray, h: int, T: int) -> float:
    """
    For stopped trajectories, estimate average number of seq_estop pairs.
    Valid decision times: t ∈ [h-1, min(τ, T-h)] inclusive.
    """
    if len(tau_arr) == 0:
        return 0.0
    t_min = h - 1
    depth = np.clip(tau_arr, t_min, T - h) - t_min + 1  # pairs per traj
    valid = tau_arr >= t_min
    if valid.sum() == 0:
        return 0.0
    return float(depth[valid].mean())


# ---------------------------------------------------------------------------
# Simulate one kappa
# ---------------------------------------------------------------------------

def simulate_kappa(delta_all, rho, lam, kappa, h, T, rng):
    N = delta_all.shape[0]
    taus = []
    stopped = 0

    for i in range(N):
        tau = simulate_estop(delta_all[i], rho=rho, lam=lam,
                              kappa=kappa, rng=rng)
        if tau is not None:
            stopped += 1
            taus.append(tau)

    stop_rate = stopped / N

    if len(taus) == 0:
        return dict(stop_rate=stop_rate, mean_tau=None, median_tau=None,
                    pct_valid=0.0, est_seq_pairs=0)

    taus_arr = np.array(taus, dtype=np.int32)
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
        description="Tune e-stop κ for PointMass by sweeping percentile-based candidates."
    )
    parser.add_argument("--pool",          type=str, required=True)
    parser.add_argument("--advantage-npz", type=str, required=True)
    parser.add_argument("--rho",     type=float, default=0.8)
    parser.add_argument("--lam",     type=float, default=2.0)
    parser.add_argument("--gamma",   type=float, default=0.99)
    parser.add_argument("--horizon", type=int,   default=4,
                        help="seq_estop window h (default: 4)")
    parser.add_argument("--skip-expert", action="store_true", default=False,
                        help="Exclude tier-0 (expert) segments from pool")
    parser.add_argument("--target-pairs", type=int, default=10000,
                        help="Target seq_estop pair count used for recommendation")
    parser.add_argument("--percentiles", type=float, nargs="+",
                        default=[10, 20, 30, 40, 50, 60, 70, 80, 90],
                        help="Percentiles of peak-H to use as κ candidates")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    h   = args.horizon

    # ------------------------------------------------------------------
    # Load pool
    # ------------------------------------------------------------------
    print("Loading pool …")
    with open(args.pool, "rb") as f:
        pool = dict(np.load(f))
    N, T = pool["obs"].shape[:2]

    if args.skip_expert:
        mask = pool["checkpoint_step"] != 0
        pool = {k: v[mask] for k, v in pool.items()}
        N    = pool["obs"].shape[0]
        print(f"  --skip-expert: {N} non-expert segments kept")
    print(f"  {N} trajectories  T={T}")

    # ------------------------------------------------------------------
    # Load VI table and compute Δ_t
    # ------------------------------------------------------------------
    print("Loading advantage table …")
    avi = AdvantageVI.load(args.advantage_npz)
    print(f"  N={avi.N}  α={avi.alpha:.4f}")

    print(f"Computing Δ_t for {N} trajectories …")
    delta_all = compute_delta(avi, pool["obs"], pool["reward"], gamma=args.gamma)
    print(f"  Δ_t  mean={delta_all.mean():.4f}  std={delta_all.std():.4f}"
          f"  p95={np.percentile(delta_all, 95):.4f}")

    # ------------------------------------------------------------------
    # Derive candidate κ values from percentiles of peak H
    # ------------------------------------------------------------------
    print("\nComputing peak H per trajectory …")
    peak_h = peak_h_per_traj(delta_all, args.rho)
    h_ss   = float(delta_all.mean()) / (1.0 - args.rho)
    print(f"  peak_H  mean={peak_h.mean():.3f}  std={peak_h.std():.3f}"
          f"  p50={np.percentile(peak_h,50):.3f}  H_ss={h_ss:.3f}")

    # κ at percentile p of peak_H → ~(100−p)% of trajectories stopped
    candidate_kappas = {
        p: float(np.percentile(peak_h, p))
        for p in args.percentiles
    }

    print(f"\nCandidate κ values (from peak-H percentiles):")
    for p, k in candidate_kappas.items():
        print(f"  p{int(p):2d} → κ = {k:.4f}")

    # ------------------------------------------------------------------
    # Sweep
    # ------------------------------------------------------------------
    print(f"\nSimulating estop for each κ  (ρ={args.rho} λ={args.lam}) …")
    print(f"  seq_estop validity requires τ ≥ h−1 = {h-1}  (h={h})")
    print(f"  target seq_estop pairs: {args.target_pairs:,}")

    results = []
    for p, kappa in candidate_kappas.items():
        rng_k = np.random.default_rng(args.seed)   # same seed for each κ
        r = simulate_kappa(delta_all, args.rho, args.lam, kappa, h, T, rng_k)
        r["percentile"] = p
        r["kappa"]      = kappa
        results.append(r)

    # ------------------------------------------------------------------
    # Print table
    # ------------------------------------------------------------------
    print()
    hdr = (f"{'pct':>4}  {'kappa':>7}  {'stop%':>6}  "
           f"{'mean_τ':>7}  {'med_τ':>6}  "
           f"{'valid%':>7}  {'est_seq_pairs':>13}")
    print(hdr)
    print("-" * len(hdr))

    for r in results:
        mt  = f"{r['mean_tau']:7.2f}"   if r['mean_tau']   is not None else "   None"
        mdt = f"{r['median_tau']:6.1f}" if r['median_tau'] is not None else "  None"
        print(
            f"  p{int(r['percentile']):2d}  {r['kappa']:7.4f}  "
            f"{r['stop_rate']*100:5.1f}%  "
            f"{mt}  {mdt}  "
            f"{r['pct_valid']*100:6.1f}%  "
            f"{r['est_seq_pairs']:>13,}"
        )

    # ------------------------------------------------------------------
    # Recommendation
    # ------------------------------------------------------------------
    # Pick the LOWEST κ (most stops) that satisfies BOTH:
    #   (a) mean_tau >= h          (stops deep enough for seq_estop windows)
    #   (b) est_seq_pairs >= target
    print(f"\nSelection criteria:")
    print(f"  (a) mean_tau >= {h}  (stops happen after at least h={h} steps)")
    print(f"  (b) est_seq_pairs >= {args.target_pairs:,}")

    candidates_ok = [
        r for r in results
        if r["mean_tau"] is not None
        and r["mean_tau"] >= h
        and r["est_seq_pairs"] >= args.target_pairs
    ]

    if candidates_ok:
        best = min(candidates_ok, key=lambda r: r["kappa"])
        print(f"\n  *** Recommended κ = {best['kappa']:.4f}  "
              f"(p{int(best['percentile'])}  "
              f"stop={best['stop_rate']*100:.1f}%  "
              f"mean_τ={best['mean_tau']:.2f}  "
              f"est_seq_pairs={best['est_seq_pairs']:,}) ***")
        print(f"\nUpdate the following files with κ = {best['kappa']:.4f}:")
        print(f"  configs/pm_navigation/cpl_estop.yaml    → kappa: {best['kappa']:.4f}")
        print(f"  configs/pm_navigation/cpl_seq_estop.yaml → kappa: {best['kappa']:.4f}")
        print(f"  slurm/pm_generate_feedback.sbatch        → KAPPA={best['kappa']:.4f}")
    else:
        print("\n  No κ satisfies both criteria.")
        print("  Options:")
        print("    • Increase pool size (more non-expert segments)")
        print(f"    • Decrease --horizon (currently h={h})")
        print("    • Lower --target-pairs")
        print("    • Try explicit κ sweep with --percentiles 91 93 95 97 99")


if __name__ == "__main__":
    main()
