"""
Calibrate the holding-model E-stop threshold c on a subset of the trajectory
pool, following the ARIC E-stop spec's threshold-tuning guide.

For each segment, evaluates Delta_t = S(H_t) - S(C_t) at EVERY eligible
candidate t (no early stopping -- unlike generate_estop_hold_labels.py, which
stops at the first crossing to save compute). This is what lets the same
calibration run answer "what intervention rate would threshold c produce?"
for many candidate c values without re-simulating anything, per the spec:
"During calibration, do not stop computation at the first crossing: you need
the whole sequence to evaluate multiple thresholds without rerunning
simulations."

Reports, for a grid of candidate thresholds (c=0, and quantile-targeted values
at 75%/50%/25% of the maximum achievable rate p_max):
  - intervention rate p_stop(c) = fraction of segments with M_i > c
  - stop-time distribution (mean/median tau, tau/L, remaining horizon)
  - oracle gap at the stop

p_max = fraction of segments with M_i > 0 is the ceiling: with a nonnegative
threshold, a segment where holding is never strictly better than continuing
can never produce a stop (spec: "do not use a negative threshold to force a
desired feedback count").

Usage:
    python scripts/tune_estop_hold_threshold.py \\
        --pool-path  datasets/mw/demo_pool/mw_drawer-open-v2/pool.npz \\
        --run-dir    runs/runs/chpc/oracle_sac_all/mw_drawer-open-v2 \\
        --n-segments 500 --n-workers 8 \\
        --output     datasets/mw/estop_hold_labels/mw_drawer-open-v2/calibration.npz
"""

import argparse
import functools
import os
import sys

import numpy as np

# See generate_estop_hold_labels.py's identical bootstrap for why this is needed:
# bare `python3 scripts/tune_estop_hold_threshold.py` does not put the repo root
# on sys.path, so `import scripts.X` would otherwise only work under pytest/-m.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scripts.estop_hold_common as ehc  # noqa: E402


def threshold_for_rate(max_gaps, target_rate):
    """Spec's quantile-based threshold selection: c_p = max(0, Q_{1-p}(M_i))."""
    threshold = max(0.0, float(np.quantile(max_gaps, 1.0 - target_rate)))
    actual_rate = float(np.mean(max_gaps > threshold))
    return threshold, actual_rate


def first_stop(times, gaps, threshold):
    """First t with Delta_t > threshold, or None (censored under this c)."""
    crossings = [t for t, d in zip(times, gaps) if d is not None and d > threshold]
    return crossings[0] if crossings else None


def main():
    parser = argparse.ArgumentParser(description="Calibrate the holding-model E-stop threshold c.")
    parser.add_argument("--pool-path", type=str, required=True)
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--oracle-checkpoint", type=str, default="best_model.pt")
    parser.add_argument("--min-horizon", type=int, default=5)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--mcmc-samples", type=int, default=32)
    parser.add_argument("--n-workers", type=int, default=1)
    parser.add_argument("--n-segments", type=int, default=500,
                        help="Calibration subset size, randomly sampled from the pool "
                             "(default: 500 -- calibration never stops early, so this is "
                             "roughly as expensive per segment as a full worst-case generation run).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=str, default=None,
                        help="Optional path to save per-segment (pool_index, times, gaps) "
                             "for later re-analysis without re-simulating.")
    args = parser.parse_args()

    with open(args.pool_path, "rb") as f:
        pool = np.load(f)
        pool_obs = pool["obs"]
        pool_action = pool["action"]
        pool_reward = pool["reward"]
        pool_state = pool["state"]

    N, T, _ = pool_obs.shape
    rng = np.random.default_rng(args.seed)
    n = min(args.n_segments, N)
    indices = np.sort(rng.choice(N, size=n, replace=False))
    print(f"Calibrating on {n}/{N} segments (T={T}, min_horizon={args.min_horizon})")

    tasks = ((int(i), pool_obs[i], pool_action[i], pool_reward[i], pool_state[i]) for i in indices)

    make_oracle_env_fn = functools.partial(ehc.make_oracle_env, args.run_dir, args.oracle_checkpoint)
    all_times, all_gaps, all_pool_idx = [], [], []
    for k, (i, result) in enumerate(ehc.run_parallel(
        tasks, args.n_workers, make_oracle_env_fn, args.discount,
        args.mcmc_samples, args.min_horizon, threshold=0.0, stop_early=False,
    )):
        times = [t for t, _ in result["gaps"]]
        gaps = [d for _, d in result["gaps"]]
        all_times.append(times)
        all_gaps.append(gaps)
        all_pool_idx.append(i)
        if (k + 1) % 50 == 0 or (k + 1) == n:
            print(f"  [{k + 1:>5}/{n}] evaluated")

    max_gaps = np.array([ehc.max_gap(list(zip(t, g))) for t, g in zip(all_times, all_gaps)])
    p_max = float(np.mean(max_gaps > 0.0))
    print(f"\nM_i statistics: mean={max_gaps.mean():.3f}  median={np.median(max_gaps):.3f}  "
          f"p_max (fraction with M_i > 0) = {p_max:.3f}")

    if args.output:
        ehc.save_npz(
            args.output,
            pool_index=np.array(all_pool_idx, dtype=np.int32),
            max_gap=max_gaps.astype(np.float32),
            # Ragged per-segment (times, gaps) saved as object arrays for later re-analysis.
            times=np.array(all_times, dtype=object),
            gaps=np.array(all_gaps, dtype=object),
        )
        print(f"Saved calibration data -> {args.output}")

    print(f"\n{'c':>10}  {'stop_rate':>10}  {'mean_tau':>10}  {'mean_tau/L':>11}  {'mean_horizon':>13}  {'mean_gap@stop':>14}")

    candidate_cs = [0.0]
    for frac in (0.75, 0.50, 0.25):
        c, _ = threshold_for_rate(max_gaps, frac * p_max)
        candidate_cs.append(c)

    for c in candidate_cs:
        taus, gaps_at_stop = [], []
        for times, gaps in zip(all_times, all_gaps):
            tau = first_stop(times, gaps, c)
            if tau is not None:
                taus.append(tau)
                gaps_at_stop.append(gaps[times.index(tau)])
        stop_rate = len(taus) / n
        if taus:
            taus_arr = np.array(taus, dtype=np.float64)
            mean_tau_frac = float(np.mean(taus_arr / T))
            mean_horizon = float(np.mean(T - taus_arr))
            mean_gap = float(np.mean(gaps_at_stop))
            print(f"{c:>10.3f}  {stop_rate:>10.1%}  {taus_arr.mean():>10.1f}  "
                  f"{mean_tau_frac:>11.2f}  {mean_horizon:>13.1f}  {mean_gap:>14.3f}")
        else:
            print(f"{c:>10.3f}  {stop_rate:>10.1%}  {'--':>10}  {'--':>11}  {'--':>13}  {'--':>14}")

    print(f"\np_max = {p_max:.3f} -- with a nonnegative threshold, at most this fraction of "
          f"segments can ever produce a stop. Freeze a threshold from the table above before "
          f"generating evaluation feedback (spec: calibrate on training data only).")


if __name__ == "__main__":
    main()
