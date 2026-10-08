"""
Phase 2 of the Cumulative E-stop model: calibrate the intervention threshold H.

For each pool segment, computes the full per-step deficit sequence d_t and its
cumulative sum C_t once (needs one frozen-oracle V(s) pass over the segment's
recorded states, plus one extra live-env restore+step to reconstruct the true
boundary state s_T -- see cum_estop_common.reconstruct_final_obs). Unlike the
holding-model E-stop, finding tau needs NO hold-controller simulation at all,
so this can run over far more segments per unit of compute.

Reports a stop-rate-vs-H table from the SAME computed cumulative trajectories
(no resimulation needed to try a different H), mirroring
scripts/tune_estop_hold_threshold.py's calibration-reuse design but with an
independent implementation.

Usage:
    python3 scripts/tune_cum_estop_threshold.py \\
        --pool-path datasets/mw_de_labels/mw_button-press-v2/pool.npz \\
        --run-dir   datasets/mw_de_labels/mw_button-press-v2/oracle \\
        --output    datasets/mw_de_labels/mw_button-press-v2/cum_estop_calibration.npz \\
        --n-samples 500
"""
import argparse
import functools
import os

import numpy as np

import scripts.cum_estop_common as cec


def _load_envs(run_dir, oracle_checkpoint, device):
    """Module-level (not a local closure) so functools.partial(...) over it
    stays picklable under multiprocessing's spawn context -- a bare nested
    closure isn't."""
    oracle, env = cec.load_policy(run_dir, os.path.join(run_dir, oracle_checkpoint), device)
    return {"oracle": oracle, "env": env}


def _process_task(task):
    i, obs_i, action_i, reward_i, state_i = task
    ctx = cec._worker_ctx
    final_obs, _, _ = cec.reconstruct_final_obs(ctx["env"], state_i[-1], action_i[-1])
    all_obs = np.concatenate([obs_i, final_obs[None]], axis=0)
    values = cec.oracle_values(all_obs, ctx["oracle"], ctx["mcmc_samples"], ctx["device"])
    deficits = cec.per_step_deficits(reward_i, values, ctx["gamma"])
    cumulative = np.cumsum(deficits)
    return i, cumulative


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-path", type=str, required=True)
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--oracle-checkpoint", type=str, default="best_model.pt")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--mcmc-samples", type=int, default=32)
    parser.add_argument("--n-workers", type=int, default=1)
    parser.add_argument("--n-samples", type=int, default=None, help="subsample this many pool segments")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--h-grid", type=str, default=None,
                         help="comma-separated H values to report; default: auto quantile grid")
    args = parser.parse_args()

    pool = np.load(args.pool_path)
    pool_obs, pool_action, pool_reward, pool_state = (
        pool["obs"], pool["action"], pool["reward"], pool["state"],
    )
    N = pool_obs.shape[0]
    if args.n_samples is not None and args.n_samples < N:
        idx = np.sort(np.random.default_rng(args.seed).choice(N, size=args.n_samples, replace=False))
    else:
        idx = np.arange(N)

    print(f"Calibrating on {len(idx)}/{N} segments (T={pool_obs.shape[1]})")

    tasks = [(int(i), pool_obs[i], pool_action[i], pool_reward[i], pool_state[i]) for i in idx]
    worker_kwargs = dict(gamma=args.gamma, mcmc_samples=args.mcmc_samples, device=args.device)
    make_envs_fn = functools.partial(_load_envs, args.run_dir, args.oracle_checkpoint, args.device)
    results = cec.run_parallel(tasks, args.n_workers, make_envs_fn, _process_task, worker_kwargs)

    T = pool_obs.shape[1]
    cumulative_all = np.zeros((len(results), T), dtype=np.float64)
    for i, cumulative in results:
        pos = list(idx).index(i)
        cumulative_all[pos, : len(cumulative)] = cumulative

    final_evidence = cumulative_all[:, -1]
    cec.save_npz(args.output, cumulative=cumulative_all.astype(np.float32), pool_index=idx.astype(np.int32))
    print(f"Saved calibration data -> {args.output}")

    print(f"\nFinal cumulative evidence: mean={final_evidence.mean():.3f} "
          f"median={np.median(final_evidence):.3f} max={final_evidence.max():.3f}")

    if args.h_grid:
        h_values = [float(x) for x in args.h_grid.split(",")]
    else:
        nonzero = final_evidence[final_evidence > 0]
        if len(nonzero) == 0:
            print("All final cumulative evidence is zero -- cannot calibrate a positive H. "
                  "Check the deficit scale / behavior pool.")
            return
        h_values = sorted(set(np.percentile(nonzero, [10, 25, 50, 75, 90]).round(4)))

    print(f"\n{'H':>10} {'stop_rate':>10} {'mean_tau':>10} {'mean_tau/T':>12} {'mean_horizon':>14}")
    for H in h_values:
        stop_mask = final_evidence >= H
        taus = []
        for row in cumulative_all[stop_mask]:
            crossed = np.nonzero(row >= H)[0]
            taus.append(int(crossed[0]) if len(crossed) else T)
        taus = np.array(taus)
        stop_rate = stop_mask.mean()
        mean_tau = taus.mean() if len(taus) else float("nan")
        print(f"{H:10.4f} {stop_rate*100:9.1f}% {mean_tau:10.1f} {mean_tau/T:12.3f} {T-mean_tau:14.1f}")


if __name__ == "__main__":
    main()
