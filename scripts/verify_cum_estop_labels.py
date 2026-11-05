"""
Structural check of one env's merged cum_estop labels against its pool, run
by slurm/cum_estop_merge_labels.sbatch before any training starts:

  stop rows    : index 1 is the untouched pool row; index 0 shares its prefix
                 through obs[tau] (the correction starts from the true s_tau
                 obs) and diverges from action[tau] on. With --oracle, every
                 row has corrected_is_preferred = True.
  no-stop rows : index 0 is the accepted pool row (preferred); index 1 is a
                 counterfactual that starts from the true s_0 obs.

Exits non-zero on any failure -- e.g. labels written by a generator older
than the 2026-11-05 oracle-correction / no-stop-order fixes.

Usage:
    python3 scripts/verify_cum_estop_labels.py \\
        --labels-dir /scratch/.../cum_estop_labels_oracle/mw_door-open-v2 \\
        --pool-path  /scratch/.../mw_de_pool/mw_door-open-v2/pool.npz --oracle
"""
import argparse
import os
import sys

import numpy as np


def _sample(f, n_rows, rng):
    n = len(f["pool_index"])
    rows = np.sort(rng.choice(n, size=min(n_rows, n), replace=False))
    return n, rows, f["pool_index"][rows], f["obs"][rows], f["action"][rows]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels-dir", type=str, required=True)
    parser.add_argument("--pool-path", type=str, required=True)
    parser.add_argument("--oracle", action="store_true", help="expect oracle corrections (always preferred)")
    parser.add_argument("--n-rows", type=int, default=500, help="rows checked per branch")
    args = parser.parse_args()

    with np.load(args.pool_path) as pool:
        pool_obs, pool_action = pool["obs"], pool["action"]
    rng = np.random.default_rng(0)
    failures = []

    def report(name, ok):
        frac = float(np.mean(ok))
        print(f"  {name:48s} {frac:.3f}")
        if frac < 1.0:
            failures.append(name)

    with np.load(os.path.join(args.labels_dir, "stop_correction_labels.npz")) as f:
        n, rows, p, obs, action = _sample(f, args.n_rows, rng)
        tau = f["stop_index"][rows]
        print(f"stop_correction: {n} rows ({len(rows)} checked)")
        report("index 1 == pool row", [np.array_equal(obs[j, 1], pool_obs[p[j]])
                                        and np.array_equal(action[j, 1], pool_action[p[j]]) for j in range(len(rows))])
        report("index 0 obs[:tau+1] == pool (true s_tau obs)",
               [np.array_equal(obs[j, 0, :tau[j] + 1], pool_obs[p[j], :tau[j] + 1]) for j in range(len(rows))])
        report("index 0 action[:tau] == pool, diverges at tau",
               [np.array_equal(action[j, 0, :tau[j]], pool_action[p[j], :tau[j]])
                and not np.array_equal(action[j, 0, tau[j]], pool_action[p[j], tau[j]]) for j in range(len(rows))])
        if args.oracle:
            report("corrected_is_preferred (all rows)", f["corrected_is_preferred"])
            print(f"  {'oracle correction scored below original':48s} {np.mean(f['gap'] < 0):.3f}  (informational)")

    with np.load(os.path.join(args.labels_dir, "no_stop_demo_labels.npz")) as f:
        n, rows, p, obs, action = _sample(f, args.n_rows, rng)
        print(f"no_stop_demo: {n} rows ({len(rows)} checked)")
        report("index 0 == pool row (accepted original)", [np.array_equal(obs[j, 0], pool_obs[p[j]])
                                                           and np.array_equal(action[j, 0], pool_action[p[j]])
                                                           for j in range(len(rows))])
        report("index 1 obs[0] == pool obs[0] (true s_0 obs)",
               [np.array_equal(obs[j, 1, 0], pool_obs[p[j], 0]) for j in range(len(rows))])

    if failures:
        print(f"FAILED: {failures}")
        sys.exit(1)
    print("OK")


if __name__ == "__main__":
    main()
