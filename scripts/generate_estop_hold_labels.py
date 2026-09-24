"""
Generate "holding" E-stop feedback labels from a trajectory pool (Model A of
the ARIC E-stop spec): at every candidate stopping time t, restore the exact
MetaWorld simulator snapshot at s_t, roll out a fixed holding controller for
the remaining h_t = T - t steps, and compare its oracle Sum-estimator score
against the recorded continuation's. Stop at the first t where holding wins by
more than --threshold. At most one preference pair per segment (H_tau > C_tau),
plus a lightweight no-stop record when the threshold is never crossed.

This is a different, physically-simulated model from generate_estop_labels.py
/ generate_seq_estop_labels.py, which never roll out an actual holding branch
(they repeat the last recorded (s, a) and zero reward). See
scripts/estop_hold_common.py's module docstring for why that distinction
matters and what's reused from the rest of the pipeline.

Computation + parallelization
------------------------------
The dominant cost is per-segment: up to O(T^2) simulator steps in the worst
case (T - min_horizon + 1 candidates, each up to a full T-step rollout),
mitigated by stopping at the first threshold crossing. Segments are
independent (spec: "process each segment independently"), so this script
parallelizes on two levels, matching the pipeline's existing convention for
its other expensive per-segment rollout generators (generate_corr_labels.py /
generate_demo_labels.py):

  1. SLURM array sharding across machines: --num-shards/--shard-id split the
     pool into contiguous chunks, one process per shard. Merge with the
     existing scripts/merge_label_shards.py (fully generic over npz keys).
  2. Intra-shard multiprocessing across CPU cores: --n-workers spawns a pool
     of worker processes, each owning its own MetaWorld env instance *and* its
     own CPU copy of the frozen oracle (loaded once via a pool initializer,
     not per segment). Segments within a shard are then fanned out across
     workers via imap. CPU-only by design (each worker gets its own oracle
     copy; running many small MLP forward passes on the CPU is far cheaper
     than juggling N separate CUDA contexts for tiny per-candidate value
     queries) -- matches --device cpu already used for corr/demo generation.

Resumable: progress is checkpointed every --save-every processed (not just
kept) segments, since every segment yields either a stopped pair or a no-stop
record -- nothing is silently skipped the way corr/demo's early-termination
skips are.

Output (--output-dir/<env>/estop_hold_labels[_shard<i>of<n>].npz):
    obs             : (M, 2, T, obs_dim)   [0]=hold suffix H_tau, [1]=original suffix C_tau
    action          : (M, 2, T, act_dim)   zero-padded beyond horizon[m]
    reward          : (M, 2, T)            zero-padded beyond horizon[m]
    horizon         : (M,)  int32          real (unpadded) suffix length h_tau = T - tau
    stop_index      : (M,)  int32          tau
    oracle_gap      : (M,)  float32        Delta_tau (> threshold by construction)
    pool_index      : (M,)  int32          source index into pool.npz (provenance / leakage check)
    checkpoint_step : (M,)  int64
    threshold       : (1,)  float32        the --threshold this file was generated with

Companion no-stop file (--output-dir/<env>/estop_hold_nostop[_shard<i>of<n>].npz),
used for threshold calibration diagnostics (spec Section 12 -- "retain the
no-stop record", and the threshold-tuning guide's M_i statistic):
    pool_index      : (K,)  int32
    max_gap         : (K,)  float32        max observed Delta_t among evaluated t (partial:
                                            only reflects candidates actually evaluated before
                                            censoring -- for the EXACT M_i over every eligible
                                            t, use scripts/tune_estop_hold_threshold.py instead,
                                            which never stops early)
    checkpoint_step : (K,)  int64

Usage:
    python scripts/generate_estop_hold_labels.py \\
        --pool-path  datasets/mw/demo_pool/mw_drawer-open-v2/pool.npz \\
        --run-dir    runs/runs/chpc/oracle_sac_all/mw_drawer-open-v2 \\
        --threshold  1.0 --min-horizon 5 --n-workers 8 \\
        --output-dir datasets/mw/estop_hold_labels
"""

import argparse
import functools
import os
import sys

import numpy as np

# Bare `python3 scripts/generate_estop_hold_labels.py` (the invocation style
# every sbatch template in this repo uses) sets sys.path[0] to this file's own
# directory (scripts/), not the repo root -- so `import scripts.X` only
# resolves under `python -m` or pytest (which add the CWD instead). Insert the
# repo root explicitly so this script works under either invocation style.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scripts.estop_hold_common as ehc  # noqa: E402


def _pad(arr, T):
    """Zero-pad a (h, ...) array up to (T, ...) along axis 0."""
    if arr.shape[0] == T:
        return arr
    pad_shape = (T - arr.shape[0],) + arr.shape[1:]
    return np.concatenate([arr, np.zeros(pad_shape, dtype=arr.dtype)], axis=0)


def main():
    parser = argparse.ArgumentParser(
        description="Generate holding-model E-stop labels (real simulated holding rollouts)."
    )
    parser.add_argument("--pool-path", type=str, required=True)
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--oracle-checkpoint", type=str, default="best_model.pt")
    parser.add_argument("--threshold", type=float, required=True,
                        help="Fixed stopping threshold c >= 0 (Delta_t > c triggers a stop). "
                             "Calibrate first with scripts/tune_estop_hold_threshold.py.")
    parser.add_argument("--min-horizon", type=int, default=5,
                        help="Only evaluate t <= T - min_horizon (default: 5).")
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--mcmc-samples", type=int, default=32,
                        help="MCMC samples for V(s) estimation (default: 32). Called many "
                             "times per segment (once per candidate t plus the continuation "
                             "pass), so kept smaller than the 64 used by the one-shot pool "
                             "scorers in generate_pref_labels.py / generate_corr_labels.py.")
    parser.add_argument("--n-workers", type=int, default=1,
                        help="Worker processes for intra-shard parallelism across segments "
                             "(default: 1). Set to match --cpus-per-task in the SLURM template.")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=200,
                        help="Checkpoint progress every N processed segments (default: 200).")
    parser.add_argument("--max-segments", type=int, default=None,
                        help="Cap segments processed per shard (debug/testing).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str,
                        default="/scratch/general/vast/u1472210/estop_hold_labels")
    args = parser.parse_args()

    assert args.threshold >= 0.0, "spec requires a nonnegative threshold c >= 0"
    np.random.seed(args.seed)

    print("=" * 65)
    print("Holding-model E-stop label generation")
    print(f"Pool path      : {args.pool_path}")
    print(f"Oracle dir     : {args.run_dir}")
    print(f"Threshold c    : {args.threshold}")
    print(f"Min horizon    : {args.min_horizon}")
    print(f"Discount gamma : {args.discount}")
    print(f"MCMC samples   : {args.mcmc_samples}")
    print(f"n-workers      : {args.n_workers}")
    print("=" * 65)

    with open(args.pool_path, "rb") as f:
        pool = np.load(f)
        pool_obs = pool["obs"]              # (N, T, obs_dim)
        pool_action = pool["action"]        # (N, T, act_dim)
        pool_reward = pool["reward"]        # (N, T)
        pool_state = pool["state"]          # (N, T, state_dim)
        pool_ckpt = pool["checkpoint_step"] # (N,)

    N, T, obs_dim = pool_obs.shape
    act_dim = pool_action.shape[-1]
    print(f"N={N:,} segments, T={T}, obs_dim={obs_dim}, act_dim={act_dim}")

    assert 0 <= args.shard_id < args.num_shards
    if args.num_shards > 1:
        chunk = -(-N // args.num_shards)
        shard_start = args.shard_id * chunk
        shard_end = min(N, shard_start + chunk)
        print(f"Sharding: shard {args.shard_id}/{args.num_shards} covers [{shard_start}, {shard_end})")
    else:
        shard_start, shard_end = 0, N

    if args.max_segments is not None:
        shard_end = min(shard_end, shard_start + args.max_segments)

    env_name = os.path.basename(os.path.dirname(args.pool_path))
    out_dir = os.path.join(args.output_dir, env_name)
    os.makedirs(out_dir, exist_ok=True)
    shard_suffix = f"_shard{args.shard_id}of{args.num_shards}" if args.num_shards > 1 else ""
    out_path = os.path.join(out_dir, f"estop_hold_labels{shard_suffix}.npz")
    nostop_path = os.path.join(out_dir, f"estop_hold_nostop{shard_suffix}.npz")
    prog_path = os.path.join(out_dir, f"estop_hold_labels{shard_suffix}_progress.npz")

    if os.path.exists(out_path):
        print(f"\nOutput already exists: {out_path}\nDelete it to regenerate.")
        return

    out_obs, out_action, out_reward = [], [], []
    out_horizon, out_stop_idx, out_gap, out_pool_idx, out_ckpt = [], [], [], [], []
    ns_pool_idx, ns_max_gap, ns_ckpt = [], [], []
    start_idx = shard_start

    if os.path.exists(prog_path):
        print(f"Resuming from progress file: {prog_path}")
        prog = np.load(prog_path, allow_pickle=True)
        start_idx = int(prog["next_pool_idx"])
        if prog["obs"].shape[0] > 0:
            for row in prog["obs"]: out_obs.append(row)
            for row in prog["action"]: out_action.append(row)
            for row in prog["reward"]: out_reward.append(row)
            for v in prog["horizon"]: out_horizon.append(int(v))
            for v in prog["stop_index"]: out_stop_idx.append(int(v))
            for v in prog["oracle_gap"]: out_gap.append(float(v))
            for v in prog["pool_index"]: out_pool_idx.append(int(v))
            for v in prog["checkpoint_step"]: out_ckpt.append(int(v))
            for v in prog["ns_pool_index"]: ns_pool_idx.append(int(v))
            for v in prog["ns_max_gap"]: ns_max_gap.append(float(v))
            for v in prog["ns_checkpoint_step"]: ns_ckpt.append(int(v))
        print(f"  Resumed at pool index {start_idx}, {len(out_obs)} pairs kept so far")

    tasks = (
        (i, pool_obs[i], pool_action[i], pool_reward[i], pool_state[i])
        for i in range(start_idx, shard_end)
    )

    def _handle_result(i, result):
        if result["stopped"]:
            out_obs.append(np.stack([_pad(result["positive"]["obs"], T), _pad(result["negative"]["obs"], T)]))
            out_action.append(np.stack([_pad(result["positive"]["action"], T), _pad(result["negative"]["action"], T)]))
            out_reward.append(np.stack([_pad(result["positive"]["reward"], T), _pad(result["negative"]["reward"], T)]))
            out_horizon.append(result["horizon"])
            out_stop_idx.append(result["stop_index"])
            out_gap.append(result["oracle_gap"])
            out_pool_idx.append(i)
            out_ckpt.append(int(pool_ckpt[i]))
        else:
            ns_pool_idx.append(i)
            ns_max_gap.append(ehc.max_gap(result["gaps"]))
            ns_ckpt.append(int(pool_ckpt[i]))

    def _save_progress(next_idx):
        ehc.save_npz(
            prog_path,
            obs=np.stack(out_obs, axis=0) if out_obs else np.zeros((0, 2, T, obs_dim), dtype=np.float32),
            action=np.stack(out_action, axis=0) if out_action else np.zeros((0, 2, T, act_dim), dtype=np.float32),
            reward=np.stack(out_reward, axis=0) if out_reward else np.zeros((0, 2, T), dtype=np.float32),
            horizon=np.array(out_horizon, dtype=np.int32),
            stop_index=np.array(out_stop_idx, dtype=np.int32),
            oracle_gap=np.array(out_gap, dtype=np.float32),
            pool_index=np.array(out_pool_idx, dtype=np.int32),
            checkpoint_step=np.array(out_ckpt, dtype=np.int64),
            ns_pool_index=np.array(ns_pool_idx, dtype=np.int32),
            ns_max_gap=np.array(ns_max_gap, dtype=np.float32),
            ns_checkpoint_step=np.array(ns_ckpt, dtype=np.int64),
            next_pool_idx=np.array(next_idx),
        )

    n_processed = 0
    n_total = shard_end - start_idx
    print(f"\nProcessing {n_total} segments with {args.n_workers} worker(s)...\n")

    make_oracle_env_fn = functools.partial(ehc.make_oracle_env, args.run_dir, args.oracle_checkpoint)
    for i, result in ehc.run_parallel(
        tasks, args.n_workers, make_oracle_env_fn, args.discount,
        args.mcmc_samples, args.min_horizon, args.threshold, stop_early=True,
    ):
        _handle_result(i, result)
        n_processed += 1
        if n_processed % 20 == 0 or n_processed == n_total:
            print(f"  [{n_processed:>5}/{n_total}]  stopped={len(out_obs)}  censored={len(ns_pool_idx)}")
        if n_processed % args.save_every == 0:
            _save_progress(i + 1)
            print(f"  [progress saved at pool idx {i + 1}]")

    M = len(out_obs)
    K = len(ns_pool_idx)
    print(f"\n{'=' * 55}")
    print(f"Segments processed : {n_total}")
    print(f"Stopped (pairs)    : {M}  ({100 * M / max(1, n_total):.1f}%)")
    print(f"Censored (no-stop) : {K}  ({100 * K / max(1, n_total):.1f}%)")

    if M > 0:
        gaps_arr = np.array(out_gap, dtype=np.float32)
        horizons_arr = np.array(out_horizon, dtype=np.int32)
        stop_arr = np.array(out_stop_idx, dtype=np.int32)
        print(f"Oracle gap at stop : mean={gaps_arr.mean():.3f}  min={gaps_arr.min():.3f}  max={gaps_arr.max():.3f}")
        print(f"Stop index tau     : mean={stop_arr.mean():.1f}  median={np.median(stop_arr):.1f}")
        print(f"Suffix horizon     : mean={horizons_arr.mean():.1f}  min={horizons_arr.min()}  max={horizons_arr.max()}")

        ehc.save_npz(
            out_path,
            obs=np.stack(out_obs, axis=0),
            action=np.stack(out_action, axis=0),
            reward=np.stack(out_reward, axis=0),
            horizon=np.array(out_horizon, dtype=np.int32),
            stop_index=np.array(out_stop_idx, dtype=np.int32),
            oracle_gap=np.array(out_gap, dtype=np.float32),
            pool_index=np.array(out_pool_idx, dtype=np.int32),
            checkpoint_step=np.array(out_ckpt, dtype=np.int64),
            threshold=np.array([args.threshold], dtype=np.float32),
        )
        print(f"\nSaved -> {out_path}")
    else:
        print("\nWARNING: 0 pairs generated. Lower --threshold (see "
              "scripts/tune_estop_hold_threshold.py) or check pool/oracle.")

    if K > 0:
        ehc.save_npz(
            nostop_path,
            pool_index=np.array(ns_pool_idx, dtype=np.int32),
            max_gap=np.array(ns_max_gap, dtype=np.float32),
            checkpoint_step=np.array(ns_ckpt, dtype=np.int64),
        )
        print(f"Saved -> {nostop_path}")

    if os.path.exists(prog_path):
        os.remove(prog_path)
        print("Progress file removed.")


if __name__ == "__main__":
    main()
