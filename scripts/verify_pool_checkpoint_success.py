"""
Verify that every checkpoint actually used to build a trajectory pool (pool.npz's
checkpoint_step field) falls within an expected eval/success range.

This does not regenerate or modify anything — it's a read-only sanity check for
the "each pool segment must come from a checkpoint with success in
[--success-min, --success-max]" requirement before that pool's segments get used
as the "original" counterfactual candidate in generate_demo_labels.py.

Usage:
    python scripts/verify_pool_checkpoint_success.py \\
        --oracle-runs-base /scratch/general/vast/u1472210/oracle_sac_seeds \\
        --pool-dir-base    /scratch/general/vast/u1472210/mw_de_pool \\
        --envs mw_button-press-v2 mw_door-open-v2 mw_drawer-open-v2 mw_plate-slide-v2 \\
        --success-min 0.5 --success-max 0.6
"""

import argparse
import os

import numpy as np

from select_checkpoints_by_success import existing_checkpoint_steps, load_success_rows


def main():
    parser = argparse.ArgumentParser(
        description="Check pool.npz checkpoint_step values against logged eval/success.",
    )
    parser.add_argument("--oracle-runs-base", type=str, required=True,
                        help="Base dir with one subdir per env, e.g. "
                             "/scratch/general/vast/u1472210/oracle_sac_seeds")
    parser.add_argument("--pool-dir-base", type=str, required=True,
                        help="Base dir with one subdir per env, each containing pool.npz, "
                             "e.g. /scratch/general/vast/u1472210/mw_de_pool")
    parser.add_argument("--oracle-seed", type=str, default="seed-1",
                        help="Seed subdir under --oracle-runs-base/<env>/ (default: seed-1)")
    parser.add_argument("--envs", nargs="+", required=True)
    parser.add_argument("--checkpoint-interval", type=int, default=20000)
    parser.add_argument("--success-min", type=float, default=0.5)
    parser.add_argument("--success-max", type=float, default=0.6)
    args = parser.parse_args()

    any_violation = False

    for env in args.envs:
        run_dir  = os.path.join(args.oracle_runs_base, env, args.oracle_seed)
        pool_path = os.path.join(args.pool_dir_base, env, "pool.npz")
        print(f"{'='*65}\n{env}\n{'='*65}")

        if not os.path.isfile(pool_path):
            print(f"  ERROR: pool.npz not found: {pool_path}")
            continue

        try:
            rows = load_success_rows(run_dir)
        except (FileNotFoundError, KeyError) as e:
            print(f"  ERROR: {e}")
            continue

        success_by_step = dict(rows)
        ckpt_steps_on_disk = existing_checkpoint_steps(run_dir)

        with open(pool_path, "rb") as f:
            pool_ckpt = np.load(f)["checkpoint_step"]

        unique_steps, counts = np.unique(pool_ckpt, return_counts=True)
        print(f"  {len(unique_steps)} unique checkpoint(s) used across {pool_ckpt.shape[0]:,} segments:\n")

        env_bad = False
        for step, count in zip(unique_steps, counts):
            step = int(step)
            on_disk = step in ckpt_steps_on_disk
            succ = success_by_step.get(step)
            in_range = succ is not None and args.success_min <= succ <= args.success_max
            flag = "" if in_range else "  <-- OUT OF RANGE"
            if not in_range:
                env_bad = True
                any_violation = True
            succ_str = f"{succ:.3f}" if succ is not None else "NO log.csv ROW"
            disk_str = "" if on_disk else "  [WARNING: no model_<step>.pt on disk]"
            print(f"    step={step:>8,}  segments={count:>6,}  success={succ_str}{flag}{disk_str}")

        if env_bad:
            print(f"\n  RESULT: FAIL -- some checkpoints outside "
                  f"[{args.success_min}, {args.success_max}]")
        else:
            print(f"\n  RESULT: OK -- all checkpoints within "
                  f"[{args.success_min}, {args.success_max}]")
        print()

    if any_violation:
        print("One or more envs have pool segments from out-of-range checkpoints. "
              "Rebuild the pool (mw_de_build_pool.sbatch) with corrected "
              "CHECKPOINT_STEPS before generating demo labels.")
    else:
        print("All checked envs: pool checkpoints are within the expected success range.")


if __name__ == "__main__":
    main()
