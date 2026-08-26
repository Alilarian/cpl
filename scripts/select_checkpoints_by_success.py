"""
Select checkpoint steps whose logged eval success rate falls within a target
range, for use with build_trajectory_pool.py --checkpoint-steps.

Reads <run_dir>/log.csv (written by research/utils/logger.py during training,
one row per eval — every trainer_kwargs.eval_freq steps) and keeps rows whose
step is a multiple of --checkpoint-interval (i.e. steps that actually have a
saved model_<step>.pt, since trainer_kwargs.checkpoint_freq may differ from
eval_freq) with eval/success in [--success-min, --success-max]. If more than
--n-checkpoints qualify, picks --n-checkpoints spread evenly across the
qualifying steps (not just the first N) so the pool spans the full range
rather than clustering. If none qualify, prints the closest available
checkpoints so you can widen the range.

Note: log.csv is rewritten (not appended) the first time a new metric key
shows up in a given training process, which can happen once around the first
eval step. This only affects rows from that same process before the reset —
if a run's row count looks implausibly small relative to
total_steps/eval_freq, cross-check against the wandb dashboard.

Usage:
    python scripts/select_checkpoints_by_success.py \\
        --oracle-runs-base /scratch/general/vast/u1472210/oracle_sac_1m \\
        --envs mw_button-press-v2 mw_door-open-v2 mw_drawer-open-v2 mw_plate-slide-v2 \\
        --success-min 0.3 --success-max 0.5 --n-checkpoints 5
"""

import argparse
import csv
import os


def load_success_rows(run_dir, step_col="step", success_col="eval/success"):
    csv_path = os.path.join(run_dir, "log.csv")
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"No log.csv in {run_dir}")
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        if success_col not in (reader.fieldnames or []):
            raise KeyError(f"Column '{success_col}' not found in {csv_path}")
        for row in reader:
            step_raw = row.get(step_col, "")
            succ_raw = row.get(success_col, "")
            if step_raw == "" or succ_raw == "":
                continue
            try:
                rows.append((int(float(step_raw)), float(succ_raw)))
            except ValueError:
                continue
    return sorted(rows, key=lambda x: x[0])


def existing_checkpoint_steps(run_dir):
    steps = set()
    for fname in os.listdir(run_dir):
        if fname.startswith("model_") and fname.endswith(".pt"):
            try:
                steps.add(int(fname[len("model_"):-len(".pt")]))
            except ValueError:
                pass
    return steps


def pick_evenly_spaced(candidates, n):
    """candidates: sorted list of (step, success). Pick up to n spread across it."""
    if len(candidates) <= n:
        return candidates
    idx = [round(i * (len(candidates) - 1) / (n - 1)) for i in range(n)]
    seen = set()
    picked = []
    for i in idx:
        if i not in seen:
            seen.add(i)
            picked.append(candidates[i])
    return picked


def main():
    parser = argparse.ArgumentParser(
        description="Select checkpoints by logged eval success rate.",
    )
    parser.add_argument("--oracle-runs-base", type=str, required=True,
                        help="Base dir with one subdir per env, e.g. "
                             "/scratch/general/vast/u1472210/oracle_sac_1m")
    parser.add_argument("--envs", nargs="+", required=True)
    parser.add_argument("--checkpoint-interval", type=int, default=20000,
                        help="Only consider steps that are multiples of this and that "
                             "have a saved model_<step>.pt (default: 20000, matching "
                             "trainer_kwargs.checkpoint_freq).")
    parser.add_argument("--success-min", type=float, default=0.3)
    parser.add_argument("--success-max", type=float, default=0.5)
    parser.add_argument("--n-checkpoints", type=int, default=5)
    args = parser.parse_args()

    for env in args.envs:
        run_dir = os.path.join(args.oracle_runs_base, env)
        print(f"{'='*65}\n{env}\n{'='*65}")

        try:
            rows = load_success_rows(run_dir)
        except (FileNotFoundError, KeyError) as e:
            print(f"  ERROR: {e}")
            continue

        ckpt_steps = existing_checkpoint_steps(run_dir)
        on_disk = [
            (step, succ) for step, succ in rows
            if step % args.checkpoint_interval == 0 and step in ckpt_steps
        ]

        candidates = [
            (step, succ) for step, succ in on_disk
            if args.success_min <= succ <= args.success_max
        ]

        if not candidates:
            print(f"  No checkpoints with success in [{args.success_min}, {args.success_max}].")
            nearby = sorted(
                on_disk,
                key=lambda x: min(abs(x[1] - args.success_min), abs(x[1] - args.success_max)),
            )[:5]
            if nearby:
                print("  Closest available checkpoints (step, success):")
                for step, succ in sorted(nearby):
                    print(f"    {step:>8,}  success={succ:.3f}")
            else:
                print(f"  No eval rows with a matching on-disk checkpoint were found at all "
                      f"(found {len(rows)} log.csv rows, {len(ckpt_steps)} checkpoint files).")
            continue

        picked = pick_evenly_spaced(candidates, args.n_checkpoints)

        print(f"  {len(candidates)} checkpoints in range, picked {len(picked)}:")
        for step, succ in picked:
            print(f"    {step:>8,}  success={succ:.3f}")

        steps_str = " ".join(str(s) for s, _ in picked)
        print(f'\n  CHECKPOINT_STEPS="{steps_str}"')

    print()


if __name__ == "__main__":
    main()
