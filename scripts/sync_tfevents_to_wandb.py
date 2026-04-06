"""Sync the most recent tfevents file per seed directory to W&B.

For each (task, seed) directory under cpl_state_dense_all and
cpl_state_sparse_all, picks the tfevents file with the largest Unix
timestamp in its filename (i.e. the most recent run) and uploads it via
wandb.tensorboard.

Each algo gets its own W&B project:
  {project}-cpl-state-dense
  {project}-cpl-state-sparse

Within each project, runs are grouped by task.

Usage
-----
python scripts/sync_tfevents_to_wandb.py \\
    --root runs/runs/chpc \\
    --project cpl-metaworld \\
    --entity alilarian23
"""

import argparse
import os
import re

import wandb


def tfevents_timestamp(filename: str) -> int:
    m = re.search(r"events\.out\.tfevents\.(\d+)", filename)
    return int(m.group(1)) if m else 0


def latest_tfevents(directory: str):
    candidates = [f for f in os.listdir(directory) if re.match(r"events\.out\.tfevents\.", f)]
    if not candidates:
        return None
    return os.path.join(directory, max(candidates, key=tfevents_timestamp))


def find_runs(root: str) -> list:
    """Return list of dicts: algo, task, seed, run_dir, tfevents."""
    runs = []
    for algo_dir in ["cpl_state_dense_all", "cpl_state_sparse_all"]:
        algo_path = os.path.join(root, algo_dir)
        if not os.path.isdir(algo_path):
            print(f"[skip] {algo_path} not found")
            continue

        algo_label = algo_dir[:-4]  # strip _all

        for task in sorted(os.listdir(algo_path)):
            task_path = os.path.join(algo_path, task)
            if not os.path.isdir(task_path):
                continue
            for path_dir in sorted(os.listdir(task_path)):
                path_full = os.path.join(task_path, path_dir)
                if not os.path.isdir(path_full):
                    continue
                for seed_dir in sorted(os.listdir(path_full)):
                    seed_full = os.path.join(path_full, seed_dir)
                    if not (os.path.isdir(seed_full) and seed_dir.startswith("seed-")):
                        continue
                    tf = latest_tfevents(seed_full)
                    if tf:
                        seed = seed_dir[len("seed-"):]
                        runs.append(dict(algo=algo_label, task=task, seed=seed, run_dir=seed_full, tfevents=tf))

    return runs


def upload_run(run: dict, project: str, entity, mode: str) -> None:
    algo, task, seed = run["algo"], run["task"], run["seed"]
    run_dir = run["tfevents_dir"] = run["run_dir"]

    algo_project = f"{project}-{algo.replace('_', '-')}"
    run_name = f"{task}__seed{seed}"

    print(f"  syncing: {algo_project} / {task} / seed{seed}  [{os.path.basename(run['tfevents'])}]")

    wandb.tensorboard.unpatch()
    wandb.tensorboard.patch(root_logdir=run_dir)
    wb_run = wandb.init(
        project=algo_project,
        entity=entity,
        name=run_name,
        group=task,
        job_type=task,
        tags=[algo, task],
        config=dict(algo=algo, task=task, seed=seed),
        sync_tensorboard=True,
        reinit="finish_previous",
        mode=mode,
    )
    wb_run.finish()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--project", type=str, required=True)
    parser.add_argument("--entity", type=str, default=None)
    parser.add_argument("--mode", type=str, default="online",
                        choices=["online", "offline", "disabled"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs = find_runs(args.root)

    if not runs:
        print("No runs found.")
        return

    print(f"Found {len(runs)} runs under {args.root}")
    for run in runs:
        upload_run(run, project=args.project, entity=args.entity, mode=args.mode)

    print("Done.")


if __name__ == "__main__":
    main()
