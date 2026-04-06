"""Upload per-run log.csv files to W&B.

Supports three directory layouts under a root path:

  oracle_sac_all/
    <task>/log.csv                                 (no seed subdirectory)

  cpl_state_dense_all/
    <task>/path-<short>/seed-<n>/log.csv

  cpl_state_sparse_all/
    <task>/path-<short>/seed-<n>/log.csv

W&B organisation
----------------
  group    = algo section  (oracle_sac / cpl_state_dense / cpl_state_sparse)
  job_type = task name     (mw_bin-picking-v2, …)
  name     = {task}__seed{n}

This creates three collapsible sections in the W&B sidebar, each with runs
categorised by task.  All numeric columns from log.csv are uploaded by default.

Usage examples
--------------
python scripts/upload_runs_to_wandb.py \\
    --root runs/runs/chpc \\
    --project my-project \\
    --entity alilarian23

# Only oracle_sac
python scripts/upload_runs_to_wandb.py \\
    --root runs/runs/chpc \\
    --project my-project \\
    --entity alilarian23 \\
    --algos oracle_sac_all
"""

import argparse
import os

import pandas as pd

# Columns that are timing / bookkeeping overhead — excluded from metric upload
_SKIP_PREFIXES = ("time/",)


def _metric_cols(df: pd.DataFrame, x_key: str) -> list[str]:
    """Return all numeric columns except x_key and timing columns."""
    cols = []
    for col in df.columns:
        if col == x_key:
            continue
        if any(col.startswith(p) for p in _SKIP_PREFIXES):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    return cols


# ---------------------------------------------------------------------------
# Directory discovery
# ---------------------------------------------------------------------------

def find_runs(root: str, algos: list) -> list:
    """Return a list of run dicts: algo, task, seed, log_path."""
    runs = []
    for algo_dir in sorted(os.listdir(root)):
        if algos and algo_dir not in algos:
            continue
        algo_path = os.path.join(root, algo_dir)
        if not os.path.isdir(algo_path):
            continue

        # Short label: strip trailing _all
        algo_label = algo_dir[:-4] if algo_dir.endswith("_all") else algo_dir

        for task in sorted(os.listdir(algo_path)):
            task_path = os.path.join(algo_path, task)
            if not os.path.isdir(task_path):
                continue

            # oracle_sac layout: task/log.csv
            direct_log = os.path.join(task_path, "log.csv")
            if os.path.exists(direct_log):
                runs.append(dict(algo=algo_label, task=task, seed="0", log_path=direct_log))
                continue

            # cpl layout: task/path-*/seed-*/log.csv
            for path_dir in sorted(os.listdir(task_path)):
                path_full = os.path.join(task_path, path_dir)
                if not os.path.isdir(path_full):
                    continue
                for seed_dir in sorted(os.listdir(path_full)):
                    seed_full = os.path.join(path_full, seed_dir)
                    log_path = os.path.join(seed_full, "log.csv")
                    if os.path.isdir(seed_full) and seed_dir.startswith("seed-") and os.path.exists(log_path):
                        seed = seed_dir[len("seed-"):]
                        runs.append(dict(algo=algo_label, task=task, seed=seed, log_path=log_path))

    return runs


# ---------------------------------------------------------------------------
# W&B upload
# ---------------------------------------------------------------------------

def upload_run(run: dict, project: str, entity, x_key: str, mode: str) -> None:
    import wandb

    algo, task, seed, log_path = run["algo"], run["task"], run["seed"], run["log_path"]

    df = pd.read_csv(log_path)
    if x_key not in df.columns:
        print(f"  [skip] {log_path} — missing x_key '{x_key}'")
        return

    metrics = _metric_cols(df, x_key)
    if not metrics:
        print(f"  [skip] {log_path} — no numeric columns found")
        return

    # Each algo gets its own project for clear separation
    algo_project = f"{project}-{algo.replace('_', '-')}"

    run_name = f"{task}__seed{seed}"
    print(f"  uploading: {algo_project} / {task} / seed{seed}  ({len(df)} rows, {len(metrics)} metrics)")

    wandb_run = wandb.init(
        project=algo_project,
        entity=entity,
        name=run_name,
        group=task,          # group by task within the algo project
        job_type=task,
        tags=[algo, task],
        config=dict(algo=algo, task=task, seed=seed),
        reinit=True,
        mode=mode,
    )

    cols = [x_key] + metrics
    for _, row in df[cols].dropna(subset=[x_key]).iterrows():
        log_dict = {k: v for k, v in row.items() if k != x_key and pd.notna(v)}
        wandb.log(log_dict, step=int(row[x_key]))

    wandb_run.finish()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload per-run log.csv files to W&B.")
    parser.add_argument("--root", type=str, required=True, help="Root directory containing algo subdirectories.")
    parser.add_argument("--project", type=str, required=True, help="W&B project name.")
    parser.add_argument("--entity", type=str, default=None, help="W&B entity (username or team).")
    parser.add_argument(
        "--algos",
        nargs="+",
        default=[],
        help="Algo directory names to include (default: all). E.g. oracle_sac_all cpl_state_dense_all",
    )
    parser.add_argument("--x-key", type=str, default="step", help="Column to use as the W&B step axis.")
    parser.add_argument(
        "--mode",
        type=str,
        default="online",
        choices=["online", "offline", "disabled"],
        help="W&B mode.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs = find_runs(args.root, args.algos)

    if not runs:
        print("No runs found. Check --root and --algos.")
        return

    print(f"Found {len(runs)} runs under {args.root}")
    for run in runs:
        upload_run(run, project=args.project, entity=args.entity, x_key=args.x_key, mode=args.mode)

    print("Done.")


if __name__ == "__main__":
    main()
