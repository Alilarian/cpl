import argparse
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True, help="Sweep root path.")
    parser.add_argument("--project", type=str, required=True, help="Weights & Biases project name.")
    parser.add_argument("--entity", type=str, default=None, help="Optional Weights & Biases entity.")
    parser.add_argument("--run-name", type=str, default=None, help="Optional Weights & Biases run name.")
    parser.add_argument("--group-key", type=str, default="contrastive_bias", help="Sweep hyperparameter name.")
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["eval/reward", "eval/success"],
        help="Metrics to aggregate over seeds.",
    )
    parser.add_argument("--x-key", type=str, default="step", help="X-axis key in log.csv.")
    parser.add_argument("--seed-prefix", type=str, default="seed-", help="Seed directory prefix.")
    parser.add_argument(
        "--wandb-mode",
        type=str,
        default="online",
        choices=["online", "offline", "disabled"],
        help="Weights & Biases mode.",
    )
    parser.add_argument(
        "--summary-stat",
        type=str,
        default="last",
        choices=["last", "best"],
        help="How to summarize each seed for the final summary table.",
    )
    parser.add_argument(
        "--maximize-metrics",
        nargs="+",
        default=["eval/reward", "eval/success"],
        help="Metrics where larger is better when --summary-stat=best.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional output directory for aggregated CSV files. Defaults to the sweep path.",
    )
    return parser.parse_args()


def safe_metric_name(metric: str) -> str:
    return metric.replace("/", "_")


def discover_group_dirs(path: str) -> List[str]:
    group_dirs = []
    for entry in sorted(os.listdir(path)):
        full_path = os.path.join(path, entry)
        if os.path.isdir(full_path):
            group_dirs.append(full_path)
    if len(group_dirs) == 0:
        raise ValueError(f"No group directories found under {path}")
    return group_dirs


def discover_seed_logs(group_dir: str, seed_prefix: str) -> List[Tuple[str, str]]:
    seed_logs = []
    for entry in sorted(os.listdir(group_dir)):
        seed_dir = os.path.join(group_dir, entry)
        log_path = os.path.join(seed_dir, "log.csv")
        if os.path.isdir(seed_dir) and entry.startswith(seed_prefix) and os.path.exists(log_path):
            seed_logs.append((entry, log_path))
    return seed_logs


def parse_group_value(group_name: str, group_key: str) -> str:
    prefix = group_key + "-"
    if group_name.startswith(prefix):
        return group_name[len(prefix) :]
    return group_name


def aggregate_group(
    group_dir: str,
    group_key: str,
    metrics: List[str],
    x_key: str,
    seed_prefix: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    group_name = os.path.basename(group_dir.rstrip("/"))
    group_value = parse_group_value(group_name, group_key)
    seed_logs = discover_seed_logs(group_dir, seed_prefix)
    if len(seed_logs) == 0:
        raise ValueError(f"No seed log.csv files found under {group_dir}")

    per_seed_rows = []
    aggregated_frames = []
    for seed_name, log_path in seed_logs:
        df = pd.read_csv(log_path)
        missing_cols = [col for col in [x_key, *metrics] if col not in df.columns]
        if missing_cols:
            raise ValueError(f"{log_path} is missing columns: {missing_cols}")

        seed_df = df[[x_key, *metrics]].copy()
        seed_df["seed"] = seed_name
        seed_df[group_key] = group_value
        per_seed_rows.append(seed_df)

    seed_history = pd.concat(per_seed_rows, ignore_index=True)

    for metric in metrics:
        metric_df = seed_history[[x_key, "seed", group_key, metric]].dropna(subset=[metric]).copy()
        grouped = metric_df.groupby([group_key, x_key])[metric]
        agg_df = grouped.agg(["mean", "std", "min", "max", "count"]).reset_index()
        agg_df["metric"] = metric
        aggregated_frames.append(agg_df)

    aggregated = pd.concat(aggregated_frames, ignore_index=True)
    return seed_history, aggregated


def summarize_seeds(
    seed_history: pd.DataFrame,
    group_key: str,
    metrics: List[str],
    x_key: str,
    summary_stat: str,
    maximize_metrics: List[str],
) -> pd.DataFrame:
    rows = []
    for (group_value, seed_name), seed_df in seed_history.groupby([group_key, "seed"]):
        seed_df = seed_df.sort_values(x_key)
        row: Dict[str, object] = {
            group_key: group_value,
            "seed": seed_name,
            "last_step": int(seed_df[x_key].iloc[-1]),
        }
        for metric in metrics:
            metric_series = seed_df[[x_key, metric]].dropna(subset=[metric])
            if len(metric_series) == 0:
                row[metric] = np.nan
                continue
            if summary_stat == "last":
                row[metric] = float(metric_series.iloc[-1][metric])
            else:
                if metric in maximize_metrics:
                    idx = metric_series[metric].idxmax()
                else:
                    idx = metric_series[metric].idxmin()
                row[metric] = float(metric_series.loc[idx, metric])
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_groups(per_seed_summary: pd.DataFrame, group_key: str, metrics: List[str]) -> pd.DataFrame:
    rows = []
    for group_value, group_df in per_seed_summary.groupby(group_key):
        row: Dict[str, object] = {
            group_key: group_value,
            "num_seeds": int(len(group_df)),
        }
        for metric in metrics:
            values = group_df[metric].dropna()
            row[f"{metric}/mean"] = float(values.mean()) if len(values) > 0 else np.nan
            row[f"{metric}/std"] = float(values.std(ddof=0)) if len(values) > 0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_key)


def log_to_wandb(
    aggregated: pd.DataFrame,
    per_seed_summary: pd.DataFrame,
    group_summary: pd.DataFrame,
    group_key: str,
    metrics: List[str],
    x_key: str,
    args: argparse.Namespace,
) -> None:
    import wandb

    run = wandb.init(
        project=args.project,
        entity=args.entity,
        name=args.run_name,
        config={
            "path": args.path,
            "group_key": group_key,
            "metrics": metrics,
            "x_key": x_key,
            "summary_stat": args.summary_stat,
        },
        mode=args.wandb_mode,
    )

    if run is None:
        return

    aggregated = aggregated.sort_values([x_key, group_key, "metric"])
    all_steps = sorted(aggregated[x_key].unique())
    for step in all_steps:
        log_row: Dict[str, float] = {}
        step_df = aggregated[aggregated[x_key] == step]
        for _, row in step_df.iterrows():
            metric_name = safe_metric_name(row["metric"])
            group_value = row[group_key]
            prefix = f"aggregate/{metric_name}/{group_key}-{group_value}"
            log_row[f"{prefix}/mean"] = float(row["mean"])
            log_row[f"{prefix}/std"] = float(0.0 if pd.isna(row["std"]) else row["std"])
            log_row[f"{prefix}/min"] = float(row["min"])
            log_row[f"{prefix}/max"] = float(row["max"])
            log_row[f"{prefix}/count"] = int(row["count"])
        wandb.log(log_row, step=int(step))

    run.log(
        {
            "aggregate/history": wandb.Table(dataframe=aggregated),
            "aggregate/per_seed_summary": wandb.Table(dataframe=per_seed_summary),
            "aggregate/group_summary": wandb.Table(dataframe=group_summary),
        }
    )

    best_metric = metrics[0]
    best_metric_key = f"{best_metric}/mean"
    if best_metric_key in group_summary.columns:
        best_row = group_summary.loc[group_summary[best_metric_key].idxmax()]
        run.summary[f"best_{group_key}"] = best_row[group_key]
        run.summary[f"best_{safe_metric_name(best_metric)}"] = float(best_row[best_metric_key])

    for _, row in group_summary.iterrows():
        group_value = row[group_key]
        for metric in metrics:
            mean_key = f"{metric}/mean"
            std_key = f"{metric}/std"
            if mean_key in row:
                run.summary[f"summary/{group_key}-{group_value}/{safe_metric_name(metric)}_mean"] = float(row[mean_key])
            if std_key in row:
                run.summary[f"summary/{group_key}-{group_value}/{safe_metric_name(metric)}_std"] = float(row[std_key])

    run.finish()


def main() -> None:
    args = parse_args()
    output_dir = args.path if args.output_dir is None else args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    group_dirs = discover_group_dirs(args.path)

    seed_histories = []
    aggregated_frames = []
    for group_dir in group_dirs:
        seed_history, aggregated = aggregate_group(
            group_dir=group_dir,
            group_key=args.group_key,
            metrics=args.metrics,
            x_key=args.x_key,
            seed_prefix=args.seed_prefix,
        )
        seed_histories.append(seed_history)
        aggregated_frames.append(aggregated)

    seed_history = pd.concat(seed_histories, ignore_index=True)
    aggregated = pd.concat(aggregated_frames, ignore_index=True)
    per_seed_summary = summarize_seeds(
        seed_history=seed_history,
        group_key=args.group_key,
        metrics=args.metrics,
        x_key=args.x_key,
        summary_stat=args.summary_stat,
        maximize_metrics=args.maximize_metrics,
    )
    group_summary = summarize_groups(per_seed_summary=per_seed_summary, group_key=args.group_key, metrics=args.metrics)

    aggregated.to_csv(os.path.join(output_dir, "wandb_aggregated_history.csv"), index=False)
    per_seed_summary.to_csv(os.path.join(output_dir, "wandb_per_seed_summary.csv"), index=False)
    group_summary.to_csv(os.path.join(output_dir, "wandb_group_summary.csv"), index=False)

    print("[research] Wrote aggregated history to", os.path.join(output_dir, "wandb_aggregated_history.csv"))
    print("[research] Wrote per-seed summary to", os.path.join(output_dir, "wandb_per_seed_summary.csv"))
    print("[research] Wrote group summary to", os.path.join(output_dir, "wandb_group_summary.csv"))
    print(group_summary.to_string(index=False))

    log_to_wandb(
        aggregated=aggregated,
        per_seed_summary=per_seed_summary,
        group_summary=group_summary,
        group_key=args.group_key,
        metrics=args.metrics,
        x_key=args.x_key,
        args=args,
    )


if __name__ == "__main__":
    main()
