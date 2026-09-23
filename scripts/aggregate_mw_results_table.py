"""
Aggregate windowed-peak eval/success rate across BC and CPL (demo + other
feedback types) runs into one envs x methods results table.

Reads (per env, per seed):
  runs/mw_de_demo_bc/<env>/bc_s<seed>/log.csv            BC
  runs/mw_de_demo_cpl/<env>/demo_s<seed>/log.csv         CPL + Demo
  runs/mw_de_feedback_cpl/<env>/pref_s<seed>/log.csv     CPL + Pref
  runs/mw_de_feedback_cpl/<env>/corr_s<seed>/log.csv     CPL + Corr
  runs/mw_de_feedback_cpl/<env>/scalar_s<seed>/log.csv   CPL + Scalar
  runs/mw_de_feedback_cpl/<env>/credit_assignment_s<seed>/log.csv   CPL + Credit

seq_estop is intentionally excluded.

Per (method, env, seed): peak of an N-point rolling mean over eval/success
(default N=8) -- same windowed-peak-smoothing idea as
scripts/plot_mw_data_efficiency.py's smooth_peak, generalized to take an
exact window size instead of a half-width (which only produces odd windows).
If a run has fewer than N eval points, falls back to the max of what's
there (same fallback smooth_peak uses).

Aggregates across seeds (mean, min, max, n) and prints an envs x methods
table (mean peak success, with seed count flagged if < expected), plus
writes a CSV.

Usage:
    python scripts/aggregate_mw_results_table.py \\
        --envs mw_button-press-v2 mw_door-open-v2 mw_drawer-open-v2 mw_plate-slide-v2 \\
        --seeds 0 1 2 \\
        --window 8 \\
        --output results/mw_de_results_table.csv
"""

import argparse
import csv
import os
from collections import defaultdict

import numpy as np

SUCCESS_COL_CANDIDATES = (
    "eval/success", "eval/succ", "eval_success", "eval_succ", "success",
)


def load_success_series(log_path):
    """Return the eval/success column (NaN/empty rows dropped) from a log.csv."""
    if not os.path.exists(log_path):
        return None
    with open(log_path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        col = next((c for c in SUCCESS_COL_CANDIDATES if c in fieldnames), None)
        if col is None:
            return None
        vals = []
        for row in reader:
            raw = row.get(col, "")
            if raw in (None, ""):
                continue
            try:
                vals.append(float(raw))
            except ValueError:
                continue
    return np.array(vals, dtype=np.float64) if vals else None


def windowed_peak(values, window):
    """Peak of a `window`-point rolling mean; falls back to max() if too short."""
    if values is None or len(values) == 0:
        return None
    if len(values) < window:
        return float(np.max(values))
    kernel = np.ones(window) / window
    smoothed = np.convolve(values, kernel, mode="valid")
    return float(np.max(smoothed))


# method -> (runs_dir, run_name template with {seed})
METHODS = {
    "BC":     ("runs/mw_de_demo_bc",      "bc_s{seed}"),
    "Demo":   ("runs/mw_de_demo_cpl",     "demo_s{seed}"),
    "Pref":   ("runs/mw_de_feedback_cpl", "pref_s{seed}"),
    "Corr":   ("runs/mw_de_feedback_cpl", "corr_s{seed}"),
    "Scalar": ("runs/mw_de_feedback_cpl", "scalar_s{seed}"),
    "Credit": ("runs/mw_de_feedback_cpl", "credit_assignment_s{seed}"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--envs", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--window", type=int, default=8,
                        help="Rolling-mean window size for peak smoothing (default: 8).")
    parser.add_argument("--repo-root", type=str, default=".")
    parser.add_argument("--output", type=str, default="results/mw_de_results_table.csv")
    args = parser.parse_args()

    # method -> env -> list of (seed, peak)
    results = defaultdict(lambda: defaultdict(list))

    print(f"Window size: {args.window}\n")

    for method, (runs_dir_rel, name_tmpl) in METHODS.items():
        runs_dir = os.path.join(args.repo_root, runs_dir_rel)
        for env in args.envs:
            for seed in args.seeds:
                run_name = name_tmpl.format(seed=seed)
                log_path = os.path.join(runs_dir, env, run_name, "log.csv")
                series = load_success_series(log_path)
                if series is None:
                    print(f"  [MISSING] {method:8s} {env:24s} seed={seed}  ({log_path})")
                    continue
                peak = windowed_peak(series, args.window)
                results[method][env].append((seed, peak))
                print(f"  {method:8s} {env:24s} seed={seed}  "
                      f"n_eval_pts={len(series):4d}  peak={peak:.3f}")

    print()

    # ------------------------------------------------------------------
    # Build envs x methods table (mean peak success across seeds)
    # ------------------------------------------------------------------
    method_names = list(METHODS.keys())
    table_rows = []
    for env in args.envs:
        row = {"env": env}
        for method in method_names:
            vals = [p for _, p in results[method][env]]
            n = len(vals)
            expected = len(args.seeds)
            if n == 0:
                row[method] = ""
                row[f"{method}_n"] = 0
            else:
                mean = float(np.mean(vals))
                lo, hi = float(np.min(vals)), float(np.max(vals))
                flag = "" if n == expected else f" [n={n}/{expected}]"
                row[method] = f"{mean:.3f} ({lo:.3f}-{hi:.3f}){flag}"
                row[f"{method}_n"] = n
        table_rows.append(row)

    # ------------------------------------------------------------------
    # Print
    # ------------------------------------------------------------------
    col_w = max(28, max(len(m) for m in method_names) + 12)
    header = f"{'env':26s}" + "".join(f"{m:>{col_w}s}" for m in method_names)
    print(header)
    print("-" * len(header))
    for row in table_rows:
        line = f"{row['env']:26s}"
        for m in method_names:
            line += f"{row[m]:>{col_w}s}"
        print(line)

    # ------------------------------------------------------------------
    # Save CSV
    # ------------------------------------------------------------------
    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["env"] + method_names)
        writer.writeheader()
        for row in table_rows:
            writer.writerow({"env": row["env"], **{m: row[m] for m in method_names}})

    print(f"\nSaved -> {args.output}")


if __name__ == "__main__":
    main()
