"""
Plot MetaWorld CPL b10k results from local log.csv files.

Reads runs/mw_b10k/<env>/<type>/<alg>_s<seed>/log.csv (the layout written by
slurm/mw_b10k_cpl.sbatch and slurm/mw_b10k_piql.sbatch: one fixed 10k-sample
budget per feedback type, 3 policy-init seeds, no budget sweep), and for each
requested metric plots per-env curves vs. training step (one line per
feedback type, mean + min/max shading across seeds), plus a peak-success
summary bar chart.

Usage
-----
python scripts/plot_mw_b10k.py
python scripts/plot_mw_b10k.py --runs-dir runs/mw_b10k --alg cpl --out results/mw_b10k
python scripts/plot_mw_b10k.py --metrics eval/success,eval/reward,train/accuracy,train/demo_loss
"""

import argparse
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# demo is excluded from the mw_b10k matrix for now (see slurm/mw_b10k_cpl.sbatch)
TYPES = ["pref", "corr", "seq_estop", "scalar", "credit_assignment"]

DEFAULT_METRICS = ["eval/success", "eval/reward", "train/accuracy", "train/demo_loss"]

COLORS = {
    "pref":              "#0072B2",
    "corr":              "#E69F00",
    "seq_estop":         "#D55E00",
    "scalar":            "#CC79A7",
    "credit_assignment": "#56B4E9",
}
LABELS = {
    "pref":              "Pref",
    "corr":              "Corr",
    "seq_estop":         "Seq E-stop",
    "scalar":            "Scalar",
    "credit_assignment": "Credit",
}


def parse_run_name(name, alg):
    """<alg>_s<seed>  →  seed, or None if it doesn't match the requested alg."""
    m = re.fullmatch(rf"{re.escape(alg)}_s(\d+)", name)
    if m is None:
        return None
    return int(m.group(1))


def _colkey(name):
    """numpy genfromtxt(names=True) mangles headers like 'eval/success' -> 'evalsuccess'."""
    return re.sub(r"[^0-9a-zA-Z_]", "", name)


def load_env_runs(env_dir, alg):
    """Returns {(type, seed): structured ndarray} for one env's run dir."""
    data = {}
    for fb_type in TYPES:
        type_dir = os.path.join(env_dir, fb_type)
        if not os.path.isdir(type_dir):
            continue
        for name in sorted(os.listdir(type_dir)):
            seed = parse_run_name(name, alg)
            if seed is None:
                continue
            log_path = os.path.join(type_dir, name, "log.csv")
            if not os.path.exists(log_path) or os.path.getsize(log_path) == 0:
                print(f"  Missing/empty log: {fb_type}/{name}")
                continue

            try:
                rows = np.atleast_1d(np.genfromtxt(log_path, delimiter=",", names=True, invalid_raise=False))
            except (IndexError, ValueError) as e:
                print(f"  Unparseable log: {fb_type}/{name} ({e})")
                continue
            if rows.size == 0 or rows.dtype.names is None or "step" not in rows.dtype.names:
                print(f"  Empty/unusable log: {fb_type}/{name}")
                continue

            data[(fb_type, seed)] = rows
            succ_col = _colkey("eval/success")
            if succ_col in rows.dtype.names:
                succ = rows[succ_col]
                succ = succ[~np.isnan(succ)]
                if len(succ):
                    print(f"  {fb_type}/{name}: {len(rows)} rows  "
                          f"last_success={succ[-1]:.3f}  max_success={succ.max():.3f}")
                    continue
            print(f"  {fb_type}/{name}: {len(rows)} rows")
    return data


def get_series(rows, metric):
    """Extract (steps, values) for one metric column, dropping NaN rows."""
    col = _colkey(metric)
    if col not in rows.dtype.names:
        return None
    vals = rows[col]
    mask = ~np.isnan(vals)
    if not mask.any():
        return None
    return rows["step"][mask], vals[mask]


def _common_grid(entries, n_points=200):
    """Interpolate a list of (steps, values) arrays onto a shared step grid."""
    max_step = max(steps[-1] for steps, _ in entries)
    grid = np.linspace(0, max_step, n_points)
    interped = [np.interp(grid, steps, vals, left=vals[0], right=vals[-1]) for steps, vals in entries]
    return grid, np.array(interped)


def plot_metric(env_name, data, metric, out_dir):
    slug = env_name.replace("/", "_")
    metric_slug = metric.replace("/", "_")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    any_curve = False
    for fb_type in TYPES:
        entries = []
        for (t, s), rows in data.items():
            if t != fb_type:
                continue
            series = get_series(rows, metric)
            if series is not None:
                entries.append(series)
        if not entries:
            continue
        grid, curves = _common_grid(entries)
        mean, lo, hi = curves.mean(axis=0), curves.min(axis=0), curves.max(axis=0)
        ax.plot(grid, mean, "-", color=COLORS[fb_type], label=f"{LABELS[fb_type]} (n={len(entries)})",
                linewidth=1.8, zorder=3)
        ax.fill_between(grid, lo, hi, color=COLORS[fb_type], alpha=0.15, zorder=2)
        any_curve = True

    if not any_curve:
        plt.close(fig)
        print(f"  No data for {metric} in {env_name} — skipping.")
        return

    if metric == "eval/success":
        ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Training step", fontsize=10)
    ax.set_ylabel(metric, fontsize=10)
    ax.set_title(f"{env_name} — {metric} vs. training step", fontsize=11)
    ax.legend(fontsize=8, loc="best", framealpha=0.9)
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)
    fig.tight_layout()

    out_path = os.path.join(out_dir, f"mw_b10k_{slug}_{metric_slug}.pdf")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  {metric} figure saved -> {out_path}")
    plt.close(fig)


def plot_peak_success_summary(env_name, data, out_dir):
    slug = env_name.replace("/", "_")
    fig, ax = plt.subplots(figsize=(6, 4))
    xs, means, los, his, labels = [], [], [], [], []
    for i, fb_type in enumerate(TYPES):
        peaks = []
        for (t, s), rows in data.items():
            if t != fb_type:
                continue
            series = get_series(rows, "eval/success")
            if series is not None:
                peaks.append(float(series[1].max()))
        if not peaks:
            continue
        xs.append(i)
        means.append(np.mean(peaks))
        # max(0, ...) guards against float rounding producing a tiny negative
        # value when all seeds tie exactly (or there's only one seed) --
        # matplotlib's errorbar rejects any negative yerr outright.
        los.append(max(0.0, np.mean(peaks) - np.min(peaks)))
        his.append(max(0.0, np.max(peaks) - np.mean(peaks)))
        labels.append(LABELS[fb_type])
        ax.bar(i, np.mean(peaks), color=COLORS[fb_type], alpha=0.85, zorder=3)

    if not xs:
        plt.close(fig)
        return
    ax.errorbar(xs, means, yerr=[los, his], fmt="none", color="black",
                 capsize=3, linewidth=1.2, zorder=5)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=9, rotation=20, ha="right")
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Peak eval success rate", fontsize=10)
    ax.set_title(f"{env_name} — peak success by feedback type (10k budget)", fontsize=11)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.4, alpha=0.5)
    fig.tight_layout()

    out_path = os.path.join(out_dir, f"mw_b10k_{slug}_peak.pdf")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Peak-summary figure saved -> {out_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", default="runs/mw_b10k",
                        help="Root runs directory: <runs-dir>/<env>/<type>/<alg>_s<seed>/")
    parser.add_argument("--alg", default="cpl", choices=["cpl", "piql"],
                        help="Which algorithm's runs to plot (default: cpl)")
    parser.add_argument("--metrics", default=",".join(DEFAULT_METRICS),
                        help="Comma-separated log.csv column names to plot vs. step "
                             f"(default: {','.join(DEFAULT_METRICS)})")
    parser.add_argument("--out", default=None,
                        help="Output directory for PDFs (default: results/mw_b10k/<alg>)")
    args = parser.parse_args()
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]

    repo_root = os.path.dirname(os.path.dirname(__file__))
    runs_dir = args.runs_dir
    if not os.path.isabs(runs_dir):
        runs_dir = os.path.join(repo_root, runs_dir)
    out_dir = args.out or os.path.join(repo_root, "results", "mw_b10k", args.alg)
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.isdir(runs_dir):
        print(f"No runs found under {runs_dir}")
        return

    env_dirs = sorted(
        d for d in os.listdir(runs_dir)
        if d.startswith("mw_") and os.path.isdir(os.path.join(runs_dir, d))
    )
    if not env_dirs:
        print(f"No mw_* env subdirectories under {runs_dir}")
        return

    print(f"Found {len(env_dirs)} environment(s): {env_dirs}")
    print(f"Metrics: {metrics}")
    for env_name in env_dirs:
        print(f"\n{'='*60}\nEnv: {env_name}\n{'='*60}")
        data = load_env_runs(os.path.join(runs_dir, env_name), args.alg)
        print(f"{len(data)} runs loaded.")
        if not data:
            continue
        for metric in metrics:
            plot_metric(env_name, data, metric, out_dir)
        plot_peak_success_summary(env_name, data, out_dir)


if __name__ == "__main__":
    main()
