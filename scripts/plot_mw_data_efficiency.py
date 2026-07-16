"""
Plot MetaWorld CPL data-efficiency from local log.csv files.

Reads runs/mw_de/<type>_b<budget>_s<seed>/log.csv, computes peak smoothed
eval success rate per (type, budget), and plots two figures:

  Plot A — 6 curves on a log-budget x-axis, min/max shading across seeds.
  Plot B — grouped bars: budget needed to first reach each success threshold
            (median bar + min/max whiskers across seeds; hatched = never reached).

Usage
-----
python scripts/plot_mw_data_efficiency.py
python scripts/plot_mw_data_efficiency.py \
    --runs-dir runs/mw_de \
    --env mw_drawer-open-v2 \
    --out results/mw_de_drawer-open.pdf
"""

import argparse
import os
import re
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TYPES   = ["pref", "corr", "demo", "seq_estop", "scalar", "credit_assignment"]
BUDGETS = [50, 100, 300, 600, 1_000, 2_000, 5_000, 10_000, 20_000, 40_000, 100_000]
THRESHOLDS = [0.5, 0.7, 0.9]

COLORS = {
    "pref":              "#0072B2",
    "corr":              "#E69F00",
    "demo":              "#009E73",
    "seq_estop":         "#D55E00",
    "scalar":            "#CC79A7",
    "credit_assignment": "#56B4E9",
}
LABELS = {
    "pref":              "Pref",
    "corr":              "Corr",
    "demo":              "Demo",
    "seq_estop":         "Seq E-stop",
    "scalar":            "Scalar",
    "credit_assignment": "Credit",
}


def parse_run_name(name):
    """<type>_b<budget>_s<seed>  →  (type, budget, seed) or None"""
    m = re.fullmatch(r"([a-z_]+)_b(\d+)_s(\d+)", name)
    if m is None:
        return None
    return m.group(1), int(m.group(2)), int(m.group(3))


def load_runs(runs_dir):
    """Returns {(type, budget, seed): np.array of eval success values}"""
    data = {}
    if not os.path.isdir(runs_dir):
        print(f"WARNING: runs dir not found: {runs_dir}")
        return data

    for name in sorted(os.listdir(runs_dir)):
        parsed = parse_run_name(name)
        if parsed is None:
            continue
        fb_type, budget, seed = parsed
        if fb_type not in TYPES:
            continue

        log_path = os.path.join(runs_dir, name, "log.csv")
        if not os.path.exists(log_path):
            print(f"  Missing log: {name}")
            continue

        rows = np.genfromtxt(log_path, delimiter=",", names=True, invalid_raise=False)

        # MetaWorld logs success under various names depending on env/logger version.
        # numpy genfromtxt strips '/' from header names (eval/success → evalsuccess).
        succ_col = None
        for candidate in ("evalsuccess", "evalsucc", "eval/success", "eval/succ",
                           "eval_success", "eval_succ", "success"):
            if candidate in rows.dtype.names:
                succ_col = candidate
                break
        if succ_col is None:
            print(f"  No success column in {name} — columns: {rows.dtype.names}")
            continue

        mask  = ~np.isnan(rows[succ_col])
        succs = rows[succ_col][mask]

        if len(succs) == 0:
            print(f"  No eval rows: {name}")
            continue

        data[(fb_type, budget, seed)] = succs
        print(f"  {name}: {len(succs)} eval pts  "
              f"last={succs[-1]:.3f}  max={succs.max():.3f}")
    return data


def smooth_peak(values, half_w=3):
    """Rolling mean (window = 2*half_w+1), return max."""
    w = 2 * half_w + 1
    if len(values) < w:
        return float(np.max(values))
    kernel   = np.ones(w) / w
    smoothed = np.convolve(values, kernel, mode="valid")
    return float(np.max(smoothed))


def compute_peaks(data, smooth_half_w):
    return {key: smooth_peak(succs, smooth_half_w) for key, succs in data.items()}


def aggregate(peaks):
    grouped = defaultdict(list)
    for (fb_type, budget, seed), val in peaks.items():
        grouped[(fb_type, budget)].append(val)

    agg = {}
    for key, vals in grouped.items():
        agg[key] = {
            "mean": float(np.mean(vals)),
            "lo":   float(np.min(vals)),
            "hi":   float(np.max(vals)),
            "n":    len(vals),
        }
    return agg


def threshold_crossing_log10(budgets_arr, succ_arr, threshold):
    """
    Running-max envelope; return log10(budget) where it first reaches threshold.
    Interpolated in log10 space.  None if never reached.
    """
    envelope = np.maximum.accumulate(succ_arr)
    above = np.where(envelope >= threshold)[0]
    if len(above) == 0:
        return None
    k = int(above[0])
    if k == 0:
        return np.log10(float(budgets_arr[0]))
    lo_log = np.log10(float(budgets_arr[k - 1]))
    hi_log = np.log10(float(budgets_arr[k]))
    lo_v, hi_v = float(envelope[k - 1]), float(envelope[k])
    if hi_v <= lo_v:
        return lo_log
    frac = (threshold - lo_v) / (hi_v - lo_v)
    return lo_log + frac * (hi_log - lo_log)


# ── Plot A ─────────────────────────────────────────────────────────────────────

def plot_a(agg, env_name, out_path):
    fig, ax = plt.subplots(figsize=(7, 4.5))

    for fb_type in TYPES:
        pts = [(b, agg[(fb_type, b)]) for b in BUDGETS if (fb_type, b) in agg]
        if not pts:
            continue
        xs    = np.array([p[0] for p in pts])
        means = np.array([p[1]["mean"] for p in pts])
        los   = np.array([p[1]["lo"]   for p in pts])
        his   = np.array([p[1]["hi"]   for p in pts])

        ax.plot(xs, means, "o-", color=COLORS[fb_type], label=LABELS[fb_type],
                linewidth=1.8, markersize=5, zorder=3)
        ax.fill_between(xs, los, his, color=COLORS[fb_type], alpha=0.15, zorder=2)

    ax.set_xscale("log")
    ax.set_xlim(38, 130_000)
    ax.set_xticks(BUDGETS)
    ax.set_xticklabels([str(b) for b in BUDGETS], fontsize=7, rotation=30, ha="right")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Budget  (n feedback pairs / examples)", fontsize=10)
    ax.set_ylabel("Peak success rate", fontsize=10)
    ax.set_title(f"{env_name} — CPL data efficiency by feedback type", fontsize=11)
    ax.legend(fontsize=8, loc="lower right", framealpha=0.9)
    ax.grid(True, which="both", linestyle="--", linewidth=0.4, alpha=0.5)

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure A saved → {out_path}")
    plt.close(fig)


# ── Plot B ─────────────────────────────────────────────────────────────────────

def plot_b(peaks, env_name, out_path, thresholds=THRESHOLDS):
    budgets_arr = np.array(BUDGETS, dtype=float)
    ceil_log    = np.log10(float(BUDGETS[-1]))
    n_types     = len(TYPES)
    n_thresh    = len(thresholds)

    ts_data = defaultdict(dict)
    for (fb_type, budget, seed), val in peaks.items():
        ts_data[(fb_type, seed)][budget] = val

    crossing_logs = [[[] for _ in range(n_thresh)] for _ in range(n_types)]
    for ti, fb_type in enumerate(TYPES):
        for (t2, seed), bdict in ts_data.items():
            if t2 != fb_type:
                continue
            succ_arr = np.array([bdict.get(b, np.nan) for b in BUDGETS])
            for j in range(1, len(succ_arr)):
                if np.isnan(succ_arr[j]) and not np.isnan(succ_arr[j - 1]):
                    succ_arr[j] = succ_arr[j - 1]
            if np.all(np.isnan(succ_arr)):
                continue
            succ_arr = np.where(np.isnan(succ_arr), 0.0, succ_arr)
            for thi, thr in enumerate(thresholds):
                c = threshold_crossing_log10(budgets_arr, succ_arr, thr)
                crossing_logs[ti][thi].append(c)

    group_spacing = 1.5
    bar_w = 0.8 / n_types
    x_centers = np.arange(n_thresh, dtype=float) * group_spacing

    fig, ax = plt.subplots(figsize=(9, 5))

    for ti, fb_type in enumerate(TYPES):
        color = COLORS[fb_type]
        label = LABELS[fb_type]
        x_off = (ti - n_types / 2 + 0.5) * bar_w

        for thi in range(n_thresh):
            vals    = crossing_logs[ti][thi]
            reached = [v for v in vals if v is not None]
            all_vals = [v if v is not None else ceil_log for v in vals]

            if len(reached) == 0:
                med_log = ceil_log
                lo_err  = 0.0
                hi_err  = 0.0
                hatch   = "////"
                edge    = "black"
            else:
                arr     = np.array(all_vals)
                med_log = float(np.median(arr))
                lo_err  = med_log - float(np.min(arr))
                hi_err  = float(np.max(arr)) - med_log
                hatch   = "////" if len(reached) < len(vals) else ""
                edge    = "black" if hatch else color

            x = x_centers[thi] + x_off
            ax.bar(x, med_log, width=bar_w * 0.88,
                   color=color, alpha=0.82, zorder=3,
                   label=label if thi == 0 else "",
                   hatch=hatch, edgecolor=edge, linewidth=0.8)
            if lo_err > 0 or hi_err > 0:
                ax.errorbar(x, med_log,
                            yerr=[[lo_err], [hi_err]],
                            fmt="none", color="black",
                            capsize=3, linewidth=1.2, zorder=5)

    ytick_b = BUDGETS
    ax.set_yticks([np.log10(b) for b in ytick_b])
    ax.set_yticklabels([str(b) for b in ytick_b], fontsize=8)
    ax.set_ylim(np.log10(BUDGETS[0]) - 0.2, ceil_log + 0.15)

    ax.set_xticks(x_centers)
    ax.set_xticklabels([f"≥ {int(t * 100)}%" for t in thresholds], fontsize=10)
    ax.set_xlim(x_centers[0] - group_spacing * 0.55,
                x_centers[-1] + group_spacing * 0.55)

    ax.set_ylabel("Budget at first threshold crossing (log scale)", fontsize=10)
    ax.set_title(f"{env_name} — Budget to reach success threshold", fontsize=11)
    ax.legend(fontsize=8, loc="upper left", framealpha=0.9, ncol=2)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.4, alpha=0.5)

    ax.axhline(ceil_log, color="gray", linewidth=0.8, linestyle=":", zorder=1)
    ax.text(ax.get_xlim()[1], ceil_log + 0.02, "ceiling",
            ha="right", va="bottom", fontsize=7, color="gray")

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Figure B saved → {out_path}")
    plt.close(fig)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _detect_envs(runs_dir):
    """
    Auto-detect environment layout under runs_dir.

    Two layouts are supported:
      Flat  : runs_dir/<type>_b<budget>_s<seed>/   → one env (use --env name)
      Nested: runs_dir/<env_name>/<type>_b<budget>_s<seed>/

    Returns list of (env_name, effective_runs_dir) pairs.
    """
    if not os.path.isdir(runs_dir):
        return []

    # Nested: look for mw_* subdirs that themselves contain run dirs
    env_dirs = sorted(
        d for d in os.listdir(runs_dir)
        if d.startswith("mw_") and os.path.isdir(os.path.join(runs_dir, d))
    )
    if env_dirs:
        return [(e, os.path.join(runs_dir, e)) for e in env_dirs]

    # Flat: the runs_dir itself contains <type>_b<budget>_s<seed>/ dirs
    return [("mw_drawer-open-v2", runs_dir)]   # default env label for flat layout


def _plot_env(env_name, env_runs_dir, out_dir, smooth):
    print(f"\n{'='*60}")
    print(f"Env: {env_name}  ({env_runs_dir})")
    print("="*60)

    data = load_runs(env_runs_dir)
    print(f"\n{len(data)} runs loaded.")

    if not data:
        print("No runs with log.csv — skipping.")
        return

    peaks = compute_peaks(data, smooth)
    agg   = aggregate(peaks)

    print(f"\n{'Type':<20}  {'Budget':>7}  {'n':>2}  {'mean':>6}  {'lo':>6}  {'hi':>6}")
    print("-" * 57)
    for fb_type in TYPES:
        for budget in BUDGETS:
            if (fb_type, budget) not in agg:
                continue
            d = agg[(fb_type, budget)]
            print(f"{fb_type:<20}  {budget:>7}  {d['n']:>2}  "
                  f"{d['mean']:>6.3f}  {d['lo']:>6.3f}  {d['hi']:>6.3f}")

    slug  = env_name.replace("/", "_")
    out_a = os.path.join(out_dir, f"mw_de_{slug}.pdf")
    out_b = os.path.join(out_dir, f"mw_de_{slug}_b.pdf")
    plot_a(agg, env_name, out_a)
    plot_b(peaks, env_name, out_b)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", default="runs/mw_de_cb075",
                        help="Root runs directory. Supports two layouts:\n"
                             "  Flat  : <runs-dir>/<type>_b<budget>_s<seed>/  (single env)\n"
                             "  Nested: <runs-dir>/<env>/<type>_b<budget>_s<seed>/  (multi-env)\n"
                             "Multi-env layout is auto-detected from mw_* subdirectories.")
    parser.add_argument("--env", default=None,
                        help="Override env name label for flat layout "
                             "(default: mw_drawer-open-v2). Ignored in nested layout.")
    parser.add_argument("--out", default=None,
                        help="Output directory for PDFs (default: results/). "
                             "Ignored when --runs-dir is given explicitly with nested layout.")
    parser.add_argument("--smooth", type=int, default=3,
                        help="Rolling-window half-width for peak smoothing (default 3)")
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.dirname(__file__))

    runs_dir = args.runs_dir
    if not os.path.isabs(runs_dir):
        runs_dir = os.path.join(repo_root, runs_dir)

    out_dir = args.out or os.path.join(repo_root, "results")
    os.makedirs(out_dir, exist_ok=True)

    envs = _detect_envs(runs_dir)
    if not envs:
        print(f"No runs found under {runs_dir}")
        return

    # Allow --env override for flat layout
    if len(envs) == 1 and args.env:
        envs = [(args.env, envs[0][1])]

    print(f"Found {len(envs)} environment(s): {[e for e, _ in envs]}")

    for env_name, env_runs_dir in envs:
        _plot_env(env_name, env_runs_dir, out_dir, args.smooth)


if __name__ == "__main__":
    main()
