"""
Plot PointMass CPL data-efficiency from local log.csv files.

Reads runs/pm_de/<type>_b<budget>_s<seed>/log.csv, computes peak
smoothed eval/succ per (type, budget), plots two figures:

  Plot A — 6 curves on a log-budget x-axis, min/max shading across seeds.
  Plot B — grouped bars: budget needed to first reach each success threshold
            (median bar + min/max whiskers across seeds; hatched = never reached).

Usage
-----
python scripts/plot_pm_data_efficiency_local.py
python scripts/plot_pm_data_efficiency_local.py --runs-dir runs/pm_de --out results/pm_data_efficiency.pdf
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
BUDGETS = [50, 100, 300, 600, 1_000, 2_000, 5_000, 10_000, 20_000]
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
    """Returns {(type, budget, seed): np.array of eval/succ values}"""
    data = {}
    for name in sorted(os.listdir(runs_dir)):
        parsed = parse_run_name(name)
        if parsed is None:
            continue
        fb_type, budget, seed = parsed
        log_path = os.path.join(runs_dir, name, "log.csv")
        if not os.path.exists(log_path):
            print(f"  Missing log: {name}")
            continue

        rows = np.genfromtxt(log_path, delimiter=",", names=True)
        mask  = ~np.isnan(rows["evalsucc"])
        succs = rows["evalsucc"][mask]

        if len(succs) == 0:
            print(f"  No eval rows: {name}")
            continue

        data[(fb_type, budget, seed)] = succs
        print(f"  {name}: {len(succs)} eval points, "
              f"last={succs[-1]:.2f}, max={succs.max():.2f}")
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
    """Returns {(type, budget, seed): float} — smoothed peak success rate."""
    return {key: smooth_peak(succs, smooth_half_w) for key, succs in data.items()}


def aggregate(peaks):
    """
    peaks: {(type, budget, seed): float}
    Returns {(type, budget): {mean, lo, hi, n}}
    """
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
    Running-max envelope of succ_arr (indexed by budgets_arr, ascending).
    Returns log10(budget) where envelope first reaches threshold,
    interpolated in log10 space.  Returns None if never reached.
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


# ── Plot A ─────────────────────────────────────────────────────────────────

def plot_a(agg, out_path):
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
    ax.set_xlim(38, 25_000)
    ax.set_xticks(BUDGETS)
    ax.set_xticklabels([str(b) for b in BUDGETS], fontsize=7)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Budget  (n feedback pairs)", fontsize=10)
    ax.set_ylabel("Peak success rate", fontsize=10)
    ax.set_title("PointMass — CPL data efficiency by feedback type", fontsize=11)
    ax.legend(fontsize=8, loc="lower right", framealpha=0.9)
    ax.grid(True, which="both", linestyle="--", linewidth=0.4, alpha=0.5)

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure A saved → {out_path}")
    plt.close(fig)


# ── Plot B ─────────────────────────────────────────────────────────────────

def plot_b(peaks, out_path, thresholds=THRESHOLDS):
    """
    Grouped bar chart: budget needed to first reach each success threshold.
    Groups = thresholds.  Bars within each group = feedback types.
    Bar height = log10(median crossing budget) over seeds.
    Whiskers = min/max over seeds.
    Hatched + placed at ceiling = never reached by any seed.
    """
    budgets_arr = np.array(BUDGETS, dtype=float)
    ceil_log    = np.log10(float(BUDGETS[-1]))
    n_types     = len(TYPES)
    n_thresh    = len(thresholds)

    # Collect per-seed data: ts_data[(type, seed)] = {budget: peak_succ}
    ts_data = defaultdict(dict)
    for (fb_type, budget, seed), val in peaks.items():
        ts_data[(fb_type, seed)][budget] = val

    # crossing_logs[ti][thi] = list of log10 budget values (one per seed, None=not reached)
    crossing_logs = [[[] for _ in range(n_thresh)] for _ in range(n_types)]
    for ti, fb_type in enumerate(TYPES):
        for (t2, seed), bdict in ts_data.items():
            if t2 != fb_type:
                continue
            succ_arr = np.array([bdict.get(b, np.nan) for b in BUDGETS])
            # Forward-fill NaNs so envelope is defined
            for j in range(1, len(succ_arr)):
                if np.isnan(succ_arr[j]) and not np.isnan(succ_arr[j - 1]):
                    succ_arr[j] = succ_arr[j - 1]
            if np.all(np.isnan(succ_arr)):
                continue
            succ_arr = np.where(np.isnan(succ_arr), 0.0, succ_arr)
            for thi, thr in enumerate(thresholds):
                c = threshold_crossing_log10(budgets_arr, succ_arr, thr)
                crossing_logs[ti][thi].append(c)

    # Layout
    group_spacing = 1.5
    bar_w = 0.8 / n_types
    x_centers = np.arange(n_thresh, dtype=float) * group_spacing

    fig, ax = plt.subplots(figsize=(9, 5))

    for ti, fb_type in enumerate(TYPES):
        color  = COLORS[fb_type]
        label  = LABELS[fb_type]
        x_off  = (ti - n_types / 2 + 0.5) * bar_w

        for thi in range(n_thresh):
            vals     = crossing_logs[ti][thi]
            reached  = [v for v in vals if v is not None]
            n_seeds  = len(vals)

            # Replace unreached seeds with ceiling for min/max computation
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
                hatch   = "////" if len(reached) < n_seeds else ""
                edge    = "black" if hatch else color

            x = x_centers[thi] + x_off
            bar = ax.bar(x, med_log, width=bar_w * 0.88,
                         color=color, alpha=0.82, zorder=3,
                         label=label if thi == 0 else "",
                         hatch=hatch, edgecolor=edge, linewidth=0.8)
            if lo_err > 0 or hi_err > 0:
                ax.errorbar(x, med_log,
                            yerr=[[lo_err], [hi_err]],
                            fmt="none", color="black",
                            capsize=3, linewidth=1.2, zorder=5)

    # Y-axis: log10 budget ticks
    ytick_b = [b for b in BUDGETS]
    ax.set_yticks([np.log10(b) for b in ytick_b])
    ax.set_yticklabels([str(b) for b in ytick_b], fontsize=8)
    ax.set_ylim(np.log10(BUDGETS[0]) - 0.2, ceil_log + 0.15)

    # X-axis: threshold labels
    ax.set_xticks(x_centers)
    ax.set_xticklabels([f"≥ {int(t * 100)}%" for t in thresholds], fontsize=10)
    ax.set_xlim(x_centers[0] - group_spacing * 0.55,
                x_centers[-1] + group_spacing * 0.55)

    ax.set_ylabel("Budget at first threshold crossing (log scale)", fontsize=10)
    ax.set_title("PointMass — Budget to reach success threshold", fontsize=11)
    ax.legend(fontsize=8, loc="upper left", framealpha=0.9, ncol=2)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.4, alpha=0.5)

    # Ceiling annotation
    ax.axhline(ceil_log, color="gray", linewidth=0.8, linestyle=":", zorder=1)
    ax.text(ax.get_xlim()[1], ceil_log + 0.02, "ceiling",
            ha="right", va="bottom", fontsize=7, color="gray")

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Figure B saved → {out_path}")
    plt.close(fig)


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", default="runs/pm_de",
                        help="Directory containing <type>_b<budget>_s<seed>/ runs")
    parser.add_argument("--out", default="results/pm_data_efficiency.pdf")
    parser.add_argument("--smooth", type=int, default=3,
                        help="Rolling-window half-width (default 3)")
    args = parser.parse_args()

    runs_dir = args.runs_dir
    if not os.path.isabs(runs_dir):
        runs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), runs_dir)

    print(f"Loading runs from {runs_dir} …")
    data = load_runs(runs_dir)
    print(f"\n{len(data)} runs loaded.")

    peaks = compute_peaks(data, args.smooth)
    agg   = aggregate(peaks)

    print(f"\n{'Type':<20}  {'Budget':>7}  {'n':>2}  {'mean':>6}  {'lo':>6}  {'hi':>6}")
    print("-" * 55)
    for fb_type in TYPES:
        for budget in BUDGETS:
            if (fb_type, budget) not in agg:
                continue
            d = agg[(fb_type, budget)]
            print(f"{fb_type:<20}  {budget:>7}  {d['n']:>2}  "
                  f"{d['mean']:>6.3f}  {d['lo']:>6.3f}  {d['hi']:>6.3f}")

    # Plot A
    plot_a(agg, args.out)

    # Plot B — derive output path from --out
    stem, ext = os.path.splitext(args.out)
    out_b = stem + "_b" + ext
    plot_b(peaks, out_b)


if __name__ == "__main__":
    main()
