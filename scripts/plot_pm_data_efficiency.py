"""
Plot the PointMass data-efficiency figure from WandB.

Fetches all runs in the pm-cpl-data-efficiency project, extracts
eval/success_rate curves, computes peak smoothed success rate per
(feedback_type, budget), and plots 6 curves on a log-budget x-axis
with min/max shading across the two seeds.

Usage
-----
python scripts/plot_pm_data_efficiency.py \\
    --entity  <wandb-entity> \\
    --project pm-cpl-data-efficiency \\
    --out     results/pm_data_efficiency.pdf

Optional flags
--------------
  --smooth   Rolling-window half-width for success-rate smoothing (default 3)
  --min-steps Ignore eval points before this training step (default 0)
"""

import argparse
import re
import sys
from collections import defaultdict

import numpy as np

# ── Matplotlib setup ──────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TYPES   = ["pref", "corr", "demo", "estop", "scalar", "credit_assignment"]
BUDGETS = [100, 300, 1_000, 3_000, 10_000, 20_000]

# Colorblind-friendly palette (Wong 2011)
COLORS = {
    "pref":              "#0072B2",   # blue
    "corr":              "#E69F00",   # orange
    "demo":              "#009E73",   # green
    "estop":             "#D55E00",   # vermillion
    "scalar":            "#CC79A7",   # pink
    "credit_assignment": "#56B4E9",   # sky blue
}
LABELS = {
    "pref":              "Pref",
    "corr":              "Corr",
    "demo":              "Demo",
    "estop":             "E-stop",
    "scalar":            "Scalar",
    "credit_assignment": "Credit",
}


def smooth_and_peak(values, half_w):
    """Apply centred rolling mean with half-width half_w, return max."""
    w = 2 * half_w + 1
    if len(values) < w:
        return float(np.max(values)) if len(values) > 0 else 0.0
    kernel = np.ones(w) / w
    smoothed = np.convolve(values, kernel, mode="valid")
    return float(np.max(smoothed))


def parse_run_name(name):
    """
    Extract (feedback_type, budget, seed) from run name pattern:
        <type>_b<budget:06d>_s<seed>
    Returns None if the name doesn't match.
    """
    m = re.fullmatch(r"([a-z_]+)_b(\d+)_s(\d+)", name)
    if m is None:
        return None
    return m.group(1), int(m.group(2)), int(m.group(3))


def fetch_runs(entity, project, min_steps):
    """
    Returns dict: (feedback_type, budget, seed) → peak_success_rate
    """
    try:
        import wandb
    except ImportError:
        print("ERROR: wandb not installed.  pip install wandb", file=sys.stderr)
        sys.exit(1)

    api  = wandb.Api()
    runs = api.runs(f"{entity}/{project}")
    print(f"Fetched {len(runs)} runs from {entity}/{project}")

    results = {}

    for run in runs:
        parsed = parse_run_name(run.name)
        if parsed is None:
            print(f"  Skipping (name doesn't match pattern): {run.name}")
            continue
        fb_type, budget, seed = parsed

        if fb_type not in TYPES or budget not in BUDGETS:
            print(f"  Skipping (unknown type/budget): {run.name}")
            continue

        if run.state not in ("finished", "running"):
            print(f"  Skipping (state={run.state}): {run.name}")
            continue

        history = run.history(keys=["eval/success_rate", "_step"], pandas=False)
        values  = [
            row["eval/success_rate"]
            for row in history
            if row.get("eval/success_rate") is not None
            and row.get("_step", 0) >= min_steps
        ]

        if not values:
            print(f"  No eval data yet: {run.name}")
            continue

        key = (fb_type, budget, seed)
        results[key] = values
        print(f"  {run.name}: {len(values)} eval points, "
              f"last={values[-1]:.2f}")

    return results


def aggregate(raw, smooth_half_w):
    """
    raw: {(type, budget, seed): [success_rate...]}
    Returns: {(type, budget): {"mean": float, "lo": float, "hi": float}}
    """
    grouped = defaultdict(list)
    for (fb_type, budget, seed), values in raw.items():
        peak = smooth_and_peak(values, smooth_half_w)
        grouped[(fb_type, budget)].append(peak)

    agg = {}
    for (fb_type, budget), peaks in grouped.items():
        agg[(fb_type, budget)] = {
            "mean": float(np.mean(peaks)),
            "lo":   float(np.min(peaks)),
            "hi":   float(np.max(peaks)),
            "n":    len(peaks),
        }
    return agg


def plot(agg, out_path):
    fig, ax = plt.subplots(figsize=(7, 4.5))

    for fb_type in TYPES:
        pts = [(b, agg[(fb_type, b)]) for b in BUDGETS if (fb_type, b) in agg]
        if not pts:
            continue
        xs    = np.array([p[0] for p in pts])
        means = np.array([p[1]["mean"] for p in pts])
        los   = np.array([p[1]["lo"]   for p in pts])
        his   = np.array([p[1]["hi"]   for p in pts])
        color = COLORS[fb_type]
        label = LABELS[fb_type]

        ax.plot(xs, means, "o-", color=color, label=label, linewidth=1.8,
                markersize=5, zorder=3)
        ax.fill_between(xs, los, his, color=color, alpha=0.15, zorder=2)

    ax.set_xscale("log")
    ax.set_xlim(80, 25_000)
    ax.set_xticks(BUDGETS)
    ax.set_xticklabels([str(b) for b in BUDGETS], fontsize=8)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Budget  (n feedback pairs)", fontsize=10)
    ax.set_ylabel("Peak success rate", fontsize=10)
    ax.set_title("PointMass — CPL data efficiency by feedback type", fontsize=11)
    ax.legend(fontsize=8, loc="lower right", framealpha=0.9)
    ax.grid(True, which="both", linestyle="--", linewidth=0.4, alpha=0.5)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved → {out_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot PointMass CPL data-efficiency figure from WandB."
    )
    parser.add_argument("--entity",   required=True,
                        help="WandB entity (username or team)")
    parser.add_argument("--project",  default="pm-cpl-data-efficiency",
                        help="WandB project name")
    parser.add_argument("--out",      default="results/pm_data_efficiency.pdf",
                        help="Output figure path (.pdf or .png)")
    parser.add_argument("--smooth",   type=int, default=3,
                        help="Rolling-window half-width for smoothing (default 3)")
    parser.add_argument("--min-steps", type=int, default=0,
                        help="Ignore eval steps before this value (default 0)")
    args = parser.parse_args()

    raw = fetch_runs(args.entity, args.project, args.min_steps)
    if not raw:
        print("No runs found — nothing to plot.", file=sys.stderr)
        sys.exit(1)

    agg = aggregate(raw, args.smooth)

    # Summary table
    print(f"\n{'Type':<20}  {'Budget':>7}  {'n':>2}  {'mean':>6}  {'lo':>6}  {'hi':>6}")
    print("-" * 55)
    for fb_type in TYPES:
        for budget in BUDGETS:
            if (fb_type, budget) not in agg:
                continue
            d = agg[(fb_type, budget)]
            print(f"{fb_type:<20}  {budget:>7}  {d['n']:>2}  "
                  f"{d['mean']:>6.3f}  {d['lo']:>6.3f}  {d['hi']:>6.3f}")

    import os
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    plot(agg, args.out)


if __name__ == "__main__":
    main()
