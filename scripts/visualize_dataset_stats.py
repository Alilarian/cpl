"""
Generate LaTeX tables and matplotlib figures from dataset_stats CSVs.

Outputs:
  results/figures/pool_stats.pdf         — pool reward distribution bar chart
  results/figures/quality_gradient.pdf   — rl_sum per K position (line plot)
  results/figures/adv_gap.pdf            — advantage gap per env (bar chart)
  results/figures/action_diversity.pdf   — choice set action diversity per env
  results/tables/pool_stats.tex          — LaTeX table: pool statistics
  results/tables/demo_stats.tex          — LaTeX table: demo label statistics
  results/tables/quality_gradient.tex    — LaTeX table: rl_sum per position

Usage:
    python scripts/visualize_dataset_stats.py \\
        --stats-dir results/dataset_stats \\
        --output-dir results
"""

import argparse
import os
import glob

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ENV_SHORT = {
    "mw_bin-picking-v2":   "Bin Pick",
    "mw_button-press-v2":  "Button",
    "mw_door-open-v2":     "Door",
    "mw_drawer-open-v2":   "Drawer",
    "mw_plate-slide-v2":   "Plate Slide",
}

COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_stats(stats_dir):
    rows = []
    for path in sorted(glob.glob(os.path.join(stats_dir, "*.csv"))):
        df = pd.read_csv(path)
        rows.append(df.iloc[0].to_dict())
    df = pd.DataFrame(rows)
    df["env_short"] = df["env"].map(lambda e: ENV_SHORT.get(e, e))
    return df


# ---------------------------------------------------------------------------
# LaTeX helpers
# ---------------------------------------------------------------------------

def fmt(val, decimals=2):
    if pd.isna(val):
        return "--"
    if isinstance(val, float):
        return f"{val:.{decimals}f}"
    return str(int(val))


def save_tex(lines, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Saved → {path}")


def latex_pool_table(df, path):
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Trajectory Pool Statistics (Phase 3). "
        r"Reward is the undiscounted sum over $T=50$ steps. "
        r"Disc.\ Reward uses $\gamma=0.99$.}",
        r"\label{tab:pool_stats}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r"Environment & $N$ & $T$ & "
        r"Reward $\mu \pm \sigma$ & Reward [min, max] & "
        r"Disc.\ Reward $\mu \pm \sigma$ & "
        r"\#Ckpts & Step Range \\",
        r"\midrule",
    ]
    for _, r in df.iterrows():
        env   = ENV_SHORT.get(r["env"], r["env"])
        lines.append(
            f"{env} & {int(r['pool_N'])} & {int(r['pool_T'])} & "
            f"${fmt(r['pool_reward_mean'])} \\pm {fmt(r['pool_reward_std'])}$ & "
            f"$[{fmt(r['pool_reward_min'])},\\, {fmt(r['pool_reward_max'])}]$ & "
            f"${fmt(r['pool_disc_reward_mean'])} \\pm {fmt(r['pool_disc_reward_std'])}$ & "
            f"{int(r['pool_ckpt_n_unique'])} & "
            f"$[{int(r['pool_ckpt_min'])},\\, {int(r['pool_ckpt_max'])}]$ \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}}",
        r"\end{table}",
    ]
    save_tex(lines, path)


def latex_demo_table(df, path):
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Demonstrative Feedback Dataset Statistics (Phase 4). "
        r"$K=7$ choice sets. Demo = position 0 (best by $\widehat{A}^*$). "
        r"Adv.\ Gap $= \text{rl\_sum}[0] - \text{rl\_sum}[K{-}1]$.}",
        r"\label{tab:demo_stats}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lrrrrrrrr}",
        r"\toprule",
        r"Environment & $N$ & Skip & "
        r"Demo Rew $\mu$ & Demo $\widehat{A}^*$ $\mu \pm \sigma$ & "
        r"Adv.\ Gap $\mu \pm \sigma$ & Adv.\ Gap [min, max] & "
        r"Disc.\ Rew $\mu$ & Diversity \\",
        r"\midrule",
    ]
    for _, r in df.iterrows():
        env = ENV_SHORT.get(r["env"], r["env"])
        skip_pct = f"{float(r['demo_skip_rate'])*100:.1f}\\%"
        lines.append(
            f"{env} & {int(r['demo_N'])} & {skip_pct} & "
            f"{fmt(r['demo_pos0_reward_mean'])} & "
            f"${fmt(r['demo_rl_sum_pos0_mean'])} \\pm {fmt(r['demo_rl_sum_pos0_std'])}$ & "
            f"${fmt(r['demo_adv_gap_mean'])} \\pm {fmt(r['demo_adv_gap_std'])}$ & "
            f"$[{fmt(r['demo_adv_gap_min'])},\\, {fmt(r['demo_adv_gap_max'])}]$ & "
            f"{fmt(r['demo_disc_reward_pos0_mean'])} & "
            f"{fmt(r['demo_action_diversity_mean'])} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}}",
        r"\end{table}",
    ]
    save_tex(lines, path)


def latex_quality_gradient_table(df, path):
    K = int(df["demo_K"].iloc[0])
    pos_cols_mean = [f"demo_rl_sum_pos{k}_mean" for k in range(K)]

    col_headers = " & ".join([f"$k={k}$" for k in range(K)])
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Mean $\widehat{A}^*$ (rl\_sum) per choice-set position $k$ "
        r"($k=0$ = demonstration, $k=K{-}1$ = worst counterfactual). "
        r"Values confirm a monotone quality gradient across positions.}",
        r"\label{tab:quality_gradient}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{l" + "r" * K + "}",
        r"\toprule",
        f"Environment & {col_headers} \\\\",
        r"\midrule",
    ]
    for _, r in df.iterrows():
        env  = ENV_SHORT.get(r["env"], r["env"])
        vals = " & ".join([fmt(r.get(c, float("nan"))) for c in pos_cols_mean])
        lines.append(f"{env} & {vals} \\\\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}}",
        r"\end{table}",
    ]
    save_tex(lines, path)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def savefig(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved → {path}")


def fig_pool_rewards(df, out_dir):
    envs   = df["env_short"].tolist()
    x      = np.arange(len(envs))
    width  = 0.35

    fig, ax = plt.subplots(figsize=(9, 4))
    bars1 = ax.bar(x - width/2, df["pool_reward_mean"],      width,
                   yerr=df["pool_reward_std"],      capsize=4,
                   color=COLORS[0], label="Undiscounted", alpha=0.85)
    bars2 = ax.bar(x + width/2, df["pool_disc_reward_mean"], width,
                   yerr=df["pool_disc_reward_std"], capsize=4,
                   color=COLORS[1], label=r"Discounted ($\gamma=0.99$)", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(envs, rotation=15, ha="right")
    ax.set_ylabel("Segment Reward (sum over $T=50$)")
    ax.set_title("Pool Segment Reward Distribution")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    savefig(fig, os.path.join(out_dir, "figures", "pool_rewards.pdf"))


def fig_quality_gradient(df, out_dir):
    K = int(df["demo_K"].iloc[0])
    pos_cols = [f"demo_rl_sum_pos{k}_mean" for k in range(K)]
    std_cols = [f"demo_rl_sum_pos{k}_std"  for k in range(K)]

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(K)

    for i, (_, row) in enumerate(df.iterrows()):
        means = [row.get(c, np.nan) for c in pos_cols]
        stds  = [row.get(c, np.nan) for c in std_cols]
        ax.plot(x, means, marker="o", color=COLORS[i],
                label=row["env_short"], linewidth=1.8)
        ax.fill_between(x,
                        [m - s for m, s in zip(means, stds)],
                        [m + s for m, s in zip(means, stds)],
                        color=COLORS[i], alpha=0.12)

    ax.set_xticks(x)
    ax.set_xticklabels([f"$k={k}$" for k in range(K)])
    ax.set_xlabel("Choice-set position $k$  (0 = demo, $K-1$ = worst)")
    ax.set_ylabel(r"Mean $\widehat{A}^*$ (rl\_sum)")
    ax.set_title("Quality Gradient Across Choice-Set Positions")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    savefig(fig, os.path.join(out_dir, "figures", "quality_gradient.pdf"))


def fig_adv_gap(df, out_dir):
    envs  = df["env_short"].tolist()
    x     = np.arange(len(envs))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x, df["demo_adv_gap_mean"], yerr=df["demo_adv_gap_std"],
           capsize=5, color=COLORS[2], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(envs, rotation=15, ha="right")
    ax.set_ylabel(r"$\widehat{A}^*(demo) - \widehat{A}^*(worst)$")
    ax.set_title("Advantage Gap: Demo vs Worst Counterfactual")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    savefig(fig, os.path.join(out_dir, "figures", "adv_gap.pdf"))


def fig_action_diversity(df, out_dir):
    envs = df["env_short"].tolist()
    x    = np.arange(len(envs))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x, df["demo_action_diversity_mean"], color=COLORS[3], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(envs, rotation=15, ha="right")
    ax.set_ylabel("Mean Pairwise L2 Distance (actions)")
    ax.set_title("Choice-Set Action Diversity")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    savefig(fig, os.path.join(out_dir, "figures", "action_diversity.pdf"))


def fig_reward_gradient(df, out_dir):
    """Undiscounted reward per position: demo, mid, worst."""
    K       = int(df["demo_K"].iloc[0])
    mid_pos = K // 2

    pos_labels = ["Demo (pos 0)", f"Mid (pos {mid_pos})", f"Worst (pos {K-1})"]
    col_keys   = [
        "demo_pos0_reward_mean",
        f"demo_pos{mid_pos}_reward_mean",
        f"demo_pos{K-1}_reward_mean",
    ]
    x     = np.arange(len(df))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 4))
    for j, (label, col) in enumerate(zip(pos_labels, col_keys)):
        vals = [r.get(col, np.nan) for _, r in df.iterrows()]
        ax.bar(x + (j - 1) * width, vals, width,
               color=COLORS[j], label=label, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(df["env_short"].tolist(), rotation=15, ha="right")
    ax.set_ylabel("Mean Undiscounted Reward")
    ax.set_title("Reward by Choice-Set Position")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    savefig(fig, os.path.join(out_dir, "figures", "reward_gradient.pdf"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate LaTeX tables and figures from dataset stats CSVs."
    )
    parser.add_argument("--stats-dir",  type=str, default="results/dataset_stats",
                        help="Directory containing per-env CSV files.")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Root output directory for tables/ and figures/.")
    args = parser.parse_args()

    print(f"Loading stats from: {args.stats_dir}")
    df = load_stats(args.stats_dir)
    print(f"  {len(df)} environments loaded: {df['env'].tolist()}\n")

    tab_dir = os.path.join(args.output_dir, "tables")
    fig_dir = args.output_dir

    # LaTeX tables
    print("Generating LaTeX tables...")
    latex_pool_table(df,              os.path.join(tab_dir, "pool_stats.tex"))
    latex_demo_table(df,              os.path.join(tab_dir, "demo_stats.tex"))
    latex_quality_gradient_table(df,  os.path.join(tab_dir, "quality_gradient.tex"))

    # Figures
    print("\nGenerating figures...")
    fig_pool_rewards(df,      fig_dir)
    fig_quality_gradient(df,  fig_dir)
    fig_adv_gap(df,           fig_dir)
    fig_action_diversity(df,  fig_dir)
    fig_reward_gradient(df,   fig_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
