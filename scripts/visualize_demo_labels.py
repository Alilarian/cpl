"""
Visualize a demonstrative example (demo + K-1 counterfactuals) from demo_labels datasets.

demo_labels structure (N, K, T, ...):
  obs          : (N, K, T, 35)   index 0 = demo (best), 1..K-1 = counterfactuals sorted descending
  reward       : (N, K, T)
  adv_scores   : (N, K)          rl_sum values, sorted descending
  checkpoint_step: (N,)

From trim_mw_obs: obs[0:3] = end-effector XYZ, obs[4:7] = object XYZ

Output figure layout per sample:
  ┌─────────────────────────────┬──────────────────────────┐
  │  XY Path (all K overlaid)   │  Reward curves (K rows)  │
  │  EE=solid, Obj=dashed       │  per-step + cumulative   │
  │  color: demo=gold, CF=blues │                          │
  ├─────────────────────────────┴──────────────────────────┤
  │  Per-trajectory XY panels (1 per K, small)             │
  └────────────────────────────────────────────────────────┘

Usage:
    # Random sample from each env in demo_labels/
    python scripts/visualize_demo_labels.py \\
        --demo-labels-dir datasets/mw/demo_labels

    # Specific env, specific sample index
    python scripts/visualize_demo_labels.py \\
        --demo-labels-dir datasets/mw/demo_labels \\
        --env mw_drawer-open-v2 \\
        --sample 42

    # Custom output dir
    python scripts/visualize_demo_labels.py \\
        --demo-labels-dir datasets/mw/demo_labels \\
        --output results/demo_vis
"""

import argparse
import os
import glob

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D


# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------

# Demo = gold, counterfactuals fade from teal → blue → grey
PALETTE = [
    "#FFD700",  # 0: demo — gold
    "#00BFFF",  # 1: CF-1 — deep sky blue
    "#1E90FF",  # 2: CF-2 — dodger blue
    "#6A5ACD",  # 3: CF-3 — slate blue
    "#9370DB",  # 4: CF-4 — medium purple
    "#C0A0C0",  # 5: CF-5 — muted pink/grey
    "#808080",  # 6: CF-6 — grey (worst)
]

ALPHA_FULL  = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4]   # alpha per rank


def rank_label(k):
    return "Demo" if k == 0 else f"CF-{k}"


# ---------------------------------------------------------------------------
# Core figure builder
# ---------------------------------------------------------------------------

def build_figure(obs_K, reward_K, adv_scores, env_name, sample_idx, K):
    """
    Build and return a matplotlib Figure visualizing one demonstrative sample.

    Args:
        obs_K:      (K, T, obs_dim) float32
        reward_K:   (K, T)          float32
        adv_scores: (K,)            float32
        env_name:   str
        sample_idx: int
        K:          int
    """
    T = obs_K.shape[1]
    timesteps = np.arange(T)

    # Extract EE and object positions
    ee_K  = obs_K[:, :, 0:3]   # (K, T, 3)
    obj_K = obs_K[:, :, 4:7]   # (K, T, 3)

    # Cumulative returns
    cum_K = np.cumsum(reward_K, axis=1)   # (K, T)

    # -----------------------------------------------------------------------
    # Layout:
    #  Row 0 (tall): [overview XY path | reward curves stacked]
    #  Row 1 (short): [K small individual XY panels side by side]
    # -----------------------------------------------------------------------
    fig = plt.figure(figsize=(16, 10), facecolor="#0d0d1a")

    gs_outer = gridspec.GridSpec(
        2, 1, figure=fig,
        height_ratios=[2.2, 1.0],
        hspace=0.35,
    )

    # Top row: overview XY (left) + reward curves (right)
    gs_top = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=gs_outer[0],
        width_ratios=[1.2, 1.0],
        wspace=0.3,
    )

    ax_xy   = fig.add_subplot(gs_top[0])
    ax_rew  = fig.add_subplot(gs_top[1])

    # Bottom row: K small individual panels
    gs_bot = gridspec.GridSpecFromSubplotSpec(
        1, K, subplot_spec=gs_outer[1],
        wspace=0.25,
    )
    ax_indiv = [fig.add_subplot(gs_bot[k]) for k in range(K)]

    dark_bg = "#0d0d1a"
    panel_bg = "#12122a"

    for ax in [ax_xy, ax_rew] + ax_indiv:
        ax.set_facecolor(panel_bg)
        ax.tick_params(colors="#aaaaaa", labelsize=6)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333355")

    # -----------------------------------------------------------------------
    # Overview XY path (all K overlaid)
    # -----------------------------------------------------------------------
    ax_xy.set_title("XY End-Effector + Object Paths (all trajectories)",
                    color="white", fontsize=9, pad=6)

    all_x = np.concatenate([ee_K[:, :, 0].ravel(), obj_K[:, :, 0].ravel()])
    all_y = np.concatenate([ee_K[:, :, 1].ravel(), obj_K[:, :, 1].ravel()])
    pad = 0.05
    ax_xy.set_xlim(all_x.min() - pad, all_x.max() + pad)
    ax_xy.set_ylim(all_y.min() - pad, all_y.max() + pad)

    for k in range(K - 1, -1, -1):   # draw worst first so demo is on top
        c  = PALETTE[k]
        al = ALPHA_FULL[k]
        lw = 2.0 if k == 0 else 1.2

        ax_xy.plot(ee_K[k, :, 0],  ee_K[k, :, 1],
                   color=c, alpha=al, linewidth=lw, solid_capstyle="round")
        ax_xy.plot(obj_K[k, :, 0], obj_K[k, :, 1],
                   color=c, alpha=al * 0.7, linewidth=lw,
                   linestyle="--", solid_capstyle="round")
        # Start marker (circle) and end marker (star)
        ax_xy.scatter(ee_K[k, 0, 0],  ee_K[k, 0, 1],
                      color=c, s=30 if k == 0 else 15,
                      alpha=al, zorder=5, marker="o")
        ax_xy.scatter(ee_K[k, -1, 0], ee_K[k, -1, 1],
                      color=c, s=50 if k == 0 else 25,
                      alpha=al, zorder=5, marker="*")

    ax_xy.set_xlabel("X", color="#aaaaaa", fontsize=7)
    ax_xy.set_ylabel("Y", color="#aaaaaa", fontsize=7)

    legend_handles = [
        Line2D([0], [0], color=PALETTE[k], linewidth=2 if k == 0 else 1.2,
               alpha=ALPHA_FULL[k],
               label=f"{rank_label(k)}  adv={adv_scores[k]:.2f}")
        for k in range(K)
    ]
    legend_handles += [
        Line2D([0], [0], color="white", linewidth=1.5, linestyle="-",
               label="End-effector"),
        Line2D([0], [0], color="white", linewidth=1.5, linestyle="--",
               label="Object"),
    ]
    ax_xy.legend(handles=legend_handles, fontsize=6, loc="upper right",
                 framealpha=0.35, facecolor="#0d0d1a", edgecolor="none",
                 labelcolor="white")

    # -----------------------------------------------------------------------
    # Reward curves (per-step + cumulative)
    # -----------------------------------------------------------------------
    ax_rew.set_title("Reward over time", color="white", fontsize=9, pad=6)
    ax2_rew = ax_rew.twinx()
    ax2_rew.set_facecolor(panel_bg)
    ax2_rew.tick_params(colors="#aaaaaa", labelsize=6)

    for k in range(K):
        c  = PALETTE[k]
        al = ALPHA_FULL[k]
        lw = 1.8 if k == 0 else 1.0
        ax_rew.plot(timesteps, reward_K[k],
                    color=c, alpha=al * 0.6, linewidth=lw)
        ax2_rew.plot(timesteps, cum_K[k],
                     color=c, alpha=al, linewidth=lw,
                     linestyle="--")

    ax_rew.set_xlabel("Timestep", color="#aaaaaa", fontsize=7)
    ax_rew.set_ylabel("Per-step reward", color="#aaaaaa", fontsize=7)
    ax2_rew.set_ylabel("Cumulative return", color="#aaaaaa", fontsize=7)
    ax_rew.axhline(0, color="#444466", linewidth=0.6, linestyle=":")

    rew_legend = [
        Line2D([0], [0], color="white", linewidth=1.5, linestyle="-",
               label="Per-step reward"),
        Line2D([0], [0], color="white", linewidth=1.5, linestyle="--",
               label="Cumulative return"),
    ]
    ax_rew.legend(handles=rew_legend, fontsize=6, loc="upper left",
                  framealpha=0.35, facecolor="#0d0d1a", edgecolor="none",
                  labelcolor="white")

    # -----------------------------------------------------------------------
    # Individual XY panels (one per trajectory)
    # -----------------------------------------------------------------------
    for k in range(K):
        ax = ax_indiv[k]
        c  = PALETTE[k]
        al = ALPHA_FULL[k]

        ax.set_xlim(all_x.min() - pad, all_x.max() + pad)
        ax.set_ylim(all_y.min() - pad, all_y.max() + pad)

        # Faint full paths of other trajectories for context
        for j in range(K):
            if j != k:
                ax.plot(ee_K[j, :, 0], ee_K[j, :, 1],
                        color="#444466", alpha=0.3, linewidth=0.6)

        ax.plot(ee_K[k, :, 0],  ee_K[k, :, 1],
                color=c, alpha=al, linewidth=1.8, solid_capstyle="round")
        ax.plot(obj_K[k, :, 0], obj_K[k, :, 1],
                color=c, alpha=al * 0.7, linewidth=1.3, linestyle="--",
                solid_capstyle="round")

        ax.scatter(ee_K[k, 0, 0],  ee_K[k, 0, 1],
                   color=c, s=20, zorder=5, marker="o")
        ax.scatter(ee_K[k, -1, 0], ee_K[k, -1, 1],
                   color=c, s=35, zorder=5, marker="*")

        ret = float(reward_K[k].sum())
        ax.set_title(
            f"{rank_label(k)}\nadv={adv_scores[k]:.2f}\nR={ret:.2f}",
            color=c, fontsize=6.5, pad=3,
        )
        ax.set_xlabel("X", color="#aaaaaa", fontsize=5)
        ax.set_ylabel("Y", color="#aaaaaa", fontsize=5)

    # -----------------------------------------------------------------------
    # Figure title
    # -----------------------------------------------------------------------
    fig.suptitle(
        f"{env_name}  |  sample {sample_idx}  |  "
        f"Demo adv={adv_scores[0]:.2f}  vs  Worst adv={adv_scores[-1]:.2f}  "
        f"(gap={adv_scores[0] - adv_scores[-1]:.2f})",
        color="white", fontsize=11, y=0.99,
    )

    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Visualize demonstrative examples from demo_labels datasets."
    )
    parser.add_argument("--demo-labels-dir", type=str,
                        default="datasets/mw/demo_labels",
                        help="Root dir containing <env>/demo_labels_K*.npz files")
    parser.add_argument("--env", type=str, default=None,
                        help="Specific env to visualize (default: all found)")
    parser.add_argument("--sample", type=str, default="random",
                        help="Sample index (int) or 'random' (default: random)")
    parser.add_argument("--output", type=str, default="results/demo_vis",
                        help="Output directory (default: results/demo_vis)")
    parser.add_argument("--dpi", type=int, default=150,
                        help="PNG DPI (default: 150)")
    args = parser.parse_args()

    # Find datasets
    if args.env:
        pattern = os.path.join(args.demo_labels_dir, args.env, "demo_labels_K*.npz")
    else:
        pattern = os.path.join(args.demo_labels_dir, "*", "demo_labels_K*.npz")

    paths = sorted(glob.glob(pattern))
    if not paths:
        print(f"No demo_labels files found at: {pattern}")
        return

    os.makedirs(args.output, exist_ok=True)

    for path in paths:
        env_name = os.path.basename(os.path.dirname(path))
        fname    = os.path.basename(path)   # demo_labels_K7.npz
        K_str    = fname.replace("demo_labels_K", "").replace(".npz", "")
        K        = int(K_str)

        print(f"\n{'='*60}")
        print(f"Env : {env_name}  ({fname})")

        data = np.load(path, allow_pickle=True)
        N = data["obs"].shape[0]

        if args.sample == "random":
            sample_idx = np.random.randint(0, N)
            print(f"Random sample index: {sample_idx}  (N={N})")
        else:
            sample_idx = int(args.sample)

        obs_K     = data["obs"][sample_idx]         # (K, T, obs_dim)
        reward_K  = data["reward"][sample_idx]      # (K, T)
        adv_scores = data["adv_scores"][sample_idx] # (K,)
        ckpt_step  = int(data["checkpoint_step"][sample_idx])

        print(f"Checkpoint step : {ckpt_step}")
        for k in range(K):
            ret = float(reward_K[k].sum())
            print(f"  {rank_label(k):6s}  adv={adv_scores[k]:8.3f}  ret={ret:8.3f}")

        fig = build_figure(obs_K, reward_K, adv_scores, env_name, sample_idx, K)

        out_path = os.path.join(
            args.output, f"{env_name}_sample{sample_idx}.png"
        )
        fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"Saved → {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
