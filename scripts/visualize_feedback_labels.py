"""
Visualize corrective, e-stop, and sequential e-stop feedback labels.

For each feedback type, samples one pair per environment and produces a
high-resolution PNG under results/<type>_vis/.

Feedback types and what is shown
---------------------------------
corr      : expert correction (index 0, gold) vs agent trajectory (index 1, blue)
            annotated with oracle adv_scores and improvement magnitude
estop     : halt prefix (index 0, gold) vs full trajectory (index 1, blue)
            red X marks stop time τ on XY path; vertical dashed line on reward plot
seq_estop : samples a pair where stop_event=1 (the actual intervention moment τ)
            stop segment (index 0, gold) vs continue segment (index 1, blue)
            both are h=10 step windows; annotated with decision timestep t

obs layout (from trim_mw_obs):
    obs[0:3]  = end-effector XYZ
    obs[4:7]  = object XYZ

Usage:
    # All envs, all feedback types
    python scripts/visualize_feedback_labels.py

    # Specific type and env
    python scripts/visualize_feedback_labels.py \\
        --type estop --env mw_drawer-open-v2 --sample 42

    # Custom data root and output dir
    python scripts/visualize_feedback_labels.py \\
        --data-root datasets/mw --output-root results --dpi 200
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

BG       = "#0d0d1a"
PANEL_BG = "#12122a"
GOLD     = "#FFD700"
BLUE     = "#4fc3f7"
RED      = "#FF6B6B"
GRID_CLR = "#222244"


def _style_ax(ax):
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors="#aaaaaa", labelsize=6)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333355")
    ax.grid(True, color=GRID_CLR, linewidth=0.4, linestyle=":")


def _legend(ax, handles, **kw):
    ax.legend(handles=handles, fontsize=6, framealpha=0.35,
              facecolor=BG, edgecolor="none", labelcolor="white", **kw)


# ---------------------------------------------------------------------------
# Core figure builder  (shared by all three feedback types)
# ---------------------------------------------------------------------------

def build_pair_figure(
    obs_pref, obs_npref,
    rew_pref, rew_npref,
    label_pref, label_npref,
    title,
    extra_annotations=None,   # callable(ax_xy, ax_rew, ax_pref, ax_npref) or None
):
    """
    Build a 16×10 figure comparing two trajectories.

    Layout
    ------
    Row 0 (tall) : [XY paths overlaid | Reward curves]
    Row 1 (short): [XY path preferred | XY path non-preferred]
    """
    T_p  = obs_pref.shape[0]
    T_np = obs_npref.shape[0]

    ee_p   = obs_pref[:,  0:3];  obj_p  = obs_pref[:,  4:7]
    ee_np  = obs_npref[:, 0:3];  obj_np = obs_npref[:, 4:7]
    cum_p  = np.cumsum(rew_pref)
    cum_np = np.cumsum(rew_npref)

    fig = plt.figure(figsize=(16, 10), facecolor=BG)
    fig.suptitle(title, color="white", fontsize=10, y=0.997)

    gs = gridspec.GridSpec(2, 1, figure=fig,
                           height_ratios=[2.2, 1.0], hspace=0.38)
    gs_top = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=gs[0], width_ratios=[1.1, 1.0], wspace=0.28)
    ax_xy  = fig.add_subplot(gs_top[0])
    ax_rew = fig.add_subplot(gs_top[1])

    gs_bot = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=gs[1], wspace=0.22)
    ax_pref  = fig.add_subplot(gs_bot[0])
    ax_npref = fig.add_subplot(gs_bot[1])

    for ax in [ax_xy, ax_rew, ax_pref, ax_npref]:
        _style_ax(ax)

    all_x = np.concatenate([ee_p[:, 0], ee_np[:, 0], obj_p[:, 0], obj_np[:, 0]])
    all_y = np.concatenate([ee_p[:, 1], ee_np[:, 1], obj_p[:, 1], obj_np[:, 1]])
    pad   = 0.05
    xlim  = (all_x.min() - pad, all_x.max() + pad)
    ylim  = (all_y.min() - pad, all_y.max() + pad)

    # ------------------------------------------------------------------
    # Top-left: overlaid XY paths
    # ------------------------------------------------------------------
    ax_xy.set_title("XY Paths — End-Effector (solid) + Object (dashed)",
                    color="white", fontsize=8.5, pad=5)
    ax_xy.set_xlim(*xlim); ax_xy.set_ylim(*ylim)
    ax_xy.set_xlabel("X", color="#aaaaaa", fontsize=7)
    ax_xy.set_ylabel("Y", color="#aaaaaa", fontsize=7)

    for ee, obj, col, lw, al in [
        (ee_np, obj_np, BLUE, 1.4, 0.75),
        (ee_p,  obj_p,  GOLD, 2.0, 1.00),
    ]:
        ax_xy.plot(ee[:, 0],  ee[:, 1],  color=col, lw=lw, alpha=al)
        ax_xy.plot(obj[:, 0], obj[:, 1], color=col, lw=lw * 0.8,
                   alpha=al * 0.65, linestyle="--")
        ax_xy.scatter(ee[0, 0],  ee[0, 1],  color=col, s=28, zorder=5, marker="o")
        ax_xy.scatter(ee[-1, 0], ee[-1, 1], color=col, s=50, zorder=5, marker="*")

    _legend(ax_xy, [
        Line2D([0], [0], color=GOLD, lw=2.0, label=label_pref),
        Line2D([0], [0], color=BLUE, lw=1.4, label=label_npref),
        Line2D([0], [0], color="white", lw=1.5, linestyle="-",  label="End-effector"),
        Line2D([0], [0], color="white", lw=1.5, linestyle="--", label="Object"),
    ], loc="upper right")

    # ------------------------------------------------------------------
    # Top-right: reward curves (per-step + cumulative)
    # ------------------------------------------------------------------
    ax_rew.set_title("Reward over Time", color="white", fontsize=8.5, pad=5)
    ax2 = ax_rew.twinx()
    ax2.set_facecolor(PANEL_BG)
    ax2.tick_params(colors="#aaaaaa", labelsize=6)

    ax_rew.plot(np.arange(T_p),  rew_pref,  color=GOLD, lw=1.8, alpha=0.65)
    ax_rew.plot(np.arange(T_np), rew_npref, color=BLUE, lw=1.2, alpha=0.65)
    ax2.plot(np.arange(T_p),  cum_p,  color=GOLD, lw=1.8, linestyle="--")
    ax2.plot(np.arange(T_np), cum_np, color=BLUE, lw=1.2, linestyle="--")
    ax_rew.axhline(0, color="#444466", lw=0.6, linestyle=":")

    ax_rew.set_xlabel("Timestep", color="#aaaaaa", fontsize=7)
    ax_rew.set_ylabel("Per-step reward",   color="#aaaaaa", fontsize=7)
    ax2.set_ylabel("Cumulative return",    color="#aaaaaa", fontsize=7)

    _legend(ax_rew, [
        Line2D([0], [0], color="white", lw=1.5, linestyle="-",  label="Per-step"),
        Line2D([0], [0], color="white", lw=1.5, linestyle="--", label="Cumulative"),
        Line2D([0], [0], color=GOLD, lw=2.0, label=label_pref),
        Line2D([0], [0], color=BLUE, lw=1.4, label=label_npref),
    ], loc="upper left")

    # ------------------------------------------------------------------
    # Bottom: individual XY panels
    # ------------------------------------------------------------------
    for ax, ee, obj, col, lbl, rew, other_ee in [
        (ax_pref,  ee_p,  obj_p,  GOLD, label_pref,  rew_pref,  ee_np),
        (ax_npref, ee_np, obj_np, BLUE, label_npref, rew_npref, ee_p),
    ]:
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.set_xlabel("X", color="#aaaaaa", fontsize=6)
        ax.set_ylabel("Y", color="#aaaaaa", fontsize=6)
        ax.plot(other_ee[:, 0], other_ee[:, 1],
                color="#444466", alpha=0.3, lw=0.8)
        ax.plot(ee[:, 0],  ee[:, 1],  color=col, lw=2.0, alpha=0.95)
        ax.plot(obj[:, 0], obj[:, 1], color=col, lw=1.4, alpha=0.65, linestyle="--")
        ax.scatter(ee[0, 0],  ee[0, 1],  color=col, s=25, zorder=5, marker="o")
        ax.scatter(ee[-1, 0], ee[-1, 1], color=col, s=45, zorder=5, marker="*")
        ax.set_title(f"{lbl}\nR={rew.sum():.3f}", color=col, fontsize=7, pad=3)

    if extra_annotations is not None:
        extra_annotations(ax_xy, ax_rew, ax_pref, ax_npref)

    return fig


# ---------------------------------------------------------------------------
# Corrective feedback
# ---------------------------------------------------------------------------

def visualize_corr(path, env_name, sample_idx, out_dir, dpi):
    data = np.load(path)
    N    = data["obs"].shape[0]
    if sample_idx is None:
        sample_idx = int(np.random.randint(0, N))

    obs    = data["obs"][sample_idx]         # (2, T, 35)
    reward = data["reward"][sample_idx]      # (2, T)
    adv    = data["adv_scores"][sample_idx]  # (2,)
    improv = float(data["improvement"][sample_idx])

    ret_corr  = float(reward[0].sum())
    ret_agent = float(reward[1].sum())

    label_pref  = f"Expert correction  adv={adv[0]:.3f}  R={ret_corr:.2f}"
    label_npref = f"Agent trajectory   adv={adv[1]:.3f}  R={ret_agent:.2f}"
    title = (f"{env_name}  |  Corrective Feedback  |  sample {sample_idx}  |  "
             f"improvement={improv:.3f}  Δadv={adv[0]-adv[1]:.3f}")

    fig = build_pair_figure(
        obs[0], obs[1], reward[0], reward[1],
        label_pref, label_npref, title,
    )
    out = os.path.join(out_dir, f"{env_name}_sample{sample_idx}.png")
    fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"    Saved → {out}")


# ---------------------------------------------------------------------------
# E-stop feedback
# ---------------------------------------------------------------------------

def visualize_estop(path, env_name, sample_idx, out_dir, dpi):
    data = np.load(path)
    N    = data["obs"].shape[0]
    if sample_idx is None:
        sample_idx = int(np.random.randint(0, N))

    obs       = data["obs"][sample_idx]      # (2, T, 35)
    reward    = data["reward"][sample_idx]   # (2, T)
    stop_time = int(data["stop_time"][sample_idx])

    T          = obs.shape[1]
    ret_halt   = float(reward[0, :stop_time + 1].sum())
    ret_full   = float(reward[1].sum())

    label_pref  = f"Halt prefix  τ={stop_time}  R={ret_halt:.2f}"
    label_npref = f"Full traj    T={T}           R={ret_full:.2f}"
    title = (f"{env_name}  |  E-stop Feedback  |  sample {sample_idx}  |  "
             f"stop time τ={stop_time}/{T-1}  ΔR={ret_halt-ret_full:.3f}")

    ee_halt = obs[0, :, 0:3]

    def annotations(ax_xy, ax_rew, ax_pref, ax_npref):
        # Vertical line at τ on reward plot
        for ax in (ax_rew,):
            ax.axvline(stop_time, color=RED, lw=1.4, linestyle="--", alpha=0.85)
            ax.text(stop_time + 0.3, ax.get_ylim()[1] * 0.90,
                    f"τ={stop_time}", color=RED, fontsize=7)
        # Red X at halt position on XY panels
        for ax in (ax_xy, ax_pref):
            ax.scatter(ee_halt[stop_time, 0], ee_halt[stop_time, 1],
                       color=RED, s=90, zorder=9, marker="X",
                       label=f"Stop τ={stop_time}")
        _legend(ax_xy, [
            Line2D([0], [0], color=GOLD, lw=2.0, label=label_pref),
            Line2D([0], [0], color=BLUE, lw=1.4, label=label_npref),
            Line2D([0], [0], color="white", lw=1.5, linestyle="-",  label="End-effector"),
            Line2D([0], [0], color="white", lw=1.5, linestyle="--", label="Object"),
            Line2D([0], [0], color=RED, lw=0, marker="X", markersize=7,
                   label=f"Stop τ={stop_time}"),
        ], loc="upper right")

    fig = build_pair_figure(
        obs[0], obs[1], reward[0], reward[1],
        label_pref, label_npref, title,
        extra_annotations=annotations,
    )
    out = os.path.join(out_dir, f"{env_name}_sample{sample_idx}.png")
    fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"    Saved → {out}")


# ---------------------------------------------------------------------------
# Sequential e-stop feedback
# ---------------------------------------------------------------------------

def visualize_seq_estop(path, env_name, sample_idx, out_dir, dpi):
    """
    Picks a row where stop_event=1 — the actual intervention moment τ.
      index 0 = stop segment   (preferred, gold)  — h steps ending at τ
      index 1 = continue segment (non-pref, blue) — h steps continuing past τ

    sample_idx indexes into the filtered stop_event=1 rows.
    """
    data       = np.load(path)
    stop_event = data["stop_event"]          # (M,)
    stop_rows  = np.where(stop_event == 1)[0]
    if len(stop_rows) == 0:
        print(f"    WARNING: no stop_event=1 rows; using random pair.")
        stop_rows = np.arange(len(stop_event))

    if sample_idx is None:
        row = int(np.random.choice(stop_rows))
    else:
        row = int(stop_rows[sample_idx % len(stop_rows)])

    obs      = data["obs"][row]        # (2, h, 35)
    reward   = data["reward"][row]     # (2, h)
    timestep = int(data["timestep"][row])
    traj_idx = int(data["traj_idx"][row])
    h        = obs.shape[1]

    ret_stop = float(reward[0].sum())
    ret_cont = float(reward[1].sum())

    label_pref  = f"Stop segment    R={ret_stop:.2f}"
    label_npref = f"Continue segment R={ret_cont:.2f}"
    title = (f"{env_name}  |  Sequential E-stop  |  "
             f"traj {traj_idx}  t={timestep}  h={h}  "
             f"(row {row})  ΔR={ret_stop-ret_cont:.3f}")

    fig = build_pair_figure(
        obs[0], obs[1], reward[0], reward[1],
        label_pref, label_npref, title,
    )
    out = os.path.join(out_dir, f"{env_name}_traj{traj_idx}_t{timestep}.png")
    fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"    Saved → {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

FEEDBACK_TYPES = {
    "corr":      ("corr_labels",      "corr_labels.npz",      visualize_corr),
    "estop":     ("estop_labels",     "estop_labels.npz",      visualize_estop),
    "seq_estop": ("seq_estop_labels", "seq_estop_labels.npz",  visualize_seq_estop),
}

ENVS = [
    "mw_bin-picking-v2",
    "mw_button-press-v2",
    "mw_door-open-v2",
    "mw_drawer-open-v2",
    "mw_plate-slide-v2",
]


def main():
    parser = argparse.ArgumentParser(
        description="Visualize corrective / e-stop / seq-e-stop feedback labels."
    )
    parser.add_argument("--type", type=str, default=None,
                        choices=list(FEEDBACK_TYPES.keys()),
                        help="Feedback type to visualize (default: all)")
    parser.add_argument("--env", type=str, default=None,
                        help="Specific env (default: all 5 MetaWorld envs)")
    parser.add_argument("--sample", type=int, default=None,
                        help="Sample index (default: random per env)")
    parser.add_argument("--data-root", type=str, default="datasets/mw",
                        help="Root containing <type>/<env>/ subdirs")
    parser.add_argument("--output-root", type=str, default="results",
                        help="Output root; saves to <output-root>/<type>_vis/")
    parser.add_argument("--dpi", type=int, default=200,
                        help="PNG resolution in DPI (default: 200)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    np.random.seed(args.seed)

    types_to_run = [args.type] if args.type else list(FEEDBACK_TYPES.keys())
    envs_to_run  = [args.env]  if args.env  else ENVS

    for ftype in types_to_run:
        subdir, fname, vis_fn = FEEDBACK_TYPES[ftype]
        out_dir = os.path.join(args.output_root, f"{ftype}_vis")
        os.makedirs(out_dir, exist_ok=True)
        print(f"\n{'='*60}")
        print(f"Feedback type : {ftype}  →  {out_dir}")
        print(f"{'='*60}")

        for env in envs_to_run:
            path = os.path.join(args.data_root, subdir, env, fname)
            if not os.path.exists(path):
                print(f"  SKIP {env}: {path} not found")
                continue
            print(f"  {env}")
            vis_fn(path, env, args.sample, out_dir, args.dpi)

    print("\nDone.")


if __name__ == "__main__":
    main()
