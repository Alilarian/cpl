"""
Visualize the V* and A* tables produced by compute_advantage_vi.py.

Three panels
------------
1. V*(s)          soft-optimal state value — shows where the agent "wants to be"
2. π*(s) quiver   direction of the advantage-maximizing action at each state
3. max_a A*(s,a)  best-action advantage — how much better the greedy action is
                  than the average at each state

Usage
-----
python scripts/visualize_advantage_vi.py \
    --npz runs/pm_sac_oracle/advantage_vi_N100.npz \
    --out results/vi_values.png
"""

import argparse
import importlib.util
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_module(dotted_name, rel_path):
    path = os.path.join(_ROOT, rel_path)
    spec = importlib.util.spec_from_file_location(dotted_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[dotted_name] = mod
    spec.loader.exec_module(mod)
    return mod


_avi_mod = _load_module("research.utils.advantage_vi", "research/utils/advantage_vi.py")
_pm_mod  = _load_module("research.envs.pointmass",     "research/envs/pointmass.py")

AdvantageVI     = _avi_mod.AdvantageVI
PointMassGymEnv = _pm_mod.PointMassGymEnv


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def add_arena(ax, env, alpha=0.35, zorder=5):
    """Overlay goal / trap circles and a unit-square border."""
    ax.add_patch(plt.Circle(env.goal, env.goal_dist_thresh,
                            color="green", alpha=alpha, zorder=zorder))
    ax.plot(*env.goal, "g^", ms=7, zorder=zorder + 1)

    ax.add_patch(plt.Circle(env.trap, env.trap_dist_thresh,
                            color="red", alpha=alpha, zorder=zorder))
    ax.plot(*env.trap, "rx", ms=7, mew=2, zorder=zorder + 1)

    for spine in ["bottom", "left", "top", "right"]:
        ax.spines[spine].set_linewidth(1.2)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("x"); ax.set_ylabel("y")


def heatmap(ax, data, title, cmap, norm=None, cbar_label=""):
    """Display (N,N) data with x on horizontal axis, y on vertical."""
    # data[i,j] = value at x=xs[i], y=xs[j]
    # imshow row 0 → bottom when origin='lower'; .T maps (i,j)→(j,i)
    im = ax.imshow(
        data.T,
        origin="lower",
        extent=[0, 1, 0, 1],
        cmap=cmap,
        norm=norm,
        aspect="equal",
        interpolation="nearest",
    )
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=cbar_label)
    ax.set_title(title, fontsize=11)
    return im


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--npz", required=True,
        help="Path to advantage_vi_*.npz produced by compute_advantage_vi.py",
    )
    parser.add_argument(
        "--out", default="results/vi_values.png",
        help="Output PNG path",
    )
    parser.add_argument(
        "--quiver-stride", type=int, default=None,
        help="Grid sub-sampling for quiver (default: N//20, min 1)",
    )
    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # Load tables
    # -----------------------------------------------------------------------
    avi = AdvantageVI.load(args.npz)
    env = PointMassGymEnv()          # default goal/trap — must match training

    N  = avi.N
    xs = avi.xs                      # (N,) cell centres
    V  = avi.V                       # (N, N)
    A  = avi.A_star                  # (N, N, K)

    print(f"[viz] loaded  N={N}  K={A.shape[2]}  α={avi.alpha:.4f}")
    print(f"[viz] V  range [{V.min():.3f}, {V.max():.3f}]")
    print(f"[viz] A* range [{A.min():.3f}, {A.max():.3f}]")

    # -----------------------------------------------------------------------
    # Derived quantities
    # -----------------------------------------------------------------------
    max_adv = A.max(axis=2)          # (N, N)  best-action advantage

    # Greedy action index at every cell
    k_best = A.argmax(axis=2)        # (N, N)
    # Direction vectors for best action (Cartesian, in [-1,1]²)
    dx = avi.action_grid[k_best, 0]  # (N, N)
    dy = avi.action_grid[k_best, 1]

    # -----------------------------------------------------------------------
    # Shared color norms
    # -----------------------------------------------------------------------
    # V*: min→max, diverge at 0 so the zero-value region stands out
    v_abs = max(abs(float(V.min())), abs(float(V.max())), 1e-6)
    v_norm = TwoSlopeNorm(vmin=-v_abs, vcenter=0.0, vmax=v_abs)

    # A*: symmetric diverging norm around 0
    a_abs = max(abs(float(max_adv.min())), abs(float(max_adv.max())), 1e-6)
    a_norm = TwoSlopeNorm(vmin=-a_abs, vcenter=0.0, vmax=a_abs)

    # -----------------------------------------------------------------------
    # Figure
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f"Soft-optimal value & advantage  (N={N}, α={avi.alpha:.3f}, γ={avi.gamma:.2f})",
        fontsize=13, y=1.01,
    )

    # --- Panel 1: V*(s) ---
    ax = axes[0]
    heatmap(ax, V, "V*(s)  —  state value", cmap="RdYlGn", norm=v_norm,
            cbar_label="V*(s)")
    add_arena(ax, env)

    # --- Panel 2: greedy policy quiver ---
    ax = axes[1]
    # Background: V* for spatial context
    heatmap(ax, V, "π*(s)  —  greedy action direction", cmap="RdYlGn",
            norm=v_norm, cbar_label="V*(s)")
    add_arena(ax, env)

    stride = args.quiver_stride or max(1, N // 20)
    qi = np.arange(stride // 2, N, stride)
    qj = np.arange(stride // 2, N, stride)
    QI, QJ = np.meshgrid(qi, qj, indexing="ij")
    qx = xs[QI]           # (ns, ns)  x positions
    qy = xs[QJ]           # (ns, ns)  y positions
    qdx = dx[QI, QJ]      # (ns, ns)  best-action x component
    qdy = dy[QI, QJ]

    # Color arrows by advantage magnitude (warm = high advantage)
    adv_color = max_adv[QI, QJ].ravel()
    ax.quiver(
        qx, qy, qdx, qdy,
        adv_color,
        cmap="plasma",
        scale=25, width=0.004,
        alpha=0.85, zorder=6,
    )

    # --- Panel 3: max_a A*(s,a) ---
    ax = axes[2]
    heatmap(ax, max_adv, "max_a A*(s,a)  —  best-action advantage",
            cmap="RdYlGn", norm=a_norm, cbar_label="A*(s, a*)")
    add_arena(ax, env)

    # -----------------------------------------------------------------------
    # Legend
    # -----------------------------------------------------------------------
    legend_patches = [
        mpatches.Patch(color="green", alpha=0.5, label="goal region"),
        mpatches.Patch(color="red",   alpha=0.5, label="trap region"),
    ]
    axes[2].legend(handles=legend_patches, fontsize=8, loc="upper right")

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"[viz] saved → {args.out}")


if __name__ == "__main__":
    main()
