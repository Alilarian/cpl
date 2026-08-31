"""
Plot oracle SAC eval/success curves (vs. training step) from local log.csv
files, one line per MetaWorld env.

Run this on CHPC where the oracle_sac_1m run dirs live — reads
<oracle-runs-base>/<env>/log.csv via select_checkpoints_by_success.load_success_rows,
and (optionally) overlays which steps have on-disk model_<step>.pt checkpoints
and which fall inside a target success band (e.g. the 30-50% range used for
build_trajectory_pool.py --checkpoint-steps).

Usage
-----
python scripts/plot_oracle_success_curves.py
python scripts/plot_oracle_success_curves.py \\
    --oracle-runs-base /scratch/general/vast/u1472210/oracle_sac_1m \\
    --envs mw_button-press-v2 mw_door-open-v2 mw_drawer-open-v2 mw_plate-slide-v2 \\
    --band-min 0.3 --band-max 0.5 \\
    --out results/oracle_success_curves.pdf
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from select_checkpoints_by_success import load_success_rows, existing_checkpoint_steps

ENVS_ALL = [
    "mw_bin-picking-v2",
    "mw_button-press-v2",
    "mw_door-open-v2",
    "mw_drawer-open-v2",
    "mw_plate-slide-v2",
    "mw_sweep-into-v2",
]

COLORS = {
    "mw_bin-picking-v2":  "#0072B2",
    "mw_button-press-v2": "#E69F00",
    "mw_door-open-v2":    "#009E73",
    "mw_drawer-open-v2":  "#D55E00",
    "mw_plate-slide-v2":  "#CC79A7",
    "mw_sweep-into-v2":   "#56B4E9",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-runs-base", type=str,
                        default="/scratch/general/vast/u1472210/oracle_sac_1m")
    parser.add_argument("--envs", nargs="+", default=ENVS_ALL)
    parser.add_argument("--band-min", type=float, default=0.3,
                        help="Lower bound of the success band to shade (default: 0.3). "
                             "Pass --band-min -1 to disable shading.")
    parser.add_argument("--band-max", type=float, default=0.5,
                        help="Upper bound of the success band to shade (default: 0.5).")
    parser.add_argument("--checkpoint-interval", type=int, default=20000,
                        help="Used only to mark which on-disk checkpoints exist "
                             "(default: 20000, matching trainer_kwargs.checkpoint_freq).")
    parser.add_argument("--out", type=str, default="results/oracle_success_curves.pdf")
    args = parser.parse_args()

    fig, ax = plt.subplots(figsize=(9, 5.5))

    if args.band_min >= 0:
        ax.axhspan(args.band_min, args.band_max, color="gray", alpha=0.12, zorder=1,
                    label=f"{args.band_min:.0%}–{args.band_max:.0%} band")

    any_loaded = False
    for env in args.envs:
        run_dir = os.path.join(args.oracle_runs_base, env)
        color = COLORS.get(env, None)

        try:
            rows = load_success_rows(run_dir)
        except (FileNotFoundError, KeyError) as e:
            print(f"  [skip] {env}: {e}")
            continue
        if not rows:
            print(f"  [skip] {env}: log.csv has no eval/success rows")
            continue

        steps = [s for s, _ in rows]
        succs = [v for _, v in rows]
        ax.plot(steps, succs, "-", color=color, linewidth=1.6, label=env, zorder=3)

        ckpt_steps = existing_checkpoint_steps(run_dir)
        ckpt_pts = [(s, v) for s, v in rows
                    if s % args.checkpoint_interval == 0 and s in ckpt_steps]
        if ckpt_pts:
            cx = [p[0] for p in ckpt_pts]
            cy = [p[1] for p in ckpt_pts]
            ax.scatter(cx, cy, color=color, s=18, zorder=4,
                       edgecolor="white", linewidth=0.5)

        any_loaded = True
        print(f"  {env}: {len(rows)} eval rows, {len(ckpt_pts)} on-disk checkpoints, "
              f"last success={succs[-1]:.3f}, max={max(succs):.3f}")

    if not any_loaded:
        print("No envs had loadable log.csv data — nothing to plot.")
        return

    ax.set_xlabel("Training step", fontsize=10)
    ax.set_ylabel("Eval success rate", fontsize=10)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Oracle SAC — eval success rate vs. training step", fontsize=11)
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)
    ax.legend(fontsize=8, loc="lower right", framealpha=0.9, ncol=2)

    fig.tight_layout()
    out_path = args.out
    if not os.path.isabs(out_path):
        out_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved → {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
