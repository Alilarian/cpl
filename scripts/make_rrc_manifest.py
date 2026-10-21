"""
Write SLURM array manifests (one line per array element) for the RRC + SAC stages.

    # off-distribution caches (one line per task)
    python scripts/make_rrc_manifest.py --stage offdist --tasks all --out slurm/manifests/rrc_sac/offdist.tsv
    # reward models: one line per (task, type), all seeds trained in that element
    python scripts/make_rrc_manifest.py --stage reward --tasks all --types all --seeds 0 \
        --pilot_task mw_door-open-v2 --pilot_types pref cum_estop --pilot_seeds 0 1 2 --out .../reward_c1.tsv
    # SAC: runs packed --pack per line, each run "task:arm:seed"
    python scripts/make_rrc_manifest.py --stage sac --tasks mw_door-open-v2 \
        --arms pref@best pref@3x cum_estop@best cum_estop@3x true_reward untrained --seeds 0 1 2 --pack 4 --out .../pilot.tsv

Line formats (tab-separated):
    offdist : <task>
    reward  : <task> <config> <seeds, space-separated> <extra flags>
    sac     : <task:arm:seed> [<task:arm:seed> ...]   (space-separated, up to --pack runs)
"""

import argparse
import os

TASKS = ["mw_button-press-v2", "mw_door-open-v2", "mw_drawer-open-v2", "mw_plate-slide-v2"]
TYPES = ["pref", "corr", "demo", "scalar", "credit", "cum_estop"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["offdist", "reward", "sac"], required=True)
    ap.add_argument("--tasks", nargs="+", default=["all"])
    ap.add_argument("--types", nargs="+", default=["all"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--pilot_task", default=None)
    ap.add_argument("--pilot_types", nargs="*", default=[])
    ap.add_argument("--pilot_seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--untrained_stats_type", default="pref", help="type whose data standardizes the untrained control")
    ap.add_argument("--arms", nargs="+", default=None, help="sac stage; default: <type>@best for --types + true_reward")
    ap.add_argument("--pack", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tasks = TASKS if args.tasks == ["all"] else args.tasks
    types = TYPES if args.types == ["all"] else args.types
    lines = []
    if args.stage == "offdist":
        lines = list(tasks)
    elif args.stage == "reward":
        for task in tasks:
            for t in types:
                pilot = task == args.pilot_task and t in args.pilot_types
                seeds = sorted(set(args.seeds) | set(args.pilot_seeds)) if pilot else args.seeds
                lines.append("\t".join([task, f"configs/rrc_sac/{t}.yaml", " ".join(map(str, seeds)), "--pilot" if pilot else ""]))
            if task == args.pilot_task:
                lines.append("\t".join([task, f"configs/rrc_sac/{args.untrained_stats_type}.yaml",
                                        " ".join(map(str, args.pilot_seeds)), "--untrained"]))
    else:
        arms = args.arms or [f"{t}@best" for t in types] + ["true_reward"]
        runs = [f"{task}:{arm}:{s}" for task in tasks for arm in arms for s in args.seeds]
        lines = [" ".join(runs[i : i + args.pack]) for i in range(0, len(runs), args.pack)]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {len(lines)} lines -> {args.out}  (submit with --array=1-{len(lines)})")


if __name__ == "__main__":
    main()
