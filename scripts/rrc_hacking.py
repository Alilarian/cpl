"""
Reward-hacking metrics (docs/rrc_sac_plan.md section 3.6) for every RRC-SAC run under --runs_root.

    python scripts/rrc_hacking.py --runs_root runs/rrc_sac --tasks mw_door-open-v2

Writes <run>/hacking.json and <run>/hacking.png for each run directory that has a log.csv
and a reward_map.json.
"""

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml

from research.reward_learning.data import load_rrc_config, resolve
from research.reward_learning.diagnostics import write_json
from research.reward_learning.hacking import hacking_metrics, plot_hacking, references


def val_disagreement(reward_ckpt: str):
    """Held-out per-step ensemble std of this checkpoint, from the reward run's report.json."""
    if reward_ckpt is None:
        return None
    rep_path = os.path.join(os.path.dirname(reward_ckpt), "report.json")
    if not os.path.exists(rep_path):
        return None
    with open(rep_path) as f:
        rep = json.load(f)
    name = os.path.splitext(os.path.basename(reward_ckpt))[0]
    exp = rep.get("exports", {}).get(name)
    return None if exp is None else exp["heldout"]["steps_val"]["disagreement"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", default="runs/rrc_sac")
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--rrc_config", default="configs/rrc_sac/base.yaml")
    ap.add_argument("--overlay", nargs="*", default=[])
    ap.add_argument("--params", default="configs/rrc_sac/hacking.yaml")
    args = ap.parse_args()

    cfg = load_rrc_config(args.rrc_config, args.overlay)
    cfg["paths"]["runs_root"] = args.runs_root
    with open(args.params) as f:
        params = yaml.safe_load(f)
    tasks = args.tasks or sorted(d for d in os.listdir(args.runs_root) if not d.startswith("_"))
    for task in tasks:
        cache = resolve(cfg["paths"]["offdist_cache"], task, cfg)
        for rm in sorted(glob.glob(os.path.join(args.runs_root, task, "*", "seed*", "sac*", "reward_map.json"))):
            run = os.path.dirname(rm)
            if not os.path.exists(os.path.join(run, "log.csv")):
                continue
            with open(rm) as f:
                rmap = json.load(f)
            refs = references(cache, rmap["mode"], rmap.get("reward_ckpt"), (rmap["a"], rmap["b"]))
            m = hacking_metrics(run, refs, val_disagreement(rmap.get("reward_ckpt")), params)
            m.update(task=task, arm=rmap["arm"], seed=rmap["seed"])
            write_json(os.path.join(run, "hacking.json"), m)
            title = f"{task} {rmap['arm']} s{rmap['seed']}"
            plot_hacking(m, os.path.join(run, "hacking.png"), title)
            print(f"{title:45s} final success {m['final']['success']:.2f}  G {m['final']['G']:+.2f} (max {m['max_G']:+.2f})  "
                  f"flag={m['flag']['hacking']}")


if __name__ == "__main__":
    main()
