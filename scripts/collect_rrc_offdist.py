"""
Cache off-distribution rollouts for one task (random policy + oracle SAC checkpoints by success).

    python scripts/collect_rrc_offdist.py --task mw_door-open-v2 [--overlay configs/rrc_sac/local.yaml]
        [--gt_policy_ckpts runs/rrc_sac/.../sac_best/final_model.pt ...]
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.reward_learning.data import load_rrc_config, resolve
from research.reward_learning.offdist import collect_offdist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--config", default="configs/rrc_sac/base.yaml")
    ap.add_argument("--overlay", nargs="*", default=[])
    ap.add_argument("--runs_root", default=None)
    ap.add_argument("--gt_policy_ckpts", nargs="*", default=[])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = load_rrc_config(args.config, args.overlay)
    if args.runs_root:
        cfg["paths"]["runs_root"] = args.runs_root
    out = resolve(cfg["paths"]["offdist_cache"], args.task, cfg)
    if os.path.exists(out) and not args.force:
        print(f"[offdist] {out} exists, skipping (use --force to rebuild)")
        return
    d = cfg["diagnostics"]
    meta = collect_offdist(resolve(cfg["paths"]["oracle_dir"], args.task, cfg), out,
                           n_random=int(d["offdist_random_episodes"]), n_per_ckpt=int(d["offdist_episodes_per_ckpt"]),
                           extra_ckpts=args.gt_policy_ckpts, seed=args.seed, device=args.device)
    with open(out.replace(".npz", "_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
