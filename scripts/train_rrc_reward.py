"""
Train RRC reward ensembles for one (task, feedback type) and write the diagnostics report.

    python scripts/train_rrc_reward.py --config configs/rrc_sac/pref.yaml --task mw_door-open-v2 --seeds 0 1 2
    python scripts/train_rrc_reward.py --config configs/rrc_sac/pref.yaml --task mw_door-open-v2 --seeds 0 --pilot
    python scripts/train_rrc_reward.py --config configs/rrc_sac/pref.yaml --task mw_door-open-v2 --seeds 0 1 2 --untrained

Outputs, per seed, in <runs_root>/<task>/<type>/seed<s>/reward/ (or .../untrained/seed<s>/reward/):
    best.pt  last.pt  epoch_XXX.pt  [epoch3x.pt best_ens.pt with --pilot]
    split.npz  metrics.csv  train_summary.json  report.json  config.yaml  complete
Label consistency does not depend on the reward model, so it is computed once per (task, type)
into <runs_root>/<task>/<type>/label_consistency.json.
Use --overlay configs/rrc_sac/local.yaml for the local data mirrors.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.reward_learning.data import PoolIndex, load_choice_set, load_rrc_config, make_episode_split, resolve
from research.reward_learning.diagnostics import heldout_report, label_consistency, offdist_report, write_json
from research.reward_learning.ensemble import RewardEnsemble
from research.reward_learning.train import export_untrained, train_reward_ensemble


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--overlay", nargs="*", default=[])
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--runs_root", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--pilot", action="store_true", help="no early stopping; keep every epoch to export the 3x checkpoint")
    ap.add_argument("--untrained", action="store_true", help="negative control: random-init ensembles (no training)")
    ap.add_argument("--skip_label_consistency", action="store_true")
    ap.add_argument("--skip_offdist", action="store_true")
    ap.add_argument("--force", action="store_true", help="retrain even if a `complete` sentinel exists")
    args = ap.parse_args()

    cfg = load_rrc_config(args.config, args.overlay, args.set)
    if args.runs_root:
        cfg["paths"]["runs_root"] = args.runs_root
    task, ftype = args.task, cfg["type"]
    runs_root = resolve(cfg["paths"]["runs_root"], task, cfg)
    arm = "untrained" if args.untrained else ftype
    t0 = time.time()

    pool = PoolIndex(resolve(cfg["paths"]["pool_path"], task, cfg), expected_episodes=cfg["data"].get("expected_episodes"))
    cs = load_choice_set(cfg, task, pool)
    split = make_episode_split(cs.episodes, pool.n_episodes, float(cfg["data"]["val_frac"]), int(cfg["data"]["split_seed"]))
    print(f"[rrc] {task} {ftype}: {len(cs)} acts ({cs.n_options} options), train {len(split['train_idx'])}, "
          f"val {len(split['val_idx'])} from {len(split['val_episodes'])}/{pool.n_episodes} episodes, "
          f"loaded in {time.time() - t0:.0f}s  meta={json.dumps({k: v for k, v in cs.meta.items() if k != 'stop_index'}, default=str)}")

    if not args.untrained and not args.skip_label_consistency:
        lc_path = os.path.join(runs_root, task, ftype, "label_consistency.json")
        if not os.path.exists(lc_path) or args.force:
            lc = label_consistency(cs, split, cfg, task, args.device)
            write_json(lc_path, lc)
            print(f"[rrc] label consistency: {json.dumps(lc, default=float)}")

    cache = resolve(cfg["paths"]["offdist_cache"], task, cfg)
    for seed in args.seeds:
        out = os.path.join(runs_root, task, arm, f"seed{seed}", "reward")
        if os.path.exists(os.path.join(out, "complete")) and not args.force:
            print(f"[rrc] {out} complete, skipping")
            continue
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, "config.yaml"), "w") as f:
            yaml.safe_dump(dict(cfg, task=task, seed=seed, pilot=args.pilot, untrained=args.untrained), f)
        np.savez(os.path.join(out, "split.npz"), **{k: v for k, v in split.items()},
                 row_index=cs.row_index, episodes=cs.episodes)
        if args.untrained:
            export_untrained(cs, split, cfg, out, seed, args.device)
            exports = ["best"]
        else:
            summary = train_reward_ensemble(cs, split, cfg, out, seed, args.device, pilot=args.pilot)
            exports = ["best", "last"] + (["epoch3x", "best_ens"] if args.pilot else [])
            print(f"[rrc] seed {seed}: {summary['epochs_run']} epochs, best epochs {summary['best_epochs']}, "
                  f"ensemble best epoch {summary['ens_best_epoch']} (val NLL {summary['ens_best_val_nll']:.4f})")
        report = dict(task=task, type=ftype, arm=arm, seed=seed, exports={})
        for name in exports:
            model = RewardEnsemble.load(os.path.join(out, f"{name}.pt"), args.device)
            rep = dict(member_epochs=model.extra.get("member_epochs"), heldout=heldout_report(model, cs, split, cfg, args.device))
            if not args.skip_offdist:
                if os.path.exists(cache):
                    rep["offdist"] = offdist_report(model, cs, cache, cfg)
                else:
                    print(f"[rrc] WARNING: no off-distribution cache at {cache}; run scripts/collect_rrc_offdist.py first")
            report["exports"][name] = rep
        write_json(os.path.join(out, "report.json"), report)
        open(os.path.join(out, "complete"), "w").close()
        b = report["exports"]["best"]["heldout"]
        print(f"[rrc] seed {seed} best: val NLL {b['val'].get('nll_ens', float('nan')):.4f} acc {b['val'].get('acc_ens', float('nan')):.3f} "
              f"gt corr {b['gt_corr_val']['all']['pearson']:.3f}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
