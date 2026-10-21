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
WandB (when logged in): one run per seed, name rrc_reward/<task>/<type>/s<seed>, group
rrc_reward/<task>/<type>, project $WANDB_PROJECT (default rrc-sac-2m) -- per-epoch train/held-out
NLL and accuracy per member and for the ensemble, plus the final diagnostics as the run summary.
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
from research.reward_learning import wandb_utils


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
    ap.add_argument("--no_wandb", action="store_true")
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

    lc = {}
    lc_path = os.path.join(runs_root, task, ftype, "label_consistency.json")
    if not args.untrained and not args.skip_label_consistency:
        if not os.path.exists(lc_path) or args.force:
            lc = label_consistency(cs, split, cfg, task, args.device)
            write_json(lc_path, lc)
            print(f"[rrc] label consistency: {json.dumps(lc, default=float)}")
    if not lc and os.path.exists(lc_path):
        with open(lc_path) as f:
            lc = json.load(f)

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
        run = None if args.no_wandb else wandb_utils.init_run(
            name=f"rrc_reward/{task}/{arm}/s{seed}", group=f"rrc_reward/{task}/{arm}", job_type="reward",
            config=dict(cfg, task=task, seed=seed, arm=arm, pilot=args.pilot, n_acts=len(cs),
                        n_train=int(len(split["train_idx"])), n_val=int(len(split["val_idx"]))),
            run_dir=out)
        on_epoch = None if run is None else (lambda row: run.log({k.replace("_", "/", 1): v for k, v in row.items()},
                                                                  step=int(row["epoch"])))
        summary = None
        if args.untrained:
            export_untrained(cs, split, cfg, out, seed, args.device)
            exports = ["best"]
        else:
            summary = train_reward_ensemble(cs, split, cfg, out, seed, args.device, pilot=args.pilot, epoch_callback=on_epoch)
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
        best = report["exports"]["best"]
        wandb_utils.finish(run, dict(
            train=None if summary is None else {k: summary[k] for k in ("epochs_run", "early_stop_epoch", "ens_best_epoch",
                                                                        "ens_best_val_nll", "true_reward_mean", "true_reward_std")
                                                if k in summary},
            best=dict(val=best["heldout"]["val"], train=best["heldout"]["train"], gt_corr_val=best["heldout"]["gt_corr_val"],
                      steps_val=dict(disagreement=best["heldout"]["steps_val"]["disagreement"],
                                     saturation_max=max(best["heldout"]["steps_val"]["saturation"])),
                      offdist={k: v for k, v in best.get("offdist", {}).items() if k != "by_policy"}),
            label_consistency={k: v for k, v in lc.items() if not isinstance(v, (list, dict))},
        ))
        b = report["exports"]["best"]["heldout"]
        print(f"[rrc] seed {seed} best: val NLL {b['val'].get('nll_ens', float('nan')):.4f} acc {b['val'].get('acc_ens', float('nan')):.3f} "
              f"gt corr {b['gt_corr_val']['all']['pearson']:.3f}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
