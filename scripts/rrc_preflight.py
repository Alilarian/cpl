"""
Stage A of RRC + SAC (docs/rrc_sac_plan.md section 5): check every input before spending compute.

    python scripts/rrc_preflight.py --tasks all [--overlay configs/rrc_sac/local.yaml] [--types ...]

Per task: pool + 200 recovered episodes; each feedback type loads through ARIC's buffers with
ARIC's rows, every act maps to a source episode, credit/scalar rows align with the pool, the
cum_estop original arm is found (and where it is stored); env spaces; the SAC config equals the
oracle config; oracle checkpoints and log.csv exist; label-oracle candidates present.
Writes <runs_root>/_preflight/<task>.json and exits non-zero if anything fails.
"""

import argparse
import json
import os
import re
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from research.reward_learning.data import PoolIndex, load_choice_set, load_rrc_config, make_episode_split, resolve

TASKS = ["mw_button-press-v2", "mw_door-open-v2", "mw_drawer-open-v2", "mw_plate-slide-v2"]
TYPES = ["pref", "corr", "demo", "scalar", "credit", "cum_estop"]


def check_task(task, types, overlays, runs_root):
    res = dict(task=task, ok=True, checks={})

    def record(name, fn):
        t0 = time.time()
        try:
            out = fn()
            res["checks"][name] = dict(ok=True, **(out or {}), seconds=round(time.time() - t0, 1))
        except Exception as e:  # keep going so one run reports everything
            res["ok"] = False
            res["checks"][name] = dict(ok=False, error=f"{type(e).__name__}: {e}", trace=traceback.format_exc(limit=3))
        status = "ok " if res["checks"][name]["ok"] else "FAIL"
        print(f"[preflight] {task:22s} {status} {name}: "
              f"{ {k: v for k, v in res['checks'][name].items() if k not in ('ok', 'trace')} }", flush=True)

    base = load_rrc_config("configs/rrc_sac/base.yaml", overlays)
    base["paths"]["runs_root"] = runs_root
    state = {}

    def pool_check():
        p = resolve(base["paths"]["pool_path"], task, base)
        pool = PoolIndex(p, expected_episodes=base["data"]["expected_episodes"])
        state["pool"] = pool
        return dict(path=p, rows=int(pool.obs.shape[0]), episodes=pool.n_episodes, states=pool.n_states,
                    reward_min=float(pool.reward.min()), reward_max=float(pool.reward.max()))

    record("pool", pool_check)

    def env_check():
        from research.utils.config import get_env

        env = get_env(task, {"sparse": False}, None, {})
        out = dict(obs_dim=int(env.observation_space.shape[0]), act_dim=int(env.action_space.shape[0]),
                   horizon=int(getattr(env.unwrapped, "_max_episode_steps", -1)))
        env.close()
        assert out["obs_dim"] == 35 and out["act_dim"] == 4, out
        return out

    record("env", env_check)

    def oracle_check():
        d = resolve(base["paths"]["oracle_dir"], task, base)
        files = os.listdir(d)
        n_ckpt = sum(bool(re.match(r"model_\d+\.pt$", f)) for f in files)
        for f in ("config.yaml", "best_model.pt", "final_model.pt", "log.csv"):
            assert f in files, f"{f} missing in {d}"
        assert n_ckpt >= 10, f"only {n_ckpt} model_<step>.pt checkpoints in {d}"
        import yaml

        from scripts.train_rrc_sac import check_against_oracle
        from research.utils.config import Config

        cfg = Config.load("configs/rrc_sac/sac_oracle.yaml")
        cfg["env"] = task
        diffs = check_against_oracle(cfg.config, os.path.join(d, "config.yaml"))
        assert not diffs, f"configs/rrc_sac/sac_oracle.yaml differs from {d}/config.yaml: {diffs}"
        cands = []
        for c in base["paths"].get("label_oracle_candidates", []):
            cd, _, ck = resolve(c, task, base).rpartition(":")
            cands.append(dict(candidate=f"{cd}:{ck}", exists=os.path.exists(os.path.join(cd, ck))))
        return dict(oracle_dir=d, n_checkpoints=n_ckpt, label_oracle_candidates=cands)

    record("oracle", oracle_check)

    for t in types:
        def type_check(t=t):
            assert "pool" in state, "pool failed"
            cfg = load_rrc_config(f"configs/rrc_sac/{t}.yaml", overlays)
            cs = load_choice_set(cfg, task, state["pool"])
            sp = make_episode_split(cs.episodes, state["pool"].n_episodes, cfg["data"]["val_frac"], cfg["data"]["split_seed"])
            out = dict(path=cs.meta["path"], acts=len(cs), options=cs.n_options, obs_shape=list(cs.obs.shape),
                       train=int(len(sp["train_idx"])), val=int(len(sp["val_idx"])), val_episodes=int(len(sp["val_episodes"])),
                       episodes_per_act=float((cs.episodes >= 0).sum(1).mean()), ties=cs.meta.get("n_ties"),
                       true_reward_range=[float(cs.true_reward.min()), float(cs.true_reward.max())])
            if t == "cum_estop":
                out.update(branch_counts={int(b): int((cs.branch == b).sum()) for b in np.unique(cs.branch)},
                           nostop_original_at_index0_frac=cs.meta["nostop_original_at_index0_frac"],
                           stop_original_preferred_frac=cs.meta["stop_original_preferred_frac"],
                           nostop_preferred=cs.meta["nostop_preferred"],
                           stop_index_range=[int(cs.meta["stop_index"][cs.branch == 0].min()),
                                             int(cs.meta["stop_index"][cs.branch == 0].max())] if (cs.branch == 0).any() else None)
            assert cs.obs.shape[-1] == 35 and cs.act.shape[-1] == 4
            assert np.isfinite(cs.obs).all() and np.isfinite(cs.true_reward).all()
            return out

        record(f"type:{t}", type_check)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=["all"])
    ap.add_argument("--types", nargs="+", default=["all"])
    ap.add_argument("--overlay", nargs="*", default=[])
    ap.add_argument("--runs_root", default="runs/rrc_sac")
    args = ap.parse_args()
    tasks = TASKS if args.tasks == ["all"] else args.tasks
    types = TYPES if args.types == ["all"] else args.types
    all_ok = True
    for task in tasks:
        res = check_task(task, types, args.overlay, args.runs_root)
        out = os.path.join(args.runs_root, "_preflight", f"{task}.json")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as f:
            json.dump(res, f, indent=2, default=str)
        all_ok &= res["ok"]
        print(f"[preflight] {task}: {'ALL OK' if res['ok'] else 'FAILURES'} -> {out}", flush=True)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
