"""
SAC on a learned (or true, or untrained) reward with the exact oracle SAC config.

    python scripts/train_rrc_sac.py --task mw_door-open-v2 --arm pref@best --seed 0
    python scripts/train_rrc_sac.py --task mw_door-open-v2 --arm true_reward --seed 0     # oracle path
    python scripts/train_rrc_sac.py --task mw_door-open-v2 --arm untrained --seed 0       # negative control

Arms and paths (under --runs_root, default runs/rrc_sac):
    <type>@best  reward <task>/<type>/seed<s>/reward/best.pt     -> run <task>/<type>/seed<s>/sac_best
    <type>@3x    reward <task>/<type>/seed<s>/reward/epoch3x.pt  -> run <task>/<type>/seed<s>/sac_3x
    true_reward  env reward through the same wrapper              -> run <task>/true_reward/seed<s>/sac
    untrained    reward <task>/untrained/seed<s>/reward/best.pt  -> run <task>/untrained/seed<s>/sac

This mirrors scripts/train.py + Config.get_trainer() exactly; the only change is that the
training env is wrapped with LearnedRewardWrapper and the eval env with ShadowRewardWrapper
(eval reward and success stay ground truth). The SAC config is asserted equal to the task's
oracle config; the only sanctioned deviation is --total_steps (the SAC step budget, recorded in
reward_map.json). --smoke is required to override anything else.
"""

import argparse
import functools
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml

from research.reward_learning.data import load_rrc_config, resolve

ARM_EXPORTS = {"best": ("best.pt", "sac_best"), "3x": ("epoch3x.pt", "sac_3x"), "last": ("last.pt", "sac_last"),
               "best_ens": ("best_ens.pt", "sac_best_ens")}
IGNORED_KEYS = ("env", "seed", "checkpoint")


def arm_paths(runs_root: str, task: str, arm: str, seed: int):
    """-> (mode, reward_ckpt or None, run_dir)"""
    if arm == "true_reward":
        return "true", None, os.path.join(runs_root, task, "true_reward", f"seed{seed}", "sac")
    if arm == "untrained":
        base = os.path.join(runs_root, task, "untrained", f"seed{seed}")
        return "learned", os.path.join(base, "reward", "best.pt"), os.path.join(base, "sac")
    ftype, _, which = arm.partition("@")
    ckpt, run = ARM_EXPORTS[which or "best"]
    base = os.path.join(runs_root, task, ftype, f"seed{seed}")
    return "learned", os.path.join(base, "reward", ckpt), os.path.join(base, run)


def check_against_oracle(sac_cfg: dict, oracle_cfg_path: str) -> list:
    with open(oracle_cfg_path) as f:
        oracle = yaml.load(f, Loader=yaml.Loader)
    diffs = []
    for k in sorted(set(sac_cfg) | set(oracle)):
        if k in IGNORED_KEYS:
            continue
        if sac_cfg.get(k) != oracle.get(k):
            diffs.append((k, sac_cfg.get(k), oracle.get(k)))
    return diffs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--runs_root", default=None)
    ap.add_argument("--rrc_config", default="configs/rrc_sac/base.yaml", help="for paths (oracle_dir, runs_root)")
    ap.add_argument("--overlay", nargs="*", default=[])
    ap.add_argument("--sac_config", default="configs/rrc_sac/sac_oracle.yaml")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--snapshot_every", type=int, default=50_000, help="save eval trajectories every N env steps")
    ap.add_argument("--total_steps", type=int, default=None,
                    help="SAC env-step budget (default: the oracle config's 1M); every other setting stays the oracle's")
    ap.add_argument("--smoke", action="store_true", help="allow --set overrides of the oracle config (testing only)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--allow_config_diff", action="store_true")
    args = ap.parse_args()
    assert not args.set or args.smoke, "--set changes the oracle SAC config; only allowed with --smoke"

    rcfg = load_rrc_config(args.rrc_config, args.overlay)
    runs_root = args.runs_root or resolve(rcfg["paths"]["runs_root"], args.task, rcfg)
    mode, reward_ckpt, path = arm_paths(runs_root, args.task, args.arm, args.seed)
    if os.path.exists(os.path.join(path, "training_complete")):
        print(f"[rrc-sac] {path} already complete, skipping")
        return
    if reward_ckpt is not None:
        assert os.path.exists(reward_ckpt), f"missing reward checkpoint {reward_ckpt} -- train it first"

    from research.reward_learning import envs as rrc_envs
    from research.utils import evaluate as research_evaluate
    from research.utils.config import Config
    from research.utils.trainer import Trainer

    config = Config.load(args.sac_config)
    config["env"] = args.task
    config["seed"] = args.seed
    oracle_cfg = os.path.join(resolve(rcfg["paths"]["oracle_dir"], args.task, rcfg), "config.yaml")
    diffs = check_against_oracle(config.config, oracle_cfg)
    if diffs and not args.allow_config_diff:
        raise SystemExit(f"SAC config differs from the oracle's ({oracle_cfg}): {diffs}")
    overrides = {}
    if args.total_steps is not None:
        config["trainer_kwargs"]["total_steps"] = int(args.total_steps)
        overrides["trainer_kwargs.total_steps"] = int(args.total_steps)
    for kv in args.set:
        dotted, _, raw = kv.partition("=")
        node = config.config
        keys = dotted.split(".")
        for k in keys[:-1]:
            node = node[k]
        node[keys[-1]] = yaml.safe_load(raw)

    # Affine map onto the true reward's scale (research SAC does not normalize rewards).
    if mode == "learned":
        model = rrc_envs.NumpyRewardModel(reward_ckpt)
        affine = (float(model.extra["sigma_true"]), float(model.extra["mu_true"]))
        reward_map = dict(mode=mode, reward_ckpt=os.path.abspath(reward_ckpt), a=affine[0], b=affine[1],
                          branch="affine: SAC does not normalize rewards; a*r_hat+b matches the true reward's "
                                 "mean/std over this type's training (s,a)", member_epochs=model.extra.get("member_epochs"),
                          reward_type=model.extra.get("type"))
    else:
        affine = (1.0, 0.0)
        reward_map = dict(mode=mode, reward_ckpt=None, a=1.0, b=0.0, branch="oracle path: env reward through the wrapper")

    os.makedirs(path, exist_ok=True)
    try:
        from scripts.train import try_wandb_setup
        try_wandb_setup(path, config)
    except Exception as e:  # wandb is optional
        print(f"[wandb] setup skipped: {e}")
    config.save(path)
    with open(os.path.join(path, "reward_map.json"), "w") as f:
        json.dump(dict(reward_map, task=args.task, arm=args.arm, seed=args.seed, config_diffs=[d[0] for d in diffs],
                       overrides=overrides, total_steps=config["trainer_kwargs"]["total_steps"],
                       smoke_overrides=args.set), f, indent=2)
    git = subprocess.Popen(["git", "rev-parse", "HEAD"], stdout=subprocess.PIPE).communicate()[0].strip()
    with open(os.path.join(path, "git_hash.txt"), "wb") as f:
        f.write(git)

    config = config.parse()
    model = config.get_model(device=args.device)
    eval_env = config["eval_env"] if config["eval_env"] is not None else config["env"]
    eval_kwargs = config["eval_env_kwargs"] if config["eval_env"] is not None else config["env_kwargs"]
    common = dict(wrapper=config["wrapper"], wrapper_kwargs=config["wrapper_kwargs"], mode=mode,
                  reward_ckpt=reward_ckpt, affine=affine)
    train_env_fn = functools.partial(rrc_envs.make_env, env_name=config["env"], env_kwargs=config["env_kwargs"],
                                     role="train", episode_log=os.path.join(path, "train_episodes.csv"), **common)
    eval_env_fn = functools.partial(rrc_envs.make_env, env_name=eval_env, env_kwargs=eval_kwargs, role="eval", **common)

    # Save eval-trajectory snapshots: wrap the configured eval fn without changing its behaviour.
    eval_fn_name = config["trainer_kwargs"].get("eval_fn")
    original_eval = getattr(research_evaluate, eval_fn_name)

    def eval_with_snapshots(env, model_, path_, step, **kwargs):
        metrics = original_eval(env, model_, path_, step, **kwargs)
        episodes = env.pop_episodes() if hasattr(env, "pop_episodes") else []
        if args.snapshot_every > 0 and step % args.snapshot_every == 0:
            rrc_envs.save_eval_episodes(episodes, os.path.join(path_, "eval_traj", f"step_{step}.npz"))
        return metrics

    setattr(research_evaluate, eval_fn_name, eval_with_snapshots)

    trainer = Trainer(model, train_env_fn, eval_env_fn, **config["trainer_kwargs"])
    trainer.train(path)
    open(os.path.join(path, "training_complete"), "w").close()


if __name__ == "__main__":
    main()
