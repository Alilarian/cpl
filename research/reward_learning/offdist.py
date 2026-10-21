"""
Off-distribution rollouts for reward-model diagnostics and reward-hacking normalization.

Cached once per task: uniform-random episodes plus deterministic episodes from oracle SAC
checkpoints chosen by their logged eval success (~0, 0.25, 0.5, 0.75, 1.0) and the final
oracle checkpoint. Random episodes almost never succeed, so the checkpoints supply the
successful episodes needed for the success-separation AUC; "random" and "oracle_final"
are the 0 and 1 references of the reward-hacking scale (hacking.py).
"""

import csv
import os
import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch


def oracle_eval_curve(oracle_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    with open(os.path.join(oracle_dir, "log.csv")) as f:
        rows = [r for r in csv.DictReader(f) if r.get("eval/success") not in (None, "")]
    steps = np.array([int(float(r["step"])) for r in rows])
    succ = np.array([float(r["eval/success"]) for r in rows])
    return steps, succ


def pick_oracle_checkpoints(oracle_dir: str, targets: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0)) -> List[Tuple[str, str, float]]:
    """[(label, checkpoint file, logged success)] -- for each target the earliest checkpoint whose
    nearest logged eval success is closest to it; plus the final checkpoint."""
    steps, succ = oracle_eval_curve(oracle_dir)
    ckpts = sorted(
        (int(m.group(1)), f) for f in os.listdir(oracle_dir) if (m := re.match(r"model_(\d+)\.pt$", f))
    )
    assert ckpts, f"no model_<step>.pt checkpoints in {oracle_dir}"
    ck_steps = np.array([s for s, _ in ckpts])
    ck_succ = np.array([succ[np.argmin(np.abs(steps - s))] for s in ck_steps])
    picked, used = [], set()
    for t in targets:
        order = np.lexsort((ck_steps, np.abs(ck_succ - t)))  # closest success, then earliest
        for j in order:
            if ckpts[j][1] not in used:
                used.add(ckpts[j][1])
                picked.append((f"oracle_s{t:.2f}", ckpts[j][1], float(ck_succ[j])))
                break
    final = "final_model.pt" if os.path.exists(os.path.join(oracle_dir, "final_model.pt")) else ckpts[-1][1]
    picked.append(("oracle_final", final, float(succ[-3:].mean())))
    return picked


def load_policy(run_dir: str, ckpt_file: str, device: str):
    from research.utils.config import Config

    config = Config.load(run_dir)
    config["checkpoint"] = None
    config = config.parse()
    env = config.get_eval_env_fn()()
    model = config.get_model(observation_space=env.observation_space, action_space=env.action_space, device=device)
    model.load(os.path.join(run_dir, ckpt_file))
    model.eval()
    return model, env


def rollout(env, policy, n_episodes: int, seed: int) -> Dict[str, np.ndarray]:
    """policy=None -> uniform random actions. Same action call as eval_policy (model.predict)."""
    obs_l, act_l, rew_l, suc_l = [], [], [], []
    env.action_space.seed(seed)
    for i in range(n_episodes):
        np.random.seed(seed * 1000 + i)  # randomize_hand draws from np.random
        obs = env.reset()
        obs = obs[0] if isinstance(obs, tuple) else obs
        o_ep, a_ep, r_ep, s_ep = [], [], [], []
        done = False
        while not done:
            if policy is None:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    action = policy.predict(dict(obs=obs))
            res = env.step(action)
            if len(res) == 5:
                next_obs, r, term, trunc, info = res
                done = term or trunc
            else:
                next_obs, r, done, info = res
            o_ep.append(np.asarray(obs, np.float32))
            a_ep.append(np.clip(np.asarray(action, np.float32), env.action_space.low, env.action_space.high))
            r_ep.append(float(r))
            s_ep.append(float(info.get("success", 0.0)))
            obs = next_obs
        obs_l.append(np.stack(o_ep)); act_l.append(np.stack(a_ep))
        rew_l.append(np.array(r_ep, np.float32)); suc_l.append(np.array(s_ep, np.float32))
    return dict(obs=np.stack(obs_l), act=np.stack(act_l), reward=np.stack(rew_l), success=np.stack(suc_l))


def collect_offdist(oracle_dir: str, out_path: str, n_random: int = 50, n_per_ckpt: int = 20,
                    extra_ckpts: Sequence[str] = (), seed: int = 0, device: str = "cpu") -> Dict:
    picked = pick_oracle_checkpoints(oracle_dir)
    _, env = load_policy(oracle_dir, picked[-1][1], device)
    parts, labels, meta = [], [], dict(oracle_dir=oracle_dir, checkpoints=[])
    rnd = rollout(env, None, n_random, seed)
    parts.append(rnd); labels += ["random"] * n_random
    for j, (label, ck, s) in enumerate(picked):
        model, _ = load_policy(oracle_dir, ck, device)
        parts.append(rollout(env, model, n_per_ckpt, seed + 1 + j)); labels += [label] * n_per_ckpt
        meta["checkpoints"].append(dict(label=label, checkpoint=ck, logged_success=s))
    for j, path in enumerate(extra_ckpts):
        run_dir, ck = os.path.dirname(path), os.path.basename(path)
        model, _ = load_policy(run_dir, ck, device)
        label = "extra:" + path
        parts.append(rollout(env, model, n_per_ckpt, seed + 100 + j)); labels += [label] * n_per_ckpt
        meta["checkpoints"].append(dict(label=label, checkpoint=path))
    out = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    out["policy"] = np.array(labels)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp = out_path + ".tmp"
    with open(tmp, "wb") as f:
        np.savez(f, **out)
    os.replace(tmp, out_path)
    meta["n_episodes"] = int(len(labels))
    meta["success_by_policy"] = {l: float(out["success"][np.array(labels) == l].max(1).mean()) for l in dict.fromkeys(labels)}
    return meta
