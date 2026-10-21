"""
Reward-model diagnostics (docs/rrc_sac_plan.md section 3.4), written to report.json.

All "predicted" quantities use the standardized ensemble reward mean_m (r_m - mu_m)/sigma_m;
correlations are invariant to its affine map to the true-reward scale.
"""

import json
import os
from typing import Dict, Optional

import numpy as np
import torch
from scipy import stats

from .data import ChoiceSet, resolve
from .ensemble import RewardEnsemble
from .train import TorchChoiceData, evaluate_split

# --------------------------------------------------------------------------------------
# Small stats helpers
# --------------------------------------------------------------------------------------


def _corr(x: np.ndarray, y: np.ndarray) -> Dict:
    x, y = np.asarray(x, np.float64).ravel(), np.asarray(y, np.float64).ravel()
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return dict(pearson=float("nan"), spearman=float("nan"), n=int(len(x)))
    return dict(pearson=float(stats.pearsonr(x, y)[0]), spearman=float(stats.spearmanr(x, y)[0]), n=int(len(x)))


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """P(score of a positive > score of a negative), ties counted 1/2 (Mann-Whitney U / n1 n0)."""
    scores, labels = np.asarray(scores, np.float64), np.asarray(labels).astype(bool)
    n1, n0 = labels.sum(), (~labels).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    ranks = stats.rankdata(scores)
    return float((ranks[labels].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


# --------------------------------------------------------------------------------------
# Held-out data
# --------------------------------------------------------------------------------------


@torch.no_grad()
def _option_pred_returns(model: RewardEnsemble, data: TorchChoiceData, rows: np.ndarray) -> np.ndarray:
    """(n, n_options) option returns under the standardized ensemble reward."""
    rows_t = torch.as_tensor(rows, device=data.device)
    raw = data.all_scores(model.net, rows_t)  # (E, n, K) from raw member rewards
    # Option returns are linear in r, so standardizing member rewards maps to
    # (R_m - mu_m * D) / sigma_m with D = sum of (masked) discount weights per option.
    ones = torch.ones_like(data.obs[rows_t][..., 0])  # (n, K, L) or (n, T)
    mask = None if data.step_mask is None else data.step_mask[rows_t]
    D = data._returns(ones, mask)  # (n, K)
    mu, sigma = model.mu.view(-1, 1, 1), model.sigma.view(-1, 1, 1)
    z = (raw - mu * D.unsqueeze(0)) / sigma
    return z.mean(0).cpu().numpy()


@torch.no_grad()
def _step_stats(model: RewardEnsemble, obs: np.ndarray, act: np.ndarray, thr: float) -> Dict:
    raw = model.member_rewards(obs, act)  # (E, n)
    z = (raw - model.mu[:, None]) / model.sigma[:, None]
    return dict(
        saturation=[float(v) for v in (raw.abs() > thr).float().mean(1).cpu().numpy()],
        disagreement=float(z.std(0, unbiased=False).mean().cpu()),
        n_steps=int(raw.shape[1]),
    )


def heldout_report(model: RewardEnsemble, cs: ChoiceSet, split: Dict, cfg: Dict, device) -> Dict:
    beta = float(cfg["rrc"].get("beta", 1.0))
    thr = float(cfg.get("diagnostics", {}).get("saturation_threshold", 0.95))
    data = TorchChoiceData(cs, device)
    tr, va = split["train_idx"], split["val_idx"]
    out = dict(train=evaluate_split(data, model.net, tr, cs.act_ids, beta),
               val=evaluate_split(data, model.net, va, cs.act_ids, beta))
    if "val_strict" in split and split["val_strict"].any() and not split["val_strict"].all():
        out["val_strict"] = evaluate_split(data, model.net, va[split["val_strict"]], cs.act_ids, beta)
    if cs.branch is not None:
        for b in np.unique(cs.branch):
            out[f"val_branch{b}"] = evaluate_split(data, model.net, va[cs.branch[va] == b], cs.act_ids, beta)

    pred = _option_pred_returns(model, data, va)
    true = cs.option_true_returns()[va]
    valid = np.ones_like(pred, bool) if cs.option_mask is None else cs.option_mask[va].astype(bool)
    centered = lambda x: x - np.where(valid, x, 0).sum(1, keepdims=True) / valid.sum(1, keepdims=True)
    out["gt_corr_val"] = dict(
        all=_corr(pred[valid], true[valid]),
        within_act=_corr(centered(pred)[valid], centered(true)[valid]),
    )
    obs_v, act_v, _ = cs.subset(va).valid_steps()
    out["steps_val"] = _step_stats(model, obs_v, act_v, thr)
    return out


# --------------------------------------------------------------------------------------
# Off-distribution rollouts
# --------------------------------------------------------------------------------------


def _windows(x: np.ndarray, L: int, stride: int) -> np.ndarray:
    """(Ne, T, ...) -> (Ne * n_windows, L, ...) windows starting every `stride` steps."""
    T = x.shape[1]
    starts = range(0, T - L + 1, stride)
    return np.concatenate([x[:, s : s + L] for s in starts], axis=0)


@torch.no_grad()
def offdist_report(model: RewardEnsemble, cs: ChoiceSet, cache_path: str, cfg: Dict) -> Dict:
    thr = float(cfg.get("diagnostics", {}).get("saturation_threshold", 0.95))
    with np.load(cache_path, allow_pickle=False) as c:
        obs, act, rew, succ, policy = c["obs"], c["act"], c["reward"], c["success"], c["policy"]
    Ne, T = rew.shape
    raw = model.member_rewards(obs, act)  # (E, Ne, T)
    z = (raw - model.mu.view(-1, 1, 1)) / model.sigma.view(-1, 1, 1)
    r_hat = z.mean(0).cpu().numpy()
    ep_success = succ.max(1) > 0
    disc = 0.99 ** np.arange(T)
    out = dict(
        n_episodes=int(Ne), n_success=int(ep_success.sum()),
        episode_corr_undiscounted=_corr(r_hat.sum(1), rew.sum(1)),
        episode_corr_discounted=_corr((r_hat * disc).sum(1), (rew * disc).sum(1)),
        success_auc_pred=auc(r_hat.sum(1), ep_success),
        success_auc_true=auc(rew.sum(1), ep_success),
        saturation=[float(v) for v in (raw.abs() > thr).float().mean((1, 2)).cpu().numpy()],
        disagreement=float(z.std(0, unbiased=False).mean().cpu()),
    )
    L = cs.window if cs.kind == "windows" else cs.obs.shape[2]
    w = cs.gamma ** np.arange(L)
    seg_pred = (_windows(r_hat, L, max(1, L // 2)) * w).sum(1)
    seg_true = (_windows(rew, L, max(1, L // 2)) * w).sum(1)
    out["segment_corr"] = dict(_corr(seg_pred, seg_true), length=int(L), gamma=cs.gamma)
    groups = {}
    for p in np.unique(policy):
        m = policy == p
        groups[str(p)] = dict(n=int(m.sum()), pred_return=float(r_hat[m].sum(1).mean()),
                              true_return=float(rew[m].sum(1).mean()), success=float(ep_success[m].mean()))
    out["by_policy"] = groups
    return out


# --------------------------------------------------------------------------------------
# Label consistency against the labeler (oracle critic) -- independent of the reward model
# --------------------------------------------------------------------------------------


def load_oracle(oracle_dir: str, ckpt: str, device: str):
    from research.utils.config import Config

    config = Config.load(oracle_dir)
    config["checkpoint"] = None
    config = config.parse()
    env = config.get_eval_env_fn()()
    model = config.get_model(observation_space=env.observation_space, action_space=env.action_space, device=device)
    model.load(os.path.join(oracle_dir, ckpt))
    model.eval()
    return model, env


@torch.no_grad()
def oracle_values(oracle, obs: np.ndarray, mcmc_samples: int, device: str, chunk: int = 4096) -> np.ndarray:
    """V(s) exactly as the label generators estimate it: critic-ensemble mean over MC actor samples."""
    flat = obs.reshape(-1, obs.shape[-1])
    out = np.empty(len(flat), np.float32)
    for i in range(0, len(flat), chunk):
        o = torch.as_tensor(flat[i : i + chunk], device=device, dtype=torch.float32)
        enc = oracle.network.encoder(o)
        exp = enc.unsqueeze(1).expand(-1, mcmc_samples, -1)
        a = oracle.network.actor(exp).sample()
        out[i : i + chunk] = oracle.network.critic(exp, a).mean(dim=0).mean(dim=1).cpu().numpy()
    return out.reshape(obs.shape[:-1])


def _chosen(cs: ChoiceSet) -> np.ndarray:
    return cs.target.argmax(1)


def _oracle_candidates(cfg: Dict, task: str):
    """[(run_dir, ckpt)] that exist: paths.label_oracle_candidates ("dir:ckpt"), else paths.oracle_dir."""
    raw = cfg["paths"].get("label_oracle_candidates") or [f"{cfg['paths']['oracle_dir']}:{cfg['paths'].get('oracle_ckpt', 'best_model.pt')}"]
    out = []
    for c in raw:
        d, _, ck = resolve(c, task, cfg).rpartition(":")
        if os.path.exists(os.path.join(d, ck)):
            out.append((d, ck))
    assert out, f"none of the label-oracle candidates exist: {raw}"
    return out


def label_consistency(cs: ChoiceSet, split: Dict, cfg: Dict, task: str, device: str, seed: int = 0) -> Dict:
    """
    (i)  the label generator's own rule recomputed with an oracle critic -- must be ~1.0 when the
         oracle is the one that labeled the data (the labeler is picked among the configured
         candidates by agreement with the stored scores on a probe sample);
    (ii) the true-reward partial-return accuracy -- the ceiling of this RRC model.
    """
    dcfg = cfg.get("diagnostics", {})
    M = int(dcfg.get("mcmc_samples", 64))
    rng = np.random.RandomState(seed)
    chosen = _chosen(cs)
    out = dict()

    pr = cs.option_true_returns()
    out["partial_return_acc_all"] = float((pr.argmax(1) == chosen).mean())
    if cs.adv_scores is not None:
        out["stored_score_acc_all"] = float((cs.adv_scores.argmax(1) == chosen).mean())

    if cs.type == "cum_estop":
        n = int(dcfg.get("cum_estop_consistency_acts", 200))
        rows = np.concatenate([rng.choice(np.nonzero(cs.branch == b)[0], size=min(n, int((cs.branch == b).sum())),
                                          replace=False) for b in (0, 1)])
    else:
        n_tr = int(dcfg.get("label_consistency_train_acts", 1000))
        tr = rng.choice(split["train_idx"], size=min(n_tr, len(split["train_idx"])), replace=False)
        rows = np.concatenate([split["val_idx"], tr])
    rows = np.sort(rows)

    cands = _oracle_candidates(cfg, task)
    has_stored = cs.adv_scores is not None and cs.type != "cum_estop"
    if len(cands) > 1 and has_stored:
        probe = np.sort(rng.choice(rows, size=min(200, len(rows)), replace=False))
        method = "spearman" if cs.type == "scalar" else "pearson"  # scalar stores percentile-normalized scores
        out["labeler_candidates"] = []
        for d, ck in cands:
            oracle, _ = load_oracle(d, ck, device)
            c = _corr(_rule_scores(oracle, cs, probe, M, device), cs.adv_scores[probe])[method]
            out["labeler_candidates"].append(dict(oracle=f"{d}:{ck}", **{f"{method}_vs_stored": c}))
        best = int(np.nanargmax([c[f"{method}_vs_stored"] for c in out["labeler_candidates"]]))
        cands = [cands[best]]
    out["labeler_oracle"] = f"{cands[0][0]}:{cands[0][1]}"
    oracle, _ = load_oracle(cands[0][0], cands[0][1], device)
    scores = _rule_scores(oracle, cs, rows, M, device)
    acc = (scores.argmax(1) == chosen[rows])
    out["oracle_rule_acc"] = float(acc.mean())
    out["oracle_rule_n"] = int(len(rows))
    if cs.type != "cum_estop":
        out["oracle_rule_acc_val"] = float(acc[np.isin(rows, split["val_idx"])].mean())
    if has_stored:
        out["oracle_rule_vs_stored_scores"] = _corr(scores, cs.adv_scores[rows])
    if cs.type == "cum_estop":
        br = cs.branch[rows]
        out["oracle_rule_acc_by_branch"] = {int(b): float(acc[br == b].mean()) for b in np.unique(br)}
        stored_acc = (scores.argmax(1) == 0)
        out["oracle_rule_acc_as_stored_index0_by_branch"] = {int(b): float(stored_acc[br == b].mean()) for b in np.unique(br)}
        out["partial_return_acc_by_branch"] = {int(b): float((pr.argmax(1) == chosen)[cs.branch == b].mean())
                                                for b in np.unique(cs.branch)}
        out["nostop_original_at_index0_frac"] = cs.meta.get("nostop_original_at_index0_frac")
        out["nostop_preferred_mode"] = cs.meta.get("nostop_preferred")
        out["note"] = "cum_estop oracle rule is approximate (endpoint V at the last stored obs, last reward dropped)"
    return out


def _rule_scores(oracle, cs: ChoiceSet, rows: np.ndarray, M: int, device: str) -> np.ndarray:
    """(n, n_options) labeler scores recomputed with `oracle`, following each generator's formula."""
    if cs.kind == "windows":
        k = cs.window
        V = oracle_values(oracle, cs.obs[rows], M, device)  # (n, T)
        r = cs.true_reward[rows].astype(np.float64)
        P = np.concatenate([np.zeros((len(rows), 1)), np.cumsum(r, 1)], axis=1)
        C = cs.n_options
        # window_score[i] = sum_{t=i}^{i+k-2} r_t + V[i+k-1] - V[i]   (generate_credit_labels.py)
        scores = np.stack([P[:, i + k - 1] - P[:, i] + V[:, i + k - 1] - V[:, i] for i in range(C)], axis=1)
    else:
        obs, rew = cs.obs[rows], cs.true_reward[rows].astype(np.float64)
        N, K, L, _ = obs.shape
        if cs.type == "cum_estop":
            # U_E from the divergence point s (0 for no-stop rows): sum_k gamma^k r_{s+k} + gamma^{L'} V(end) - V(s_s).
            # The stored arrays lack the post-terminal obs, so V is taken at the last stored obs and
            # the last reward is dropped (same endpoint convention as the rl_sum labelers) -- approximate.
            g = cs.gamma
            s = np.where(cs.branch[rows] == 0, cs.meta["stop_index"][rows], 0)
            idx = np.stack([s, np.full(N, L - 1)], 1)
            pts = np.stack([obs[np.arange(N), k][np.arange(N)[:, None], idx] for k in range(K)], 1)  # (N, K, 2, D)
            V = oracle_values(oracle, pts, M, device)  # (N, K, 2)
            scores = np.zeros((N, K))
            for i in range(N):
                Lp = L - s[i]
                w = g ** np.arange(Lp - 1)
                for kk in range(K):
                    scores[i, kk] = (rew[i, kk, s[i] : L - 1] * w).sum() + g ** (Lp - 1) * V[i, kk, 1] - V[i, kk, 0]
        else:
            pts = obs[:, :, [0, L - 1]]  # (N, K, 2, D)
            V = oracle_values(oracle, pts, M, device)
            # rl_sum = sum_{t<L-1} r_t + V(s_{L-1}) - V(s_0)   (generate_{pref,corr,demo,scalar}_labels.py)
            scores = rew[:, :, : L - 1].sum(-1) + V[:, :, 1] - V[:, :, 0]
    return scores


def write_json(path: str, obj: Dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=float)
    os.replace(tmp, path)
