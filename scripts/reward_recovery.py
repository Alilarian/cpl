"""
Reward recovery sanity check (Phase-1 sanity check #2 in the reward-CPL baseline plan).

For a feedback type, computes the Spearman correlation of each scorer's segment
score against the oracle ranking already stored in the frozen *_labels.npz:
  - r_theta(s,a) segment scores, from a trained RewardCPL/RewardCreditCPL/
    RewardEstopHoldCPL checkpoint (Phase 1 of the reward-model baseline).
  - alpha * log pi(a|s) segment scores, from the matching ARIC checkpoint
    (CPL/DemoCPL/CreditAssignmentCPL/EstopHoldCPL).
Both scorers are run through the exact same score_segments reshape used to train
them, so this also cross-checks that the two scorers are being compared on
identical segments.

Cross-reference the printed traj_r_var (also logged by RewardCPL* during
validation) against these Spearman numbers to check the shift-identifiability
prediction: scalar/credit assignment (within-trajectory-only comparisons) should
show worse reward-side Spearman correlation than pairwise/demo (cross-trajectory),
while the ARIC/log-pi side should not degrade the same way -- pi is normalized per
state, so alpha*log pi cannot absorb a free per-trajectory shift the way an
unconstrained r_theta can (CPL Sec. 3.1's argument for why the distributional
constraint fixes the shift-invariance of Boltzmann preference models).

Usage:
  python scripts/reward_recovery.py \
      --type demo \
      --labels datasets/mw/demo_labels/mw_drawer-open-v2/demo_labels_K7.npz \
      --reward-config configs/mw_state_dense/reward_demo.yaml \
      --reward-checkpoint runs/reward_demo/best_model.pt \
      --aric-config configs/mw_state_dense/demo_cpl.yaml \
      --aric-checkpoint runs/demo_cpl/best_model.pt \
      --out results/reward_recovery.csv
"""

import argparse
import csv
import os
from typing import Optional, Tuple

import numpy as np
import torch
from scipy.stats import spearmanr

from research.algs.scoring import score_segments
from research.utils.config import Config


def _oracle_score(npz: np.lib.npyio.NpzFile) -> Tuple[np.ndarray, str]:
    """(N, K) oracle score per segment, plus a string noting which field was used."""
    if "reward" in npz.files:
        return npz["reward"].sum(axis=-1), "reward.sum(-1)"
    if "adv_scores" in npz.files:
        return npz["adv_scores"], "adv_scores"
    if "chosen_idx" in npz.files:
        # Credit assignment: only an ordinal winner is stored, no continuous score.
        chosen = npz["chosen_idx"]
        N, C = npz["obs"].shape[0], npz["obs"].shape[1]
        oracle = np.zeros((N, C), dtype=np.float32)
        oracle[np.arange(N), chosen] = 1.0
        return oracle, "one_hot(chosen_idx)  [WARNING: ordinal only, Spearman will be a coarse proxy]"
    raise ValueError("labels npz has none of reward/adv_scores/chosen_idx -- cannot derive an oracle score.")


def _load_model(config_path: str, checkpoint_path: Optional[str], obs_dim: int, act_dim: int, device: str):
    import gym

    obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
    act_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32)
    config = Config.load(config_path).parse()
    model = config.get_model(observation_space=obs_space, action_space=act_space, device=device)
    if checkpoint_path is not None:
        model.load(checkpoint_path, strict=True)
    model.eval()
    return model


def _reward_scorer(model):
    def scorer(obs, act):
        with torch.no_grad():
            return model.network.reward(obs, act).mean(dim=0)

    return scorer


def _policy_scorer(model, alpha: float):
    def scorer(obs, act):
        with torch.no_grad():
            obs_enc = model.network.encoder(obs)
            dist = model.network.actor(obs_enc)
            if isinstance(dist, torch.distributions.Distribution):
                lp = dist.log_prob(act)
            else:
                lp = -torch.square(dist - act).sum(dim=-1)
            return alpha * lp

    return scorer


def compute_recovery(
    feedback_type: str,
    labels_path: str,
    reward_config: Optional[str],
    reward_checkpoint: Optional[str],
    aric_config: Optional[str],
    aric_checkpoint: Optional[str],
    alpha: float,
    discount: float,
    device: str = "cpu",
):
    npz = np.load(labels_path)
    obs = torch.from_numpy(npz["obs"].astype(np.float32))
    action = torch.from_numpy(npz["action"].astype(np.float32))
    oracle, oracle_source = _oracle_score(npz)
    obs_dim, act_dim = obs.shape[-1], action.shape[-1]

    rows = []
    if reward_config is not None:
        reward_model = _load_model(reward_config, reward_checkpoint, obs_dim, act_dim, device)
        reward_scores = score_segments(obs, action, _reward_scorer(reward_model), discount=discount).numpy()
        rho, p = spearmanr(reward_scores.flatten(), oracle.flatten())
        rows.append(dict(type=feedback_type, scorer="r_theta", spearman_rho=rho, spearman_p=p, oracle_source=oracle_source))

    if aric_config is not None:
        aric_model = _load_model(aric_config, aric_checkpoint, obs_dim, act_dim, device)
        aric_scores = score_segments(obs, action, _policy_scorer(aric_model, alpha), discount=1.0).numpy()
        rho, p = spearmanr(aric_scores.flatten(), oracle.flatten())
        rows.append(
            dict(type=feedback_type, scorer="alpha*log_pi", spearman_rho=rho, spearman_p=p, oracle_source=oracle_source)
        )

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", required=True, help="Feedback type label for the output CSV (e.g. demo, scalar).")
    parser.add_argument("--labels", required=True, help="Path to the frozen *_labels.npz.")
    parser.add_argument("--reward-config", default=None, help="Phase-1 RewardCPL* config (e.g. reward_demo.yaml).")
    parser.add_argument("--reward-checkpoint", default=None, help="Phase-1 RewardCPL* checkpoint (best_model.pt).")
    parser.add_argument("--aric-config", default=None, help="Matching ARIC config (e.g. demo_cpl.yaml).")
    parser.add_argument("--aric-checkpoint", default=None, help="Matching ARIC checkpoint (best_model.pt).")
    parser.add_argument("--alpha", type=float, default=0.1, help="alpha used for the ARIC config (must match its yaml).")
    parser.add_argument("--discount", type=float, default=1.0, help="discount used for the reward config (1.0 unless e-stop).")
    parser.add_argument("--out", default="results/reward_recovery.csv")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    rows = compute_recovery(
        args.type,
        args.labels,
        args.reward_config,
        args.reward_checkpoint,
        args.aric_config,
        args.aric_checkpoint,
        args.alpha,
        args.discount,
        args.device,
    )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    write_header = not os.path.exists(args.out)
    with open(args.out, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["type", "scorer", "spearman_rho", "spearman_p", "oracle_source"])
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)
            print(row)
    print(f"Appended {len(rows)} row(s) to {args.out}")


if __name__ == "__main__":
    main()
