"""
Converts research/'s bc_pool.npz (produced by scripts/score_pool_for_bc.py, the
same frozen top-X%-by-oracle-rl_sum pool ARIC's own BC warmup reads via
research/datasets/bc_buffer.py::BCBuffer) into the pickle format
multi_type_feedback/train_bc.py's load_demonstrations() already expects, applying
the identical research/envs/metaworld.py obs-trim/reward-scale transform so PPO's
BC-pretrained reference policy mu is trained on the exact same demonstration data
and observation convention as ARIC's bc_pool warmup and r_theta -- this is what
gives Phase-2 PPO the "equivalent warmup" the spec calls for, since Phase-1's
reward-model training has no BC-warmup analogue of its own.

bc_pool.npz stores T observations paired 1:1 with T actions per segment.
train_bc.py's load_demonstrations expects imitation's Trajectory convention: T+1
observations for T actions (it drops the last action via acts[:-1]). Bridge this
by duplicating each segment's final (obs, action) pair once more as a synthetic
terminal step -- consistent with how research/datasets/estop_buffer.py's padded
tail repeats the last (s, a) pair elsewhere in this codebase.

Usage:
  python multi_type_feedback/convert_research_pool_to_demos.py \
      --bc-pool datasets/mw/bc_pool/mw_drawer-open-v2/bc_pool.npz \
      --out feedback_regen/research_demo_mw_drawer-open-v2.pkl
"""

import argparse
import pickle

import numpy as np


def trim_mw_obs(obs: np.ndarray) -> np.ndarray:
    # Matches research/envs/metaworld.py::trim_mw_obs exactly.
    return np.concatenate((obs[..., :18], obs[..., 22:]), axis=-1).astype(np.float32)


def convert(bc_pool_path: str, sparse: bool = False, apply_trim: bool = True):
    data = np.load(bc_pool_path)
    obs = data["obs"]  # (N, T, obs_dim) -- 39-dim raw MetaWorld obs if not yet trimmed
    action = data["action"]  # (N, T, act_dim)
    N, T = obs.shape[0], obs.shape[1]

    if apply_trim and obs.shape[-1] == 39:
        obs = trim_mw_obs(obs)

    demos = []
    for i in range(N):
        demo = []
        for t in range(T):
            done = False
            demo.append((obs[i, t].astype(np.float32), action[i, t].astype(np.float32), done))
        # Duplicate the final (obs, action) pair as a synthetic terminal step so
        # load_demonstrations sees T+1 observations / T actions (see module docstring).
        demo.append((obs[i, T - 1].astype(np.float32), action[i, T - 1].astype(np.float32), True))
        demos.append(demo)

    return {"demos": demos}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bc-pool", required=True, help="Path to research/'s bc_pool.npz.")
    parser.add_argument("--out", required=True, help="Output .pkl path for train_bc.py's load_demonstrations.")
    parser.add_argument(
        "--no-trim",
        action="store_true",
        help="Skip the 39->35 dim obs trim (use only if bc_pool.npz was already trimmed upstream).",
    )
    args = parser.parse_args()

    feedback_data = convert(args.bc_pool, apply_trim=not args.no_trim)
    with open(args.out, "wb") as f:
        pickle.dump(feedback_data, f)
    print(f"Wrote {len(feedback_data['demos'])} demonstration trajectories to {args.out}")


if __name__ == "__main__":
    main()
