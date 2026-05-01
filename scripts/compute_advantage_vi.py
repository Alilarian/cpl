"""
Precompute the soft-optimal advantage table for PointMassGymEnv.

Runs discrete value iteration on a Cartesian state × action grid and saves
V*, Q*, A* tables to a compressed .npz file that the preference-labeling
script can load in microseconds per query.

Usage
-----
# With a trained SAC checkpoint (reads learned alpha automatically):
python scripts/compute_advantage_vi.py \
    --out runs/pm_sac_oracle/advantage_vi.npz \
    --model runs/pm_sac_oracle/best_model.pt

# With an explicit temperature (e.g. for ablations):
python scripts/compute_advantage_vi.py \
    --out runs/pm_sac_oracle/advantage_vi.npz \
    --alpha 0.05

# Finer grid (slower, ~4× memory):
python scripts/compute_advantage_vi.py \
    --out runs/pm_sac_oracle/advantage_vi_200.npz \
    --model runs/pm_sac_oracle/best_model.pt \
    --n-states 200 --n-actions 15
"""

import argparse
import importlib.util
import os
import sys
import time

import numpy as np

# Load PointMassGymEnv and AdvantageVI directly from their source files to
# avoid importing the research package __init__.py, which triggers
# MetaWorld → mujoco_py → compile failure on machines without a working
# MuJoCo build.  Neither module depends on anything from that chain.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_module(dotted_name: str, rel_path: str):
    path = os.path.join(_ROOT, rel_path)
    spec = importlib.util.spec_from_file_location(dotted_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[dotted_name] = mod
    spec.loader.exec_module(mod)
    return mod


_pm = _load_module("research.envs.pointmass", "research/envs/pointmass.py")
_avi = _load_module("research.utils.advantage_vi", "research/utils/advantage_vi.py")

PointMassGymEnv = _pm.PointMassGymEnv
AdvantageVI = _avi.AdvantageVI


# ---------------------------------------------------------------------------
# Alpha loading
# ---------------------------------------------------------------------------

def alpha_from_checkpoint(model_path: str) -> float:
    """Read the learned SAC temperature from a saved checkpoint.

    The base Algorithm saves nn.Parameters (including log_alpha) directly as
    tensors under their attribute name.  So the checkpoint layout is:
        { "log_alpha": <scalar Parameter>, "network": <state_dict>, ... }
    """
    import torch

    ckpt = torch.load(model_path, map_location="cpu")
    if "log_alpha" not in ckpt:
        raise KeyError(
            f"'log_alpha' not found in checkpoint.  "
            f"Available keys: {list(ckpt.keys())}"
        )
    alpha = float(ckpt["log_alpha"].exp().item())
    print(f"[VI] read α={alpha:.5f} from {model_path}")
    return alpha


# ---------------------------------------------------------------------------
# Sanity-check table
# ---------------------------------------------------------------------------

SANITY_CASES = [
    ("at goal     ", [0.3, 0.3], [0.0,  0.0]),
    ("at trap     ", [0.7, 0.7], [0.0,  0.0]),
    ("toward goal ", [0.0, 0.0], [1.0,  1.0]),   # corner → moving right-up
    ("toward trap ", [0.5, 0.5], [0.5,  0.5]),   # center → toward trap
    ("mid-field   ", [0.5, 0.99], [0.0, 0.0]),   # far from both, zero action
]


def sanity_check(avi: AdvantageVI) -> None:
    print("\n[VI] Sanity check")
    print(f"  {'description':15s}  {'obs':>18s}  {'A*':>8s}  {'V':>8s}")
    print("  " + "-" * 57)
    cell = 1.0 / avi.N
    for desc, obs, act in SANITY_CASES:
        obs_a = np.array(obs, dtype=np.float32)
        act_a = np.array(act, dtype=np.float32)
        a_val = avi.query(obs_a, act_a)
        i = int(np.clip(obs_a[0] / cell, 0, avi.N - 1))
        j = int(np.clip(obs_a[1] / cell, 0, avi.N - 1))
        v_val = float(avi.V[i, j])
        print(f"  {desc}  obs={obs}  A*={a_val:+8.4f}  V={v_val:+8.4f}")

    # Value range
    v_min, v_max = float(avi.V.min()), float(avi.V.max())
    a_min, a_max = float(avi.A_star.min()), float(avi.A_star.max())
    print(f"\n  V  range: [{v_min:.4f}, {v_max:.4f}]")
    print(f"  A* range: [{a_min:.4f}, {a_max:.4f}]")

    # Quick policy-consistency check: expert direction should have A* ≥ 0
    # from a point that is not in the goal or trap.
    env = avi.env
    if env is not None:
        expert = env.make_expert_policy()
        test_obs = np.array([0.0, 0.0], dtype=np.float32)
        test_act = expert(test_obs)
        a_expert = avi.query(test_obs, test_act)
        random_adv = np.array([
            avi.query(test_obs, env.action_space.sample()) for _ in range(20)
        ])
        print(
            f"\n  Expert action A* from (0,0): {a_expert:+.4f}  "
            f"(mean of 20 random actions: {random_adv.mean():+.4f})"
        )
        if a_expert < random_adv.mean():
            print("  [WARNING] expert advantage is below random mean — check alpha")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Precompute soft-optimal advantage table via value iteration."
    )
    parser.add_argument(
        "--out", required=True,
        help="Output path for the .npz table (e.g. runs/pm_sac_oracle/advantage_vi.npz)",
    )
    parser.add_argument(
        "--model", default=None,
        help="Path to SAC best_model.pt; alpha is read from its log_alpha parameter",
    )
    parser.add_argument(
        "--alpha", type=float, default=None,
        help="Explicit temperature α (overrides --model if both are given)",
    )
    parser.add_argument("--n-states", type=int, default=100,
                        help="Grid side length N (N² states, default 100)")
    parser.add_argument("--n-actions", type=int, default=11,
                        help="Action-grid points per axis (n² actions, default 11→121)")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--max-iter", type=int, default=3000)
    args = parser.parse_args()

    # Resolve alpha
    if args.alpha is not None:
        alpha = args.alpha
        if args.model is not None:
            print("[VI] --alpha overrides --model for temperature")
    elif args.model is not None:
        alpha = alpha_from_checkpoint(args.model)
    else:
        alpha = 0.1
        print(f"[VI] Warning: no --alpha or --model given; using default α={alpha}")

    # Build env with the same goal/trap/thresholds used during SAC training
    env = PointMassGymEnv(step_cost=0.01)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)

    t0 = time.time()
    avi = AdvantageVI(
        env,
        alpha=alpha,
        N=args.n_states,
        n_act=args.n_actions,
        gamma=args.gamma,
        tol=args.tol,
        max_iter=args.max_iter,
    ).build(verbose=True)
    elapsed = time.time() - t0
    print(f"[VI] build time: {elapsed:.1f} s")

    avi.save(args.out)
    sanity_check(avi)


if __name__ == "__main__":
    main()
