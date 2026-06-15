"""
Generate a trajectory pool for the PointMass navigation environment.

Collects fixed-length segments from policy tiers that span the quality
spectrum from expert to random.  The pool has the same schema as the MetaWorld
pool produced by build_trajectory_pool.py and is consumed by
generate_pm_feedback.py.

Policy tiers
------------
  0  expert              — heuristic straight-to-goal with trap avoidance
  1  noisy expert ε=0.05 — 5%  random, 95% expert
  2  noisy expert ε=0.10 — 10% random, 90% expert
  3  noisy expert ε=0.25 — 25% random, 75% expert
  4  noisy expert ε=0.50 — 50% random, 50% expert
  5  noisy expert ε=0.75 — 75% random, 25% expert

Output (--out)
--------------
  obs             : (N, T, 2)   float32  positions after each step
  action          : (N, T, 2)   float32  Cartesian actions in [-1,1]²
  reward          : (N, T)      float32  shaped reward (step_cost=0.01)
  state           : (N, T, 3)   float32  [x, y, ep_step] BEFORE each action
                                         state[:,0] = s_0 for rollout_from_state
  checkpoint_step : (N,)        int64    tier index in place of checkpoint step

Usage
-----
python scripts/generate_pm_pool.py \\
    --out datasets/pm/pool.npz \\
    --n-per-tier 500 \\
    --segment-len 50 \\
    --seed 42
"""

import argparse
import importlib.util
import io
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_module(dotted_name, rel_path):
    path = os.path.join(_ROOT, rel_path)
    spec = importlib.util.spec_from_file_location(dotted_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[dotted_name] = mod
    spec.loader.exec_module(mod)
    return mod


_pm = _load_module("research.envs.pointmass", "research/envs/pointmass.py")
PointMassGymEnv = _pm.PointMassGymEnv


# Tier index → (label, epsilon)
TIERS = {
    0: ("expert",              0.00),
    1: ("noisy expert ε=0.05", 0.05),
    2: ("noisy expert ε=0.10", 0.10),
    3: ("noisy expert ε=0.25", 0.25),
    4: ("noisy expert ε=0.50", 0.50),
    5: ("noisy expert ε=0.75", 0.75),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def save_npz(path, **arrays):
    tmp = path + ".tmp"
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with io.BytesIO() as buf:
        np.savez_compressed(buf, **arrays)
        buf.seek(0)
        with open(tmp, "wb") as f:
            f.write(buf.read())
    os.replace(tmp, path)


def rollout_episodes(env, policy_fn, n_episodes):
    """
    Roll out policy_fn for n_episodes.  state captured BEFORE each step so
    that state[0] of any segment is the exact env state when the first action
    was taken — compatible with env.set_state() for expert-rollout replay.

    Returns list of dicts:
        {"obs": (L,2), "action": (L,2), "reward": (L,), "state": (L,3)}
    """
    episodes = []
    for _ in range(n_episodes):
        obs = env.reset()
        ep_obs, ep_act, ep_rew, ep_states = [], [], [], []
        done = False
        while not done:
            state_before = env.get_state()          # [x, y, step] before action
            action = policy_fn(obs)
            obs, reward, done, _ = env.step(action)
            ep_obs.append(obs.copy())
            ep_act.append(action.copy())
            ep_rew.append(float(reward))
            ep_states.append(state_before.copy())
        if len(ep_obs) > 0:
            episodes.append({
                "obs":    np.array(ep_obs,    dtype=np.float32),
                "action": np.array(ep_act,    dtype=np.float32),
                "reward": np.array(ep_rew,    dtype=np.float32),
                "state":  np.array(ep_states, dtype=np.float32),
            })
    return episodes

def sample_segments(episodes, n_segments, segment_length, rng):
    """
    Sample n_segments of exactly segment_length steps from episodes.
    Longer episodes contribute proportionally more segments.
    Returns None if no episode is long enough.
    """
    valid = [ep for ep in episodes if ep["obs"].shape[0] >= segment_length]
    if not valid:
        return None
    weights = np.array([ep["obs"].shape[0] - segment_length + 1
                        for ep in valid], dtype=np.float64)
    weights /= weights.sum()

    seg_obs, seg_act, seg_rew, seg_states = [], [], [], []
    for _ in range(n_segments):
        idx   = rng.choice(len(valid), p=weights)
        ep    = valid[idx]
        L     = ep["obs"].shape[0]
        start = int(rng.integers(0, L - segment_length + 1))
        end   = start + segment_length
        seg_obs.append(ep["obs"][start:end])
        seg_act.append(ep["action"][start:end])
        seg_rew.append(ep["reward"][start:end])
        seg_states.append(ep["state"][start:end])

    return {
        "obs":    np.stack(seg_obs,    axis=0),
        "action": np.stack(seg_act,    axis=0),
        "reward": np.stack(seg_rew,    axis=0),
        "state":  np.stack(seg_states, axis=0),
    }

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate PointMass trajectory pool for preference labeling."
    )
    parser.add_argument("--out",         type=str, required=True,
                        help="Output .npz path")
    parser.add_argument("--n-per-tier",  type=int, default=500,
                        help="Segments per tier (default: 500)")
    parser.add_argument("--segment-len", type=int, default=50,
                        help="Fixed segment length T (default: 50)")
    parser.add_argument("--n-rollouts",  type=int, default=None,
                        help="Episodes to roll out per tier before sampling. "
                             "Default: max(200, 5 × n-per-tier).")
    parser.add_argument("--tiers",       type=int, nargs="+",
                        default=list(TIERS.keys()),
                        help="Tier indices to include (default: 0..5)")
    parser.add_argument("--seed",        type=int, default=42)
    args = parser.parse_args()

    n_rollouts = args.n_rollouts or max(200, 5 * args.n_per_tier)
    rng = np.random.default_rng(args.seed)

    # Shaped reward so pool segments carry a meaningful reward signal
    env = PointMassGymEnv(step_cost=0.01)

    print("=" * 65)
    print("PointMass pool generation")
    print(f"  Tiers         : {[TIERS[t][0] for t in args.tiers]}")
    print(f"  Segs / tier   : {args.n_per_tier}")
    print(f"  Rollouts/tier : {n_rollouts}")
    print(f"  Segment len T : {args.segment_len}")
    print(f"  Seed          : {args.seed}")
    print(f"  Output        : {args.out}")
    print("=" * 65)

    all_obs, all_act, all_rew, all_state, all_ckpt = [], [], [], [], []

    for tier in args.tiers:
        label, eps = TIERS[tier]
        # Per-tier seed keeps tiers reproducible independently
        np.random.seed(args.seed + tier)

        policy_fn = (env.make_expert_policy() if eps == 0.0
                     else env.make_noisy_expert_policy(eps=eps))

        print(f"\n[tier {tier}] {label}")
        print(f"  Rolling out {n_rollouts} episodes …")
        episodes = rollout_episodes(env, policy_fn, n_rollouts)

        valid = [ep for ep in episodes if ep["obs"].shape[0] >= args.segment_len]
        ep_lengths = [ep["obs"].shape[0] for ep in episodes]
        succ = sum(1 for ep in episodes
                   if any(r > 0 for r in ep["reward"]))
        print(f"  Episodes: total={len(episodes)}  valid_for_segs={len(valid)}"
              f"  success_rate={100*succ/len(episodes):.0f}%")
        print(f"  Episode length: mean={np.mean(ep_lengths):.0f}"
              f"  min={np.min(ep_lengths)}  max={np.max(ep_lengths)}")

        if not valid:
            print(f"  WARNING: no episodes long enough for tier {tier}, skipping.")
            continue

        segs = sample_segments(valid, args.n_per_tier, args.segment_len, rng)
        if segs is None:
            print(f"  WARNING: segment sampling failed for tier {tier}, skipping.")
            continue

        avg_rew = segs["reward"].sum(axis=1).mean()
        print(f"  Sampled {args.n_per_tier} segments  "
              f"avg_episode_reward={avg_rew:.3f}")

        all_obs.append(segs["obs"])
        all_act.append(segs["action"])
        all_rew.append(segs["reward"])
        all_state.append(segs["state"])
        all_ckpt.append(np.full(args.n_per_tier, tier, dtype=np.int64))

    if not all_obs:
        print("\nERROR: No segments collected. Check tier settings.")
        return

    obs_out   = np.concatenate(all_obs,   axis=0).astype(np.float32)
    act_out   = np.concatenate(all_act,   axis=0).astype(np.float32)
    rew_out   = np.concatenate(all_rew,   axis=0).astype(np.float32)
    state_out = np.concatenate(all_state, axis=0).astype(np.float32)
    ckpt_out  = np.concatenate(all_ckpt,  axis=0)

    N = obs_out.shape[0]
    print(f"\n{'=' * 65}")
    print(f"Pool summary")
    print(f"  Total segments  : {N}")
    print(f"  obs             : {obs_out.shape}")
    print(f"  action          : {act_out.shape}")
    print(f"  reward          : {rew_out.shape}")
    print(f"  state           : {state_out.shape}  (state[:,0] = s_0)")
    print(f"  checkpoint_step : {ckpt_out.shape}  (tier index 0-5)")
    print(f"  Avg total reward: {rew_out.sum(axis=1).mean():.3f}")
    for tier in args.tiers:
        mask = ckpt_out == tier
        if mask.any():
            print(f"    tier {tier} ({TIERS[tier][0]}): "
                  f"avg_reward={rew_out[mask].sum(axis=1).mean():.3f}")

    save_npz(
        args.out,
        obs=obs_out,
        action=act_out,
        reward=rew_out,
        state=state_out,
        checkpoint_step=ckpt_out,
    )
    print(f"\nSaved → {args.out}")


if __name__ == "__main__":
    main()
