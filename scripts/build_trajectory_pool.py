"""
Phase 3: Build a trajectory pool for demonstrative feedback generation.

Follows the same pattern as create_dataset.py + create_comparison_dataset.py:
  - Rolls out each checkpoint and stores (obs, action, reward, state) at every step
  - Randomly samples fixed-length segments within episodes (same as sample_sequence)
  - state[:,0] of each segment = s_0, used in Phase 4 to restore env for expert rollout

For each environment, iterates over checkpoints at --checkpoint-interval steps,
rolls out --n-episodes-per-checkpoint episodes per checkpoint, randomly samples
--n-segments-per-checkpoint segments of --segment-length from those episodes.

Resumable: each checkpoint's segments are saved immediately to a staging directory
(<out_dir>/<env>/ckpts/step_<N>.npz).  On re-run, already-saved checkpoints are
skipped automatically.  When all checkpoints are done they are merged into pool.npz
and the staging directory is removed.

Output per environment (--output-dir/<env>/pool.npz):
    obs             : (N, T, obs_dim)
    action          : (N, T, act_dim)
    reward          : (N, T)
    state           : (N, T, state_dim)   state[:,0] is s_0 for each segment
    checkpoint_step : (N,)                which checkpoint produced this segment

Usage:
    python scripts/build_trajectory_pool.py \\
        --oracle-runs-base runs/runs/chpc/oracle_sac_all \\
        --envs mw_drawer-open-v2 mw_door-open-v2 mw_bin-picking-v2 \\
               mw_button-press-v2 mw_plate-slide-v2 \\
        --checkpoint-interval 20000 \\
        --n-episodes-per-checkpoint 10 \\
        --n-segments-per-checkpoint 400 \\
        --segment-length 50 \\
        --output-dir datasets/demo_pool
"""

import argparse
import io
import os
import shutil

import numpy as np
import torch

from research.utils.config import Config


ENVS_ALL = [
    "mw_bin-picking-v2",
    "mw_button-press-v2",
    "mw_door-open-v2",
    "mw_drawer-open-v2",
    "mw_plate-slide-v2",
]


def get_checkpoint_paths(run_dir, checkpoint_interval):
    """Return sorted list of (step, path) for model_<step>.pt at the given interval."""
    checkpoints = []
    for fname in os.listdir(run_dir):
        if not fname.startswith("model_") or not fname.endswith(".pt"):
            continue
        try:
            step = int(fname[len("model_"):-len(".pt")])
        except ValueError:
            continue
        if step % checkpoint_interval == 0:
            checkpoints.append((step, os.path.join(run_dir, fname)))
    checkpoints.sort(key=lambda x: x[0])
    return checkpoints


def load_model(run_dir, checkpoint_path, device):
    config = Config.load(run_dir)
    config["checkpoint"] = None
    config = config.parse()
    env_fn = config.get_train_env_fn() or config.get_eval_env_fn()
    env = env_fn()
    model = config.get_model(
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
    )
    model.load(checkpoint_path)
    model.eval()
    return model, env


def rollout_episodes(model, env, n_episodes):
    """
    Roll out n_episodes full episodes deterministically.
    State is captured before each step, matching create_dataset.py convention.
    Returns list of per-episode dicts with arrays of shape (ep_len, dim).
    """
    assert hasattr(env, "get_state"), "Environment must support get_state()"

    episodes = []
    for _ in range(n_episodes):
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]

        ep_obs, ep_act, ep_rew, ep_states = [], [], [], []
        done = False

        while not done:
            # Capture state before the step (same as create_dataset.py)
            state = env.get_state()

            with torch.no_grad():
                action = model.predict(dict(obs=obs), sample=False)
            action = np.clip(action, env.action_space.low, env.action_space.high)

            result = env.step(action)
            if len(result) == 5:
                next_obs, reward, terminated, truncated, _ = result
                done = terminated or truncated
            else:
                next_obs, reward, done, _ = result

            ep_obs.append(obs.astype(np.float32))
            ep_act.append(action.astype(np.float32))
            ep_rew.append(float(reward))
            ep_states.append(state.astype(np.float32))

            obs = next_obs

        episodes.append({
            "obs":    np.stack(ep_obs,    axis=0),
            "action": np.stack(ep_act,    axis=0),
            "reward": np.array(ep_rew,    dtype=np.float32),
            "state":  np.stack(ep_states, axis=0),
        })

    return episodes


def sample_segments(episodes, n_segments, segment_length):
    """
    Randomly sample n_segments of fixed length from the collected episodes.

    Matches create_comparison_dataset.py / sample_sequence logic:
      - Only sample from episodes long enough to contain a full segment
      - Episode selection is weighted proportionally to the number of valid
        start positions (longer episodes contribute more segments)
      - Start position is uniformly random within the valid range

    Returns dict of arrays shaped (n_segments, segment_length, dim).
    state[:,0] of the returned array is s_0 for each segment.
    """
    valid_eps = [ep for ep in episodes if ep["obs"].shape[0] >= segment_length]
    assert valid_eps, f"No episodes with length >= {segment_length}"

    # Number of valid start positions per episode
    n_valid_starts = np.array([ep["obs"].shape[0] - segment_length + 1 for ep in valid_eps])
    probs = n_valid_starts / n_valid_starts.sum()

    seg_obs, seg_act, seg_rew, seg_states = [], [], [], []

    for _ in range(n_segments):
        ep_idx = np.random.choice(len(valid_eps), p=probs)
        ep     = valid_eps[ep_idx]
        ep_len = ep["obs"].shape[0]

        start = np.random.randint(0, ep_len - segment_length + 1)
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


def save_npz(path, **arrays):
    """Atomically save a compressed npz file (write to tmp, then rename)."""
    tmp_path = path + ".tmp"
    with io.BytesIO() as buf:
        np.savez_compressed(buf, **arrays)
        buf.seek(0)
        with open(tmp_path, "wb") as f:
            f.write(buf.read())
    os.replace(tmp_path, path)


def build_pool_for_env(run_dir, checkpoints, n_episodes_per_checkpoint,
                       n_segments_per_checkpoint, segment_length, staging_dir, device):
    """
    Build the full pool for one environment across all checkpoints.

    Saves each checkpoint's segments immediately to staging_dir/step_<N>.npz.
    Already-saved checkpoints are skipped on resume.
    Returns concatenated arrays across all checkpoints.
    """
    os.makedirs(staging_dir, exist_ok=True)

    for ckpt_idx, (step, ckpt_path) in enumerate(checkpoints):
        stage_path = os.path.join(staging_dir, f"step_{step}.npz")

        if os.path.exists(stage_path):
            print(f"  [{ckpt_idx+1:>2}/{len(checkpoints)}] step={step:>7,}  "
                  f"already done — skipping")
            continue

        print(f"  [{ckpt_idx+1:>2}/{len(checkpoints)}] step={step:>7,}  "
              f"rolling out {n_episodes_per_checkpoint} eps ...", end="", flush=True)

        model, env = load_model(run_dir, ckpt_path, device)
        episodes   = rollout_episodes(model, env, n_episodes_per_checkpoint)
        segs       = sample_segments(episodes, n_segments_per_checkpoint, segment_length)

        n = segs["obs"].shape[0]
        avg_reward = segs["reward"].sum(axis=1).mean()
        print(f"  → {n} segments  avg_seg_reward={avg_reward:.2f}")

        save_npz(
            stage_path,
            obs=segs["obs"],
            action=segs["action"],
            reward=segs["reward"],
            state=segs["state"],
            checkpoint_step=np.full(n, step, dtype=np.int64),
        )

    # Merge all staging files in sorted order
    all_obs, all_action, all_reward, all_state, all_ckpt_step = [], [], [], [], []
    for step, _ in checkpoints:
        stage_path = os.path.join(staging_dir, f"step_{step}.npz")
        d = np.load(stage_path)
        all_obs.append(d["obs"])
        all_action.append(d["action"])
        all_reward.append(d["reward"])
        all_state.append(d["state"])
        all_ckpt_step.append(d["checkpoint_step"])

    return (
        np.concatenate(all_obs,       axis=0),
        np.concatenate(all_action,    axis=0),
        np.concatenate(all_reward,    axis=0),
        np.concatenate(all_state,     axis=0),
        np.concatenate(all_ckpt_step, axis=0),
    )


def main():
    parser = argparse.ArgumentParser(description="Build trajectory pool for demonstrative feedback.")
    parser.add_argument(
        "--oracle-runs-base", type=str, required=True,
        help="Base dir with one subdir per env, e.g. runs/runs/chpc/oracle_sac_all",
    )
    parser.add_argument(
        "--envs", nargs="+", default=ENVS_ALL,
        help="Environments to process (default: all 5 non-sweep envs).",
    )
    parser.add_argument(
        "--checkpoint-interval", type=int, default=20000,
        help="Use checkpoints every N training steps (default: 20000).",
    )
    parser.add_argument(
        "--n-episodes-per-checkpoint", type=int, default=10,
        help="Episodes to roll out per checkpoint (default: 10).",
    )
    parser.add_argument(
        "--n-segments-per-checkpoint", type=int, default=400,
        help="Segments randomly sampled from those episodes (default: 400). "
             "50 checkpoints × 400 = 20k segments/env.",
    )
    parser.add_argument(
        "--segment-length", type=int, default=50,
        help="Fixed segment length in steps (default: 50).",
    )
    parser.add_argument(
        "--output-dir", type=str, default="datasets/demo_pool",
        help="Root output directory (default: datasets/demo_pool).",
    )
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Device: {device}\n")

    for env_name in args.envs:
        run_dir = os.path.join(args.oracle_runs_base, env_name)
        assert os.path.isdir(run_dir), f"Run dir not found: {run_dir}"

        out_dir     = os.path.join(args.output_dir, env_name)
        pool_path   = os.path.join(out_dir, "pool.npz")
        staging_dir = os.path.join(out_dir, "ckpts")

        print(f"{'='*65}")
        print(f"Environment : {env_name}")

        if os.path.exists(pool_path):
            print(f"  pool.npz already exists — skipping (delete to regenerate)")
            print(f"{'='*65}\n")
            continue

        checkpoints = get_checkpoint_paths(run_dir, args.checkpoint_interval)
        assert checkpoints, f"No checkpoints at interval={args.checkpoint_interval} in {run_dir}"

        total_segs = len(checkpoints) * args.n_segments_per_checkpoint
        print(f"Checkpoints : {len(checkpoints)}  "
              f"(steps {checkpoints[0][0]:,} → {checkpoints[-1][0]:,})")
        print(f"Episodes/ckpt={args.n_episodes_per_checkpoint}  "
              f"Segments/ckpt={args.n_segments_per_checkpoint}  "
              f"Total≈{total_segs}")
        print(f"{'='*65}")

        obs_arr, action_arr, reward_arr, state_arr, ckpt_arr = build_pool_for_env(
            run_dir, checkpoints,
            args.n_episodes_per_checkpoint,
            args.n_segments_per_checkpoint,
            args.segment_length,
            staging_dir,
            device,
        )

        N = obs_arr.shape[0]
        print(f"\nPool summary for {env_name}:")
        print(f"  N segments  : {N}")
        print(f"  obs         : {obs_arr.shape}")
        print(f"  action      : {action_arr.shape}")
        print(f"  state       : {state_arr.shape}  ([:,0] = s_0)")
        print(f"  ckpt steps  : {np.unique(ckpt_arr)}")

        os.makedirs(out_dir, exist_ok=True)
        save_npz(
            pool_path,
            obs=obs_arr,
            action=action_arr,
            reward=reward_arr,
            state=state_arr,
            checkpoint_step=ckpt_arr,
        )
        print(f"  Saved → {pool_path}")

        # Remove staging files now that pool.npz is safely written
        shutil.rmtree(staging_dir)
        print(f"  Staging dir removed: {staging_dir}\n")

    print("Done.")


if __name__ == "__main__":
    main()
