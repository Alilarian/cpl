"""
Phase 3: Build a trajectory pool for demonstrative feedback generation.

Follows the same pattern as create_dataset.py + create_comparison_dataset.py:
  - Rolls out each checkpoint and stores (obs, action, reward, state) at every step
  - Randomly samples fixed-length segments within episodes (same as sample_sequence)
  - state[:,0] of each segment = s_0, used in Phase 4 to restore env for expert rollout

For each environment, iterates over checkpoints at --checkpoint-interval steps,
rolls out --n-episodes-per-checkpoint episodes per checkpoint, randomly samples
--n-segments-per-checkpoint segments of --segment-length from those episodes.

Resumable at two levels:
  - Across checkpoints: each checkpoint's segments are saved to a staging
    directory (<out_dir>/<env>/ckpts/<seed>_step_<N>.npz). Already-finished
    checkpoints are skipped on re-run. Once all checkpoints are done they are
    merged into pool.npz and the staging directory is removed.
  - Within a checkpoint: segments are generated in batches of --save-every
    (rolling out --n-episodes-per-checkpoint fresh episodes per batch), and
    each batch is written to its own small file under
    <out_dir>/<env>/ckpts/<seed>_step_<N>_batches/batch_<k>.npz — a true
    append, not a rewrite of everything collected so far, so cost stays
    linear even for a large --n-segments-per-checkpoint (e.g. 100k from a
    single checkpoint). If the job is killed mid-way, re-running resumes from
    the last completed batch instead of restarting that checkpoint. Batches
    are merged into the checkpoint's staging file (and the batch dir removed)
    once its target segment count is reached.

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


def _scan_checkpoints(search_dir, model_dir, checkpoint_interval=None, checkpoint_steps=None):
    """Find model_<step>.pt files in search_dir, either at a fixed interval or
    matching an explicit set of steps. Returns list of (step, ckpt_path, model_dir).
    """
    wanted = set(checkpoint_steps) if checkpoint_steps is not None else None
    results = []
    for fname in os.listdir(search_dir):
        if not fname.startswith("model_") or not fname.endswith(".pt"):
            continue
        try:
            step = int(fname[len("model_"):-len(".pt")])
        except ValueError:
            continue
        if wanted is not None:
            if step in wanted:
                results.append((step, os.path.join(search_dir, fname), model_dir))
        elif step % checkpoint_interval == 0:
            results.append((step, os.path.join(search_dir, fname), model_dir))
    return results


def get_checkpoint_paths(run_dir, checkpoint_interval=None, checkpoint_steps=None):
    """Return sorted list of (step, ckpt_path, model_dir) tuples.

    Supports two layouts:
      flat  : run_dir/model_<step>.pt          (model_dir = run_dir)
      seeded: run_dir/seed-*/model_<step>.pt   (model_dir = seed subdir)
    Seeded layout is detected when no model_*.pt files exist directly in run_dir.

    If checkpoint_steps is given, only those exact steps are matched (and
    checkpoint_interval is ignored) — raises FileNotFoundError if any requested
    step has no corresponding model_<step>.pt.
    """
    flat = _scan_checkpoints(run_dir, run_dir, checkpoint_interval, checkpoint_steps)
    if flat:
        result = sorted(flat, key=lambda x: x[0])
    else:
        checkpoints = []
        seed_dirs = sorted(
            e for e in os.listdir(run_dir)
            if e.startswith("seed-") and os.path.isdir(os.path.join(run_dir, e))
        )
        for seed in seed_dirs:
            seed_dir = os.path.join(run_dir, seed)
            checkpoints.extend(_scan_checkpoints(seed_dir, seed_dir, checkpoint_interval, checkpoint_steps))
        result = sorted(checkpoints, key=lambda x: x[0])

    if checkpoint_steps is not None:
        missing = set(checkpoint_steps) - {s for s, _, _ in result}
        if missing:
            raise FileNotFoundError(f"Missing checkpoints for steps {sorted(missing)} in {run_dir}")
    return result


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


def generate_checkpoint_segments(model, env, n_segments, segment_length,
                                  episodes_per_batch, batch_dir, save_every):
    """
    Generate n_segments for one checkpoint in batches, appending each batch as
    its own file under batch_dir instead of rewriting everything collected so
    far — keeps per-save cost O(batch), not O(total-so-far).

    Each batch rolls out `episodes_per_batch` fresh episodes and samples
    min(save_every, remaining) segments from them. On resume, already-written
    batch files are counted and generation continues from there.

    Returns list of batch file paths, in order.
    """
    os.makedirs(batch_dir, exist_ok=True)

    batch_paths = sorted(
        (os.path.join(batch_dir, f) for f in os.listdir(batch_dir) if f.startswith("batch_")),
        key=lambda p: int(os.path.basename(p)[len("batch_"):-len(".npz")]),
    )
    def _batch_len(path):
        with np.load(path) as d:
            return d["obs"].shape[0]

    n_done = sum(_batch_len(p) for p in batch_paths)
    if n_done > 0:
        print(f"      resuming: {n_done}/{n_segments} segments already batched "
              f"({len(batch_paths)} batch files)")

    next_idx = len(batch_paths)
    while n_done < n_segments:
        batch_n = min(save_every, n_segments - n_done)
        episodes = rollout_episodes(model, env, episodes_per_batch)
        segs = sample_segments(episodes, batch_n, segment_length)

        batch_path = os.path.join(batch_dir, f"batch_{next_idx:05d}.npz")
        save_npz(batch_path, obs=segs["obs"], action=segs["action"],
                 reward=segs["reward"], state=segs["state"])
        batch_paths.append(batch_path)

        n_done += batch_n
        next_idx += 1
        print(f"      [{n_done:>7,}/{n_segments:>7,} segments batched]")

    return batch_paths


def build_pool_for_env(checkpoints, n_episodes_per_checkpoint,
                       n_segments_per_checkpoint, segment_length, staging_dir,
                       device, save_every):
    """
    Build the full pool for one environment across all checkpoints.

    checkpoints: list of (step, ckpt_path, model_dir) from get_checkpoint_paths.
    Saves each checkpoint's segments immediately to staging_dir/<seed>_step_<N>.npz,
    generated incrementally via generate_checkpoint_segments (see there for the
    within-checkpoint resume mechanism). Already-finished checkpoints are
    skipped on resume. Returns concatenated arrays across all checkpoints.
    """
    os.makedirs(staging_dir, exist_ok=True)

    for ckpt_idx, (step, ckpt_path, model_dir) in enumerate(checkpoints):
        seed_tag   = os.path.basename(model_dir)
        stage_path = os.path.join(staging_dir, f"{seed_tag}_step_{step}.npz")
        batch_dir  = os.path.join(staging_dir, f"{seed_tag}_step_{step}_batches")

        if os.path.exists(stage_path):
            print(f"  [{ckpt_idx+1:>2}/{len(checkpoints)}] {seed_tag} step={step:>7,}  "
                  f"already done — skipping")
            continue

        print(f"  [{ckpt_idx+1:>2}/{len(checkpoints)}] {seed_tag} step={step:>7,}  "
              f"generating {n_segments_per_checkpoint} segments "
              f"({n_episodes_per_checkpoint} eps/batch, save every {save_every}) ...")

        model, env = load_model(model_dir, ckpt_path, device)
        batch_paths = generate_checkpoint_segments(
            model, env, n_segments_per_checkpoint, segment_length,
            n_episodes_per_checkpoint, batch_dir, save_every,
        )

        obs_b, act_b, rew_b, state_b = [], [], [], []
        for p in batch_paths:
            with np.load(p) as d:
                obs_b.append(d["obs"]); act_b.append(d["action"])
                rew_b.append(d["reward"]); state_b.append(d["state"])
        obs, action, reward, state = (
            np.concatenate(obs_b, axis=0), np.concatenate(act_b, axis=0),
            np.concatenate(rew_b, axis=0), np.concatenate(state_b, axis=0),
        )

        n = obs.shape[0]
        avg_reward = reward.sum(axis=1).mean()
        print(f"  → {n} segments  avg_seg_reward={avg_reward:.2f}")

        save_npz(
            stage_path,
            obs=obs,
            action=action,
            reward=reward,
            state=state,
            checkpoint_step=np.full(n, step, dtype=np.int64),
        )
        shutil.rmtree(batch_dir)

    # Merge all staging files in sorted order
    all_obs, all_action, all_reward, all_state, all_ckpt_step = [], [], [], [], []
    for step, _, model_dir in checkpoints:
        seed_tag   = os.path.basename(model_dir)
        stage_path = os.path.join(staging_dir, f"{seed_tag}_step_{step}.npz")
        with np.load(stage_path) as d:
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
        help="Use checkpoints every N training steps (default: 20000). "
             "Ignored if --checkpoint-steps is given.",
    )
    parser.add_argument(
        "--checkpoint-steps", type=int, nargs="+", default=None,
        help="Explicit list of checkpoint steps to use instead of a fixed interval, "
             "e.g. --checkpoint-steps 340000 420000 500000 580000 660000. "
             "Only these exact model_<step>.pt files are rolled out.",
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
        "--save-every", type=int, default=10000,
        help="Segments per batch within a checkpoint (default: 10000). Each batch is "
             "written as its own file (append, not rewrite) so a walltime kill loses "
             "at most one batch's worth of work — see generate_checkpoint_segments.",
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

        checkpoints = get_checkpoint_paths(
            run_dir,
            checkpoint_interval=args.checkpoint_interval,
            checkpoint_steps=args.checkpoint_steps,
        )
        if args.checkpoint_steps is not None:
            assert checkpoints, f"None of the requested checkpoint steps were found in {run_dir}"
        else:
            assert checkpoints, f"No checkpoints at interval={args.checkpoint_interval} in {run_dir}"

        total_segs = len(checkpoints) * args.n_segments_per_checkpoint
        print(f"Checkpoints : {len(checkpoints)}  "
              f"(steps {checkpoints[0][0]:,} → {checkpoints[-1][0]:,})")
        print(f"Episodes/ckpt={args.n_episodes_per_checkpoint}  "
              f"Segments/ckpt={args.n_segments_per_checkpoint}  "
              f"Total≈{total_segs}")
        print(f"{'='*65}")

        obs_arr, action_arr, reward_arr, state_arr, ckpt_arr = build_pool_for_env(
            checkpoints,
            args.n_episodes_per_checkpoint,
            args.n_segments_per_checkpoint,
            args.segment_length,
            staging_dir,
            device,
            args.save_every,
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
