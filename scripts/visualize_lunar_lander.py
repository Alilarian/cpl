"""
Visualize a LunarLander episode and save to an MP4.

Usage:
    python scripts/visualize_lunar_lander.py                  # random policy
    python scripts/visualize_lunar_lander.py --checkpoint path/to/model.pt
    python scripts/visualize_lunar_lander.py --output my_video.mp4 --episodes 3
"""

import argparse
import os

import gymnasium as gymn
import numpy as np

# Optional: only needed when loading a trained checkpoint
try:
    import torch
    import research
    import gym
    HAS_RESEARCH = True
except ImportError:
    HAS_RESEARCH = False


def collect_episode(env, policy_fn):
    """Roll out one episode. Returns list of (frame, obs, reward, done)."""
    obs, _ = env.reset()
    frames, total_reward, steps = [], 0.0, 0
    done = False
    while not done:
        frame = env.render()
        action = policy_fn(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        frames.append(frame)
        total_reward += reward
        steps += 1
    return frames, total_reward, steps


def save_mp4(frames, path, fps=50):
    try:
        import cv2
        h, w = frames[0].shape[:2]
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()
        return "cv2"
    except ImportError:
        pass

    try:
        import imageio.v3 as iio
        import numpy as np
        iio.imwrite(path, np.stack(frames), fps=fps, codec="libx264")
        return "imageio"
    except ImportError:
        pass

    raise RuntimeError(
        "Neither opencv-python nor imageio is installed.\n"
        "Install one of them:  pip install imageio[ffmpeg]  or  pip install opencv-python"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="lunar_lander.mp4")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to a CPL/SAC checkpoint to load a trained policy")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    env = gymn.make(
        "LunarLander-v3",
        continuous=True,
        enable_wind=False,
        turbulence_power=0.0,
        render_mode="rgb_array",
    )
    env.reset(seed=args.seed)

    # Build policy
    if args.checkpoint is not None:
        if not HAS_RESEARCH:
            raise RuntimeError("Could not import research package — needed to load a checkpoint.")
        print(f"Loading checkpoint: {args.checkpoint}")
        # Wrap in gym 0.23 API for CPL model
        ll_env = gym.make("LunarLanderContinuous-cpl-v0")
        from research.utils.config import Config
        model = torch.load(args.checkpoint, map_location="cpu")
        model.eval()
        def policy_fn(obs):
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                action = model.predict({"obs": obs_t}, is_batched=True, sample=False)
            return np.array(action).squeeze(0)
    else:
        print("No checkpoint provided — using random policy.")
        def policy_fn(obs):
            return env.action_space.sample()

    all_frames = []
    for ep in range(args.episodes):
        frames, total_reward, steps = collect_episode(env, policy_fn)
        all_frames.extend(frames)
        print(f"  Episode {ep + 1}: {steps} steps, return = {total_reward:.1f}")

    env.close()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    backend = save_mp4(all_frames, args.output)
    print(f"\nSaved {len(all_frames)} frames to: {args.output}  (via {backend})")


if __name__ == "__main__":
    main()
