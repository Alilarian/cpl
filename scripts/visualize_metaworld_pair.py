"""
Visualize a paired MetaWorld trajectory comparison.

Each frame shows (side-by-side):
  [Traj A render | XY path panel | Traj B render]

Overlays on each rendered frame:
  - Current reward in top-right corner (green = positive, red = negative)
  - Cumulative return in bottom-right corner
  - Timestep counter in top-left corner

The XY path panel shows:
  - End-effector (EE) and object XY paths for both trajectories
  - A moving dot at the current timestep

Dataset structure:
  - 2N segments total; pair i = segment i (traj A) vs segment i+N (traj B)
  - obs[:, 0:3]  = end-effector XYZ (from trim_mw_obs)
  - obs[:, 4:7]  = object XYZ
  - reward[:, t] = per-timestep reward

Usage:
    python scripts/visualize_metaworld_pair.py \\
        --dataset datasets/mw/pref/mw_drawer-open-v2_ep2500_n0.3.npz \\
        --env mw_drawer-open-v2 \\
        --pair 0

    python scripts/visualize_metaworld_pair.py \\
        --dataset datasets/mw/pref/mw_plate-slide-v2_ep2500_n0.3.npz \\
        --env mw_plate-slide-v2 \\
        --pair random \\
        --output results/plate_pair \\
        --resolution 128 \\
        --label-key rl_sum
"""

import argparse
import os
import sys

import numpy as np


# ---------------------------------------------------------------------------
# Frame annotation helpers
# ---------------------------------------------------------------------------

def annotate_frame(frame, reward, cum_return, timestep, max_timestep,
                   advantage=None):
    """Overlay reward and return text on an (H, W, 3) uint8 frame.

    Requires Pillow.
    """
    from PIL import Image, ImageDraw
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    h, w = frame.shape[:2]

    # Reward in top-right (green if positive, red if negative/zero)
    rew_color = (80, 220, 80) if reward > 0 else (220, 80, 80)
    draw.text((w - 80, 4), f"r={reward:+.3f}", fill=rew_color)

    # Cumulative return in bottom-right
    draw.text((w - 80, h - 16), f"R={cum_return:+.2f}", fill=(220, 220, 80))

    # Timestep top-left
    draw.text((4, 4), f"t={timestep}/{max_timestep}", fill=(200, 200, 200))

    # Optional advantage in top-right below reward
    if advantage is not None:
        adv_color = (80, 180, 220) if advantage >= 0 else (220, 120, 80)
        draw.text((w - 80, 18), f"A={advantage:+.3f}", fill=adv_color)

    return np.array(img)


# ---------------------------------------------------------------------------
# Path panel helpers
# ---------------------------------------------------------------------------

def make_path_panel(ee_a, obj_a, ee_b, obj_b, current_t, size=256):
    """Render an XY bird's-eye path panel as a (size, size, 3) uint8 array.

    Args:
        ee_a:  (T, 3) end-effector positions for traj A
        obj_a: (T, 3) object positions for traj A
        ee_b:  (T, 3) end-effector positions for traj B
        obj_b: (T, 3) object positions for traj B
        current_t: current timestep (dot drawn here)
        size: panel pixel size
    Returns:
        (size, size, 3) uint8 numpy array
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import io
    from PIL import Image

    fig, ax = plt.subplots(figsize=(3, 3), dpi=size // 3)
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    all_x = np.concatenate([ee_a[:, 0], ee_b[:, 0], obj_a[:, 0], obj_b[:, 0]])
    all_y = np.concatenate([ee_a[:, 1], ee_b[:, 1], obj_a[:, 1], obj_b[:, 1]])
    pad = 0.05
    xmin, xmax = all_x.min() - pad, all_x.max() + pad
    ymin, ymax = all_y.min() - pad, all_y.max() + pad
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    T = ee_a.shape[0]
    # Color along trajectory by timestep (light = early, dark = late)
    for (ee, obj, ee_col, obj_col, label) in [
        (ee_a, obj_a, "#4fc3f7", "#80cbc4", "A"),
        (ee_b, obj_b, "#f48fb1", "#ffcc80", "B"),
    ]:
        # Full faded path
        ax.plot(ee[:, 0], ee[:, 1], color=ee_col, alpha=0.25, linewidth=1.2)
        ax.plot(obj[:, 0], obj[:, 1], color=obj_col, alpha=0.25, linewidth=1.2,
                linestyle="--")
        # Active path up to current_t
        t = min(current_t + 1, T)
        ax.plot(ee[:t, 0], ee[:t, 1], color=ee_col, linewidth=1.8,
                label=f"EE {label}")
        ax.plot(obj[:t, 0], obj[:t, 1], color=obj_col, linewidth=1.8,
                linestyle="--", label=f"Obj {label}")
        # Current position dot
        ax.scatter(ee[current_t, 0], ee[current_t, 1], color=ee_col,
                   s=30, zorder=5)
        ax.scatter(obj[current_t, 0], obj[current_t, 1], color=obj_col,
                   s=25, marker="^", zorder=5)

    ax.legend(fontsize=5, loc="upper right", framealpha=0.4,
              labelcolor="white", facecolor="#1a1a2e", edgecolor="none")
    ax.set_xlabel("X", color="white", fontsize=7)
    ax.set_ylabel("Y", color="white", fontsize=7)
    ax.tick_params(colors="gray", labelsize=6)
    ax.set_title(f"XY Path  t={current_t}", color="white", fontsize=7, pad=3)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444466")

    fig.tight_layout(pad=0.3)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    panel = np.array(Image.open(buf).convert("RGB"))
    buf.close()
    return panel


def resize_panel(panel, target_h):
    """Resize panel to target_h while keeping aspect ratio."""
    from PIL import Image
    ph, pw = panel.shape[:2]
    new_w = int(pw * target_h / ph)
    return np.array(Image.fromarray(panel).resize((new_w, target_h),
                                                   Image.LANCZOS))


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def render_trajectory(env, states):
    """Render every timestep of a trajectory by restoring MuJoCo state.

    Returns list of (H, W, 3) uint8 frames.
    """
    frames = []
    for t in range(states.shape[0]):
        env.set_state(states[t])
        img = env._get_image()              # (3, H, W)
        frames.append(img.transpose(1, 2, 0))   # → (H, W, 3)
    return frames


# ---------------------------------------------------------------------------
# Video / image savers
# ---------------------------------------------------------------------------

def save_mp4(frames, path, fps=20):
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
        iio.imwrite(path, np.stack(frames), fps=fps, codec="libx264")
        return "imageio"
    except ImportError:
        pass
    raise RuntimeError(
        "Need opencv-python or imageio[ffmpeg]:\n"
        "  pip install imageio[ffmpeg]   or   pip install opencv-python"
    )


def save_contact_sheet(frames_a, frames_b, path, ee_a, obj_a, ee_b, obj_b,
                       rewards_a, rewards_b, n_cols=8,
                       label_a="Traj A", label_b="Traj B"):
    """Save PNG contact sheet: sampled frames (rows A/B) + full path chart."""
    from PIL import Image, ImageDraw
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import io

    T = len(frames_a)
    indices = np.linspace(0, T - 1, n_cols, dtype=int)
    h, w = frames_a[0].shape[:2]
    banner_h = 22
    margin = 3
    path_h = h * 2 + banner_h * 2 + margin * 3   # match total height of both rows

    # Build path chart at full height
    fig, ax = plt.subplots(figsize=(3, path_h / (h * 2 / 3)), dpi=80)
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    all_x = np.concatenate([ee_a[:, 0], ee_b[:, 0], obj_a[:, 0], obj_b[:, 0]])
    all_y = np.concatenate([ee_a[:, 1], ee_b[:, 1], obj_a[:, 1], obj_b[:, 1]])
    pad = 0.05
    ax.set_xlim(all_x.min() - pad, all_x.max() + pad)
    ax.set_ylim(all_y.min() - pad, all_y.max() + pad)

    for (ee, obj, ec, oc, lbl) in [
        (ee_a, obj_a, "#4fc3f7", "#80cbc4", "A"),
        (ee_b, obj_b, "#f48fb1", "#ffcc80", "B"),
    ]:
        ax.plot(ee[:, 0], ee[:, 1], color=ec, linewidth=2.0, label=f"EE {lbl}")
        ax.plot(obj[:, 0], obj[:, 1], color=oc, linewidth=2.0, linestyle="--",
                label=f"Obj {lbl}")
        ax.scatter(ee[0, 0], ee[0, 1], color=ec, marker="o", s=50, zorder=5)
        ax.scatter(ee[-1, 0], ee[-1, 1], color=ec, marker="*", s=80, zorder=5)

    ax.legend(fontsize=7, loc="upper right", framealpha=0.4,
              labelcolor="white", facecolor="#1a1a2e", edgecolor="none")
    ax.set_xlabel("X", color="white", fontsize=8)
    ax.set_ylabel("Y", color="white", fontsize=8)
    ax.set_title("Full XY Path", color="white", fontsize=9, pad=4)
    ax.tick_params(colors="gray", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444466")
    fig.tight_layout(pad=0.4)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    path_img = np.array(Image.open(buf).convert("RGB"))
    buf.close()

    # Resize path chart to match total frame-rows height
    ph, pw = path_img.shape[:2]
    scale = path_h / ph
    new_pw = int(pw * scale)
    path_img = np.array(Image.fromarray(path_img).resize((new_pw, path_h),
                                                          Image.LANCZOS))

    # Build sheet
    total_w = n_cols * (w + margin) - margin + margin + path_img.shape[1]
    total_h = path_h
    sheet = Image.new("RGB", (total_w, total_h), color=(20, 20, 30))
    draw = ImageDraw.Draw(sheet)

    for row, (frames, lbl, rewards) in enumerate([
        (frames_a, label_a, rewards_a),
        (frames_b, label_b, rewards_b),
    ]):
        y_banner = row * (banner_h + h + margin)
        y_img = y_banner + banner_h
        draw.text((2, y_banner + 4), lbl, fill=(220, 220, 220))
        for col, t in enumerate(indices):
            x = col * (w + margin)
            tile = Image.fromarray(frames[t])
            sheet.paste(tile, (x, y_img))
            # Reward text on frame
            rew = rewards[t]
            rc = (80, 220, 80) if rew > 0 else (220, 80, 80)
            draw.text((x + w - 52, y_img + 3), f"r={rew:+.2f}", fill=rc)
            draw.text((x + 2, y_img + 3), f"t={t}", fill=(180, 180, 180))

    # Paste path chart on the right
    x_chart = n_cols * (w + margin) - margin + margin
    sheet.paste(Image.fromarray(path_img), (x_chart, 0))

    sheet.save(path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Visualize a pairwise MetaWorld trajectory comparison."
    )
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to .npz pref dataset")
    parser.add_argument("--env", type=str, required=True,
                        help="Gym env ID, e.g. mw_drawer-open-v2")
    parser.add_argument("--pair", type=str, default="0",
                        help="Pair index (int) or 'random'")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file prefix (default: results/<env>_pair<i>)")
    parser.add_argument("--resolution", type=int, default=128,
                        help="Render resolution in pixels (default 128)")
    parser.add_argument("--fps", type=int, default=15,
                        help="Frames per second for video (default 15)")
    parser.add_argument("--contact-cols", type=int, default=8,
                        help="Columns in the contact sheet (default 8)")
    parser.add_argument("--label-key", type=str, default="rl_sum",
                        help="Dataset scalar quality label (default: rl_sum)")
    parser.add_argument("--no-video", action="store_true",
                        help="Skip saving MP4 video")
    parser.add_argument("--no-sheet", action="store_true",
                        help="Skip saving PNG contact sheet")
    parser.add_argument("--camera", type=str, default="corner2",
                        help="MuJoCo camera name (default: corner2)")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------------
    print(f"Loading dataset: {args.dataset}")
    data = np.load(args.dataset, allow_pickle=True)

    states  = data["state"]    # (2N, T, state_dim)
    obs_all = data["obs"]      # (2N, T, obs_dim=35)
    rewards = data["reward"]   # (2N, T)
    labels  = data.get(args.label_key, None)

    total_segs = states.shape[0]
    N = total_segs // 2

    if args.pair.lower() == "random":
        pair_idx = np.random.randint(0, N)
        print(f"Random pair index: {pair_idx}")
    else:
        pair_idx = int(args.pair)

    if pair_idx >= N:
        print(f"Error: pair {pair_idx} out of range (dataset has {N} pairs).")
        sys.exit(1)

    idx_a, idx_b = pair_idx, pair_idx + N

    states_a  = states[idx_a]       # (T, state_dim)
    states_b  = states[idx_b]
    rewards_a = rewards[idx_a]      # (T,)
    rewards_b = rewards[idx_b]
    obs_a     = obs_all[idx_a]      # (T, 35)
    obs_b     = obs_all[idx_b]

    # End-effector XYZ and object XYZ from trimmed obs
    ee_a  = obs_a[:, 0:3]          # end-effector
    ee_b  = obs_b[:, 0:3]
    obj_a = obs_a[:, 4:7]          # first object
    obj_b = obs_b[:, 4:7]

    ret_a = float(rewards_a.sum())
    ret_b = float(rewards_b.sum())

    if labels is not None:
        score_a = float(labels[idx_a])
        score_b = float(labels[idx_b])
        preferred = ("A" if score_a > score_b else
                     "B" if score_b > score_a else "tie")
        label_a = (f"Traj A  idx={idx_a}  ret={ret_a:.2f}"
                   f"  {args.label_key}={score_a:.3f}")
        label_b = (f"Traj B  idx={idx_b}  ret={ret_b:.2f}"
                   f"  {args.label_key}={score_b:.3f}"
                   f"  ← preferred: {preferred}")
    else:
        preferred = ("A" if ret_a > ret_b else
                     "B" if ret_b > ret_a else "tie")
        label_a = f"Traj A  idx={idx_a}  ret={ret_a:.2f}"
        label_b = f"Traj B  idx={idx_b}  ret={ret_b:.2f}"

    print(f"  {label_a}")
    print(f"  {label_b}")
    print(f"  Preferred by return: {preferred}")

    # ------------------------------------------------------------------
    # Build MetaWorld image env
    # ------------------------------------------------------------------
    print(f"\nCreating env: {args.env}  resolution={args.resolution}")
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import gym
    import research  # registers mw_* envs
    from research.envs.metaworld import MetaWorldSawyerImageWrapper

    base_env = gym.make(args.env)
    env = MetaWorldSawyerImageWrapper(
        base_env,
        width=args.resolution,
        height=args.resolution,
        camera=args.camera,
    )
    env.reset()     # configures camera position and FOV

    # ------------------------------------------------------------------
    # Render both trajectories
    # ------------------------------------------------------------------
    print("Rendering traj A...")
    frames_a = render_trajectory(env, states_a)
    print("Rendering traj B...")
    frames_b = render_trajectory(env, states_b)

    if hasattr(env, "close"):
        env.close()

    T = len(frames_a)

    # ------------------------------------------------------------------
    # Annotate frames with per-timestep reward + cumulative return
    # ------------------------------------------------------------------
    print("Annotating frames...")
    cum_a, cum_b = 0.0, 0.0
    ann_a, ann_b = [], []
    for t in range(T):
        cum_a += rewards_a[t]
        cum_b += rewards_b[t]
        ann_a.append(annotate_frame(frames_a[t], rewards_a[t], cum_a, t, T - 1))
        ann_b.append(annotate_frame(frames_b[t], rewards_b[t], cum_b, t, T - 1))

    # ------------------------------------------------------------------
    # Output paths
    # ------------------------------------------------------------------
    if args.output is None:
        env_slug = args.env.replace("/", "_")
        args.output = os.path.join("results", f"{env_slug}_pair{pair_idx}")

    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # PNG contact sheet
    # ------------------------------------------------------------------
    if not args.no_sheet:
        sheet_path = args.output + "_contact.png"
        print(f"\nSaving contact sheet → {sheet_path}")
        save_contact_sheet(
            ann_a, ann_b, sheet_path,
            ee_a, obj_a, ee_b, obj_b,
            rewards_a, rewards_b,
            n_cols=args.contact_cols,
            label_a=label_a,
            label_b=label_b,
        )
        print(f"  Saved.")

    # ------------------------------------------------------------------
    # MP4 video: [traj_A | path_panel | traj_B]
    # ------------------------------------------------------------------
    if not args.no_video:
        print("\nBuilding composite video frames...")
        divider = np.full((args.resolution, 4, 3), 60, dtype=np.uint8)
        composite = []
        for t in range(T):
            panel = make_path_panel(ee_a, obj_a, ee_b, obj_b,
                                    current_t=t, size=args.resolution)
            panel = resize_panel(panel, args.resolution)
            frame = np.concatenate([ann_a[t], divider, panel, divider, ann_b[t]],
                                   axis=1)
            composite.append(frame)

        path_vid = args.output + "_comparison.mp4"
        path_a   = args.output + "_trajA.mp4"
        path_b   = args.output + "_trajB.mp4"

        print(f"Saving traj A       → {path_a}")
        save_mp4(ann_a, path_a, fps=args.fps)

        print(f"Saving traj B       → {path_b}")
        save_mp4(ann_b, path_b, fps=args.fps)

        print(f"Saving composite    → {path_vid}")
        backend = save_mp4(composite, path_vid, fps=args.fps)
        print(f"  (backend: {backend})")

    print("\nDone.")


if __name__ == "__main__":
    main()
