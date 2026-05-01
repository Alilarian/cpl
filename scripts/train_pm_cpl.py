"""
Standalone CPL training for PointMass with pairwise preference feedback.

Does NOT import the full research package — works on machines without
a working MuJoCo/MetaWorld build.

Features
--------
- BC warmup phase followed by CPL preference loss
- Periodic rollout evaluation in PointMassGymEnv
- Vector-field snapshots saved to <out>/frames/policy_<step>.png
- Optional animated GIF from frames (requires Pillow: pip install Pillow)
- Checkpoints: best_model.pt (best eval reward) + final_model.pt

Usage
-----
# Basic
python scripts/train_pm_cpl.py \\
    --pref datasets/pm/pref_labels.npz \\
    --out  runs/pm_cpl_pref

# With VI overlay on vector-field plots
python scripts/train_pm_cpl.py \\
    --pref          datasets/pm/pref_labels.npz \\
    --advantage-npz runs/pm_sac_oracle/advantage_vi_N100.npz \\
    --out           runs/pm_cpl_pref \\
    --total-steps   100000 \\
    --bc-steps      20000  \\
    --vis-freq      5000   \\
    --make-gif
"""

import argparse
import importlib.util
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# Direct module loading (avoids research/__init__.py → MetaWorld chain)
# ---------------------------------------------------------------------------

def _load_module(dotted_name, rel_path):
    if dotted_name in sys.modules:
        return sys.modules[dotted_name]
    path = os.path.join(_ROOT, rel_path)
    spec = importlib.util.spec_from_file_location(dotted_name, path)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[dotted_name] = mod
    spec.loader.exec_module(mod)
    return mod


_pm  = _load_module("research.envs.pointmass",     "research/envs/pointmass.py")
_avi = _load_module("research.utils.advantage_vi", "research/utils/advantage_vi.py")

PointMassGymEnv = _pm.PointMassGymEnv
AdvantageVI     = _avi.AdvantageVI


# ---------------------------------------------------------------------------
# Policy: deterministic MLP with tanh output (ContinuousMLPActor analogue)
# ---------------------------------------------------------------------------

class Policy(nn.Module):
    """
    Deterministic policy: obs → tanh(MLP(obs)) ∈ [-1,1]².

    log_prob proxy: -||action - mean||²  (unit-variance Gaussian, same as
    MetaWorld CPL with ContinuousMLPActor).
    """

    def __init__(self, obs_dim=2, act_dim=2, hidden=(256, 256)):
        super().__init__()
        dims   = [obs_dim] + list(hidden)
        layers = []
        for i in range(len(dims) - 1):
            layers += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU()]
        layers += [nn.Linear(dims[-1], act_dim), nn.Tanh()]
        self.net = nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)

    def forward(self, obs):
        """obs: (*batch, obs_dim) → (*batch, act_dim) in [-1,1]²"""
        return self.net(obs)

    def log_prob(self, obs, action):
        """Per-step log-prob proxy: (B,T,D),(B,T,D) → (B,T)"""
        mean = self.forward(obs)
        return -torch.square(action - mean).sum(dim=-1)

    @torch.no_grad()
    def act(self, obs_np: np.ndarray) -> np.ndarray:
        """Single-step inference: numpy (obs_dim,) → numpy (act_dim,)"""
        device = next(self.parameters()).device
        obs = torch.tensor(obs_np, dtype=torch.float32, device=device).unsqueeze(0)
        return np.clip(self.forward(obs).squeeze(0).cpu().numpy(), -1.0, 1.0)


# ---------------------------------------------------------------------------
# CPL loss
# ---------------------------------------------------------------------------

def biased_bce_with_logits(adv1, adv2, y, bias=1.0):
    """
    Biased BCE with logits for pairwise preferences.
    y = 1.0 if segment 2 is preferred, 0.0 if segment 1 is preferred.
    """
    logit21 = adv2 - bias * adv1
    logit12 = adv1 - bias * adv2
    max21    = torch.clamp(-logit21, min=0)
    max12    = torch.clamp(-logit12, min=0)
    nlp21    = torch.log(torch.exp(-max21) + torch.exp(-logit21 - max21)) + max21
    nlp12    = torch.log(torch.exp(-max12) + torch.exp(-logit12 - max12)) + max12
    loss     = (y * nlp21 + (1 - y) * nlp12).mean()
    with torch.no_grad():
        accuracy = ((adv2 > adv1) == torch.round(y)).float().mean()
    return loss, accuracy


# ---------------------------------------------------------------------------
# Preference dataset
# ---------------------------------------------------------------------------

class PrefDataset:
    """
    Loads pref_labels.npz from generate_pm_feedback.py --type pref.

    Schema: obs (N,2,T,obs_dim), action (N,2,T,act_dim), adv_scores (N,2).
    """

    def __init__(self, path, batch_size=64, capacity=None, action_eps=1e-5):
        data   = np.load(path)
        obs    = data["obs"].astype(np.float32)           # (N, 2, T, obs_dim)
        action = data["action"].astype(np.float32)        # (N, 2, T, act_dim)
        scores = data["adv_scores"].astype(np.float32)    # (N, 2)

        N = obs.shape[0]
        if capacity is not None and N > capacity:
            obs    = obs[:capacity]
            action = action[:capacity]
            scores = scores[:capacity]
            N = capacity

        lim = 1.0 - action_eps
        action = np.clip(action, -lim, lim)

        hard   = (scores[:, 1] > scores[:, 0]).astype(np.float32)
        soft   = 0.5 * (scores[:, 0] == scores[:, 1]).astype(np.float32)

        self.obs_0   = obs[:, 0]     # (N, T, obs_dim)
        self.obs_1   = obs[:, 1]
        self.act_0   = action[:, 0]  # (N, T, act_dim)
        self.act_1   = action[:, 1]
        self.labels  = hard + soft   # (N,)  1.0 if seg1 preferred
        self.N       = N
        self.T       = obs.shape[2]
        self.batch_size = batch_size

    def __len__(self):
        return self.N

    def sample(self):
        idx = np.random.randint(0, self.N, size=self.batch_size)
        return {
            "obs_1":    torch.from_numpy(self.obs_0[idx]),    # (B, T, obs_dim)
            "obs_2":    torch.from_numpy(self.obs_1[idx]),
            "action_1": torch.from_numpy(self.act_0[idx]),   # (B, T, act_dim)
            "action_2": torch.from_numpy(self.act_1[idx]),
            "label":    torch.from_numpy(self.labels[idx]),  # (B,)
        }


# ---------------------------------------------------------------------------
# Rollout evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(policy: Policy, env: PointMassGymEnv,
             n_episodes: int = 20) -> dict:
    policy.eval()
    successes, total_reward, total_len = 0, 0.0, 0
    for _ in range(n_episodes):
        obs  = env.reset()
        done = False
        ep_r = 0.0
        ep_l = 0
        while not done:
            obs, r, done, _ = env.step(policy.act(obs))
            ep_r += r
            ep_l += 1
        if ep_r > 0:   # positive reward ↔ reached goal
            successes += 1
        total_reward += ep_r
        total_len    += ep_l
    policy.train()
    return {
        "success_rate": successes / n_episodes,
        "mean_reward":  total_reward / n_episodes,
        "mean_length":  total_len / n_episodes,
    }


# ---------------------------------------------------------------------------
# Vector-field visualization
# ---------------------------------------------------------------------------

@torch.no_grad()
def plot_vector_field(
    policy: Policy,
    env: PointMassGymEnv,
    step: int,
    out_dir: str,
    device: torch.device,
    avi=None,
    grid_n: int = 30,
) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    policy.eval()
    xs       = np.linspace(0.0, 1.0, grid_n)
    GX, GY   = np.meshgrid(xs, xs, indexing="ij")
    states   = np.stack([GX.ravel(), GY.ravel()], axis=1).astype(np.float32)
    obs_t    = torch.from_numpy(states).to(device)
    mean_act = policy(obs_t).cpu().numpy()               # (grid_n², act_dim)
    DX  = mean_act[:, 0].reshape(grid_n, grid_n)
    DY  = mean_act[:, 1].reshape(grid_n, grid_n)
    mag = np.hypot(DX, DY)

    fig, ax = plt.subplots(figsize=(5, 5))

    if avi is not None:
        im = ax.imshow(
            avi.V.T, origin="lower", extent=[0, 1, 0, 1],
            cmap="RdYlGn", aspect="equal",
            interpolation="nearest", alpha=0.55,
        )
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="V*(s)")

    ax.quiver(GX, GY, DX, DY, mag.ravel(),
              cmap="plasma", scale=30, width=0.003, alpha=0.85)

    ax.add_patch(plt.Circle(env.goal, env.goal_dist_thresh,
                            color="green", alpha=0.35, zorder=5))
    ax.plot(*env.goal, "g^", ms=8, zorder=6)
    ax.add_patch(plt.Circle(env.trap, env.trap_dist_thresh,
                            color="red", alpha=0.35, zorder=5))
    ax.plot(*env.trap, "rx", ms=8, mew=2, zorder=6)

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title(f"Policy vector field  (step {step:,})", fontsize=10)
    ax.legend(handles=[
        mpatches.Patch(color="green", alpha=0.5, label="goal"),
        mpatches.Patch(color="red",   alpha=0.5, label="trap"),
    ], fontsize=7, loc="upper right")

    os.makedirs(out_dir, exist_ok=True)
    fname = os.path.join(out_dir, f"policy_{step:07d}.png")
    fig.tight_layout()
    fig.savefig(fname, dpi=100, bbox_inches="tight")
    plt.close(fig)
    policy.train()

    try:
        import wandb
        if wandb.run is not None:
            wandb.log({"policy_vector_field": wandb.Image(fname)}, step=step)
    except ImportError:
        pass

    return fname


# ---------------------------------------------------------------------------
# GIF creation
# ---------------------------------------------------------------------------

def frames_to_gif(frames_dir: str, out_gif: str, fps: int = 3) -> None:
    from PIL import Image
    pngs = sorted(f for f in os.listdir(frames_dir) if f.endswith(".png"))
    if not pngs:
        print("[gif] no frames found in", frames_dir)
        return
    imgs = [Image.open(os.path.join(frames_dir, f)).convert("RGBA") for f in pngs]
    imgs[0].save(
        out_gif, save_all=True, append_images=imgs[1:],
        duration=int(1000 / fps), loop=0,
    )
    print(f"[gif] saved → {out_gif}  ({len(imgs)} frames @ {fps} fps)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train CPL on PointMass pairwise preferences (standalone)."
    )
    parser.add_argument("--pref",          type=str, required=True,
                        help="Path to pref_labels.npz")
    parser.add_argument("--advantage-npz", type=str, default=None,
                        help="VI table .npz for V*(s) overlay in vector-field plots")
    parser.add_argument("--out",           type=str, required=True,
                        help="Output directory for checkpoints and figures")
    parser.add_argument("--total-steps",   type=int, default=100_000)
    parser.add_argument("--bc-steps",      type=int, default=20_000,
                        help="BC warmup steps before CPL loss kicks in")
    parser.add_argument("--batch-size",    type=int, default=64)
    parser.add_argument("--lr",            type=float, default=1e-4)
    parser.add_argument("--alpha",         type=float, default=1.0,
                        help="CPL temperature α (log-prob scaling)")
    parser.add_argument("--bias",          type=float, default=0.5,
                        help="Contrastive bias (0 < bias ≤ 1)")
    parser.add_argument("--capacity",      type=int, default=None,
                        help="Max preference pairs to load (None = all)")
    parser.add_argument("--hidden",        type=int, nargs="+", default=[256, 256],
                        help="Hidden layer sizes (default: 256 256)")
    parser.add_argument("--eval-freq",     type=int, default=5000)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--vis-freq",      type=int, default=5000,
                        help="Steps between vector-field snapshots")
    parser.add_argument("--device",        type=str, default="auto")
    parser.add_argument("--seed",          type=int, default=42)
    parser.add_argument("--make-gif",      action="store_true",
                        help="Create animated GIF from frames after training")
    parser.add_argument("--gif-fps",       type=int, default=3)
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.out, exist_ok=True)
    frames_dir = os.path.join(args.out, "frames")

    # W&B — opt-in: set WANDB_API_KEY (or run `wandb login`) before training
    try:
        import wandb
        api_key = os.getenv("WANDB_API_KEY") or wandb.api.api_key
        if api_key:
            project = os.getenv("WANDB_PROJECT", "cpl-pointmass")
            wandb.init(project=project, name=os.path.basename(args.out),
                       config=vars(args))
            print(f"W&B: {wandb.run.url}")
    except ImportError:
        pass

    print(f"Device : {device}")
    print(f"Output : {args.out}")

    # --- Data -----------------------------------------------------------
    print(f"\nLoading pref data from {args.pref} …")
    dataset = PrefDataset(
        args.pref,
        batch_size=args.batch_size,
        capacity=args.capacity,
    )
    seg0_pref = 1.0 - float(dataset.labels.mean())  # fraction where seg0 is preferred
    print(f"  {len(dataset):,} pairs  T={dataset.T}  "
          f"seg0_preferred={seg0_pref:.1%}")

    # --- VI table (optional) -------------------------------------------
    avi = None
    if args.advantage_npz and os.path.exists(args.advantage_npz):
        avi = AdvantageVI.load(args.advantage_npz)
        print(f"VI table: {args.advantage_npz}  (N={avi.N})")

    # --- Environment (for eval) ----------------------------------------
    env = PointMassGymEnv(step_cost=0.01)

    # --- Policy ---------------------------------------------------------
    policy = Policy(obs_dim=2, act_dim=2, hidden=tuple(args.hidden)).to(device)
    n_params = sum(p.numel() for p in policy.parameters())
    print(f"\nPolicy params: {n_params:,}")

    # --- Optimizer + warmup LR schedule --------------------------------
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    warmup_steps = 10_000
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda s: min(1.0, s / max(1, warmup_steps)),
    )

    # --- Training loop --------------------------------------------------
    print(f"\n{'='*62}")
    print(f"  total_steps={args.total_steps:,}  bc_steps={args.bc_steps:,}"
          f"  α={args.alpha}  bias={args.bias}")
    print(f"{'='*62}")

    log_every      = 500
    best_reward    = -float("inf")
    t0             = time.time()
    cpl_buf, bc_buf, acc_buf = [], [], []

    # Save frame of random (untrained) policy
    plot_vector_field(policy, env, 0, frames_dir, device, avi=avi)

    policy.train()
    for step in range(1, args.total_steps + 1):

        batch    = dataset.sample()
        obs_1    = batch["obs_1"].to(device)      # (B, T, 2)
        obs_2    = batch["obs_2"].to(device)
        act_1    = batch["action_1"].to(device)   # (B, T, 2)
        act_2    = batch["action_2"].to(device)
        label    = batch["label"].to(device)      # (B,)

        obs    = torch.cat([obs_1, obs_2], dim=0)   # (2B, T, 2)
        action = torch.cat([act_1, act_2], dim=0)   # (2B, T, 2)
        B      = obs_1.shape[0]

        # Forward: log_prob per step
        lp = policy.log_prob(obs, action)             # (2B, T)

        # BC loss
        bc_loss = -lp.mean()

        # CPL loss
        adv         = args.alpha * lp
        seg_adv     = adv.sum(dim=1)                  # (2B,)
        adv1, adv2  = seg_adv[:B], seg_adv[B:]
        cpl_loss, accuracy = biased_bce_with_logits(adv1, adv2, label, bias=args.bias)

        if step <= args.bc_steps:
            loss    = bc_loss
            cpl_val = 0.0
            acc_val = 0.0
        else:
            loss    = cpl_loss
            cpl_val = cpl_loss.item()
            acc_val = accuracy.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        cpl_buf.append(cpl_val)
        bc_buf.append(bc_loss.item())
        acc_buf.append(acc_val)

        # --- Logging ----------------------------------------------------
        if step % log_every == 0:
            phase = "BC " if step <= args.bc_steps else "CPL"
            cpl_mean = np.mean(cpl_buf[-log_every:])
            bc_mean  = np.mean(bc_buf[-log_every:])
            acc_mean = np.mean(acc_buf[-log_every:])
            lr_val   = scheduler.get_last_lr()[0]
            print(f"[{phase}] step {step:>7,}  "
                  f"cpl={cpl_mean:.4f}  "
                  f"bc={bc_mean:.4f}  "
                  f"acc={acc_mean:.3f}  "
                  f"lr={lr_val:.2e}  "
                  f"t={time.time()-t0:.0f}s")
            try:
                import wandb
                if wandb.run is not None:
                    wandb.log({"train/cpl_loss": cpl_mean,
                               "train/bc_loss":  bc_mean,
                               "train/accuracy": acc_mean,
                               "train/lr":       lr_val}, step=step)
            except ImportError:
                pass

        # --- Evaluation -------------------------------------------------
        if step % args.eval_freq == 0:
            m = evaluate(policy, env, n_episodes=args.eval_episodes)
            print(f"  [eval] success={m['success_rate']:.2f}  "
                  f"reward={m['mean_reward']:.1f}  "
                  f"len={m['mean_length']:.0f}")
            try:
                import wandb
                if wandb.run is not None:
                    wandb.log({"eval/success_rate": m["success_rate"],
                               "eval/mean_reward":  m["mean_reward"],
                               "eval/mean_length":  m["mean_length"]}, step=step)
            except ImportError:
                pass
            if m["mean_reward"] > best_reward:
                best_reward = m["mean_reward"]
                ckpt = os.path.join(args.out, "best_model.pt")
                torch.save({"policy": policy.state_dict(),
                            "step": step, "reward": best_reward,
                            "args": vars(args)}, ckpt)
                print(f"  [eval] ★ best reward={best_reward:.1f}  → {ckpt}")

        # --- Vector-field snapshot --------------------------------------
        if step % args.vis_freq == 0:
            fname = plot_vector_field(policy, env, step,
                                      frames_dir, device, avi=avi)
            print(f"  [viz]  → {fname}")

    # --- Final checkpoint -----------------------------------------------
    final = os.path.join(args.out, "final_model.pt")
    torch.save({"policy": policy.state_dict(),
                "step": args.total_steps, "args": vars(args)}, final)
    print(f"\nFinal model → {final}")

    # --- GIF ------------------------------------------------------------
    if args.make_gif:
        gif_path = os.path.join(args.out, "policy_evolution.gif")
        try:
            frames_to_gif(frames_dir, gif_path, fps=args.gif_fps)
        except ImportError:
            print("[gif] Pillow not available — skipping  (pip install Pillow)")

    elapsed = time.time() - t0
    print(f"Done  ({elapsed:.0f}s  best_reward={best_reward:.1f})")


if __name__ == "__main__":
    main()
