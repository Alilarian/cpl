"""
Check whether seq-estop's "preferred" label actually matches higher oracle
return-based advantage.

generate_seq_estop_labels.py assigns index 0 = preferred purely by *position*
relative to the simulated stop time tau:
    t == tau  -> stop_seg (obs[t-h+1:t+1]) preferred over cont_seg (obs[t:t+h])
    t <  tau  -> cont_seg preferred over stop_seg
It never compares a scalar return/advantage between the two candidate
segments (unlike generate_pref_labels.py, which picks whichever of two
segments has the higher oracle rl_sum and asserts it at generation time).

CPL's contrastive loss (research/algs/cpl.py) only works if "preferred" tracks
true advantage: it trains the policy so index 0's summed log-prob exceeds
index 1's, on the theoretical premise that log pi*(a|s) is proportional to
advantage. If the label doesn't reliably track a real advantage difference,
CPL is trained on effectively mislabeled pairs, which can wreck the policy.

This script re-scores every stored (preferred, non-preferred) pair with the
SAME oracle used to generate the labels, using the same one-step TD quantity
that feeds Delta_t/H_t -- but signed and un-clamped:

    rl_sum_t = r_t + gamma * V(s_{t+1}) - V(s_t)      (t = 0..h-2, pad last = 0)
    segment_score = sum_t rl_sum_t                     (per h-step segment)

For every pair it then checks whether segment_score[preferred] >
segment_score[non-preferred] ("advantage-consistency"), and reports the rate
overall, split by stop_event (stop-time pairs vs continue-time pairs), and by
checkpoint_step tercile (pool segments from early vs late oracle checkpoints).

Usage:
    python scripts/analyze_seq_estop_advantage.py \\
        --labels-path /scratch/.../seq_estop_labels/mw_drawer-open-v2/seq_estop_labels.npz \\
        --run-dir     runs/runs/chpc/oracle_sac_seeds/mw_drawer-open-v2/seed-1 \\
        --output-dir  results/seq_estop_advantage \\
        --max-pairs   20000
"""

import argparse
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from research.utils.config import Config


# ---------------------------------------------------------------------------
# Shared utilities (same pattern as generate_seq_estop_labels.py)
# ---------------------------------------------------------------------------

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
    return model


def compute_signed_advantage(obs_b, reward_b, oracle, mcmc_samples, discount, device):
    """
    rl_sum_t = r_t + gamma * V(s_{t+1}) - V(s_t)  for t = 0..h-2, padded 0 at t=h-1.

    Same V(s) estimator as generate_seq_estop_labels.py's
    compute_per_step_disadvantage, but returns the *signed, un-clamped*
    one-step TD advantage instead of max(0, -rl_sum_t).

    Args:
        obs_b    : (B, h, obs_dim) numpy
        reward_b : (B, h)          numpy

    Returns:
        rl_sum : (B, h) numpy
    """
    obs_t    = torch.from_numpy(obs_b).float().to(device)
    reward_t = torch.from_numpy(reward_b).float().to(device)

    with torch.no_grad():
        obs_enc = oracle.network.encoder(obs_t)

        obs_exp   = obs_enc.unsqueeze(2).expand(-1, -1, mcmc_samples, -1)
        sampled_a = oracle.network.actor(obs_exp).sample()
        v = oracle.network.critic(obs_exp, sampled_a).mean(dim=0)
        v = v.mean(dim=2)

        v_curr = v[:, :-1]
        v_next = v[:, 1:]
        r_curr = reward_t[:, :-1]
        rl_sum = r_curr + discount * v_next - v_curr

        pad = torch.zeros(obs_b.shape[0], 1, device=device, dtype=rl_sum.dtype)
        rl_sum = torch.cat([rl_sum, pad], dim=1)

    return rl_sum.cpu().numpy()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Check advantage-consistency of seq-estop preferred labels."
    )
    parser.add_argument("--labels-path", type=str, required=True,
                         help="Path to seq_estop_labels.npz")
    parser.add_argument("--run-dir", type=str, required=True,
                         help="Oracle SAC run dir used to generate the labels")
    parser.add_argument("--oracle-checkpoint", type=str, default="best_model.pt")
    parser.add_argument("--discount", type=float, default=0.99,
                         help="Must match the value used at label-generation time")
    parser.add_argument("--mcmc-samples", type=int, default=32,
                         help="Must match the value used at label-generation time")
    parser.add_argument("--batch-size", type=int, default=256,
                         help="Segments scored per forward pass")
    parser.add_argument("--max-pairs", type=int, default=None,
                         help="Randomly subsample this many pairs before scoring "
                              "(default: score all pairs in the file)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="results/seq_estop_advantage")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else (
        "cpu" if args.device == "auto" else args.device
    )

    env_name = os.path.basename(os.path.dirname(args.labels_path))
    out_dir = os.path.join(args.output_dir, env_name)
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 65)
    print(f"Environment    : {env_name}")
    print(f"Labels path    : {args.labels_path}")
    print(f"Oracle dir     : {args.run_dir}")
    print(f"Discount gamma : {args.discount}")
    print(f"MCMC samples   : {args.mcmc_samples}")
    print(f"Device         : {device}")
    print("=" * 65)

    # ------------------------------------------------------------------
    # Load labels
    # ------------------------------------------------------------------
    print("\nLoading labels ...")
    with open(args.labels_path, "rb") as f:
        d = np.load(f)
        obs        = d["obs"]              # (M, 2, h, obs_dim)
        reward     = d["reward"]           # (M, 2, h)
        stop_event = d["stop_event"]       # (M,)
        timestep   = d["timestep"]         # (M,)
        traj_idx   = d["traj_idx"]         # (M,)
        ckpt       = d["checkpoint_step"]  # (M,)

    M, K, h, obs_dim = obs.shape
    assert K == 2, f"expected K=2 pairs, got K={K}"
    print(f"  M={M:,} pairs, h={h}, obs_dim={obs_dim}")
    print(f"  stop pairs     : {int(stop_event.sum()):,}")
    print(f"  continue pairs : {int((1 - stop_event).sum()):,}")

    rng = np.random.default_rng(args.seed)
    if args.max_pairs is not None and M > args.max_pairs:
        keep = np.sort(rng.choice(M, size=args.max_pairs, replace=False))
        obs, reward = obs[keep], reward[keep]
        stop_event, timestep = stop_event[keep], timestep[keep]
        traj_idx, ckpt = traj_idx[keep], ckpt[keep]
        M = args.max_pairs
        print(f"  Subsampled to  : {M:,} pairs")

    # ------------------------------------------------------------------
    # Load oracle
    # ------------------------------------------------------------------
    oracle_ckpt = os.path.join(args.run_dir, args.oracle_checkpoint)
    print(f"\nLoading oracle: {oracle_ckpt}")
    oracle = load_model(args.run_dir, oracle_ckpt, device)

    # ------------------------------------------------------------------
    # Score every segment (flatten M*2 segments into one batched pass)
    # ------------------------------------------------------------------
    print(f"\nScoring {M * 2:,} segments (h={h} steps each) ...")
    obs_flat    = obs.reshape(M * 2, h, obs_dim)
    reward_flat = reward.reshape(M * 2, h)

    rl_sum_flat = np.empty((M * 2, h), dtype=np.float32)
    n_batches = (M * 2 + args.batch_size - 1) // args.batch_size
    for b in range(n_batches):
        s = b * args.batch_size
        e = min(s + args.batch_size, M * 2)
        rl_sum_flat[s:e] = compute_signed_advantage(
            obs_flat[s:e], reward_flat[s:e], oracle, args.mcmc_samples, args.discount, device,
        )
        if (b + 1) % 20 == 0 or b == n_batches - 1:
            print(f"  [{b+1:>4}/{n_batches}]  scored {e:>7,}/{M*2:,}")

    segment_score = rl_sum_flat.sum(axis=1).reshape(M, 2)  # (M, 2)
    margin = segment_score[:, 0] - segment_score[:, 1]     # preferred - non-preferred
    win = margin > 0
    tie = margin == 0

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    def report(mask, label):
        n = int(mask.sum())
        if n == 0:
            print(f"  {label:<28}: n=0")
            return
        w = win[mask]
        m = margin[mask]
        print(f"  {label:<28}: n={n:>7,}  "
              f"advantage-consistent={100*w.mean():5.1f}%  "
              f"mean_margin={m.mean():+.4f}  median_margin={np.median(m):+.4f}")

    print("\n" + "=" * 65)
    print("ADVANTAGE-CONSISTENCY  (preferred segment_score > non-preferred?)")
    print("=" * 65)
    report(np.ones(M, dtype=bool), "ALL PAIRS")
    print()
    report(stop_event == 1, "stop pairs   (t == tau)")
    report(stop_event == 0, "continue pairs (t < tau)")

    print()
    ckpt_terciles = np.quantile(ckpt.astype(np.float64), [1/3, 2/3])
    low  = ckpt <= ckpt_terciles[0]
    mid  = (ckpt > ckpt_terciles[0]) & (ckpt <= ckpt_terciles[1])
    high = ckpt > ckpt_terciles[1]
    report(low,  "checkpoint tercile: early")
    report(mid,  "checkpoint tercile: mid")
    report(high, "checkpoint tercile: late")

    print(f"\n  ties (margin == 0): {int(tie.sum()):,} / {M:,}")
    print(f"\nInterpretation: 50% == coin flip (label carries no information about\n"
          f"true oracle advantage); 100% == label always matches the oracle's own\n"
          f"ranking of the two segments (CPL's assumption is fully satisfied).")

    # ------------------------------------------------------------------
    # Save arrays + plot
    # ------------------------------------------------------------------
    out_npz = os.path.join(out_dir, "advantage_consistency.npz")
    np.savez_compressed(
        out_npz,
        segment_score=segment_score,
        margin=margin,
        win=win,
        stop_event=stop_event,
        timestep=timestep,
        traj_idx=traj_idx,
        checkpoint_step=ckpt,
    )
    print(f"\nSaved arrays -> {out_npz}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    ax = axes[0]
    bins = np.linspace(np.percentile(margin, 1), np.percentile(margin, 99), 60)
    ax.hist(margin[stop_event == 0], bins=bins, alpha=0.6, label="continue pairs", color="#4C72B0")
    ax.hist(margin[stop_event == 1], bins=bins, alpha=0.6, label="stop pairs", color="#C44E52")
    ax.axvline(0.0, color="black", linewidth=1, linestyle="--")
    ax.set_xlabel("segment_score[preferred] - segment_score[non-preferred]")
    ax.set_ylabel("count")
    ax.set_title(f"{env_name}: advantage margin")
    ax.legend()

    ax = axes[1]
    groups = ["all", "stop", "continue", "ckpt-early", "ckpt-mid", "ckpt-late"]
    masks  = [np.ones(M, dtype=bool), stop_event == 1, stop_event == 0, low, mid, high]
    rates  = [100 * win[m].mean() if m.sum() > 0 else 0.0 for m in masks]
    bar_colors = ["#55A868"] + ["#4C72B0", "#C44E52"] + ["#8172B2"] * 3
    ax.bar(groups, rates, color=bar_colors)
    ax.axhline(50.0, color="black", linewidth=1, linestyle="--")
    ax.set_ylabel("advantage-consistent (%)")
    ax.set_ylim(0, 100)
    ax.set_title("preferred > non-preferred rate")
    ax.tick_params(axis="x", rotation=30)

    fig.tight_layout()
    out_png = os.path.join(out_dir, "advantage_consistency.png")
    fig.savefig(out_png, dpi=150)
    print(f"Saved plot   -> {out_png}")


if __name__ == "__main__":
    main()
