"""
Generate corrective feedback dataset from a trajectory pool.

Corrective feedback pairs each pool segment (the agent's own trajectory) with an
expert rollout from the *same* initial state s_0.  This gives the signal:
  "from *your* state, here is what you should have done instead."

Unlike demonstrative feedback (which uses stratified counterfactuals), corrective
feedback is strictly pairwise (K=2):
  - index 0: better trajectory  (expert if expert > original, else original)
  - index 1: worse trajectory   (original if expert > original, else expert)

Pairs are always kept regardless of which trajectory is better — the pair is simply
flipped so that index 0 is always the winner.  This matches the multi-type-feedback
convention and guarantees N_corr ≈ N_pool (no direction-based dropping).
Only pairs where |rl_sum gap| < --min-adv-gap are discarded (truly ambiguous pairs).
Pairs are optionally sorted by |improvement| magnitude (largest gap first).

Relationship to generate_demo_labels.py
---------------------------------------
Reuses: load_model, score_trajectories, rollout_from_state, get_checkpoint_paths,
        save_npz — logic is identical; only the candidate construction differs.
There are no stratified tiers here: K is always 2.

Output (--output-dir/<env>/corr_labels.npz):
    obs           : (N, 2, T, obs_dim)   [0]=better traj, [1]=worse traj
    action        : (N, 2, T, act_dim)
    reward        : (N, 2, T)
    adv_scores    : (N, 2)               [better_rl_sum, worse_rl_sum]  (always sorted descending)
    improvement   : (N,)                 |adv_scores[:,0] - adv_scores[:,1]|
    checkpoint_step : (N,)

Usage:
    python scripts/generate_corr_labels.py \\
        --pool-path datasets/mw/demo_pool/mw_drawer-open-v2/pool.npz \\
        --run-dir runs/runs/chpc/oracle_sac_all/mw_drawer-open-v2 \\
        --min-adv-gap 0.0 \\
        --output-dir datasets/mw/corr_labels

    # All envs (loop):
    for ENV in mw_drawer-open-v2 mw_door-open-v2 mw_bin-picking-v2 \\
               mw_button-press-v2 mw_plate-slide-v2; do
        python scripts/generate_corr_labels.py \\
            --pool-path datasets/mw/demo_pool/$ENV/pool.npz \\
            --run-dir runs/runs/chpc/oracle_sac_all/$ENV \\
            --output-dir datasets/mw/corr_labels
    done
"""

import argparse
import io
import os

import numpy as np
import torch

from research.utils.config import Config


# ---------------------------------------------------------------------------
# Shared utilities (identical to generate_demo_labels.py)
# ---------------------------------------------------------------------------

def save_npz(path, **arrays):
    """Atomically save a compressed npz (write to .tmp, then rename)."""
    tmp = path + ".tmp"
    with io.BytesIO() as buf:
        np.savez_compressed(buf, **arrays)
        buf.seek(0)
        with open(tmp, "wb") as f:
            f.write(buf.read())
    os.replace(tmp, path)


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


def score_trajectories(obs, action, reward, oracle, discount, mcmc_samples, device):
    """
    Score a batch of K trajectories with oracle advantage metrics.

    Primary metric: rl_sum = Σ_t r_t + V(s_T) - V(s_0)

    Args:
        obs    : (K, T, obs_dim)
        action : (K, T, act_dim)
        reward : (K, T)

    Returns:
        dict  key → np.ndarray (K,)
    """
    K, T, _ = obs.shape
    obs_t    = torch.from_numpy(obs).float().to(device)
    action_t = torch.from_numpy(action).float().to(device)
    reward_t = torch.from_numpy(reward).float().to(device)

    out = {}

    with torch.no_grad():
        obs_enc = oracle.network.encoder(obs_t)
        dist    = oracle.network.actor(obs_enc)

        if isinstance(dist, torch.distributions.Distribution):
            lp = dist.log_prob(torch.clamp(action_t, min=-0.999, max=0.999))
            out["rl_lp"] = lp.sum(dim=-1).cpu().numpy()
        else:
            lp = -torch.square(dist - action_t).sum(dim=-1)
            out["rl_lp"] = lp.sum(dim=-1).cpu().numpy()

        if hasattr(oracle.network, "critic"):
            q = oracle.network.critic(obs_enc, action_t).mean(dim=0)

            obs_exp   = obs_enc.unsqueeze(2).expand(-1, -1, mcmc_samples, -1)
            sampled_a = oracle.network.actor(obs_exp).sample()
            v         = oracle.network.critic(obs_exp, sampled_a).mean(dim=0)
            v         = v.mean(dim=2)

            disc = torch.pow(
                discount * torch.ones(T, device=device),
                torch.arange(T, device=device).float(),
            ).unsqueeze(0)

            adv = q - v
            out["rl_dir"]     = adv.sum(dim=-1).cpu().numpy()
            out["rl_dis_dir"] = (disc * adv).sum(dim=-1).cpu().numpy()
            out["rl_sum"]     = (
                reward_t[:, :-1].sum(dim=-1) + v[:, -1] - v[:, 0]
            ).cpu().numpy()
            out["rl_dis_sum"] = (
                (disc * reward_t)[:, :-1].sum(dim=-1)
                + (discount ** T) * v[:, -1]
                - v[:, 0]
            ).cpu().numpy()

    return out


def rollout_from_state(model, env, s0, segment_length, device):
    """
    Restore env to s0 and roll out model deterministically for segment_length steps.

    Returns (obs, action, reward) each (T, dim), or None if episode ends early.
    """
    env.set_state(s0)
    w = env
    while w is not None:
        if hasattr(w, '_elapsed_steps'):
            w._elapsed_steps = 0
        if hasattr(w, '_has_reset'):
            w._has_reset = True
        w = getattr(w, 'env', None)
    obs = env.get_obs().astype(np.float32)

    ep_obs, ep_act, ep_rew = [], [], []

    for _ in range(segment_length):
        with torch.no_grad():
            action = model.predict(dict(obs=obs), sample=False)
        action = np.clip(action, env.action_space.low, env.action_space.high)

        result = env.step(action)
        if len(result) == 5:
            next_obs, reward, terminated, truncated, _ = result
            done = terminated or truncated
        else:
            next_obs, reward, done, _ = result

        ep_obs.append(obs)
        ep_act.append(action.astype(np.float32))
        ep_rew.append(float(reward))
        obs = next_obs.astype(np.float32)

        if done:
            break

    if len(ep_obs) < segment_length:
        return None

    return (
        np.stack(ep_obs,  axis=0),
        np.stack(ep_act,  axis=0),
        np.array(ep_rew,  dtype=np.float32),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate corrective feedback labels (K=2: expert vs original)."
    )
    parser.add_argument("--pool-path", type=str, required=True,
                        help="Path to pool.npz from build_trajectory_pool.py")
    parser.add_argument("--run-dir", type=str, required=True,
                        help="Oracle SAC run dir (config.yaml + best_model.pt)")
    parser.add_argument("--expert-checkpoint", type=str, default="best_model.pt",
                        help="Expert checkpoint filename (default: best_model.pt)")
    parser.add_argument("--min-adv-gap", type=float, default=0.0,
                        help="Min |rl_sum gap| to keep a pair (default: 0.0). "
                             "Pairs are kept regardless of direction — flipped so "
                             "index 0 is always the better trajectory. Only truly "
                             "ambiguous pairs (|gap| < threshold) are discarded.")
    parser.add_argument("--sort-by-improvement", action="store_true", default=True,
                        help="Sort output by improvement magnitude descending "
                             "(most corrective pairs first, default: True)")
    parser.add_argument("--no-sort", dest="sort_by_improvement",
                        action="store_false",
                        help="Disable sorting by improvement magnitude")
    parser.add_argument("--discount",        type=float, default=0.99)
    parser.add_argument("--mcmc-samples",   type=int,   default=64,
                        help="MCMC samples for V(s) estimation (default: 64)")
    parser.add_argument("--score-batch-size", type=int, default=64,
                        help="Score this many segments in one forward pass (default: 64). "
                             "Higher values amortize PyTorch overhead on CPU.")
    parser.add_argument("--save-every",   type=int,   default=500,
                        help="Save progress every N kept samples (default: 500)")
    parser.add_argument("--output-dir",   type=str,   default="datasets/mw/corr_labels")
    parser.add_argument("--device",       type=str,   default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Device: {device}")
    print(f"K=2  (index 0 = better trajectory, index 1 = worse trajectory)")
    print(f"Scoring metric: rl_sum = Σ r_t + V(s_T) - V(s_0)")
    print(f"Min |adv gap|: {args.min_adv_gap}  (pairs always kept, just flipped)")

    # ------------------------------------------------------------------
    # Load pool
    # ------------------------------------------------------------------
    print(f"\nLoading pool: {args.pool_path}")
    with open(args.pool_path, "rb") as f:
        pool        = np.load(f)
        pool_obs    = pool["obs"]             # (N, T, obs_dim)
        pool_action = pool["action"]          # (N, T, act_dim)
        pool_reward = pool["reward"]          # (N, T)
        pool_state  = pool["state"]           # (N, T, state_dim)
        pool_ckpt   = pool["checkpoint_step"] # (N,)

    N, T, obs_dim = pool_obs.shape
    act_dim = pool_action.shape[-1]
    print(f"  N={N} segments, T={T} steps, obs_dim={obs_dim}")

    # ------------------------------------------------------------------
    # Load expert (also used as oracle for scoring)
    # ------------------------------------------------------------------
    expert_path = os.path.join(args.run_dir, args.expert_checkpoint)
    print(f"\nLoading expert + oracle: {expert_path}")
    expert_model, env = load_model(args.run_dir, expert_path, device)
    oracle = expert_model

    # ------------------------------------------------------------------
    # Output paths + resume
    # ------------------------------------------------------------------
    env_name  = os.path.basename(os.path.dirname(args.pool_path))
    out_dir   = os.path.join(args.output_dir, env_name)
    os.makedirs(out_dir, exist_ok=True)
    out_path  = os.path.join(out_dir, "corr_labels.npz")
    prog_path = os.path.join(out_dir, "corr_labels_progress.npz")

    if os.path.exists(out_path):
        print(f"\nOutput already exists: {out_path}")
        print("Delete it to regenerate.")
        return

    out_obs, out_action, out_reward, out_adv, out_impr, out_ckpt = \
        [], [], [], [], [], []
    n_skip_done = 0
    n_skip_gap  = 0
    start_idx   = 0

    if os.path.exists(prog_path):
        print(f"Resuming from progress file: {prog_path}")
        prog = np.load(prog_path, allow_pickle=True)
        start_idx   = int(prog["next_pool_idx"])
        n_skip_done = int(prog["n_skip_done"])
        n_skip_gap  = int(prog["n_skip_gap"])
        if prog["obs"].shape[0] > 0:
            for row in prog["obs"]:         out_obs.append(row)
            for row in prog["action"]:      out_action.append(row)
            for row in prog["reward"]:      out_reward.append(row)
            for row in prog["adv_scores"]:  out_adv.append(row)
            for v   in prog["improvement"]: out_impr.append(float(v))
            for v   in prog["checkpoint_step"]: out_ckpt.append(int(v))
        print(f"  Resumed at pool index {start_idx}, "
              f"{len(out_obs)} samples kept so far\n")

    # ------------------------------------------------------------------
    # Process pool segments with batched scoring
    # Rollouts are sequential (env state required), but score_trajectories
    # is called once per batch of --score-batch-size segments, giving 3-5×
    # speedup on CPU by amortizing PyTorch overhead and improving BLAS use.
    # ------------------------------------------------------------------
    print(f"\nProcessing {N - start_idx} segments "
          f"(score batch size={args.score_batch_size})...\n")

    # Rollout buffer: collect before scoring
    buf_pool_idx = []   # pool indices for buffered entries
    buf_expert   = []   # (obs, act, rew) tuples from expert rollout
    buf_orig_obs = []
    buf_orig_act = []
    buf_orig_rew = []

    prev_milestone = len(out_obs) // args.save_every

    for i in range(start_idx, N):
        if i % 200 == 0:
            print(f"  [{i:>5}/{N}]  kept={len(out_obs)}  "
                  f"skip_done={n_skip_done}  skip_gap={n_skip_gap}")

        s0 = pool_state[i, 0]

        result = rollout_from_state(expert_model, env, s0, T, device)
        if result is None:
            n_skip_done += 1
        else:
            buf_pool_idx.append(i)
            buf_expert.append(result)
            buf_orig_obs.append(pool_obs[i])
            buf_orig_act.append(pool_action[i])
            buf_orig_rew.append(pool_reward[i])

        flush = len(buf_pool_idx) >= args.score_batch_size or i == N - 1
        if not flush or not buf_pool_idx:
            continue

        # -- Batch score all buffered segments in one forward pass ----------
        B = len(buf_pool_idx)
        # Build (B, 2, T, dim) arrays: slot 0 = expert, slot 1 = original
        obs_flat = np.empty((B * 2, T, obs_dim), dtype=np.float32)
        act_flat = np.empty((B * 2, T, act_dim), dtype=np.float32)
        rew_flat = np.empty((B * 2, T),          dtype=np.float32)
        for j in range(B):
            obs_flat[j * 2]     = buf_expert[j][0]
            obs_flat[j * 2 + 1] = buf_orig_obs[j]
            act_flat[j * 2]     = buf_expert[j][1]
            act_flat[j * 2 + 1] = buf_orig_act[j]
            rew_flat[j * 2]     = buf_expert[j][2]
            rew_flat[j * 2 + 1] = buf_orig_rew[j]

        scores_dict = score_trajectories(
            obs_flat, act_flat, rew_flat,
            oracle, args.discount, args.mcmc_samples, device,
        )
        rl_sum = scores_dict["rl_sum"].reshape(B, 2)  # (B, 2)

        for j in range(B):
            expert_score = float(rl_sum[j, 0])
            orig_score   = float(rl_sum[j, 1])
            improvement  = abs(expert_score - orig_score)

            if improvement < args.min_adv_gap:
                n_skip_gap += 1
                continue

            if expert_score >= orig_score:
                pair_obs = np.stack([buf_expert[j][0], buf_orig_obs[j]])
                pair_act = np.stack([buf_expert[j][1], buf_orig_act[j]])
                pair_rew = np.stack([buf_expert[j][2], buf_orig_rew[j]])
                pair_adv = np.array([expert_score, orig_score], dtype=np.float32)
            else:
                pair_obs = np.stack([buf_orig_obs[j], buf_expert[j][0]])
                pair_act = np.stack([buf_orig_act[j], buf_expert[j][1]])
                pair_rew = np.stack([buf_orig_rew[j], buf_expert[j][2]])
                pair_adv = np.array([orig_score, expert_score], dtype=np.float32)

            out_obs.append(pair_obs)
            out_action.append(pair_act)
            out_reward.append(pair_rew)
            out_adv.append(pair_adv)
            out_impr.append(improvement)
            out_ckpt.append(int(pool_ckpt[buf_pool_idx[j]]))

        # Save progress after each batch when a new milestone is crossed
        milestone = len(out_obs) // args.save_every
        if milestone > prev_milestone and len(out_obs) > 0:
            save_npz(
                prog_path,
                obs=np.stack(out_obs, axis=0),
                action=np.stack(out_action, axis=0),
                reward=np.stack(out_reward, axis=0),
                adv_scores=np.stack(out_adv, axis=0),
                improvement=np.array(out_impr, dtype=np.float32),
                checkpoint_step=np.array(out_ckpt, dtype=np.int64),
                next_pool_idx=np.array(i + 1),
                n_skip_done=np.array(n_skip_done),
                n_skip_gap=np.array(n_skip_gap),
            )
            print(f"  [progress saved at pool idx {i+1}, {len(out_obs)} kept]")
            prev_milestone = milestone

        buf_pool_idx.clear()
        buf_expert.clear()
        buf_orig_obs.clear()
        buf_orig_act.clear()
        buf_orig_rew.clear()

    # ------------------------------------------------------------------
    # Final assembly
    # ------------------------------------------------------------------
    N_out = len(out_obs)
    print(f"\n{'='*55}")
    print(f"Segments processed  : {N}")
    print(f"Kept                : {N_out}")
    print(f"Skipped (done early): {n_skip_done}")
    print(f"Skipped (gap)       : {n_skip_gap}")

    if N_out == 0:
        print("WARNING: 0 samples kept. Lower --min-adv-gap or check pool/expert.")
        return

    obs_out    = np.stack(out_obs,    axis=0)   # (N_out, 2, T, obs_dim)
    action_out = np.stack(out_action, axis=0)
    reward_out = np.stack(out_reward, axis=0)
    adv_out    = np.stack(out_adv,    axis=0)   # (N_out, 2)
    impr_out   = np.array(out_impr,   dtype=np.float32)
    ckpt_out   = np.array(out_ckpt,   dtype=np.int64)

    # Sort by improvement magnitude (largest corrective signal first)
    if args.sort_by_improvement:
        order      = np.argsort(impr_out)[::-1]
        obs_out    = obs_out[order]
        action_out = action_out[order]
        reward_out = reward_out[order]
        adv_out    = adv_out[order]
        impr_out   = impr_out[order]
        ckpt_out   = ckpt_out[order]
        print(f"Sorted by improvement (largest first)")

    print(f"\nImprovement (rl_sum expert - original):")
    print(f"  mean={impr_out.mean():.3f}  std={impr_out.std():.3f}"
          f"  min={impr_out.min():.3f}  max={impr_out.max():.3f}")

    print(f"\nOutput shapes:")
    print(f"  obs    : {obs_out.shape}  (K=2: 0=expert, 1=original)")
    print(f"  action : {action_out.shape}")
    print(f"  reward : {reward_out.shape}")
    print(f"  adv    : {adv_out.shape}  (rl_sum per K)")
    print(f"  impr   : {impr_out.shape} (expert - original rl_sum)")

    save_npz(
        out_path,
        obs=obs_out,
        action=action_out,
        reward=reward_out,
        adv_scores=adv_out,
        improvement=impr_out,
        checkpoint_step=ckpt_out,
    )
    print(f"\nSaved → {out_path}")

    if os.path.exists(prog_path):
        os.remove(prog_path)
        print(f"Progress file removed.")


if __name__ == "__main__":
    main()
