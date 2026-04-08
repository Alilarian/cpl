"""
Phase 4: Generate demonstrative feedback dataset from the trajectory pool.

For each segment in pool.npz:
  1. Restore env to s_0 = state[i, 0]
  2. Roll out expert (best_model.pt) from s_0                     → 1 candidate
  3. Roll out 1 checkpoint per stratified tier from s_0            → n_tiers candidates
  4. Include the original pool segment as a candidate              → 1 candidate
     (already starts at s_0, no extra rollout needed)
  5. Score all candidates with oracle rl_sum = Σ r_t + V(s_T) - V(s_0)
  6. τ_D  = argmax rl_sum  → placed at output index 0
     τ_1..K-1 = remaining candidates sorted descending
  7. Quality filter: skip if rl_sum(τ_D) - rl_sum(τ_worst) < --min-adv-gap

Total candidates K = 1 (expert) + n_tiers + 1 (original)
                   = --n-counterfactuals + 1

n_tiers is derived automatically from --n-counterfactuals:
    n_tiers = n_counterfactuals - 1  (expert and original fill the other 2 slots)

Stratified tier sampling:
  - Splits the checkpoint step range into n_tiers equal-width bands
  - Pre-loads --checkpoints-per-tier candidate models per tier
  - Per segment: randomly picks 1 from each tier's pre-loaded pool
  - Guarantees counterfactuals span the full quality spectrum

Resumable: progress is saved every --save-every segments to
  <output-dir>/<env>/demo_labels_K<K>_progress.npz
  On re-run the script reloads this file and continues from the last saved index.
  The final output replaces the progress file once all segments are processed.

Output (--output-dir/<env>/demo_labels_K<K>.npz):
    obs              : (N, K, T, obs_dim)   index 0 = demo, 1..K-1 = counterfactuals
    action           : (N, K, T, act_dim)
    reward           : (N, K, T)
    adv_scores       : (N, K)               rl_sum values, sorted descending
    checkpoint_step  : (N,)                 source checkpoint of original pool segment

Usage:
    python scripts/generate_demo_labels.py \\
        --pool-path datasets/demo_pool/mw_drawer-open-v2/pool.npz \\
        --run-dir runs/runs/chpc/oracle_sac_all/mw_drawer-open-v2 \\
        --n-counterfactuals 6 \\
        --checkpoints-per-tier 2 \\
        --min-adv-gap 0.0 \\
        --output-dir datasets/demo_labels
"""

import argparse
import io
import os

import numpy as np
import torch

from research.utils.config import Config


def save_npz(path, **arrays):
    """Atomically save a compressed npz file (write to tmp, then rename)."""
    tmp_path = path + ".tmp"
    with io.BytesIO() as buf:
        np.savez_compressed(buf, **arrays)
        buf.seek(0)
        with open(tmp_path, "wb") as f:
            f.write(buf.read())
    os.replace(tmp_path, path)


# ---------------------------------------------------------------------------
# Model loading
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
    return model, env


# ---------------------------------------------------------------------------
# Oracle scoring  (rl_sum is primary; all metrics computed for analysis)
# ---------------------------------------------------------------------------

def score_trajectories(obs, action, reward, oracle, discount, mcmc_samples, device):
    """
    Score a batch of trajectories using oracle advantage metrics.

    Primary metric: rl_sum = Σ_t r_t + V(s_T) - V(s_0)
    Also computes rl_lp, rl_dir, rl_dis_dir, rl_dis_sum for analysis.

    Args:
        obs    : (B, T, obs_dim)
        action : (B, T, act_dim)
        reward : (B, T)

    Returns:
        dict  key → np.ndarray (B,)
    """
    B, T, _ = obs.shape
    obs_t    = torch.from_numpy(obs).float().to(device)
    action_t = torch.from_numpy(action).float().to(device)
    reward_t = torch.from_numpy(reward).float().to(device)

    out = {}

    with torch.no_grad():
        obs_enc = oracle.network.encoder(obs_t)         # (B, T, D)
        dist    = oracle.network.actor(obs_enc)

        if isinstance(dist, torch.distributions.Distribution):
            lp = dist.log_prob(torch.clamp(action_t, min=-0.999, max=0.999))  # (B, T)
            out["rl_lp"] = lp.sum(dim=-1).cpu().numpy()
        else:
            lp = -torch.square(dist - action_t).sum(dim=-1)
            out["rl_lp"] = lp.sum(dim=-1).cpu().numpy()

        if hasattr(oracle.network, "critic"):
            q = oracle.network.critic(obs_enc, action_t).mean(dim=0)  # (B, T)

            # V(s) via MCMC: sample M actions, average Q values
            obs_exp   = obs_enc.unsqueeze(2).expand(-1, -1, mcmc_samples, -1)  # (B, T, M, D)
            sampled_a = oracle.network.actor(obs_exp).sample()                  # (B, T, M, Da)
            v         = oracle.network.critic(obs_exp, sampled_a).mean(dim=0)   # (B, T, M)
            v         = v.mean(dim=2)                                           # (B, T)

            disc = torch.pow(
                discount * torch.ones(T, device=device),
                torch.arange(T, device=device).float(),
            ).unsqueeze(0)                                                      # (1, T)

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


# ---------------------------------------------------------------------------
# Rollout from a saved state
# ---------------------------------------------------------------------------

def rollout_from_state(model, env, s0, segment_length, device):
    """
    Restore env to s0, roll out model deterministically for segment_length steps.

    Returns (obs, action, reward) each (T, dim), or None if episode ends early.
    """
    env.set_state(s0)
    # Reset wrapper state counters so the episode can proceed without calling reset()
    # Walk the wrapper stack and fix any relevant flags/counters
    w = env
    while w is not None:
        if hasattr(w, '_elapsed_steps'):   # TimeLimit
            w._elapsed_steps = 0
        if hasattr(w, '_has_reset'):       # OrderEnforcing
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
        np.stack(ep_obs, axis=0),
        np.stack(ep_act, axis=0),
        np.array(ep_rew, dtype=np.float32),
    )


# ---------------------------------------------------------------------------
# Stratified checkpoint pool
# ---------------------------------------------------------------------------

def get_checkpoint_paths(run_dir, checkpoint_interval):
    """Return sorted (step, path) list for model_<step>.pt at the given interval."""
    ckpts = []
    for fname in os.listdir(run_dir):
        if not fname.startswith("model_") or not fname.endswith(".pt"):
            continue
        try:
            step = int(fname[len("model_"):-len(".pt")])
        except ValueError:
            continue
        if step % checkpoint_interval == 0:
            ckpts.append((step, os.path.join(run_dir, fname)))
    ckpts.sort(key=lambda x: x[0])
    return ckpts


def build_stratified_pool(run_dir, checkpoints, n_tiers, checkpoints_per_tier, device):
    """
    Divide checkpoints into n_tiers equal-width step ranges.
    Pre-load up to checkpoints_per_tier models from each tier.

    Returns:
        list of n_tiers lists, each containing (step, model) tuples.
        Per segment, 1 model is randomly drawn from each inner list.
    """
    if not checkpoints:
        return []

    steps      = np.array([s for s, _ in checkpoints])
    tier_edges = np.linspace(steps[0], steps[-1], n_tiers + 1)
    tier_models = []

    for t in range(n_tiers):
        lo, hi = tier_edges[t], tier_edges[t + 1]
        mask   = (steps >= lo) & (steps <= hi) if t == n_tiers - 1 else (steps >= lo) & (steps < hi)

        tier_ckpts = [(s, p) for (s, p), m in zip(checkpoints, mask) if m]
        if not tier_ckpts:
            print(f"  [WARN] Tier {t+1} ({int(lo):,}–{int(hi):,}) has no checkpoints — skipping")
            tier_models.append([])
            continue

        n_sample = min(checkpoints_per_tier, len(tier_ckpts))
        chosen   = [tier_ckpts[j] for j in
                    np.random.choice(len(tier_ckpts), n_sample, replace=False)]

        print(f"  Tier {t+1}/{n_tiers}  steps {int(lo):>7,}–{int(hi):>7,}  "
              f"loading ckpts: " + ", ".join(f"{s:,}" for s, _ in chosen))

        loaded = []
        for s, path in chosen:
            m, _ = load_model(run_dir, path, device)
            loaded.append((s, m))
        tier_models.append(loaded)

    return tier_models


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate demonstrative feedback labels (Phase 4)."
    )
    parser.add_argument("--pool-path", type=str, required=True,
                        help="Path to pool.npz from build_trajectory_pool.py")
    parser.add_argument("--run-dir", type=str, required=True,
                        help="Oracle SAC run dir (config.yaml + best_model.pt + checkpoints)")
    parser.add_argument("--expert-checkpoint", type=str, default="best_model.pt",
                        help="Expert checkpoint filename (default: best_model.pt)")
    parser.add_argument("--checkpoint-interval", type=int, default=20000,
                        help="Step interval for intermediate checkpoints (default: 20000)")
    parser.add_argument("--n-counterfactuals", type=int, default=6,
                        help="Number of counterfactuals per demo (default: 6). "
                             "K = n_counterfactuals + 1 = 7. "
                             "n_tiers = n_counterfactuals - 1 (expert + original fill 2 slots).")
    parser.add_argument("--checkpoints-per-tier", type=int, default=2,
                        help="Candidate checkpoints pre-loaded per tier (default: 2). "
                             "1 is randomly chosen per segment.")
    parser.add_argument("--min-adv-gap", type=float, default=0.0,
                        help="Min rl_sum gap (best - worst) to keep a sample (default: 0.0)")
    parser.add_argument("--discount",     type=float, default=0.99)
    parser.add_argument("--mcmc-samples", type=int,   default=64,
                        help="MCMC samples for V(s) (default: 64)")
    parser.add_argument("--save-every",  type=int,   default=500,
                        help="Save progress checkpoint every N kept samples (default: 500).")
    parser.add_argument("--output-dir", type=str, default="datasets/demo_labels")
    parser.add_argument("--device",     type=str, default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Device: {device}")

    # n_tiers derived so that: 1 (expert) + n_tiers + 1 (original) = n_counterfactuals + 1 = K
    # → n_tiers = n_counterfactuals - 1
    n_tiers = args.n_counterfactuals - 1
    K       = args.n_counterfactuals + 1
    assert n_tiers >= 1, "--n-counterfactuals must be >= 2"
    print(f"\nChoice set K={K}  (1 demo + {args.n_counterfactuals} counterfactuals)")
    print(f"Composition: 1 expert + {n_tiers} stratified tiers + 1 original")

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
    print(f"  N={N} segments, T={T} steps, obs_dim={obs_dim}")

    # ------------------------------------------------------------------
    # Load expert (also used as oracle for scoring)
    # ------------------------------------------------------------------
    expert_path = os.path.join(args.run_dir, args.expert_checkpoint)
    print(f"\nLoading expert + oracle: {expert_path}")
    expert_model, env = load_model(args.run_dir, expert_path, device)
    oracle = expert_model

    # ------------------------------------------------------------------
    # Build stratified counterfactual pool
    # ------------------------------------------------------------------
    print(f"\nBuilding stratified pool ({n_tiers} tiers × {args.checkpoints_per_tier} ckpts/tier):")
    all_ckpts   = get_checkpoint_paths(args.run_dir, args.checkpoint_interval)
    tier_models = build_stratified_pool(
        args.run_dir, all_ckpts, n_tiers, args.checkpoints_per_tier, device,
    )
    n_active_tiers = sum(1 for t in tier_models if t)
    actual_K = 1 + n_active_tiers + 1   # expert + active tiers + original
    if actual_K != K:
        print(f"  [WARN] Some tiers were empty. Actual K={actual_K} (expected {K})")
        K = actual_K

    print(f"\nScoring metric: rl_sum = Σ r_t + V(s_T) - V(s_0)\n")

    # ------------------------------------------------------------------
    # Output paths + resume
    # ------------------------------------------------------------------
    env_name    = os.path.basename(os.path.dirname(args.pool_path))
    out_dir     = os.path.join(args.output_dir, env_name)
    os.makedirs(out_dir, exist_ok=True)
    out_path    = os.path.join(out_dir, f"demo_labels_K{K}.npz")
    prog_path   = os.path.join(out_dir, f"demo_labels_K{K}_progress.npz")

    if os.path.exists(out_path):
        print(f"Output already exists: {out_path}")
        print("Delete it to regenerate.")
        return

    out_obs, out_action, out_reward, out_adv, out_ckpt = [], [], [], [], []
    n_skip_done  = 0
    n_skip_gap   = 0
    start_idx    = 0

    if os.path.exists(prog_path):
        print(f"Resuming from progress file: {prog_path}")
        prog = np.load(prog_path, allow_pickle=True)
        start_idx   = int(prog["next_pool_idx"])
        n_skip_done = int(prog["n_skip_done"])
        n_skip_gap  = int(prog["n_skip_gap"])
        if prog["obs"].shape[0] > 0:
            for row in prog["obs"]:    out_obs.append(row)
            for row in prog["action"]: out_action.append(row)
            for row in prog["reward"]: out_reward.append(row)
            for row in prog["adv_scores"]: out_adv.append(row)
            for v in prog["checkpoint_step"]: out_ckpt.append(int(v))
        print(f"  Resumed at pool index {start_idx}, {len(out_obs)} samples kept so far\n")

    # ------------------------------------------------------------------
    # Process each pool segment
    # ------------------------------------------------------------------
    for i in range(start_idx, N):
        if i % 200 == 0:
            print(f"  [{i:>5}/{N}]  kept={len(out_obs)}  "
                  f"skip_done={n_skip_done}  skip_gap={n_skip_gap}")

        s0 = pool_state[i, 0]   # full env physics state at segment start

        # ---- Build candidate set -----------------------------------------

        cand_obs, cand_act, cand_rew = [], [], []

        # 1. Expert rollout from s_0
        result = rollout_from_state(expert_model, env, s0, T, device)
        if result is None:
            n_skip_done += 1
            continue
        cand_obs.append(result[0])
        cand_act.append(result[1])
        cand_rew.append(result[2])

        # 2. One rollout per stratified tier
        tier_ok = True
        for tier in tier_models:
            if not tier:
                continue
            _, m = tier[np.random.randint(len(tier))]
            result = rollout_from_state(m, env, s0, T, device)
            if result is None:
                tier_ok = False
                break
            cand_obs.append(result[0])
            cand_act.append(result[1])
            cand_rew.append(result[2])

        if not tier_ok:
            n_skip_done += 1
            continue

        # 3. Original pool segment (starts at s_0 by construction — no re-rollout)
        cand_obs.append(pool_obs[i])
        cand_act.append(pool_action[i])
        cand_rew.append(pool_reward[i])

        # ---- Score with oracle rl_sum ------------------------------------

        obs_arr    = np.stack(cand_obs, axis=0)  # (K, T, obs_dim)
        action_arr = np.stack(cand_act, axis=0)  # (K, T, act_dim)
        reward_arr = np.stack(cand_rew, axis=0)  # (K, T)

        scores_dict = score_trajectories(
            obs_arr, action_arr, reward_arr,
            oracle, args.discount, args.mcmc_samples, device,
        )
        scores = scores_dict["rl_sum"]   # (K,)

        # ---- Quality filter ----------------------------------------------

        if scores.max() - scores.min() < args.min_adv_gap:
            n_skip_gap += 1
            continue

        # ---- Rank best → worst; demo is index 0 --------------------------

        order      = np.argsort(scores)[::-1]
        obs_arr    = obs_arr[order]
        action_arr = action_arr[order]
        reward_arr = reward_arr[order]
        scores     = scores[order]

        out_obs.append(obs_arr)
        out_action.append(action_arr)
        out_reward.append(reward_arr)
        out_adv.append(scores)
        out_ckpt.append(pool_ckpt[i])

        # ---- Periodic progress save -----------------------------------------
        if len(out_obs) % args.save_every == 0:
            save_npz(
                prog_path,
                obs=np.stack(out_obs, axis=0),
                action=np.stack(out_action, axis=0),
                reward=np.stack(out_reward, axis=0),
                adv_scores=np.stack(out_adv, axis=0),
                checkpoint_step=np.array(out_ckpt, dtype=np.int64),
                next_pool_idx=np.array(i + 1),
                n_skip_done=np.array(n_skip_done),
                n_skip_gap=np.array(n_skip_gap),
            )
            print(f"  [progress saved at pool idx {i+1}, {len(out_obs)} kept]")

    # ------------------------------------------------------------------
    # Save final output
    # ------------------------------------------------------------------
    N_out = len(out_obs)
    print(f"\n{'='*55}")
    print(f"Segments processed : {N}")
    print(f"Kept               : {N_out}")
    print(f"Skipped (done)     : {n_skip_done}")
    print(f"Skipped (gap)      : {n_skip_gap}")
    print(f"K (choice set size): {K}")

    if N_out == 0:
        print("WARNING: 0 samples kept. Lower --min-adv-gap or check pool.")
        return

    obs_out    = np.stack(out_obs,    axis=0)  # (N_out, K, T, obs_dim)
    action_out = np.stack(out_action, axis=0)
    reward_out = np.stack(out_reward, axis=0)
    adv_out    = np.stack(out_adv,    axis=0)  # (N_out, K)
    ckpt_out   = np.array(out_ckpt, dtype=np.int64)

    print(f"\nOutput shapes:")
    print(f"  obs    : {obs_out.shape}  (K dim: 0=demo, 1..{K-1}=counterfactuals)")
    print(f"  action : {action_out.shape}")
    print(f"  reward : {reward_out.shape}")
    print(f"  adv    : {adv_out.shape}  (rl_sum, descending)")

    save_npz(
        out_path,
        obs=obs_out,
        action=action_out,
        reward=reward_out,
        adv_scores=adv_out,
        checkpoint_step=ckpt_out,
    )
    print(f"\nSaved → {out_path}")

    # Remove progress file now that the final output is safely written
    if os.path.exists(prog_path):
        os.remove(prog_path)
        print(f"Progress file removed: {prog_path}")


if __name__ == "__main__":
    main()
