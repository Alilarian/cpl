"""
Generate Sequential E-stop (seq-estop) feedback labels from a trajectory pool.

Simulates an oracle human who monitors each trajectory and presses the
e-stop button when accumulated intervention pressure exceeds a threshold.

Algorithm (per trajectory):
    Step 1: Per-step disadvantage  Δ_t = max(0, V*(s_t) - Q*(s_t, a_t))
    Step 2: Intervention score     H_t = ρ H_{t-1} + Δ_t,  H_0 = Δ_0
    Step 3: Stop probability       p_t = σ(λ(H_t − κ))
    Step 4: Sample stop time       τ ~ first Bernoulli(p_t)=1, or censored
    Step 5: Labels                 continue for t < τ, stop at t = τ
    Step 6: Grounded segments:
              stop_seg  = obs[t-h+1 : t+1]   (last h steps ending at t)
              cont_seg  = obs[t     : t+h]    (next h steps starting at t)

Valid decision times: t ∈ [h-1, T-h]  (both segments need h steps available)

Output format is K=2 pairs compatible with CorrBuffer / DemoCPL.
Index 0 is always the preferred segment (label always = 0):
    - at stop time τ:        preferred = stop_seg,  non-preferred = cont_seg
    - at continue times t<τ: preferred = cont_seg,  non-preferred = stop_seg
Censored trajectories (no stop) produce all-continue pairs.

Output (--output-dir/<env>/seq_estop_labels.npz):
    obs             : (M, 2, h, obs_dim)   [0]=preferred, [1]=non-preferred
    action          : (M, 2, h, act_dim)
    reward          : (M, 2, h)
    stop_event      : (M,)                 1 at actual stop time τ, 0 otherwise
    timestep        : (M,)                 decision time t within trajectory
    traj_idx        : (M,)                 source trajectory index in pool
    checkpoint_step : (M,)

Usage:
    python scripts/generate_seq_estop_labels.py \\
        --pool-path  /scratch/.../demo_pool/mw_drawer-open-v2/pool.npz \\
        --run-dir    /scratch/.../oracle_sac_all/mw_drawer-open-v2 \\
        --rho 0.8 --lam 2.0 --kappa 1.5 --horizon 10 \\
        --output-dir /scratch/.../seq_estop_labels

    # All envs (loop):
    for ENV in mw_bin-picking-v2 mw_button-press-v2 mw_door-open-v2 \\
               mw_drawer-open-v2 mw_plate-slide-v2; do
        python scripts/generate_seq_estop_labels.py \\
            --pool-path  /scratch/.../demo_pool/$ENV/pool.npz \\
            --run-dir    /scratch/.../oracle_sac_all/$ENV \\
            --rho 0.8 --lam 2.0 --kappa 1.5 --horizon 10 \\
            --output-dir /scratch/.../seq_estop_labels
    done

Parameters:
    --rho    ρ ∈ [0,1]: memory decay for running score H_t       (default: 0.8)
    --lam    λ > 0:     sigmoid slope / strictness                (default: 2.0)
    --kappa  κ:         stop threshold (H_t ≈ κ → p_stop ≈ 0.5) (default: 1.5)
    --horizon h:        local continuation window length          (default: 10)
"""

import argparse
import io
import os

import numpy as np
import torch

from research.utils.config import Config


# ---------------------------------------------------------------------------
# Shared utilities (same pattern as generate_pref_labels.py)
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
    return model


# ---------------------------------------------------------------------------
# Per-step disadvantage via one-step rl_sum (TD advantage)
# ---------------------------------------------------------------------------

def compute_per_step_disadvantage(obs_b, reward_b, oracle, mcmc_samples, discount, device):
    """
    Compute Δ_t = max(0, V*(s_t) − r_t − γ V*(s_{t+1})) for a batch of B segments.

    This is the negative one-step rl_sum (TD advantage):
        rl_sum_t = r_t + γ V(s_{t+1}) − V(s_t)   (consistent with generate_pref_labels.py)
        Δ_t      = max(0, −rl_sum_t)

    V*(s_t) is estimated by MCMC-averaging Q over oracle-policy actions — no
    direct Q(s_t, a_t) forward pass required.  The last step (t=T-1) has no
    successor in the segment and is padded with Δ_{T-1} = 0.

    Args:
        obs_b    : (B, T, obs_dim)  numpy
        reward_b : (B, T)           numpy
        discount : float γ

    Returns:
        delta : (B, T)  numpy, clamped to >= 0
    """
    obs_t    = torch.from_numpy(obs_b).float().to(device)     # (B, T, obs_dim)
    reward_t = torch.from_numpy(reward_b).float().to(device)  # (B, T)

    with torch.no_grad():
        obs_enc = oracle.network.encoder(obs_t)   # (B, T, D)

        # V(s_t) ≈ E_{a~π*}[Q(s_t, a)] via MCMC
        obs_exp   = obs_enc.unsqueeze(2).expand(-1, -1, mcmc_samples, -1)  # (B, T, M, D)
        sampled_a = oracle.network.actor(obs_exp).sample()                  # (B, T, M, act_dim)
        v = oracle.network.critic(obs_exp, sampled_a).mean(dim=0)           # (B, T, M)
        v = v.mean(dim=2)                                                    # (B, T)

        # One-step rl_sum: rl_sum_t = r_t + γ V(s_{t+1}) − V(s_t)  for t = 0..T-2
        v_curr   = v[:, :-1]           # (B, T-1)
        v_next   = v[:, 1:]            # (B, T-1)
        r_curr   = reward_t[:, :-1]    # (B, T-1)
        td_adv   = r_curr + discount * v_next - v_curr   # (B, T-1)

        # Δ_t = max(0, −rl_sum_t); pad last step with 0 (no successor in segment)
        delta_core = (-td_adv).clamp(min=0.0)             # (B, T-1)
        pad        = torch.zeros(obs_b.shape[0], 1, device=device, dtype=delta_core.dtype)
        delta      = torch.cat([delta_core, pad], dim=1)  # (B, T)

    return delta.cpu().numpy()


# ---------------------------------------------------------------------------
# Sequential e-stop simulation (per trajectory, pure numpy)
# ---------------------------------------------------------------------------

def simulate_seq_estop(delta, rho, lam, kappa, rng):
    """
    Run the sequential e-stop simulation for a single trajectory.

    Args:
        delta : (T,) per-step disadvantage Δ_t
        rho   : float ∈ [0,1], memory decay
        lam   : float > 0, sigmoid slope
        kappa : float, stop threshold
        rng   : numpy.random.Generator

    Returns:
        tau : int (stop time) or None (censored — no stop occurred)
        H   : (T,) running intervention scores
        p   : (T,) stop probabilities
    """
    T = len(delta)
    H = np.empty(T, dtype=np.float64)
    H[0] = delta[0]
    for t in range(1, T):
        H[t] = rho * H[t - 1] + delta[t]

    # Numerically stable sigmoid: σ(x) = 1/(1+e^{-x})
    x = lam * (H - kappa)
    p = np.where(x >= 0,
                 1.0 / (1.0 + np.exp(-x)),
                 np.exp(x) / (1.0 + np.exp(x)))

    # Sequential Bernoulli sampling — stop at first success
    for t in range(T):
        if rng.random() < p[t]:
            return t, H, p

    return None, H, p  # censored


# ---------------------------------------------------------------------------
# Segment pair builder (per trajectory)
# ---------------------------------------------------------------------------

def build_pairs(traj_obs, traj_action, traj_reward, tau, h):
    """
    Build (stop, continue) grounded segment pairs for one trajectory.

    Valid decision times: t ∈ [h-1, T-h] so that:
        stop_seg  = traj_obs[t-h+1 : t+1]  has exactly h steps
        cont_seg  = traj_obs[t     : t+h]   has exactly h steps

    Index 0 is always the preferred segment (label = 0, compatible with CorrBuffer):
        t == tau  → stop preferred  → obs[0]=stop_seg, obs[1]=cont_seg
        t <  tau  → cont preferred  → obs[0]=cont_seg, obs[1]=stop_seg
    Censored trajectories (tau=None) produce all-continue pairs.

    Args:
        traj_obs    : (T, obs_dim)
        traj_action : (T, act_dim)
        traj_reward : (T,)
        tau         : int or None
        h           : int, local horizon

    Returns:
        list of dicts with keys:
            obs (2,h,obs_dim), action (2,h,act_dim), reward (2,h),
            stop_event (float 0/1), timestep (int)
    """
    T = traj_obs.shape[0]
    t_min = h - 1      # stop_seg needs t - h + 1 >= 0  ⟹  t >= h - 1
    t_max = T - h      # cont_seg needs t + h - 1 < T   ⟹  t <= T - h

    if t_max < t_min:
        return []  # trajectory too short for any comparison

    # Clamp tau to valid range; skip if tau is before any valid decision time
    if tau is None:
        t_end = t_max        # censored: all-continue up to t_max
    else:
        t_end = min(tau, t_max)
        if t_end < t_min:
            return []        # stop occurred before any valid decision time

    pairs = []
    for t in range(t_min, t_end + 1):
        is_stop = (tau is not None) and (t == tau)

        stop_obs = traj_obs[t - h + 1: t + 1]         # (h, obs_dim)
        stop_act = traj_action[t - h + 1: t + 1]      # (h, act_dim)
        stop_rew = traj_reward[t - h + 1: t + 1]      # (h,)

        cont_obs = traj_obs[t: t + h]                  # (h, obs_dim)
        cont_act = traj_action[t: t + h]               # (h, act_dim)
        cont_rew = traj_reward[t: t + h]               # (h,)

        # Always put preferred segment at index 0
        if is_stop:
            p_obs, p_act, p_rew = stop_obs, stop_act, stop_rew
            n_obs, n_act, n_rew = cont_obs, cont_act, cont_rew
        else:
            p_obs, p_act, p_rew = cont_obs, cont_act, cont_rew
            n_obs, n_act, n_rew = stop_obs, stop_act, stop_rew

        pairs.append({
            "obs":        np.stack([p_obs, n_obs], axis=0),    # (2, h, obs_dim)
            "action":     np.stack([p_act, n_act], axis=0),    # (2, h, act_dim)
            "reward":     np.stack([p_rew, n_rew], axis=0),    # (2, h)
            "stop_event": 1.0 if is_stop else 0.0,
            "timestep":   t,
        })

    return pairs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Simulate sequential e-stop feedback and build ARIC segment pairs."
    )
    parser.add_argument(
        "--pool-path", type=str, required=True,
        help="Path to pool.npz from build_trajectory_pool.py",
    )
    parser.add_argument(
        "--run-dir", type=str, required=True,
        help="Oracle SAC run dir (config.yaml + checkpoint)",
    )
    parser.add_argument(
        "--oracle-checkpoint", type=str, default="best_model.pt",
        help="Oracle checkpoint filename (default: best_model.pt)",
    )
    # Simulator hyper-parameters
    parser.add_argument(
        "--rho", type=float, default=0.8,
        help="Memory decay ρ ∈ [0,1] for running score H_t (default: 0.8)",
    )
    parser.add_argument(
        "--lam", type=float, default=2.0,
        help="Sigmoid slope λ (strictness) (default: 2.0)",
    )
    parser.add_argument(
        "--kappa", type=float, default=1.5,
        help="Stop threshold κ: H_t ≈ κ ⟹ p_stop ≈ 0.5 (default: 1.5)",
    )
    parser.add_argument(
        "--horizon", type=int, default=10,
        help="Local continuation horizon h in steps (default: 10)",
    )
    # Oracle computation
    parser.add_argument(
        "--batch-size", type=int, default=256,
        help="Segments scored per GPU forward pass (default: 256). Reduce if OOM.",
    )
    parser.add_argument(
        "--mcmc-samples", type=int, default=64,
        help="MCMC samples for V(s) estimation (default: 64)",
    )
    parser.add_argument(
        "--discount", type=float, default=0.99,
        help="Discount γ for one-step rl_sum: r_t + γ V(s_{t+1}) − V(s_t) (default: 0.99)",
    )
    # Output
    parser.add_argument(
        "--max-pairs", type=int, default=None,
        help="Cap total output pairs (stratified subsample preserving stop/continue "
             "ratio). Default: no cap.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for stop-time sampling (default: 42)",
    )
    parser.add_argument(
        "--output-dir", type=str,
        default="/scratch/general/vast/u1472210/seq_estop_labels",
    )
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    env_name = os.path.basename(os.path.dirname(args.pool_path))

    print("=" * 65)
    print(f"Environment      : {env_name}")
    print(f"Pool path        : {args.pool_path}")
    print(f"Oracle dir       : {args.run_dir}")
    print(f"Simulator params : ρ={args.rho}  λ={args.lam}  κ={args.kappa}  h={args.horizon}")
    print(f"Batch size       : {args.batch_size}")
    print(f"MCMC samples     : {args.mcmc_samples}")
    print(f"Discount γ       : {args.discount}")
    print(f"Max pairs        : {args.max_pairs if args.max_pairs else 'no cap'}")
    print(f"Seed             : {args.seed}")
    print(f"Device           : {device}")
    print("=" * 65)

    # ------------------------------------------------------------------
    # Guard
    # ------------------------------------------------------------------
    out_dir  = os.path.join(args.output_dir, env_name)
    out_path = os.path.join(out_dir, "seq_estop_labels.npz")
    if os.path.exists(out_path):
        print(f"\nOutput already exists: {out_path}")
        print("Delete it to regenerate.")
        return

    # ------------------------------------------------------------------
    # Load pool
    # ------------------------------------------------------------------
    print(f"\nLoading pool ...")
    with open(args.pool_path, "rb") as f:
        pool = np.load(f)
        pool_obs    = pool["obs"]             # (N, T, obs_dim)
        pool_action = pool["action"]          # (N, T, act_dim)
        pool_reward = pool["reward"]          # (N, T)
        pool_ckpt   = pool["checkpoint_step"] # (N,)

    N, T, obs_dim = pool_obs.shape
    act_dim = pool_action.shape[2]
    h = args.horizon
    print(f"  N={N:,} segments, T={T}, obs_dim={obs_dim}, act_dim={act_dim}")
    print(f"  Valid decision times per trajectory: "
          f"t ∈ [{h-1}, {T-h}]  ({max(0, T - 2*h + 2)} max comparisons/traj)")

    if T < 2 * h:
        print(f"ERROR: T={T} < 2*h={2*h}. Increase pool segment length or reduce --horizon.")
        return

    # ------------------------------------------------------------------
    # Load oracle
    # ------------------------------------------------------------------
    oracle_ckpt = os.path.join(args.run_dir, args.oracle_checkpoint)
    print(f"\nLoading oracle: {oracle_ckpt}")
    oracle = load_model(args.run_dir, oracle_ckpt, device)

    # ------------------------------------------------------------------
    # Compute per-step Δ_t for all pool trajectories (batched)
    # ------------------------------------------------------------------
    print(f"\nComputing per-step disadvantage Δ_t for all {N:,} trajectories ...")
    delta_all = np.empty((N, T), dtype=np.float32)
    n_batches = (N + args.batch_size - 1) // args.batch_size

    for b in range(n_batches):
        s = b * args.batch_size
        e = min(s + args.batch_size, N)
        delta_all[s:e] = compute_per_step_disadvantage(
            pool_obs[s:e], pool_reward[s:e], oracle, args.mcmc_samples, args.discount, device,
        )
        if (b + 1) % 10 == 0 or b == n_batches - 1:
            print(f"  [{b+1:>4}/{n_batches}]  scored {e:>6,}/{N:,}")

    print(f"\nΔ_t statistics over all {N*T:,} steps:")
    print(f"  mean={delta_all.mean():.4f}  std={delta_all.std():.4f}"
          f"  p50={np.median(delta_all):.4f}"
          f"  p95={np.percentile(delta_all, 95):.4f}"
          f"  p99={np.percentile(delta_all, 99):.4f}")

    # Report steady-state H relative to κ (helps diagnose κ setting)
    h_ss = float(delta_all.mean()) / (1.0 - args.rho)
    p_ss = 1.0 / (1.0 + np.exp(-args.lam * (h_ss - args.kappa)))
    print(f"\nSteady-state H = E[Δ]/(1−ρ) = {h_ss:.4f}  "
          f"→ p_stop = {p_ss:.3f}  (κ={args.kappa})")
    if p_ss < 0.05:
        print("  NOTE: p_stop at steady-state is very low — "
              "most trajectories will be censored. Consider lowering κ.")
    elif p_ss > 0.95:
        print("  NOTE: p_stop at steady-state is very high — "
              "most trajectories will stop early. Consider raising κ.")

    # ------------------------------------------------------------------
    # Simulate seq-estop and build segment pairs
    # ------------------------------------------------------------------
    print(f"\nSimulating seq-estop and building segment pairs ...")
    rng = np.random.default_rng(args.seed)

    all_obs        = []
    all_action     = []
    all_reward     = []
    all_stop_event = []
    all_timestep   = []
    all_traj_idx   = []
    all_ckpt       = []

    n_stopped  = 0
    n_censored = 0
    tau_values = []

    for i in range(N):
        tau, _H, _p = simulate_seq_estop(
            delta_all[i], args.rho, args.lam, args.kappa, rng,
        )

        if tau is None:
            n_censored += 1
        else:
            n_stopped += 1
            tau_values.append(tau)

        pairs = build_pairs(
            pool_obs[i], pool_action[i], pool_reward[i], tau, h,
        )

        for pair in pairs:
            all_obs.append(pair["obs"])
            all_action.append(pair["action"])
            all_reward.append(pair["reward"])
            all_stop_event.append(pair["stop_event"])
            all_timestep.append(pair["timestep"])
            all_traj_idx.append(i)
            all_ckpt.append(int(pool_ckpt[i]))

        if (i + 1) % 2000 == 0 or i == N - 1:
            print(f"  [{i+1:>5}/{N}]  pairs={len(all_obs):>7,}  "
                  f"stopped={n_stopped}  censored={n_censored}")

    M = len(all_obs)
    if M == 0:
        print("\nERROR: 0 pairs generated. "
              "Check pool/oracle or adjust κ/horizon parameters.")
        return

    print(f"\nSimulation summary:")
    print(f"  Trajectories stopped  : {n_stopped:,}  ({100*n_stopped/N:.1f}%)")
    print(f"  Trajectories censored : {n_censored:,}  ({100*n_censored/N:.1f}%)")
    if tau_values:
        tau_arr = np.array(tau_values)
        print(f"  Stop time τ           : "
              f"mean={tau_arr.mean():.1f}  median={np.median(tau_arr):.1f}"
              f"  p5={np.percentile(tau_arr, 5):.0f}"
              f"  p95={np.percentile(tau_arr, 95):.0f}")
    stop_count = int(sum(all_stop_event))
    print(f"  Total pairs           : {M:,}  "
          f"(stop={stop_count}, continue={M - stop_count})")
    print(f"  Stop fraction         : {100*stop_count/M:.2f}%")

    # ------------------------------------------------------------------
    # Stack into arrays
    # ------------------------------------------------------------------
    obs_out        = np.stack(all_obs,    axis=0).astype(np.float32)  # (M, 2, h, obs_dim)
    action_out     = np.stack(all_action, axis=0).astype(np.float32)  # (M, 2, h, act_dim)
    reward_out     = np.stack(all_reward, axis=0).astype(np.float32)  # (M, 2, h)
    stop_event_out = np.array(all_stop_event, dtype=np.float32)       # (M,)
    timestep_out   = np.array(all_timestep,   dtype=np.int32)         # (M,)
    traj_idx_out   = np.array(all_traj_idx,   dtype=np.int32)         # (M,)
    ckpt_out       = np.array(all_ckpt,       dtype=np.int64)         # (M,)

    # ------------------------------------------------------------------
    # Optional cap (stratified to preserve stop/continue ratio)
    # ------------------------------------------------------------------
    if args.max_pairs is not None and M > args.max_pairs:
        print(f"\nSubsampling to max_pairs={args.max_pairs:,} "
              f"(stratified by stop/continue) ...")
        stop_idx = np.where(stop_event_out == 1.0)[0]
        cont_idx = np.where(stop_event_out == 0.0)[0]
        frac_stop    = len(stop_idx) / M
        n_stop_keep  = max(1, int(round(args.max_pairs * frac_stop)))
        n_cont_keep  = args.max_pairs - n_stop_keep

        keep_stop = rng.choice(stop_idx, size=min(n_stop_keep, len(stop_idx)), replace=False)
        keep_cont = rng.choice(cont_idx, size=min(n_cont_keep, len(cont_idx)), replace=False)
        keep      = np.sort(np.concatenate([keep_stop, keep_cont]))

        obs_out        = obs_out[keep]
        action_out     = action_out[keep]
        reward_out     = reward_out[keep]
        stop_event_out = stop_event_out[keep]
        timestep_out   = timestep_out[keep]
        traj_idx_out   = traj_idx_out[keep]
        ckpt_out       = ckpt_out[keep]
        M = len(obs_out)
        print(f"  After cap: {M:,} pairs  "
              f"(stop={int(stop_event_out.sum())}, "
              f"continue={M - int(stop_event_out.sum())})")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    os.makedirs(out_dir, exist_ok=True)
    save_npz(
        out_path,
        obs=obs_out,
        action=action_out,
        reward=reward_out,
        stop_event=stop_event_out,
        timestep=timestep_out,
        traj_idx=traj_idx_out,
        checkpoint_step=ckpt_out,
    )

    print(f"\nSaved → {out_path}")
    print(f"  obs             : {obs_out.shape}  (K=2, h={h})")
    print(f"  action          : {action_out.shape}")
    print(f"  reward          : {reward_out.shape}")
    print(f"  stop_event      : {stop_event_out.shape}  "
          f"({int(stop_event_out.sum())} stops, "
          f"{M - int(stop_event_out.sum())} continues)")
    print(f"  timestep        : {timestep_out.shape}")
    print(f"  traj_idx        : {traj_idx_out.shape}")
    print(f"  checkpoint_step : {ckpt_out.shape}")
    print(f"\nTrain with:")
    print(f"  dataset: CorrBuffer")
    print(f"  dataset_kwargs:")
    print(f"    path: .../seq_estop_labels/{env_name}/seq_estop_labels.npz")
    print(f"    batch_size: 96")


if __name__ == "__main__":
    main()
