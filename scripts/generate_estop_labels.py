"""
Generate (non-sequential) E-stop feedback labels from a trajectory pool.

Simulates an oracle human who intervenes at time τ and produces ONE pair per
stopped trajectory:

    preferred (index 0) : prefix   σ_{0:τ}^{halt}  — trajectory halted at τ:
                          obs[τ] and action[τ] are repeated for steps τ+1..T-1
                          and reward is zeroed out after τ.
    non-preferred (idx 1): full    σ_{0:T}  — trajectory allowed to continue

Both segments are stored at length T.  The stop_time field records τ so the
loss applies a mask when computing the discounted advantage sum for the prefix.

ARIC loss (implemented in EstopCPL):
    J_π(σ_{0:t}) = α Σ_{k=0}^{t} γ^k log π(a_k|s_k)
    L = -log σ(β'(J_π(σ_{0:τ}) − J_π(σ_{0:T})))

Simulation (same hyper-parameters as seq-estop):
    Δ_t = max(0, V*(s_t) − r_t − γ V*(s_{t+1}))     (one-step rl_sum)
    H_t = ρ H_{t-1} + Δ_t
    p_t = σ(λ(H_t − κ))
    τ   = first t where Bernoulli(p_t) = 1  (censored → skip)

Only stopped trajectories produce a pair; censored ones are discarded.

Output (--output-dir/<env>/estop_labels.npz):
    obs             : (M, 2, T, obs_dim)   [0]=halt prefix, [1]=full
    action          : (M, 2, T, act_dim)
    reward          : (M, 2, T)
    stop_time       : (M,)  int32  τ for each pair
    checkpoint_step : (M,)  int64

Usage:
    python scripts/generate_estop_labels.py \\
        --pool-path  /scratch/.../demo_pool/mw_drawer-open-v2/pool.npz \\
        --run-dir    /scratch/.../oracle_sac_all/mw_drawer-open-v2 \\
        --rho 0.8 --lam 2.0 --kappa 1.5 \\
        --output-dir /scratch/.../estop_labels

    # All envs:
    for ENV in mw_bin-picking-v2 mw_button-press-v2 mw_door-open-v2 \\
               mw_drawer-open-v2 mw_plate-slide-v2; do
        python scripts/generate_estop_labels.py \\
            --pool-path  /scratch/.../demo_pool/$ENV/pool.npz \\
            --run-dir    /scratch/.../oracle_sac_all/$ENV \\
            --rho 0.8 --lam 2.0 --kappa 1.5 \\
            --output-dir /scratch/.../estop_labels
    done
"""

import argparse
import io
import os

import numpy as np
import torch

from research.utils.config import Config


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def save_npz(path, **arrays):
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
# Per-step disadvantage via one-step rl_sum (same as generate_seq_estop_labels)
# ---------------------------------------------------------------------------

def compute_per_step_disadvantage(obs_b, reward_b, oracle, mcmc_samples, discount, device):
    """
    Δ_t = max(0, V*(s_t) − r_t − γ V*(s_{t+1}))  for t = 0..T-2, Δ_{T-1} = 0.
    """
    obs_t    = torch.from_numpy(obs_b).float().to(device)
    reward_t = torch.from_numpy(reward_b).float().to(device)

    with torch.no_grad():
        obs_enc   = oracle.network.encoder(obs_t)
        obs_exp   = obs_enc.unsqueeze(2).expand(-1, -1, mcmc_samples, -1)
        sampled_a = oracle.network.actor(obs_exp).sample()
        v         = oracle.network.critic(obs_exp, sampled_a).mean(dim=0).mean(dim=2)  # (B, T)

        v_curr  = v[:, :-1]
        v_next  = v[:, 1:]
        r_curr  = reward_t[:, :-1]
        td_adv  = r_curr + discount * v_next - v_curr

        delta_core = (-td_adv).clamp(min=0.0)
        pad        = torch.zeros(obs_b.shape[0], 1, device=device, dtype=delta_core.dtype)
        delta      = torch.cat([delta_core, pad], dim=1)

    return delta.cpu().numpy()


# ---------------------------------------------------------------------------
# E-stop simulation (identical logic to seq-estop simulator)
# ---------------------------------------------------------------------------

def simulate_estop(delta, rho, lam, kappa, rng):
    """Returns tau (int stop time) or None (censored)."""
    T = len(delta)
    H = np.empty(T, dtype=np.float64)
    H[0] = delta[0]
    for t in range(1, T):
        H[t] = rho * H[t - 1] + delta[t]

    x = lam * (H - kappa)
    p = np.where(x >= 0,
                 1.0 / (1.0 + np.exp(-x)),
                 np.exp(x) / (1.0 + np.exp(x)))

    for t in range(T):
        if rng.random() < p[t]:
            return t
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate non-sequential e-stop labels (halt prefix vs full trajectory)."
    )
    parser.add_argument("--pool-path", type=str, required=True)
    parser.add_argument("--run-dir",   type=str, required=True)
    parser.add_argument("--oracle-checkpoint", type=str, default="best_model.pt")
    parser.add_argument("--rho",   type=float, default=0.8,
                        help="Memory decay ρ ∈ [0,1] (default: 0.8)")
    parser.add_argument("--lam",   type=float, default=2.0,
                        help="Sigmoid slope λ (default: 2.0)")
    parser.add_argument("--kappa", type=float, default=1.5,
                        help="Stop threshold κ (default: 1.5)")
    parser.add_argument("--discount",     type=float, default=0.99)
    parser.add_argument("--batch-size",   type=int,   default=256)
    parser.add_argument("--mcmc-samples", type=int,   default=64)
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument(
        "--output-dir", type=str,
        default="/scratch/general/vast/u1472210/estop_labels",
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
    print(f"Simulator params : ρ={args.rho}  λ={args.lam}  κ={args.kappa}")
    print(f"Discount γ       : {args.discount}")
    print(f"Batch size       : {args.batch_size}")
    print(f"MCMC samples     : {args.mcmc_samples}")
    print(f"Seed             : {args.seed}")
    print(f"Device           : {device}")
    print("=" * 65)

    out_dir  = os.path.join(args.output_dir, env_name)
    out_path = os.path.join(out_dir, "estop_labels.npz")
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
    print(f"  N={N:,} trajectories, T={T}, obs_dim={obs_dim}, act_dim={act_dim}")

    # ------------------------------------------------------------------
    # Load oracle
    # ------------------------------------------------------------------
    oracle_ckpt = os.path.join(args.run_dir, args.oracle_checkpoint)
    print(f"\nLoading oracle: {oracle_ckpt}")
    oracle = load_model(args.run_dir, oracle_ckpt, device)

    # ------------------------------------------------------------------
    # Compute per-step Δ_t (batched)
    # ------------------------------------------------------------------
    print(f"\nComputing Δ_t for {N:,} trajectories ...")
    delta_all = np.empty((N, T), dtype=np.float32)
    n_batches = (N + args.batch_size - 1) // args.batch_size

    for b in range(n_batches):
        s = b * args.batch_size
        e = min(s + args.batch_size, N)
        delta_all[s:e] = compute_per_step_disadvantage(
            pool_obs[s:e], pool_reward[s:e],
            oracle, args.mcmc_samples, args.discount, device,
        )
        if (b + 1) % 10 == 0 or b == n_batches - 1:
            print(f"  [{b+1:>4}/{n_batches}]  scored {e:>6,}/{N:,}")

    print(f"\nΔ_t  mean={delta_all.mean():.4f}  std={delta_all.std():.4f}"
          f"  p95={np.percentile(delta_all, 95):.4f}")

    h_ss = float(delta_all.mean()) / (1.0 - args.rho)
    p_ss = 1.0 / (1.0 + np.exp(-args.lam * (h_ss - args.kappa)))
    print(f"Steady-state H={h_ss:.4f}  → p_stop≈{p_ss:.3f}  (κ={args.kappa})")
    if p_ss < 0.05:
        print("  NOTE: very low stop probability — most trajectories will be censored.")
    elif p_ss > 0.95:
        print("  NOTE: very high stop probability — most trajectories will stop early.")

    # ------------------------------------------------------------------
    # Simulate e-stop, build halt-prefix vs full pairs
    # ------------------------------------------------------------------
    print(f"\nSimulating e-stop ...")
    rng = np.random.default_rng(args.seed)

    all_obs       = []
    all_action    = []
    all_reward    = []
    all_stop_time = []
    all_ckpt      = []

    n_stopped  = 0
    n_censored = 0
    tau_values = []

    for i in range(N):
        tau = simulate_estop(delta_all[i], args.rho, args.lam, args.kappa, rng)

        if tau is None:
            n_censored += 1
            continue  # censored — no e-stop signal

        n_stopped += 1
        tau_values.append(tau)

        # Halt prefix: obs[0:τ+1] then repeat s_τ for remaining steps.
        # action: a_τ repeated; reward: zeroed out after τ (halted, no reward).
        prefix_obs = pool_obs[i].copy()        # (T, obs_dim)
        prefix_obs[tau + 1:] = pool_obs[i, tau]   # repeat halted state

        prefix_act = pool_action[i].copy()     # (T, act_dim)
        prefix_act[tau + 1:] = pool_action[i, tau]  # repeat halted action

        prefix_rew = pool_reward[i].copy()     # (T,)
        prefix_rew[tau + 1:] = 0.0             # no reward after halt

        # Full trajectory (unchanged)
        full_obs = pool_obs[i]
        full_act = pool_action[i]
        full_rew = pool_reward[i]

        all_obs.append(np.stack([prefix_obs, full_obs], axis=0))      # (2, T, obs_dim)
        all_action.append(np.stack([prefix_act, full_act], axis=0))
        all_reward.append(np.stack([prefix_rew, full_rew], axis=0))
        all_stop_time.append(tau)
        all_ckpt.append(int(pool_ckpt[i]))

        if (i + 1) % 2000 == 0 or i == N - 1:
            print(f"  [{i+1:>5}/{N}]  stopped={n_stopped}  censored={n_censored}")

    M = len(all_obs)
    if M == 0:
        print("\nERROR: 0 pairs generated. "
              "Adjust κ so more trajectories are stopped.")
        return

    tau_arr = np.array(tau_values)
    print(f"\nSimulation summary:")
    print(f"  Trajectories stopped  : {n_stopped:,}  ({100*n_stopped/N:.1f}%)")
    print(f"  Trajectories censored : {n_censored:,}  ({100*n_censored/N:.1f}%)")
    print(f"  Stop time τ           : "
          f"mean={tau_arr.mean():.1f}  median={np.median(tau_arr):.1f}"
          f"  p5={np.percentile(tau_arr,5):.0f}"
          f"  p95={np.percentile(tau_arr,95):.0f}")
    print(f"  Total pairs           : {M:,}  (one per stopped trajectory)")

    # ------------------------------------------------------------------
    # Stack and save
    # ------------------------------------------------------------------
    obs_out       = np.stack(all_obs,       axis=0).astype(np.float32)  # (M, 2, T, obs_dim)
    action_out    = np.stack(all_action,    axis=0).astype(np.float32)
    reward_out    = np.stack(all_reward,    axis=0).astype(np.float32)
    stop_time_out = np.array(all_stop_time, dtype=np.int32)
    ckpt_out      = np.array(all_ckpt,      dtype=np.int64)

    os.makedirs(out_dir, exist_ok=True)
    save_npz(
        out_path,
        obs=obs_out,
        action=action_out,
        reward=reward_out,
        stop_time=stop_time_out,
        checkpoint_step=ckpt_out,
    )

    print(f"\nSaved → {out_path}")
    print(f"  obs             : {obs_out.shape}  ([0]=halt prefix, [1]=full)")
    print(f"  action          : {action_out.shape}")
    print(f"  reward          : {reward_out.shape}")
    print(f"  stop_time       : {stop_time_out.shape}"
          f"  (mean τ={stop_time_out.mean():.1f})")
    print(f"  checkpoint_step : {ckpt_out.shape}")
    print(f"\nTrain with:")
    print(f"  dataset: EstopBuffer")
    print(f"  dataset_kwargs:")
    print(f"    path: .../estop_labels/{env_name}/estop_labels.npz")


if __name__ == "__main__":
    main()
