"""
Phases 3, 5, 6 of the Cumulative E-stop model: for each pool segment, find
the first cumulative-deficit threshold crossing (or none), then either
  - generate and verify an imperfect correction from an intermediate-quality
    checkpoint bank (Pi_M), starting exactly at the stop state s_tau, or
  - (no stop) generate and verify one substantially-worse counterfactual from
    a weak checkpoint bank (Pi_B), starting at s_0, keeping the full original
    rollout as the accepted demonstration.

Output is two label files, each in the SAME (obs, action, reward) K=2 schema
CorrBuffer already reads (both arms are now full, equal, T-length
trajectories -- no padding/masking needed at all, so no new buffer class is
needed on the training side):

    <output-dir>/<env>/stop_correction_labels.npz
        obs/action/reward : (N, 2, T, ...)   index 0 = corrected (preferred),
                                               index 1 = original (non-preferred)
        stop_index, checkpoint_step, gap, attempts : (N,)  provenance

    <output-dir>/<env>/no_stop_demo_labels.npz
        obs/action/reward : (N, 2, T, ...)   index 0 = accepted demo (preferred),
                                               index 1 = poor counterfactual
        checkpoint_step, gap, attempts : (N,)  provenance

    <output-dir>/<env>/unresolved_events.npz
        pool_index, event_type (0=stop_unresolved, 1=no_stop_unpaired) : (N,)

Usage:
    python3 scripts/generate_cum_estop_labels.py \\
        --pool-path   datasets/mw_de_labels/mw_button-press-v2/pool.npz \\
        --run-dir     datasets/mw_de_labels/mw_button-press-v2/oracle \\
        --bands-path  datasets/mw_de_labels/mw_button-press-v2/checkpoint_bands.json \\
        --H 10.0 \\
        --output-dir  datasets/cum_estop_labels
"""
import argparse
import json
import os
import re

import numpy as np

import scripts.cum_estop_common as cec


def _make_envs(run_dir, oracle_checkpoint, bands, device):
    def factory():
        oracle, env = cec.load_policy(run_dir, os.path.join(run_dir, oracle_checkpoint), device)
        obs_space, act_space = env.observation_space, env.action_space
        intermediate = {
            b["checkpoint"]: cec.load_policy_weights_only(
                run_dir, os.path.join(run_dir, b["checkpoint"]), obs_space, act_space, device,
            )
            for b in bands["intermediate"]
        }
        weak = {
            b["checkpoint"]: cec.load_policy_weights_only(
                run_dir, os.path.join(run_dir, b["checkpoint"]), obs_space, act_space, device,
            )
            for b in bands["weak"]
        }
        return {"oracle": oracle, "env": env, "intermediate": intermediate, "weak": weak}
    return factory


def _try_correction(ctx, pool_index, attempt, tau, T, obs_i, action_i, reward_i, state_i,
                     original_suffix_score, prefix_score_values):
    """One candidate correction attempt. Returns (passed, result_dict_or_None)."""
    names = list(ctx["intermediate"].keys())
    rng = np.random.default_rng((ctx["seed"], pool_index, attempt))
    name = names[rng.integers(len(names))]
    policy = ctx["intermediate"][name]
    horizon = T - tau

    correction = cec.rollout_policy(ctx["env"], state_i[tau], policy, horizon, ctx["device"],
                                     stochastic=True)
    if correction["n_steps"] < horizon:
        return False, None  # early termination -- treat as a failed candidate, retry

    boundary_obs = np.concatenate([correction["obs"], correction["final_obs"][None]], axis=0)
    values = cec.oracle_values(boundary_obs, ctx["oracle"], ctx["mcmc_samples"], ctx["device"])
    correction_score = cec.segment_score(correction["reward"], values, ctx["gamma"])

    gap = correction_score - original_suffix_score
    margin = ctx["delta_c"] * cec.discounted_length(horizon, ctx["gamma"])
    if gap < margin:
        return False, None

    corrected_full_obs = np.concatenate([obs_i[:tau], correction["obs"]], axis=0)
    corrected_full_action = np.concatenate([action_i[:tau], correction["action"]], axis=0)
    corrected_full_reward = np.concatenate([reward_i[:tau], correction["reward"]], axis=0)
    return True, {
        "obs": corrected_full_obs, "action": corrected_full_action, "reward": corrected_full_reward,
        "checkpoint": name, "gap": gap, "attempts": attempt + 1,
    }


def _try_negative(ctx, pool_index, attempt, T, obs_i, action_i, reward_i, state_i,
                   accepted_score):
    names = list(ctx["weak"].keys())
    rng = np.random.default_rng((ctx["seed"], pool_index, attempt, 1))  # 1 = negative-branch discriminator
    name = names[rng.integers(len(names))]
    policy = ctx["weak"][name]

    bad = cec.rollout_policy(ctx["env"], state_i[0], policy, T, ctx["device"], stochastic=True)
    if bad["n_steps"] < T:
        return False, None

    boundary_obs = np.concatenate([bad["obs"], bad["final_obs"][None]], axis=0)
    values = cec.oracle_values(boundary_obs, ctx["oracle"], ctx["mcmc_samples"], ctx["device"])
    bad_score = cec.segment_score(bad["reward"], values, ctx["gamma"])

    gap = accepted_score - bad_score
    margin = ctx["delta_n"] * cec.discounted_length(T, ctx["gamma"])
    if gap < margin:
        return False, None

    return True, {
        "obs": bad["obs"], "action": bad["action"], "reward": bad["reward"],
        "checkpoint": name, "gap": gap, "attempts": attempt + 1,
    }


def _process_task(task):
    i, obs_i, action_i, reward_i, state_i = task
    ctx = cec._worker_ctx
    T = len(action_i)

    final_obs, _, _ = cec.reconstruct_final_obs(ctx["env"], state_i[-1], action_i[-1])
    all_obs = np.concatenate([obs_i, final_obs[None]], axis=0)
    values = cec.oracle_values(all_obs, ctx["oracle"], ctx["mcmc_samples"], ctx["device"])
    deficits = cec.per_step_deficits(reward_i, values, ctx["gamma"])
    tau = cec.cumulative_stop_index(deficits, ctx["H"])

    if tau is not None:
        original_suffix_score = cec.segment_score(reward_i[tau:], values[tau:], ctx["gamma"])
        for attempt in range(ctx["correction_budget"]):
            passed, result = _try_correction(
                ctx, i, attempt, tau, T, obs_i, action_i, reward_i, state_i,
                original_suffix_score, values,
            )
            if passed:
                return i, "stop_correction", tau, result, {
                    "original_obs": obs_i, "original_action": action_i, "original_reward": reward_i,
                }
        return i, "stop_unresolved", tau, None, None

    accepted_score = cec.segment_score(reward_i, values, ctx["gamma"])
    for attempt in range(ctx["negative_budget"]):
        passed, result = _try_negative(ctx, i, attempt, T, obs_i, action_i, reward_i, state_i, accepted_score)
        if passed:
            return i, "no_stop_demo", None, result, {
                "original_obs": obs_i, "original_action": action_i, "original_reward": reward_i,
            }
    return i, "no_stop_unpaired", None, None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-path", type=str, required=True)
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--oracle-checkpoint", type=str, default="best_model.pt")
    parser.add_argument("--bands-path", type=str, required=True)
    parser.add_argument("--H", type=float, required=True, help="cumulative deficit threshold")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--mcmc-samples", type=int, default=32)
    parser.add_argument("--delta-c", type=float, default=0.1,
                         help="min average per-step improvement required to accept a correction")
    parser.add_argument("--delta-n", type=float, default=0.3,
                         help="min average per-step gap required to accept a no-stop negative "
                              "(should exceed --delta-c, per spec Section 7.2)")
    parser.add_argument("--correction-budget", type=int, default=8)
    parser.add_argument("--negative-budget", type=int, default=8)
    parser.add_argument("--n-workers", type=int, default=1)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--subsample-n", type=int, default=None)
    parser.add_argument("--subsample-seed", type=int, default=0)
    parser.add_argument("--max-segments", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--env-name", type=str, required=True)
    args = parser.parse_args()

    assert args.delta_n >= args.delta_c, "--delta-n should be >= --delta-c (Section 7.2)"

    with open(args.bands_path) as f:
        bands = json.load(f)

    pool = np.load(args.pool_path)
    pool_obs, pool_action, pool_reward, pool_state = (
        pool["obs"], pool["action"], pool["reward"], pool["state"],
    )
    N = pool_obs.shape[0]
    orig_pool_idx = np.arange(N)

    if args.subsample_n is not None:
        assert args.subsample_n <= N
        rng = np.random.default_rng(args.subsample_seed)
        keep = np.sort(rng.choice(N, size=args.subsample_n, replace=False))
        pool_obs, pool_action, pool_reward, pool_state = (
            pool_obs[keep], pool_action[keep], pool_reward[keep], pool_state[keep],
        )
        orig_pool_idx = keep
        N = args.subsample_n

    if args.max_segments is not None and args.max_segments < N:
        pool_obs, pool_action, pool_reward, pool_state = (
            pool_obs[: args.max_segments], pool_action[: args.max_segments],
            pool_reward[: args.max_segments], pool_state[: args.max_segments],
        )
        orig_pool_idx = orig_pool_idx[: args.max_segments]
        N = args.max_segments

    shard_idx = np.arange(args.shard_id, N, args.num_shards)
    print(f"Shard {args.shard_id}/{args.num_shards}: {len(shard_idx)}/{N} segments, H={args.H}")

    tasks = [
        (int(orig_pool_idx[i]), pool_obs[i], pool_action[i], pool_reward[i], pool_state[i])
        for i in shard_idx
    ]
    worker_kwargs = dict(
        gamma=args.gamma, mcmc_samples=args.mcmc_samples, H=args.H, device=args.device,
        delta_c=args.delta_c, delta_n=args.delta_n,
        correction_budget=args.correction_budget, negative_budget=args.negative_budget,
        seed=args.seed,
    )
    make_envs_fn = _make_envs(args.run_dir, args.oracle_checkpoint, bands, args.device)
    results = cec.run_parallel(tasks, args.n_workers, make_envs_fn, _process_task, worker_kwargs)

    stop_rows, demo_rows, unresolved_rows = [], [], []
    for i, event_type, tau, result, original in results:
        if event_type == "stop_correction":
            stop_rows.append((i, tau, result, original))
        elif event_type == "no_stop_demo":
            demo_rows.append((i, result, original))
        elif event_type == "stop_unresolved":
            unresolved_rows.append((i, 0))
        elif event_type == "no_stop_unpaired":
            unresolved_rows.append((i, 1))

    def ckpt_step(name):
        # Parse the step from the checkpoint filename itself (e.g.
        # "model_100000.pt" -> 100000) -- NOT bands[...]["step"], which is the
        # nearest log.csv row's step and can differ from the actual file
        # loaded whenever select_checkpoint_bands.py snapped to a different
        # on-disk checkpoint.
        m = re.match(r"model_(\d+)\.pt$", name)
        return int(m.group(1)) if m else -1

    out_dir = os.path.join(args.output_dir, args.env_name)
    suffix = f"_shard{args.shard_id}of{args.num_shards}"

    if stop_rows:
        obs = np.stack([np.stack([r[2]["obs"], r[3]["original_obs"]]) for r in stop_rows])
        action = np.stack([np.stack([r[2]["action"], r[3]["original_action"]]) for r in stop_rows])
        reward = np.stack([np.stack([r[2]["reward"], r[3]["original_reward"]]) for r in stop_rows])
        cec.save_npz(
            os.path.join(out_dir, f"stop_correction_labels{suffix}.npz"),
            obs=obs.astype(np.float32), action=action.astype(np.float32), reward=reward.astype(np.float32),
            pool_index=np.array([r[0] for r in stop_rows], dtype=np.int32),
            stop_index=np.array([r[1] for r in stop_rows], dtype=np.int32),
            checkpoint_step=np.array([ckpt_step(r[2]["checkpoint"]) for r in stop_rows], dtype=np.int64),
            gap=np.array([r[2]["gap"] for r in stop_rows], dtype=np.float32),
            attempts=np.array([r[2]["attempts"] for r in stop_rows], dtype=np.int32),
            H=np.full(len(stop_rows), args.H, dtype=np.float32),
        )
    if demo_rows:
        obs = np.stack([np.stack([r[1]["obs"], r[2]["original_obs"]]) for r in demo_rows])
        action = np.stack([np.stack([r[1]["action"], r[2]["original_action"]]) for r in demo_rows])
        reward = np.stack([np.stack([r[1]["reward"], r[2]["original_reward"]]) for r in demo_rows])
        cec.save_npz(
            os.path.join(out_dir, f"no_stop_demo_labels{suffix}.npz"),
            obs=obs.astype(np.float32), action=action.astype(np.float32), reward=reward.astype(np.float32),
            pool_index=np.array([r[0] for r in demo_rows], dtype=np.int32),
            checkpoint_step=np.array([ckpt_step(r[1]["checkpoint"]) for r in demo_rows], dtype=np.int64),
            gap=np.array([r[1]["gap"] for r in demo_rows], dtype=np.float32),
            attempts=np.array([r[1]["attempts"] for r in demo_rows], dtype=np.int32),
            H=np.full(len(demo_rows), args.H, dtype=np.float32),
        )
    cec.save_npz(
        os.path.join(out_dir, f"unresolved_events{suffix}.npz"),
        pool_index=np.array([r[0] for r in unresolved_rows], dtype=np.int32),
        event_type=np.array([r[1] for r in unresolved_rows], dtype=np.int32),
    )

    print(f"stop_correction pairs: {len(stop_rows)}  no_stop_demo pairs: {len(demo_rows)}  "
          f"unresolved: {len(unresolved_rows)}  (of {len(shard_idx)} segments in this shard)")


if __name__ == "__main__":
    main()
