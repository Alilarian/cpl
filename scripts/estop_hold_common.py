"""
Shared core for the "holding" E-stop model (Model A from the ARIC E-stop spec):
an oracle human who, at every candidate stopping time t, compares the recorded
continuation against a *physically simulated* holding rollout of the same
length, and intervenes at the first t where holding wins by more than a fixed
threshold. This is distinct from (and does not replace) the existing
generate_estop_labels.py / generate_seq_estop_labels.py, which never actually
simulate a holding branch -- they repeat the last recorded (s, a) and zero the
reward, which is a bookkeeping trick, not a holding counterfactual (see
tests/test_seq_estop_consistency.py for the consequences of that shortcut on
the sequential variant).

Reused verbatim from the rest of the pipeline (no need to re-derive):
  - pool.npz already stores a restorable MetaWorld snapshot at *every* step
    (state[:, t], from build_trajectory_pool.py), not just t=0. That is the
    hard part of "capture_env/restore_env" (spec Section 2) -- it's already
    solved by MetaWorldSawyerEnv.get_state()/set_state()
    (research/envs/metaworld.py), which folds qpos/qvel/mocap/goal-rand-vec
    into one restorable vector.
  - The oracle value-estimation pattern (encoder -> actor.sample() -> critic,
    MCMC-averaged) is identical to generate_seq_estop_labels.py /
    generate_pref_labels.py / generate_corr_labels.py.
  - save_npz / load_model / the wrapper-elapsed-steps reset trick are copied
    from generate_corr_labels.py, which is the closest existing analogue
    (restore env state -> roll out a controller -> score with the oracle).

Endpoint convention (spec Section 5): this MetaWorld formulation
(research/envs/metaworld.py::MetaWorldSawyerEnv.step) only ever sets done=True
via a time-limit cutoff at _max_episode_steps (and marks info["discount"]=1.0
there, i.e. "infinite bootstrap" -- a truncation, not a true MDP terminal).
There is no sparse-success early-termination in this env. So every segment
endpoint here is either an "ordinary dataset cut" or a "time-limit truncation"
in the spec's Section 5 table -- both rows say: bootstrap with V(s_h), never
zero it. This lets the whole pipeline use one endpoint rule instead of a
terminal/truncated branch. If this is ever pointed at an env with true
absorbing terminals, `reconstruct_final_obs`'s `done` return value is exactly
the hook needed to special-case that (not implemented here, per spec Section 5:
"for an initial implementation, use segments ... without an earlier true
termination").
"""

import io
import multiprocessing as mp
import os

import numpy as np
import torch

from research.utils.config import Config


# ---------------------------------------------------------------------------
# I/O + model loading (identical pattern to generate_corr_labels.py)
# ---------------------------------------------------------------------------

def save_npz(path, **arrays):
    """Atomically save a compressed npz (write to .tmp, then rename). Creates
    the parent directory if it doesn't exist yet."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp = path + ".tmp"
    with io.BytesIO() as buf:
        np.savez_compressed(buf, **arrays)
        buf.seek(0)
        with open(tmp, "wb") as f:
            f.write(buf.read())
    os.replace(tmp, path)


def load_model(run_dir, checkpoint_path, device):
    """Load the frozen oracle (also used as the env's config source). Returns (model, env)."""
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
# Env restoration (spec Section 2's capture_env/restore_env adapters)
# ---------------------------------------------------------------------------

def reset_episode_counters(env):
    """
    Walk the wrapper chain resetting any elapsed-step bookkeeping so a
    restored episode can run a full fresh horizon instead of truncating early.
    Copied from generate_corr_labels.py::rollout_from_state -- defensive for
    whatever wrapper stack config.get_train_env_fn() produces around
    MetaWorldSawyerEnv (which already resets its own _episode_steps /
    curr_path_length inside set_state(), but this guards a generic TimeLimit-
    style wrapper too).
    """
    w = env
    while w is not None:
        if hasattr(w, "_elapsed_steps"):
            w._elapsed_steps = 0
        if hasattr(w, "_has_reset"):
            w._has_reset = True
        w = getattr(w, "env", None)


def restore_state(env, state):
    """capture_env/restore_env adapter (spec Section 2): restore a snapshot, return obs."""
    env.set_state(state)
    reset_episode_counters(env)
    return env.get_obs().astype(np.float32)


def env_step(env, action):
    """Normalize both the old gym 4-tuple and gymnasium 5-tuple step() APIs."""
    result = env.step(action)
    if len(result) == 5:
        next_obs, reward, terminated, truncated, _ = result
        done = terminated or truncated
    else:
        next_obs, reward, done, _ = result
    return next_obs.astype(np.float32), float(reward), bool(done)


def reconstruct_final_obs(env, state_before_last, last_action):
    """
    The pool stores exactly T (obs, action, reward) triples aligned so obs[t]
    is the state BEFORE action[t] -- there is no stored "obs[T]" (the state
    after the segment's last recorded action). Recover the true s_T with one
    extra restore + replay step instead of silently reusing obs[T-1] as an
    approximate boundary (as generate_pref_labels.py / generate_corr_labels.py
    do for their own, different, scoring purposes).

    Returns (final_obs, final_reward, done).
    """
    restore_state(env, state_before_last)
    return env_step(env, last_action)


# ---------------------------------------------------------------------------
# Frozen oracle value function (identical MCMC pattern used throughout the
# rest of this pipeline's generate_*_labels.py scripts)
# ---------------------------------------------------------------------------

def oracle_state_values(obs, oracle, mcmc_samples, device):
    """
    obs: (N, obs_dim) numpy array of states (a flat batch -- caller decides
         what the N states are, e.g. one segment's T+1 boundary states, or a
         hold rollout's horizon+1 boundary states).

    Returns (N,) numpy V(s) estimates: critic-ensemble mean over `mcmc_samples`
    actions sampled from the frozen oracle actor. Same MCMC value-estimation
    pattern as generate_seq_estop_labels.py::compute_per_step_disadvantage,
    just with a single flat batch dim instead of (segment, time).
    """
    obs_t = torch.from_numpy(obs).float().to(device)
    with torch.no_grad():
        obs_enc = oracle.network.encoder(obs_t)                      # (N, D)
        obs_exp = obs_enc.unsqueeze(1).expand(-1, mcmc_samples, -1)  # (N, M, D)
        sampled_a = oracle.network.actor(obs_exp).sample()             # (N, M, act_dim)
        v = oracle.network.critic(obs_exp, sampled_a).mean(dim=0)      # (N, M)  ensemble mean
        v = v.mean(dim=1)                                              # (N,)    MCMC mean
    return v.cpu().numpy()


# ---------------------------------------------------------------------------
# Sum estimator (spec Section 4/6): backward Bellman-consistent telescoping
# ---------------------------------------------------------------------------

def continuation_scores(rewards, values, gamma):
    """
    S(C_t) = sum_{k=t}^{L-1} gamma^(k-t) r_k + gamma^(L-t) V(s_L) - V(s_t)
           = B_t - V(s_t),   B_L = V(s_L),  B_t = r_t + gamma * B_{t+1}

    rewards: (L,)    values: (L+1,) with the endpoint convention already applied.
    Returns (L,) scores, one per candidate t. O(L), excluding value-net calls.
    """
    L = len(rewards)
    assert len(values) == L + 1, f"expected {L + 1} values, got {len(values)}"
    scores = np.empty(L, dtype=np.float64)
    bootstrapped_return = float(values[L])
    for t in reversed(range(L)):
        bootstrapped_return = float(rewards[t]) + gamma * bootstrapped_return
        scores[t] = bootstrapped_return - float(values[t])
    return scores


# ---------------------------------------------------------------------------
# Fixed holding controller (spec Section 3)
# ---------------------------------------------------------------------------

def gripper_command_at(action_i, t):
    """
    The gripper command "in effect" when the hold controller takes over at
    candidate time t: the last action actually applied to reach s_t, i.e.
    action[t-1] (obs[t]/action[t]/reward[t] are aligned as a triple, so
    action[t-1] is what produced s_t). At t=0 there is no prior action inside
    this segment; fall back to action[0] (the about-to-be-issued command) as
    the least-arbitrary choice -- spec Section 3 explicitly requires the
    gripper command to be defined at t=0 too.
    """
    idx = t - 1 if t > 0 else 0
    return float(action_i[idx, -1])


def make_hold_action(gripper_command, act_dim):
    """
    pi_hold: zero end-effector displacement + the held gripper command.
    Verified against the standard MetaWorld-v2 SawyerXYZEnv convention: the
    first (act_dim - 1) dims are added as a delta to the mocap target (so zero
    holds the current target fixed) and the last dim drives the gripper
    actuator. If the installed metaworld version's gripper actuation is not
    itself "hold under a constant command", replace this with a feedback
    controller instead (spec Section 3).
    """
    action = np.zeros(act_dim, dtype=np.float32)
    action[-1] = gripper_command
    return action


def rollout_hold(env, state_t, gripper_command, horizon, act_dim):
    """
    Restore env to state_t and roll the fixed hold controller forward for
    `horizon` steps.

    Returns dict:
        obs      : (n_steps, obs_dim)  pre-step states, index 0 = s_t
        action   : (n_steps, act_dim)  the (identical) hold action, repeated
        reward   : (n_steps,)          oracle-definition reward at each hold step
        final_obs: (obs_dim,)          state after the last executed hold step
        n_steps  : int                 steps actually executed (< horizon only
                                        if the env signalled done early -- see
                                        module docstring on the endpoint
                                        convention for why that should not
                                        happen for ordinary segments here)
    """
    obs = restore_state(env, state_t)
    hold_action = make_hold_action(gripper_command, act_dim)

    obs_list, act_list, rew_list = [], [], []
    for _ in range(horizon):
        obs_list.append(obs)
        next_obs, reward, done = env_step(env, hold_action)
        act_list.append(hold_action.copy())
        rew_list.append(reward)
        obs = next_obs
        if done:
            break

    return {
        "obs": np.stack(obs_list, axis=0).astype(np.float32),
        "action": np.stack(act_list, axis=0).astype(np.float32),
        "reward": np.array(rew_list, dtype=np.float32),
        "final_obs": obs.astype(np.float32),
        "n_steps": len(act_list),
    }


# ---------------------------------------------------------------------------
# Per-segment search (spec Sections 7-10)
# ---------------------------------------------------------------------------

def evaluate_segment(
    env,
    oracle,
    obs_i,
    action_i,
    reward_i,
    state_i,
    gamma,
    mcmc_samples,
    device,
    min_horizon=1,
    threshold=0.0,
    stop_early=True,
):
    """
    Run the deterministic stopping search over one pool segment.

    obs_i/action_i/reward_i: (T, ...) arrays for this segment.
    state_i: (T, state_dim) restorable MetaWorld snapshots, one per step.
    stop_early: True for label generation (stop at tau = first t with
        Delta_t > threshold, spec Section 8). False for threshold calibration
        (spec's "do not stop at the first crossing ... you need the whole
        sequence to evaluate multiple thresholds without rerunning
        simulations") -- evaluates every eligible t regardless of threshold.

    Returns a dict:
        stopped     : bool
        stop_index  : int or None (tau)
        horizon     : int or None (= T - tau)
        oracle_gap  : float or None (Delta_tau)
        positive    : {"obs","action","reward"} hold suffix H_tau, or None
        negative    : {"obs","action","reward"} original suffix C_tau, or None
        gaps        : [(t, delta_or_None), ...] for every evaluated t, in order
                      (delta is None when the hold rollout terminated early --
                      spec Section 5: never silently shorten only one branch)
    """
    T = len(action_i)
    act_dim = action_i.shape[-1]

    final_obs, _final_reward, _final_done = reconstruct_final_obs(
        env, state_i[T - 1], action_i[T - 1],
    )

    all_obs = np.concatenate([obs_i, final_obs[None]], axis=0)  # (T+1, obs_dim)
    values = oracle_state_values(all_obs, oracle, mcmc_samples, device)
    cont_scores = continuation_scores(reward_i, values, gamma)  # (T,)

    gaps = []
    for t in range(0, T - min_horizon + 1):
        horizon = T - t
        gripper = gripper_command_at(action_i, t)
        hold = rollout_hold(env, state_i[t], gripper, horizon, act_dim)

        if hold["n_steps"] < horizon:
            gaps.append((t, None))
            continue

        hold_boundary_obs = np.concatenate([hold["obs"], hold["final_obs"][None]], axis=0)
        hold_values = oracle_state_values(hold_boundary_obs, oracle, mcmc_samples, device)

        disc = gamma ** np.arange(horizon)
        hold_score = float(
            (disc * hold["reward"]).sum()
            + (gamma ** horizon) * hold_values[-1]
            - hold_values[0]
        )
        delta = hold_score - float(cont_scores[t])
        gaps.append((t, delta))

        if stop_early and delta > threshold:
            return {
                "stopped": True,
                "stop_index": t,
                "horizon": horizon,
                "oracle_gap": delta,
                "positive": {
                    "obs": hold["obs"],
                    "action": hold["action"],
                    "reward": hold["reward"],
                },
                "negative": {
                    "obs": obs_i[t:T].copy(),
                    "action": action_i[t:T].copy(),
                    "reward": reward_i[t:T].copy(),
                },
                "gaps": gaps,
            }

    return {
        "stopped": False,
        "stop_index": None,
        "horizon": None,
        "oracle_gap": None,
        "positive": None,
        "negative": None,
        "gaps": gaps,
    }


def max_gap(gaps):
    """M_i = max_t Delta_t over evaluated (non-None) candidates, or -inf if none."""
    vals = [d for _, d in gaps if d is not None]
    return max(vals) if vals else float("-inf")


# ---------------------------------------------------------------------------
# Parallel map across segments (spec: "process each segment independently" --
# shared by generate_estop_hold_labels.py, stop_early=True, and
# tune_estop_hold_threshold.py, stop_early=False). Each worker owns one
# MetaWorld env + one CPU oracle copy for its whole lifetime, loaded once via
# the pool initializer rather than per segment.
# ---------------------------------------------------------------------------

_worker_ctx = {}


def make_oracle_env(run_dir, oracle_checkpoint, device="cpu"):
    """Default (oracle, env) factory for run_parallel -- picklable by reference
    (module-level function + plain string/None args), safe under spawn."""
    return load_model(run_dir, os.path.join(run_dir, oracle_checkpoint), device)


def _init_worker(make_oracle_env_fn, gamma, mcmc_samples, min_horizon, threshold, stop_early):
    oracle, env = make_oracle_env_fn()
    _worker_ctx.update(
        oracle=oracle, env=env, gamma=gamma, mcmc_samples=mcmc_samples,
        min_horizon=min_horizon, threshold=threshold, stop_early=stop_early,
    )


def _process_task(task):
    i, obs_i, action_i, reward_i, state_i = task
    ctx = _worker_ctx
    result = evaluate_segment(
        ctx["env"], ctx["oracle"], obs_i, action_i, reward_i, state_i,
        ctx["gamma"], ctx["mcmc_samples"], "cpu",
        min_horizon=ctx["min_horizon"], threshold=ctx["threshold"], stop_early=ctx["stop_early"],
    )
    return i, result


def run_parallel(tasks, n_workers, make_oracle_env_fn, gamma, mcmc_samples,
                  min_horizon, threshold, stop_early):
    """
    Yields (i, result) pairs from evaluate_segment for every task in `tasks`
    (an iterable of (i, obs_i, action_i, reward_i, state_i)).

    make_oracle_env_fn: zero-arg callable returning (oracle, env), called once
        per worker (not per segment). Must be picklable under spawn when
        n_workers > 1 -- a module-level function (optionally via
        functools.partial with picklable args) works; a closure/lambda does
        not. Production code passes functools.partial(make_oracle_env, run_dir,
        oracle_checkpoint); tests can inject a synthetic factory with no
        MetaWorld/CHPC dependency at all.

    n_workers <= 1: sequential, in-process (no multiprocessing overhead).
    n_workers > 1 : a spawn-context multiprocessing.Pool with one persistent
        worker per process (spawn, not fork, for MuJoCo/CUDA safety); results
        are yielded in submission order (chunksize=1 for even load balancing
        across variable-cost segments, at the cost of some IPC overhead).
    """
    init_args = (make_oracle_env_fn, gamma, mcmc_samples, min_horizon, threshold, stop_early)
    if n_workers <= 1:
        _init_worker(*init_args)
        for task in tasks:
            yield _process_task(task)
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=n_workers, initializer=_init_worker, initargs=init_args) as pool_exec:
            for i, result in pool_exec.imap(_process_task, tasks, chunksize=1):
                yield i, result
