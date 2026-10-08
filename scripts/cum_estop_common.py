"""
Shared helpers for the Cumulative E-stop with Correction and Implicit
Acceptance model (see cumulative_estop_model.md). Written independently of
the holding-model E-stop pipeline (scripts/estop_hold_common.py) -- no
imports from it, no shared code, by design.

Core pieces:
  - frozen-oracle value estimation (MCMC mean over sampled actions)
  - per-step realized advantage / deficit, and the cumulative stopping rule
  - the Sum estimator U_E(suffix), used only to verify corrections/negatives
  - env snapshot restore + rollout of an arbitrary policy checkpoint
  - a worker-pool parallel map, mirroring the two-level (SLURM array x
    multiprocessing) parallelization used elsewhere in this repo
"""
import multiprocessing as mp
import os
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from research.utils.config import Config


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def save_npz(path: str, **arrays) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "wb") as f:
        np.savez(f, **arrays)  # pass a file handle, not a path -- np.savez
                                # auto-appends ".npz" to string paths, which
                                # would turn "foo.npz.tmp" into "foo.npz.tmp.npz"
    os.replace(tmp_path, path)


# ---------------------------------------------------------------------------
# Policy / oracle loading
# ---------------------------------------------------------------------------

def load_policy_weights_only(run_dir: str, checkpoint_path: str, observation_space, action_space,
                              device: str = "cpu"):
    """Like load_policy, but reuses an already-created env's observation/action
    space instead of constructing a brand new (expensive) MetaWorld env --
    for loading additional checkpoints (Pi_M/Pi_B bank members) of the same
    task after the first (oracle) load already created the shared env."""
    config = Config.load(run_dir)
    config["checkpoint"] = None
    config = config.parse()
    model = config.get_model(observation_space=observation_space, action_space=action_space, device=device)
    model.load(checkpoint_path)
    model.eval()
    return model


def load_policy(run_dir: str, checkpoint_path: str, device: str = "cpu"):
    """Load a frozen policy checkpoint (also used as the env's config source).
    Returns (model, env). Same role as the oracle loader used elsewhere in
    this repo, reimplemented independently."""
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
    env.reset()  # gym's OrderEnforcing wrapper requires one reset() before any step()
    return model, env


# ---------------------------------------------------------------------------
# Env restore / step (MetaWorldSawyerEnv.get_state()/set_state() adapter)
# ---------------------------------------------------------------------------

def reset_episode_counters(env) -> None:
    """Walk the wrapper chain resetting elapsed-step bookkeeping so a
    restored episode can run a full horizon instead of truncating early."""
    e = env
    while hasattr(e, "env") or hasattr(e, "_episode_steps"):
        if hasattr(e, "_episode_steps"):
            e._episode_steps = 0
        if hasattr(e, "_elapsed_steps"):
            e._elapsed_steps = 0
        if not hasattr(e, "env"):
            break
        e = e.env


def restore_state(env, state: np.ndarray) -> np.ndarray:
    env.set_state(state)
    reset_episode_counters(env)
    return env.get_obs().astype(np.float32)


def env_step(env, action: np.ndarray) -> Tuple[np.ndarray, float, bool]:
    """Normalize both the old gym 4-tuple and gymnasium 5-tuple step() APIs."""
    result = env.step(action)
    if len(result) == 5:
        next_obs, reward, terminated, truncated, _ = result
        done = terminated or truncated
    else:
        next_obs, reward, done, _ = result
    return next_obs.astype(np.float32), float(reward), bool(done)


def reconstruct_final_obs(env, state_before_last: np.ndarray, last_action: np.ndarray):
    """A pool segment stores exactly T (obs, action, reward) triples aligned so
    obs[t] is the state BEFORE action[t] -- there is no stored obs[T]. Recover
    the true boundary state with one extra restore + replay step."""
    restore_state(env, state_before_last)
    return env_step(env, last_action)


# ---------------------------------------------------------------------------
# Frozen oracle value function: V(s) via MCMC mean over sampled actions
# ---------------------------------------------------------------------------

def oracle_values(obs: np.ndarray, oracle, mcmc_samples: int, device: str) -> np.ndarray:
    """obs: (N, obs_dim). Returns (N,) V(s) estimates: critic-ensemble mean
    over `mcmc_samples` actions sampled from the frozen oracle actor."""
    obs_t = torch.from_numpy(obs).float().to(device)
    with torch.no_grad():
        obs_enc = oracle.network.encoder(obs_t)
        obs_exp = obs_enc.unsqueeze(1).expand(-1, mcmc_samples, -1)
        sampled_a = oracle.network.actor(obs_exp).sample()
        v = oracle.network.critic(obs_exp, sampled_a).mean(dim=0)
        v = v.mean(dim=1)
    return v.cpu().numpy()


# ---------------------------------------------------------------------------
# Section 4: cumulative deficit and the first-crossing stopping rule
#
# g_t is read here as the standard realized 1-step advantage of the action
# actually taken, A_t = r_t + gamma*V(s_{t+1}) - V(s_t), rather than the
# spec's literal max_a Q_E(s,a) formulation -- this needs only V(s) (the
# same MCMC-mean estimator above), not a second max-seeking estimator over
# actions. d_t = [-A_t]_+ : how much worse than the critic's own expectation
# this step turned out to be.
# ---------------------------------------------------------------------------

def per_step_deficits(reward: np.ndarray, values: np.ndarray, gamma: float) -> np.ndarray:
    """reward: (T,).  values: (T+1,) V(s_0..s_T) with the endpoint convention
    already applied (values[T] = V at the true boundary state, via
    reconstruct_final_obs). Returns (T,) deficits d_t >= 0."""
    T = len(reward)
    assert len(values) == T + 1, f"expected {T + 1} values, got {len(values)}"
    advantage = reward + gamma * values[1:] - values[:-1]
    return np.clip(-advantage, a_min=0.0, a_max=None)


def cumulative_stop_index(deficits: np.ndarray, H: float) -> Optional[int]:
    """tau = min{t : C_t >= H}, C_t = sum_{k<=t} d_k. None if never crossed
    (censored / no-stop segment)."""
    cumulative = np.cumsum(deficits)
    crossed = np.nonzero(cumulative >= H)[0]
    return int(crossed[0]) if len(crossed) > 0 else None


# ---------------------------------------------------------------------------
# Section 5: the Sum estimator U_E(suffix) -- used only to verify candidate
# corrections/negatives, never to find tau (that only uses per-step deficits)
# ---------------------------------------------------------------------------

def segment_score(reward: np.ndarray, values: np.ndarray, gamma: float) -> float:
    """U_E(zeta) = sum_k gamma^k r_k + gamma^L V(s_L) - V(s_0).
    reward: (L,).  values: (L+1,) = [V(s_0), ..., V(s_L)]."""
    L = len(reward)
    assert len(values) == L + 1
    discounted_return = float(values[L])
    for t in reversed(range(L)):
        discounted_return = float(reward[t]) + gamma * discounted_return
    return discounted_return - float(values[0])


def discounted_length(L: int, gamma: float) -> float:
    """W_L = sum_{k=0}^{L-1} gamma^k, used to scale acceptance margins by
    horizon (Sections 6.3 / 7.2): m(L) = delta * W_L."""
    if gamma == 1.0:
        return float(L)
    return float((1.0 - gamma ** L) / (1.0 - gamma))


# ---------------------------------------------------------------------------
# Rollout an arbitrary policy checkpoint from a restored snapshot
# ---------------------------------------------------------------------------

def rollout_policy(env, state_t: np.ndarray, policy, horizon: int, device: str,
                    stochastic: bool = True) -> Dict:
    """Restore env to state_t and roll `policy` forward for `horizon` steps.

    Returns dict:
        obs       : (n_steps, obs_dim)  pre-step states, index 0 = s_t
        action    : (n_steps, act_dim)
        reward    : (n_steps,)
        final_obs : (obs_dim,)  state after the last executed step
        n_steps   : int  (< horizon only if the env signalled done early)
    """
    obs = restore_state(env, state_t)
    obs_list, act_list, rew_list = [], [], []
    done = False
    for _ in range(horizon):
        obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(device)
        with torch.no_grad():
            obs_enc = policy.network.encoder(obs_t)
            dist = policy.network.actor(obs_enc)
            if isinstance(dist, torch.distributions.Distribution):
                action = dist.sample() if stochastic else dist.mean
            else:
                action = dist
            action = action.squeeze(0).cpu().numpy()
        obs_list.append(obs)
        next_obs, reward, done = env_step(env, action)
        act_list.append(action.copy())
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
# Parallel map across pool segments (SLURM array x multiprocessing, same
# two-level pattern used elsewhere in this repo -- one persistent worker
# process per core, each owning its own env + oracle + checkpoint-bank
# policies for its whole lifetime).
# ---------------------------------------------------------------------------

_worker_ctx: Dict = {}


def _init_worker(make_envs_fn: Callable, worker_kwargs: Dict) -> None:
    _worker_ctx.update(make_envs_fn())
    _worker_ctx.update(worker_kwargs)


def run_parallel(tasks: List, n_workers: int, make_envs_fn: Callable,
                  process_task_fn: Callable, worker_kwargs: Dict) -> List:
    """tasks: list of picklable task tuples, passed to process_task_fn(task)
    inside each worker (process_task_fn must be a module-level function that
    reads shared state from the _worker_ctx global via this module)."""
    if n_workers <= 1:
        _init_worker(make_envs_fn, worker_kwargs)
        return [process_task_fn(t) for t in tasks]

    ctx = mp.get_context("spawn")
    with ctx.Pool(
        n_workers, initializer=_init_worker, initargs=(make_envs_fn, worker_kwargs)
    ) as pool:
        return pool.map(process_task_fn, tasks)
