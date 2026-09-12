"""
Wraps multi-type-feedback's existing ppo_make_metaworld_env (utils.py) to match
research/envs/metaworld.py::MetaWorldSawyerEnv's observation/reward/horizon
convention -- the one r_theta and ARIC's policy were both trained on. Without
this, PPO would train on raw 39-dim obs / unscaled reward / a 500-step horizon /
no hand-position randomization at reset, breaking the "only the scorer changed"
premise the whole comparison depends on (see research/envs/metaworld.py for the
reference implementation this ports).

Built as a proper gymnasium.Wrapper AROUND ppo_make_metaworld_env's existing
output (not a reimplementation from ALL_V2_ENVIRONMENTS) so it inherits whatever
step()/reset() tuple convention that already-working function uses, and composes
correctly with downstream wrappers that assert isinstance(env, gym.Env) (e.g.
MetaWorldMonitor, Monitor). The rest of this framework standardizes on
gymnasium's 5-tuple (obs, reward, terminated, truncated, info) API (see utils.py's
own `import gymnasium as gym` and its use of gymnasium.wrappers.TimeLimit), so
this wrapper assumes the same -- UNTESTED against real MetaWorld V2 in this
environment (the installed metaworld here is a newer, incompatible V3-only
release with no working V2 env dict), so this should get a real smoke run on the
training cluster before being trusted.

Registered behind a new env-name prefix ("research-metaworld-<name>") wherever
env construction dispatches on env name, so the original ppo_make_metaworld_env
path is untouched for any other experiment.
"""

import os
from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.vec_env import DummyVecEnv
from train_baselines.utils import ppo_make_metaworld_env
from train_baselines.wrappers import MetaWorldMonitor

RESEARCH_MW_HORIZON = 250
RESEARCH_MW_PREFIX = "research-metaworld-"


def trim_mw_obs(obs: np.ndarray) -> np.ndarray:
    # Matches research/envs/metaworld.py::trim_mw_obs exactly: drop the duplicated
    # robot-state block, keep two object observations, for a more Markovian obs.
    return np.concatenate((obs[:18], obs[22:]), dtype=np.float32)


class ResearchMetaWorldCompatWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, sparse: bool = False, randomize_hand: bool = True):
        super().__init__(env)
        self.sparse = sparse
        self.randomize_hand = randomize_hand

        low, high = self.env.observation_space.low, self.env.observation_space.high
        assert low.shape[0] == 39, "expected raw 39-dim MetaWorld obs before trimming"
        self.observation_space = gym.spaces.Box(low=trim_mw_obs(low), high=trim_mw_obs(high), dtype=np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if self.randomize_hand:
            raw_env = self.env.unwrapped
            high = np.array([0.25, 0.15, 0.2], dtype=np.float32)
            hand_init_pos = raw_env.hand_init_pos + np.random.uniform(low=-high, high=high)
            hand_init_pos = np.clip(hand_init_pos, a_min=raw_env.mocap_low, a_max=raw_env.mocap_high)
            hand_init_pos = np.expand_dims(hand_init_pos, axis=0)
            for _ in range(50):
                raw_env.data.set_mocap_pos("mocap", hand_init_pos)
                raw_env.data.set_mocap_quat("mocap", np.array([1, 0, 1, 0]))
                raw_env.do_simulation([-1, 1], raw_env.frame_skip)
            obs = raw_env._get_obs()
        return trim_mw_obs(obs.astype(np.float32)), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self.sparse:
            reward = float(info.get("success", 0.0))
        else:
            reward = reward / 10.0
        return trim_mw_obs(obs.astype(np.float32)), reward, terminated, truncated, info


def ppo_make_research_metaworld_env(env_id: str, seed: Optional[int] = None, sparse: bool = False) -> gym.Env:
    """Drop-in analogue of ppo_make_metaworld_env, matching research/'s obs/reward/horizon convention."""
    base_env = ppo_make_metaworld_env(env_id, seed)
    # Nesting TimeLimits is safe: gymnasium's TimeLimit tracks elapsed steps
    # independently, so this tighter 250-step limit fires before the inner 500.
    limited_env = TimeLimit(base_env, max_episode_steps=RESEARCH_MW_HORIZON)
    return ResearchMetaWorldCompatWrapper(limited_env, sparse=sparse)


def make_vec_research_metaworld_env(
    env_id: str,
    n_envs: int = 1,
    seed: Optional[int] = None,
    start_index: int = 0,
    monitor_dir: Optional[str] = None,
    wrapper_class=None,
    vec_env_cls=None,
    vec_env_kwargs=None,
    monitor_kwargs=None,
    sparse: bool = False,
    **_ignored,
):
    """Mirrors train_baselines.utils.make_vec_metaworld_env's body exactly, swapping in
    ppo_make_research_metaworld_env. Kept as a close structural copy (not a call into the
    original) since the two factories take different constructor args (env_id, seed, sparse
    vs. just env_id, seed)."""
    vec_env_kwargs = {} if vec_env_kwargs is None else vec_env_kwargs
    monitor_kwargs = {} if monitor_kwargs is None else monitor_kwargs
    real_env_id = env_id[len(RESEARCH_MW_PREFIX) :] if env_id.startswith(RESEARCH_MW_PREFIX) else env_id

    def make_env(rank):
        def _init():
            env_seed = seed + rank if seed is not None else None
            env = ppo_make_research_metaworld_env(real_env_id, env_seed, sparse=sparse)
            if seed is not None:
                env.action_space.seed(env_seed)
            monitor_path = os.path.join(monitor_dir, str(rank)) if monitor_dir is not None else None
            if monitor_path is not None:
                os.makedirs(monitor_dir, exist_ok=True)
            env = MetaWorldMonitor(env, filename=monitor_path, **monitor_kwargs)
            if wrapper_class is not None:
                env = wrapper_class(env)
            return env

        return _init

    vec_env_cls = DummyVecEnv if vec_env_cls is None else vec_env_cls
    return vec_env_cls([make_env(i + start_index) for i in range(n_envs)], **vec_env_kwargs)


def install_research_metaworld_dispatch(sparse: bool = False) -> None:
    """
    Monkeypatches train_baselines.exp_manager's module-level `make_vec_metaworld_env`
    reference so that ExperimentManager.create_envs -- unmodified -- routes any env
    name starting with "research-metaworld-" through make_vec_research_metaworld_env
    (and everything else through the original function). create_envs resolves that
    name from its enclosing module's globals at call time, so patching the module
    attribute (not the original train_baselines.utils function) is what redirects it.

    This is the only way this bridge touches exp_manager.py's behavior, and it is
    purely additive: any env name that doesn't start with "research-metaworld-" is
    passed through to the original, untouched function.

    Call this once, before constructing any ExperimentManager.
    """
    import train_baselines.exp_manager as exp_manager_module

    original_fn = exp_manager_module.make_vec_metaworld_env

    def dispatch(env_id, **kwargs):
        if str(env_id).startswith(RESEARCH_MW_PREFIX):
            return make_vec_research_metaworld_env(str(env_id), sparse=sparse, **kwargs)
        return original_fn(env_id, **kwargs)

    exp_manager_module.make_vec_metaworld_env = dispatch
