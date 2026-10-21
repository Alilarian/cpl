"""Reward wrappers on the real MetaWorld env (skipped when MetaWorld/mujoco is unavailable)."""
import csv
import functools
import os
import pickle

import numpy as np
import pytest
import torch

try:
    import cloudpickle
    from research.utils.config import get_env

    _env = get_env("mw_button-press-v2", {"sparse": False}, None, {})
    _env.close()
    HAVE_MW = True
except Exception:  # pragma: no cover
    HAVE_MW = False

pytestmark = pytest.mark.skipif(not HAVE_MW, reason="MetaWorld env not available")

from research.reward_learning.ensemble import EnsembleMLP, RewardEnsemble
from research.reward_learning.envs import LearnedRewardWrapper, ShadowRewardWrapper, make_env

ENV = "mw_button-press-v2"


def _rollout(env, actions, seed):
    np.random.seed(seed)
    obs = env.reset()
    out = []
    for a in actions:
        o, r, d, info = env.step(a)
        out.append((o.copy(), r, d))
        if d:
            break
    return obs, out


def _ckpt(tmp_path):
    net = EnsembleMLP(35, 4, ensemble_size=3, hidden_layers=(32, 32), seeds=[1, 2, 3])
    path = str(tmp_path / "r.pt")
    RewardEnsemble(net, torch.tensor([0.1, 0.0, -0.1]), torch.tensor([1.0, 0.5, 2.0])).save(
        path, extra=dict(mu_true=0.6, sigma_true=0.3))
    return path


def test_true_mode_is_identical_to_the_unwrapped_env(tmp_path):
    actions = np.random.RandomState(0).uniform(-1, 1, (250, 4)).astype(np.float32)
    plain = get_env(ENV, {"sparse": False}, None, {})
    log = str(tmp_path / "eps.csv")
    wrapped = LearnedRewardWrapper(get_env(ENV, {"sparse": False}, None, {}), "true", episode_log=log)
    o1, s1 = _rollout(plain, actions, 7)
    o2, s2 = _rollout(wrapped, actions, 7)
    assert np.array_equal(o1, o2) and len(s1) == len(s2) == 250
    for (a, ra, da), (b, rb, db) in zip(s1, s2):
        assert np.array_equal(a, b) and ra == rb and da == db
    with open(log) as f:
        row = list(csv.DictReader(f))[0]
    assert np.isclose(float(row["true_return"]), sum(r for _, r, _ in s1))
    assert np.isclose(float(row["learned_return"]), float(row["true_return"]))
    assert int(row["env_step"]) == 250


def test_learned_mode_reward_and_shadow_eval(tmp_path):
    path = _ckpt(tmp_path)
    a_map, b_map = 0.3, 0.6
    model = RewardEnsemble.load(path)
    actions = np.random.RandomState(1).uniform(-1.2, 1.2, (20, 4)).astype(np.float32)  # some out of bounds
    env = LearnedRewardWrapper(get_env(ENV, {"sparse": False}, None, {}), "learned", path, (a_map, b_map))
    np.random.seed(3)
    obs = env.reset()
    for a in actions:
        nxt, r, d, info = env.step(a)
        expected = a_map * float(model.reward(obs[None], np.clip(a, -1, 1)[None])[0]) + b_map
        assert np.isclose(r, expected, atol=1e-5)
        assert "true_reward" in info
        obs = nxt
    shadow = ShadowRewardWrapper(get_env(ENV, {"sparse": False}, None, {}), "learned", path, (a_map, b_map))
    plain = get_env(ENV, {"sparse": False}, None, {})
    np.random.seed(4); shadow.reset()
    np.random.seed(4); plain.reset()
    for a in actions:
        _, r1, _, info = shadow.step(a)
        _, r2, _, _ = plain.step(a)
        assert r1 == r2  # eval reward stays ground truth
        assert "learned_reward" in info and "learned_reward_std" in info
    eps = shadow.pop_episodes()
    assert len(eps) == 1 and eps[0]["obs"].shape == (20, 35) and eps[0]["member_reward"].shape == (20, 3)


def test_env_factory_pickles_for_async_env(tmp_path):
    fn = functools.partial(make_env, env_name=ENV, env_kwargs={"sparse": False}, wrapper=None, wrapper_kwargs={},
                           role="train", mode="learned", reward_ckpt=_ckpt(tmp_path), affine=(0.3, 0.6),
                           episode_log=str(tmp_path / "e.csv"))
    env = pickle.loads(cloudpickle.dumps(fn))()
    assert isinstance(env, LearnedRewardWrapper)
    env.close()
