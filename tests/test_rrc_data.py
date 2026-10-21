"""Data loading on tiny synthetic files: ARIC subsample, streaming, episode recovery, split, cum_estop, credit."""
import numpy as np
import pytest

from research.reward_learning.data import (PoolIndex, aric_subsample_indices, load_choice_set, make_episode_split,
                                           read_npz_rows, tie_aware_target)

D, A = 35, 4


def _episodes(n_eps=4, T_ep=40, seed=0):
    rng = np.random.RandomState(seed)
    obs = rng.randn(n_eps, T_ep, D).astype(np.float32)
    act = rng.uniform(-0.9, 0.9, (n_eps, T_ep, A)).astype(np.float32)
    rew = rng.uniform(0, 1, (n_eps, T_ep)).astype(np.float32)
    return obs, act, rew


def _make_pool(path, n_rows=60, T=16, seed=0):
    eo, ea, er = _episodes(seed=seed)
    rng = np.random.RandomState(seed + 1)
    ep = rng.randint(0, eo.shape[0], n_rows)
    st = rng.randint(0, eo.shape[1] - T + 1, n_rows)
    obs = np.stack([eo[e, s : s + T] for e, s in zip(ep, st)])
    act = np.stack([ea[e, s : s + T] for e, s in zip(ep, st)])
    rew = np.stack([er[e, s : s + T] for e, s in zip(ep, st)])
    state = np.zeros((n_rows, T, 3), np.float32)
    np.savez_compressed(path, obs=obs, action=act, reward=rew, state=state, checkpoint_step=np.zeros(n_rows, np.int64))
    return obs, act, rew, ep


def test_aric_subsample_matches_corr_buffer(tmp_path):
    import gym
    from research.datasets import CorrBuffer

    rng = np.random.RandomState(0)
    obs = rng.randn(50, 2, 8, D).astype(np.float32)
    act = rng.uniform(-0.5, 0.5, (50, 2, 8, A)).astype(np.float32)
    rew = rng.rand(50, 2, 8).astype(np.float32)
    p = str(tmp_path / "c.npz")
    np.savez_compressed(p, obs=obs, action=act, reward=rew)
    space = gym.spaces.Box(-np.inf, np.inf, (D,), np.float32)
    buf = CorrBuffer(space, space, path=p, split="all", subsample_n=20, subsample_seed=3)
    keep = aric_subsample_indices(50, 20, 3)
    assert np.array_equal(buf.obs, obs[keep]) and np.array_equal(buf.reward, rew[keep])


def test_read_npz_rows_streams_selected_rows(tmp_path):
    x = np.random.RandomState(0).randn(1000, 3, 5).astype(np.float32)
    p = str(tmp_path / "x.npz")
    np.savez_compressed(p, x=x)
    rows = np.array([0, 1, 7, 513, 998, 999])
    assert np.array_equal(read_npz_rows(p, "x", rows, chunk_rows=64), x[rows])


def test_pool_index_recovers_episodes(tmp_path):
    p = str(tmp_path / "pool.npz")
    obs, _, _, ep = _make_pool(p)
    pool = PoolIndex(p)
    assert pool.n_episodes == len(np.unique(ep))
    lab = pool.row_episode
    mapping = {}
    for e, l in zip(ep, lab):
        mapping.setdefault(e, set()).add(l)
    # rows of one source episode share one recovered id (windows cover each episode densely here),
    # and different source episodes get different ids
    assert all(len(v) == 1 for v in mapping.values())
    assert len({next(iter(v)) for v in mapping.values()}) == len(mapping)
    assert np.all(pool.lookup_episodes(obs[:, 0]) == lab)
    assert np.all(pool.lookup_states(np.full((2, D), 123.0, np.float32)) == -1)


def test_episode_split_has_no_shared_episode():
    rng = np.random.RandomState(0)
    eps = np.stack([rng.randint(0, 50, 400), rng.randint(0, 50, 400)], 1)  # pref-like: two episodes per act
    sp = make_episode_split(eps, 50, 0.1, 0)
    assert len(sp["val_idx"]) >= 0.1 * 400
    tr_eps = set(eps[sp["train_idx"]].ravel())
    assert not tr_eps & set(sp["val_episodes"].tolist())
    assert len(sp["train_idx"]) + len(sp["val_idx"]) == 400
    assert sp["val_strict"].dtype == bool


def test_tie_aware_target():
    scores = np.array([[1.0, 1.0, 0.0], [2.0, 1.0, 0.0]])
    t, n = tie_aware_target(np.array([0, 0]), scores, 3)
    assert n == 1
    assert np.allclose(t, [[0.5, 0.5, 0.0], [1.0, 0.0, 0.0]])


def test_cum_estop_preferred_arm_is_the_original_on_nostop_rows(tmp_path):
    pp = str(tmp_path / "pool.npz")
    pobs, pact, prew, _ = _make_pool(pp, n_rows=30, T=16)
    rng = np.random.RandomState(5)
    rows_o, rows_a, rows_r, src = [], [], [], []
    for i in range(12):
        orig = (pobs[i], pact[i], prew[i])
        if i % 2 == 0:  # stop row [preferred, other], shared prefix up to tau
            tau = 5 + i % 4
            corr_o = orig[0].copy(); corr_o[tau:] = rng.randn(16 - tau, D)
            corr_a = orig[1].copy(); corr_a[tau:] = rng.uniform(-0.5, 0.5, (16 - tau, A))
            corr_r = orig[2].copy(); corr_r[tau:] = rng.rand(16 - tau)
            pair = [(corr_o, corr_a, corr_r), orig] if i % 4 == 0 else [orig, (corr_o, corr_a, corr_r)]
            src.append(0)
        else:  # no-stop row stored as the generator does: [weak counterfactual, accepted original]
            bad_o = rng.randn(16, D).astype(np.float32); bad_o[0] = orig[0][0]
            pair = [(bad_o, rng.uniform(-0.5, 0.5, (16, A)), rng.rand(16)), orig]
            src.append(1)
        rows_o.append(np.stack([p[0] for p in pair])); rows_a.append(np.stack([p[1] for p in pair]))
        rows_r.append(np.stack([p[2] for p in pair]))
    mp = str(tmp_path / "mix.npz")
    np.savez(mp, obs=np.stack(rows_o).astype(np.float32), action=np.stack(rows_a).astype(np.float32),
             reward=np.stack(rows_r).astype(np.float32), source=np.array(src, np.int32))
    pool = PoolIndex(pp)
    cfg = dict(type="cum_estop", paths={}, rrc=dict(gamma=0.99),
               data=dict(file=mp, subsample_n=None, nostop_preferred="original"))
    cs = load_choice_set(cfg, "t", pool)
    chosen = cs.target.argmax(1)
    assert np.all(chosen[cs.branch == 1] == 1)  # the original, stored at index 1
    assert np.all(chosen[cs.branch == 0] == 0)
    assert cs.meta["nostop_original_at_index0_frac"] == 0.0
    taus = [5 + i % 4 for i in range(0, 12, 2)]
    assert np.array_equal(cs.meta["stop_index"][cs.branch == 0], taus)
    cfg["data"]["nostop_preferred"] = "as_stored"
    assert np.all(load_choice_set(cfg, "t", pool).target.argmax(1) == 0)


def test_credit_rebuilds_base_segment_from_windows(tmp_path):
    pp = str(tmp_path / "pool.npz")
    pobs, pact, prew, _ = _make_pool(pp, n_rows=20, T=16)
    k = 5
    C = 16 - k + 1
    wo = np.stack([pobs[:, i : i + k] for i in range(C)], 1)
    wa = np.stack([pact[:, i : i + k] for i in range(C)], 1)
    adv = np.random.RandomState(0).randn(20, C).astype(np.float32)
    cp = str(tmp_path / "credit.npz")
    np.savez_compressed(cp, obs=wo, action=wa, adv_scores=adv, chosen_idx=adv.argmax(1).astype(np.int32),
                        checkpoint_step=np.zeros(20, np.int64))
    pool = PoolIndex(pp)
    cfg = dict(type="credit", paths={}, rrc=dict(gamma=1.0), data=dict(file=cp, subsample_n=12, subsample_seed=0))
    cs = load_choice_set(cfg, "t", pool)
    rows = aric_subsample_indices(20, 12, 0)
    assert cs.kind == "windows" and cs.window == k and cs.n_options == C
    assert np.array_equal(cs.obs, pobs[rows]) and np.allclose(cs.true_reward, prew[rows])
    assert np.array_equal(cs.target.argmax(1), adv[rows].argmax(1))
