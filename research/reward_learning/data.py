"""
Feedback data -> RRC choice structures, plus the episode-level train/val split.

Rows are loaded through ARIC's own buffer classes (CorrBuffer / DemoBuffer) with
the exact subsample ARIC trained on, so the grounding is ARIC's. Credit
assignment is the one exception: its label file stores 45 overlapping 20-step
windows per row (~12.6 GB decompressed for 100k rows), so only the selected rows
are streamed and the 64-step base segment is rebuilt from them.

None of the label files stores rollout/episode ids, so they are recovered from
pool.npz: rows that share any identical observation belong to the same source
episode (the pool is 200 deterministic episodes cut into overlapping windows).
"""

import copy
import os
import re
import zipfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import gym
import numpy as np
import yaml

OBS_DIM = 35
ACT_DIM = 4

# --------------------------------------------------------------------------------------
# Config loading (deep-merging `import:`; ARIC's BareConfig merges shallowly, which drops
# nested keys -- see the _cpl_all alg_kwargs incident).
# --------------------------------------------------------------------------------------


def _deep_update(base: Dict, new: Dict) -> Dict:
    for k, v in new.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = copy.deepcopy(v)
    return base


def load_yaml_config(path: str) -> Dict:
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    imports = data.pop("import", [])
    imports = [imports] if isinstance(imports, str) else imports
    cfg: Dict = {}
    for p in imports:
        _deep_update(cfg, load_yaml_config(p))
    return _deep_update(cfg, data)


def load_rrc_config(path: str, overlays: Sequence[str] = (), sets: Sequence[str] = ()) -> Dict:
    """Type config + optional overlay files (e.g. configs/rrc_sac/local.yaml) + k.a.b=v overrides."""
    cfg = load_yaml_config(path)
    for o in overlays:
        _deep_update(cfg, load_yaml_config(o))
    for kv in sets:
        key, _, raw = kv.partition("=")
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(raw)
    return cfg


def resolve(template: Optional[str], task: str, cfg: Dict) -> Optional[str]:
    """Fill {task} and {<paths key>} placeholders (repeatedly, so paths may reference each other)."""
    if template is None:
        return None
    values = dict(task=task, **{k: v for k, v in cfg.get("paths", {}).items() if isinstance(v, str)})
    out = template
    for _ in range(5):
        new = re.sub(r"\{(\w+)\}", lambda m: str(values.get(m.group(1), m.group(0))), out)
        if new == out:
            break
        out = new
    return out


# --------------------------------------------------------------------------------------
# Choice structures
# --------------------------------------------------------------------------------------


@dataclass
class ChoiceSet:
    """
    kind == "stacked": option c of act i is the segment obs[i, c, :L] (step_mask marks real steps).
    kind == "windows": option c of act i is the window obs[i, c : c + window] of one base segment.
    """

    type: str
    kind: str
    obs: np.ndarray  # stacked (N,K,L,D) | windows (N,T,D)
    act: np.ndarray
    true_reward: np.ndarray  # stacked (N,K,L) | windows (N,T)
    target: np.ndarray  # (N, n_options) soft target, rows sum to 1
    gamma: float
    episodes: np.ndarray  # (N, M) source episode ids, -1 padded
    row_index: np.ndarray  # (N,) row in the source label file
    act_ids: np.ndarray  # (N,) acts group choice structures (default: one row = one act)
    step_mask: Optional[np.ndarray] = None
    option_mask: Optional[np.ndarray] = None
    window: Optional[int] = None
    adv_scores: Optional[np.ndarray] = None  # stored labeler scores, (N, n_options)
    branch: Optional[np.ndarray] = None  # cum_estop: 0 = stop+correction, 1 = no-stop
    pool_rows: Optional[np.ndarray] = None  # (N,) pool row for credit/scalar
    meta: Dict = field(default_factory=dict)

    def __len__(self) -> int:
        return self.obs.shape[0]

    @property
    def n_options(self) -> int:
        return self.target.shape[1]

    def subset(self, idx: np.ndarray) -> "ChoiceSet":
        out = copy.copy(self)
        for name in ("obs", "act", "true_reward", "target", "episodes", "row_index", "act_ids", "step_mask",
                     "option_mask", "adv_scores", "branch", "pool_rows"):
            v = getattr(self, name)
            if v is not None:
                setattr(out, name, v[idx])
        out.meta = dict(self.meta)
        return out

    def option_true_returns(self) -> np.ndarray:
        """(N, n_options) true discounted option returns under ARIC's grounding."""
        if self.kind == "windows":
            k, T = self.window, self.true_reward.shape[1]
            w = self.gamma ** np.arange(k)
            return np.stack([(self.true_reward[:, i : i + k] * w).sum(1) for i in range(T - k + 1)], axis=1)
        r = self.true_reward if self.step_mask is None else self.true_reward * self.step_mask
        return (r * self.gamma ** np.arange(r.shape[-1])).sum(-1)

    def valid_steps(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """All real (s, a, r_true) steps, flattened -- used for standardization and reward-scale stats."""
        if self.kind == "windows":
            return self.obs.reshape(-1, OBS_DIM), self.act.reshape(-1, ACT_DIM), self.true_reward.reshape(-1)
        m = np.ones(self.true_reward.shape, bool) if self.step_mask is None else self.step_mask.astype(bool)
        if self.option_mask is not None:
            m = m & self.option_mask[..., None].astype(bool)
        return self.obs[m], self.act[m], self.true_reward[m]


def one_hot(n: int, k: int, idx: np.ndarray) -> np.ndarray:
    t = np.zeros((n, k), np.float32)
    t[np.arange(n), idx] = 1.0
    return t


def tie_aware_target(chosen: np.ndarray, scores: Optional[np.ndarray], k: int) -> Tuple[np.ndarray, int]:
    """One-hot on chosen; uniform over options tied with the chosen one's labeler score, if any."""
    n = len(chosen)
    t = one_hot(n, k, chosen)
    if scores is None:
        return t, 0
    best = scores[np.arange(n), chosen][:, None]
    tied = np.isclose(scores, best, rtol=0.0, atol=1e-7)
    n_ties = int((tied.sum(1) > 1).sum())
    if n_ties:
        rows = tied.sum(1) > 1
        t[rows] = tied[rows] / tied[rows].sum(1, keepdims=True)
    return t, n_ties


def aric_subsample_indices(n_total: int, subsample_n: Optional[int], subsample_seed: int) -> np.ndarray:
    """The exact rows CorrBuffer / DemoBuffer / PMCreditAssignmentBuffer keep for split='all'."""
    if subsample_n is None:
        return np.arange(n_total)
    assert subsample_n <= n_total, f"subsample_n={subsample_n} exceeds available {n_total} rows"
    return np.sort(np.random.RandomState(subsample_seed).permutation(n_total)[:subsample_n])


def npz_shape(path: str, key: str) -> Tuple[int, ...]:
    with zipfile.ZipFile(path) as z, z.open(key + ".npy") as f:
        version = np.lib.format.read_magic(f)
        shape, _, _ = np.lib.format._read_array_header(f, version)
    return shape


def read_npz_rows(path: str, key: str, rows: np.ndarray, chunk_rows: int = 2048) -> np.ndarray:
    """Stream one array of an .npz and keep only `rows` (sorted), without materializing the array."""
    rows = np.asarray(rows)
    assert np.all(np.diff(rows) > 0), "rows must be sorted and unique"
    with zipfile.ZipFile(path) as z, z.open(key + ".npy") as f:
        version = np.lib.format.read_magic(f)
        shape, fortran, dtype = np.lib.format._read_array_header(f, version)
        assert not fortran, "Fortran-ordered arrays are not supported"
        row_shape = shape[1:]
        row_items = int(np.prod(row_shape)) if row_shape else 1
        row_bytes = row_items * dtype.itemsize
        out = np.empty((len(rows),) + tuple(row_shape), dtype=dtype)
        j = 0
        for start in range(0, shape[0], chunk_rows):
            if j >= len(rows):
                break
            n = min(chunk_rows, shape[0] - start)
            buf = f.read(n * row_bytes)
            assert len(buf) == n * row_bytes, f"short read in {path}:{key}"
            j_end = np.searchsorted(rows, start + n)
            if j_end > j:
                block = np.frombuffer(buf, dtype=dtype).reshape((n,) + tuple(row_shape))
                out[j:j_end] = block[rows[j:j_end] - start]
                j = j_end
    return out


# --------------------------------------------------------------------------------------
# Pool: state table and episode recovery
# --------------------------------------------------------------------------------------

_HASH_W = np.random.RandomState(12345).randint(1, 2**62, size=OBS_DIM, dtype=np.int64).astype(np.uint64) | np.uint64(1)


def hash_obs(obs: np.ndarray, chunk: int = 1 << 18) -> np.ndarray:
    """64-bit hash of each exact float32 observation vector. obs: (M, D) -> (M,) uint64."""
    obs = np.ascontiguousarray(obs, dtype=np.float32)
    words = obs.view(np.uint32)
    out = np.empty(len(obs), np.uint64)
    with np.errstate(over="ignore"):
        for i in range(0, len(obs), chunk):
            w = words[i : i + chunk].astype(np.uint64)
            out[i : i + chunk] = (w * _HASH_W[: w.shape[1]]).sum(axis=1)
    return out


class PoolIndex:
    """Maps observation vectors to the pool's source episodes."""

    def __init__(self, pool_path: str, expected_episodes: Optional[int] = None, load_state: bool = False):
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components

        self.path = pool_path
        with np.load(pool_path) as pool:
            self.obs = pool["obs"].astype(np.float32)  # (R, T, D)
            self.action = pool["action"].astype(np.float32)
            self.reward = pool["reward"].astype(np.float32)
            self.state = pool["state"] if load_state else None
        R, T, D = self.obs.shape
        assert D == OBS_DIM, f"pool obs dim {D} != {OBS_DIM}"
        h = hash_obs(self.obs.reshape(-1, D))
        self.hashes, state_of_step = np.unique(h, return_inverse=True)
        S = len(self.hashes)
        self.row_states = state_of_step.reshape(R, T)
        rows = np.repeat(np.arange(R), T)
        graph = coo_matrix((np.ones(R * T, np.int8), (rows, R + state_of_step)), shape=(R + S, R + S))
        n_comp, labels = connected_components(graph, directed=False)
        row_lab = labels[:R]
        uniq, self.row_episode = np.unique(row_lab, return_inverse=True)
        remap = {lab: i for i, lab in enumerate(uniq)}
        self.state_episode = np.array([remap[l] for l in labels[R:]], dtype=np.int64)
        self.n_episodes = len(uniq)
        self.n_states = S
        if expected_episodes is not None:
            assert self.n_episodes == expected_episodes, (
                f"recovered {self.n_episodes} source episodes from {pool_path}, expected {expected_episodes}"
            )

    def lookup_states(self, obs: np.ndarray) -> np.ndarray:
        """(M, D) observations -> (M,) pool state ids, -1 where not a pool state."""
        h = hash_obs(obs.reshape(-1, OBS_DIM))
        pos = np.searchsorted(self.hashes, h)
        pos = np.clip(pos, 0, len(self.hashes) - 1)
        found = self.hashes[pos] == h
        return np.where(found, pos, -1).reshape(obs.shape[:-1])

    def lookup_episodes(self, obs: np.ndarray) -> np.ndarray:
        s = self.lookup_states(obs)
        return np.where(s >= 0, self.state_episode[np.maximum(s, 0)], -1)


def _episodes_from_first_obs(pool: PoolIndex, obs: np.ndarray) -> np.ndarray:
    """obs (N, K, L, D): episode of each arm's first observation -> (N, M) unique ids, -1 padded."""
    ep = pool.lookup_episodes(obs[:, :, 0])  # (N, K)
    N = ep.shape[0]
    out = np.full((N, ep.shape[1]), -1, np.int64)
    for i in range(N):
        u = np.unique(ep[i][ep[i] >= 0])
        out[i, : len(u)] = u
    width = max(1, int((out >= 0).sum(1).max()))
    out = out[:, :width]
    unmatched = int((out[:, 0] < 0).sum())
    assert unmatched == 0, f"{unmatched}/{N} acts have no arm whose first obs is a pool state"
    return out


# --------------------------------------------------------------------------------------
# Per-type loaders
# --------------------------------------------------------------------------------------


def _spaces():
    obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32)
    act_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(ACT_DIM,), dtype=np.float32)
    return obs_space, act_space


def _load_aric_stacked(buffer_name: str, path: str, subsample_n, subsample_seed, action_eps):
    from research.datasets import CorrBuffer, DemoBuffer

    buf_cls = dict(CorrBuffer=CorrBuffer, DemoBuffer=DemoBuffer)[buffer_name]
    obs_space, act_space = _spaces()
    n_total = npz_shape(path, "obs")[0]
    if subsample_n is not None and subsample_n >= n_total:
        subsample_n = None  # the whole file (identical to a full sorted permutation)
    buf = buf_cls(
        obs_space, act_space, path=path, batch_size=1, split="all",
        subsample_n=subsample_n, subsample_seed=subsample_seed, action_eps=action_eps,
    )
    rows = aric_subsample_indices(n_total, subsample_n, subsample_seed)
    assert len(rows) == len(buf), (len(rows), len(buf))
    return buf.obs, buf.action, buf.reward, rows


def _small_arrays(path: str, keys: Sequence[str], rows: np.ndarray) -> Dict[str, np.ndarray]:
    out = {}
    with np.load(path) as f:
        for k in keys:
            if k in f.files:
                out[k] = f[k][rows]
    return out


def load_choice_set(cfg: Dict, task: str, pool: PoolIndex) -> ChoiceSet:
    t = cfg["type"]
    d = cfg["data"]
    path = resolve(d["file"], task, cfg)
    gamma = float(cfg["rrc"]["gamma"])
    eps = float(d.get("action_eps", 1e-5))
    sub_n, sub_seed = d.get("subsample_n"), int(d.get("subsample_seed", 0))
    meta = dict(path=path, subsample_n=sub_n, subsample_seed=sub_seed)

    if t in ("pref", "corr", "demo", "scalar", "cum_estop"):
        buffer_name = "DemoBuffer" if t == "demo" else "CorrBuffer"
        obs, act, rew, rows = _load_aric_stacked(buffer_name, path, sub_n, sub_seed, eps)
        N, K, L, D = obs.shape
        assert D == OBS_DIM and act.shape[-1] == ACT_DIM, (obs.shape, act.shape)
        extra = _small_arrays(path, ["adv_scores", "traj_id", "start_idx", "source"], rows)
        adv = extra.get("adv_scores")
        chosen = np.zeros(N, np.int64)  # ARIC: preferred / demo arm pinned at index 0
        episodes = _episodes_from_first_obs(pool, obs)
        cs = ChoiceSet(
            type=t, kind="stacked", obs=obs, act=act, true_reward=rew, target=None, gamma=gamma,
            episodes=episodes, row_index=rows, act_ids=np.arange(N), adv_scores=adv, meta=meta,
        )
        if t == "scalar":
            cs.pool_rows = extra["traj_id"].astype(np.int64)
            starts = extra["start_idx"]
            ok = np.array([np.array_equal(obs[i, k, 0], pool.obs[cs.pool_rows[i], starts[i, k]])
                           for i in range(min(N, 500)) for k in range(K)])
            assert ok.all(), "scalar traj_id/start_idx do not point at the stored subsegments in this pool"
            cs.episodes = pool.row_episode[cs.pool_rows][:, None]
            if d.get("act_group", "row") == "traj_id":
                cs.act_ids = cs.pool_rows.copy()
        if t == "cum_estop":
            assert "source" in extra, "cum_estop mix file has no `source` (branch) array"
            cs.branch = extra["source"].astype(np.int64)
            in_pool = pool.lookup_states(obs) >= 0  # (N, K, L)
            original = in_pool.all(axis=2)  # (N, K): arm is a full pool trajectory
            diverge = np.argmax(np.any(obs[:, 0] != obs[:, 1], axis=-1) | np.any(act[:, 0] != act[:, 1], axis=-1), axis=1)
            cs.meta["stop_index"] = diverge.astype(np.int64)
            nostop = cs.branch == 1
            n_orig = original.sum(1)
            assert np.all(n_orig[nostop] == 1), "no-stop rows must have exactly one full-pool (original) arm"
            orig_idx = np.argmax(original, axis=1)
            cs.meta["nostop_original_at_index0_frac"] = float((orig_idx[nostop] == 0).mean()) if nostop.any() else None
            cs.meta["stop_original_preferred_frac"] = (
                float((orig_idx[~nostop] == 0).mean()) if (~nostop).any() else None
            )
            mode = d.get("nostop_preferred", "original")
            assert mode in ("original", "as_stored"), mode
            if mode == "original":
                chosen = np.where(nostop, orig_idx, 0)
            cs.meta["nostop_preferred"] = mode
        cs.target, n_ties = tie_aware_target(chosen, None, K)
        cs.meta["n_ties"] = n_ties
        return cs

    if t == "credit":
        n_total = npz_shape(path, "obs")[0]
        if sub_n is not None and sub_n >= n_total:
            sub_n = None
        rows = aric_subsample_indices(n_total, sub_n, sub_seed)
        win_obs = read_npz_rows(path, "obs", rows).astype(np.float32)  # (N, C, k, D)
        win_act = read_npz_rows(path, "action", rows).astype(np.float32)
        N, C, k, D = win_obs.shape
        T = C + k - 1
        base_obs = np.concatenate([win_obs[:, :, 0], win_obs[:, -1, 1:]], axis=1)  # (N, T, D)
        base_act = np.concatenate([win_act[:, :, 0], win_act[:, -1, 1:]], axis=1)
        for i in range(C):  # every window must be a slice of the rebuilt base segment
            assert np.array_equal(win_obs[:, i], base_obs[:, i : i + k]), f"credit window {i} inconsistent"
        del win_obs, win_act
        extra = _small_arrays(path, ["chosen_idx", "adv_scores"], rows)
        assert np.array_equal(base_obs, pool.obs[rows]), (
            "credit row n is not pool row n -- credit labels were not generated from this pool"
        )
        base_act = np.clip(base_act, -(1 - eps), 1 - eps)
        chosen = extra["chosen_idx"].astype(np.int64)
        target, n_ties = tie_aware_target(chosen, extra["adv_scores"], C)
        meta.update(n_ties=n_ties)
        return ChoiceSet(
            type=t, kind="windows", obs=base_obs, act=base_act, true_reward=pool.reward[rows].copy(),
            target=target, gamma=gamma, episodes=pool.row_episode[rows][:, None], row_index=rows,
            act_ids=np.arange(N), window=k, adv_scores=extra["adv_scores"], pool_rows=rows, meta=meta,
        )

    raise ValueError(f"unknown feedback type {t!r}")


# --------------------------------------------------------------------------------------
# Episode-level split
# --------------------------------------------------------------------------------------


def make_episode_split(episodes: np.ndarray, n_episodes: int, val_frac: float = 0.1, seed: int = 0) -> Dict:
    """
    Draw a held-out episode set E_val (episodes added in a seeded random order until >= val_frac
    of acts touch E_val). An act is in val if ANY of its source episodes is in E_val, so no
    source rollout appears in both splits.
    """
    order = np.random.RandomState(seed).permutation(n_episodes)
    N = episodes.shape[0]
    valid = episodes >= 0
    touched = np.zeros(N, bool)
    chosen = []
    for e in order:
        if touched.mean() >= val_frac:
            break
        chosen.append(e)
        touched |= np.any((episodes == e) & valid, axis=1)
    val_eps = np.array(sorted(chosen), dtype=np.int64)
    in_val = np.isin(episodes, val_eps) & valid
    val = np.any(in_val, axis=1)
    strict = val & np.all(in_val | ~valid, axis=1)
    train_idx, val_idx = np.nonzero(~val)[0], np.nonzero(val)[0]
    train_eps = np.unique(episodes[train_idx][valid[train_idx]])
    assert not np.isin(train_eps, val_eps).any()
    return dict(train_idx=train_idx, val_idx=val_idx, val_episodes=val_eps, val_strict=strict[val_idx])
