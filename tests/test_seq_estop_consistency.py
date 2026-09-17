"""
Regression + statistical-significance tests for the seq-estop
advantage-consistency bug.

Both MetaWorld's build_pairs() (scripts/generate_seq_estop_labels.py) and
PointMass's build_seq_pairs() (scripts/generate_pm_feedback.py) assign the
"preferred" segment purely by position relative to the simulated stop time
tau (t == tau -> stop_seg preferred, t < tau -> cont_seg preferred). Neither
ever compares the two candidate segments' actual return/advantage -- unlike
generate_pref_labels.py, which explicitly scores both candidates and asserts
the higher-scoring one is preferred. See scripts/toy_seq_estop_bug_demo.py
for a standalone walkthrough.

Part A (always runs, no CHPC/oracle needed): a deterministic toy trajectory
with known per-step "oracle quality" reproduces the exact failure mode in
BOTH environments' pair-builder functions. This pins the bug as a regression
test -- if generate_seq_estop_labels.py / generate_pm_feedback.py are fixed
to enforce advantage-consistency, this test's expectations must change too.

Part B (CHPC only; auto-skipped elsewhere): re-scores a random subsample of
a REAL generated seq_estop_labels.npz with the real oracle used to build it
(reusing scripts/analyze_seq_estop_advantage.py's scoring code) and runs a
two-sided binomial test against the null hypothesis that "preferred" carries
no information (consistency rate = 50%). With tens of thousands of pairs,
sampling noise cannot explain a mean this far from 50% -- so this turns "the
numbers looked bad" into a hard, statistically-grounded pass/fail.

Run everywhere:  pytest tests/test_seq_estop_consistency.py -v
On CHPC, point at a real label file / oracle dir if the defaults don't match:
  SEQ_ESTOP_LABELS_PATH=/scratch/.../seq_estop_labels.npz \\
  SEQ_ESTOP_ORACLE_DIR=runs/runs/chpc/oracle_sac_seeds/<env>/seed-1 \\
  pytest tests/test_seq_estop_consistency.py -v
"""

import os

import numpy as np
import pytest
from scipy import stats

from scripts.generate_pm_feedback import build_seq_pairs as pm_build_pairs
from scripts.generate_seq_estop_labels import build_pairs as mw_build_pairs

# ---------------------------------------------------------------------------
# Part A: deterministic toy trajectory, no oracle/CHPC needed
# ---------------------------------------------------------------------------

# quality[t] embedded directly as the 1-D "obs" value, so summing a segment's
# obs IS its total oracle quality -- no model/forward-pass needed. Matches
# the trajectory that motivated this investigation: good -> okay -> degrading
# -> bad, with a human e-stop firing at tau=6.
TOY_QUALITY = np.array([0.9, 0.8, 0.7, 0.5, 0.2, -0.2, -0.8, -1.0, -1.1, -1.1], dtype=np.float32)
TOY_H = 3
TOY_TAU = 6


def _score_pairs(pairs):
    """[(timestep, stop_event, is_quality_consistent), ...] for build_pairs() output."""
    out = []
    for pair in pairs:
        p_score = float(pair["obs"][0].sum())
        n_score = float(pair["obs"][1].sum())
        out.append((pair["timestep"], pair["stop_event"], p_score > n_score))
    return out


@pytest.mark.parametrize("build_pairs_fn,label", [
    (mw_build_pairs, "MetaWorld"),
    (pm_build_pairs, "PointMass"),
])
def test_positional_label_ignores_actual_quality(build_pairs_fn, label):
    """
    On a trajectory that declines monotonically toward its stop time, every
    continue-pair (t < tau) compares a backward-looking window (recent, still
    okay) against a forward-looking window (further into the decline) and
    labels the WORSE one preferred -- guaranteed by construction, not chance.
    The single stop-pair (t == tau) is the one case where the comparison
    direction happens to line up. This is the exact mechanism behind the
    ~38% / 62% split measured on real CHPC-generated data.
    """
    T = len(TOY_QUALITY)
    traj_obs = TOY_QUALITY.reshape(T, 1)
    traj_action = np.zeros((T, 1), dtype=np.float32)
    traj_reward = np.zeros(T, dtype=np.float32)

    pairs = build_pairs_fn(traj_obs, traj_action, traj_reward, TOY_TAU, TOY_H)
    results = _score_pairs(pairs)

    assert len(results) == 5, f"{label}: expected 5 decision points (t=2..6), got {len(results)}"

    continue_results = [r for r in results if r[1] == 0.0]
    stop_results     = [r for r in results if r[1] == 1.0]

    assert len(stop_results) == 1 and stop_results[0][2] is True, (
        f"{label}: the single stop-pair (t=tau) should be quality-consistent "
        f"on this monotonically-declining toy trajectory"
    )
    assert len(continue_results) == 4 and all(not r[2] for r in continue_results), (
        f"{label}: every continue-pair (t<tau) should be quality-INCONSISTENT "
        f"on this monotonically-declining toy trajectory -- got {continue_results}"
    )

    n_consistent = sum(r[2] for r in results)
    assert n_consistent / len(results) == pytest.approx(0.2), (
        f"{label}: expected exactly 1/5 = 20% overall consistency, "
        f"got {n_consistent}/{len(results)}"
    )


def test_mw_and_pm_pair_builders_are_the_same_algorithm():
    """
    Documents that MetaWorld's build_pairs and PointMass's build_seq_pairs
    are the same positional-assignment algorithm (same bug, not an MW-only
    quirk): given identical inputs they produce identical preferred/
    non-preferred segments and stop_event/timestep bookkeeping.
    """
    T = len(TOY_QUALITY)
    traj_obs = TOY_QUALITY.reshape(T, 1)
    traj_action = np.zeros((T, 1), dtype=np.float32)
    traj_reward = TOY_QUALITY.copy()  # reuse as a stand-in reward trace too

    mw_pairs = mw_build_pairs(traj_obs, traj_action, traj_reward, TOY_TAU, TOY_H)
    pm_pairs = pm_build_pairs(traj_obs, traj_action, traj_reward, TOY_TAU, TOY_H)

    assert len(mw_pairs) == len(pm_pairs)
    for a, b in zip(mw_pairs, pm_pairs):
        assert a["timestep"] == b["timestep"]
        assert a["stop_event"] == b["stop_event"]
        np.testing.assert_array_equal(a["obs"], b["obs"])
        np.testing.assert_array_equal(a["reward"], b["reward"])


# ---------------------------------------------------------------------------
# Part B: real generated data + real oracle (CHPC only)
# ---------------------------------------------------------------------------

DEFAULT_LABELS_PATH = os.environ.get(
    "SEQ_ESTOP_LABELS_PATH",
    "/scratch/general/vast/u1472210/mw_de_labels/mw_drawer-open-v2/seq_estop_labels.npz",
)
DEFAULT_ORACLE_DIR = os.environ.get(
    "SEQ_ESTOP_ORACLE_DIR",
    "runs/runs/chpc/oracle_sac_seeds/mw_drawer-open-v2/seed-1",
)

_LABELS_AVAILABLE = os.path.exists(DEFAULT_LABELS_PATH)
_CHPC_DATA_AVAILABLE = _LABELS_AVAILABLE and os.path.isdir(DEFAULT_ORACLE_DIR)


def _reconstruct_trace(timestep, stop_event, h):
    """
    Reconstruct the (stop_indices, cont_indices, label) trace for one
    trajectory's pairs, sorted by timestep. Mirrors build_pairs()'s slicing
    formula exactly:  stop_seg = obs[t-h+1 : t+1]  ->  indices [t-h+1 .. t]
                       cont_seg = obs[t     : t+h]  ->  indices [t     .. t+h-1]
    Only timestep/stop_event/h are needed -- the index ranges don't depend
    on the segment content, only on t and h.
    """
    order = np.argsort(timestep)
    ts = timestep[order]
    se = stop_event[order]
    stop_rows_t = ts[se == 1]
    tau = int(stop_rows_t[0]) if len(stop_rows_t) else None

    rows = []
    for t_raw, s in zip(ts, se):
        t = int(t_raw)
        rows.append({
            "t": t,
            "stop_indices": (t - h + 1, t),
            "cont_indices": (t, t + h - 1),
            "label": "STOP" if s == 1.0 else "CONTINUE",
        })
    return tau, rows


def _format_trace(env_name, traj_idx, tau, rows):
    lines = [f"trajectory {traj_idx}  (env={env_name})", f"tau = {tau}", ""]
    for r in rows:
        lo_s, hi_s = r["stop_indices"]
        lo_c, hi_c = r["cont_indices"]
        lines.append(f"t={r['t']}")
        lines.append(f"stop_seg = [{lo_s} ... {hi_s}]")
        lines.append(f"cont_seg = [{lo_c} ... {hi_c}]")
        lines.append(f"label = {r['label']}")
        lines.append("")
    return "\n".join(lines)


@pytest.mark.skipif(
    not _LABELS_AVAILABLE,
    reason=(
        "Needs a real seq_estop_labels.npz (CHPC only; no oracle needed for "
        f"this one). Looked for LABELS_PATH={DEFAULT_LABELS_PATH!r} -- override "
        "with SEQ_ESTOP_LABELS_PATH."
    ),
)
def test_real_pair_trace_matches_expected_pattern():
    """
    For a handful of REAL stopped trajectories from the generated label file,
    reconstructs and saves the exact per-t (tau, stop_indices, cont_indices,
    label) trace -- the concrete "tau=35 / t=30 stop_seg=[21..30]
    cont_seg=[30..39] label=CONTINUE / ... / t=35 ... label=STOP" pattern --
    and asserts it matches the structural claim the labeling scheme makes:
    CONTINUE for every t < tau, STOP exactly once at t == tau.

    Does NOT need the oracle (index bookkeeping only, no scoring), so it's
    cheap and runs even before/without the statistical test above. Saves the
    full trace to results/seq_estop_advantage/<env>/pair_trace.txt on CHPC.
    """
    with open(DEFAULT_LABELS_PATH, "rb") as f:
        d = np.load(f)
        timestep   = d["timestep"]
        stop_event = d["stop_event"]
        traj_idx   = d["traj_idx"]

    h = int(os.environ.get("SEQ_ESTOP_TRACE_H", "10"))          # matches production default
    n_traj = int(os.environ.get("SEQ_ESTOP_TRACE_N_TRAJ", "5"))

    stopped_traj_ids = np.unique(traj_idx[stop_event == 1])
    assert len(stopped_traj_ids) > 0, "No stopped trajectories found in the label file"
    chosen = stopped_traj_ids[:n_traj]

    env_name = os.path.basename(os.path.dirname(DEFAULT_LABELS_PATH)) or "unknown_env"
    out_dir = os.path.join("results", "seq_estop_advantage", env_name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "pair_trace.txt")

    n_overlap_total = 0
    with open(out_path, "w") as fh:
        for tid in chosen:
            mask = traj_idx == tid
            tau, rows = _reconstruct_trace(timestep[mask], stop_event[mask], h)

            # This IS the bug, made explicit and checked against real data:
            assert tau is not None, f"traj {tid}: expected a stop row, found none"
            stop_rows = [r for r in rows if r["label"] == "STOP"]
            cont_rows = [r for r in rows if r["label"] == "CONTINUE"]
            assert len(stop_rows) == 1 and stop_rows[0]["t"] == tau, (
                f"traj {tid}: expected exactly one STOP row at t=tau={tau}, "
                f"got {stop_rows}"
            )
            assert all(r["t"] < tau for r in cont_rows), (
                f"traj {tid}: found a CONTINUE row with t >= tau={tau}"
            )

            # How many CONTINUE-labeled ("preferred") forward windows literally
            # reach into the STOP window itself -- i.e. contain the very
            # behavior that triggered the human's e-stop.
            overlapping = [r for r in cont_rows if r["cont_indices"][1] >= tau]
            n_overlap_total += len(overlapping)

            trace_text = _format_trace(env_name, int(tid), tau, rows)
            print("\n" + trace_text)
            if overlapping:
                print(f"  -> {len(overlapping)}/{len(cont_rows)} CONTINUE-labeled "
                      f"cont_seg windows reach t>=tau (include the stop-triggering step).")

            fh.write(trace_text + "\n")

    print(f"\nSaved pair trace for {len(chosen)} trajectories -> {out_path}")
    print(f"Total CONTINUE pairs whose 'preferred' window reaches the stop step: "
          f"{n_overlap_total}")


@pytest.mark.skipif(
    not _CHPC_DATA_AVAILABLE,
    reason=(
        "Needs a real seq_estop_labels.npz + oracle checkpoint (CHPC only). "
        f"Looked for LABELS_PATH={DEFAULT_LABELS_PATH!r} "
        f"ORACLE_DIR={DEFAULT_ORACLE_DIR!r} -- override with "
        "SEQ_ESTOP_LABELS_PATH / SEQ_ESTOP_ORACLE_DIR env vars."
    ),
)
def test_real_seq_estop_continue_pairs_are_significantly_advantage_inconsistent():
    """
    Re-scores a random subsample of pairs from the real generated label file
    with the same oracle used to build it (same scoring code as
    scripts/analyze_seq_estop_advantage.py), then runs a two-sided binomial
    test of the continue-pair consistency rate against the null p=0.5.

    Fails loudly (rather than silently passing) if the observed rate is NOT
    significantly below 50%, since that would mean either the pipeline
    changed since this diagnosis or something is wrong with this test itself
    -- either way worth knowing.
    """
    from scripts.analyze_seq_estop_advantage import compute_signed_advantage, load_model

    rng = np.random.default_rng(0)
    max_pairs = int(os.environ.get("SEQ_ESTOP_TEST_MAX_PAIRS", "5000"))

    with open(DEFAULT_LABELS_PATH, "rb") as f:
        d = np.load(f)
        obs, reward, stop_event = d["obs"], d["reward"], d["stop_event"]

    M = obs.shape[0]
    if M > max_pairs:
        keep = np.sort(rng.choice(M, size=max_pairs, replace=False))
        obs, reward, stop_event = obs[keep], reward[keep], stop_event[keep]
        M = max_pairs

    oracle_ckpt = os.path.join(DEFAULT_ORACLE_DIR, "best_model.pt")
    oracle = load_model(DEFAULT_ORACLE_DIR, oracle_ckpt, "cpu")

    _, K, h, obs_dim = obs.shape
    assert K == 2
    obs_flat = obs.reshape(M * K, h, obs_dim)
    reward_flat = reward.reshape(M * K, h)
    rl_sum = compute_signed_advantage(
        obs_flat, reward_flat, oracle, mcmc_samples=32, discount=0.99, device="cpu",
    )
    segment_score = rl_sum.sum(axis=1).reshape(M, K)
    win = segment_score[:, 0] > segment_score[:, 1]

    cont_mask = stop_event == 0
    n_cont = int(cont_mask.sum())
    k_cont = int(win[cont_mask].sum())
    rate = k_cont / n_cont

    result = stats.binomtest(k_cont, n_cont, p=0.5, alternative="two-sided")
    print(
        f"\ncontinue-pair advantage-consistency: {rate:.1%} (k={k_cont}, n={n_cont}), "
        f"binomial p-value vs 50% = {result.pvalue:.3e}"
    )

    assert result.pvalue < 1e-6, (
        f"Expected continue-pair consistency to be significantly different from 50% "
        f"(that IS the bug) -- got p={result.pvalue:.3e} at rate={rate:.1%}. Either the "
        f"generation pipeline changed, or increase SEQ_ESTOP_TEST_MAX_PAIRS and retry."
    )
    assert rate < 0.5, (
        f"Continue-pair consistency ({rate:.1%}) is significantly different from 50% but "
        f"in the OPPOSITE direction from the known bug -- re-open the investigation "
        f"before trusting this diagnosis."
    )
