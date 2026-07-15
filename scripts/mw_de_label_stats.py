"""
Compute and save detailed statistics for all MW DE label files.

For each env × feedback type, reports:
  - N examples/pairs, shape info, file size on disk
  - Reward stats (mean/std/min/max) for best and worst candidate
  - Advantage gap stats (signal strength: larger = clearer label)
  - Unique checkpoint steps covered (proxy for trajectory diversity)
  - Estimated unique pool segments contributing to the dataset
  - Budget coverage: how many examples support each DE budget level
  - Type-specific stats (improvement for corr, K for demo, etc.)

Unique pool segment estimation:
  corr / demo / credit  : N per env (one label per pool segment)
  seq_estop             : len(unique(traj_idx)) if traj_idx stored
  pref / scalar         : unique checkpoint steps × (N / unique_ckpts)
                          (approximation; segments re-sampled from pool)

Output:
  --output-dir/mw_de_label_stats.txt   human-readable report
  --output-dir/mw_de_label_stats.csv   machine-readable, one row per env×type

Usage:
  python scripts/mw_de_label_stats.py \\
      --labels-dir /scratch/general/vast/u1472210/mw_de_labels \\
      --output-dir results/
"""

import argparse
import csv
import os

import numpy as np

ENVS = [
    "mw_drawer-open-v2",
    "mw_bin-picking-v2",
    "mw_button-press-v2",
    "mw_door-open-v2",
    "mw_plate-slide-v2",
    "mw_sweep-into-v2",
]

LABEL_FILES = {
    "pref"             : "pref_labels.npz",
    "corr"             : "corr_labels.npz",
    "demo"             : "demo_labels_K7.npz",
    "seq_estop"        : "seq_estop_labels.npz",
    "scalar"           : "scalar_labels.npz",
    "credit_assignment": "credit_labels.npz",
}

BUDGETS = [50, 100, 300, 600, 1000, 2000, 5000, 10000, 20000, 40000, 100000]

GAMMA = 0.99


def stat(arr, prefix):
    return {
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_std" : float(np.std(arr)),
        f"{prefix}_min" : float(np.min(arr)),
        f"{prefix}_max" : float(np.max(arr)),
        f"{prefix}_p25" : float(np.percentile(arr, 25)),
        f"{prefix}_p50" : float(np.percentile(arr, 50)),
        f"{prefix}_p75" : float(np.percentile(arr, 75)),
    }


def discounted_sum(reward, gamma):
    """reward: (N, T) → (N,) discounted return."""
    T = reward.shape[1]
    disc = gamma ** np.arange(T)
    return (reward * disc[None, :]).sum(axis=1)


def budget_coverage(N):
    """For each DE budget, report N_available (min(N, budget)) and whether it's clamped."""
    return {f"budget_{b}": min(N, b) for b in BUDGETS}


def file_size_mb(path):
    return os.path.getsize(path) / (1024 ** 2)


# ---------------------------------------------------------------------------
# Per-type stats
# ---------------------------------------------------------------------------

def stats_pref_or_corr(d, ftype):
    """pref_labels.npz and corr_labels.npz: obs (N,2,T,D), adv_scores (N,2)."""
    obs        = d["obs"]          # (N, 2, T, D)
    reward     = d["reward"]       # (N, 2, T)
    adv_scores = d["adv_scores"]   # (N, 2)
    ckpt       = d["checkpoint_step"]  # (N,)

    N, K, T, obs_dim = obs.shape
    act_dim = d["action"].shape[-1]

    # Segment rewards: best (pos 0) and worst (pos 1)
    rew_best  = reward[:, 0, :].sum(axis=1)
    rew_worst = reward[:, 1, :].sum(axis=1)
    disc_best = discounted_sum(reward[:, 0, :], GAMMA)

    gap = adv_scores[:, 0] - adv_scores[:, 1]

    # Unique pool segments: approximation (segments re-sampled from pool for pref/scalar)
    unique_ckpts = np.unique(ckpt)
    n_unique_ckpts = len(unique_ckpts)
    # Each pair uses 2 segments → upper bound on unique segments = 2N
    # Lower bound via checkpoint diversity
    approx_unique_segs = min(2 * N, N)  # conservative: N distinct pool segs at most

    row = {
        "N"                 : N,
        "K"                 : K,
        "T"                 : T,
        "obs_dim"           : obs_dim,
        "act_dim"           : act_dim,
        "n_unique_ckpts"    : n_unique_ckpts,
        "ckpt_min"          : int(unique_ckpts.min()),
        "ckpt_max"          : int(unique_ckpts.max()),
        "approx_unique_segs": approx_unique_segs,
        "pct_gap_positive"  : float((gap > 0).mean()),
    }
    row.update(stat(rew_best,  "rew_best"))
    row.update(stat(rew_worst, "rew_worst"))
    row.update(stat(disc_best, "disc_best"))
    row.update(stat(gap,       "adv_gap"))

    if ftype == "corr" and "improvement" in d:
        row.update(stat(d["improvement"], "improvement"))

    row.update(budget_coverage(N))
    return row


def stats_demo(d):
    """demo_labels_K7.npz: obs (N,K,T,D), adv_scores (N,K)."""
    obs        = d["obs"]          # (N, K, T, D)
    reward     = d["reward"]       # (N, K, T)
    adv_scores = d["adv_scores"]   # (N, K)
    ckpt       = d["checkpoint_step"]

    N, K, T, obs_dim = obs.shape
    act_dim = d["action"].shape[-1]

    rew_demo  = reward[:, 0, :].sum(axis=1)   # best (demo) position
    rew_worst = reward[:, -1, :].sum(axis=1)  # worst position
    disc_demo = discounted_sum(reward[:, 0, :], GAMMA)

    gap = adv_scores[:, 0] - adv_scores[:, -1]

    unique_ckpts = np.unique(ckpt)

    # Per-position mean adv_score (quality gradient)
    pos_adv_means = {f"adv_pos{k}_mean": float(adv_scores[:, k].mean()) for k in range(K)}

    # Action diversity: mean pairwise L2 between K candidates (subsample 200)
    n_sub = min(N, 200)
    idx = np.random.choice(N, n_sub, replace=False)
    act_flat = d["action"][idx].reshape(n_sub, K, -1)  # (sub, K, T*A)
    dists = []
    for i in range(K):
        for j in range(i + 1, K):
            dists.append(np.linalg.norm(act_flat[:, i] - act_flat[:, j], axis=1).mean())
    action_diversity = float(np.mean(dists))

    row = {
        "N"                 : N,
        "K"                 : K,
        "T"                 : T,
        "obs_dim"           : obs_dim,
        "act_dim"           : act_dim,
        "n_unique_ckpts"    : len(unique_ckpts),
        "ckpt_min"          : int(unique_ckpts.min()),
        "ckpt_max"          : int(unique_ckpts.max()),
        "approx_unique_segs": N,   # one demo per pool segment
        "action_diversity"  : action_diversity,
    }
    row.update(stat(rew_demo,  "rew_best"))
    row.update(stat(rew_worst, "rew_worst"))
    row.update(stat(disc_demo, "disc_best"))
    row.update(stat(gap,       "adv_gap"))
    row.update(pos_adv_means)
    row.update(budget_coverage(N))
    return row


def stats_seq_estop(d):
    """seq_estop_labels.npz: obs (N,2,T,D), stop_event, timestep, traj_idx."""
    obs    = d["obs"]     # (N, 2, T, D)
    reward = d["reward"]  # (N, 2, T)
    ckpt   = d["checkpoint_step"]

    N, K, T, obs_dim = obs.shape
    act_dim = d["action"].shape[-1]

    rew_pref = reward[:, 0, :].sum(axis=1)
    disc_pref = discounted_sum(reward[:, 0, :], GAMMA)

    unique_ckpts = np.unique(ckpt)

    # Unique pool trajectories (traj_idx stored per pair)
    if "traj_idx" in d:
        unique_segs = int(len(np.unique(d["traj_idx"])))
    else:
        unique_segs = None

    # Stop time distribution
    row = {
        "N"                 : N,
        "K"                 : K,
        "T"                 : T,
        "obs_dim"           : obs_dim,
        "act_dim"           : act_dim,
        "n_unique_ckpts"    : len(unique_ckpts),
        "ckpt_min"          : int(unique_ckpts.min()),
        "ckpt_max"          : int(unique_ckpts.max()),
        "approx_unique_segs": unique_segs if unique_segs is not None else "N/A",
    }
    row.update(stat(rew_pref,  "rew_best"))
    row.update(stat(disc_pref, "disc_best"))

    if "timestep" in d:
        row.update(stat(d["timestep"].astype(float), "stop_timestep"))

    if "adv_scores" in d:
        gap = d["adv_scores"][:, 0] - d["adv_scores"][:, -1]
        row.update(stat(gap, "adv_gap"))

    row.update(budget_coverage(N))
    return row


def stats_credit(d):
    """credit_labels.npz: obs (N,C,k,D), adv_scores (N,C), chosen_idx (N,)."""
    obs        = d["obs"]        # (N, C, k, D)
    adv_scores = d["adv_scores"] # (N, C)
    chosen_idx = d["chosen_idx"] # (N,)
    ckpt       = d["checkpoint_step"]

    N, C, k, obs_dim = obs.shape
    act_dim = d["action"].shape[-1]

    gap = adv_scores.max(axis=1) - adv_scores.min(axis=1)
    unique_ckpts = np.unique(ckpt)

    # chosen_idx==0 means best window is at start of segment
    pct_chosen_zero = float((chosen_idx == 0).mean())
    chosen_adv  = adv_scores[np.arange(N), chosen_idx]
    worst_adv   = adv_scores.min(axis=1)

    row = {
        "N"                 : N,
        "K"                 : C,      # C = number of candidates (analogous to K)
        "T"                 : k,      # subsegment length
        "obs_dim"           : obs_dim,
        "act_dim"           : act_dim,
        "n_unique_ckpts"    : len(unique_ckpts),
        "ckpt_min"          : int(unique_ckpts.min()),
        "ckpt_max"          : int(unique_ckpts.max()),
        "approx_unique_segs": N,      # one credit example per pool segment
        "pct_chosen_zero"   : pct_chosen_zero,
    }
    row.update(stat(chosen_adv, "rew_best"))    # best window adv score
    row.update(stat(worst_adv,  "rew_worst"))
    row.update(stat(gap,        "adv_gap"))
    row.update(budget_coverage(N))
    return row


STATS_FN = {
    "pref"             : lambda d: stats_pref_or_corr(d, "pref"),
    "corr"             : lambda d: stats_pref_or_corr(d, "corr"),
    "demo"             : lambda d: stats_demo(d),
    "seq_estop"        : lambda d: stats_seq_estop(d),
    "scalar"           : lambda d: stats_pref_or_corr(d, "scalar"),
    "credit_assignment": lambda d: stats_credit(d),
}


# ---------------------------------------------------------------------------
# Report printing
# ---------------------------------------------------------------------------

def print_row(row, ftype, env, file_mb):
    print(f"\n  [{ftype}]  N={row['N']:,}  K={row.get('K','?')}  T={row.get('T','?')}  "
          f"obs_dim={row.get('obs_dim','?')}  file={file_mb:.1f}MB")
    print(f"    unique_ckpts={row['n_unique_ckpts']}  "
          f"ckpt_range=[{row['ckpt_min']},{row['ckpt_max']}]  "
          f"approx_unique_segs={row.get('approx_unique_segs','N/A')}")
    if "rew_best_mean" in row:
        print(f"    rew_best : mean={row['rew_best_mean']:.3f}  "
              f"std={row['rew_best_std']:.3f}  "
              f"min={row['rew_best_min']:.3f}  "
              f"max={row['rew_best_max']:.3f}")
    if "rew_worst_mean" in row:
        print(f"    rew_worst: mean={row['rew_worst_mean']:.3f}  "
              f"std={row['rew_worst_std']:.3f}")
    if "adv_gap_mean" in row:
        print(f"    adv_gap  : mean={row['adv_gap_mean']:.3f}  "
              f"std={row['adv_gap_std']:.3f}  "
              f"p50={row['adv_gap_p50']:.3f}  "
              f"min={row['adv_gap_min']:.3f}  "
              f"max={row['adv_gap_max']:.3f}")
    if "improvement_mean" in row:
        print(f"    improvement (corr): mean={row['improvement_mean']:.3f}  "
              f"std={row['improvement_std']:.3f}")
    if "stop_timestep_mean" in row:
        print(f"    stop_tau : mean={row['stop_timestep_mean']:.1f}  "
              f"p50={row['stop_timestep_p50']:.1f}")
    if "pct_chosen_zero" in row:
        print(f"    pct_chosen_zero={row['pct_chosen_zero']*100:.1f}%  "
              f"(best window at start of segment)")
    if "pct_gap_positive" in row:
        print(f"    pct_gap_positive={row['pct_gap_positive']*100:.1f}%")
    # Budget coverage
    budgets_ok = [b for b in BUDGETS if row.get(f"budget_{b}", 0) == b]
    budgets_clamped = [b for b in BUDGETS if row.get(f"budget_{b}", 0) < b]
    if budgets_ok:
        print(f"    budgets fully covered : {budgets_ok}")
    if budgets_clamped:
        print(f"    budgets clamped (N<budget): {budgets_clamped}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels-dir",  type=str,
                        default="/scratch/general/vast/u1472210/mw_de_labels")
    parser.add_argument("--output-dir",  type=str, default="results")
    parser.add_argument("--envs",        nargs="+", default=ENVS)
    parser.add_argument("--seed",        type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    txt_path = os.path.join(args.output_dir, "mw_de_label_stats.txt")
    csv_path = os.path.join(args.output_dir, "mw_de_label_stats.csv")

    all_rows = []

    with open(txt_path, "w") as txt:
        for env in args.envs:
            header = f"{'='*65}\nENV: {env}\n{'='*65}"
            print(header)
            txt.write(header + "\n")

            for ftype, fname in LABEL_FILES.items():
                path = os.path.join(args.labels_dir, env, fname)
                if not os.path.exists(path):
                    msg = f"  [{ftype}]: MISSING"
                    print(msg)
                    txt.write(msg + "\n")
                    all_rows.append({"env": env, "type": ftype, "status": "missing"})
                    continue

                file_mb = file_size_mb(path)
                try:
                    d = np.load(path, allow_pickle=False)
                    row = STATS_FN[ftype](d)
                    row["env"]    = env
                    row["type"]   = ftype
                    row["status"] = "ok"
                    row["file_mb"] = round(file_mb, 1)
                    all_rows.append(row)

                    print_row(row, ftype, env, file_mb)
                    # Also write to txt
                    import io, contextlib
                    buf = io.StringIO()
                    with contextlib.redirect_stdout(buf):
                        print_row(row, ftype, env, file_mb)
                    txt.write(buf.getvalue())

                except Exception as e:
                    msg = f"  [{ftype}]: ERROR — {e}"
                    print(msg)
                    txt.write(msg + "\n")
                    all_rows.append({"env": env, "type": ftype, "status": f"error: {e}"})

            txt.write("\n")
            print()

    # Write CSV
    all_keys = ["env", "type", "status", "file_mb"]
    for r in all_rows:
        for k in r:
            if k not in all_keys:
                all_keys.append(k)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nSaved:\n  {txt_path}\n  {csv_path}")


if __name__ == "__main__":
    main()
