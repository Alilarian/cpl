"""
Debug script: inspect generated pref datasets to diagnose data-efficiency issues.

Checks:
  1. Does median gap collapse at higher budgets?
  2. Are many pairs same-tier comparisons?
  3. Are preferred tiers often lower (worse) than non-preferred?
  4. Are large budgets dominated by near-duplicate segments?
  5. Is n_out actually equal to the requested budget?

Usage:
  python scripts/debug_pref.py
  python scripts/debug_pref.py --seed 0 --data-dir datasets/pm/data_eff
"""

import argparse
import os
import numpy as np

BUDGETS = [50, 100, 300, 600, 1000, 2000, 5000, 10000, 20000]


def inspect_pref(path, budget):
    if not os.path.exists(path):
        print(f"  MISSING: {path}")
        return

    d    = np.load(path)
    gap  = d["gap"]                    # (B,)   always positive
    ckpt = d["checkpoint_step"]        # (B, 2) col0=preferred, col1=non-preferred
    obs  = d["obs"]                    # (B, 2, T, obs_dim)
    B    = len(gap)

    sep = "─" * 62
    print(sep)
    print(f"  File   : {os.path.basename(path)}")

    # ── Q5: actual vs requested budget ──────────────────────────────────
    match = "✓" if B == budget else f"✗ (requested {budget})"
    print(f"  Q5  n_out = {B}  {match}")

    # ── Q1: gap distribution ─────────────────────────────────────────────
    pcts = [5, 25, 50, 75, 95]
    pv   = np.percentile(gap, pcts)
    print(f"  Q1  gap   mean={gap.mean():.4f}  "
          f"med={np.median(gap):.4f}  "
          f"min={gap.min():.4f}  max={gap.max():.4f}")
    print(f"        " + "  ".join(f"p{p}={v:.4f}" for p, v in zip(pcts, pv)))

    # fraction of pairs with very small gap
    for thresh in [0.01, 0.05, 0.1, 0.5]:
        frac = float((gap < thresh).mean())
        print(f"        gap < {thresh:.2f} : {100*frac:5.1f}%")

    # ── Q2: same-tier comparisons ────────────────────────────────────────
    pref_tier  = ckpt[:, 0]
    npref_tier = ckpt[:, 1]
    same_tier  = float((pref_tier == npref_tier).mean())
    print(f"  Q2  same-tier pairs : {100*same_tier:.1f}%")

    # Tier distribution matrix (capped at 8 tiers for readability)
    tiers     = np.unique(np.concatenate([pref_tier, npref_tier]))
    n_tiers   = min(len(tiers), 8)
    top_tiers = tiers[:n_tiers]
    mat       = np.zeros((n_tiers, n_tiers), dtype=int)
    tier_map  = {t: i for i, t in enumerate(top_tiers)}
    for p, n in zip(pref_tier, npref_tier):
        pi = tier_map.get(p, -1)
        ni = tier_map.get(n, -1)
        if pi >= 0 and ni >= 0:
            mat[pi, ni] += 1
    print(f"        tier matrix  rows=preferred  cols=non-preferred")
    print(f"        tiers: {top_tiers.tolist()}")
    for i, row in enumerate(mat):
        print(f"          tier {top_tiers[i]:6d}: {row.tolist()}")

    # ── Q3: preferred tier vs non-preferred tier ordering ───────────────
    # Higher checkpoint_step = later in training = generally better policy
    pref_higher  = float((pref_tier >  npref_tier).mean())
    pref_lower   = float((pref_tier <  npref_tier).mean())
    pref_equal   = float((pref_tier == npref_tier).mean())
    print(f"  Q3  pref ckpt > non-pref ckpt : {100*pref_higher:.1f}%  (expected high)")
    print(f"      pref ckpt < non-pref ckpt : {100*pref_lower:.1f}%  (inverted — bad)")
    print(f"      pref ckpt = non-pref ckpt : {100*pref_equal:.1f}%  (same tier)")

    # ── Q4: near-duplicate detection via obs fingerprint ─────────────────
    # Use first obs of preferred segment as a cheap fingerprint
    first_obs    = obs[:, 0, 0, :]           # (B, obs_dim) — start state of preferred
    # Round to 2 decimal places to cluster near-identical starting states
    fingerprints = np.round(first_obs, 2)
    _, counts    = np.unique(fingerprints, axis=0, return_counts=True)
    dup_frac     = float((counts > 1).sum()) / len(counts)
    max_repeat   = int(counts.max())
    print(f"  Q4  unique start-states (preferred, rounded to 0.01):")
    print(f"        {len(counts)} unique / {B} total  "
          f"({100*(1 - len(counts)/B):.1f}% repeated)")
    print(f"        most-repeated start-state appears {max_repeat}×")
    # Distribution of repeat counts
    for thresh in [2, 5, 10, 50]:
        n = int((counts >= thresh).sum())
        print(f"        start-states appearing ≥ {thresh:2d}× : {n}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",     type=int, default=0)
    parser.add_argument("--data-dir", default="datasets/pm/data_eff")
    args = parser.parse_args()

    print(f"\nPref dataset inspection  (seed={args.seed})")
    print(f"Data dir: {args.data_dir}\n")

    for B in BUDGETS:
        path = os.path.join(args.data_dir, f"pref_b{B:06d}_s{args.seed}.npz")
        inspect_pref(path, B)
        print()


if __name__ == "__main__":
    main()
