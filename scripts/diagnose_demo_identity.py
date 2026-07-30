"""
Diagnostic: identify which candidate slot is the expert rollout in already-generated
demo labels, without regenerating the dataset.

Motivation
----------
generate_demo_labels.py builds K candidates per pool segment (expert rollout,
stratified-tier rollouts, original pool segment), scores them all with the oracle,
sorts descending by rl_sum, and saves only the sorted (obs, action, reward, adv_scores)
arrays.  The identity of each slot (which one was "expert") is discarded at generation
time — demo_labels_K{K}.npz cannot tell you whether index 0 ("the demo") really is the
expert rollout or a counterfactual that happened to score higher.

This script recovers that identity for an already-generated demo_labels file, at a
fraction of the original cost:
  - Re-rolls ONLY the expert model per segment (not all K candidates).
  - Does NOT re-run oracle scoring (adv_scores are already saved and reused as-is).
  - Can run on a random subsample instead of the full dataset.

Two matching steps per demo example:
  1. Pool-index recovery: exactly one of the K candidates is an unmodified copy of a
     pool.npz row (the "original" — see generate_demo_labels.py's `pool_obs[i]`
     candidate). It's found by exact byte-match against pool.npz, keyed by
     checkpoint_step (already stored per demo example) to narrow the search.
     This also recovers s_0 = pool_state[pool_idx, 0] needed for step 2, and reports
     which slot the "original" candidate landed in as a free byproduct.
  2. Expert-identity recovery: roll the expert model out from the same s_0 and match
     (allclose) against the K stored candidates to find which slot (if any) is the
     expert rollout.

Interpreting output:
  - If pool-index matches are common but expert matches are rare, the wrong
    --run-dir/--expert-checkpoint was probably given (labels were generated with a
    different oracle checkpoint) — try the other candidate run dir.
  - If pool-index matches themselves are rare, --pool-path likely doesn't correspond
    to the same pool.npz used to generate --demo-labels-path.

Usage:
    python scripts/diagnose_demo_identity.py \\
        --pool-path        /scratch/.../mw_de_pool/mw_plate-slide-v2/pool.npz \\
        --demo-labels-path datasets/mw/demo_labels/mw_plate-slide-v2/demo_labels_K7.npz \\
        --run-dir          runs/runs/chpc/oracle_sac_seeds/mw_plate-slide-v2/seed-1 \\
        --n-samples        2000 \\
        --output           results/demo_identity_mw_plate-slide-v2.npz
"""

import argparse
import hashlib
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_demo_labels import load_model, rollout_from_state  # noqa: E402


def hash_row(arr):
    """Cheap exact-match fingerprint for a (T, dim) float32 array."""
    return hashlib.md5(np.ascontiguousarray(arr, dtype=np.float32).tobytes()).digest()


def build_pool_lookup(pool_obs, pool_ckpt):
    """(checkpoint_step, obs-hash) -> pool index, for exact "original" matching."""
    lookup = {}
    for i in range(pool_obs.shape[0]):
        key = (int(pool_ckpt[i]), hash_row(pool_obs[i]))
        lookup[key] = i
    return lookup


def main():
    parser = argparse.ArgumentParser(
        description="Recover expert-rollout identity in an already-generated demo_labels file."
    )
    parser.add_argument("--pool-path", type=str, required=True,
                        help="Path to the pool.npz used to generate --demo-labels-path.")
    parser.add_argument("--demo-labels-path", type=str, required=True,
                        help="Path to demo_labels_K{K}.npz to diagnose.")
    parser.add_argument("--run-dir", type=str, required=True,
                        help="Oracle run dir (config.yaml + checkpoints) believed to have "
                             "generated these labels.")
    parser.add_argument("--expert-checkpoint", type=str, default="best_model.pt",
                        help="Expert checkpoint filename (default: best_model.pt)")
    parser.add_argument("--n-samples", type=int, default=2000,
                        help="Number of demo examples to check (random subsample). "
                             "Use <=0 to process all. (default: 2000)")
    parser.add_argument("--seed", type=int, default=0,
                        help="RNG seed for subsampling (default: 0)")
    parser.add_argument("--atol", type=float, default=1e-4,
                        help="Absolute tolerance for expert-rollout match (default: 1e-4)")
    parser.add_argument("--rtol", type=float, default=1e-4,
                        help="Relative tolerance for expert-rollout match (default: 1e-4)")
    parser.add_argument("--output", type=str, default=None,
                        help="Where to save per-example results (.npz). "
                             "Default: <demo-labels-dir>/demo_identity_diagnostic.npz")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # Load pool + demo labels
    # ------------------------------------------------------------------
    print(f"\nLoading pool: {args.pool_path}")
    with open(args.pool_path, "rb") as f:
        pool = np.load(f)
        pool_obs   = pool["obs"]             # (N_pool, T, obs_dim)
        pool_state = pool["state"]           # (N_pool, T, state_dim)
        pool_ckpt  = pool["checkpoint_step"] # (N_pool,)
    print(f"  N_pool={pool_obs.shape[0]}")

    print(f"\nLoading demo labels: {args.demo_labels_path}")
    demo = np.load(args.demo_labels_path)
    demo_obs    = demo["obs"]             # (N, K, T, obs_dim)
    demo_action = demo["action"]          # (N, K, T, act_dim)
    demo_adv    = demo["adv_scores"]      # (N, K)  sorted descending
    demo_ckpt   = demo["checkpoint_step"] # (N,)  checkpoint of the ORIGINAL candidate
    N, K, T, obs_dim = demo_obs.shape
    print(f"  N={N} examples, K={K} candidates, T={T} steps")

    # ------------------------------------------------------------------
    # Build exact-match lookup for pool-index recovery
    # ------------------------------------------------------------------
    print("\nBuilding pool lookup table (checkpoint_step, obs-hash) -> pool index ...")
    pool_lookup = build_pool_lookup(pool_obs, pool_ckpt)
    print(f"  {len(pool_lookup)} unique entries")

    # ------------------------------------------------------------------
    # Load expert model (also doubles as env source for rollouts)
    # ------------------------------------------------------------------
    expert_path = os.path.join(args.run_dir, args.expert_checkpoint)
    print(f"\nLoading expert: {expert_path}")
    expert_model, env = load_model(args.run_dir, expert_path, device)

    # ------------------------------------------------------------------
    # Sample demo examples to check
    # ------------------------------------------------------------------
    rng = np.random.default_rng(args.seed)
    if args.n_samples <= 0 or args.n_samples >= N:
        indices = np.arange(N)
    else:
        indices = rng.choice(N, size=args.n_samples, replace=False)
    indices.sort()
    print(f"\nChecking {len(indices)}/{N} demo examples "
          f"(1 expert rollout each; no rescoring)...\n")

    # ------------------------------------------------------------------
    # Per-example diagnostics
    # ------------------------------------------------------------------
    pool_idx_out      = np.full(len(indices), -1, dtype=np.int64)
    original_rank_out = np.full(len(indices), -1, dtype=np.int8)
    expert_rank_out   = np.full(len(indices), -1, dtype=np.int8)  # -1 = no match / skipped
    gap_to_winner_out = np.full(len(indices), np.nan, dtype=np.float32)

    n_pool_lookup_failed = 0
    n_expert_rollout_failed = 0
    n_expert_no_match = 0

    for out_i, n in enumerate(indices):
        if out_i % 200 == 0:
            print(f"  [{out_i:>5}/{len(indices)}]  "
                  f"pool_fail={n_pool_lookup_failed}  "
                  f"rollout_fail={n_expert_rollout_failed}  "
                  f"no_match={n_expert_no_match}")

        # ---- Step 1: recover pool index via exact "original" match ----
        ckpt_n = int(demo_ckpt[n])
        pool_idx = None
        original_rank = -1
        for k in range(K):
            key = (ckpt_n, hash_row(demo_obs[n, k]))
            if key in pool_lookup:
                pool_idx = pool_lookup[key]
                original_rank = k
                break

        if pool_idx is None:
            n_pool_lookup_failed += 1
            continue
        pool_idx_out[out_i] = pool_idx
        original_rank_out[out_i] = original_rank

        # ---- Step 2: re-roll expert from the same s_0 ----
        s0 = pool_state[pool_idx, 0]
        result = rollout_from_state(expert_model, env, s0, T, device)
        if result is None:
            n_expert_rollout_failed += 1
            continue
        exp_obs, exp_action, _ = result

        # ---- Match against the K stored candidates ----
        expert_rank = -1
        for k in range(K):
            if (np.allclose(exp_obs, demo_obs[n, k], atol=args.atol, rtol=args.rtol) and
                    np.allclose(exp_action, demo_action[n, k], atol=args.atol, rtol=args.rtol)):
                expert_rank = k
                break

        if expert_rank == -1:
            n_expert_no_match += 1
            continue

        expert_rank_out[out_i] = expert_rank
        if expert_rank != 0:
            gap_to_winner_out[out_i] = demo_adv[n, 0] - demo_adv[n, expert_rank]

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    n_checked = len(indices)
    n_pool_matched = n_checked - n_pool_lookup_failed
    n_rollout_ok = n_pool_matched - n_expert_rollout_failed
    n_identified = n_rollout_ok - n_expert_no_match

    print(f"\n{'='*60}")
    print(f"Checked                    : {n_checked}")
    print(f"Pool index recovered       : {n_pool_matched}  ({100*n_pool_matched/n_checked:.1f}%)")
    print(f"Expert rollout succeeded   : {n_rollout_ok}  ({100*n_rollout_ok/max(1,n_pool_matched):.1f}% of recovered)")
    print(f"Expert identity matched    : {n_identified}  ({100*n_identified/max(1,n_rollout_ok):.1f}% of rollouts)")

    if n_pool_matched > 0 and n_pool_matched / n_checked < 0.5:
        print("\n[WARNING] Pool-index recovery rate is low — --pool-path likely does not "
              "correspond to the same pool.npz used to generate --demo-labels-path.")
    elif n_rollout_ok > 0 and n_identified / n_rollout_ok < 0.5:
        print("\n[WARNING] Expert identity match rate is low despite pool matches succeeding — "
              "--run-dir/--expert-checkpoint is likely NOT the oracle used to generate these "
              "labels. Try the other candidate run dir (e.g. oracle_sac_all vs "
              "oracle_sac_seeds/seed-N).")

    if n_identified > 0:
        matched_mask = expert_rank_out >= 0
        ranks = expert_rank_out[matched_mask]
        is_demo = ranks == 0
        print(f"\nAmong {n_identified} identified examples:")
        print(f"  expert IS the demo (rank 0)      : {is_demo.sum()}  ({100*is_demo.mean():.1f}%)")
        print(f"  expert relegated to counterfactual: {(~is_demo).sum()}  ({100*(~is_demo).mean():.1f}%)")

        print(f"\n  Expert-rank histogram (0 = demo, {K-1} = worst):")
        for k in range(K):
            cnt = (ranks == k).sum()
            print(f"    rank {k}: {cnt:>5}  ({100*cnt/len(ranks):.1f}%)")

        demoted_gaps = gap_to_winner_out[~np.isnan(gap_to_winner_out)]
        if len(demoted_gaps) > 0:
            print(f"\n  Among demoted examples (expert lost to a noise-selected winner):")
            print(f"    gap (winner rl_sum - expert rl_sum): "
                  f"mean={demoted_gaps.mean():.3f}  median={np.median(demoted_gaps):.3f}  "
                  f"p90={np.percentile(demoted_gaps,90):.3f}")

        orig_matched = original_rank_out[original_rank_out >= 0]
        orig_is_demo = orig_matched == 0
        print(f"\n  [bonus] original pool segment IS the demo (rank 0): "
              f"{orig_is_demo.sum()}/{len(orig_matched)}  ({100*orig_is_demo.mean():.1f}%)")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    out_path = args.output
    if out_path is None:
        out_dir = os.path.dirname(os.path.abspath(args.demo_labels_path))
        out_path = os.path.join(out_dir, "demo_identity_diagnostic.npz")
    np.savez_compressed(
        out_path,
        demo_index=indices,
        pool_index=pool_idx_out,
        original_rank=original_rank_out,
        expert_rank=expert_rank_out,
        gap_to_winner=gap_to_winner_out,
    )
    print(f"\nSaved per-example results -> {out_path}")


if __name__ == "__main__":
    main()
