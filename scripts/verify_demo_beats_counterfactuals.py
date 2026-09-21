"""
Verify that the demo (K-index 0, expert rollout) scores at least as well as
every counterfactual (K-index 1..K-1) in an already-generated demo_labels_K<K>.npz.

generate_demo_labels.py pins the expert at index 0 unconditionally and only
keeps a row if rl_sum(expert) - rl_sum(best counterfactual) >= --min-adv-gap
(default 0.0) -- so with the default gap, adv_scores[:, 0] >= adv_scores[:, 1:].max(1)
should hold for every row by construction. This script re-checks that directly
against the saved adv_scores (no rollouts, no model loading -- just numpy), as a
fast regression check independent of that generation-time filter.

This answers a different question than scripts/diagnose_demo_identity.py, which
re-rolls the expert to check whether rank 0 really *is* the expert rollout (a
provenance check). This script only checks whether rank 0's *score* dominates,
using whatever is already saved.

Usage:
    python scripts/verify_demo_beats_counterfactuals.py \\
        --demo-labels datasets/demo_labels/mw_drawer-open-v2/demo_labels_K9.npz \\
        --min-adv-gap 0.0
"""

import argparse

import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Check that adv_scores[:, 0] (the demo) >= max(adv_scores[:, 1:]) "
                    "(the counterfactuals) for every row of a demo_labels_K<K>.npz file.",
    )
    parser.add_argument("--demo-labels", type=str, nargs="+", required=True,
                        help="One or more demo_labels_K<K>.npz paths.")
    parser.add_argument("--min-adv-gap", type=float, default=0.0,
                        help="Same --min-adv-gap used at generation time (default: 0.0). "
                             "A row fails here iff adv_scores[i,0] - max(adv_scores[i,1:]) "
                             "< --min-adv-gap.")
    args = parser.parse_args()

    any_violation = False

    for path in args.demo_labels:
        print(f"{'='*65}\n{path}\n{'='*65}")
        with open(path, "rb") as f:
            raw = np.load(f)
            adv = raw["adv_scores"]  # (N, K)

        N, K = adv.shape
        demo_score = adv[:, 0]
        best_cf_score = adv[:, 1:].max(axis=1)
        gap = demo_score - best_cf_score

        n_fail = int((gap < args.min_adv_gap).sum())
        print(f"  N={N} examples, K={K} candidates")
        print(f"  gap (demo - best counterfactual): "
              f"mean={gap.mean():.3f}  min={gap.min():.3f}  max={gap.max():.3f}")

        if n_fail > 0:
            any_violation = True
            worst = np.argsort(gap)[:min(10, n_fail)]
            print(f"\n  FAIL: {n_fail}/{N} ({100*n_fail/N:.2f}%) rows have "
                  f"demo - best_counterfactual < {args.min_adv_gap}")
            print(f"  Worst offenders (row index, gap):")
            for i in worst:
                print(f"    row {i:>7}: gap={gap[i]:.4f}")
        else:
            print(f"\n  OK: all {N} rows satisfy demo >= best counterfactual "
                  f"(gap >= {args.min_adv_gap})")
        print()

    if any_violation:
        print("One or more files have rows where the demo does NOT beat its "
              "counterfactuals. This should not happen given --min-adv-gap filtering "
              "at generation time -- investigate before training on this data.")
    else:
        print("All checked files: the demo dominates its counterfactuals on every row.")


if __name__ == "__main__":
    main()
