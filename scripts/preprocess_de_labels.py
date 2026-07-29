#!/usr/bin/env python3
"""
Preprocess MetaWorld DE label files so that capacity=N at training time
gives an unbiased, diverse N samples instead of the first N in pool order.

Strategy per type
-----------------
seq_estop   : stratified sampling — keep exactly 1 pair per source trajectory
              (removes within-trajectory duplication), then shuffle.
              713k pairs → ~161k (one per traj), random order.

scalar      : shuffle only — already <1 pair per trajectory on average.
              Pool ordering bias is removed; diversity is preserved.

credit      : shuffle only — exactly 1 example per trajectory by construction.
              Pool ordering bias is removed.

Output
------
Preprocessed files are written to --output-dir/<env>/<same_filename>.
The original files are never touched.
Set LABELS_DIR=<output-dir> when calling mw_de_train.sbatch.
"""
import argparse
import os
import sys
import numpy as np


# ── per-type label filename ────────────────────────────────────────────────────
FILE_MAP = {
    "seq_estop":        "seq_estop_labels.npz",
    "scalar":           "scalar_labels.npz",
    "credit_assignment": "credit_labels.npz",
}


def stratify_estop(arrays: dict, rng: np.random.Generator) -> dict:
    """
    Keep exactly one pair per source trajectory, chosen at random,
    then shuffle the result.

    Requires a 'traj_idx' key (M,) in arrays.
    """
    traj_idx = arrays["traj_idx"]   # (M,)

    # Sort pairs by trajectory so we can group them efficiently.
    sort_order = np.argsort(traj_idx, kind="stable")
    sorted_trajs = traj_idx[sort_order]

    _, first_occ, counts = np.unique(
        sorted_trajs, return_index=True, return_counts=True
    )
    # Random offset within each trajectory's group.
    offsets = rng.integers(0, counts)           # (n_trajs,)
    selected_sorted = first_occ + offsets       # index into sort_order
    selected = sort_order[selected_sorted]      # back to original row indices

    rng.shuffle(selected)
    return {k: v[selected] for k, v in arrays.items()}


def shuffle_all(arrays: dict, rng: np.random.Generator) -> dict:
    N = next(iter(arrays.values())).shape[0]
    perm = rng.permutation(N)
    return {k: v[perm] for k, v in arrays.items()}


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess seq_estop / scalar / credit label files."
    )
    parser.add_argument("--labels-dir", required=True,
                        help="Root label directory (contains <env>/ subdirs).")
    parser.add_argument("--output-dir", required=True,
                        help="Root output directory for preprocessed files.")
    parser.add_argument("--env", required=True,
                        help="Env subdirectory, e.g. mw_drawer-open-v2.")
    parser.add_argument("--type", required=True,
                        choices=list(FILE_MAP.keys()),
                        help="Feedback type to preprocess.")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed for reproducibility (default: 42).")
    args = parser.parse_args()

    fname   = FILE_MAP[args.type]
    in_path = os.path.join(args.labels_dir, args.env, fname)

    out_env_dir = os.path.join(args.output_dir, args.env)
    os.makedirs(out_env_dir, exist_ok=True)
    out_path = os.path.join(out_env_dir, fname)

    if not os.path.exists(in_path):
        print(f"ERROR: input file not found: {in_path}", file=sys.stderr)
        sys.exit(1)

    mb = os.path.getsize(in_path) / 1e6
    print(f"Loading {in_path}  ({mb:.0f} MB compressed) ...")
    arrays = dict(np.load(in_path))

    N_in = next(iter(arrays.values())).shape[0]
    print(f"  N_in = {N_in:,}  keys = {list(arrays.keys())}")

    rng = np.random.default_rng(args.seed)

    if args.type == "seq_estop":
        if "traj_idx" not in arrays:
            print("ERROR: seq_estop file missing 'traj_idx' key.", file=sys.stderr)
            sys.exit(1)
        n_trajs = len(np.unique(arrays["traj_idx"]))
        print(f"Stratifying (1 pair / trajectory) over {n_trajs:,} trajectories ...")
        result = stratify_estop(arrays, rng)
    else:
        print(f"Shuffling {N_in:,} samples ...")
        result = shuffle_all(arrays, rng)

    N_out = next(iter(result.values())).shape[0]
    print(f"  N_out = {N_out:,}")

    print(f"Saving to {out_path} ...")
    np.savez_compressed(out_path, **result)
    out_mb = os.path.getsize(out_path) / 1e6
    print(f"Done — {out_mb:.0f} MB written.")
    print()
    print("To train with these preprocessed labels set:")
    print(f"  LABELS_DIR={args.output_dir}")


if __name__ == "__main__":
    main()
