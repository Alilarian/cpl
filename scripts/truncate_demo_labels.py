"""
Truncate a merged demo_labels_K<K>.npz to an exact row count.

generate_demo_labels.py has no "keep generating until N rows survive" mode --
--max-segments takes a fixed prefix of the pool and some of those get dropped
by the skip_done / skip_gap filters, so the merged output lands at whatever
count happens to survive. To pin an exact row count for downstream use (e.g.
matching N across envs), overshoot --max-segments so the merge comfortably
exceeds the target, then truncate to exactly --n here.

This is a plain prefix truncation (first --n rows, in generation order) --
there's no natural cross-row quality ranking to sort by (only the K
candidates *within* a row are ranked, by generate_demo_labels.py itself).

Usage:
    python scripts/truncate_demo_labels.py \\
        --demo-labels /scratch/.../mw_de_labels/mw_button-press-v2/demo_labels_K9.npz \\
        --n 10000
"""

import argparse
import io
import os

import numpy as np


def save_npz(path, **arrays):
    """Atomically save a compressed npz file (write to tmp, then rename)."""
    tmp_path = path + ".tmp"
    with io.BytesIO() as buf:
        np.savez_compressed(buf, **arrays)
        buf.seek(0)
        with open(tmp_path, "wb") as f:
            f.write(buf.read())
    os.replace(tmp_path, path)


def main():
    parser = argparse.ArgumentParser(
        description="Truncate a merged demo_labels_K<K>.npz to exactly --n rows."
    )
    parser.add_argument("--demo-labels", type=str, required=True,
                        help="Path to a merged demo_labels_K<K>.npz (in place).")
    parser.add_argument("--n", type=int, required=True,
                        help="Exact row count to keep (first --n rows).")
    args = parser.parse_args()

    with open(args.demo_labels, "rb") as f:
        raw = dict(np.load(f))

    N_full = raw["obs"].shape[0]
    if N_full < args.n:
        raise SystemExit(
            f"Only {N_full} rows available, cannot truncate up to {args.n}. "
            f"Rerun generation with a larger --max-segments first."
        )
    if N_full == args.n:
        print(f"Already exactly {args.n} rows -- nothing to do.")
        return

    truncated = {k: v[:args.n] for k, v in raw.items()}
    save_npz(args.demo_labels, **truncated)
    print(f"Truncated {args.demo_labels}: {N_full} -> {args.n} rows")


if __name__ == "__main__":
    main()
