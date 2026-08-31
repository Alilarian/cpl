"""
Merge sharded feedback-label npz files (from --num-shards > 1 runs of
generate_demo_labels.py / generate_corr_labels.py) into one final file.

Generic: concatenates whatever array keys each shard file has along axis 0,
so it works for demo_labels_K<K>_shard<i>of<n>.npz, corr_labels_shard<i>of<n>.npz,
or any future sharded output with the same shape-per-row convention.

Usage:
    python scripts/merge_label_shards.py \\
        --shard-glob 'datasets/mw/demo_labels/mw_door-open-v2/demo_labels_K7_shard*of4.npz' \\
        --output      datasets/mw/demo_labels/mw_door-open-v2/demo_labels_K7.npz

    # or an explicit list, any order:
    python scripts/merge_label_shards.py \\
        --shards datasets/mw/corr_labels/mw_door-open-v2/corr_labels_shard0of4.npz \\
                 datasets/mw/corr_labels/mw_door-open-v2/corr_labels_shard1of4.npz \\
                 datasets/mw/corr_labels/mw_door-open-v2/corr_labels_shard2of4.npz \\
                 datasets/mw/corr_labels/mw_door-open-v2/corr_labels_shard3of4.npz \\
        --output  datasets/mw/corr_labels/mw_door-open-v2/corr_labels.npz
"""

import argparse
import glob
import io
import os

import numpy as np


def save_npz(path, **arrays):
    """Atomically save a compressed npz (write to .tmp, then rename)."""
    tmp = path + ".tmp"
    with io.BytesIO() as buf:
        np.savez_compressed(buf, **arrays)
        buf.seek(0)
        with open(tmp, "wb") as f:
            f.write(buf.read())
    os.replace(tmp, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", nargs="+", default=None,
                        help="Explicit list of shard npz files, any order.")
    parser.add_argument("--shard-glob", type=str, default=None,
                        help="Glob pattern matching shard npz files "
                             "(alternative to --shards).")
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    if args.shards:
        paths = list(args.shards)
    elif args.shard_glob:
        paths = sorted(glob.glob(args.shard_glob))
    else:
        raise SystemExit("Pass --shards <files...> or --shard-glob '<pattern>'")

    if not paths:
        raise SystemExit(f"No shard files found ({args.shard_glob or args.shards}).")

    if os.path.exists(args.output):
        print(f"Output already exists: {args.output}. Delete it to regenerate.")
        return

    print(f"Merging {len(paths)} shard files:")
    merged = {}
    total_n = 0
    for p in paths:
        with np.load(p) as d:
            first_key = d.files[0]
            n = d[first_key].shape[0]
            print(f"  {p}: {n:,} rows")
            total_n += n
            for k in d.files:
                merged.setdefault(k, []).append(d[k])

    out = {k: np.concatenate(v, axis=0) for k, v in merged.items()}

    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)
    save_npz(args.output, **out)

    print(f"\nMerged {total_n:,} total rows -> {args.output}")
    for k, v in out.items():
        print(f"  {k:16s}: {v.shape}")


if __name__ == "__main__":
    main()
