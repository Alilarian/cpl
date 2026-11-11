"""
Derive larger-stride credit assignment labels from an existing stride-1 file.

generate_credit_labels.py scores every window independently, so the oracle
score of the window starting at step s does not depend on which other windows
are candidates.  Keeping the windows at starts 0, stride, 2*stride, ... of a
stride-1 file and re-taking the argmax therefore gives exactly the labels the
generator would produce with --stride, with the same oracle that labeled the
source file.  No oracle or simulator is needed.

Every window must be covered: stride <= k and (T-k) % stride == 0, where
T = C_src + k - 1 is the length of the base segment.  For T=64, k=20 the
valid strides are 1, 2, 4 and 11.

The obs/action arrays are streamed from the compressed npz in row chunks, so
peak RAM is the output plus one chunk (the 100k-row MW DE files are ~12.6 GB
of obs uncompressed).

Output keeps the PMCreditAssignmentBuffer schema:
    obs           : (N, C, k, obs_dim)  windows at window_starts
    action        : (N, C, k, act_dim)
    adv_scores    : (N, C)              source oracle score of each kept window
    chosen_idx    : (N,) int32          argmax over the kept windows
    checkpoint_step : (N,)
    window_starts : (C,) int32          start step of each window in the base segment
    stride        : ()   int64

Usage:
    python scripts/restride_credit_labels.py \\
        --input  datasets/mw_lowsucc/mw_button-press-v2/credit_same_labels.npz \\
        --output datasets/mw_lowsucc/mw_button-press-v2/credit_same_stride11_labels.npz \\
        --stride 11
"""

import argparse
import io
import os
import zipfile

import numpy as np
import numpy.lib.format as npformat


def save_npz(path, **arrays):
    """Atomically save a compressed npz (write to .tmp, then rename)."""
    tmp = path + ".tmp"
    with io.BytesIO() as buf:
        np.savez_compressed(buf, **arrays)
        buf.seek(0)
        with open(tmp, "wb") as f:
            f.write(buf.read())
    os.replace(tmp, path)


def npz_header(zf, key):
    """(shape, dtype) of one array in an npz without reading its data."""
    with zf.open(key + ".npy") as fh:
        version = npformat.read_magic(fh)
        shape, fortran, dtype = npformat._read_array_header(fh, version)
    assert not fortran, f"{key}: Fortran-ordered arrays are not supported"
    return shape, dtype


def stream_select(zf, key, starts, chunk_rows):
    """
    Read a (N, C_src, k, D) window array in row chunks and keep the windows
    at `starts`.  Checks that the source windows are stride-1 slices of one
    base segment (window c shifted by one step equals window c+1).
    """
    shape, dtype = npz_header(zf, key)
    N, C_src, k, D = shape
    row_bytes = C_src * k * D * dtype.itemsize
    out = np.empty((N, len(starts), k, D), dtype=np.float32)
    with zf.open(key + ".npy") as fh:
        version = npformat.read_magic(fh)
        npformat._read_array_header(fh, version)
        for s in range(0, N, chunk_rows):
            e = min(s + chunk_rows, N)
            buf = fh.read((e - s) * row_bytes)
            win = np.frombuffer(buf, dtype=dtype).reshape(e - s, C_src, k, D)
            if not np.array_equal(win[:, 1:, :-1], win[:, :-1, 1:]):
                raise ValueError(f"{key}: source windows are not stride-1 slices of one segment "
                                 f"(rows {s}-{e})")
            out[s:e] = win[:, starts]
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Derive larger-stride credit assignment labels from a stride-1 file."
    )
    parser.add_argument("--input", type=str, required=True,
                        help="Stride-1 credit_labels.npz from generate_credit_labels.py")
    parser.add_argument("--output", type=str, required=True,
                        help="Output npz path")
    parser.add_argument("--stride", type=int, default=11,
                        help="Stride between kept window starts (default: 11)")
    parser.add_argument("--chunk-rows", type=int, default=2000,
                        help="Rows decompressed per chunk (default: 2000)")
    args = parser.parse_args()

    if os.path.exists(args.output):
        print(f"\nOutput already exists: {args.output}")
        print("Delete it to regenerate.")
        return

    zf = zipfile.ZipFile(args.input)
    keys = {n[:-4] for n in zf.namelist()}
    if "window_starts" in keys:
        print(f"ERROR: {args.input} already has window_starts -- source must be a stride-1 file")
        return

    (N, C_src, k, obs_dim), _ = npz_header(zf, "obs")
    T = C_src + k - 1
    starts = np.arange(0, T - k + 1, args.stride)
    C = len(starts)

    print("=" * 65)
    print(f"Input            : {args.input}")
    print(f"Output           : {args.output}")
    print(f"N={N:,}  T={T}  k={k}  C_src={C_src}")
    print(f"Stride           : {args.stride}  ->  C={C}  starts={starts.tolist()}")
    print("=" * 65)

    if args.stride > k or starts[-1] + k != T:
        print(f"ERROR: --stride {args.stride} leaves steps uncovered "
              f"(need stride <= k={k} and (T-k)={T - k} divisible by stride)")
        return
    if C < 2:
        print(f"ERROR: only {C} window -- need C >= 2")
        return

    src = np.load(args.input)
    src_scores = src["adv_scores"].astype(np.float32)   # (N, C_src)
    src_chosen = src["chosen_idx"].astype(np.int64)     # (N,)
    ckpt       = src["checkpoint_step"]
    n_bad = int((src_scores.argmax(axis=1) != src_chosen).sum())
    if n_bad:
        print(f"ERROR: source chosen_idx != argmax(adv_scores) in {n_bad:,} rows")
        return

    win_scores = src_scores[:, starts]                              # (N, C)
    chosen_idx = np.argmax(win_scores, axis=1).astype(np.int32)     # (N,)

    print(f"\nStreaming obs / action windows ...")
    obs    = stream_select(zf, "obs", starts, args.chunk_rows)
    action = stream_select(zf, "action", starts, args.chunk_rows)

    # ------------------------------------------------------------------
    # Statistics (new labels vs the stride-1 source)
    # ------------------------------------------------------------------
    srt    = np.sort(win_scores, axis=1)[:, ::-1]
    gap    = srt[:, 0] - srt[:, 1]
    spread = srt[:, 0] - srt[:, -1]
    new_start = starts[chosen_idx]
    nearest   = starts[np.abs(starts[None, :] - src_chosen[:, None]).argmin(axis=1)]
    pct = [5, 25, 50, 75, 95]

    print(f"\n{'─'*60}")
    print(f"Stride-{args.stride} credit assignment statistics")
    print(f"{'─'*60}")
    print(f"  Best-vs-2nd gap        : median={np.median(gap):.4f}  "
          f"{'  '.join(f'p{p}={v:.3f}' for p, v in zip(pct, np.percentile(gap, pct)))}")
    print(f"  Gap / (best - worst)   : median={np.median(gap / np.maximum(spread, 1e-9)):.3f}")
    print(f"  Near-ties (gap < 0.05) : {(gap < 0.05).mean():.3f}")
    print(f"  Same window as stride 1: {(new_start == src_chosen).mean():.3f}")
    print(f"  Nearest grid window    : {(new_start == nearest).mean():.3f}")
    counts = np.bincount(chosen_idx, minlength=C)
    print(f"  Chosen window counts   : "
          + "  ".join(f"[{s}]={c:,}" for s, c in zip(starts, counts)))
    print(f"{'─'*60}")

    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)
    save_npz(
        args.output,
        obs=obs,
        action=action,
        adv_scores=win_scores,
        chosen_idx=chosen_idx,
        checkpoint_step=ckpt,
        window_starts=starts.astype(np.int32),
        stride=np.int64(args.stride),
    )

    print(f"\nSaved → {args.output}")
    print(f"  obs           : {obs.shape}  (N, C, k, obs_dim)")
    print(f"  action        : {action.shape}")
    print(f"  adv_scores    : {win_scores.shape}")
    print(f"  chosen_idx    : {chosen_idx.shape}")
    print(f"  window_starts : {starts.tolist()}")


if __name__ == "__main__":
    main()
