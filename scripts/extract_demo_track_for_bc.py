"""
Extract the demo track (K-index 0, the expert rollout) out of a
demo_labels_K<K>.npz file produced by generate_demo_labels.py, dropping the
counterfactuals, and save it in the same flat schema as score_pool_for_bc.py's
bc_pool.npz -- so it can be fed straight into
multi-type-feedback/multi_type_feedback/convert_research_pool_to_demos.py
without any changes to that converter.

This is how the BC baseline gets trained on "the main demo, not the
counterfactuals": BC never sees demo_labels_K<K>.npz's K dimension at all,
only the K=0 slice extracted here.

Output (--output-dir/<env>/demo_track_bc.npz):
    obs             : (N, T, obs_dim)   demo (K=0) track only
    action          : (N, T, act_dim)
    reward          : (N, T)
    rl_sum          : (N,)              adv_scores[:, 0] -- the expert's own oracle score
    checkpoint_step : (N,)              source checkpoint of the underlying pool segment

Usage:
    python scripts/extract_demo_track_for_bc.py \\
        --demo-labels datasets/demo_labels/mw_drawer-open-v2/demo_labels_K9.npz \\
        --output-dir  datasets/demo_track_bc

    python multi-type-feedback/multi_type_feedback/convert_research_pool_to_demos.py \\
        --bc-pool datasets/demo_track_bc/mw_drawer-open-v2/demo_track_bc.npz \\
        --out multi-type-feedback/feedback_regen/research_demo_metaworld-drawer-open-v2_0.pkl
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
        description="Extract the K=0 demo track from demo_labels_K<K>.npz for BC training."
    )
    parser.add_argument("--demo-labels", type=str, required=True,
                        help="Path to demo_labels_K<K>.npz from generate_demo_labels.py")
    parser.add_argument("--output-dir", type=str, default="datasets/demo_track_bc",
                        help="Root output directory. demo_track_bc.npz is written to "
                             "<output-dir>/<env>/.")
    args = parser.parse_args()

    env_name = os.path.basename(os.path.dirname(args.demo_labels))
    out_dir  = os.path.join(args.output_dir, env_name)
    out_path = os.path.join(out_dir, "demo_track_bc.npz")

    if os.path.exists(out_path):
        print(f"Output already exists: {out_path}")
        print("Delete it to regenerate.")
        return

    print(f"Loading {args.demo_labels}")
    with open(args.demo_labels, "rb") as f:
        raw = np.load(f)
        obs             = raw["obs"]              # (N, K, T, obs_dim)
        action          = raw["action"]            # (N, K, T, act_dim)
        reward          = raw["reward"]             # (N, K, T)
        adv_scores      = raw["adv_scores"]          # (N, K)
        checkpoint_step = raw["checkpoint_step"]      # (N,)

    N, K, T, obs_dim = obs.shape
    print(f"  N={N} choice sets, K={K}, T={T}, obs_dim={obs_dim}")
    print(f"  Keeping K-index 0 only (the demo/expert track)")

    os.makedirs(out_dir, exist_ok=True)
    save_npz(
        out_path,
        obs=obs[:, 0],
        action=action[:, 0],
        reward=reward[:, 0],
        rl_sum=adv_scores[:, 0],
        checkpoint_step=checkpoint_step,
    )
    print(f"\nSaved -> {out_path}")
    print(f"  obs    : {obs[:, 0].shape}")
    print(f"  action : {action[:, 0].shape}")
    print(f"  reward : {reward[:, 0].shape}")


if __name__ == "__main__":
    main()
