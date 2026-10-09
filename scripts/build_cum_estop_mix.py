"""
Build a reproducible, pre-shuffled 50/50 mix of the Cumulative E-stop
model's two branches (stop_correction, no_stop_demo) into ONE flat
CorrBuffer-compatible file, for both CPL and PIQL training. A single file
(rather than MixedFeedbackBuffer's per-component sub-batches) is needed
because PIQL has no component-weighted mixing mechanism, and because the
rows should be genuinely interleaved, not just two separately-shuffled
named sub-batches drawn in lockstep each step.

Usage:
    python3 scripts/build_cum_estop_mix.py \\
        --stop-correction /scratch/.../stop_correction_labels.npz \\
        --no-stop-demo    /scratch/.../no_stop_demo_labels.npz \\
        --n-per-branch 5000 --seed 0 \\
        --output /scratch/.../cum_estop_mix10k.npz
"""
import argparse

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stop-correction", type=str, required=True)
    parser.add_argument("--no-stop-demo", type=str, required=True)
    parser.add_argument("--n-per-branch", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    stop = np.load(args.stop_correction)
    demo = np.load(args.no_stop_demo)

    rng = np.random.default_rng(args.seed)

    def subsample(data, n, source_label):
        N = data["obs"].shape[0]
        assert n <= N, f"--n-per-branch={n} exceeds available {N} rows in {data}"
        keep = np.sort(rng.choice(N, size=n, replace=False))
        return data["obs"][keep], data["action"][keep], data["reward"][keep], \
            np.full(n, source_label, dtype=np.int32)

    stop_obs, stop_action, stop_reward, stop_source = subsample(stop, args.n_per_branch, 0)
    demo_obs, demo_action, demo_reward, demo_source = subsample(demo, args.n_per_branch, 1)

    obs = np.concatenate([stop_obs, demo_obs], axis=0)
    action = np.concatenate([stop_action, demo_action], axis=0)
    reward = np.concatenate([stop_reward, demo_reward], axis=0)
    source = np.concatenate([stop_source, demo_source], axis=0)

    total = len(obs)
    shuffle = rng.permutation(total)  # interleave the two branches, not two contiguous blocks
    obs, action, reward, source = obs[shuffle], action[shuffle], reward[shuffle], source[shuffle]

    with open(args.output, "wb") as f:
        np.savez(f, obs=obs, action=action, reward=reward, source=source)

    print(f"stop_correction: {args.n_per_branch}  no_stop_demo: {args.n_per_branch}  "
          f"total: {total} (shuffled) -> {args.output}")
    print(f"source==0 (stop_correction) fraction in first 10 rows: {source[:10]}")


if __name__ == "__main__":
    main()
