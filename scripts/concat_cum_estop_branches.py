"""
Concatenate the Cumulative E-stop model's two branch files (stop_correction,
no_stop_demo -- both already in CorrBuffer's (obs, action, reward) K=2
schema) into one flat file for standalone PIQL training, which has no
component-weighted mixing mechanism (unlike MixedCPL -- see
configs/mw_state_dense/cum_estop_piql.yaml).

Usage:
    python3 scripts/concat_cum_estop_branches.py \\
        --stop-correction datasets/cum_estop_labels/mw_drawer-open-v2/stop_correction_labels.npz \\
        --no-stop-demo    datasets/cum_estop_labels/mw_drawer-open-v2/no_stop_demo_labels.npz \\
        --output          datasets/cum_estop_labels/mw_drawer-open-v2/cum_estop_concat.npz
"""
import argparse

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stop-correction", type=str, required=True)
    parser.add_argument("--no-stop-demo", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    a = np.load(args.stop_correction)
    b = np.load(args.no_stop_demo)

    out = {}
    for key in ("obs", "action", "reward"):
        out[key] = np.concatenate([a[key], b[key]], axis=0)

    with open(args.output, "wb") as f:
        np.savez(f, **out)

    print(f"stop_correction: {len(a['obs'])}  no_stop_demo: {len(b['obs'])}  "
          f"total: {len(out['obs'])} -> {args.output}")


if __name__ == "__main__":
    main()
