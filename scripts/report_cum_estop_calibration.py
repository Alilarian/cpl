"""
Re-derive the stop-rate-vs-H calibration table from an already-computed
cum_estop_calibration.npz (scripts/tune_cum_estop_threshold.py's output) --
no oracle/simulator access needed, since the npz already holds each
segment's full cumulative-deficit trajectory. Lets you inspect the
calibration table after downloading the npz from CHPC, or re-query it with a
different --h-grid than whatever was printed at generation time.

Usage:
    python3 scripts/report_cum_estop_calibration.py \\
        --calibration datasets/cum_estop_labels/mw_button-press-v2/cum_estop_calibration.npz \\
        [--h-grid 5,10,15,20]
"""
import argparse

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", type=str, required=True)
    parser.add_argument("--h-grid", type=str, default=None,
                         help="comma-separated H values; default: auto quantile grid")
    args = parser.parse_args()

    data = np.load(args.calibration)
    cumulative_all = data["cumulative"]
    N, T = cumulative_all.shape
    final_evidence = cumulative_all[:, -1]

    print(f"Calibration file: {args.calibration}")
    print(f"{N} segments, T={T}")
    print(f"Final cumulative evidence: mean={final_evidence.mean():.3f} "
          f"median={np.median(final_evidence):.3f} max={final_evidence.max():.3f}")

    if args.h_grid:
        h_values = [float(x) for x in args.h_grid.split(",")]
    else:
        nonzero = final_evidence[final_evidence > 0]
        if len(nonzero) == 0:
            print("All final cumulative evidence is zero -- cannot calibrate a positive H.")
            return
        h_values = sorted(set(np.percentile(nonzero, [10, 25, 50, 75, 90]).round(4)))

    print(f"\n{'H':>10} {'stop_rate':>10} {'mean_tau':>10} {'mean_tau/T':>12} {'mean_horizon':>14}")
    for H in h_values:
        stop_mask = final_evidence >= H
        taus = []
        for row in cumulative_all[stop_mask]:
            crossed = np.nonzero(row >= H)[0]
            taus.append(int(crossed[0]) if len(crossed) else T)
        taus = np.array(taus)
        stop_rate = stop_mask.mean()
        mean_tau = taus.mean() if len(taus) else float("nan")
        print(f"{H:10.4f} {stop_rate*100:9.1f}% {mean_tau:10.1f} {mean_tau/T:12.3f} {T-mean_tau:14.1f}")


if __name__ == "__main__":
    main()
