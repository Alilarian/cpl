"""
Phase 4 of the Cumulative E-stop model: select the intermediate-quality
checkpoint bank Pi_M (for corrections) and the weak checkpoint bank Pi_B
(for no-stop negatives) for one MetaWorld env, from the oracle training
run's own log.csv -- no new rollout benchmarking needed, since eval/success
and eval/reward are already logged there every few thousand steps.

Usage:
    python3 scripts/select_checkpoint_bands.py --run-dir datasets/mw_de_labels/mw_button-press-v2/oracle \\
        --output datasets/mw_de_labels/mw_button-press-v2/checkpoint_bands.json

Band selection (defaults, override via flags):
    weak (Pi_B)         : eval/success <= --weak-max-success   (default 0.1)
    intermediate (Pi_M) : --mid-min-success <= eval/success <= --mid-max-success
                          (default 0.3 to 0.85)
    best (excluded)     : the checkpoint with the single highest eval/success
                          (ties broken by eval/reward), always excluded from Pi_M

Each selected log row's `step` is mapped to the nearest existing
model_<step>.pt file actually on disk (log steps don't always land exactly
on a saved checkpoint step).
"""
import argparse
import csv
import glob
import json
import os
import re


def available_checkpoint_steps(run_dir: str):
    steps = []
    for path in glob.glob(os.path.join(run_dir, "model_*.pt")):
        m = re.match(r"model_(\d+)\.pt$", os.path.basename(path))
        if m:
            steps.append(int(m.group(1)))
    return sorted(steps)


def nearest_checkpoint(step: int, available: list) -> int:
    return min(available, key=lambda s: abs(s - step))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--weak-max-success", type=float, default=0.1)
    parser.add_argument("--mid-min-success", type=float, default=0.3)
    parser.add_argument("--mid-max-success", type=float, default=0.85)
    parser.add_argument("--max-weak", type=int, default=5, help="cap on Pi_B size")
    parser.add_argument("--max-mid", type=int, default=10, help="cap on Pi_M size")
    args = parser.parse_args()

    log_path = os.path.join(args.run_dir, "log.csv")
    with open(log_path) as f:
        rows = list(csv.DictReader(f))

    available = available_checkpoint_steps(args.run_dir)
    assert available, f"no model_<step>.pt files found under {args.run_dir}"

    entries = []
    for row in rows:
        step = int(float(row["step"]))
        success = float(row["eval/success"])
        reward = float(row["eval/reward"])
        entries.append({"step": step, "success": success, "reward": reward})
    entries.sort(key=lambda e: e["step"])

    best = max(entries, key=lambda e: (e["success"], e["reward"]))
    best_checkpoint = f"model_{nearest_checkpoint(best['step'], available)}.pt"

    weak = [e for e in entries if e["success"] <= args.weak_max_success]
    mid = [
        e for e in entries
        if args.mid_min_success <= e["success"] <= args.mid_max_success
        and nearest_checkpoint(e["step"], available) != nearest_checkpoint(best["step"], available)
    ]

    def dedup_checkpoints(band_entries, cap):
        seen = set()
        out = []
        for e in band_entries:
            ckpt_step = nearest_checkpoint(e["step"], available)
            if ckpt_step in seen:
                continue
            seen.add(ckpt_step)
            out.append({"checkpoint": f"model_{ckpt_step}.pt", "step": e["step"], "success": e["success"]})
        return out[:cap]

    weak_bank = dedup_checkpoints(weak, args.max_weak)
    mid_bank = dedup_checkpoints(mid, args.max_mid)

    assert mid_bank, (
        f"no checkpoints found with eval/success in "
        f"[{args.mid_min_success}, {args.mid_max_success}] -- adjust --mid-min-success/--mid-max-success. "
        f"Observed success values: {sorted(set(round(e['success'], 3) for e in entries))}"
    )
    assert weak_bank, (
        f"no checkpoints found with eval/success <= {args.weak_max_success} -- "
        f"adjust --weak-max-success. Observed success values: "
        f"{sorted(set(round(e['success'], 3) for e in entries))}"
    )

    result = {
        "run_dir": args.run_dir,
        "best": {"checkpoint": best_checkpoint, "step": best["step"], "success": best["success"]},
        "intermediate": mid_bank,
        "weak": weak_bank,
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

    print(f"best (excluded): {result['best']}")
    print(f"intermediate (Pi_M, {len(mid_bank)}): {[b['checkpoint'] for b in mid_bank]}")
    print(f"weak (Pi_B, {len(weak_bank)}): {[b['checkpoint'] for b in weak_bank]}")
    print(f"Saved -> {args.output}")


if __name__ == "__main__":
    main()
