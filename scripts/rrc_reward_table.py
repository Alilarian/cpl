"""
Reward-model diagnostics table (docs/rrc_sac_plan.md section 3.4) from every report.json under --runs_root.

    python scripts/rrc_reward_table.py --runs_root runs/rrc_sac --out results/rrc_reward_diagnostics.md

One row per (task, type, export), mean +- std over seeds; label-consistency columns come from
<task>/<type>/label_consistency.json. Also writes the per-seed rows to <out>.csv.
"""

import argparse
import csv
import glob
import json
import os
from collections import defaultdict

import numpy as np

COLS = [
    ("val NLL", lambda r: r["heldout"]["val"]["nll_ens"]),
    ("val acc", lambda r: r["heldout"]["val"]["acc_ens"]),
    ("train NLL", lambda r: r["heldout"]["train"]["nll_ens"]),
    ("GT corr val (r)", lambda r: r["heldout"]["gt_corr_val"]["all"]["pearson"]),
    ("GT corr val (rho)", lambda r: r["heldout"]["gt_corr_val"]["all"]["spearman"]),
    ("GT corr within act", lambda r: r["heldout"]["gt_corr_val"]["within_act"]["pearson"]),
    ("offdist ep corr (r)", lambda r: r["offdist"]["episode_corr_undiscounted"]["pearson"]),
    ("offdist ep corr (rho)", lambda r: r["offdist"]["episode_corr_undiscounted"]["spearman"]),
    ("offdist seg corr", lambda r: r["offdist"]["segment_corr"]["pearson"]),
    ("success AUC", lambda r: r["offdist"]["success_auc_pred"]),
    ("success AUC (true return)", lambda r: r["offdist"]["success_auc_true"]),
    ("saturation val (max member)", lambda r: max(r["heldout"]["steps_val"]["saturation"])),
    ("saturation offdist (max member)", lambda r: max(r["offdist"]["saturation"])),
    ("disagreement val", lambda r: r["heldout"]["steps_val"]["disagreement"]),
    ("disagreement offdist", lambda r: r["offdist"]["disagreement"]),
]
LC_COLS = [("labeler-rule acc", "oracle_rule_acc"), ("partial-return acc (ceiling)", "partial_return_acc_all"),
           ("labeler oracle", "labeler_oracle")]


def _get(fn, rep):
    try:
        return float(fn(rep))
    except (KeyError, TypeError, ValueError):
        return float("nan")


def fmt(vals):
    v = np.array(vals, float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return "–"
    return f"{v.mean():.3f}" if len(v) == 1 else f"{v.mean():.3f} ± {v.std(ddof=1):.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", default="runs/rrc_sac")
    ap.add_argument("--out", default="results/rrc_reward_diagnostics.md")
    ap.add_argument("--exports", nargs="*", default=["best", "epoch3x"])
    args = ap.parse_args()

    groups = defaultdict(list)
    flat = []
    for rp in sorted(glob.glob(os.path.join(args.runs_root, "*", "*", "seed*", "reward", "report.json"))):
        with open(rp) as f:
            rep = json.load(f)
        for name, r in rep["exports"].items():
            if name not in args.exports:
                continue
            vals = {c: _get(fn, r) for c, fn in COLS}
            groups[(rep["task"], rep["arm"], name)].append(vals)
            flat.append(dict(task=rep["task"], type=rep["arm"], export=name, seed=rep["seed"],
                             member_epochs=r.get("member_epochs"), **vals))
    lines = ["# RRC reward-model diagnostics", "",
             "Mean ± std over seeds. Predicted returns use the standardized ensemble reward. "
             "Labeler-rule acc should be ≈ 1.0; partial-return acc is the ceiling of the RRC likelihood "
             "(labels were generated from oracle advantage, not partial return).", ""]
    header = ["task", "type", "export", "n"] + [c for c, _ in COLS] + [c for c, _ in LC_COLS]
    lines += ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for (task, ftype, name), rows in sorted(groups.items()):
        lc_path = os.path.join(args.runs_root, task, ftype, "label_consistency.json")
        lc = json.load(open(lc_path)) if os.path.exists(lc_path) else {}
        lc_vals = []
        for _, k in LC_COLS:
            v = lc.get(k)
            lc_vals.append(f"{v:.3f}" if isinstance(v, float) else (os.path.basename(os.path.dirname(v.split(":")[0])) + "/" + os.path.basename(v.split(":")[0]) if isinstance(v, str) else "–"))
        cells = [task, ftype, name, str(len(rows))] + [fmt([r[c] for r in rows]) for c, _ in COLS] + lc_vals
        lines.append("| " + " | ".join(cells) + " |")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.splitext(args.out)[0] + ".csv", "w", newline="") as f:
        if flat:
            w = csv.DictWriter(f, fieldnames=list(flat[0].keys()))
            w.writeheader()
            w.writerows(flat)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
