"""
Aggregate RRC + SAC results: final success, learning curves, reward-hacking table.

    python scripts/aggregate_rrc_sac.py --runs_root runs/rrc_sac --out docs/rrc_sac_results.md [--tasks ...]

Per (task, arm): final success = mean of the last 3 evals of each run, reported as mean ± std over
seeds (the headline with 3 seeds), plus the hacking summary from hacking.json (run
scripts/rrc_hacking.py first). Across tasks per arm: IQM of all (task, seed) final successes with a
95% stratified bootstrap CI (seeds resampled within task). Writes <out>, <out>.csv and
<out dir>/rrc_sac_curves_<task>.png.
"""

import argparse
import csv
import glob
import json
import os
from collections import defaultdict

import numpy as np
from scipy import stats

FINAL_EVALS = 3


def arm_of(run_dir: str) -> str:
    leaf = os.path.basename(run_dir)
    kind = os.path.basename(os.path.dirname(os.path.dirname(run_dir)))
    if kind in ("true_reward", "untrained"):
        return kind
    return f"{kind}@{leaf[len('sac_'):]}" if leaf.startswith("sac_") else kind


def eval_curve(run_dir):
    with open(os.path.join(run_dir, "log.csv")) as f:
        rows = [r for r in csv.DictReader(f) if r.get("eval/success") not in (None, "")]
    step = np.array([float(r["step"]) for r in rows])
    succ = np.array([float(r["eval/success"]) for r in rows])
    true = np.array([float(r["eval/reward"]) for r in rows])
    order = np.argsort(step, kind="stable")
    return step[order], succ[order], true[order]


def iqm(x):
    return float(stats.trim_mean(np.asarray(x, float), 0.25))


def stratified_bootstrap_iqm(by_task, n_boot=10_000, seed=0):
    rng = np.random.RandomState(seed)
    vals = [np.asarray(v, float) for v in by_task.values() if len(v)]
    boots = [iqm(np.concatenate([v[rng.randint(0, len(v), len(v))] for v in vals])) for _ in range(n_boot)]
    return np.percentile(boots, [2.5, 97.5])


def ms(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return "–"
    return f"{v.mean():.2f}" if len(v) == 1 else f"{v.mean():.2f} ± {v.std(ddof=1):.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", default="runs/rrc_sac")
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--out", default="docs/rrc_sac_results.md")
    ap.add_argument("--oracle_dir", default="/scratch/general/vast/u1472210/oracle_sac_1m/{task}",
                    help="original oracle run, shown as a reference row")
    args = ap.parse_args()

    tasks = args.tasks or sorted(d for d in os.listdir(args.runs_root) if not d.startswith("_"))
    runs = defaultdict(list)  # (task, arm) -> list of dicts
    for task in tasks:
        for log in glob.glob(os.path.join(args.runs_root, task, "*", "seed*", "sac*", "log.csv")):
            run = os.path.dirname(log)
            step, succ, true = eval_curve(run)
            if len(step) == 0:
                continue
            seed = int(os.path.basename(os.path.dirname(run))[len("seed"):])
            h = None
            if os.path.exists(os.path.join(run, "hacking.json")):
                with open(os.path.join(run, "hacking.json")) as f:
                    h = json.load(f)
            runs[(task, arm_of(run))].append(dict(seed=seed, step=step, succ=succ, true=true, hacking=h,
                                                  final=float(succ[-FINAL_EVALS:].mean()), last_step=float(step[-1]),
                                                  complete=os.path.exists(os.path.join(run, "training_complete"))))

    lines = ["# RRC + SAC results", "",
             f"Final success = mean of the last {FINAL_EVALS} evals (20 episodes each, every 10k steps), "
             "mean ± std over seeds. G = normalized learned − true return (0 = random, 1 = oracle policy, "
             "under the same reward); flag = §3.6 hacking test.", "",
             "| task | arm | seeds (complete) | last step | final success | final true return | final G | max G | "
             "success drop | hacking flags | disagreement ratio (final) | step corr learned/true (last snapshot) |",
             "|" + "---|" * 12]
    csv_rows = []
    for (task, arm), rs in sorted(runs.items()):
        hs = [r["hacking"] for r in rs if r["hacking"]]
        g = lambda f: [f(h) for h in hs]
        last_corr = []
        for h in hs:
            snaps = h.get("agreement_on_policy_states", {})
            if snaps:
                last_corr.append(snaps[max(snaps, key=int)]["step_corr"]["pearson"])
        cells = [task, arm, f"{len(rs)} ({sum(r['complete'] for r in rs)})", f"{min(r['last_step'] for r in rs):.0f}",
                 ms([r["final"] for r in rs]), ms([r["true"][-FINAL_EVALS:].mean() for r in rs]),
                 ms(g(lambda h: h["final"]["G"])), ms(g(lambda h: h["max_G"])), ms(g(lambda h: h["overopt_drop"]["success"])),
                 f"{sum(h['flag']['hacking'] for h in hs)}/{len(hs)}" if hs else "–",
                 ms(g(lambda h: h.get("disagreement_ratio", {}).get("final", np.nan))), ms(last_corr)]
        lines.append("| " + " | ".join(cells) + " |")
        for r in rs:
            h = r["hacking"] or {}
            csv_rows.append(dict(task=task, arm=arm, seed=r["seed"], final_success=r["final"], last_step=r["last_step"],
                                 complete=r["complete"], final_G=h.get("final", {}).get("G"), max_G=h.get("max_G"),
                                 hacking_flag=h.get("flag", {}).get("hacking")))
    for task in tasks:
        od = args.oracle_dir.replace("{task}", task)
        if os.path.exists(os.path.join(od, "log.csv")):
            s, su, tr = eval_curve(od)
            lines.append(f"| {task} | oracle SAC (original run, seed 0) | 1 | {s[-1]:.0f} | {su[-FINAL_EVALS:].mean():.2f} | "
                         f"{tr[-FINAL_EVALS:].mean():.2f} | – | – | – | – | – | – |")

    arms = sorted({a for _, a in runs})
    if len(tasks) > 1:
        lines += ["", "## Across tasks (IQM of final success, 95% stratified bootstrap CI)", "",
                  "| arm | runs | IQM | 95% CI | mean |", "|---|---|---|---|---|"]
        for arm in arms:
            by_task = {t: [r["final"] for r in runs.get((t, arm), [])] for t in tasks}
            allv = [v for vs in by_task.values() for v in vs]
            if not allv:
                continue
            lo, hi = stratified_bootstrap_iqm({t: v for t, v in by_task.items() if v})
            lines.append(f"| {arm} | {len(allv)} | {iqm(allv):.2f} | [{lo:.2f}, {hi:.2f}] | {np.mean(allv):.2f} |")

    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        for task in tasks:
            fig, ax = plt.subplots(figsize=(7, 4))
            for arm in arms:
                rs = runs.get((task, arm), [])
                if not rs:
                    continue
                n = min(len(r["step"]) for r in rs)
                S = rs[0]["step"][:n] / 1e6
                Y = np.stack([r["succ"][:n] for r in rs])
                ax.plot(S, Y.mean(0), label=f"{arm} (n={len(rs)})")
                if len(rs) > 1:
                    ax.fill_between(S, Y.mean(0) - Y.std(0), Y.mean(0) + Y.std(0), alpha=0.15)
            ax.set_xlabel("env steps (M)"); ax.set_ylabel("eval success"); ax.set_title(task)
            ax.legend(fontsize=7); fig.tight_layout()
            p = os.path.join(out_dir, f"rrc_sac_curves_{task}.png")
            fig.savefig(p, dpi=120); plt.close(fig)
            lines += ["", f"![{task}]({os.path.basename(p)})"]
    except Exception as e:
        print(f"[aggregate] plotting skipped: {e}")

    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.splitext(args.out)[0] + ".csv", "w", newline="") as f:
        if csv_rows:
            w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            w.writeheader()
            w.writerows(csv_rows)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
