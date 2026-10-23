"""
Push the post-hoc RRC results to WandB (project $WANDB_PROJECT, default rrc-sac-2m; entity $WANDB_ENTITY).

    # one finished SAC run: resume its WandB run, add hacking metrics, plot and training-episode curves
    python scripts/rrc_wandb_sync.py --stage sac_run --run_dir runs/rrc_sac/<task>/<type>/seed<s>/sac_best
    # summary run "rrc_summary/reward_models": reward-diagnostics + label-consistency tables
    python scripts/rrc_wandb_sync.py --stage reward --runs_root runs/rrc_sac
    # summary run "rrc_summary/final": per-run results, per (task, arm) table, learning-curve images
    python scripts/rrc_wandb_sync.py --stage final --runs_root runs/rrc_sac --results_dir runs/rrc_sac/_results

Only finished SAC runs (training_complete) are synced, so this never writes into a run that a
training process is still logging to.
"""

import argparse
import csv
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROJECT = os.getenv("WANDB_PROJECT", "rrc-sac-2m")
ENTITY = os.getenv("WANDB_ENTITY") or None


def _scalars(d, prefix):
    out = {}
    for k, v in d.items():
        key = f"{prefix}/{k}"
        if isinstance(v, dict):
            out.update(_scalars(v, key))
        elif isinstance(v, bool):
            out[key] = int(v)
        elif isinstance(v, (int, float)):
            out[key] = v
    return out


def _read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def _table(wandb, rows):
    """Native wandb.Table when possible; wandb.Table needs a working pandas, so fall back to an HTML table."""
    if not rows:
        return None
    cols = list(rows[0].keys())

    def conv(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return v

    data = [[conv(r.get(c)) for c in cols] for r in rows]
    try:
        return wandb.Table(columns=cols, data=data)
    except ImportError:
        fmt = lambda v: f"{v:.4g}" if isinstance(v, float) else ("" if v is None else str(v))
        head = "".join(f"<th>{c}</th>" for c in cols)
        body = "".join("<tr>" + "".join(f"<td>{fmt(v)}</td>" for v in row) + "</tr>" for row in data)
        return wandb.Html(f"<table border='1' style='border-collapse:collapse;font-size:12px'><tr>{head}</tr>{body}</table>")


def _line_png(path, xs, ys, keys, title, xlabel="env steps (M)"):
    """Matplotlib line plot saved to `path` (uploaded as wandb.Image; wandb.plot needs pandas)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 3.6))
    for y, k in zip(ys, keys):
        ax.plot(xs, y, label=k, lw=1.2)
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def sync_sac_run(wandb, run_dir):
    if not os.path.exists(os.path.join(run_dir, "training_complete")):
        print(f"[wandb-sync] {run_dir} not finished, skipping")
        return
    hp = os.path.join(run_dir, "hacking.json")
    if not os.path.exists(hp):
        print(f"[wandb-sync] {run_dir} has no hacking.json (run scripts/rrc_hacking.py first), skipping")
        return
    with open(hp) as f:
        h = json.load(f)
    id_file = os.path.join(run_dir, "wandb_run_id.txt")
    run_id = open(id_file).read().strip() if os.path.exists(id_file) else None
    rmap = json.load(open(os.path.join(run_dir, "reward_map.json")))
    name = f"rrc_sac/{h['task']}/{h['arm']}/s{h['seed']}"
    run = wandb.init(entity=ENTITY, project=PROJECT, id=run_id, resume="allow" if run_id else None, name=name,
                     group=f"rrc_sac/{h['task']}/{h['arm']}", dir=os.path.join(run_dir, ".."))
    summ = {}
    summ.update(_scalars(h["final"], "hacking/final"))
    summ.update(_scalars(h["overopt_drop"], "hacking/overopt_drop"))
    summ.update(_scalars(h["second_half"], "hacking/second_half"))
    summ.update(_scalars(h["flag"], "hacking/flag"))
    summ.update(_scalars(h["refs"], "hacking/refs"))
    summ["hacking/max_G"] = h["max_G"]
    if "disagreement_ratio" in h:
        summ["hacking/disagreement_ratio_final"] = h["disagreement_ratio"]["final"]
        summ["hacking/disagreement_ratio_max"] = h["disagreement_ratio"]["max"]
    if "final_train" in h:
        summ.update(_scalars(h["final_train"], "hacking/final_train"))
    summ["reward_map/a"], summ["reward_map/b"] = rmap["a"], rmap["b"]
    run.summary.update(summ)

    payload = {}
    png = os.path.join(run_dir, "hacking.png")
    if os.path.exists(png):
        payload["hacking/plot"] = wandb.Image(png)
    c = h["curves"]
    xs = [st / 1e6 for st in c["step"]]
    png_dir = os.path.join(run_dir, "wandb_plots")
    os.makedirs(png_dir, exist_ok=True)
    payload["hacking/normalized_returns"] = wandb.Image(_line_png(
        os.path.join(png_dir, "normalized_returns.png"), xs, [c["L_norm"], c["T_norm"], c["success"]],
        ["learned return (norm.)", "true return (norm.)", "eval success"], "eval: learned vs true return (0 = random, 1 = oracle)"))
    payload["hacking/G_curve"] = wandb.Image(_line_png(
        os.path.join(png_dir, "G.png"), xs, [c["G"]], ["G = learned - true"], "hacking gap G"))
    snaps = h.get("agreement_on_policy_states", {})
    if snaps:
        steps = sorted(snaps, key=int)
        payload["hacking/learned_true_step_corr"] = wandb.Image(_line_png(
            os.path.join(png_dir, "step_corr.png"), [int(st) / 1e6 for st in steps],
            [[snaps[st]["step_corr"]["pearson"] for st in steps]], ["corr(learned r, true r) per step"],
            "learned vs true reward on the policy's own eval states"))
    tr_path = os.path.join(run_dir, "train_episodes.csv")
    if os.path.exists(tr_path):
        rows = _read_csv(tr_path)
        x = [int(r["env_step"]) / 1e6 for r in rows]
        payload["train_episodes/returns"] = wandb.Image(_line_png(
            os.path.join(png_dir, "train_returns.png"), x,
            [[float(r["true_return"]) for r in rows], [float(r["learned_return"]) for r in rows]],
            ["true return", "learned return"], "training episodes: true vs learned return"))
        payload["train_episodes/success"] = wandb.Image(_line_png(
            os.path.join(png_dir, "train_success.png"), x, [[float(r["success"]) for r in rows]], ["success"],
            "training episodes: success"))
    run.log(payload)
    run.finish()
    print(f"[wandb-sync] synced {name}")


def sync_reward(wandb, runs_root, results_dir):
    run = wandb.init(entity=ENTITY, project=PROJECT, name="rrc_summary/reward_models", job_type="summary",
                     group="rrc_summary", dir=results_dir)
    csv_path = os.path.join(results_dir, "rrc_reward_diagnostics.csv")
    if os.path.exists(csv_path):
        run.log({"reward_models/diagnostics": _table(wandb, _read_csv(csv_path))})
    lc_rows = []
    for p in sorted(glob.glob(os.path.join(runs_root, "*", "*", "label_consistency.json"))):
        task, ftype = p.split(os.sep)[-3], p.split(os.sep)[-2]
        lc = json.load(open(p))
        lc_rows.append(dict(task=task, type=ftype, labeler_oracle=lc.get("labeler_oracle"),
                            labeler_rule_acc=lc.get("oracle_rule_acc"), partial_return_acc=lc.get("partial_return_acc_all"),
                            corr_with_stored_scores=(lc.get("oracle_rule_vs_stored_scores") or {}).get("pearson")))
    if lc_rows:
        run.log({"reward_models/label_consistency": _table(wandb, lc_rows)})
    run.finish()
    print("[wandb-sync] synced rrc_summary/reward_models")


def sync_final(wandb, runs_root, results_dir):
    run = wandb.init(entity=ENTITY, project=PROJECT, name="rrc_summary/final", job_type="summary",
                     group="rrc_summary", dir=results_dir)
    payload = {}
    res_csv = os.path.join(results_dir, "rrc_sac_results.csv")
    if os.path.exists(res_csv):
        rows = _read_csv(res_csv)
        payload["results/per_run"] = _table(wandb, rows)
        groups = {}
        for r in rows:
            groups.setdefault((r["task"], r["arm"]), []).append(r)
        agg = []
        for (task, arm), rs in sorted(groups.items()):
            f = [float(r["final_success"]) for r in rs]
            g = [float(r["final_G"]) for r in rs if r.get("final_G") not in (None, "", "None")]
            m = lambda v: sum(v) / len(v) if v else float("nan")
            sd = lambda v: (sum((x - m(v)) ** 2 for x in v) / (len(v) - 1)) ** 0.5 if len(v) > 1 else 0.0
            agg.append(dict(task=task, arm=arm, n=len(rs), final_success_mean=m(f), final_success_std=sd(f),
                            final_G_mean=m(g), hacking_flags=sum(r.get("hacking_flag") == "True" for r in rs)))
        payload["results/per_task_arm"] = _table(wandb, agg)
    diag = os.path.join(results_dir, "rrc_reward_diagnostics.csv")
    if os.path.exists(diag):
        payload["reward_models/diagnostics"] = _table(wandb, _read_csv(diag))
    for png in sorted(glob.glob(os.path.join(results_dir, "rrc_sac_curves_*.png"))):
        payload[f"curves/{os.path.basename(png)[len('rrc_sac_curves_'):-4]}"] = wandb.Image(png)
    run.log(payload)
    run.finish()
    print("[wandb-sync] synced rrc_summary/final")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["sac_run", "reward", "final"], required=True)
    ap.add_argument("--run_dir", nargs="*", default=[])
    ap.add_argument("--runs_root", default="runs/rrc_sac")
    ap.add_argument("--results_dir", default=None)
    args = ap.parse_args()
    import wandb

    results_dir = args.results_dir or os.path.join(args.runs_root, "_results")
    os.makedirs(results_dir, exist_ok=True)
    if args.stage == "sac_run":
        for d in args.run_dir:
            sync_sac_run(wandb, d)
    elif args.stage == "reward":
        sync_reward(wandb, args.runs_root, results_dir)
    else:
        sync_final(wandb, args.runs_root, results_dir)


if __name__ == "__main__":
    main()
