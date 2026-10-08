"""
Generates one concrete config.yaml for standalone Cumulative E-stop CPL
training, given the resolved per-env label paths (and an optional per-branch
subsample_n). Exists because scripts/train.py's --set dotted-path mechanism
can't index into a YAML list (dataset_kwargs.components is a list) -- see
research/utils/config.py / scripts/train.py's `obj = obj[k]` walk, which
only supports dict keys, not list indices. Writing a real file sidesteps
that limitation entirely, the same way the mixed-manifest generators do.

Usage:
    python3 scripts/generate_cum_estop_standalone_config.py \\
        --env-name mw_button-press-v2 \\
        --stop-correction-path /scratch/.../stop_correction_labels.npz \\
        --no-stop-demo-path    /scratch/.../no_stop_demo_labels.npz \\
        --output /path/to/run/config.yaml \\
        [--subsample-n 5000] [--subsample-seed 0] [--total-steps 250000]
"""
import argparse
import os

import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-name", type=str, required=True)
    parser.add_argument("--stop-correction-path", type=str, required=True)
    parser.add_argument("--no-stop-demo-path", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--subsample-n", type=int, default=None)
    parser.add_argument("--subsample-seed", type=int, default=0)
    parser.add_argument("--total-steps", type=int, default=None)
    args = parser.parse_args()

    def component(name, path):
        dataset_kwargs = {"path": path, "capacity": None}
        if args.subsample_n is not None:
            dataset_kwargs["subsample_n"] = args.subsample_n
            dataset_kwargs["subsample_seed"] = args.subsample_seed
        return {"name": name, "dataset_class": "CorrBuffer", "weight": 0.5, "dataset_kwargs": dataset_kwargs}

    config = {
        "import": os.path.join(REPO_ROOT, "configs/mw_state_dense/cum_estop_cpl.yaml"),
        "eval_env": args.env_name,
        "dataset_kwargs": {
            "batch_size": 96,
            "components": [
                component("stop_correction", args.stop_correction_path),
                component("no_stop_demo", args.no_stop_demo_path),
            ],
        },
    }
    if args.total_steps is not None:
        # Config.load's `import` merge is a SHALLOW dict.update() -- restate
        # every trainer_kwargs field from cum_estop_cpl.yaml explicitly, not
        # just total_steps, or this silently drops train_dataloader_kwargs
        # and re-enables PyTorch's default batch_size=1 auto-collation (see
        # generate_mixed_cpl_manifest.py's build_config docstring for the
        # exact incident this caused before).
        config["trainer_kwargs"] = {
            "total_steps": args.total_steps,
            "log_freq": 500,
            "profile_freq": 500,
            "eval_freq": 5000,
            "eval_fn": "eval_policy",
            "eval_kwargs": {"num_ep": 25},
            "loss_metric": "reward",
            "train_dataloader_kwargs": {"num_workers": 0, "batch_size": None},
        }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        yaml.dump(config, f, sort_keys=False)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
