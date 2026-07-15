import argparse
import os
import subprocess

from research.utils.config import Config


def try_wandb_setup(path, config):
    try:
        import wandb
    except ImportError:
        return

    # Accept the key from the environment variable (SLURM / CI) or fall back
    # to whatever wandb already has stored (netrc / wandb login).
    api_key = os.getenv("WANDB_API_KEY") or wandb.api.api_key
    if not api_key:
        print("[wandb] No API key found — skipping. Set WANDB_API_KEY or run `wandb login`.")
        return

    project_dir = os.path.dirname(os.path.dirname(__file__))
    project_name = os.getenv("WANDB_PROJECT", os.path.basename(project_dir))
    wandb_dir = os.path.join(os.path.dirname(project_dir), "wandb")
    os.makedirs(wandb_dir, exist_ok=True)

    # Resume an existing run if a run-ID was saved from a previous attempt.
    # This keeps metrics continuous across walltime-limited restarts.
    run_id_file = os.path.join(path, "wandb_run_id.txt")
    run_id = None
    if os.path.exists(run_id_file):
        with open(run_id_file) as f:
            run_id = f.read().strip() or None

    # WANDB_RUN_NAME allows the sbatch to set a meaningful name that includes
    # the env, type, budget, and seed (e.g. "mw_drawer-open-v2/pref_b000050_s0").
    run_name = os.getenv("WANDB_RUN_NAME", os.path.basename(path))
    group    = os.getenv("WANDB_GROUP", None)

    run = wandb.init(
        project=project_name,
        name=run_name,
        group=group,
        config=config.flatten(separator="-"),
        dir=wandb_dir,
        id=run_id,
        resume="allow",
    )

    # Persist the run ID so future restarts resume this exact run.
    if run is not None:
        with open(run_id_file, "w") as f:
            f.write(run.id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, default=None)
    parser.add_argument("--path", "-p", type=str, default=None)
    parser.add_argument("--device", "-d", type=str, default="auto")
    parser.add_argument("--seed", "-s", type=int, default=None, help="Override the seed in the config")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="Override a config value using dot-notation: e.g. "
                             "--set dataset_kwargs.path=foo.npz --set trainer_kwargs.total_steps=50000")
    args = parser.parse_args()

    config = Config.load(args.config)
    if args.seed is not None:
        config["seed"] = args.seed

    for kv in args.set:
        dotted, _, raw = kv.partition("=")
        keys = dotted.split(".")
        # Cast: try int, float, then bool/null keywords, else keep as string
        val: object
        for cast in (int, float):
            try:
                val = cast(raw)
                break
            except ValueError:
                pass
        else:
            val = {"null": None, "none": None, "true": True, "false": False}.get(raw.lower(), raw)
        obj = config.config
        for k in keys[:-1]:
            obj = obj[k]
        obj[keys[-1]] = val
    print(config)
    os.makedirs(args.path, exist_ok=True)  # Allow resuming from existing directories
    try_wandb_setup(args.path, config)
    config.save(args.path)  # Save the config
    # save the git hash
    process = subprocess.Popen(["git", "rev-parse", "HEAD"], shell=False, stdout=subprocess.PIPE)
    git_head_hash = process.communicate()[0].strip()
    with open(os.path.join(args.path, "git_hash.txt"), "wb") as f:
        f.write(git_head_hash)
    # Parse the config file to resolve names.
    config = config.parse()
    # Get everything at once.
    trainer = config.get_trainer(device=args.device)
    # Train the model
    trainer.train(args.path)
