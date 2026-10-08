"""
Generates the SLURM manifest for mixing the Cumulative E-stop model
(cumulative_estop_model.md) with each of demo/pref/corr in turn, at a 90/10
ratio: the cum_estop model (its own two branches, stop_correction and
no_stop_demo, each getting half of the 90% to preserve their internal
lambda=0.5 equal-branch weighting) as the dominant signal, mixed with one
secondary feedback type taking the remaining 10% -- 3 separate 2-branch-vs-
1-secondary combos (not combined into a single 4-way run), across the 4
MetaWorld envs and 3 policy-init seeds -- 36 runs total.

Mirrors scripts/generate_estop_mixed_cpl_manifest.py's pattern (itself
mirroring generate_mixed_cpl_manifest.py's base+secondary structure), but
written independently for this model -- no shared code with either.

Submits through the existing generic slurm/train_array.sbatch:
    sbatch --array=1-36 --gres=gpu:1 --mem=64G --time=06:00:00 \\
        --account=<acct> --partition=<part> --qos=<qos> \\
        DEVICE=cuda CONDA_ENV_NAME=<env> \\
        slurm/train_array.sbatch slurm/manifests/cum_estop_mixed_cpl_all/manifest.tsv
"""

import os

import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ENVS = [
    "mw_button-press-v2",
    "mw_door-open-v2",
    "mw_drawer-open-v2",
    "mw_plate-slide-v2",
]
SECONDARY_TYPES = ["demo", "pref", "corr"]
SEEDS = [0, 1, 2]

# The cum_estop model's own 90% share is split evenly across its two
# branches (preserving lambda=0.5 internally); the secondary type gets 10%.
CUM_ESTOP_BRANCH_WEIGHT = 0.45
SECONDARY_WEIGHT = 0.1
CUM_ESTOP_DISCOUNT = 0.99

SUBSAMPLE_SEED = 0
BATCH_SIZE = 96
ALPHA = 0.1
CONTRASTIVE_BIAS = 0.75
BC_STEPS, BC_COEFF = 0, 0.0

CUM_ESTOP_LABELS_DIR = "/scratch/general/vast/u1472210/cum_estop_labels"
SECONDARY_LABELS_DIR = "/scratch/general/vast/u1472210/mw_de_labels"

DATASET_CLASS = {"demo": "DemoBuffer", "pref": "CorrBuffer", "corr": "CorrBuffer"}
LABEL_FILENAME = {"demo": "demo_labels_K9.npz", "pref": "pref_labels.npz", "corr": "corr_labels.npz"}
SECONDARY_N = {"demo": 1000, "pref": 1000, "corr": 1000}


def cum_estop_component(name: str, env: str) -> dict:
    return {
        "name": name,
        "dataset_class": "CorrBuffer",
        "weight": CUM_ESTOP_BRANCH_WEIGHT,
        "dataset_kwargs": {
            "path": f"{CUM_ESTOP_LABELS_DIR}/{env}/{name}_labels.npz",
            "capacity": None,
        },
    }


def secondary_component(name: str, env: str) -> dict:
    return {
        "name": name,
        "dataset_class": DATASET_CLASS[name],
        "weight": SECONDARY_WEIGHT,
        "dataset_kwargs": {
            "path": f"{SECONDARY_LABELS_DIR}/{env}/{LABEL_FILENAME[name]}",
            "capacity": None,
            "subsample_n": SECONDARY_N[name],
            "subsample_seed": SUBSAMPLE_SEED,
        },
    }


def build_config(secondary: str, env: str, seed: int) -> dict:
    """See generate_mixed_cpl_manifest.py's build_config docstring for why every
    alg_kwargs/trainer_kwargs field must be restated explicitly here: Config.load's
    `import` merge is a SHALLOW dict.update(), so a partial override silently
    drops sibling fields from the imported mixed_cpl.yaml."""
    return {
        "import": "configs/mw_state_dense/mixed_cpl.yaml",
        "eval_env": env,
        "seed": seed,
        "alg_kwargs": {
            "alpha": ALPHA,
            "contrastive_bias": CONTRASTIVE_BIAS,
            "bc_steps": BC_STEPS,
            "bc_coeff": BC_COEFF,
            "component_weights": {
                "stop_correction": CUM_ESTOP_BRANCH_WEIGHT,
                "no_stop_demo": CUM_ESTOP_BRANCH_WEIGHT,
                secondary: SECONDARY_WEIGHT,
            },
            "component_discounts": {
                "stop_correction": CUM_ESTOP_DISCOUNT,
                "no_stop_demo": CUM_ESTOP_DISCOUNT,
            },
        },
        "dataset_kwargs": {
            "batch_size": BATCH_SIZE,
            "components": [
                cum_estop_component("stop_correction", env),
                cum_estop_component("no_stop_demo", env),
                secondary_component(secondary, env),
            ],
        },
    }


def run_path(secondary: str, env: str, seed: int) -> str:
    return f"runs/mw_cum_estop_mixed/{env}/cum_estop+{secondary}_10pct/cpl_s{seed}"


def main():
    manifest_dir = os.path.join(REPO_ROOT, "slurm", "manifests", "cum_estop_mixed_cpl_all")
    configs_dir = os.path.join(manifest_dir, "configs")
    os.makedirs(configs_dir, exist_ok=True)

    rows = []
    task_id = 0
    for secondary in SECONDARY_TYPES:
        for env in ENVS:
            for seed in SEEDS:
                task_id += 1
                config = build_config(secondary, env, seed)
                config_rel = f"slurm/manifests/cum_estop_mixed_cpl_all/configs/{task_id:04d}.yaml"
                with open(os.path.join(REPO_ROOT, config_rel), "w") as f:
                    yaml.dump(config, f, sort_keys=False)
                rows.append(f"{config_rel}\t{run_path(secondary, env, seed)}")

    manifest_path = os.path.join(manifest_dir, "manifest.tsv")
    with open(manifest_path, "w") as f:
        f.write("\n".join(rows) + "\n")

    print(f"Wrote {task_id} configs + manifest to {manifest_dir}")
    print(f"Submit with: DEVICE=cuda CONDA_ENV_NAME=<env> sbatch --array=1-{task_id} --gres=gpu:1 \\")
    print(f"    --mem=64G --time=06:00:00 --account=<acct> --partition=<part> --qos=<qos> \\")
    print(f"    slurm/train_array.sbatch {os.path.relpath(manifest_path, REPO_ROOT)}")


if __name__ == "__main__":
    main()
