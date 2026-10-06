"""
Generates the SLURM manifest for the mixed-feedback-type CPL experiment
family: holding-model E-stop (`estop_hold`) as a 90% "base" signal, mixed
9:1 with `demo`, `pref`, and `corr` in turn -- 3 separate 2-component
combos, each run independently (not combined in one run) -- across the 4
MetaWorld envs estop_hold data exists for and 3 policy-init seeds -- 36
runs total.

Mirrors scripts/generate_mixed_cpl_manifest.py's exact base+secondary
pattern (there: credit_assignment/scalar as base; here: estop_hold as the
single base), and submits through the same generic slurm/train_array.sbatch
-- no new sbatch script needed:

    sbatch --array=1-36 --gres=gpu:1 --mem=64G --time=06:00:00 \\
        --account=rai --partition=rai-gpu-grn --qos=rai-gpu-grn \\
        DEVICE=cuda CONDA_ENV_NAME=cpl_gpu \\
        slurm/train_array.sbatch slurm/manifests/estop_mixed_cpl_all/manifest.tsv

Each generated config imports configs/mw_state_dense/mixed_cpl.yaml and fills
in `dataset_kwargs.components` (research/datasets/mixed_buffer.py) and
`alg_kwargs.component_weights` (research/algs/mixed_cpl.py). EstopHoldBuffer
is zero-padded (not repeat-padded like pref/demo/corr) -- MixedCPL masks by
each component's own `horizon` field when present, so mixing it in is safe
(see research/algs/mixed_cpl.py::MixedCPL._get_component_loss).

subsample_seed=0 is used for every component (same convention as
generate_mixed_cpl_manifest.py), so estop_hold's "9000 rows" and each
secondary's "1000 rows" are reproducible draws from the full merged label
files.
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

BASE_NAME = "estop_hold"
BASE_WEIGHT, SECONDARY_WEIGHT = 0.9, 0.1  # dataset/batch composition
BASE_N, SECONDARY_N = 9000, 1000  # subsample_n per slot
SUBSAMPLE_SEED = 0
BATCH_SIZE = 96
ALPHA = 0.1
CONTRASTIVE_BIAS = 0.75
BC_STEPS, BC_COEFF = 0, 0.0
# total_steps is NOT set here -- it lives solely in
# configs/mw_state_dense/mixed_cpl.yaml's trainer_kwargs. See build_config()'s
# docstring note for why per-run configs must never override trainer_kwargs.

ESTOP_LABELS_DIR = "/scratch/general/vast/u1472210/estop_hold_labels"
ESTOP_LABEL_FILENAME = "estop_hold_labels.npz"

# pref/demo/corr label files all live under this one directory, per-env
# subfolder (same as generate_mixed_cpl_manifest.py).
SECONDARY_LABELS_DIR = "/scratch/general/vast/u1472210/mw_de_labels"

DATASET_CLASS = {
    BASE_NAME: "EstopHoldBuffer",
    "pref": "CorrBuffer",
    "corr": "CorrBuffer",
    "demo": "DemoBuffer",
}

LABEL_FILENAME = {
    "pref": "pref_labels.npz",
    "corr": "corr_labels.npz",
    "demo": "demo_labels_K9.npz",
}


def label_path(feedback_type: str, env: str) -> str:
    if feedback_type == BASE_NAME:
        return f"{ESTOP_LABELS_DIR}/{env}/{ESTOP_LABEL_FILENAME}"
    return f"{SECONDARY_LABELS_DIR}/{env}/{LABEL_FILENAME[feedback_type]}"


def component(name: str, weight: float, subsample_n: int, env: str) -> dict:
    return {
        "name": name,
        "dataset_class": DATASET_CLASS[name],
        "weight": weight,
        "dataset_kwargs": {
            "path": label_path(name, env),
            "capacity": None,
            "subsample_n": subsample_n,
            "subsample_seed": SUBSAMPLE_SEED,
        },
    }


def build_config(secondary: str, env: str, seed: int) -> dict:
    """
    IMPORTANT (see scripts/generate_mixed_cpl_manifest.py::build_config for
    the full history): Config.load's `import` merge is a SHALLOW
    dict.update() -- every alg_kwargs/trainer_kwargs field must be restated
    explicitly here, never partially overridden, or it silently drops
    sibling fields from the imported mixed_cpl.yaml (alpha fell back to
    CPL's default 1.0, and a dropped train_dataloader_kwargs re-enabled
    PyTorch's default batch_size=1 auto-collation, in two separate past
    incidents). trainer_kwargs is therefore never touched at all here.
    """
    return {
        "import": "configs/mw_state_dense/mixed_cpl.yaml",
        "eval_env": env,
        "seed": seed,
        "alg_kwargs": {
            "alpha": ALPHA,
            "contrastive_bias": CONTRASTIVE_BIAS,
            "bc_steps": BC_STEPS,
            "bc_coeff": BC_COEFF,
            "component_weights": {BASE_NAME: 1.0, secondary: SECONDARY_WEIGHT},
        },
        "dataset_kwargs": {
            "batch_size": BATCH_SIZE,
            "components": [
                component(BASE_NAME, BASE_WEIGHT, BASE_N, env),
                component(secondary, SECONDARY_WEIGHT, SECONDARY_N, env),
            ],
        },
    }


def run_path(secondary: str, env: str, seed: int) -> str:
    return f"runs/mw_estop_mixed/{env}/{BASE_NAME}+{secondary}_10pct/cpl_s{seed}"


def main():
    manifest_dir = os.path.join(REPO_ROOT, "slurm", "manifests", "estop_mixed_cpl_all")
    configs_dir = os.path.join(manifest_dir, "configs")
    os.makedirs(configs_dir, exist_ok=True)

    rows = []
    task_id = 0
    for secondary in SECONDARY_TYPES:
        for env in ENVS:
            for seed in SEEDS:
                task_id += 1
                config = build_config(secondary, env, seed)
                config_rel = f"slurm/manifests/estop_mixed_cpl_all/configs/{task_id:04d}.yaml"
                with open(os.path.join(REPO_ROOT, config_rel), "w") as f:
                    yaml.dump(config, f, sort_keys=False)
                rows.append(f"{config_rel}\t{run_path(secondary, env, seed)}")

    manifest_path = os.path.join(manifest_dir, "manifest.tsv")
    with open(manifest_path, "w") as f:
        f.write("\n".join(rows) + "\n")

    print(f"Wrote {task_id} configs + manifest to {manifest_dir}")
    print(f"Submit with: DEVICE=cuda CONDA_ENV_NAME=cpl_gpu sbatch --array=1-{task_id} --gres=gpu:1 \\")
    print(f"    --mem=64G --time=06:00:00 --account=<acct> --partition=<part> --qos=<qos> \\")
    print(f"    slurm/train_array.sbatch {os.path.relpath(manifest_path, REPO_ROOT)}")


if __name__ == "__main__":
    main()
