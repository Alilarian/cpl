"""
Generates the SLURM manifest for the single mixed-feedback-type CPL
experiment: holding-model E-stop (`estop_hold`) as a 90% "base" signal, mixed
with `pref`, `demo`, and `corr` simultaneously -- each taking an equal share
of the remaining 10% -- across the 4 MetaWorld envs estop_hold data exists
for and 3 policy-init seeds -- 12 runs total.

Generalizes scripts/generate_mixed_cpl_manifest.py's 2-component (base 90% /
one secondary 10%) pattern to 4 components in a single run, and submits
through the exact same generic slurm/train_array.sbatch -- no new sbatch
script needed:

    sbatch --array=1-12 --gres=gpu:1 --mem=64G --time=06:00:00 \\
        --account=dbrown --partition=coe-class-grn --qos=coe-students-grn \\
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
secondary's ~333 rows are reproducible draws from the full merged label
files, independent across runs only in which *env*/*seed* they're for.
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
SECONDARY_TYPES = ["pref", "demo", "corr"]
SEEDS = [0, 1, 2]

BASE_NAME = "estop_hold"
BASE_WEIGHT = 0.9           # dataset/batch composition share for estop_hold
BASE_N = 9000                # subsample_n for estop_hold

# The remaining 10% is split evenly across the 3 secondary types.
SECONDARY_WEIGHT_TOTAL = 0.1
SECONDARY_N_TOTAL = 1000
SECONDARY_WEIGHT = SECONDARY_WEIGHT_TOTAL / len(SECONDARY_TYPES)
SECONDARY_N = {
    name: SECONDARY_N_TOTAL // len(SECONDARY_TYPES)
    for name in SECONDARY_TYPES
}
# Remainder (1000 % 3 = 1) absorbed by the last-listed secondary type, so the
# three subsample_n values still sum exactly to SECONDARY_N_TOTAL.
SECONDARY_N[SECONDARY_TYPES[-1]] += SECONDARY_N_TOTAL - sum(SECONDARY_N.values())

# Loss weight (alg_kwargs.component_weights) given to each secondary type,
# independently -- matches the per-type weight generate_mixed_cpl_manifest.py
# already uses for its single-secondary combos (0.1), rather than splitting
# the loss weight further across 3 types the way the batch-composition
# weight is split. Rationale: each secondary type is independently a
# "10%-of-a-mix-strength" signal on its own; this keeps that calibration the
# same whether it's mixed in alone or alongside other secondaries.
SECONDARY_COMPONENT_WEIGHT = 0.1

SUBSAMPLE_SEED = 0
BATCH_SIZE = 96
ALPHA = 0.1
CONTRASTIVE_BIAS = 0.75
BC_STEPS, BC_COEFF = 0, 0.0

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


def build_config(env: str, seed: int) -> dict:
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
    component_weights = {BASE_NAME: 1.0}
    component_weights.update({name: SECONDARY_COMPONENT_WEIGHT for name in SECONDARY_TYPES})

    components = [component(BASE_NAME, BASE_WEIGHT, BASE_N, env)]
    components += [
        component(name, SECONDARY_WEIGHT, SECONDARY_N[name], env) for name in SECONDARY_TYPES
    ]

    return {
        "import": "configs/mw_state_dense/mixed_cpl.yaml",
        "eval_env": env,
        "seed": seed,
        "alg_kwargs": {
            "alpha": ALPHA,
            "contrastive_bias": CONTRASTIVE_BIAS,
            "bc_steps": BC_STEPS,
            "bc_coeff": BC_COEFF,
            "component_weights": component_weights,
        },
        "dataset_kwargs": {
            "batch_size": BATCH_SIZE,
            "components": components,
        },
    }


def run_path(env: str, seed: int) -> str:
    return f"runs/mw_estop_mixed/{env}/estop_hold+pref+demo+corr_90_10/cpl_s{seed}"


def main():
    manifest_dir = os.path.join(REPO_ROOT, "slurm", "manifests", "estop_mixed_cpl_all")
    configs_dir = os.path.join(manifest_dir, "configs")
    os.makedirs(configs_dir, exist_ok=True)

    rows = []
    task_id = 0
    for env in ENVS:
        for seed in SEEDS:
            task_id += 1
            config = build_config(env, seed)
            config_rel = f"slurm/manifests/estop_mixed_cpl_all/configs/{task_id:04d}.yaml"
            with open(os.path.join(REPO_ROOT, config_rel), "w") as f:
                yaml.dump(config, f, sort_keys=False)
            rows.append(f"{config_rel}\t{run_path(env, seed)}")

    manifest_path = os.path.join(manifest_dir, "manifest.tsv")
    with open(manifest_path, "w") as f:
        f.write("\n".join(rows) + "\n")

    print(f"Wrote {task_id} configs + manifest to {manifest_dir}")
    print(f"Submit with: DEVICE=cuda CONDA_ENV_NAME=cpl_gpu sbatch --array=1-{task_id} --gres=gpu:1 \\")
    print(f"    --mem=64G --time=06:00:00 --account=<acct> --partition=<part> --qos=<qos> \\")
    print(f"    slurm/train_array.sbatch {os.path.relpath(manifest_path, REPO_ROOT)}")


if __name__ == "__main__":
    main()
