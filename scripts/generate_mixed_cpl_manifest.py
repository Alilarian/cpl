"""
Generates the SLURM manifest for the mixed-feedback-type CPL experiment
family: `credit_assignment` and `scalar` each act as a 90% "base" signal,
mixed 9:1 with `demo`, `pref`, and `corr` in turn (6 combos), across 4
MetaWorld envs and 3 policy-init seeds -- 72 runs total.

Mirrors the exact structure already used by slurm/manifests/{demo,credit,
pref,corr,scalar}_cpl_all/ (one full config.yaml per run + a manifest.tsv),
so it submits through the existing generic slurm/train_array.sbatch -- no new
sbatch script needed:

    sbatch --array=1-72 --account=<acct> --partition=<part> \\
        slurm/train_array.sbatch slurm/manifests/mixed_cpl_all/manifest.tsv

Each generated config imports configs/mw_state_dense/mixed_cpl.yaml and fills
in `dataset_kwargs.components` (see research/datasets/mixed_buffer.py) and
`alg_kwargs.component_weights` (see research/algs/mixed_cpl.py).

subsample_seed=0 is used for every component, matching the already-completed
mw_b10k baselines (which also used subsample_seed=0 on the same 100k-row
label files) -- since numpy's RandomState(0).permutation(N) is identical
regardless of how many elements you take from the front, this makes each
mixed run's "9000 credit rows" a strict prefix-subset of the exact 10000 rows
the corresponding mw_b10k/<env>/credit_assignment/cpl_s<seed> baseline
already trained on (and likewise for pref/corr/scalar's "1000 rows"). demo
has no such 10k baseline in mw_b10k (it was excluded from that matrix), but
its raw demo_labels_K9.npz file still exists and subsample_n=1000 draws
reproducibly from it the same way.

NOTE: LABELS_DIR / DEMO_LABELS_DIR below are reconstructed from the *already
completed* mw_b10k runs' saved config.yaml files in this repo checkout
(runs/mw_b10k/mw_drawer-open-v2/<type>/cpl_s0/config.yaml). They should still
be correct, but are kept as top-of-file constants specifically so a path fix
on CHPC doesn't require touching the mixing logic below.
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
BASE_TYPES = ["credit_assignment", "scalar"]
SECONDARY_TYPES = ["demo", "pref", "corr"]
SEEDS = [0, 1, 2]

BASE_WEIGHT, SECONDARY_WEIGHT = 0.9, 0.1  # dataset/batch composition
BASE_N, SECONDARY_N = 9000, 1000  # subsample_n per slot
SUBSAMPLE_SEED = 0
BATCH_SIZE = 96
CONTRASTIVE_BIAS = 0.75
TOTAL_STEPS = 250000  # matches the static per-type yaml files under
# configs/mw_state_dense/ (the real mw_b10k baselines used 500000 instead).

LABELS_DIR = "/scratch/general/vast/u1472210/mw_de_labels"  # credit/scalar/pref/corr
DEMO_LABELS_DIR = "/scratch/general/vast/u1472210/demo_labels"  # demo lives separately
DEMO_LABELS_FILENAME = "demo_labels_K9.npz"  # K9, not K7 -- confirmed against the
# actual CHPC scratch directory (demo_cpl_all's older manifest predates a
# K7->K9 regeneration and is stale on this point).

DATASET_CLASS = {
    "credit_assignment": "PMCreditAssignmentBuffer",
    "scalar": "CorrBuffer",
    "pref": "CorrBuffer",
    "corr": "CorrBuffer",
    "demo": "DemoBuffer",
}

# Label filenames under LABELS_DIR/<env>/ -- NOT simply "<type>_labels.npz"
# for credit_assignment (it's "credit_labels.npz"). Confirmed against the
# already-completed mw_b10k runs' saved config.yaml files.
LABEL_FILENAME = {
    "credit_assignment": "credit_labels.npz",
    "scalar": "scalar_labels.npz",
    "pref": "pref_labels.npz",
    "corr": "corr_labels.npz",
}


def label_path(feedback_type: str, env: str) -> str:
    if feedback_type == "demo":
        return f"{DEMO_LABELS_DIR}/{env}/{DEMO_LABELS_FILENAME}"
    return f"{LABELS_DIR}/{env}/{LABEL_FILENAME[feedback_type]}"


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


def build_config(base: str, secondary: str, env: str, seed: int) -> dict:
    return {
        "import": "configs/mw_state_dense/mixed_cpl.yaml",
        "eval_env": env,
        "seed": seed,
        "alg_kwargs": {
            "contrastive_bias": CONTRASTIVE_BIAS,
            "component_weights": {base: 1.0, secondary: SECONDARY_WEIGHT},
        },
        "dataset_kwargs": {
            "batch_size": BATCH_SIZE,
            "components": [
                component(base, BASE_WEIGHT, BASE_N, env),
                component(secondary, SECONDARY_WEIGHT, SECONDARY_N, env),
            ],
        },
        "trainer_kwargs": {
            "total_steps": TOTAL_STEPS,
        },
    }


def run_path(base: str, secondary: str, env: str, seed: int) -> str:
    return f"runs/mw_b10k_mixed/{env}/{base}+{secondary}_10pct/cpl_s{seed}"


def main():
    manifest_dir = os.path.join(REPO_ROOT, "slurm", "manifests", "mixed_cpl_all")
    configs_dir = os.path.join(manifest_dir, "configs")
    os.makedirs(configs_dir, exist_ok=True)

    rows = []
    task_id = 0
    for base in BASE_TYPES:
        for secondary in SECONDARY_TYPES:
            for env in ENVS:
                for seed in SEEDS:
                    task_id += 1
                    config = build_config(base, secondary, env, seed)
                    config_rel = f"slurm/manifests/mixed_cpl_all/configs/{task_id:04d}.yaml"
                    with open(os.path.join(REPO_ROOT, config_rel), "w") as f:
                        yaml.dump(config, f, sort_keys=False)
                    rows.append(f"{config_rel}\t{run_path(base, secondary, env, seed)}")

    manifest_path = os.path.join(manifest_dir, "manifest.tsv")
    with open(manifest_path, "w") as f:
        f.write("\n".join(rows) + "\n")

    print(f"Wrote {task_id} configs + manifest to {manifest_dir}")
    print(f"Submit with: sbatch --array=1-{task_id} --account=<acct> --partition=<part> \\")
    print(f"    slurm/train_array.sbatch {os.path.relpath(manifest_path, REPO_ROOT)}")


if __name__ == "__main__":
    main()
