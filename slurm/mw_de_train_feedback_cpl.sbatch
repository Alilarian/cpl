#!/bin/bash
# Train CPL on a non-demo feedback type (pref | corr | scalar | credit_assignment)
# using the full dataset (no budget sweep) for one MetaWorld env, across
# N_SEEDS seeds via a SLURM array. Structured identically to
# mw_de_train_demo_cpl.sbatch (resumability, WandB naming, GPU fail-fast
# check, thread pinning) but parametrized by FEEDBACK_TYPE since these 4
# types share the same config/label-file shape (unlike demo, which uses a
# different dataset class and label format).
#
# All 4 label files (pref_labels.npz, corr_labels.npz, scalar_labels.npz,
# credit_labels.npz) are assumed to already exist per env under LABELS_DIR --
# this script only trains, it does not generate labels.
#
# Usage (CPU):
#   FEEDBACK_TYPE=pref ENV_NAME=mw_button-press-v2 SEEDS="0 1 2" \
#     sbatch --array=0-2 \
#            --account=dbrown --partition=coe-class-grn --qos=coe-students-grn \
#            slurm/mw_de_train_feedback_cpl.sbatch
#
# Usage (GPU):
#   FEEDBACK_TYPE=pref ENV_NAME=mw_button-press-v2 SEEDS="0 1 2" \
#   CONDA_ENV_NAME=cpl_gpu DEVICE=cuda \
#     sbatch --array=0-2 --gres=gpu:1 \
#            --account=dbrown-gpu-np --partition=dbrown-gpu-np --qos=dbrown-gpu-np \
#            slurm/mw_de_train_feedback_cpl.sbatch
#
# SEEDS and --array must agree: --array=0-(N_SEEDS-1) (0-indexed position into
# SEEDS, not the seed value itself).
#
# Resume: if a job is killed by walltime, resubmit the same command -- the
# trainer auto-resumes from final_model.pt; completed runs (training_complete
# sentinel exists) are automatically skipped.
#
# Resource overrides (command-line flags override #SBATCH defaults below):
#   --time=D-HH:MM:SS   default: 12:00:00
#   --cpus-per-task=N   default: 8
#   --mem=XG            default: 64G
#
# Optional env overrides:
#   FEEDBACK_TYPE      required: pref | corr | scalar | credit_assignment
#   ENV_NAME           default: mw_drawer-open-v2
#   SEEDS              default: "0 1 2"
#   LABELS_DIR         default: /scratch/general/vast/u1472210/mw_de_labels
#   RUNS_DIR           default: runs/mw_de_feedback_cpl
#   CONFIG_DIR         default: configs/mw_state_dense
#   WANDB_PROJECT      default: mw-cpl-de
#   CONTRASTIVE_BIAS   override alg_kwargs.contrastive_bias (e.g. 0.5)
#   TOTAL_STEPS        override trainer_kwargs.total_steps  (default in every
#                                     config here is already 250000)
#   DEVICE             default: cpu (scripts/train.py --device; "cuda" needs
#                                    --gres=gpu:1 and a GPU-capable
#                                    CONDA_ENV_NAME, e.g. cpl_gpu)

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/uufs/chpc.utah.edu/common/home/u1472210/cpl/slurm/logs/%x_%A_%a.out
#SBATCH --error=/uufs/chpc.utah.edu/common/home/u1472210/cpl/slurm/logs/%x_%A_%a.err

set -euo pipefail

mkdir -p "${SLURM_SUBMIT_DIR:-$PWD}/slurm/logs"

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "ERROR: submit with --array=0-(N_SEEDS-1), matching SEEDS"
    exit 1
fi

if [[ "${FEEDBACK_TYPE:-}" != "pref" && "${FEEDBACK_TYPE:-}" != "corr" \
   && "${FEEDBACK_TYPE:-}" != "scalar" && "${FEEDBACK_TYPE:-}" != "credit_assignment" ]]; then
    echo "ERROR: set FEEDBACK_TYPE=pref|corr|scalar|credit_assignment"
    exit 1
fi

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
if [[ ! -f "${SUBMIT_DIR}/setup_shell.sh" ]]; then
    echo "ERROR: no setup_shell.sh in ${SUBMIT_DIR}"
    exit 1
fi
REPO_ROOT="$SUBMIT_DIR"

ENV_NAME="${ENV_NAME:-mw_drawer-open-v2}"
SEEDS="${SEEDS:-0 1 2}"
LABELS_DIR="${LABELS_DIR:-/scratch/general/vast/u1472210/mw_de_labels}"
RUNS_DIR="${RUNS_DIR:-runs/mw_de_feedback_cpl}"
CONFIG_DIR="${CONFIG_DIR:-configs/mw_state_dense}"
DEVICE="${DEVICE:-cpu}"

declare -A TYPE_CONFIG
TYPE_CONFIG[pref]=pref_cpl.yaml
TYPE_CONFIG[corr]=corr_cpl.yaml
TYPE_CONFIG[scalar]=scalar_cpl.yaml
TYPE_CONFIG[credit_assignment]=credit_cpl.yaml

declare -A TYPE_LABEL_FILE
TYPE_LABEL_FILE[pref]=pref_labels.npz
TYPE_LABEL_FILE[corr]=corr_labels.npz
TYPE_LABEL_FILE[scalar]=scalar_labels.npz
TYPE_LABEL_FILE[credit_assignment]=credit_labels.npz

CONFIG_FILE="${TYPE_CONFIG[$FEEDBACK_TYPE]}"
LABEL_FILE="${TYPE_LABEL_FILE[$FEEDBACK_TYPE]}"

read -r -a SEED_ARR <<< "$SEEDS"
SEED="${SEED_ARR[$SLURM_ARRAY_TASK_ID]:-}"
if [[ -z "$SEED" ]]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} out of range for SEEDS=(${SEEDS})"
    exit 1
fi

[[ "$LABELS_DIR" != /* ]] && LABELS_DIR="${REPO_ROOT}/${LABELS_DIR}"
[[ "$RUNS_DIR"   != /* ]] && RUNS_DIR="${REPO_ROOT}/${RUNS_DIR}"

CONFIG_PATH="${REPO_ROOT}/${CONFIG_DIR}/${CONFIG_FILE}"
LABELS_NPZ="${LABELS_DIR}/${ENV_NAME}/${LABEL_FILE}"
RUN_PATH="${RUNS_DIR}/${ENV_NAME}/${FEEDBACK_TYPE}_s${SEED}"

echo "=============================="
echo "SLURM_JOB_ID   = ${SLURM_JOB_ID:-}"
echo "ARRAY_TASK_ID  = ${SLURM_ARRAY_TASK_ID}"
echo "FEEDBACK_TYPE  = ${FEEDBACK_TYPE}"
echo "ENV_NAME       = ${ENV_NAME}"
echo "SEED           = ${SEED}"
echo "CONFIG         = ${CONFIG_PATH}"
echo "LABELS         = ${LABELS_NPZ}"
echo "RUN_PATH       = ${RUN_PATH}"
echo "DEVICE         = ${DEVICE}"
echo "CONDA_ENV_NAME = ${CONDA_ENV_NAME:-cpl (default)}"
echo "=============================="

if [[ -f "${RUN_PATH}/training_complete" ]]; then
    echo "Already completed (training_complete sentinel exists) — skipping."
    exit 0
fi

if [[ ! -f "$LABELS_NPZ" ]]; then
    echo "ERROR: label file not found: ${LABELS_NPZ}"
    exit 1
fi
if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "ERROR: config not found: ${CONFIG_PATH}"
    exit 1
fi

. "${REPO_ROOT}/setup_shell.sh"
mkdir -p "$RUN_PATH"

# Fail fast and clearly if DEVICE=cuda was requested but the active conda env's
# torch can't actually see a GPU -- otherwise this surfaces ~20s later as a
# confusing "Torch not compiled with CUDA enabled" traceback deep inside model
# construction, with no indication that CONDA_ENV_NAME is the likely cause.
if [[ "$DEVICE" == "cuda" ]]; then
    if ! python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
        echo "ERROR: DEVICE=cuda requested, but torch.cuda.is_available() is False"
        echo "in the active conda env (CONDA_ENV_NAME=${CONDA_ENV_NAME:-cpl (default)})."
        echo "Did you forget to set CONDA_ENV_NAME=cpl_gpu (or your GPU-capable env)?"
        exit 1
    fi
fi

# Match PyTorch's CPU intra-op thread pool to what SLURM actually allocated --
# nothing else in this codebase sets this, so without it PyTorch's own default
# detection is trusted implicitly.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

EXTRA_SETS=()
[[ -n "${CONTRASTIVE_BIAS:-}" ]] && EXTRA_SETS+=(--set "alg_kwargs.contrastive_bias=${CONTRASTIVE_BIAS}")
[[ -n "${TOTAL_STEPS:-}" ]]      && EXTRA_SETS+=(--set "trainer_kwargs.total_steps=${TOTAL_STEPS}")

# WandB
# Run name: "<env>/<type>_s<seed>"  — unique across all envs and types
# Group:    "<env>"                  — groups runs per env in the UI
export WANDB_PROJECT="${WANDB_PROJECT:-mw-cpl-de}"
export WANDB_RUN_NAME="${ENV_NAME}/${FEEDBACK_TYPE}_s${SEED}"
export WANDB_GROUP="${ENV_NAME}"

python3 "${REPO_ROOT}/scripts/train.py" \
    --config  "$CONFIG_PATH" \
    --path    "$RUN_PATH"   \
    --seed    "$SEED"       \
    --device  "$DEVICE"     \
    --set "eval_env=${ENV_NAME}"            \
    --set "dataset_kwargs.path=${LABELS_NPZ}" \
    --set "dataset_kwargs.capacity=null"    \
    "${EXTRA_SETS[@]+"${EXTRA_SETS[@]}"}"

# Write sentinel only on clean exit (all steps reached).
touch "${RUN_PATH}/training_complete"
echo "Training complete — sentinel written to ${RUN_PATH}/training_complete"
