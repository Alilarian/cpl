#!/bin/bash
# Train CPL (DemoCPL) on the full demonstrative-feedback dataset
# (demo_labels_K<K>.npz) for one MetaWorld env, across N_SEEDS seeds via a
# SLURM array -- one task per seed.
#
# Unlike mw_de_train.sbatch's full 6-type x 11-budget x 5-seed sweep (which is
# also hardcoded to demo_labels_K7.npz), this trains only "demo", uses the
# full dataset (dataset_kwargs.capacity=null, no budget clamping), and takes
# an arbitrary caller-chosen seed list.
#
# Usage:
#   ENV_NAME=mw_button-press-v2 SEEDS="0 1 2" \
#     sbatch --array=0-2 --account=<account> --partition=<partition> \
#            [--time=HH:MM:SS] [--cpus-per-task=N] [--mem=XG] \
#            slurm/mw_de_train_demo_cpl.sbatch
#
# SEEDS and --array must agree: --array=0-(N_SEEDS-1) (0-indexed position into
# SEEDS, not the seed value itself).
#
# Resume: if a job is killed by walltime, resubmit the same command -- the
# trainer auto-resumes from final_model.pt; completed runs (training_complete
# sentinel exists) are automatically skipped.
#
# Resource overrides (command-line flags override #SBATCH defaults below):
#   --time=HH:MM:SS     default: 48:00:00
#   --cpus-per-task=N   default: 4
#   --mem=XG            default: 16G
#
# Optional env overrides:
#   ENV_NAME           default: mw_drawer-open-v2
#   SEEDS              default: "0 1 2"
#   LABELS_DIR         default: /scratch/general/vast/u1472210/mw_de_labels
#   DEMO_LABEL_FILE    default: demo_labels_K9.npz
#   RUNS_DIR           default: runs/mw_de_demo_cpl
#   CONFIG_DIR         default: configs/mw_state_dense
#   CONFIG_FILE        default: demo_cpl.yaml
#   WANDB_PROJECT      default: mw-cpl-de
#   CONTRASTIVE_BIAS   override alg_kwargs.contrastive_bias (e.g. 0.5)
#   TOTAL_STEPS        override trainer_kwargs.total_steps  (e.g. 500000)

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --output=/uufs/chpc.utah.edu/common/home/u1472210/cpl/slurm/logs/%x_%A_%a.out
#SBATCH --error=/uufs/chpc.utah.edu/common/home/u1472210/cpl/slurm/logs/%x_%A_%a.err

set -euo pipefail

mkdir -p "${SLURM_SUBMIT_DIR:-$PWD}/slurm/logs"

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "ERROR: submit with --array=0-(N_SEEDS-1), matching SEEDS"
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
DEMO_LABEL_FILE="${DEMO_LABEL_FILE:-demo_labels_K9.npz}"
RUNS_DIR="${RUNS_DIR:-runs/mw_de_demo_cpl}"
CONFIG_DIR="${CONFIG_DIR:-configs/mw_state_dense}"
CONFIG_FILE="${CONFIG_FILE:-demo_cpl.yaml}"

read -r -a SEED_ARR <<< "$SEEDS"
SEED="${SEED_ARR[$SLURM_ARRAY_TASK_ID]:-}"
if [[ -z "$SEED" ]]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} out of range for SEEDS=(${SEEDS})"
    exit 1
fi

[[ "$LABELS_DIR" != /* ]] && LABELS_DIR="${REPO_ROOT}/${LABELS_DIR}"
[[ "$RUNS_DIR"   != /* ]] && RUNS_DIR="${REPO_ROOT}/${RUNS_DIR}"

CONFIG_PATH="${REPO_ROOT}/${CONFIG_DIR}/${CONFIG_FILE}"
LABELS_NPZ="${LABELS_DIR}/${ENV_NAME}/${DEMO_LABEL_FILE}"
RUN_PATH="${RUNS_DIR}/${ENV_NAME}/demo_s${SEED}"

echo "=============================="
echo "SLURM_JOB_ID   = ${SLURM_JOB_ID:-}"
echo "ARRAY_TASK_ID  = ${SLURM_ARRAY_TASK_ID}"
echo "ENV_NAME       = ${ENV_NAME}"
echo "SEED           = ${SEED}"
echo "CONFIG         = ${CONFIG_PATH}"
echo "LABELS         = ${LABELS_NPZ}"
echo "RUN_PATH       = ${RUN_PATH}"
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

EXTRA_SETS=()
[[ -n "${CONTRASTIVE_BIAS:-}" ]] && EXTRA_SETS+=(--set "alg_kwargs.contrastive_bias=${CONTRASTIVE_BIAS}")
[[ -n "${TOTAL_STEPS:-}" ]]      && EXTRA_SETS+=(--set "trainer_kwargs.total_steps=${TOTAL_STEPS}")

# WandB
# Run name: "<env>/demo_s<seed>"  — unique across all envs
# Group:    "<env>"                — groups runs per env in the UI
export WANDB_PROJECT="${WANDB_PROJECT:-mw-cpl-de}"
export WANDB_RUN_NAME="${ENV_NAME}/demo_s${SEED}"
export WANDB_GROUP="${ENV_NAME}"

python3 "${REPO_ROOT}/scripts/train.py" \
    --config  "$CONFIG_PATH" \
    --path    "$RUN_PATH"   \
    --seed    "$SEED"       \
    --device  cpu           \
    --set "eval_env=${ENV_NAME}"                          \
    --set "dataset_kwargs.path=${LABELS_NPZ}"             \
    --set "dataset_kwargs.capacity=null"                  \
    "${EXTRA_SETS[@]+"${EXTRA_SETS[@]}"}"

# Write sentinel only on clean exit (all steps reached).
touch "${RUN_PATH}/training_complete"
echo "Training complete — sentinel written to ${RUN_PATH}/training_complete"
