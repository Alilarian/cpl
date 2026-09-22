#!/bin/bash
# Train BC (research/algs/bc.py::BehaviorCloning) on the demo track (K-index 0,
# no counterfactuals) of demo_labels_K<K>.npz, for one MetaWorld env across
# N_SEEDS seeds via a SLURM array.
#
# Uses this repo's own scripts/train.py pipeline -- same structure as
# mw_de_train_demo_cpl.sbatch -- NOT multi-type-feedback's train_bc.py (that
# needs a separate imitation/stable-baselines3/metaworld-from-git install and
# is not what this script runs).
#
# Prerequisite (run once BEFORE submitting this, not part of this script), per
# env, to produce the flat (no-K-dimension) BCBuffer-compatible npz:
#   python scripts/extract_demo_track_for_bc.py \
#       --demo-labels <LABELS_DIR>/<env>/demo_labels_K<K>.npz \
#       --output-dir  datasets/demo_track_bc
#
# Usage (CPU):
#   ENV_NAME=mw_button-press-v2 SEEDS="0 1 2" \
#     sbatch --array=0-2 \
#            --account=dbrown --partition=coe-class-grn --qos=coe-students-grn \
#            [--time=D-HH:MM:SS] [--cpus-per-task=N] [--mem=XG] \
#            slurm/mw_de_train_demo_bc.sbatch
#
# SEEDS and --array must agree: --array=0-(N_SEEDS-1) (0-indexed position into
# SEEDS, not the seed value itself).
#
# Resume: if a job is killed by walltime, resubmit the same command -- the
# trainer auto-resumes from final_model.pt; completed runs (training_complete
# sentinel exists) are automatically skipped.
#
# Resource overrides (command-line flags override #SBATCH defaults below):
#   --time=D-HH:MM:SS   default: 2-00:00:00 (2 days)
#   --cpus-per-task=N   default: 8
#   --mem=XG            default: 16G
#
# Optional env overrides:
#   ENV_NAME           default: mw_drawer-open-v2
#   SEEDS              default: "0 1 2"
#   BC_DATA_DIR        default: datasets/demo_track_bc
#   BC_DATA_FILE       default: demo_track_bc.npz
#   RUNS_DIR           default: runs/mw_de_demo_bc
#   CONFIG_DIR         default: configs/mw_state_dense
#   CONFIG_FILE        default: demo_bc.yaml
#   WANDB_PROJECT      default: mw-cpl-de
#   TOTAL_STEPS        override trainer_kwargs.total_steps  (e.g. 500000)
#   DEVICE             default: cpu (scripts/train.py --device; set to "cuda"
#                                    to go back to GPU, and pass --gres=gpu:1)

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00
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
BC_DATA_DIR="${BC_DATA_DIR:-datasets/demo_track_bc}"
BC_DATA_FILE="${BC_DATA_FILE:-demo_track_bc.npz}"
RUNS_DIR="${RUNS_DIR:-runs/mw_de_demo_bc}"
CONFIG_DIR="${CONFIG_DIR:-configs/mw_state_dense}"
CONFIG_FILE="${CONFIG_FILE:-demo_bc.yaml}"
DEVICE="${DEVICE:-cpu}"

read -r -a SEED_ARR <<< "$SEEDS"
SEED="${SEED_ARR[$SLURM_ARRAY_TASK_ID]:-}"
if [[ -z "$SEED" ]]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} out of range for SEEDS=(${SEEDS})"
    exit 1
fi

[[ "$BC_DATA_DIR" != /* ]] && BC_DATA_DIR="${REPO_ROOT}/${BC_DATA_DIR}"
[[ "$RUNS_DIR"    != /* ]] && RUNS_DIR="${REPO_ROOT}/${RUNS_DIR}"

CONFIG_PATH="${REPO_ROOT}/${CONFIG_DIR}/${CONFIG_FILE}"
DATA_NPZ="${BC_DATA_DIR}/${ENV_NAME}/${BC_DATA_FILE}"
RUN_PATH="${RUNS_DIR}/${ENV_NAME}/bc_s${SEED}"

echo "=============================="
echo "SLURM_JOB_ID   = ${SLURM_JOB_ID:-}"
echo "ARRAY_TASK_ID  = ${SLURM_ARRAY_TASK_ID}"
echo "ENV_NAME       = ${ENV_NAME}"
echo "SEED           = ${SEED}"
echo "CONFIG         = ${CONFIG_PATH}"
echo "DATA           = ${DATA_NPZ}"
echo "RUN_PATH       = ${RUN_PATH}"
echo "DEVICE         = ${DEVICE}"
echo "=============================="

if [[ -f "${RUN_PATH}/training_complete" ]]; then
    echo "Already completed (training_complete sentinel exists) — skipping."
    exit 0
fi

if [[ ! -f "$DATA_NPZ" ]]; then
    echo "ERROR: data file not found: ${DATA_NPZ}"
    echo "Run scripts/extract_demo_track_for_bc.py first."
    exit 1
fi
if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "ERROR: config not found: ${CONFIG_PATH}"
    exit 1
fi

. "${REPO_ROOT}/setup_shell.sh"
mkdir -p "$RUN_PATH"

EXTRA_SETS=()
[[ -n "${TOTAL_STEPS:-}" ]] && EXTRA_SETS+=(--set "trainer_kwargs.total_steps=${TOTAL_STEPS}")

# WandB
# Run name: "<env>/bc_s<seed>"  — unique across all envs
# Group:    "<env>"              — groups runs per env in the UI
export WANDB_PROJECT="${WANDB_PROJECT:-mw-cpl-de}"
export WANDB_RUN_NAME="${ENV_NAME}/bc_s${SEED}"
export WANDB_GROUP="${ENV_NAME}"

python3 "${REPO_ROOT}/scripts/train.py" \
    --config  "$CONFIG_PATH" \
    --path    "$RUN_PATH"   \
    --seed    "$SEED"       \
    --device  "$DEVICE"     \
    --set "eval_env=${ENV_NAME}"            \
    --set "dataset_kwargs.path=${DATA_NPZ}" \
    --set "dataset_kwargs.capacity=null"    \
    "${EXTRA_SETS[@]+"${EXTRA_SETS[@]}"}"

# Write sentinel only on clean exit (all steps reached).
touch "${RUN_PATH}/training_complete"
echo "Training complete — sentinel written to ${RUN_PATH}/training_complete"
