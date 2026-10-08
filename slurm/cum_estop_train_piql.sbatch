#!/bin/bash
# Train the Cumulative E-stop model with PIQL for one MetaWorld env, across
# N_SEEDS seeds via a SLURM array -- one task per seed.
#
# PIQL has no component-weighted mixing mechanism (unlike MixedCPL), so the
# two branches (stop_correction, no_stop_demo) are concatenated into one flat
# file first (scripts/concat_cum_estop_branches.py) -- natural-frequency
# weighting across branches, not the lambda=0.5 equal-branch weighting CPL
# gets. The concat step runs once per env and is skipped if its output
# already exists.
#
# Usage (GPU):
#   ENV_NAME=mw_button-press-v2 SEEDS="0 1 2" DEVICE=cuda CONDA_ENV_NAME=<env> \
#     sbatch --array=0-2 --gres=gpu:1 \
#            --account=<acct> --partition=<gpu-partition> \
#            [--time=D-HH:MM:SS] [--cpus-per-task=N] [--mem=XG] \
#            slurm/cum_estop_train_piql.sbatch
#
# Optional env overrides:
#   ENV_NAME           default: mw_drawer-open-v2
#   SEEDS              default: "0 1 2"
#   LABELS_DIR         default: /scratch/general/vast/u1472210/cum_estop_labels
#   RUNS_DIR           default: runs/mw_cum_estop_piql
#   CONFIG_DIR         default: configs/mw_state_dense
#   CONFIG_FILE        default: cum_estop_piql.yaml
#   WANDB_PROJECT      default: mw-piql-cum-estop
#   TOTAL_STEPS        override trainer_kwargs.total_steps (config default: 500000)
#   DEVICE             default: cpu (set to "cuda" to run on GPU, with --gres=gpu:1)
#   CONDA_ENV_NAME     default: cpl (set to your GPU-capable env)

#SBATCH --job-name=cum_estop_train_piql
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

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
if [[ ! -f "${SUBMIT_DIR}/setup_shell.sh" ]]; then
    echo "ERROR: no setup_shell.sh in ${SUBMIT_DIR}"
    exit 1
fi
REPO_ROOT="$SUBMIT_DIR"

ENV_NAME="${ENV_NAME:-mw_drawer-open-v2}"
SEEDS="${SEEDS:-0 1 2}"
LABELS_DIR="${LABELS_DIR:-/scratch/general/vast/u1472210/cum_estop_labels}"
RUNS_DIR="${RUNS_DIR:-runs/mw_cum_estop_piql}"
CONFIG_DIR="${CONFIG_DIR:-configs/mw_state_dense}"
CONFIG_FILE="${CONFIG_FILE:-cum_estop_piql.yaml}"
DEVICE="${DEVICE:-cpu}"

read -r -a SEED_ARR <<< "$SEEDS"
SEED="${SEED_ARR[$SLURM_ARRAY_TASK_ID]:-}"
if [[ -z "$SEED" ]]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} out of range for SEEDS=(${SEEDS})"
    exit 1
fi

[[ "$LABELS_DIR" != /* ]] && LABELS_DIR="${REPO_ROOT}/${LABELS_DIR}"
[[ "$RUNS_DIR"   != /* ]] && RUNS_DIR="${REPO_ROOT}/${RUNS_DIR}"

CONFIG_PATH="${REPO_ROOT}/${CONFIG_DIR}/${CONFIG_FILE}"
STOP_NPZ="${LABELS_DIR}/${ENV_NAME}/stop_correction_labels.npz"
DEMO_NPZ="${LABELS_DIR}/${ENV_NAME}/no_stop_demo_labels.npz"
CONCAT_NPZ="${LABELS_DIR}/${ENV_NAME}/cum_estop_concat.npz"
RUN_PATH="${RUNS_DIR}/${ENV_NAME}/cum_estop_s${SEED}"

echo "=============================="
echo "SLURM_JOB_ID   = ${SLURM_JOB_ID:-}"
echo "ENV_NAME       = ${ENV_NAME}"
echo "SEED           = ${SEED}"
echo "CONCAT_NPZ     = ${CONCAT_NPZ}"
echo "RUN_PATH       = ${RUN_PATH}"
echo "DEVICE         = ${DEVICE}"
echo "CONDA_ENV_NAME = ${CONDA_ENV_NAME:-cpl (default)}"
echo "=============================="

if [[ -f "${RUN_PATH}/training_complete" ]]; then
    echo "Already completed — skipping."
    exit 0
fi
if [[ ! -f "$STOP_NPZ" || ! -f "$DEMO_NPZ" ]]; then
    echo "ERROR: label file(s) not found: ${STOP_NPZ} / ${DEMO_NPZ}"
    echo "Generate first with slurm/cum_estop_generate_labels_sharded.sbatch + scripts/merge_label_shards.py."
    exit 1
fi
if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "ERROR: config not found: ${CONFIG_PATH}"
    exit 1
fi

. "${REPO_ROOT}/setup_shell.sh"
mkdir -p "$RUN_PATH"

if [[ ! -f "$CONCAT_NPZ" ]]; then
    echo "Concatenating branches -> ${CONCAT_NPZ}"
    python3 "${REPO_ROOT}/scripts/concat_cum_estop_branches.py" \
        --stop-correction "$STOP_NPZ" --no-stop-demo "$DEMO_NPZ" --output "$CONCAT_NPZ"
fi

if [[ "$DEVICE" == "cuda" ]]; then
    if ! python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
        echo "ERROR: DEVICE=cuda requested, but torch.cuda.is_available() is False"
        echo "in the active conda env (CONDA_ENV_NAME=${CONDA_ENV_NAME:-cpl (default)})."
        exit 1
    fi
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

EXTRA_SETS=()
[[ -n "${TOTAL_STEPS:-}" ]] && EXTRA_SETS+=(--set "trainer_kwargs.total_steps=${TOTAL_STEPS}")

export WANDB_PROJECT="${WANDB_PROJECT:-mw-piql-cum-estop}"
export WANDB_RUN_NAME="${ENV_NAME}/cum_estop_s${SEED}"
export WANDB_GROUP="${ENV_NAME}"

python3 "${REPO_ROOT}/scripts/train.py" \
    --config "$CONFIG_PATH" \
    --path "$RUN_PATH" \
    --seed "$SEED" \
    --device "$DEVICE" \
    --set "eval_env=${ENV_NAME}" \
    --set "dataset_kwargs.path=${CONCAT_NPZ}" \
    "${EXTRA_SETS[@]+"${EXTRA_SETS[@]}"}"

touch "${RUN_PATH}/training_complete"
echo "Training complete — sentinel written to ${RUN_PATH}/training_complete"
