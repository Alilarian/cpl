#!/bin/bash
# Train CPL on PointMass for all 6 feedback types (array job).
#
# Each array task trains one feedback type with its dedicated WandB project:
#   1 = pref       → WANDB_PROJECT=pm-cpl-pref
#   2 = corr       → WANDB_PROJECT=pm-cpl-corr
#   3 = demo       → WANDB_PROJECT=pm-cpl-demo
#   4 = estop      → WANDB_PROJECT=pm-cpl-estop
#   5 = seq_estop  → WANDB_PROJECT=pm-cpl-seq_estop
#   6 = scalar     → WANDB_PROJECT=pm-cpl-scalar
#
# No BC warmup (bc_steps=0 in all configs).
#
# Depends on:
#   - datasets/pm/{type}_labels.npz          from pm_generate_feedback.sbatch
#   - runs/pm_sac_oracle/advantage_vi_N100.npz  from pm_compute_vi.sbatch
#     (only for vector-field overlay; training proceeds without it if absent)
#
# Pipeline order on CHPC:
#   1. sbatch slurm/pm_generate_pool.sbatch
#   2. sbatch slurm/pm_compute_vi.sbatch
#   3. sbatch --array=1-6 slurm/pm_generate_feedback.sbatch
#   4. sbatch --array=1-6 slurm/pm_train_cpl.sbatch          (this script)
#
# Usage:
#   sbatch --array=1-6 \
#          --job-name=pm_train_cpl \
#          --time=4:00:00 \
#          --cpus-per-task=4 \
#          --mem=16G \
#          --gres=gpu:1 \
#          --account=<account> \
#          --partition=<partition> \
#          slurm/pm_train_cpl.sbatch
#
# Optional overrides (set before sbatch or via --export):
#   RUNS_DIR   default: runs          (output base directory)
#   DATA_DIR   default: datasets/pm   (directory with *_labels.npz files)

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=slurm/logs/%x_%A_%a.out
#SBATCH --error=slurm/logs/%x_%A_%a.err

set -euo pipefail

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "SLURM_ARRAY_TASK_ID is not set. Submit with sbatch --array=1-5 ..."
    exit 1
fi

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
if [[ ! -f "${SUBMIT_DIR}/setup_shell.sh" ]]; then
    echo "ERROR: could not locate repo root (no setup_shell.sh in ${SUBMIT_DIR})"
    exit 1
fi
REPO_ROOT="$SUBMIT_DIR"

# Map task ID → feedback type
TYPES=(pref corr demo estop seq_estop scalar)
FEEDBACK_TYPE="${TYPES[$((SLURM_ARRAY_TASK_ID - 1))]}"

RUNS_DIR="${RUNS_DIR:-runs}"
DATA_DIR="${DATA_DIR:-datasets/pm}"

[[ "$RUNS_DIR" != /* ]] && RUNS_DIR="${REPO_ROOT}/${RUNS_DIR}"
[[ "$DATA_DIR" != /* ]] && DATA_DIR="${REPO_ROOT}/${DATA_DIR}"

CONFIG_PATH="${REPO_ROOT}/configs/pm_navigation/cpl_${FEEDBACK_TYPE}.yaml"
RUN_PATH="${RUNS_DIR}/pm_cpl_${FEEDBACK_TYPE}"

# Verify the feedback labels exist before spending GPU time
LABELS_NPZ="${DATA_DIR}/${FEEDBACK_TYPE}_labels.npz"
if [[ ! -f "$LABELS_NPZ" ]]; then
    echo "ERROR: labels not found at ${LABELS_NPZ}"
    echo "Run pm_generate_feedback.sbatch first."
    exit 1
fi

echo "=============================="
echo "SLURM_JOB_ID    = ${SLURM_JOB_ID:-}"
echo "ARRAY_TASK_ID   = ${SLURM_ARRAY_TASK_ID}"
echo "FEEDBACK_TYPE   = ${FEEDBACK_TYPE}"
echo "CONFIG_PATH     = ${CONFIG_PATH}"
echo "RUN_PATH        = ${RUN_PATH}"
echo "WANDB_PROJECT   = pm-cpl-${FEEDBACK_TYPE}"
echo "=============================="

. "${REPO_ROOT}/setup_shell.sh"
mkdir -p "$RUN_PATH"

export WANDB_PROJECT="pm-cpl-${FEEDBACK_TYPE}"

python3 "${REPO_ROOT}/scripts/train.py" \
    --config "$CONFIG_PATH" \
    --path   "$RUN_PATH"
