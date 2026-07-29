#!/bin/bash
# One-time preprocessing of seq_estop, scalar, and credit_assignment label files.
#
# Removes pool-ordering bias so that capacity=N at training time gives an
# unbiased, diverse N samples instead of the first N (worst) pool trajectories.
#
#   seq_estop   : stratified (1 pair/trajectory) + shuffle  713k → ~161k rows
#   scalar      : shuffle only                              100k rows
#   credit      : shuffle only                              161k rows
#
# Array task mapping  (2 envs × 3 types = 6 tasks):
#   Task 1  mw_drawer-open-v2  seq_estop
#   Task 2  mw_drawer-open-v2  scalar
#   Task 3  mw_drawer-open-v2  credit_assignment
#   Task 4  mw_plate-slide-v2  seq_estop
#   Task 5  mw_plate-slide-v2  scalar
#   Task 6  mw_plate-slide-v2  credit_assignment
#
# Usage:
#   sbatch --array=1-6 --account=<acct> --partition=<part> \
#          slurm/mw_de_preprocess_labels.sbatch
#
# Optional env overrides:
#   LABELS_DIR   default: /scratch/general/vast/u1472210/mw_de_labels
#   OUTPUT_DIR   default: /scratch/general/vast/u1472210/mw_de_labels_preproc
#   SEED         default: 42

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --output=/uufs/chpc.utah.edu/common/home/u1472210/cpl/slurm/logs/%x_%A_%a.out
#SBATCH --error=/uufs/chpc.utah.edu/common/home/u1472210/cpl/slurm/logs/%x_%A_%a.err

set -euo pipefail

mkdir -p "${SLURM_SUBMIT_DIR:-$PWD}/slurm/logs"

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "ERROR: submit with --array=1-6"
    exit 1
fi

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
REPO_ROOT="$SUBMIT_DIR"

# ── Matrix ────────────────────────────────────────────────────────────────────
ENVS=(
    mw_drawer-open-v2
    mw_drawer-open-v2
    mw_drawer-open-v2
    mw_plate-slide-v2
    mw_plate-slide-v2
    mw_plate-slide-v2
)
TYPES=(
    seq_estop
    scalar
    credit_assignment
    seq_estop
    scalar
    credit_assignment
)

IDX=$(( SLURM_ARRAY_TASK_ID - 1 ))
ENV_NAME="${ENVS[$IDX]}"
FEEDBACK_TYPE="${TYPES[$IDX]}"

# ── Paths ─────────────────────────────────────────────────────────────────────
LABELS_DIR="${LABELS_DIR:-/scratch/general/vast/u1472210/mw_de_labels}"
OUTPUT_DIR="${OUTPUT_DIR:-/scratch/general/vast/u1472210/mw_de_labels_preproc}"
SEED="${SEED:-42}"

echo "=============================="
echo "ARRAY_TASK_ID  = ${SLURM_ARRAY_TASK_ID}"
echo "ENV_NAME       = ${ENV_NAME}"
echo "FEEDBACK_TYPE  = ${FEEDBACK_TYPE}"
echo "LABELS_DIR     = ${LABELS_DIR}"
echo "OUTPUT_DIR     = ${OUTPUT_DIR}"
echo "SEED           = ${SEED}"
echo "=============================="

. "${REPO_ROOT}/setup_shell.sh"

python3 "${REPO_ROOT}/scripts/preprocess_de_labels.py" \
    --labels-dir  "$LABELS_DIR"  \
    --output-dir  "$OUTPUT_DIR"  \
    --env         "$ENV_NAME"    \
    --type        "$FEEDBACK_TYPE" \
    --seed        "$SEED"
