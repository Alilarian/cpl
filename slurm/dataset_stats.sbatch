#!/bin/bash
# Compute dataset statistics (pool + demo labels) — one SLURM array task per environment.
#
# Usage:
#   sbatch --array=1-5 \
#          --job-name=dataset_stats \
#          --time=1:00:00 \
#          --cpus-per-task=4 \
#          --mem=32G \
#          slurm/dataset_stats.sbatch
#
# Optional overrides (via --export):
#   POOL_DIR    default: /scratch/general/vast/u1472210/demo_pool
#   LABELS_DIR  default: /scratch/general/vast/u1472210/demo_labels
#   OUTPUT_DIR  default: results/dataset_stats   (relative to repo root)
#   ENVS_FILE   default: slurm/manifests/demo_pool/envs.txt

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=slurm/logs/%x_%A_%a.out
#SBATCH --error=slurm/logs/%x_%A_%a.err

# Account and partition are NOT set here so they can be passed via --account /
# --partition flags on the sbatch command line (required on most clusters).

set -euo pipefail

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "SLURM_ARRAY_TASK_ID is not set. Submit with sbatch --array=1-5 ..."
    exit 1
fi

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"

if [[ -f "${SUBMIT_DIR}/setup_shell.sh" ]]; then
    REPO_ROOT="$SUBMIT_DIR"
else
    echo "Could not locate repo root (no setup_shell.sh in ${SUBMIT_DIR})"
    exit 1
fi

# Defaults
POOL_DIR="${POOL_DIR:-/scratch/general/vast/u1472210/demo_pool}"
LABELS_DIR="${LABELS_DIR:-/scratch/general/vast/u1472210/demo_labels}"
OUTPUT_DIR="${OUTPUT_DIR:-results/dataset_stats}"
ENVS_FILE="${ENVS_FILE:-slurm/manifests/demo_pool/envs.txt}"

# Resolve relative paths
[[ "$OUTPUT_DIR" != /* ]] && OUTPUT_DIR="${REPO_ROOT}/${OUTPUT_DIR}"
[[ "$ENVS_FILE"  != /* ]] && ENVS_FILE="${REPO_ROOT}/${ENVS_FILE}"

# Pick the Nth environment
TASK_ENV="$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$ENVS_FILE")"
if [[ -z "$TASK_ENV" ]]; then
    echo "No environment for SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} in ${ENVS_FILE}"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"
OUTPUT_CSV="${OUTPUT_DIR}/${TASK_ENV}.csv"

echo "=============================="
echo "SLURM_JOB_ID  = ${SLURM_JOB_ID:-}"
echo "ARRAY_TASK_ID = ${SLURM_ARRAY_TASK_ID}"
echo "TASK_ENV       = ${TASK_ENV}"
echo "POOL_DIR       = ${POOL_DIR}"
echo "LABELS_DIR     = ${LABELS_DIR}"
echo "OUTPUT_CSV     = ${OUTPUT_CSV}"
echo "=============================="

. "${REPO_ROOT}/setup_shell.sh"

python3 "${REPO_ROOT}/scripts/dataset_stats.py" \
    --pool-dir   "$POOL_DIR" \
    --labels-dir "$LABELS_DIR" \
    --envs       "$TASK_ENV" \
    --output     "$OUTPUT_CSV"
