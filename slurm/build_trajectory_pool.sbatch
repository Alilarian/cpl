#!/bin/bash
# Build trajectory pool (Phase 3) — one SLURM array task per environment.
#
# Usage:
#   sbatch --array=1-5 \
#          --job-name=demo_pool \
#          --time=4:00:00 \
#          --cpus-per-task=4 \
#          --mem=32G \
#          --gres=gpu:1 \
#          slurm/build_trajectory_pool.sbatch
#
# Optional overrides (set before sbatch or via --export):
#   ORACLE_RUNS_BASE   default: runs/runs/chpc/oracle_sac_all
#   OUTPUT_DIR         default: datasets/demo_pool
#   CHECKPOINT_INTERVAL default: 20000
#   N_EPISODES         default: 10
#   N_SEGMENTS         default: 400  (50 ckpts × 400 = 20k segments/env)
#   SEGMENT_LENGTH     default: 50
#   ENVS_FILE          default: slurm/manifests/demo_pool/envs.txt

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

# Locate repo root (same logic as train_array.sbatch)
if [[ -f "${SUBMIT_DIR}/setup_shell.sh" ]]; then
    REPO_ROOT="$SUBMIT_DIR"
else
    echo "Could not locate repo root (no setup_shell.sh in ${SUBMIT_DIR})"
    exit 1
fi

# Defaults (can be overridden via environment variables)
ORACLE_RUNS_BASE="${ORACLE_RUNS_BASE:-runs/runs/chpc/oracle_sac_all}"
OUTPUT_DIR="${OUTPUT_DIR:-datasets/demo_pool}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-20000}"
N_EPISODES="${N_EPISODES:-10}"
N_SEGMENTS="${N_SEGMENTS:-400}"
SEGMENT_LENGTH="${SEGMENT_LENGTH:-50}"
ENVS_FILE="${ENVS_FILE:-slurm/manifests/demo_pool/envs.txt}"

# Resolve relative paths against repo root
[[ "$ORACLE_RUNS_BASE" != /* ]] && ORACLE_RUNS_BASE="${REPO_ROOT}/${ORACLE_RUNS_BASE}"
[[ "$OUTPUT_DIR"       != /* ]] && OUTPUT_DIR="${REPO_ROOT}/${OUTPUT_DIR}"
[[ "$ENVS_FILE"        != /* ]] && ENVS_FILE="${REPO_ROOT}/${ENVS_FILE}"

# Pick the Nth environment from the list
ENV_NAME="$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$ENVS_FILE")"
if [[ -z "$ENV_NAME" ]]; then
    echo "No environment for SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} in ${ENVS_FILE}"
    exit 1
fi

echo "=============================="
echo "SLURM_JOB_ID    = ${SLURM_JOB_ID:-}"
echo "ARRAY_TASK_ID   = ${SLURM_ARRAY_TASK_ID}"
echo "ENV              = ${ENV_NAME}"
echo "ORACLE_RUNS_BASE = ${ORACLE_RUNS_BASE}"
echo "OUTPUT_DIR       = ${OUTPUT_DIR}"
echo "=============================="

. "${REPO_ROOT}/setup_shell.sh"

python3 "${REPO_ROOT}/scripts/build_trajectory_pool.py" \
    --oracle-runs-base "$ORACLE_RUNS_BASE" \
    --envs "$ENV_NAME" \
    --checkpoint-interval "$CHECKPOINT_INTERVAL" \
    --n-episodes-per-checkpoint "$N_EPISODES" \
    --n-segments-per-checkpoint "$N_SEGMENTS" \
    --segment-length "$SEGMENT_LENGTH" \
    --output-dir "$OUTPUT_DIR" \
    --device auto
