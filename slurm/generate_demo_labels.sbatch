#!/bin/bash
# Generate demonstrative feedback labels (Phase 4) — one SLURM array task per environment.
#
# Depends on pool.npz files produced by build_trajectory_pool.sbatch (Phase 3).
#
# Usage:
#   sbatch --array=1-5 \
#          --job-name=demo_labels \
#          --time=8:00:00 \
#          --cpus-per-task=4 \
#          --mem=32G \
#          --gres=gpu:1 \
#          slurm/generate_demo_labels.sbatch
#
# Optional overrides (set before sbatch or via --export):
#   ORACLE_RUNS_BASE      default: /scratch/general/vast/u1472210/oracle_sac_all
#   POOL_DIR              default: datasets/demo_pool
#   OUTPUT_DIR            default: datasets/demo_labels
#   N_COUNTERFACTUALS     default: 6  (K=7: 1 demo + 6 counterfactuals)
#   CHECKPOINTS_PER_TIER  default: 2
#   MIN_ADV_GAP           default: 0.0
#   CHECKPOINT_INTERVAL   default: 20000
#   MCMC_SAMPLES          default: 64
#   ENVS_FILE             default: slurm/manifests/demo_pool/envs.txt

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

# Locate repo root
if [[ -f "${SUBMIT_DIR}/setup_shell.sh" ]]; then
    REPO_ROOT="$SUBMIT_DIR"
else
    echo "Could not locate repo root (no setup_shell.sh in ${SUBMIT_DIR})"
    exit 1
fi

# Defaults
ORACLE_RUNS_BASE="${ORACLE_RUNS_BASE:-/scratch/general/vast/u1472210/oracle_sac_all}"
POOL_DIR="${POOL_DIR:-datasets/demo_pool}"
OUTPUT_DIR="${OUTPUT_DIR:-datasets/demo_labels}"
N_COUNTERFACTUALS="${N_COUNTERFACTUALS:-6}"
CHECKPOINTS_PER_TIER="${CHECKPOINTS_PER_TIER:-2}"
MIN_ADV_GAP="${MIN_ADV_GAP:-0.0}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-20000}"
MCMC_SAMPLES="${MCMC_SAMPLES:-64}"
ENVS_FILE="${ENVS_FILE:-slurm/manifests/demo_pool/envs.txt}"

# Resolve relative paths
[[ "$ORACLE_RUNS_BASE" != /* ]] && ORACLE_RUNS_BASE="${REPO_ROOT}/${ORACLE_RUNS_BASE}"
[[ "$POOL_DIR"         != /* ]] && POOL_DIR="${REPO_ROOT}/${POOL_DIR}"
[[ "$OUTPUT_DIR"       != /* ]] && OUTPUT_DIR="${REPO_ROOT}/${OUTPUT_DIR}"
[[ "$ENVS_FILE"        != /* ]] && ENVS_FILE="${REPO_ROOT}/${ENVS_FILE}"

# Pick the Nth environment
ENV_NAME="$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$ENVS_FILE")"
if [[ -z "$ENV_NAME" ]]; then
    echo "No environment for SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} in ${ENVS_FILE}"
    exit 1
fi

POOL_PATH="${POOL_DIR}/${ENV_NAME}/pool.npz"
RUN_DIR="${ORACLE_RUNS_BASE}/${ENV_NAME}"

echo "=============================="
echo "SLURM_JOB_ID         = ${SLURM_JOB_ID:-}"
echo "ARRAY_TASK_ID        = ${SLURM_ARRAY_TASK_ID}"
echo "ENV                  = ${ENV_NAME}"
echo "POOL_PATH            = ${POOL_PATH}"
echo "RUN_DIR              = ${RUN_DIR}"
echo "N_COUNTERFACTUALS    = ${N_COUNTERFACTUALS}  (K=$((N_COUNTERFACTUALS + 1)))"
echo "CHECKPOINTS_PER_TIER = ${CHECKPOINTS_PER_TIER}"
echo "OUTPUT_DIR           = ${OUTPUT_DIR}"
echo "=============================="

if [[ ! -f "$POOL_PATH" ]]; then
    echo "ERROR: pool.npz not found at ${POOL_PATH}"
    echo "Run build_trajectory_pool.sbatch first."
    exit 1
fi

. "${REPO_ROOT}/setup_shell.sh"

python3 "${REPO_ROOT}/scripts/generate_demo_labels.py" \
    --pool-path "$POOL_PATH" \
    --run-dir "$RUN_DIR" \
    --n-counterfactuals "$N_COUNTERFACTUALS" \
    --checkpoints-per-tier "$CHECKPOINTS_PER_TIER" \
    --min-adv-gap "$MIN_ADV_GAP" \
    --checkpoint-interval "$CHECKPOINT_INTERVAL" \
    --mcmc-samples "$MCMC_SAMPLES" \
    --output-dir "$OUTPUT_DIR" \
    --device auto
