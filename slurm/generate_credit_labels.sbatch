#!/bin/bash
# Generate credit assignment feedback labels from the trajectory pool — one array task per env.
#
# For each pool trajectory, scores all C = T-k+1 contiguous windows of length k
# using a precomputed per-step V table (MCMC from oracle SAC, same cost as scoring
# the pool once), then selects the highest-scoring window as the oracle credit.
#
# Output: credit_labels.npz with (N, C, k, obs_dim) format for PMCreditAssignmentBuffer
#         and CreditAssignmentCPL (bc_steps=0).
#
# Depends on pool.npz from build_trajectory_pool.sbatch.
#
# Usage:
#   sbatch --array=1-5 \
#          --job-name=credit_labels \
#          --time=4:00:00 \
#          --cpus-per-task=4 \
#          --mem=64G \
#          --gres=gpu:1 \
#          slurm/generate_credit_labels.sbatch
#
# Optional overrides (set before sbatch or via --export):
#   ORACLE_RUNS_BASE  default: /scratch/general/vast/u1472210/oracle_sac_all
#   POOL_DIR          default: /scratch/general/vast/u1472210/demo_pool
#   OUTPUT_DIR        default: /scratch/general/vast/u1472210/credit_labels
#   SUBSEGMENT_LEN    default: 20    (window length k; C = T-k+1)
#   N_EXAMPLES        default: 300000
#   MCMC_SAMPLES      default: 64
#   ENVS_FILE         default: slurm/manifests/demo_pool/envs.txt

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

if [[ -f "${SUBMIT_DIR}/setup_shell.sh" ]]; then
    REPO_ROOT="$SUBMIT_DIR"
else
    echo "Could not locate repo root (no setup_shell.sh in ${SUBMIT_DIR})"
    exit 1
fi

# Defaults
ORACLE_RUNS_BASE="${ORACLE_RUNS_BASE:-/scratch/general/vast/u1472210/oracle_sac_all}"
POOL_DIR="${POOL_DIR:-/scratch/general/vast/u1472210/demo_pool}"
OUTPUT_DIR="${OUTPUT_DIR:-/scratch/general/vast/u1472210/credit_labels}"
SUBSEGMENT_LEN="${SUBSEGMENT_LEN:-20}"
N_EXAMPLES="${N_EXAMPLES:-300000}"
MCMC_SAMPLES="${MCMC_SAMPLES:-64}"
ENVS_FILE="${ENVS_FILE:-slurm/manifests/demo_pool/envs.txt}"

# Resolve relative paths
[[ "$ORACLE_RUNS_BASE" != /* ]] && ORACLE_RUNS_BASE="${REPO_ROOT}/${ORACLE_RUNS_BASE}"
[[ "$POOL_DIR"         != /* ]] && POOL_DIR="${REPO_ROOT}/${POOL_DIR}"
[[ "$OUTPUT_DIR"       != /* ]] && OUTPUT_DIR="${REPO_ROOT}/${OUTPUT_DIR}"
[[ "$ENVS_FILE"        != /* ]] && ENVS_FILE="${REPO_ROOT}/${ENVS_FILE}"

TASK_ENV="$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$ENVS_FILE")"
if [[ -z "$TASK_ENV" ]]; then
    echo "No environment for SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} in ${ENVS_FILE}"
    exit 1
fi

POOL_PATH="${POOL_DIR}/${TASK_ENV}/pool.npz"
RUN_DIR="${ORACLE_RUNS_BASE}/${TASK_ENV}"

echo "=============================="
echo "SLURM_JOB_ID    = ${SLURM_JOB_ID:-}"
echo "ARRAY_TASK_ID   = ${SLURM_ARRAY_TASK_ID}"
echo "TASK_ENV        = ${TASK_ENV}"
echo "POOL_PATH       = ${POOL_PATH}"
echo "RUN_DIR         = ${RUN_DIR}"
echo "OUTPUT_DIR      = ${OUTPUT_DIR}"
echo "SUBSEGMENT_LEN  = ${SUBSEGMENT_LEN}"
echo "N_EXAMPLES      = ${N_EXAMPLES}"
echo "MCMC_SAMPLES    = ${MCMC_SAMPLES}"
echo "=============================="

if [[ ! -f "$POOL_PATH" ]]; then
    echo "ERROR: pool.npz not found at ${POOL_PATH}"
    echo "Run build_trajectory_pool.sbatch first."
    exit 1
fi

. "${REPO_ROOT}/setup_shell.sh"

python3 "${REPO_ROOT}/scripts/generate_credit_labels.py" \
    --pool-path       "$POOL_PATH"      \
    --run-dir         "$RUN_DIR"        \
    --subsegment-len  "$SUBSEGMENT_LEN" \
    --n-examples      "$N_EXAMPLES"     \
    --mcmc-samples    "$MCMC_SAMPLES"   \
    --output-dir      "$OUTPUT_DIR"     \
    --device auto
