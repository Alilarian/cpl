#!/bin/bash
# Score trajectory pool segments with oracle rl_sum and save top-X% as BC dataset.
#
# No rollouts — segments are scored directly in batches through the oracle critic.
# Much faster than generate_demo/corr_labels.sbatch (~5-10 min per env on GPU).
#
# Depends on pool.npz files from build_trajectory_pool.sbatch (Phase 3).
#
# Usage:
#   sbatch --array=1-5 \
#          --job-name=score_bc_pool \
#          --time=1:00:00 \
#          --cpus-per-task=2 \
#          --mem=16G \
#          --gres=gpu:1 \
#          slurm/score_pool_for_bc.sbatch
#
# Optional overrides (set before sbatch or via --export):
#   ORACLE_RUNS_BASE  default: /scratch/general/vast/u1472210/oracle_sac_all
#   POOL_DIR          default: /scratch/general/vast/u1472210/demo_pool
#   OUTPUT_DIR        default: /scratch/general/vast/u1472210/bc_pool
#   TOP_PCT           default: 20.0   (top 20% of pool by rl_sum)
#   BATCH_SIZE        default: 256    (segments per GPU forward pass)
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
OUTPUT_DIR="${OUTPUT_DIR:-/scratch/general/vast/u1472210/bc_pool}"
TOP_PCT="${TOP_PCT:-20.0}"
BATCH_SIZE="${BATCH_SIZE:-256}"
MCMC_SAMPLES="${MCMC_SAMPLES:-64}"
ENVS_FILE="${ENVS_FILE:-slurm/manifests/demo_pool/envs.txt}"

# Resolve relative paths
[[ "$ORACLE_RUNS_BASE" != /* ]] && ORACLE_RUNS_BASE="${REPO_ROOT}/${ORACLE_RUNS_BASE}"
[[ "$POOL_DIR"         != /* ]] && POOL_DIR="${REPO_ROOT}/${POOL_DIR}"
[[ "$OUTPUT_DIR"       != /* ]] && OUTPUT_DIR="${REPO_ROOT}/${OUTPUT_DIR}"
[[ "$ENVS_FILE"        != /* ]] && ENVS_FILE="${REPO_ROOT}/${ENVS_FILE}"

# Pick the Nth environment
TASK_ENV="$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$ENVS_FILE")"
if [[ -z "$TASK_ENV" ]]; then
    echo "No environment for SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} in ${ENVS_FILE}"
    exit 1
fi

POOL_PATH="${POOL_DIR}/${TASK_ENV}/pool.npz"
RUN_DIR="${ORACLE_RUNS_BASE}/${TASK_ENV}"

echo "=============================="
echo "SLURM_JOB_ID      = ${SLURM_JOB_ID:-}"
echo "ARRAY_TASK_ID     = ${SLURM_ARRAY_TASK_ID}"
echo "TASK_ENV          = ${TASK_ENV}"
echo "POOL_PATH         = ${POOL_PATH}"
echo "RUN_DIR           = ${RUN_DIR}"
echo "OUTPUT_DIR        = ${OUTPUT_DIR}"
echo "TOP_PCT           = ${TOP_PCT}"
echo "BATCH_SIZE        = ${BATCH_SIZE}"
echo "MCMC_SAMPLES      = ${MCMC_SAMPLES}"
echo "=============================="

if [[ ! -f "$POOL_PATH" ]]; then
    echo "ERROR: pool.npz not found at ${POOL_PATH}"
    echo "Run build_trajectory_pool.sbatch first."
    exit 1
fi

. "${REPO_ROOT}/setup_shell.sh"

python3 "${REPO_ROOT}/scripts/score_pool_for_bc.py" \
    --pool-path     "$POOL_PATH" \
    --run-dir       "$RUN_DIR" \
    --top-pct       "$TOP_PCT" \
    --batch-size    "$BATCH_SIZE" \
    --mcmc-samples  "$MCMC_SAMPLES" \
    --output-dir    "$OUTPUT_DIR" \
    --device auto
