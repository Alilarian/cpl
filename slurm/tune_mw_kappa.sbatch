#!/bin/bash
# Tune e-stop kappa for MetaWorld seq-estop — one array task per env.
#
# Subsamples --n-sample trajectories from the full pool (fast, ~10-30 min on GPU),
# computes per-step disadvantage via MCMC, sweeps percentile-based kappa candidates,
# and prints the recommended kappa + estimated pairs for the full pool.
#
# Run this BEFORE generate_seq_estop_labels.sbatch.
# Copy the recommended KAPPA value into generate_seq_estop_labels.sbatch.
#
# Usage:
#   sbatch --array=1-5 \
#          --job-name=tune_kappa \
#          --time=1:00:00 \
#          --cpus-per-task=4 \
#          --mem=32G \
#          --gres=gpu:1 \
#          slurm/tune_mw_kappa.sbatch
#
# Optional overrides:
#   ORACLE_RUNS_BASE  default: /scratch/general/vast/u1472210/oracle_sac_all
#   POOL_DIR          default: /scratch/general/vast/u1472210/demo_pool
#   N_SAMPLE          default: 30000   (subsample for tuning)
#   TARGET_PAIRS      default: 10000   (target pairs from N_SAMPLE)
#   RHO               default: 0.8
#   LAM               default: 2.0
#   HORIZON           default: 10
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
N_SAMPLE="${N_SAMPLE:-30000}"
TARGET_PAIRS="${TARGET_PAIRS:-10000}"
RHO="${RHO:-0.8}"
LAM="${LAM:-2.0}"
HORIZON="${HORIZON:-10}"
MCMC_SAMPLES="${MCMC_SAMPLES:-64}"
ENVS_FILE="${ENVS_FILE:-slurm/manifests/demo_pool/envs.txt}"

# Resolve relative paths
[[ "$ORACLE_RUNS_BASE" != /* ]] && ORACLE_RUNS_BASE="${REPO_ROOT}/${ORACLE_RUNS_BASE}"
[[ "$POOL_DIR"         != /* ]] && POOL_DIR="${REPO_ROOT}/${POOL_DIR}"
[[ "$ENVS_FILE"        != /* ]] && ENVS_FILE="${REPO_ROOT}/${ENVS_FILE}"

TASK_ENV="$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$ENVS_FILE")"
if [[ -z "$TASK_ENV" ]]; then
    echo "No environment for SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} in ${ENVS_FILE}"
    exit 1
fi

POOL_PATH="${POOL_DIR}/${TASK_ENV}/pool.npz"
RUN_DIR="${ORACLE_RUNS_BASE}/${TASK_ENV}"

echo "=============================="
echo "SLURM_JOB_ID   = ${SLURM_JOB_ID:-}"
echo "ARRAY_TASK_ID  = ${SLURM_ARRAY_TASK_ID}"
echo "TASK_ENV       = ${TASK_ENV}"
echo "POOL_PATH      = ${POOL_PATH}"
echo "RUN_DIR        = ${RUN_DIR}"
echo "N_SAMPLE       = ${N_SAMPLE}"
echo "TARGET_PAIRS   = ${TARGET_PAIRS}"
echo "ρ=${RHO}  λ=${LAM}  h=${HORIZON}"
echo "MCMC_SAMPLES   = ${MCMC_SAMPLES}"
echo "=============================="

if [[ ! -f "$POOL_PATH" ]]; then
    echo "ERROR: pool.npz not found at ${POOL_PATH}"
    echo "Run build_trajectory_pool.sbatch first."
    exit 1
fi

. "${REPO_ROOT}/setup_shell.sh"

python3 "${REPO_ROOT}/scripts/tune_mw_kappa.py" \
    --pool-path      "$POOL_PATH"   \
    --run-dir        "$RUN_DIR"     \
    --rho            "$RHO"         \
    --lam            "$LAM"         \
    --horizon        "$HORIZON"     \
    --n-sample       "$N_SAMPLE"    \
    --target-pairs   "$TARGET_PAIRS" \
    --mcmc-samples   "$MCMC_SAMPLES" \
    --device auto
