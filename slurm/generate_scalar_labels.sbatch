#!/bin/bash
# Generate scalar feedback labels from the trajectory pool — one array task per env.
#
# Scores each pool segment with oracle rl_sum over first --segment-len steps,
# normalizes to [-1, 1], adds Gaussian noise, then extracts pairwise preferences
# within a sliding window of W adjacent pool segments.
#
# Output: scalar_labels.npz compatible with CorrBuffer / DemoCPL (bc_steps=0).
#
# Depends on pool.npz from build_trajectory_pool.sbatch.
#
# Usage:
#   sbatch --array=1-5 \
#          --job-name=scalar_labels \
#          --time=4:00:00 \
#          --cpus-per-task=4 \
#          --mem=48G \
#          --gres=gpu:1 \
#          slurm/generate_scalar_labels.sbatch
#
# Optional overrides (set before sbatch or via --export):
#   ORACLE_RUNS_BASE  default: /scratch/general/vast/u1472210/oracle_sac_all
#   POOL_DIR          default: /scratch/general/vast/u1472210/demo_pool
#   OUTPUT_DIR        default: /scratch/general/vast/u1472210/scalar_labels
#   SEGMENT_LEN       default: 20    (h steps scored per segment)
#   WINDOW_SIZE       default: 10    (W pool segments per window)
#   NOISE_STD         default: 0.1
#   SCALAR_DELTA      default: 0.0   (indifference threshold)
#   N_PAIRS           default: 300000
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
OUTPUT_DIR="${OUTPUT_DIR:-/scratch/general/vast/u1472210/scalar_labels}"
SEGMENT_LEN="${SEGMENT_LEN:-20}"
WINDOW_SIZE="${WINDOW_SIZE:-10}"
NOISE_STD="${NOISE_STD:-0.1}"
SCALAR_DELTA="${SCALAR_DELTA:-0.0}"
N_PAIRS="${N_PAIRS:-300000}"
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
echo "SLURM_JOB_ID   = ${SLURM_JOB_ID:-}"
echo "ARRAY_TASK_ID  = ${SLURM_ARRAY_TASK_ID}"
echo "TASK_ENV       = ${TASK_ENV}"
echo "POOL_PATH      = ${POOL_PATH}"
echo "RUN_DIR        = ${RUN_DIR}"
echo "OUTPUT_DIR     = ${OUTPUT_DIR}"
echo "SEGMENT_LEN    = ${SEGMENT_LEN}"
echo "WINDOW_SIZE    = ${WINDOW_SIZE}"
echo "NOISE_STD      = ${NOISE_STD}"
echo "SCALAR_DELTA   = ${SCALAR_DELTA}"
echo "N_PAIRS        = ${N_PAIRS}"
echo "MCMC_SAMPLES   = ${MCMC_SAMPLES}"
echo "=============================="

if [[ ! -f "$POOL_PATH" ]]; then
    echo "ERROR: pool.npz not found at ${POOL_PATH}"
    echo "Run build_trajectory_pool.sbatch first."
    exit 1
fi

. "${REPO_ROOT}/setup_shell.sh"

python3 "${REPO_ROOT}/scripts/generate_scalar_labels.py" \
    --pool-path       "$POOL_PATH"      \
    --run-dir         "$RUN_DIR"        \
    --segment-len     "$SEGMENT_LEN"    \
    --window-size     "$WINDOW_SIZE"    \
    --noise-std       "$NOISE_STD"      \
    --scalar-delta    "$SCALAR_DELTA"   \
    --n-pairs         "$N_PAIRS"        \
    --mcmc-samples    "$MCMC_SAMPLES"   \
    --output-dir      "$OUTPUT_DIR"     \
    --device auto
