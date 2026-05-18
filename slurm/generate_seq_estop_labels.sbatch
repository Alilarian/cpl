#!/bin/bash
# Generate sequential e-stop labels from the trajectory pool — one array task per env.
#
# For each pool trajectory:
#   1. Compute per-step disadvantage  Δ_t = max(0, V*(s_t) - r_t - γV*(s_{t+1}))  (rl_sum)
#   2. Accumulate intervention score  H_t = ρ H_{t-1} + Δ_t
#   3. Convert to stop probability    p_t = σ(λ(H_t − κ))
#   4. Sample stop time               τ ~ first Bernoulli success
#   5. Build K=2 (stop, continue) segment pairs for each decision time t ≤ τ
#
# Output: seq_estop_labels.npz — compatible with CorrBuffer / DemoCPL.
#
# Depends on pool.npz from build_trajectory_pool.sbatch.
#
# Usage:
#   sbatch --array=1-5 \
#          --job-name=seq_estop_labels \
#          --time=2:00:00 \
#          --cpus-per-task=2 \
#          --mem=32G \
#          slurm/generate_seq_estop_labels.sbatch
#
# Optional overrides (set before sbatch or via --export):
#   ORACLE_RUNS_BASE  default: /scratch/general/vast/u1472210/oracle_sac_all
#   POOL_DIR          default: /scratch/general/vast/u1472210/demo_pool
#   OUTPUT_DIR        default: /scratch/general/vast/u1472210/seq_estop_labels
#   RHO               default: 0.8
#   LAM               default: 2.0
#   KAPPA             default: 1.5
#   HORIZON           default: 10
#   MCMC_SAMPLES      default: 64
#   DISCOUNT          default: 0.99
#   MAX_PAIRS         default: (no cap)
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
OUTPUT_DIR="${OUTPUT_DIR:-/scratch/general/vast/u1472210/seq_estop_labels}"
RHO="${RHO:-0.8}"
LAM="${LAM:-2.0}"

# Per-environment kappa (tuned via slurm/tune_mw_kappa.sbatch).
# Fill in the values from tune_kappa log files before submitting.
# Override all at once with --export=ALL,KAPPA=X, or edit the table below.
if [[ -z "${KAPPA:-}" ]]; then
    TASK_ENV_TMP="$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${ENVS_FILE:-slurm/manifests/demo_pool/envs.txt}")"
    case "$TASK_ENV_TMP" in
        mw_bin-picking-v2)   KAPPA="1.8854" ;;
        mw_button-press-v2)  KAPPA="1.6575" ;;
        mw_door-open-v2)     KAPPA="1.5825" ;;
        mw_drawer-open-v2)   KAPPA="1.5236" ;;
        mw_plate-slide-v2)   KAPPA="1.8353" ;;
        *)                   KAPPA="1.5" ;;
    esac
fi
HORIZON="${HORIZON:-10}"
MCMC_SAMPLES="${MCMC_SAMPLES:-64}"
DISCOUNT="${DISCOUNT:-0.99}"
MAX_PAIRS="${MAX_PAIRS:-}"
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
echo "SLURM_JOB_ID   = ${SLURM_JOB_ID:-}"
echo "ARRAY_TASK_ID  = ${SLURM_ARRAY_TASK_ID}"
echo "TASK_ENV       = ${TASK_ENV}"
echo "POOL_PATH      = ${POOL_PATH}"
echo "RUN_DIR        = ${RUN_DIR}"
echo "OUTPUT_DIR     = ${OUTPUT_DIR}"
echo "ρ=${RHO}  λ=${LAM}  κ=${KAPPA}  h=${HORIZON}"
echo "MCMC_SAMPLES   = ${MCMC_SAMPLES}"
echo "MAX_PAIRS      = ${MAX_PAIRS:-no cap}"
echo "=============================="

if [[ ! -f "$POOL_PATH" ]]; then
    echo "ERROR: pool.npz not found at ${POOL_PATH}"
    echo "Run build_trajectory_pool.sbatch first."
    exit 1
fi

. "${REPO_ROOT}/setup_shell.sh"

MAX_PAIRS_ARG=""
if [[ -n "$MAX_PAIRS" ]]; then
    MAX_PAIRS_ARG="--max-pairs $MAX_PAIRS"
fi

python3 "${REPO_ROOT}/scripts/generate_seq_estop_labels.py" \
    --pool-path       "$POOL_PATH"      \
    --run-dir         "$RUN_DIR"        \
    --rho             "$RHO"            \
    --lam             "$LAM"            \
    --kappa           "$KAPPA"          \
    --horizon         "$HORIZON"        \
    --mcmc-samples    "$MCMC_SAMPLES"   \
    --output-dir      "$OUTPUT_DIR"     \
    --device auto                       \
    $MAX_PAIRS_ARG
