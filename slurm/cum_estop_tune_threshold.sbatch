#!/bin/bash
# Phase 2 of the Cumulative E-stop model: calibrate the intervention
# threshold H for one MetaWorld env. Needs no hold/correction simulation --
# only a frozen-oracle V(s) pass + one boundary-state restore per segment --
# so it can run over far more segments per unit of compute than the
# holding-model E-stop's calibration job.
#
# Usage:
#   ENV_NAME=mw_button-press-v2 N_SAMPLES=2000 \
#     sbatch --account=<acct> --partition=<part> [--time=HH:MM:SS] \
#            slurm/cum_estop_tune_threshold.sbatch
#
# Optional env overrides:
#   ENV_NAME          default: mw_drawer-open-v2
#   ORACLE_RUNS_BASE  default: /scratch/general/vast/u1472210/oracle_sac_1m
#   ORACLE_CHECKPOINT default: best_model.pt
#   POOL_DIR          default: /scratch/general/vast/u1472210/mw_de_pool
#   OUTPUT_DIR        default: /scratch/general/vast/u1472210/cum_estop_labels
#   GAMMA             default: 0.99
#   MCMC_SAMPLES      default: 32
#   N_SAMPLES         default: unset (full pool)
#   N_WORKERS         default: $SLURM_CPUS_PER_TASK

#SBATCH --job-name=cum_estop_tune_threshold
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/uufs/chpc.utah.edu/common/home/u1472210/cpl/slurm/logs/%x_%j.out
#SBATCH --error=/uufs/chpc.utah.edu/common/home/u1472210/cpl/slurm/logs/%x_%j.err

set -euo pipefail
mkdir -p "${SLURM_SUBMIT_DIR:-$PWD}/slurm/logs"

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
if [[ ! -f "${SUBMIT_DIR}/setup_shell.sh" ]]; then
    echo "ERROR: no setup_shell.sh in ${SUBMIT_DIR}"
    exit 1
fi
REPO_ROOT="$SUBMIT_DIR"

ENV_NAME="${ENV_NAME:-mw_drawer-open-v2}"
ORACLE_RUNS_BASE="${ORACLE_RUNS_BASE:-/scratch/general/vast/u1472210/oracle_sac_1m}"
ORACLE_CHECKPOINT="${ORACLE_CHECKPOINT:-best_model.pt}"
POOL_DIR="${POOL_DIR:-/scratch/general/vast/u1472210/mw_de_pool}"
OUTPUT_DIR="${OUTPUT_DIR:-/scratch/general/vast/u1472210/cum_estop_labels}"
GAMMA="${GAMMA:-0.99}"
MCMC_SAMPLES="${MCMC_SAMPLES:-32}"
N_SAMPLES="${N_SAMPLES:-}"
N_WORKERS="${N_WORKERS:-${SLURM_CPUS_PER_TASK:-8}}"

[[ "$ORACLE_RUNS_BASE" != /* ]] && ORACLE_RUNS_BASE="${REPO_ROOT}/${ORACLE_RUNS_BASE}"
[[ "$POOL_DIR"         != /* ]] && POOL_DIR="${REPO_ROOT}/${POOL_DIR}"
[[ "$OUTPUT_DIR"       != /* ]] && OUTPUT_DIR="${REPO_ROOT}/${OUTPUT_DIR}"

POOL_PATH="${POOL_DIR}/${ENV_NAME}/pool.npz"
ORACLE_DIR="${ORACLE_RUNS_BASE}/${ENV_NAME}"
OUT_PATH="${OUTPUT_DIR}/${ENV_NAME}/cum_estop_calibration.npz"

echo "=============================="
echo "ENV_NAME      = ${ENV_NAME}"
echo "POOL_PATH     = ${POOL_PATH}"
echo "ORACLE_DIR    = ${ORACLE_DIR}"
echo "GAMMA         = ${GAMMA}"
echo "MCMC_SAMPLES  = ${MCMC_SAMPLES}"
echo "N_SAMPLES     = ${N_SAMPLES:-<unset, full pool>}"
echo "N_WORKERS     = ${N_WORKERS}"
echo "OUT_PATH      = ${OUT_PATH}"
echo "=============================="

if [[ ! -f "$POOL_PATH" ]]; then
    echo "ERROR: pool.npz not found: ${POOL_PATH}"
    exit 1
fi

. "${REPO_ROOT}/setup_shell.sh"
mkdir -p "$(dirname "$OUT_PATH")"

python3 "${REPO_ROOT}/scripts/tune_cum_estop_threshold.py" \
    --pool-path "$POOL_PATH" \
    --run-dir "$ORACLE_DIR" \
    --oracle-checkpoint "$ORACLE_CHECKPOINT" \
    --gamma "$GAMMA" \
    --mcmc-samples "$MCMC_SAMPLES" \
    --n-workers "$N_WORKERS" \
    --output "$OUT_PATH" \
    ${N_SAMPLES:+--n-samples "$N_SAMPLES"}
