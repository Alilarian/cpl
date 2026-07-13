#!/bin/bash
# Tune the seq-estop κ (kappa) hyperparameter for a MetaWorld environment.
#
# Subsamples --n-sample segments from the pool, computes Δ_t via MCMC with
# the oracle critic, sweeps κ candidates (percentiles of peak-H), and prints
# a recommendation with estimated seq_estop pair counts scaled to the full pool.
#
# Usage:
#   sbatch --account=<account> --partition=<partition> \
#          [--time=HH:MM:SS] [--cpus-per-task=N] [--mem=XG] \
#          slurm/mw_tune_kappa.sbatch
#
# Resource overrides (command-line flags override #SBATCH defaults below):
#   --time=HH:MM:SS     default: 2:00:00
#   --cpus-per-task=N   default: 4
#   --mem=XG            default: 32G
#
# Optional env overrides:
#   ENV_NAME          default: mw_drawer-open-v2
#   ORACLE_RUNS_BASE  default: runs/runs/chpc/oracle_sac_seeds  (relative to repo root)
#   ORACLE_SEED       default: seed-1
#   POOL_DIR          default: /scratch/general/vast/u1472210/mw_de_pool
#   N_SAMPLE          default: 30000   subsample for speed; scales to full pool
#   RHO               default: 0.8
#   LAM               default: 2.0
#   HORIZON           default: 10
#   TARGET_PAIRS      default: 10000   minimum est. pairs from subsample
#   MCMC_SAMPLES      default: 32

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
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
ORACLE_RUNS_BASE="${ORACLE_RUNS_BASE:-runs/runs/chpc/oracle_sac_seeds}"
ORACLE_SEED="${ORACLE_SEED:-seed-1}"
POOL_DIR="${POOL_DIR:-/scratch/general/vast/u1472210/mw_de_pool}"
N_SAMPLE="${N_SAMPLE:-30000}"
RHO="${RHO:-0.8}"
LAM="${LAM:-2.0}"
HORIZON="${HORIZON:-10}"
TARGET_PAIRS="${TARGET_PAIRS:-10000}"
MCMC_SAMPLES="${MCMC_SAMPLES:-32}"

[[ "$ORACLE_RUNS_BASE" != /* ]] && ORACLE_RUNS_BASE="${REPO_ROOT}/${ORACLE_RUNS_BASE}"
[[ "$POOL_DIR"         != /* ]] && POOL_DIR="${REPO_ROOT}/${POOL_DIR}"

POOL_PATH="${POOL_DIR}/${ENV_NAME}/pool.npz"
ORACLE_DIR="${ORACLE_RUNS_BASE}/${ENV_NAME}/${ORACLE_SEED}"

echo "=============================="
echo "SLURM_JOB_ID  = ${SLURM_JOB_ID:-}"
echo "ENV_NAME      = ${ENV_NAME}"
echo "ORACLE_DIR    = ${ORACLE_DIR}"
echo "POOL_PATH     = ${POOL_PATH}"
echo "N_SAMPLE      = ${N_SAMPLE}"
echo "RHO           = ${RHO}"
echo "LAM           = ${LAM}"
echo "HORIZON       = ${HORIZON}"
echo "TARGET_PAIRS  = ${TARGET_PAIRS}"
echo "MCMC_SAMPLES  = ${MCMC_SAMPLES}"
echo "=============================="

if [[ ! -f "$POOL_PATH" ]]; then
    echo "ERROR: pool not found: ${POOL_PATH}"
    echo "Run mw_de_build_pool.sbatch first."
    exit 1
fi

if [[ ! -d "$ORACLE_DIR" ]]; then
    echo "ERROR: oracle dir not found: ${ORACLE_DIR}"
    exit 1
fi

. "${REPO_ROOT}/setup_shell.sh"

python3 "${REPO_ROOT}/scripts/tune_mw_kappa.py" \
    --pool-path      "$POOL_PATH"   \
    --run-dir        "$ORACLE_DIR"  \
    --rho            "$RHO"         \
    --lam            "$LAM"         \
    --horizon        "$HORIZON"     \
    --n-sample       "$N_SAMPLE"    \
    --mcmc-samples   "$MCMC_SAMPLES" \
    --target-pairs   "$TARGET_PAIRS" \
    --device         cpu
