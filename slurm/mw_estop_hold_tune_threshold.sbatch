#!/bin/bash
# Calibrate the holding-model E-stop threshold c for one or more MetaWorld
# envs (array job, one task per env) via scripts/tune_estop_hold_threshold.py.
#
# Usage (4 envs, one array task each):
#   ENVS="mw_bin-picking-v2 mw_button-press-v2 mw_door-open-v2 mw_plate-slide-v2" \
#     sbatch --array=0-3 --account=<acct> --partition=<partition> --qos=<qos> \
#            [--time=HH:MM:SS] [--cpus-per-task=N] [--mem=XG] \
#            slurm/mw_estop_hold_tune_threshold.sbatch
#
# ENVS and --array must agree: --array=0-(N_ENVS-1) (0-indexed position into
# ENVS, not env name).
#
# Output goes to slurm/logs/mw_estop_hold_tune_threshold_<jobid>_<taskid>.out --
# grep "M_i statistics" -A6 on it for the calibration table.
#
# Resource overrides (command-line flags override #SBATCH defaults below):
#   --time=HH:MM:SS     default: 01:00:00
#   --cpus-per-task=N   default: 8
#   --mem=XG            default: 32G
#
# Optional env overrides:
#   ENVS               required: space-separated env names
#   POOL_DIR           default: /scratch/general/vast/u1472210/mw_de_pool
#   ORACLE_RUNS_BASE   default: /scratch/general/vast/u1472210/oracle_sac_1m
#   ORACLE_SEED        default: ""  (flat <env>/ layout, no seed subdir)
#   OUTPUT_DIR         default: /scratch/general/vast/u1472210/estop_hold_labels
#   N_SEGMENTS         default: 500
#   N_WORKERS          default: $SLURM_CPUS_PER_TASK
#   MIN_HORIZON        default: 5
#   CONDA_ENV_NAME     default: cpl (set to your GPU-capable env if needed,
#                                     e.g. cpl_gpu -- this job itself is CPU-only)

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/uufs/chpc.utah.edu/common/home/u1472210/cpl/slurm/logs/%x_%A_%a.out
#SBATCH --error=/uufs/chpc.utah.edu/common/home/u1472210/cpl/slurm/logs/%x_%A_%a.err

set -euo pipefail

mkdir -p "${SLURM_SUBMIT_DIR:-$PWD}/slurm/logs"

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "ERROR: submit with --array=0-(N_ENVS-1), matching ENVS"
    exit 1
fi

if [[ -z "${ENVS:-}" ]]; then
    echo "ERROR: set ENVS=\"env1 env2 ...\""
    exit 1
fi

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
if [[ ! -f "${SUBMIT_DIR}/setup_shell.sh" ]]; then
    echo "ERROR: no setup_shell.sh in ${SUBMIT_DIR}"
    exit 1
fi
REPO_ROOT="$SUBMIT_DIR"

read -r -a ENV_ARR <<< "$ENVS"
ENV_NAME="${ENV_ARR[$SLURM_ARRAY_TASK_ID]:-}"
if [[ -z "$ENV_NAME" ]]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} out of range for ENVS=(${ENVS})"
    exit 1
fi

POOL_DIR="${POOL_DIR:-/scratch/general/vast/u1472210/mw_de_pool}"
ORACLE_RUNS_BASE="${ORACLE_RUNS_BASE:-/scratch/general/vast/u1472210/oracle_sac_1m}"
ORACLE_SEED="${ORACLE_SEED-}"
OUTPUT_DIR="${OUTPUT_DIR:-/scratch/general/vast/u1472210/estop_hold_labels}"
N_SEGMENTS="${N_SEGMENTS:-500}"
N_WORKERS="${N_WORKERS:-${SLURM_CPUS_PER_TASK:-8}}"
MIN_HORIZON="${MIN_HORIZON:-5}"

[[ "$POOL_DIR"         != /* ]] && POOL_DIR="${REPO_ROOT}/${POOL_DIR}"
[[ "$ORACLE_RUNS_BASE" != /* ]] && ORACLE_RUNS_BASE="${REPO_ROOT}/${ORACLE_RUNS_BASE}"
[[ "$OUTPUT_DIR"       != /* ]] && OUTPUT_DIR="${REPO_ROOT}/${OUTPUT_DIR}"

POOL_PATH="${POOL_DIR}/${ENV_NAME}/pool.npz"
if [[ -n "$ORACLE_SEED" ]]; then
    ORACLE_DIR="${ORACLE_RUNS_BASE}/${ENV_NAME}/${ORACLE_SEED}"
else
    ORACLE_DIR="${ORACLE_RUNS_BASE}/${ENV_NAME}"
fi
OUT_PATH="${OUTPUT_DIR}/${ENV_NAME}/calibration.npz"

echo "=============================="
echo "SLURM_JOB_ID  = ${SLURM_JOB_ID:-}"
echo "ARRAY_TASK_ID = ${SLURM_ARRAY_TASK_ID}"
echo "ENV_NAME      = ${ENV_NAME}"
echo "POOL_PATH     = ${POOL_PATH}"
echo "ORACLE_DIR    = ${ORACLE_DIR}"
echo "OUT_PATH      = ${OUT_PATH}"
echo "N_SEGMENTS    = ${N_SEGMENTS}"
echo "N_WORKERS     = ${N_WORKERS}"
echo "=============================="

if [[ ! -f "$POOL_PATH" ]]; then
    echo "ERROR: pool.npz not found: ${POOL_PATH}"
    exit 1
fi
if [[ ! -d "$ORACLE_DIR" ]]; then
    echo "ERROR: oracle dir not found: ${ORACLE_DIR}"
    exit 1
fi

. "${REPO_ROOT}/setup_shell.sh"

python3 "${REPO_ROOT}/scripts/tune_estop_hold_threshold.py" \
    --pool-path   "$POOL_PATH" \
    --run-dir     "$ORACLE_DIR" \
    --n-segments  "$N_SEGMENTS" \
    --n-workers   "$N_WORKERS" \
    --min-horizon "$MIN_HORIZON" \
    --output      "$OUT_PATH"

echo "Done -- calibration for ${ENV_NAME} written to ${OUT_PATH}"
