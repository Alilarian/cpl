#!/bin/bash
# Phase 4 of the Cumulative E-stop model: select checkpoint bands (Pi_M, Pi_B)
# for one MetaWorld env from its oracle training run's log.csv. Cheap (just
# parses a CSV) -- runs as a single short CPU job, one per env.
#
# Usage:
#   ENV_NAME=mw_button-press-v2 sbatch --account=<acct> --partition=<part> \
#       slurm/cum_estop_select_bands.sbatch
#
# Optional env overrides:
#   ENV_NAME          default: mw_drawer-open-v2
#   ORACLE_RUNS_BASE  default: /scratch/general/vast/u1472210/oracle_sac_1m
#   OUTPUT_DIR        default: /scratch/general/vast/u1472210/cum_estop_labels
#   WEAK_MAX_SUCCESS, MID_MIN_SUCCESS, MID_MAX_SUCCESS : band thresholds,
#       see scripts/select_checkpoint_bands.py --help for defaults

#SBATCH --job-name=cum_estop_select_bands
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
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
OUTPUT_DIR="${OUTPUT_DIR:-/scratch/general/vast/u1472210/cum_estop_labels}"

[[ "$ORACLE_RUNS_BASE" != /* ]] && ORACLE_RUNS_BASE="${REPO_ROOT}/${ORACLE_RUNS_BASE}"
[[ "$OUTPUT_DIR"       != /* ]] && OUTPUT_DIR="${REPO_ROOT}/${OUTPUT_DIR}"

ORACLE_DIR="${ORACLE_RUNS_BASE}/${ENV_NAME}"
OUT_PATH="${OUTPUT_DIR}/${ENV_NAME}/checkpoint_bands.json"

echo "ENV_NAME   = ${ENV_NAME}"
echo "ORACLE_DIR = ${ORACLE_DIR}"
echo "OUT_PATH   = ${OUT_PATH}"

. "${REPO_ROOT}/setup_shell.sh"
mkdir -p "$(dirname "$OUT_PATH")"

python3 "${REPO_ROOT}/scripts/select_checkpoint_bands.py" \
    --run-dir "$ORACLE_DIR" \
    --output  "$OUT_PATH" \
    ${WEAK_MAX_SUCCESS:+--weak-max-success "$WEAK_MAX_SUCCESS"} \
    ${MID_MIN_SUCCESS:+--mid-min-success "$MID_MIN_SUCCESS"} \
    ${MID_MAX_SUCCESS:+--mid-max-success "$MID_MAX_SUCCESS"}
