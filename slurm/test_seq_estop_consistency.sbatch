#!/bin/bash
# Run tests/test_seq_estop_consistency.py on CHPC.
#
# Part A (toy-trajectory regression tests) always runs. Part B (real-data
# statistical test) auto-activates here because the default LABELS_PATH /
# ORACLE_DIR point at CHPC scratch/run paths -- it re-scores a subsample of
# the real seq_estop_labels.npz with the real oracle and runs a binomial
# test of continue-pair advantage-consistency against the null p=0.5.
#
# Usage:
#   sbatch --account=<account> --partition=<partition> \
#          [--time=HH:MM:SS] [--cpus-per-task=N] [--mem=XG] \
#          slurm/test_seq_estop_consistency.sbatch
#
# Optional env overrides:
#   ENV_NAME          default: mw_drawer-open-v2
#   ORACLE_RUNS_BASE  default: runs/runs/chpc/oracle_sac_seeds  (relative to repo root)
#   ORACLE_SEED       default: seed-1
#   LABELS_DIR        default: /scratch/general/vast/u1472210/mw_de_labels
#   MAX_PAIRS         default: 5000   (subsample scored by the binomial test)

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:00:00
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
LABELS_DIR="${LABELS_DIR:-/scratch/general/vast/u1472210/mw_de_labels}"
MAX_PAIRS="${MAX_PAIRS:-5000}"

[[ "$ORACLE_RUNS_BASE" != /* ]] && ORACLE_RUNS_BASE="${REPO_ROOT}/${ORACLE_RUNS_BASE}"

export SEQ_ESTOP_LABELS_PATH="${LABELS_DIR}/${ENV_NAME}/seq_estop_labels.npz"
export SEQ_ESTOP_ORACLE_DIR="${ORACLE_RUNS_BASE}/${ENV_NAME}/${ORACLE_SEED}"
export SEQ_ESTOP_TEST_MAX_PAIRS="$MAX_PAIRS"

echo "=============================="
echo "SLURM_JOB_ID          = ${SLURM_JOB_ID:-}"
echo "SEQ_ESTOP_LABELS_PATH = ${SEQ_ESTOP_LABELS_PATH}"
echo "SEQ_ESTOP_ORACLE_DIR  = ${SEQ_ESTOP_ORACLE_DIR}"
echo "SEQ_ESTOP_TEST_MAX_PAIRS = ${SEQ_ESTOP_TEST_MAX_PAIRS}"
echo "=============================="

. "${REPO_ROOT}/setup_shell.sh"

cd "$REPO_ROOT"
python3 -m pytest tests/test_seq_estop_consistency.py -v
