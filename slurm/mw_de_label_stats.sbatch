#!/bin/bash
# Compute and save detailed statistics for all MW DE label files.
# Iterates over all envs × 6 feedback types and writes:
#   results/mw_de_label_stats.txt   human-readable report
#   results/mw_de_label_stats.csv   one row per env×type
#
# Usage:
#   sbatch --account=<account> --partition=<partition> \
#          slurm/mw_de_label_stats.sbatch
#
# Optional env overrides:
#   LABELS_DIR  default: /scratch/general/vast/u1472210/mw_de_labels
#   OUTPUT_DIR  default: results

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
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

LABELS_DIR="${LABELS_DIR:-/scratch/general/vast/u1472210/mw_de_labels}"
OUTPUT_DIR="${OUTPUT_DIR:-results}"

[[ "$LABELS_DIR" != /* ]] && LABELS_DIR="${REPO_ROOT}/${LABELS_DIR}"
[[ "$OUTPUT_DIR"  != /* ]] && OUTPUT_DIR="${REPO_ROOT}/${OUTPUT_DIR}"

echo "=============================="
echo "SLURM_JOB_ID = ${SLURM_JOB_ID:-}"
echo "LABELS_DIR   = ${LABELS_DIR}"
echo "OUTPUT_DIR   = ${OUTPUT_DIR}"
echo "=============================="

. "${REPO_ROOT}/setup_shell.sh"

python3 "${REPO_ROOT}/scripts/mw_de_label_stats.py" \
    --labels-dir "$LABELS_DIR" \
    --output-dir "$OUTPUT_DIR"
