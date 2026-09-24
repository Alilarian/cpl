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
#   ENVS        default: all 6 MW envs (space-separated to restrict, e.g.
#                         "mw_button-press-v2 mw_door-open-v2")
#   DEMO_K      default: 7 (K in demo_labels_K<K>.npz to look for -- set to the
#                         actual K used for a given generation run, e.g. 9)

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=3:00:00
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
ENVS="${ENVS:-}"
DEMO_K="${DEMO_K:-7}"

[[ "$LABELS_DIR" != /* ]] && LABELS_DIR="${REPO_ROOT}/${LABELS_DIR}"
[[ "$OUTPUT_DIR"  != /* ]] && OUTPUT_DIR="${REPO_ROOT}/${OUTPUT_DIR}"

echo "=============================="
echo "SLURM_JOB_ID = ${SLURM_JOB_ID:-}"
echo "LABELS_DIR   = ${LABELS_DIR}"
echo "OUTPUT_DIR   = ${OUTPUT_DIR}"
echo "ENVS         = ${ENVS:-<unset, all 6 default envs>}"
echo "DEMO_K       = ${DEMO_K}"
echo "=============================="

. "${REPO_ROOT}/setup_shell.sh"

EXTRA_ARGS=()
if [[ -n "$ENVS" ]]; then
    read -r -a ENV_ARR <<< "$ENVS"
    EXTRA_ARGS+=(--envs "${ENV_ARR[@]}")
fi

python3 "${REPO_ROOT}/scripts/mw_de_label_stats.py" \
    --labels-dir "$LABELS_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --demo-k "$DEMO_K" \
    "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
