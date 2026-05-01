#!/bin/bash
# Generate the PointMass trajectory pool (prerequisite for all PM feedback types).
#
# Produces datasets/pm/pool.npz with segments from 6 policy tiers
# (expert ε=0.0 to noisy expert ε=0.75).
#
# Usage:
#   sbatch --job-name=pm_pool \
#          --time=0:30:00 \
#          --cpus-per-task=2 \
#          --mem=8G \
#          --account=<account> \
#          --partition=<partition> \
#          slurm/pm_generate_pool.sbatch
#
# Optional overrides (set before sbatch or via --export):
#   OUTPUT_DIR    default: datasets/pm
#   N_PER_TIER    default: 500   (6 tiers × 500 = 3000 segments)
#   SEGMENT_LEN   default: 50
#   SEED          default: 42

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
if [[ ! -f "${SUBMIT_DIR}/setup_shell.sh" ]]; then
    echo "ERROR: could not locate repo root (no setup_shell.sh in ${SUBMIT_DIR})"
    exit 1
fi
REPO_ROOT="$SUBMIT_DIR"

OUTPUT_DIR="${OUTPUT_DIR:-datasets/pm}"
N_PER_TIER="${N_PER_TIER:-500}"
SEGMENT_LEN="${SEGMENT_LEN:-50}"
SEED="${SEED:-42}"

[[ "$OUTPUT_DIR" != /* ]] && OUTPUT_DIR="${REPO_ROOT}/${OUTPUT_DIR}"

echo "=============================="
echo "SLURM_JOB_ID = ${SLURM_JOB_ID:-}"
echo "OUTPUT_DIR   = ${OUTPUT_DIR}"
echo "N_PER_TIER   = ${N_PER_TIER}"
echo "SEGMENT_LEN  = ${SEGMENT_LEN}"
echo "SEED         = ${SEED}"
echo "=============================="

. "${REPO_ROOT}/setup_shell.sh"
mkdir -p "$OUTPUT_DIR"

python3 "${REPO_ROOT}/scripts/generate_pm_pool.py" \
    --out         "${OUTPUT_DIR}/pool.npz" \
    --n-per-tier  "$N_PER_TIER" \
    --segment-len "$SEGMENT_LEN" \
    --seed        "$SEED"
