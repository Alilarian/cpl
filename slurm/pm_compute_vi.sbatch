#!/bin/bash
# Compute the PointMass VI advantage table — prerequisite for feedback generation.
#
# Runs discrete value iteration on a state × action grid and saves V*, Q*, A*
# to a .npz file consumed by pm_generate_feedback.sbatch for oracle scoring.
#
# Pipeline order on CHPC:
#   1. sbatch slurm/pm_generate_pool.sbatch          (generate trajectory pool)
#   2. sbatch slurm/pm_compute_vi.sbatch             (this script)
#   3. sbatch --array=1-5 slurm/pm_generate_feedback.sbatch
#   4. sbatch --array=1-5 slurm/pm_train_cpl.sbatch
#
# Usage:
#   sbatch --job-name=pm_vi \
#          --time=0:15:00 \
#          --cpus-per-task=2 \
#          --mem=4G \
#          --account=<account> \
#          --partition=<partition> \
#          slurm/pm_compute_vi.sbatch
#
# Optional overrides (set before sbatch or via --export):
#   OUTPUT_NPZ   default: runs/pm_sac_oracle/advantage_vi_N100.npz
#   ALPHA        default: 0.01
#   N_STATES     default: 100   (grid resolution; 100 → 10000 cells)
#   N_ACTIONS    default: 11    (per dimension)

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

OUTPUT_NPZ="${OUTPUT_NPZ:-runs/pm_sac_oracle/advantage_vi_N100.npz}"
ALPHA="${ALPHA:-0.01}"
N_STATES="${N_STATES:-100}"
N_ACTIONS="${N_ACTIONS:-11}"

[[ "$OUTPUT_NPZ" != /* ]] && OUTPUT_NPZ="${REPO_ROOT}/${OUTPUT_NPZ}"

echo "=============================="
echo "SLURM_JOB_ID = ${SLURM_JOB_ID:-}"
echo "OUTPUT_NPZ   = ${OUTPUT_NPZ}"
echo "ALPHA        = ${ALPHA}"
echo "N_STATES     = ${N_STATES}"
echo "N_ACTIONS    = ${N_ACTIONS}"
echo "=============================="

. "${REPO_ROOT}/setup_shell.sh"
mkdir -p "$(dirname "$OUTPUT_NPZ")"

python3 "${REPO_ROOT}/scripts/compute_advantage_vi.py" \
    --out       "$OUTPUT_NPZ" \
    --alpha     "$ALPHA"      \
    --n-states  "$N_STATES"   \
    --n-actions "$N_ACTIONS"
