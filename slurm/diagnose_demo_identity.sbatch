#!/bin/bash
# Diagnostic: recover expert-rollout identity in an already-generated demo_labels file.
#
# Re-rolls ONLY the expert model per sampled segment (not the full K=7 candidate set),
# and does not re-run oracle scoring — adv_scores are read from the existing
# demo_labels_K{K}.npz. Cheap relative to Phase 4 generation: ~1/K the rollout cost,
# zero scoring cost, and runs on a random subsample by default.
#
# Requires (already produced by mw_de_build_pool.sbatch / mw_de_generate_labels.sbatch):
#   - pool.npz            at POOL_DIR/ENV_NAME/pool.npz
#   - demo_labels_K*.npz  at LABELS_DIR/ENV_NAME/demo_labels_K*.npz
#   - oracle run dir with best_model.pt
#
# The oracle run dir used to originally generate the labels is not recorded anywhere,
# so this defaults to the same default mw_de_generate_labels.sbatch uses
# (oracle_sac_seeds/seed-1). If the script prints a "[WARNING] Expert identity match
# rate is low" line, rerun with ORACLE_RUNS_BASE=runs/runs/chpc/oracle_sac_all (no
# ORACLE_SEED subdir) to try the other candidate.
#
# Usage:
#   # Single env:
#   sbatch --account=<account> --partition=<partition> \
#          --export=ENV_NAME=mw_plate-slide-v2 \
#          slurm/diagnose_demo_identity.sbatch
#
#   # Array over the 5 MW envs:
#   sbatch --array=1-5 --account=<account> --partition=<partition> \
#          slurm/diagnose_demo_identity.sbatch
#
# Optional env overrides:
#   ENV_NAME          default: mw_drawer-open-v2 (ignored if SLURM_ARRAY_TASK_ID set)
#   ORACLE_RUNS_BASE  default: runs/runs/chpc/oracle_sac_seeds  (relative to repo root)
#   ORACLE_SEED       default: seed-1  (subdirectory; set to empty to use ORACLE_RUNS_BASE/ENV_NAME directly)
#   POOL_DIR          default: /scratch/general/vast/u1472210/mw_de_pool
#   LABELS_DIR        default: /scratch/general/vast/u1472210/mw_de_labels
#   OUTPUT_DIR        default: results/demo_identity
#   N_SAMPLES         default: 2000  (<=0 for all)
#   SEED              default: 0

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/uufs/chpc.utah.edu/common/home/u1472210/cpl/slurm/logs/%x_%A_%a.out
#SBATCH --error=/uufs/chpc.utah.edu/common/home/u1472210/cpl/slurm/logs/%x_%A_%a.err

set -euo pipefail

mkdir -p "${SLURM_SUBMIT_DIR:-$PWD}/slurm/logs"

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
if [[ ! -f "${SUBMIT_DIR}/setup_shell.sh" ]]; then
    echo "ERROR: no setup_shell.sh in ${SUBMIT_DIR}"
    exit 1
fi
REPO_ROOT="$SUBMIT_DIR"

ENVS=(mw_drawer-open-v2 mw_plate-slide-v2 mw_door-open-v2 mw_bin-picking-v2 mw_button-press-v2)
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    ENV_NAME="${ENVS[$(( SLURM_ARRAY_TASK_ID - 1 ))]}"
else
    ENV_NAME="${ENV_NAME:-mw_drawer-open-v2}"
fi

ORACLE_RUNS_BASE="${ORACLE_RUNS_BASE:-runs/runs/chpc/oracle_sac_seeds}"
ORACLE_SEED="${ORACLE_SEED:-seed-1}"
POOL_DIR="${POOL_DIR:-/scratch/general/vast/u1472210/mw_de_pool}"
LABELS_DIR="${LABELS_DIR:-/scratch/general/vast/u1472210/mw_de_labels}"
OUTPUT_DIR="${OUTPUT_DIR:-results/demo_identity}"
N_SAMPLES="${N_SAMPLES:-2000}"
SEED="${SEED:-0}"

[[ "$ORACLE_RUNS_BASE" != /* ]] && ORACLE_RUNS_BASE="${REPO_ROOT}/${ORACLE_RUNS_BASE}"
[[ "$POOL_DIR"         != /* ]] && POOL_DIR="${REPO_ROOT}/${POOL_DIR}"
[[ "$LABELS_DIR"       != /* ]] && LABELS_DIR="${REPO_ROOT}/${LABELS_DIR}"
[[ "$OUTPUT_DIR"       != /* ]] && OUTPUT_DIR="${REPO_ROOT}/${OUTPUT_DIR}"

POOL_PATH="${POOL_DIR}/${ENV_NAME}/pool.npz"
if [[ -n "$ORACLE_SEED" ]]; then
    ORACLE_DIR="${ORACLE_RUNS_BASE}/${ENV_NAME}/${ORACLE_SEED}"
else
    ORACLE_DIR="${ORACLE_RUNS_BASE}/${ENV_NAME}"
fi

DEMO_LABELS_PATH="$(ls "${LABELS_DIR}/${ENV_NAME}"/demo_labels_K*.npz 2>/dev/null | head -1 || true)"

echo "=============================="
echo "SLURM_JOB_ID      = ${SLURM_JOB_ID:-}"
echo "ARRAY_TASK_ID     = ${SLURM_ARRAY_TASK_ID:-n/a}"
echo "ENV_NAME          = ${ENV_NAME}"
echo "ORACLE_DIR        = ${ORACLE_DIR}"
echo "POOL_PATH         = ${POOL_PATH}"
echo "DEMO_LABELS_PATH  = ${DEMO_LABELS_PATH}"
echo "N_SAMPLES         = ${N_SAMPLES}"
echo "SEED              = ${SEED}"
echo "OUTPUT_DIR        = ${OUTPUT_DIR}"
echo "=============================="

if [[ ! -f "$POOL_PATH" ]]; then
    echo "ERROR: pool.npz not found: ${POOL_PATH}"
    echo "Run mw_de_build_pool.sbatch first, or point POOL_DIR at wherever it was written."
    exit 1
fi
if [[ -z "$DEMO_LABELS_PATH" ]]; then
    echo "ERROR: no demo_labels_K*.npz found under ${LABELS_DIR}/${ENV_NAME}/"
    exit 1
fi
if [[ ! -d "$ORACLE_DIR" ]]; then
    echo "ERROR: oracle run dir not found: ${ORACLE_DIR}"
    exit 1
fi

. "${REPO_ROOT}/setup_shell.sh"
mkdir -p "${OUTPUT_DIR}"

python3 "${REPO_ROOT}/scripts/diagnose_demo_identity.py" \
    --pool-path         "$POOL_PATH" \
    --demo-labels-path  "$DEMO_LABELS_PATH" \
    --run-dir           "$ORACLE_DIR" \
    --n-samples         "$N_SAMPLES" \
    --seed              "$SEED" \
    --output            "${OUTPUT_DIR}/${ENV_NAME}_identity_diagnostic.npz" \
    --device cpu

echo "Done — ${ENV_NAME} identity diagnostic written to ${OUTPUT_DIR}/${ENV_NAME}_identity_diagnostic.npz"
