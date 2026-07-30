#!/bin/bash
# Quick sanity check for the pinned-expert fix in generate_demo_labels.py.
#
# Generates a small --max-segments demo_labels sample and immediately validates
# it with diagnose_demo_identity.py. Should complete in ~15-20 minutes (vs.
# ~20-30h for a full env), so this is meant to be run BEFORE committing to the
# full 72h regeneration across all 5 envs.
#
# Expected result: "expert IS the demo (rank 0)" should read ~100% among
# matched examples (it was 61.5% before the fix, on mw_plate-slide-v2).
#
# Usage:
#   sbatch --export=ENV_NAME=mw_plate-slide-v2 slurm/sanity_check_demo_labels.sbatch
#
# Optional env overrides:
#   ENV_NAME          default: mw_plate-slide-v2
#   MAX_SEGMENTS       default: 100
#   ORACLE_RUNS_BASE  default: runs/runs/chpc/oracle_sac_seeds  (relative to repo root)
#   ORACLE_SEED       default: seed-1
#   POOL_DIR          default: /scratch/general/vast/u1472210/mw_de_pool
#   OUTPUT_DIR        default: /scratch/general/vast/u1472210/mw_de_labels_v2

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --partition=soc-gpu-class-grn
#SBATCH --qos=soc-gpu-students-grn
#SBATCH --account=dbrown
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

ENV_NAME="${ENV_NAME:-mw_plate-slide-v2}"
MAX_SEGMENTS="${MAX_SEGMENTS:-100}"
ORACLE_RUNS_BASE="${ORACLE_RUNS_BASE:-runs/runs/chpc/oracle_sac_seeds}"
ORACLE_SEED="${ORACLE_SEED:-seed-1}"
POOL_DIR="${POOL_DIR:-/scratch/general/vast/u1472210/mw_de_pool}"
OUTPUT_DIR="${OUTPUT_DIR:-/scratch/general/vast/u1472210/mw_de_labels_v2}"

[[ "$ORACLE_RUNS_BASE" != /* ]] && ORACLE_RUNS_BASE="${REPO_ROOT}/${ORACLE_RUNS_BASE}"
[[ "$POOL_DIR"         != /* ]] && POOL_DIR="${REPO_ROOT}/${POOL_DIR}"
[[ "$OUTPUT_DIR"       != /* ]] && OUTPUT_DIR="${REPO_ROOT}/${OUTPUT_DIR}"

POOL_PATH="${POOL_DIR}/${ENV_NAME}/pool.npz"
ORACLE_DIR="${ORACLE_RUNS_BASE}/${ENV_NAME}/${ORACLE_SEED}"

echo "=============================="
echo "SLURM_JOB_ID   = ${SLURM_JOB_ID:-}"
echo "ENV_NAME       = ${ENV_NAME}"
echo "MAX_SEGMENTS   = ${MAX_SEGMENTS}"
echo "ORACLE_DIR     = ${ORACLE_DIR}"
echo "POOL_PATH      = ${POOL_PATH}"
echo "OUTPUT_DIR     = ${OUTPUT_DIR}"
echo "=============================="

if [[ ! -f "$POOL_PATH" ]]; then
    echo "ERROR: pool.npz not found: ${POOL_PATH}"
    exit 1
fi
if [[ ! -d "$ORACLE_DIR" ]]; then
    echo "ERROR: oracle run dir not found: ${ORACLE_DIR}"
    exit 1
fi

. "${REPO_ROOT}/setup_shell.sh"
mkdir -p "${OUTPUT_DIR}/${ENV_NAME}"
mkdir -p "${REPO_ROOT}/results/demo_identity"

echo ""
echo "--- Step 1/2: generate sanity batch (--max-segments ${MAX_SEGMENTS}) ---"
python3 "${REPO_ROOT}/scripts/generate_demo_labels.py" \
    --pool-path            "$POOL_PATH" \
    --run-dir              "$ORACLE_DIR" \
    --n-counterfactuals    6 \
    --checkpoints-per-tier 2 \
    --min-adv-gap          0.0 \
    --mcmc-samples         16 \
    --score-batch-size     32 \
    --max-segments         "$MAX_SEGMENTS" \
    --output-dir            "$OUTPUT_DIR" \
    --device cpu

SANITY_PATH="${OUTPUT_DIR}/${ENV_NAME}/demo_labels_K7_sanity${MAX_SEGMENTS}.npz"

echo ""
echo "--- Step 2/2: validate expert-is-demo rate ---"
python3 "${REPO_ROOT}/scripts/diagnose_demo_identity.py" \
    --pool-path         "$POOL_PATH" \
    --demo-labels-path  "$SANITY_PATH" \
    --run-dir           "$ORACLE_DIR" \
    --n-samples         -1 \
    --output            "${REPO_ROOT}/results/demo_identity/${ENV_NAME}_sanity${MAX_SEGMENTS}_check.npz" \
    --device cpu

echo ""
echo "Done. Check 'expert IS the demo (rank 0)' above — should read ~100% if the fix is correct."
