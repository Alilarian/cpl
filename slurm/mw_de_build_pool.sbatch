#!/bin/bash
# Phase 1: Build trajectory pool for MetaWorld DE sweep.
#
# Rolls out oracle SAC checkpoints every --checkpoint-interval steps,
# samples fixed-length segments, and writes pool.npz.
#
# With defaults (interval=20k, 41 ckpts × 1000 segs): ~41k segments total.
# Re-run safe: already-completed checkpoints are skipped automatically.
#
# Usage:
#   sbatch --account=<account> --partition=<partition> \
#          [--time=HH:MM:SS] [--cpus-per-task=N] [--mem=XG] \
#          slurm/mw_de_build_pool.sbatch
#
# Resource overrides (command-line flags override #SBATCH defaults below):
#   --time=HH:MM:SS     default: 12:00:00
#   --cpus-per-task=N   default: 4
#   --mem=XG            default: 32G
#
# Optional env overrides (via --export or environment):
#   ENV_NAME            default: mw_drawer-open-v2
#   ORACLE_RUNS_BASE    default: /scratch/general/vast/u1472210/oracle_sac_all
#   OUTPUT_DIR          default: /scratch/general/vast/u1472210/mw_de_pool
#   CHECKPOINT_INTERVAL default: 20000   → 41 checkpoints for drawer-open @ 820k
#   N_EPISODES          default: 20      episodes rolled out per checkpoint
#   N_SEGMENTS          default: 4000    segments sampled per checkpoint
#                                        → 41 ckpts × 4000 = ~164k segments total
#                                          covers the 100k budget for all types
#   SEGMENT_LENGTH      default: 50

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err

set -euo pipefail

mkdir -p "${SLURM_SUBMIT_DIR:-$PWD}/slurm/logs"

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
if [[ ! -f "${SUBMIT_DIR}/setup_shell.sh" ]]; then
    echo "ERROR: no setup_shell.sh in ${SUBMIT_DIR}"
    exit 1
fi
REPO_ROOT="$SUBMIT_DIR"

ENV_NAME="${ENV_NAME:-mw_drawer-open-v2}"
ORACLE_RUNS_BASE="${ORACLE_RUNS_BASE:-/scratch/general/vast/u1472210/oracle_sac_all}"
OUTPUT_DIR="${OUTPUT_DIR:-/scratch/general/vast/u1472210/mw_de_pool}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-20000}"
N_EPISODES="${N_EPISODES:-20}"
N_SEGMENTS="${N_SEGMENTS:-4000}"
SEGMENT_LENGTH="${SEGMENT_LENGTH:-50}"

[[ "$ORACLE_RUNS_BASE" != /* ]] && ORACLE_RUNS_BASE="${REPO_ROOT}/${ORACLE_RUNS_BASE}"
[[ "$OUTPUT_DIR"       != /* ]] && OUTPUT_DIR="${REPO_ROOT}/${OUTPUT_DIR}"

echo "=============================="
echo "SLURM_JOB_ID        = ${SLURM_JOB_ID:-}"
echo "ENV_NAME            = ${ENV_NAME}"
echo "ORACLE_RUNS_BASE    = ${ORACLE_RUNS_BASE}"
echo "OUTPUT_DIR          = ${OUTPUT_DIR}"
echo "CHECKPOINT_INTERVAL = ${CHECKPOINT_INTERVAL}"
echo "N_EPISODES          = ${N_EPISODES}"
echo "N_SEGMENTS          = ${N_SEGMENTS}"
echo "SEGMENT_LENGTH      = ${SEGMENT_LENGTH}"
echo "=============================="

. "${REPO_ROOT}/setup_shell.sh"

python3 "${REPO_ROOT}/scripts/build_trajectory_pool.py" \
    --oracle-runs-base          "$ORACLE_RUNS_BASE" \
    --envs                      "$ENV_NAME" \
    --checkpoint-interval       "$CHECKPOINT_INTERVAL" \
    --n-episodes-per-checkpoint "$N_EPISODES" \
    --n-segments-per-checkpoint "$N_SEGMENTS" \
    --segment-length            "$SEGMENT_LENGTH" \
    --output-dir                "$OUTPUT_DIR" \
    --device cpu
