#!/bin/bash
# Generate PointMass data-efficiency datasets.
#
# Matrix: 6 feedback types × 6 budgets × 2 seeds = 72 tasks
#
#   Types  : pref  corr  demo  estop  scalar  credit_assignment
#   Budgets: 100   300   1000  3000   10000   20000
#   Seeds  : 0  1
#
# Task-ID mapping (1-indexed):
#   type_idx   = (TASK_ID - 1) / 12          (0..5)
#   budget_idx = ((TASK_ID - 1) % 12) / 2    (0..5)
#   seed_idx   = (TASK_ID - 1) % 2           (0 or 1)
#
# By-type task ranges (useful for splitting across clusters):
#   pref              1-12
#   corr             13-24
#   demo             25-36
#   estop            37-48
#   scalar           49-60
#   credit_assignment 61-72
#
# Outputs: datasets/pm/data_eff/<type>_b<budget:06d>_s<seed>.npz
#
# Usage:
#   sbatch --array=1-72 \        # full sweep
#          --job-name=pm_de_gen \
#          --time=1:00:00 \
#          --cpus-per-task=2 \
#          --mem=16G \
#          --account=<account> \
#          --partition=<partition> \
#          slurm/pm_de_generate.sbatch
#
#   sbatch --array=1-12 ...      # smoke-test: pref only
#   sbatch --array=1-36 ...      # cluster A
#   sbatch --array=37-72 ...     # cluster B (same shared FS)
#
# Optional env overrides:
#   POOL_NPZ        default: datasets/pm/pool.npz
#   ADVANTAGE_NPZ   default: runs/pm_sac_oracle/advantage_vi_N100.npz
#   OUTPUT_DIR      default: datasets/pm/data_eff

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=slurm/logs/%x_%A_%a.out
#SBATCH --error=slurm/logs/%x_%A_%a.err

set -euo pipefail

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "ERROR: submit with --array=1-72 (or a sub-range)"
    exit 1
fi

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
if [[ ! -f "${SUBMIT_DIR}/setup_shell.sh" ]]; then
    echo "ERROR: no setup_shell.sh in ${SUBMIT_DIR}"
    exit 1
fi
REPO_ROOT="$SUBMIT_DIR"

# ── Matrix ────────────────────────────────────────────────────────────────
TYPES=(pref corr demo estop scalar credit_assignment)
BUDGETS=(100 300 1000 3000 10000 20000)
SEEDS=(0 1)

ID=$(( SLURM_ARRAY_TASK_ID - 1 ))
TYPE_IDX=$(( ID / 12 ))
BUDGET_IDX=$(( (ID % 12) / 2 ))
SEED_IDX=$(( ID % 2 ))

FEEDBACK_TYPE="${TYPES[$TYPE_IDX]}"
BUDGET="${BUDGETS[$BUDGET_IDX]}"
SEED="${SEEDS[$SEED_IDX]}"
BUDGET_STR=$(printf "%06d" "$BUDGET")

# ── Paths ─────────────────────────────────────────────────────────────────
POOL_NPZ="${POOL_NPZ:-datasets/pm/pool.npz}"
ADVANTAGE_NPZ="${ADVANTAGE_NPZ:-runs/pm_sac_oracle/advantage_vi_N100.npz}"
OUTPUT_DIR="${OUTPUT_DIR:-datasets/pm/data_eff}"

[[ "$POOL_NPZ"      != /* ]] && POOL_NPZ="${REPO_ROOT}/${POOL_NPZ}"
[[ "$ADVANTAGE_NPZ" != /* ]] && ADVANTAGE_NPZ="${REPO_ROOT}/${ADVANTAGE_NPZ}"
[[ "$OUTPUT_DIR"    != /* ]] && OUTPUT_DIR="${REPO_ROOT}/${OUTPUT_DIR}"

OUTPUT_NPZ="${OUTPUT_DIR}/${FEEDBACK_TYPE}_b${BUDGET_STR}_s${SEED}.npz"

echo "=============================="
echo "ARRAY_TASK_ID = ${SLURM_ARRAY_TASK_ID}"
echo "FEEDBACK_TYPE = ${FEEDBACK_TYPE}"
echo "BUDGET        = ${BUDGET}"
echo "SEED          = ${SEED}"
echo "OUTPUT_NPZ    = ${OUTPUT_NPZ}"
echo "=============================="

# Skip if already generated (safe to re-submit or split across clusters)
if [[ -f "$OUTPUT_NPZ" ]]; then
    echo "Already exists — skipping."
    exit 0
fi

if [[ ! -f "$POOL_NPZ" ]]; then
    echo "ERROR: pool.npz not found at ${POOL_NPZ}"
    exit 1
fi
if [[ ! -f "$ADVANTAGE_NPZ" ]]; then
    echo "ERROR: advantage_vi.npz not found at ${ADVANTAGE_NPZ}"
    exit 1
fi

. "${REPO_ROOT}/setup_shell.sh"
mkdir -p "$OUTPUT_DIR"

# ── Type-specific args (mirrors pm_generate_feedback.sbatch defaults) ─────
EXTRA_ARGS=()
case "$FEEDBACK_TYPE" in
    pref)
        EXTRA_ARGS=(--skip-expert)
        ;;
    corr)
        EXTRA_ARGS=(--skip-expert)
        ;;
    demo)
        EXTRA_ARGS=(--n-counterfactuals 4 --skip-expert)
        ;;
    estop)
        EXTRA_ARGS=(--skip-expert)
        ;;
    scalar)
        EXTRA_ARGS=(--segment-len 8 --window-size 10 --noise-std 0.1 \
                    --scalar-delta 0.0 --skip-expert)
        ;;
    credit_assignment)
        EXTRA_ARGS=(--subsegment-len 12 --skip-expert)
        ;;
esac

python3 "${REPO_ROOT}/scripts/generate_pm_feedback.py" \
    --type          "$FEEDBACK_TYPE" \
    --pool          "$POOL_NPZ"      \
    --advantage-npz "$ADVANTAGE_NPZ" \
    --out           "$OUTPUT_NPZ"    \
    --n-pairs       "$BUDGET"        \
    --seed          "$SEED"          \
    "${EXTRA_ARGS[@]}"
