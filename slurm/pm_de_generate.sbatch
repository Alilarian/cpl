#!/bin/bash
# Generate PointMass data-efficiency datasets.
#
# Matrix: 6 feedback types × 9 budgets × 3 seeds = 162 tasks
#
#   Types  : pref  corr  demo  seq_estop  scalar  credit_assignment
#   Budgets: 50  100  300  600  1000  2000  5000  10000  20000
#   Seeds  : 0  1  2
#
# Task-ID mapping (1-indexed):
#   type_idx   = (TASK_ID - 1) / 27          (0..5)
#   budget_idx = ((TASK_ID - 1) % 27) / 3    (0..8)
#   seed_idx   = (TASK_ID - 1) % 3           (0, 1, or 2)
#
# By-type task ranges (useful for splitting across clusters):
#   pref              1-27
#   corr             28-54
#   demo             55-81
#   seq_estop        82-108
#   scalar          109-135
#   credit_assignment 136-162
#
# Outputs: datasets/pm/data_eff/<type>_b<budget:06d>_s<seed>.npz
#
# Usage:
#   sbatch --array=1-162 \       # full sweep
#          --job-name=pm_de_gen \
#          --time=1:00:00 \
#          --cpus-per-task=2 \
#          --mem=16G \
#          --account=<account> \
#          --partition=<partition> \
#          slurm/pm_de_generate.sbatch
#
#   sbatch --array=1-27 ...      # smoke-test: pref only
#   sbatch --array=1-81 ...      # cluster A
#   sbatch --array=82-162 ...    # cluster B (same shared FS)
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
    echo "ERROR: submit with --array=1-162 (or a sub-range)"
    exit 1
fi

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
if [[ ! -f "${SUBMIT_DIR}/setup_shell.sh" ]]; then
    echo "ERROR: no setup_shell.sh in ${SUBMIT_DIR}"
    exit 1
fi
REPO_ROOT="$SUBMIT_DIR"

# ── Matrix ────────────────────────────────────────────────────────────────
TYPES=(pref corr demo seq_estop scalar credit_assignment)
BUDGETS=(50 100 300 600 1000 2000 5000 10000 20000)
SEEDS=(0 1 2)

ID=$(( SLURM_ARRAY_TASK_ID - 1 ))
TYPE_IDX=$(( ID / 27 ))
BUDGET_IDX=$(( (ID % 27) / 3 ))
SEED_IDX=$(( ID % 3 ))

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

# ── Type-specific args ─────────────────────────────────────────────────────
EXTRA_ARGS=()
case "$FEEDBACK_TYPE" in
    pref)
        # --n-candidates fixed at max_budget(20000) × 3 = 60000 so every budget
        # level draws from the same candidate pool → smaller budget IS a subset
        EXTRA_ARGS=(--skip-expert --n-candidates 60000 --min-adv-gap 0.5)
        ;;
    corr)
        EXTRA_ARGS=(--skip-expert --min-adv-gap 0.5)
        ;;
    demo)
        EXTRA_ARGS=(--n-counterfactuals 4 --skip-expert --min-adv-gap 0.5)
        ;;
    seq_estop)
        EXTRA_ARGS=(--rho 0.8 --lam 2.0 --kappa 1.9101 --horizon 4 --skip-expert)
        ;;
    scalar)
        EXTRA_ARGS=(--segment-len 8 --window-size 10 --noise-std 0.0 \
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
