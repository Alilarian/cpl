#!/bin/bash
# Train CPL on PointMass data-efficiency sweep.
#
# Same 72-task matrix as pm_de_generate.sbatch — must run after it.
#
#   Types  : pref  corr  demo  estop  scalar  credit_assignment
#   Budgets: 100   300   1000  3000   10000   20000
#   Seeds  : 0  1
#
# By-type task ranges:
#   pref              1-12
#   corr             13-24
#   demo             25-36
#   estop            37-48
#   scalar           49-60
#   credit_assignment 61-72
#
# Runs land in:  runs/pm_de/<type>_b<budget:06d>_s<seed>/
# WandB project: pm-cpl-data-efficiency
# WandB name:    <type>_b<budget:06d>_s<seed>
#
# Usage:
#   sbatch --array=1-72 \
#          --job-name=pm_de_train \
#          --time=4:00:00 \
#          --cpus-per-task=4 \
#          --mem=16G \
#          --gres=gpu:1 \
#          --account=<account> \
#          --partition=<partition> \
#          slurm/pm_de_train.sbatch
#
#   sbatch --array=1-12 ...      # smoke-test: pref only
#   sbatch --array=1-36 ...      # cluster A
#   sbatch --array=37-72 ...     # cluster B (same shared FS)
#
# Optional env overrides:
#   RUNS_DIR    default: runs
#   DATA_DIR    default: datasets/pm/data_eff

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

# ── Config: one yaml per feedback type (existing configs, unchanged) ───────
declare -A TYPE_CONFIG
TYPE_CONFIG[pref]=cpl_pref.yaml
TYPE_CONFIG[corr]=cpl_corr.yaml
TYPE_CONFIG[demo]=cpl_demo.yaml
TYPE_CONFIG[estop]=cpl_estop.yaml
TYPE_CONFIG[scalar]=cpl_scalar.yaml
TYPE_CONFIG[credit_assignment]=cpl_credit_assignment.yaml

CONFIG_PATH="${REPO_ROOT}/configs/pm_navigation/${TYPE_CONFIG[$FEEDBACK_TYPE]}"

# ── Paths ─────────────────────────────────────────────────────────────────
RUNS_DIR="${RUNS_DIR:-runs}"
DATA_DIR="${DATA_DIR:-datasets/pm/data_eff}"

[[ "$RUNS_DIR" != /* ]] && RUNS_DIR="${REPO_ROOT}/${RUNS_DIR}"
[[ "$DATA_DIR" != /* ]] && DATA_DIR="${REPO_ROOT}/${DATA_DIR}"

LABELS_NPZ="${DATA_DIR}/${FEEDBACK_TYPE}_b${BUDGET_STR}_s${SEED}.npz"
RUN_PATH="${RUNS_DIR}/pm_de/${FEEDBACK_TYPE}_b${BUDGET_STR}_s${SEED}"

echo "=============================="
echo "ARRAY_TASK_ID = ${SLURM_ARRAY_TASK_ID}"
echo "FEEDBACK_TYPE = ${FEEDBACK_TYPE}"
echo "BUDGET        = ${BUDGET}"
echo "SEED          = ${SEED}"
echo "CONFIG        = ${CONFIG_PATH}"
echo "LABELS        = ${LABELS_NPZ}"
echo "RUN_PATH      = ${RUN_PATH}"
echo "=============================="

if [[ ! -f "$LABELS_NPZ" ]]; then
    echo "ERROR: labels not found at ${LABELS_NPZ}"
    echo "Run pm_de_generate.sbatch first."
    exit 1
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "ERROR: config not found at ${CONFIG_PATH}"
    exit 1
fi

. "${REPO_ROOT}/setup_shell.sh"
mkdir -p "$RUN_PATH"

export WANDB_PROJECT="${WANDB_PROJECT:-pm-cpl-data-efficiency}"

python3 "${REPO_ROOT}/scripts/train.py" \
    --config "$CONFIG_PATH" \
    --path   "$RUN_PATH"   \
    --seed   "$SEED"       \
    --set    "dataset_kwargs.path=${LABELS_NPZ}" \
    --set    "budget=${BUDGET}"
