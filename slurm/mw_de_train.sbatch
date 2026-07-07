#!/bin/bash
# Phase 3: Train CPL for the MetaWorld data-efficiency sweep.
#
# Matrix: 6 feedback types × 11 budgets × 5 seeds = 330 tasks
#
#   Types  : pref  corr  demo  seq_estop  scalar  credit_assignment
#   Budgets: 50  100  300  600  1000  2000  5000  10000  20000  40000  100000
#   Seeds  : 0  1  2  3  4
#
# Task-ID mapping (1-indexed):
#   type_idx   = (TASK_ID - 1) / 55    (0..5)
#   budget_idx = ((TASK_ID-1) % 55)/5  (0..10)
#   seed_idx   = (TASK_ID - 1) % 5     (0..4)
#
# By-type ranges (useful for splitting across partitions):
#   pref                1–55
#   corr               56–110
#   demo              111–165
#   seq_estop         166–220
#   scalar            221–275
#   credit_assignment 276–330
#
# Budget clamping: if budget > available labels (corr/demo/credit max ~41k),
#   the run trains on all available data rather than erroring.
#
# Outputs: ${RUNS_DIR}/<type>_b<budget:06d>_s<seed>/
# WandB  : project=${WANDB_PROJECT}, run=<type>_b<budget:06d>_s<seed>
#
# Usage:
#   sbatch --array=1-330 \
#          --account=<account> --partition=<partition> \
#          [--time=HH:MM:SS] [--cpus-per-task=N] [--mem=XG] \
#          slurm/mw_de_train.sbatch
#
#   # Smoke-test (pref only, 1 seed):
#   sbatch --array=1 --account=<account> --partition=<partition> \
#          slurm/mw_de_train.sbatch
#
#   # By type:
#   sbatch --array=1-55   ...    # pref
#   sbatch --array=56-110 ...    # corr
#   sbatch --array=111-165 ...   # demo
#   sbatch --array=166-330 ...   # remaining
#
# Resource overrides (command-line flags override #SBATCH defaults below):
#   --time=HH:MM:SS     default: 24:00:00
#   --cpus-per-task=N   default: 4
#   --mem=XG            default: 16G
#
# Optional env overrides:
#   LABELS_DIR     default: /scratch/general/vast/u1472210/mw_de_labels
#   RUNS_DIR       default: runs/mw_de_cb075
#   ENV_NAME       default: mw_drawer-open-v2
#   CONFIG_DIR     default: configs/mw_state_dense
#   WANDB_PROJECT  default: mw-cpl-de-cb075

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=slurm/logs/%x_%A_%a.out
#SBATCH --error=slurm/logs/%x_%A_%a.err

set -euo pipefail

mkdir -p "${SLURM_SUBMIT_DIR:-$PWD}/slurm/logs"

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "ERROR: submit with --array=1-330 (or a sub-range)"
    exit 1
fi

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
if [[ ! -f "${SUBMIT_DIR}/setup_shell.sh" ]]; then
    echo "ERROR: no setup_shell.sh in ${SUBMIT_DIR}"
    exit 1
fi
REPO_ROOT="$SUBMIT_DIR"

# ── Matrix ────────────────────────────────────────────────────────────────────
TYPES=(pref corr demo seq_estop scalar credit_assignment)
BUDGETS=(50 100 300 600 1000 2000 5000 10000 20000 40000 100000)
SEEDS=(0 1 2 3 4)

ID=$(( SLURM_ARRAY_TASK_ID - 1 ))
TYPE_IDX=$(( ID / 55 ))
BUDGET_IDX=$(( (ID % 55) / 5 ))
SEED_IDX=$(( ID % 5 ))

FEEDBACK_TYPE="${TYPES[$TYPE_IDX]}"
BUDGET="${BUDGETS[$BUDGET_IDX]}"
SEED="${SEEDS[$SEED_IDX]}"
BUDGET_STR=$(printf "%06d" "$BUDGET")

# ── Paths ─────────────────────────────────────────────────────────────────────
LABELS_DIR="${LABELS_DIR:-/scratch/general/vast/u1472210/mw_de_labels}"
RUNS_DIR="${RUNS_DIR:-runs/mw_de_cb075}"
ENV_NAME="${ENV_NAME:-mw_drawer-open-v2}"
CONFIG_DIR="${CONFIG_DIR:-configs/mw_state_dense}"

[[ "$LABELS_DIR" != /* ]] && LABELS_DIR="${REPO_ROOT}/${LABELS_DIR}"
[[ "$RUNS_DIR"   != /* ]] && RUNS_DIR="${REPO_ROOT}/${RUNS_DIR}"

# ── Config and label file per feedback type ───────────────────────────────────
declare -A TYPE_CONFIG
TYPE_CONFIG[pref]=pref_cpl.yaml
TYPE_CONFIG[corr]=corr_cpl.yaml
TYPE_CONFIG[demo]=demo_cpl.yaml
TYPE_CONFIG[seq_estop]=seq_estop_cpl.yaml
TYPE_CONFIG[scalar]=scalar_cpl.yaml
TYPE_CONFIG[credit_assignment]=credit_cpl.yaml

declare -A TYPE_LABEL_FILE
TYPE_LABEL_FILE[pref]=pref_labels.npz
TYPE_LABEL_FILE[corr]=corr_labels.npz
TYPE_LABEL_FILE[demo]=demo_labels_K7.npz
TYPE_LABEL_FILE[seq_estop]=seq_estop_labels.npz
TYPE_LABEL_FILE[scalar]=scalar_labels.npz
TYPE_LABEL_FILE[credit_assignment]=credit_labels.npz

CONFIG_PATH="${REPO_ROOT}/${CONFIG_DIR}/${TYPE_CONFIG[$FEEDBACK_TYPE]}"
LABELS_NPZ="${LABELS_DIR}/${ENV_NAME}/${TYPE_LABEL_FILE[$FEEDBACK_TYPE]}"
RUN_PATH="${RUNS_DIR}/${FEEDBACK_TYPE}_b${BUDGET_STR}_s${SEED}"

echo "=============================="
echo "ARRAY_TASK_ID = ${SLURM_ARRAY_TASK_ID}"
echo "FEEDBACK_TYPE = ${FEEDBACK_TYPE}"
echo "BUDGET        = ${BUDGET}"
echo "SEED          = ${SEED}"
echo "ENV_NAME      = ${ENV_NAME}"
echo "CONFIG        = ${CONFIG_PATH}"
echo "LABELS        = ${LABELS_NPZ}"
echo "RUN_PATH      = ${RUN_PATH}"
echo "=============================="

# Skip if already trained
if [[ -f "${RUN_PATH}/best_model.pt" ]]; then
    echo "Already trained — skipping."
    exit 0
fi

if [[ ! -f "$LABELS_NPZ" ]]; then
    echo "ERROR: label file not found: ${LABELS_NPZ}"
    echo "Run mw_de_generate_labels.sbatch first."
    exit 1
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "ERROR: config not found: ${CONFIG_PATH}"
    exit 1
fi

# ── Auto-clamp budget to actual dataset size ───────────────────────────────────
# corr/demo/credit produce ~41k examples; budgets of 100k auto-clamp to all.
N_AVAILABLE=$(python3 -c "import numpy as np; d=np.load('${LABELS_NPZ}'); print(d['obs'].shape[0])")
if (( BUDGET > N_AVAILABLE )); then
    echo "NOTE: budget ${BUDGET} > available ${N_AVAILABLE} — using full dataset (capacity=null)."
    EFFECTIVE_CAPACITY="null"
else
    EFFECTIVE_CAPACITY="${BUDGET}"
fi

. "${REPO_ROOT}/setup_shell.sh"
mkdir -p "$RUN_PATH"

export WANDB_PROJECT="${WANDB_PROJECT:-mw-cpl-de-cb075}"

python3 "${REPO_ROOT}/scripts/train.py" \
    --config  "$CONFIG_PATH" \
    --path    "$RUN_PATH"   \
    --seed    "$SEED"       \
    --device  cpu           \
    --set "dataset_kwargs.path=${LABELS_NPZ}"            \
    --set "dataset_kwargs.capacity=${EFFECTIVE_CAPACITY}" \
    --set "budget=${BUDGET}"
