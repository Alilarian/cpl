#!/bin/bash
# Phase 3: Train CPL for the MetaWorld data-efficiency sweep — all envs.
#
# Matrix per env: 6 feedback types × 11 budgets × 5 seeds = 330 tasks
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
# By-type task ranges (useful for submitting one type at a time):
#   pref                1–55
#   corr               56–110
#   demo              111–165
#   seq_estop         166–220
#   scalar            221–275
#   credit_assignment 276–330
#
# Usage — all types, one env at a time:
#   ENV_NAME=mw_drawer-open-v2   sbatch --array=1-330 --account=<acct> --partition=<part> slurm/mw_de_train.sbatch
#   ENV_NAME=mw_bin-picking-v2   sbatch --array=1-330 --account=<acct> --partition=<part> slurm/mw_de_train.sbatch
#   ENV_NAME=mw_button-press-v2  sbatch --array=1-330 --account=<acct> --partition=<part> slurm/mw_de_train.sbatch
#   ENV_NAME=mw_door-open-v2     sbatch --array=1-330 --account=<acct> --partition=<part> slurm/mw_de_train.sbatch
#   ENV_NAME=mw_plate-slide-v2   sbatch --array=1-330 --account=<acct> --partition=<part> slurm/mw_de_train.sbatch
#
# Usage — single feedback type across all envs:
#   for ENV in mw_drawer-open-v2 mw_bin-picking-v2 mw_button-press-v2 mw_door-open-v2 mw_plate-slide-v2; do
#     ENV_NAME=$ENV sbatch --array=1-55   --account=<acct> --partition=<part> slurm/mw_de_train.sbatch  # pref
#     ENV_NAME=$ENV sbatch --array=56-110  ...   # corr
#     ENV_NAME=$ENV sbatch --array=111-165 ...   # demo
#     ENV_NAME=$ENV sbatch --array=166-220 ...   # seq_estop
#     ENV_NAME=$ENV sbatch --array=221-275 ...   # scalar
#     ENV_NAME=$ENV sbatch --array=276-330 ...   # credit_assignment
#   done
#
# Resume: if a job is killed by walltime, resubmit with the same command.
#   The trainer auto-resumes from final_model.pt; wandb run continues via
#   wandb_run_id.txt stored in the run directory.
#   Completed runs (training_complete sentinel exists) are automatically skipped.
#
# WandB setup (do once on CHPC before submitting):
#   export WANDB_API_KEY=<your-key>          # or: wandb login
#   echo "export WANDB_API_KEY=<your-key>" >> ~/.bashrc
#
# Resource overrides (command-line flags override #SBATCH defaults below):
#   --time=HH:MM:SS     default: 48:00:00
#   --cpus-per-task=N   default: 4
#   --mem=XG            default: 16G
#
# Optional env overrides:
#   ENV_NAME           default: mw_drawer-open-v2
#   LABELS_DIR         default: /scratch/general/vast/u1472210/mw_de_labels
#   RUNS_DIR           default: runs/mw_de_cb075
#   CONFIG_DIR         default: configs/mw_state_dense
#   WANDB_PROJECT      default: mw-cpl-de
#   CONTRASTIVE_BIAS   override alg_kwargs.contrastive_bias (e.g. 0.5)
#   TOTAL_STEPS        override trainer_kwargs.total_steps  (e.g. 500000)

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --output=/uufs/chpc.utah.edu/common/home/u1472210/cpl/slurm/logs/%x_%A_%a.out
#SBATCH --error=/uufs/chpc.utah.edu/common/home/u1472210/cpl/slurm/logs/%x_%A_%a.err

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
ENV_NAME="${ENV_NAME:-mw_drawer-open-v2}"
LABELS_DIR="${LABELS_DIR:-/scratch/general/vast/u1472210/mw_de_labels}"
RUNS_DIR="${RUNS_DIR:-runs/mw_de_cb075}"
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

# ENV_NAME is part of the run path so runs for different envs never collide.
RUN_PATH="${RUNS_DIR}/${ENV_NAME}/${FEEDBACK_TYPE}_b${BUDGET_STR}_s${SEED}"

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

# ── Skip if already completed ─────────────────────────────────────────────────
# The sentinel file is written only after train.py exits successfully at 250k steps.
# If a job was killed mid-training (no sentinel), the trainer resumes from
# final_model.pt which is saved every eval_freq steps (every 5k steps).
if [[ -f "${RUN_PATH}/training_complete" ]]; then
    echo "Already completed (training_complete sentinel exists) — skipping."
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
N_AVAILABLE=$(python3 -c "import numpy as np; d=np.load('${LABELS_NPZ}'); print(d['obs'].shape[0])")
if (( BUDGET > N_AVAILABLE )); then
    echo "NOTE: budget ${BUDGET} > available ${N_AVAILABLE} — using full dataset (capacity=null)."
    EFFECTIVE_CAPACITY="null"
else
    EFFECTIVE_CAPACITY="${BUDGET}"
fi

. "${REPO_ROOT}/setup_shell.sh"
mkdir -p "$RUN_PATH"

# ── Optional per-run overrides ────────────────────────────────────────────────
EXTRA_SETS=()
[[ -n "${CONTRASTIVE_BIAS:-}" ]] && EXTRA_SETS+=(--set "alg_kwargs.contrastive_bias=${CONTRASTIVE_BIAS}")
[[ -n "${TOTAL_STEPS:-}" ]]      && EXTRA_SETS+=(--set "trainer_kwargs.total_steps=${TOTAL_STEPS}")

# ── WandB ─────────────────────────────────────────────────────────────────────
# Project: mw-cpl-de  (set WANDB_PROJECT to override)
# Run name: "<env>/<type>_b<budget>_s<seed>"  — unique across all envs
# Group:    "<env>"                            — groups runs per env in the UI
# Resume:   wandb_run_id.txt persists the run ID across walltime restarts
export WANDB_PROJECT="${WANDB_PROJECT:-mw-cpl-de}"
export WANDB_RUN_NAME="${ENV_NAME}/${FEEDBACK_TYPE}_b${BUDGET_STR}_s${SEED}"
export WANDB_GROUP="${ENV_NAME}"

# ── Train ─────────────────────────────────────────────────────────────────────
python3 "${REPO_ROOT}/scripts/train.py" \
    --config  "$CONFIG_PATH" \
    --path    "$RUN_PATH"   \
    --seed    "$SEED"       \
    --device  cpu           \
    --set "eval_env=${ENV_NAME}"                                   \
    --set "dataset_kwargs.path=${LABELS_NPZ}"                      \
    --set "dataset_kwargs.capacity=${EFFECTIVE_CAPACITY}"          \
    --set "budget=${BUDGET}"                                       \
    "${EXTRA_SETS[@]+"${EXTRA_SETS[@]}"}"

# Write sentinel only on clean exit (all steps reached).
# A walltime kill (SIGTERM) prevents this, so the next resubmit resumes training.
touch "${RUN_PATH}/training_complete"
echo "Training complete — sentinel written to ${RUN_PATH}/training_complete"
