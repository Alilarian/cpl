#!/bin/bash
# Train CPL on window-choice scalar feedback at the b10k budget.
#
# Same design as mw_b10k_credit_s11.sbatch (500k steps, same CPL settings and
# WandB project); the label file and the budget draw differ:
#   scalar_window_labels.npz -- from scripts/generate_scalar_window_labels.py.
#   Each 64-step segment is rated every 10 steps (6 non-overlapping ratings);
#   windows of 3 ratings slide by 1, and 2 random windows per segment are kept
#   as 3-way choice sets (chosen = highest rating). No noise: V(s) uses the
#   oracle's deterministic action.
#   Rows are grouped by segment (its 2 windows adjacent, segments in random
#   order), so the budget is a PREFIX (dataset_kwargs.capacity), which keeps
#   whole segments: 10k labels = 5000 random segments x 2 windows. The prefix
#   is the same for every seed, so the seed sweep trains on identical data.
#   subsample_n would draw rows independently and split segments.
#
# Runs are stored and logged as feedback type "scalar_window", next to the b10k
# baselines:  runs/mw_b10k/<env>/scalar_window/cpl_s<seed>,
# WandB run "<env>/scalar_window/cpl_s<seed>" in mw-cpl-b10k.
#
# Matrix: 4 envs x 3 seeds = 12 tasks
#   env_idx  = (TASK_ID - 1) / 3   button-press, door-open, drawer-open, plate-slide
#   seed_idx = (TASK_ID - 1) % 3
#
# Usage (from the repo root on CHPC):  bash slurm/submit_scalar_window.sh
#
# Resume: resubmit the same task; the trainer auto-resumes from final_model.pt,
#   wandb continues via wandb_run_id.txt, and completed runs (training_complete
#   sentinel) are skipped.
#
# Optional env overrides:
#   LABELS_DIR         default: /scratch/general/vast/u1472210/scalar_window_labels
#   LABEL_FILE         default: scalar_window_labels.npz
#   TYPE_NAME          default: scalar_window  (run dir / WandB name component)
#   RUNS_DIR           default: runs/mw_b10k
#   CONFIG_DIR         default: configs/mw_state_dense
#   CPL_WANDB_PROJECT  default: mw-cpl-b10k
#   BUDGET             default: 10000  (labels; keep it a multiple of 2)
#   TOTAL_STEPS        default: 500000
#   DEVICE             default: cuda

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/uufs/chpc.utah.edu/common/home/u1472210/cpl/slurm/logs/%x_%A_%a.out
#SBATCH --error=/uufs/chpc.utah.edu/common/home/u1472210/cpl/slurm/logs/%x_%A_%a.err

set -euo pipefail

mkdir -p "${SLURM_SUBMIT_DIR:-$PWD}/slurm/logs"

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "ERROR: submit with --array=1-12 (or a sub-range)"
    exit 1
fi

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
if [[ ! -f "${SUBMIT_DIR}/setup_shell.sh" ]]; then
    echo "ERROR: no setup_shell.sh in ${SUBMIT_DIR}"
    exit 1
fi
REPO_ROOT="$SUBMIT_DIR"

# ── Matrix ────────────────────────────────────────────────────────────────────
ENVS=(mw_button-press-v2 mw_door-open-v2 mw_drawer-open-v2 mw_plate-slide-v2)
SEEDS=(0 1 2)

ID=$(( SLURM_ARRAY_TASK_ID - 1 ))
ENV_NAME="${ENVS[$(( ID / 3 ))]}"
SEED="${SEEDS[$(( ID % 3 ))]}"
ALG="cpl"

BUDGET="${BUDGET:-10000}"
TOTAL_STEPS="${TOTAL_STEPS:-500000}"
DEVICE="${DEVICE:-cuda}"
TYPE_NAME="${TYPE_NAME:-scalar_window}"

# ── Paths ─────────────────────────────────────────────────────────────────────
LABELS_DIR="${LABELS_DIR:-/scratch/general/vast/u1472210/scalar_window_labels}"
LABEL_FILE="${LABEL_FILE:-scalar_window_labels.npz}"
RUNS_DIR="${RUNS_DIR:-runs/mw_b10k}"
CONFIG_DIR="${CONFIG_DIR:-configs/mw_state_dense}"

[[ "$LABELS_DIR" != /* ]] && LABELS_DIR="${REPO_ROOT}/${LABELS_DIR}"
[[ "$RUNS_DIR"   != /* ]] && RUNS_DIR="${REPO_ROOT}/${RUNS_DIR}"

CONFIG_PATH="${REPO_ROOT}/${CONFIG_DIR}/scalar_window_cpl.yaml"
LABELS_NPZ="${LABELS_DIR}/${ENV_NAME}/${LABEL_FILE}"
RUN_PATH="${RUNS_DIR}/${ENV_NAME}/${TYPE_NAME}/${ALG}_s${SEED}"

echo "=============================="
echo "ARRAY_TASK_ID  = ${SLURM_ARRAY_TASK_ID}"
echo "ALG            = ${ALG}"
echo "FEEDBACK_TYPE  = ${TYPE_NAME}"
echo "BUDGET         = ${BUDGET}"
echo "SEED           = ${SEED}"
echo "TOTAL_STEPS    = ${TOTAL_STEPS}"
echo "DEVICE         = ${DEVICE}"
echo "ENV_NAME       = ${ENV_NAME}"
echo "CONFIG         = ${CONFIG_PATH}"
echo "LABELS         = ${LABELS_NPZ}"
echo "RUN_PATH       = ${RUN_PATH}"
echo "=============================="

if [[ -f "${RUN_PATH}/training_complete" ]]; then
    echo "Already completed (training_complete sentinel exists) — skipping."
    exit 0
fi

if [[ ! -f "$LABELS_NPZ" ]]; then
    echo "ERROR: label file not found: ${LABELS_NPZ}"
    exit 1
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "ERROR: config not found: ${CONFIG_PATH}"
    exit 1
fi

. "${REPO_ROOT}/setup_shell.sh"

# ── Label file: row count and choice-set shape ────────────────────────────────
LABEL_INFO=$(python3 - <<PYEOF
import numpy as np, zipfile
with zipfile.ZipFile("${LABELS_NPZ}") as z, z.open("obs.npy") as f:
    version = np.lib.format.read_magic(f)
    shape = np.lib.format._read_array_header(f, version)[0]
print(shape[0], shape[1], shape[2])
PYEOF
)
read -r N_AVAILABLE N_OPTIONS SUB_LEN <<< "$LABEL_INFO"
echo "Label file: N=${N_AVAILABLE}  options/label=${N_OPTIONS}  subsegment length=${SUB_LEN}"
if (( BUDGET > N_AVAILABLE )); then
    echo "NOTE: budget ${BUDGET} > available ${N_AVAILABLE} — using full dataset (capacity=null)."
    EFFECTIVE_CAPACITY="null"
else
    EFFECTIVE_CAPACITY="${BUDGET}"
    if (( BUDGET % 2 )); then
        echo "NOTE: odd budget ${BUDGET} keeps only one window of the last segment."
    fi
fi

mkdir -p "$RUN_PATH"

# ── CPL settings (identical to the b10k baselines) ────────────────────────────
export WANDB_PROJECT="${CPL_WANDB_PROJECT:-mw-cpl-b10k}"
export WANDB_RUN_NAME="${ENV_NAME}/${TYPE_NAME}/${ALG}_s${SEED}"
export WANDB_GROUP="${ENV_NAME}/${TYPE_NAME}"

# ── Train ─────────────────────────────────────────────────────────────────────
python3 "${REPO_ROOT}/scripts/train.py" \
    --config  "$CONFIG_PATH" \
    --path    "$RUN_PATH"   \
    --seed    "$SEED"       \
    --device  "$DEVICE"     \
    --set "eval_env=${ENV_NAME}"                                     \
    --set "dataset_kwargs.path=${LABELS_NPZ}"                        \
    --set "dataset_kwargs.capacity=${EFFECTIVE_CAPACITY}"            \
    --set "dataset_kwargs.subsample_n=null"                          \
    --set "budget=${BUDGET}"                                         \
    --set "trainer_kwargs.total_steps=${TOTAL_STEPS}"                \
    --set "alg_kwargs.alpha=0.1"                                     \
    --set "alg_kwargs.contrastive_bias=0.75"                         \
    --set "alg_kwargs.bc_steps=0"                                    \
    --set "alg_kwargs.bc_coeff=0.0"

# Write sentinel only on clean exit (all steps reached).
touch "${RUN_PATH}/training_complete"
echo "Training complete — sentinel written to ${RUN_PATH}/training_complete"
