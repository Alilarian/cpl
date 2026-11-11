#!/bin/bash
# Train CPL and PIQL on stride-11 credit assignment feedback at the b10k budget.
#
# Same design as mw_b10k_cpl.sbatch / mw_b10k_piql.sbatch (10k-row random
# subsample, SUBSAMPLE_SEED fixed across the seed sweep, 500k steps, same
# configs and WandB projects); only the label file changes:
#   credit_stride11_labels.npz -- 5 windows of k=20 at starts 0,11,22,33,44 of
#   each 64-step segment, derived from credit_labels.npz (45 windows, stride 1)
#   by scripts/restride_credit_labels.py.  Both files have N=100k rows in the
#   same order, so subsample_seed=0 picks the same 10k segments as the stride-1
#   credit_assignment baseline.
#
# Runs are stored and logged as feedback type "credit_s11", next to the b10k
# baselines:  runs/mw_b10k/<env>/credit_s11/<alg>_s<seed>,
# WandB run "<env>/credit_s11/<alg>_s<seed>" in mw-cpl-b10k / mw-piql-b10k.
#
# Matrix: 2 algs x 4 envs x 3 seeds = 24 tasks
#   alg_idx  = (TASK_ID - 1) / 12        cpl 1-12, piql 13-24
#   env_idx  = ((TASK_ID - 1) % 12) / 3  button-press, door-open, drawer-open, plate-slide
#   seed_idx = (TASK_ID - 1) % 3
#
# Usage (from the repo root on CHPC):  bash slurm/submit_credit_s11.sh [cpl|piql|all]
#
# Resume: resubmit the same task; the trainer auto-resumes from final_model.pt,
#   wandb continues via wandb_run_id.txt, and completed runs (training_complete
#   sentinel) are skipped.
#
# Optional env overrides:
#   LABELS_DIR         default: /scratch/general/vast/u1472210/mw_de_labels
#   LABEL_FILE         default: credit_stride11_labels.npz
#   TYPE_NAME          default: credit_s11  (run dir / WandB name component)
#   RUNS_DIR           default: runs/mw_b10k
#   CONFIG_DIR         default: configs/mw_state_dense
#   CPL_WANDB_PROJECT  default: mw-cpl-b10k
#   PIQL_WANDB_PROJECT default: mw-piql-b10k
#   SUBSAMPLE_SEED     default: 0
#   BUDGET             default: 10000
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
    echo "ERROR: submit with --array=1-24 (or a sub-range)"
    exit 1
fi

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
if [[ ! -f "${SUBMIT_DIR}/setup_shell.sh" ]]; then
    echo "ERROR: no setup_shell.sh in ${SUBMIT_DIR}"
    exit 1
fi
REPO_ROOT="$SUBMIT_DIR"

# ── Matrix ────────────────────────────────────────────────────────────────────
ALGS=(cpl piql)
ENVS=(mw_button-press-v2 mw_door-open-v2 mw_drawer-open-v2 mw_plate-slide-v2)
SEEDS=(0 1 2)

ID=$(( SLURM_ARRAY_TASK_ID - 1 ))
ALG="${ALGS[$(( ID / 12 ))]}"
ENV_NAME="${ENVS[$(( (ID % 12) / 3 ))]}"
SEED="${SEEDS[$(( ID % 3 ))]}"

BUDGET="${BUDGET:-10000}"
SUBSAMPLE_SEED="${SUBSAMPLE_SEED:-0}"
TOTAL_STEPS="${TOTAL_STEPS:-500000}"
DEVICE="${DEVICE:-cuda}"
TYPE_NAME="${TYPE_NAME:-credit_s11}"

# ── Paths ─────────────────────────────────────────────────────────────────────
LABELS_DIR="${LABELS_DIR:-/scratch/general/vast/u1472210/mw_de_labels}"
LABEL_FILE="${LABEL_FILE:-credit_stride11_labels.npz}"
RUNS_DIR="${RUNS_DIR:-runs/mw_b10k}"
CONFIG_DIR="${CONFIG_DIR:-configs/mw_state_dense}"

[[ "$LABELS_DIR" != /* ]] && LABELS_DIR="${REPO_ROOT}/${LABELS_DIR}"
[[ "$RUNS_DIR"   != /* ]] && RUNS_DIR="${REPO_ROOT}/${RUNS_DIR}"

CONFIG_PATH="${REPO_ROOT}/${CONFIG_DIR}/credit_${ALG}.yaml"
LABELS_NPZ="${LABELS_DIR}/${ENV_NAME}/${LABEL_FILE}"
RUN_PATH="${RUNS_DIR}/${ENV_NAME}/${TYPE_NAME}/${ALG}_s${SEED}"

echo "=============================="
echo "ARRAY_TASK_ID  = ${SLURM_ARRAY_TASK_ID}"
echo "ALG            = ${ALG}"
echo "FEEDBACK_TYPE  = ${TYPE_NAME}"
echo "BUDGET         = ${BUDGET}"
echo "SEED           = ${SEED}"
echo "SUBSAMPLE_SEED = ${SUBSAMPLE_SEED}"
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

# ── Label file: row count, and window layout for the stride-11 file ───────────
LABEL_INFO=$(python3 - <<PYEOF
import numpy as np, zipfile
path = "${LABELS_NPZ}"
with zipfile.ZipFile(path) as z, z.open("obs.npy") as f:
    version = np.lib.format.read_magic(f)
    shape = np.lib.format._read_array_header(f, version)[0]
d = np.load(path)
starts = d["window_starts"].tolist() if "window_starts" in d.files else list(range(shape[1]))
print(shape[0], shape[1], ",".join(map(str, starts)))
PYEOF
)
read -r N_AVAILABLE N_WINDOWS WINDOW_STARTS <<< "$LABEL_INFO"
echo "Label file: N=${N_AVAILABLE}  C=${N_WINDOWS}  window starts=${WINDOW_STARTS}"
if (( BUDGET > N_AVAILABLE )); then
    echo "NOTE: budget ${BUDGET} > available ${N_AVAILABLE} — using full dataset (subsample_n=null)."
    EFFECTIVE_SUBSAMPLE_N="null"
else
    EFFECTIVE_SUBSAMPLE_N="${BUDGET}"
fi

mkdir -p "$RUN_PATH"

# ── Per-alg settings (identical to the b10k baselines) ────────────────────────
EXTRA_SETS=(--set "trainer_kwargs.total_steps=${TOTAL_STEPS}")
if [[ "$ALG" == "cpl" ]]; then
    export WANDB_PROJECT="${CPL_WANDB_PROJECT:-mw-cpl-b10k}"
    EXTRA_SETS+=(--set "alg_kwargs.alpha=0.1")
    EXTRA_SETS+=(--set "alg_kwargs.contrastive_bias=0.75")
    EXTRA_SETS+=(--set "alg_kwargs.bc_steps=0")
    EXTRA_SETS+=(--set "alg_kwargs.bc_coeff=0.0")
else
    export WANDB_PROJECT="${PIQL_WANDB_PROJECT:-mw-piql-b10k}"
fi

# ── WandB ─────────────────────────────────────────────────────────────────────
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
    --set "dataset_kwargs.capacity=null"                             \
    --set "dataset_kwargs.subsample_n=${EFFECTIVE_SUBSAMPLE_N}"      \
    --set "dataset_kwargs.subsample_seed=${SUBSAMPLE_SEED}"          \
    --set "budget=${BUDGET}"                                         \
    "${EXTRA_SETS[@]}"

# Write sentinel only on clean exit (all steps reached).
touch "${RUN_PATH}/training_complete"
echo "Training complete — sentinel written to ${RUN_PATH}/training_complete"
