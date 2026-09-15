#!/bin/bash
# Train CPL on 5 feedback types (demo excluded for now) at a fixed 10k-comparison
# budget, drawn as a genuinely random (not top-B/nested) reproducible subsample of
# the pre-generated 100k-per-type CHPC label files.
#
# This is a sibling of mw_de_train.sbatch, not a replacement: mw_de_train.sbatch
# sweeps 11 NESTED budgets via dataset_kwargs.capacity (a prefix-slice of a
# quality-sorted file, so capacity=50 subset-of capacity=100 subset-of ... by
# construction) -- a deliberate, different experimental design (a data-efficiency
# curve). This script instead does ONE budget (10k) via dataset_kwargs.subsample_n,
# a full random permutation of the label file's row index -- the only correct way
# to draw a representative subset from scalar/seq_estop's label files, which are
# NOT laid out in i.i.d. row order (scalar is blocked by temporal window-offset;
# seq_estop is blocked by source trajectory). See research/datasets/corr_buffer.py's
# CorrBuffer docstring for the full explanation.
#
# Matrix: 5 feedback types x 3 policy-init seeds = 15 tasks
#   Types : pref  corr  seq_estop  scalar  credit_assignment   (demo excluded for now)
#   Seeds : 0  1  2
#
# The DATA subsample seed (SUBSAMPLE_SEED, default 0) is deliberately independent
# of the policy-init seed and fixed across the whole seed sweep -- all 3 seeds of
# a given type train on the *exact same* 10k-row subsample, so seed-to-seed
# variance is attributable to policy init/dropout/shuffling only, never to which
# rows got sampled.
#
# Task-ID mapping (1-indexed):
#   type_idx = (TASK_ID - 1) / 3   (0..4)
#   seed_idx = (TASK_ID - 1) % 3   (0..2)
#
# By-type task ranges (submit one type at a time):
#   pref                1–3
#   corr                4–6
#   seq_estop           7–9
#   scalar             10–12
#   credit_assignment  13–15
#
# demo is excluded from this matrix for now -- re-add it to TYPES below to bring
# it back.
#
# Usage:
#   ENV_NAME=mw_drawer-open-v2  sbatch --array=1-15 --account=<acct> --partition=<part> slurm/mw_b10k_cpl.sbatch
#
# Resume: same convention as mw_de_train.sbatch -- resubmit the same command;
#   the trainer auto-resumes from final_model.pt, wandb continues via
#   wandb_run_id.txt, and completed runs (training_complete sentinel) are skipped.
#
# WandB setup (do once on CHPC before submitting):
#   export WANDB_API_KEY=<your-key>          # or: wandb login
#
# Optional env overrides:
#   ENV_NAME         default: mw_drawer-open-v2
#   LABELS_DIR       default: /scratch/general/vast/u1472210/mw_de_labels
#   RUNS_DIR         default: runs/mw_b10k
#   CONFIG_DIR       default: configs/mw_state_dense
#   WANDB_PROJECT    default: mw-cpl-b10k   (CPL-only project; see mw_b10k_piql.sbatch
#                              for the separate PIQL project)
#   SUBSAMPLE_SEED   default: 0  (fixed across the seed sweep -- see above)
#   BUDGET           default: 10000
#   CONTRASTIVE_BIAS override alg_kwargs.contrastive_bias (e.g. 0.5)
#   TOTAL_STEPS      default: 500000  (override trainer_kwargs.total_steps)
#   BC_STEPS         default: 0  (no BC warmup; override alg_kwargs.bc_steps)

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=/uufs/chpc.utah.edu/common/home/u1472210/cpl/slurm/logs/%x_%A_%a.out
#SBATCH --error=/uufs/chpc.utah.edu/common/home/u1472210/cpl/slurm/logs/%x_%A_%a.err

set -euo pipefail

mkdir -p "${SLURM_SUBMIT_DIR:-$PWD}/slurm/logs"

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "ERROR: submit with --array=1-15 (or a sub-range)"
    exit 1
fi

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
if [[ ! -f "${SUBMIT_DIR}/setup_shell.sh" ]]; then
    echo "ERROR: no setup_shell.sh in ${SUBMIT_DIR}"
    exit 1
fi
REPO_ROOT="$SUBMIT_DIR"

# ── Matrix ────────────────────────────────────────────────────────────────────
# demo excluded for now -- add it back to TYPES to re-include it.
TYPES=(pref corr seq_estop scalar credit_assignment)
SEEDS=(0 1 2)

ID=$(( SLURM_ARRAY_TASK_ID - 1 ))
TYPE_IDX=$(( ID / 3 ))
SEED_IDX=$(( ID % 3 ))

FEEDBACK_TYPE="${TYPES[$TYPE_IDX]}"
SEED="${SEEDS[$SEED_IDX]}"
BUDGET="${BUDGET:-10000}"
SUBSAMPLE_SEED="${SUBSAMPLE_SEED:-0}"
TOTAL_STEPS="${TOTAL_STEPS:-500000}"
BC_STEPS="${BC_STEPS:-0}"
ALG="cpl"

# ── Paths ─────────────────────────────────────────────────────────────────────
ENV_NAME="${ENV_NAME:-mw_drawer-open-v2}"
LABELS_DIR="${LABELS_DIR:-/scratch/general/vast/u1472210/mw_de_labels}"
RUNS_DIR="${RUNS_DIR:-runs/mw_b10k}"
CONFIG_DIR="${CONFIG_DIR:-configs/mw_state_dense}"

[[ "$LABELS_DIR" != /* ]] && LABELS_DIR="${REPO_ROOT}/${LABELS_DIR}"
[[ "$RUNS_DIR"   != /* ]] && RUNS_DIR="${REPO_ROOT}/${RUNS_DIR}"

# ── Config and label file per feedback type ───────────────────────────────────
declare -A TYPE_CONFIG
TYPE_CONFIG[pref]=pref_cpl.yaml
TYPE_CONFIG[corr]=corr_cpl.yaml
TYPE_CONFIG[seq_estop]=seq_estop_cpl.yaml
TYPE_CONFIG[scalar]=scalar_cpl.yaml
TYPE_CONFIG[credit_assignment]=credit_cpl.yaml

declare -A TYPE_LABEL_FILE
TYPE_LABEL_FILE[pref]=pref_labels.npz
TYPE_LABEL_FILE[corr]=corr_labels.npz
TYPE_LABEL_FILE[seq_estop]=seq_estop_labels.npz
TYPE_LABEL_FILE[scalar]=scalar_labels.npz
TYPE_LABEL_FILE[credit_assignment]=credit_labels.npz

CONFIG_PATH="${REPO_ROOT}/${CONFIG_DIR}/${TYPE_CONFIG[$FEEDBACK_TYPE]}"
LABELS_NPZ="${LABELS_DIR}/${ENV_NAME}/${TYPE_LABEL_FILE[$FEEDBACK_TYPE]}"

# type/alg/seed on disk -- groups CPL and PIQL runs of the same type together.
RUN_PATH="${RUNS_DIR}/${ENV_NAME}/${FEEDBACK_TYPE}/${ALG}_s${SEED}"

echo "=============================="
echo "ARRAY_TASK_ID  = ${SLURM_ARRAY_TASK_ID}"
echo "ALG            = ${ALG}"
echo "FEEDBACK_TYPE  = ${FEEDBACK_TYPE}"
echo "BUDGET         = ${BUDGET}"
echo "SEED           = ${SEED}"
echo "SUBSAMPLE_SEED = ${SUBSAMPLE_SEED}"
echo "TOTAL_STEPS    = ${TOTAL_STEPS}"
echo "BC_STEPS       = ${BC_STEPS}"
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

# ── Auto-clamp budget to actual dataset size ───────────────────────────────────
N_AVAILABLE=$(python3 - <<PYEOF
import numpy as np, zipfile, io
path = "${LABELS_NPZ}"
try:
    d = np.load(path, mmap_mode='r')
    print(d['obs'].shape[0])
except Exception:
    with zipfile.ZipFile(path) as z:
        with z.open('obs.npy') as f:
            buf = io.BytesIO(f.read(512))
            np.lib.format.read_magic(buf)
            shape, _, _ = np.lib.format.read_array_header_1_0(buf)
            print(shape[0])
PYEOF
)
if (( BUDGET > N_AVAILABLE )); then
    echo "NOTE: budget ${BUDGET} > available ${N_AVAILABLE} — using full dataset (subsample_n=null)."
    EFFECTIVE_SUBSAMPLE_N="null"
else
    EFFECTIVE_SUBSAMPLE_N="${BUDGET}"
fi

. "${REPO_ROOT}/setup_shell.sh"
mkdir -p "$RUN_PATH"

# ── Optional per-run overrides ────────────────────────────────────────────────
EXTRA_SETS=()
[[ -n "${CONTRASTIVE_BIAS:-}" ]] && EXTRA_SETS+=(--set "alg_kwargs.contrastive_bias=${CONTRASTIVE_BIAS}")
EXTRA_SETS+=(--set "trainer_kwargs.total_steps=${TOTAL_STEPS}")
EXTRA_SETS+=(--set "alg_kwargs.bc_steps=${BC_STEPS}")
EXTRA_SETS+=(--set "alg_kwargs.bc_coeff=0.0")

# ── WandB ─────────────────────────────────────────────────────────────────────
# Project: mw-cpl-b10k  (CPL-only; see mw_b10k_piql.sbatch for the PIQL project)
# Run name: "<env>/<type>/<alg>_s<seed>"
# Group:    "<env>/<type>"   — groups CPL and PIQL runs of the same type together
export WANDB_PROJECT="${WANDB_PROJECT:-mw-cpl-b10k}"
export WANDB_RUN_NAME="${ENV_NAME}/${FEEDBACK_TYPE}/${ALG}_s${SEED}"
export WANDB_GROUP="${ENV_NAME}/${FEEDBACK_TYPE}"

# ── Train ─────────────────────────────────────────────────────────────────────
python3 "${REPO_ROOT}/scripts/train.py" \
    --config  "$CONFIG_PATH" \
    --path    "$RUN_PATH"   \
    --seed    "$SEED"       \
    --device  cpu           \
    --set "eval_env=${ENV_NAME}"                                     \
    --set "dataset_kwargs.path=${LABELS_NPZ}"                        \
    --set "dataset_kwargs.capacity=null"                             \
    --set "dataset_kwargs.subsample_n=${EFFECTIVE_SUBSAMPLE_N}"      \
    --set "dataset_kwargs.subsample_seed=${SUBSAMPLE_SEED}"          \
    --set "budget=${BUDGET}"                                         \
    "${EXTRA_SETS[@]+"${EXTRA_SETS[@]}"}"

# Write sentinel only on clean exit (all steps reached).
touch "${RUN_PATH}/training_complete"
echo "Training complete — sentinel written to ${RUN_PATH}/training_complete"
