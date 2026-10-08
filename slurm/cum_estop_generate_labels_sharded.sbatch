#!/bin/bash
# Phases 3/5/6 of the Cumulative E-stop model: sharded label generation for
# one MetaWorld env (array job, one shard per task). Each shard also
# parallelizes across CPU cores internally via --n-workers.
#
# Prerequisites:
#   1. slurm/cum_estop_select_bands.sbatch   -> checkpoint_bands.json
#   2. slurm/cum_estop_tune_threshold.sbatch  -> pick H from its printed table
#
# After all shards finish, merge each branch separately with the existing
# generic merge tool:
#   python3 scripts/merge_label_shards.py \
#       --shard-glob '<OUTPUT_DIR>/<ENV_NAME>/stop_correction_labels_shard*of<NUM_SHARDS>.npz' \
#       --output      <OUTPUT_DIR>/<ENV_NAME>/stop_correction_labels.npz
#   python3 scripts/merge_label_shards.py \
#       --shard-glob '<OUTPUT_DIR>/<ENV_NAME>/no_stop_demo_labels_shard*of<NUM_SHARDS>.npz' \
#       --output      <OUTPUT_DIR>/<ENV_NAME>/no_stop_demo_labels.npz
#   python3 scripts/merge_label_shards.py \
#       --shard-glob '<OUTPUT_DIR>/<ENV_NAME>/unresolved_events_shard*of<NUM_SHARDS>.npz' \
#       --output      <OUTPUT_DIR>/<ENV_NAME>/unresolved_events.npz
#
# Usage:
#   ENV_NAME=mw_button-press-v2 H=10.27 NUM_SHARDS=8 SUBSAMPLE_N=30000 \
#     sbatch --array=0-7 --account=<acct> --partition=<part> \
#            [--time=HH:MM:SS] [--cpus-per-task=N] [--mem=XG] \
#            slurm/cum_estop_generate_labels_sharded.sbatch
#
# NUM_SHARDS and --array must agree: --array=0-(NUM_SHARDS-1).
#
# Optional env overrides:
#   ENV_NAME            default: mw_drawer-open-v2
#   H                   required: cumulative deficit threshold (calibrate first)
#   ORACLE_RUNS_BASE    default: /scratch/general/vast/u1472210/oracle_sac_1m
#   ORACLE_CHECKPOINT   default: best_model.pt
#   BANDS_PATH          default: <OUTPUT_DIR>/<ENV_NAME>/checkpoint_bands.json
#   POOL_DIR            default: /scratch/general/vast/u1472210/mw_de_pool
#   OUTPUT_DIR          default: /scratch/general/vast/u1472210/cum_estop_labels
#   GAMMA               default: 0.99
#   MCMC_SAMPLES        default: 32
#   DELTA_N             default: 0.3
#   CORRECTION_BUDGET   default: 8 (retries only on simulator early-termination, not quality -- a
#                                 correction is accepted on the first completed rollout regardless of
#                                 whether it beats the original; whichever side scores higher wins)
#   NEGATIVE_BUDGET     default: 8
#   NUM_SHARDS          default: 8
#   N_WORKERS           default: $SLURM_CPUS_PER_TASK
#   SUBSAMPLE_N         default: unset (full pool; see mw_estop_hold_generate_labels_sharded.sbatch's
#                                 note on why this, not MAX_SEGMENTS, is the right way to cap pool size)
#   SUBSAMPLE_SEED      default: 0
#   SEED                default: 0 (checkpoint-sampling seed for correction/negative retries)

#SBATCH --job-name=cum_estop_generate_labels
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/uufs/chpc.utah.edu/common/home/u1472210/cpl/slurm/logs/%x_%A_%a.out
#SBATCH --error=/uufs/chpc.utah.edu/common/home/u1472210/cpl/slurm/logs/%x_%A_%a.err

set -euo pipefail
mkdir -p "${SLURM_SUBMIT_DIR:-$PWD}/slurm/logs"

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "ERROR: submit with --array=0-(NUM_SHARDS-1)"
    exit 1
fi
if [[ -z "${H:-}" ]]; then
    echo "ERROR: set H=<threshold>. Calibrate first with slurm/cum_estop_tune_threshold.sbatch."
    exit 1
fi

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
if [[ ! -f "${SUBMIT_DIR}/setup_shell.sh" ]]; then
    echo "ERROR: no setup_shell.sh in ${SUBMIT_DIR}"
    exit 1
fi
REPO_ROOT="$SUBMIT_DIR"

ENV_NAME="${ENV_NAME:-mw_drawer-open-v2}"
ORACLE_RUNS_BASE="${ORACLE_RUNS_BASE:-/scratch/general/vast/u1472210/oracle_sac_1m}"
ORACLE_CHECKPOINT="${ORACLE_CHECKPOINT:-best_model.pt}"
POOL_DIR="${POOL_DIR:-/scratch/general/vast/u1472210/mw_de_pool}"
OUTPUT_DIR="${OUTPUT_DIR:-/scratch/general/vast/u1472210/cum_estop_labels}"
GAMMA="${GAMMA:-0.99}"
MCMC_SAMPLES="${MCMC_SAMPLES:-32}"
DELTA_N="${DELTA_N:-0.3}"
CORRECTION_BUDGET="${CORRECTION_BUDGET:-8}"
NEGATIVE_BUDGET="${NEGATIVE_BUDGET:-8}"
NUM_SHARDS="${NUM_SHARDS:-8}"
N_WORKERS="${N_WORKERS:-${SLURM_CPUS_PER_TASK:-8}}"
SUBSAMPLE_N="${SUBSAMPLE_N:-}"
SUBSAMPLE_SEED="${SUBSAMPLE_SEED:-0}"
SEED="${SEED:-0}"
SHARD_ID="$SLURM_ARRAY_TASK_ID"

[[ "$ORACLE_RUNS_BASE" != /* ]] && ORACLE_RUNS_BASE="${REPO_ROOT}/${ORACLE_RUNS_BASE}"
[[ "$POOL_DIR"         != /* ]] && POOL_DIR="${REPO_ROOT}/${POOL_DIR}"
[[ "$OUTPUT_DIR"       != /* ]] && OUTPUT_DIR="${REPO_ROOT}/${OUTPUT_DIR}"

POOL_PATH="${POOL_DIR}/${ENV_NAME}/pool.npz"
ORACLE_DIR="${ORACLE_RUNS_BASE}/${ENV_NAME}"
BANDS_PATH="${BANDS_PATH:-${OUTPUT_DIR}/${ENV_NAME}/checkpoint_bands.json}"

echo "=============================="
echo "SLURM_JOB_ID  = ${SLURM_JOB_ID:-}"
echo "ARRAY_TASK_ID = ${SLURM_ARRAY_TASK_ID}"
echo "ENV_NAME      = ${ENV_NAME}"
echo "H             = ${H}"
echo "POOL_PATH     = ${POOL_PATH}"
echo "ORACLE_DIR    = ${ORACLE_DIR}"
echo "BANDS_PATH    = ${BANDS_PATH}"
echo "OUTPUT_DIR    = ${OUTPUT_DIR}"
echo "NUM_SHARDS    = ${NUM_SHARDS}"
echo "N_WORKERS     = ${N_WORKERS}"
echo "SUBSAMPLE_N   = ${SUBSAMPLE_N:-<unset, full pool>}"
echo "SHARD_ID      = ${SHARD_ID}"
echo "=============================="

if [[ ! -f "$POOL_PATH" ]]; then
    echo "ERROR: pool.npz not found: ${POOL_PATH}"
    exit 1
fi
if [[ ! -f "$BANDS_PATH" ]]; then
    echo "ERROR: checkpoint_bands.json not found: ${BANDS_PATH}"
    echo "Generate it first with slurm/cum_estop_select_bands.sbatch."
    exit 1
fi

. "${REPO_ROOT}/setup_shell.sh"
mkdir -p "${OUTPUT_DIR}/${ENV_NAME}"

EXTRA_ARGS=()
if [[ -n "$SUBSAMPLE_N" ]]; then
    EXTRA_ARGS+=(--subsample-n "$SUBSAMPLE_N" --subsample-seed "$SUBSAMPLE_SEED")
fi

python3 "${REPO_ROOT}/scripts/generate_cum_estop_labels.py" \
    --pool-path "$POOL_PATH" \
    --run-dir "$ORACLE_DIR" \
    --oracle-checkpoint "$ORACLE_CHECKPOINT" \
    --bands-path "$BANDS_PATH" \
    --H "$H" \
    --gamma "$GAMMA" \
    --mcmc-samples "$MCMC_SAMPLES" \
    --delta-n "$DELTA_N" \
    --correction-budget "$CORRECTION_BUDGET" \
    --negative-budget "$NEGATIVE_BUDGET" \
    --n-workers "$N_WORKERS" \
    --seed "$SEED" \
    --num-shards "$NUM_SHARDS" \
    --shard-id "$SHARD_ID" \
    --output-dir "$OUTPUT_DIR" \
    --env-name "$ENV_NAME" \
    "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"

echo "Done — shard ${SHARD_ID}/${NUM_SHARDS} written to ${OUTPUT_DIR}/${ENV_NAME}/"
