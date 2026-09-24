#!/bin/bash
# Sharded holding-model E-stop label generation for a MetaWorld env (array job,
# one shard per task). Mirrors mw_de_generate_labels_sharded.sbatch's
# corr/demo sharding convention -- generate_estop_hold_labels.py has the same
# "restore env state, roll out a controller, score with the oracle" cost
# profile as corr/demo (actually worse, up to O(T^2) simulator steps per
# segment before an early stop). On top of the SLURM-level sharding, each
# shard also parallelizes across CPU cores internally via --n-workers (set to
# --cpus-per-task): segments are independent, so this is a second, finer-
# grained level of parallelism corr/demo don't need as badly since they only
# do one rollout per segment (from s_0), not up to T per segment.
#
# Calibrate --threshold FIRST with scripts/tune_estop_hold_threshold.py --
# there is no default; a stale/arbitrary threshold either produces ~0 pairs
# or intervenes on nearly every segment.
#
# After all shards finish, merge with:
#   python3 scripts/merge_label_shards.py \
#       --shard-glob '<OUTPUT_DIR>/<ENV_NAME>/estop_hold_labels_shard*of<NUM_SHARDS>.npz' \
#       --output      <OUTPUT_DIR>/<ENV_NAME>/estop_hold_labels.npz
#   python3 scripts/merge_label_shards.py \
#       --shard-glob '<OUTPUT_DIR>/<ENV_NAME>/estop_hold_nostop_shard*of<NUM_SHARDS>.npz' \
#       --output      <OUTPUT_DIR>/<ENV_NAME>/estop_hold_nostop.npz
#
# Usage:
#   ENV_NAME=mw_drawer-open-v2 THRESHOLD=1.0 NUM_SHARDS=8 \
#     sbatch --array=0-7 --account=<account> --partition=<partition> \
#            [--time=HH:MM:SS] [--cpus-per-task=N] [--mem=XG] \
#            slurm/mw_estop_hold_generate_labels_sharded.sbatch
#
# NUM_SHARDS and --array must agree: --array=0-(NUM_SHARDS-1).
#
# Resource overrides (command-line flags override #SBATCH defaults below):
#   --time=HH:MM:SS     default: 12:00:00  (per shard; scale with pool size / NUM_SHARDS)
#   --cpus-per-task=N   default: 8         (also sets --n-workers, see below)
#   --mem=XG            default: 64G
#
# Optional env overrides:
#   ENV_NAME            default: mw_drawer-open-v2
#   THRESHOLD           required: fixed stopping threshold c >= 0
#   MIN_HORIZON         default: 5
#   ORACLE_RUNS_BASE    default: runs/runs/chpc/oracle_sac_seeds  (relative to repo root)
#   ORACLE_SEED         default: seed-1  (set to "" for a flat <env>/ layout with no
#                                         seed subdir, e.g. /scratch/.../oracle_sac_1m)
#   POOL_DIR            default: /scratch/general/vast/u1472210/mw_de_pool
#   OUTPUT_DIR          default: /scratch/general/vast/u1472210/estop_hold_labels
#   NUM_SHARDS          default: 8
#   N_WORKERS           default: $SLURM_CPUS_PER_TASK (intra-shard multiprocessing)
#   MAX_SEGMENTS        default: unset (debug/testing cap, applied per shard)

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/uufs/chpc.utah.edu/common/home/u1472210/cpl/slurm/logs/%x_%A_%a.out
#SBATCH --error=/uufs/chpc.utah.edu/common/home/u1472210/cpl/slurm/logs/%x_%A_%a.err

set -euo pipefail

mkdir -p "${SLURM_SUBMIT_DIR:-$PWD}/slurm/logs"

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "ERROR: submit with --array=0-(NUM_SHARDS-1)"
    exit 1
fi

if [[ -z "${THRESHOLD:-}" ]]; then
    echo "ERROR: set THRESHOLD=<c>. Calibrate first with scripts/tune_estop_hold_threshold.py."
    exit 1
fi

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
if [[ ! -f "${SUBMIT_DIR}/setup_shell.sh" ]]; then
    echo "ERROR: no setup_shell.sh in ${SUBMIT_DIR}"
    exit 1
fi
REPO_ROOT="$SUBMIT_DIR"

ENV_NAME="${ENV_NAME:-mw_drawer-open-v2}"
MIN_HORIZON="${MIN_HORIZON:-5}"
ORACLE_RUNS_BASE="${ORACLE_RUNS_BASE:-runs/runs/chpc/oracle_sac_seeds}"
ORACLE_SEED="${ORACLE_SEED-seed-1}"
POOL_DIR="${POOL_DIR:-/scratch/general/vast/u1472210/mw_de_pool}"
OUTPUT_DIR="${OUTPUT_DIR:-/scratch/general/vast/u1472210/estop_hold_labels}"
NUM_SHARDS="${NUM_SHARDS:-8}"
N_WORKERS="${N_WORKERS:-${SLURM_CPUS_PER_TASK:-8}}"
MAX_SEGMENTS="${MAX_SEGMENTS:-}"
SHARD_ID="$SLURM_ARRAY_TASK_ID"

[[ "$ORACLE_RUNS_BASE" != /* ]] && ORACLE_RUNS_BASE="${REPO_ROOT}/${ORACLE_RUNS_BASE}"
[[ "$POOL_DIR"         != /* ]] && POOL_DIR="${REPO_ROOT}/${POOL_DIR}"
[[ "$OUTPUT_DIR"       != /* ]] && OUTPUT_DIR="${REPO_ROOT}/${OUTPUT_DIR}"

POOL_PATH="${POOL_DIR}/${ENV_NAME}/pool.npz"
if [[ -n "$ORACLE_SEED" ]]; then
    ORACLE_DIR="${ORACLE_RUNS_BASE}/${ENV_NAME}/${ORACLE_SEED}"
else
    ORACLE_DIR="${ORACLE_RUNS_BASE}/${ENV_NAME}"
fi

echo "=============================="
echo "SLURM_JOB_ID  = ${SLURM_JOB_ID:-}"
echo "ARRAY_TASK_ID = ${SLURM_ARRAY_TASK_ID}"
echo "ENV_NAME      = ${ENV_NAME}"
echo "THRESHOLD     = ${THRESHOLD}"
echo "MIN_HORIZON   = ${MIN_HORIZON}"
echo "ORACLE_SEED   = ${ORACLE_SEED}"
echo "POOL_PATH     = ${POOL_PATH}"
echo "ORACLE_DIR    = ${ORACLE_DIR}"
echo "OUTPUT_DIR    = ${OUTPUT_DIR}"
echo "NUM_SHARDS    = ${NUM_SHARDS}"
echo "N_WORKERS     = ${N_WORKERS}"
echo "MAX_SEGMENTS  = ${MAX_SEGMENTS:-<unset, full shard>}"
echo "SHARD_ID      = ${SHARD_ID}"
echo "=============================="

if [[ ! -f "$POOL_PATH" ]]; then
    echo "ERROR: pool.npz not found: ${POOL_PATH}"
    exit 1
fi

. "${REPO_ROOT}/setup_shell.sh"
mkdir -p "${OUTPUT_DIR}/${ENV_NAME}"

EXTRA_ARGS=()
if [[ -n "$MAX_SEGMENTS" ]]; then
    EXTRA_ARGS+=(--max-segments "$MAX_SEGMENTS")
fi

python3 "${REPO_ROOT}/scripts/generate_estop_hold_labels.py" \
    --pool-path    "$POOL_PATH" \
    --run-dir      "$ORACLE_DIR" \
    --threshold    "$THRESHOLD" \
    --min-horizon  "$MIN_HORIZON" \
    --n-workers    "$N_WORKERS" \
    --num-shards   "$NUM_SHARDS" \
    --shard-id     "$SHARD_ID" \
    --output-dir   "$OUTPUT_DIR" \
    "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"

echo "Done — shard ${SHARD_ID}/${NUM_SHARDS} written to ${OUTPUT_DIR}/${ENV_NAME}/"
