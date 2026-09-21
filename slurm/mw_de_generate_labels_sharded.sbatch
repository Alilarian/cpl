#!/bin/bash
# Sharded corr / demo generation for a MetaWorld env (array job, one shard per task).
#
# corr and demo are the two feedback types that roll out the expert (and, for
# demo, several stratified counterfactual checkpoints) from every pool
# segment's s_0 — far slower than the pool-only scoring used by pref/seq_estop/
# scalar/credit_assignment. Splitting the pool into --num-shards contiguous
# chunks and running one array task per shard lets them finish in parallel
# instead of as one long single job. Each shard is independently resumable
# (see generate_corr_labels.py / generate_demo_labels.py --save-every).
#
# After all shards finish, merge with:
#   python3 scripts/merge_label_shards.py \
#       --shard-glob '<OUTPUT_DIR>/<ENV_NAME>/corr_labels_shard*of<NUM_SHARDS>.npz' \
#       --output      <OUTPUT_DIR>/<ENV_NAME>/corr_labels.npz
#   # (demo: shard-glob 'demo_labels_K<K>_shard*of<NUM_SHARDS>.npz', K = N_COUNTERFACTUALS+1)
#
# Usage:
#   FEEDBACK_TYPE=corr ENV_NAME=mw_drawer-open-v2 NUM_SHARDS=8 \
#     sbatch --array=0-7 --account=<account> --partition=<partition> \
#            [--time=HH:MM:SS] [--cpus-per-task=N] [--mem=XG] \
#            slurm/mw_de_generate_labels_sharded.sbatch
#
# NUM_SHARDS and --array must agree: --array=0-(NUM_SHARDS-1).
#
# Resource overrides (command-line flags override #SBATCH defaults below):
#   --time=HH:MM:SS     default: 12:00:00  (per shard; scale with pool size / NUM_SHARDS)
#   --cpus-per-task=N   default: 8
#   --mem=XG            default: 32G
#
# Optional env overrides:
#   FEEDBACK_TYPE       required: corr | demo
#   ENV_NAME            default: mw_drawer-open-v2
#   ORACLE_RUNS_BASE    default: runs/runs/chpc/oracle_sac_seeds  (relative to repo root)
#   ORACLE_SEED         default: seed-1
#   POOL_DIR            default: /scratch/general/vast/u1472210/mw_de_pool
#   OUTPUT_DIR          default: /scratch/general/vast/u1472210/mw_de_labels
#   NUM_SHARDS          default: 8
#   N_COUNTERFACTUALS   default: 6   (demo only; K = N_COUNTERFACTUALS + 1;
#                                     n_tiers = N_COUNTERFACTUALS - 1, e.g. 8 -> 7
#                                     tier checkpoints + 1 original pool segment)
#   SUCCESS_MIN         default: 0.2 (demo only; lower bound of eval/success for
#                                     the stratified tier counterfactual checkpoints)
#   SUCCESS_MAX         default: 0.6 (demo only; upper bound, see --success-min/max
#                                     in generate_demo_labels.py. Does not affect
#                                     the expert checkpoint or the pool's own
#                                     "original" segment, whose quality is instead
#                                     controlled by how the pool itself was built
#                                     -- see scripts/verify_pool_checkpoint_success.py)

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/uufs/chpc.utah.edu/common/home/u1472210/cpl/slurm/logs/%x_%A_%a.out
#SBATCH --error=/uufs/chpc.utah.edu/common/home/u1472210/cpl/slurm/logs/%x_%A_%a.err

set -euo pipefail

mkdir -p "${SLURM_SUBMIT_DIR:-$PWD}/slurm/logs"

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "ERROR: submit with --array=0-(NUM_SHARDS-1)"
    exit 1
fi

if [[ "${FEEDBACK_TYPE:-}" != "corr" && "${FEEDBACK_TYPE:-}" != "demo" ]]; then
    echo "ERROR: set FEEDBACK_TYPE=corr or FEEDBACK_TYPE=demo"
    exit 1
fi

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
if [[ ! -f "${SUBMIT_DIR}/setup_shell.sh" ]]; then
    echo "ERROR: no setup_shell.sh in ${SUBMIT_DIR}"
    exit 1
fi
REPO_ROOT="$SUBMIT_DIR"

ENV_NAME="${ENV_NAME:-mw_drawer-open-v2}"
ORACLE_RUNS_BASE="${ORACLE_RUNS_BASE:-runs/runs/chpc/oracle_sac_seeds}"
ORACLE_SEED="${ORACLE_SEED:-seed-1}"
POOL_DIR="${POOL_DIR:-/scratch/general/vast/u1472210/mw_de_pool}"
OUTPUT_DIR="${OUTPUT_DIR:-/scratch/general/vast/u1472210/mw_de_labels}"
NUM_SHARDS="${NUM_SHARDS:-8}"
N_COUNTERFACTUALS="${N_COUNTERFACTUALS:-6}"
SUCCESS_MIN="${SUCCESS_MIN:-0.2}"
SUCCESS_MAX="${SUCCESS_MAX:-0.6}"
SHARD_ID="$SLURM_ARRAY_TASK_ID"

[[ "$ORACLE_RUNS_BASE" != /* ]] && ORACLE_RUNS_BASE="${REPO_ROOT}/${ORACLE_RUNS_BASE}"
[[ "$POOL_DIR"         != /* ]] && POOL_DIR="${REPO_ROOT}/${POOL_DIR}"
[[ "$OUTPUT_DIR"       != /* ]] && OUTPUT_DIR="${REPO_ROOT}/${OUTPUT_DIR}"

POOL_PATH="${POOL_DIR}/${ENV_NAME}/pool.npz"
ORACLE_DIR="${ORACLE_RUNS_BASE}/${ENV_NAME}/${ORACLE_SEED}"

echo "=============================="
echo "SLURM_JOB_ID    = ${SLURM_JOB_ID:-}"
echo "ARRAY_TASK_ID   = ${SLURM_ARRAY_TASK_ID}"
echo "FEEDBACK_TYPE   = ${FEEDBACK_TYPE}"
echo "ENV_NAME        = ${ENV_NAME}"
echo "ORACLE_SEED     = ${ORACLE_SEED}"
echo "POOL_PATH       = ${POOL_PATH}"
echo "ORACLE_DIR      = ${ORACLE_DIR}"
echo "OUTPUT_DIR      = ${OUTPUT_DIR}"
echo "NUM_SHARDS      = ${NUM_SHARDS}"
echo "N_COUNTERFACTUALS = ${N_COUNTERFACTUALS}  (demo only)"
echo "SUCCESS_MIN     = ${SUCCESS_MIN}  (demo only)"
echo "SUCCESS_MAX     = ${SUCCESS_MAX}  (demo only)"
echo "SHARD_ID        = ${SHARD_ID}"
echo "=============================="

if [[ ! -f "$POOL_PATH" ]]; then
    echo "ERROR: pool.npz not found: ${POOL_PATH}"
    exit 1
fi

. "${REPO_ROOT}/setup_shell.sh"
mkdir -p "${OUTPUT_DIR}/${ENV_NAME}"

case "$FEEDBACK_TYPE" in

    corr)
        python3 "${REPO_ROOT}/scripts/generate_corr_labels.py" \
            --pool-path        "$POOL_PATH" \
            --run-dir          "$ORACLE_DIR" \
            --min-adv-gap      0.0 \
            --mcmc-samples     16 \
            --score-batch-size 64 \
            --num-shards       "$NUM_SHARDS" \
            --shard-id         "$SHARD_ID" \
            --output-dir       "$OUTPUT_DIR" \
            --device cpu
        ;;

    demo)
        python3 "${REPO_ROOT}/scripts/generate_demo_labels.py" \
            --pool-path            "$POOL_PATH" \
            --run-dir              "$ORACLE_DIR" \
            --n-counterfactuals    "$N_COUNTERFACTUALS" \
            --checkpoints-per-tier 2 \
            --success-min          "$SUCCESS_MIN" \
            --success-max          "$SUCCESS_MAX" \
            --min-adv-gap          0.0 \
            --mcmc-samples         16 \
            --score-batch-size     32 \
            --num-shards           "$NUM_SHARDS" \
            --shard-id             "$SHARD_ID" \
            --output-dir           "$OUTPUT_DIR" \
            --device cpu
        ;;
esac

echo "Done — shard ${SHARD_ID}/${NUM_SHARDS} of ${FEEDBACK_TYPE} written to ${OUTPUT_DIR}/${ENV_NAME}/"
