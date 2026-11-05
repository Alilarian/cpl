#!/bin/bash
# Merge one env's cum_estop label shards (slurm/cum_estop_generate_labels_sharded.sbatch),
# verify them against the pool (scripts/verify_cum_estop_labels.py), and build
# the 50/50 training mix once -- so the CPL/PIQL array tasks that follow never
# race to build the same mix file themselves.
#
# Usage:
#   ENV_NAME=mw_button-press-v2 OUTPUT_DIR=<labels dir> NUM_SHARDS=4 \
#     sbatch --account=<acct> --partition=<part> slurm/cum_estop_merge_labels.sbatch
#
# Rerunning is safe: merged files and the mix are skipped if they already exist.
#
# Optional env overrides:
#   ENV_NAME           required
#   OUTPUT_DIR         default: /scratch/general/vast/u1472210/cum_estop_labels
#   NUM_SHARDS         default: 8
#   POOL_DIR           default: /scratch/general/vast/u1472210/mw_de_pool
#   CORRECTION_POLICY  default: oracle (verify expects oracle corrections)
#   N_PER_BRANCH       default: 5000 -- must match the training sbatch's N_PER_BRANCH
#   MIX_SEED           default: 0    -- must match the training sbatch's MIX_SEED

#SBATCH --job-name=cum_estop_merge_labels
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/uufs/chpc.utah.edu/common/home/u1472210/cpl/slurm/logs/%x_%j.out
#SBATCH --error=/uufs/chpc.utah.edu/common/home/u1472210/cpl/slurm/logs/%x_%j.err

set -euo pipefail
mkdir -p "${SLURM_SUBMIT_DIR:-$PWD}/slurm/logs"

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
if [[ ! -f "${SUBMIT_DIR}/setup_shell.sh" ]]; then
    echo "ERROR: no setup_shell.sh in ${SUBMIT_DIR}"
    exit 1
fi
REPO_ROOT="$SUBMIT_DIR"

if [[ -z "${ENV_NAME:-}" ]]; then
    echo "ERROR: set ENV_NAME"
    exit 1
fi
OUTPUT_DIR="${OUTPUT_DIR:-/scratch/general/vast/u1472210/cum_estop_labels}"
NUM_SHARDS="${NUM_SHARDS:-8}"
POOL_DIR="${POOL_DIR:-/scratch/general/vast/u1472210/mw_de_pool}"
CORRECTION_POLICY="${CORRECTION_POLICY:-oracle}"
N_PER_BRANCH="${N_PER_BRANCH:-5000}"
MIX_SEED="${MIX_SEED:-0}"

[[ "$OUTPUT_DIR" != /* ]] && OUTPUT_DIR="${REPO_ROOT}/${OUTPUT_DIR}"
[[ "$POOL_DIR"   != /* ]] && POOL_DIR="${REPO_ROOT}/${POOL_DIR}"
LABELS="${OUTPUT_DIR}/${ENV_NAME}"
MIX_NPZ="${LABELS}/cum_estop_mix$((N_PER_BRANCH * 2)).npz"

echo "=============================="
echo "SLURM_JOB_ID      = ${SLURM_JOB_ID:-}"
echo "ENV_NAME          = ${ENV_NAME}"
echo "LABELS            = ${LABELS}"
echo "NUM_SHARDS        = ${NUM_SHARDS}"
echo "CORRECTION_POLICY = ${CORRECTION_POLICY}"
echo "MIX_NPZ           = ${MIX_NPZ}"
echo "=============================="

. "${REPO_ROOT}/setup_shell.sh"

if [[ -f "${LABELS}/stop_correction_labels.npz" && -f "${LABELS}/no_stop_demo_labels.npz" \
      && -f "${LABELS}/unresolved_events.npz" ]]; then
    echo "Already merged: ${LABELS}"
else
    # Every shard writes unresolved_events (possibly empty), so a missing one means a shard never finished.
    n_done=$(compgen -G "${LABELS}/unresolved_events_shard*of${NUM_SHARDS}.npz" | wc -l)
    if [[ "$n_done" -ne "$NUM_SHARDS" ]]; then
        echo "ERROR: ${n_done}/${NUM_SHARDS} shards finished in ${LABELS}"
        exit 1
    fi
    for name in stop_correction_labels no_stop_demo_labels unresolved_events; do
        python3 "${REPO_ROOT}/scripts/merge_label_shards.py" \
            --shard-glob "${LABELS}/${name}_shard*of${NUM_SHARDS}.npz" \
            --output "${LABELS}/${name}.npz"
    done
fi

VERIFY_ARGS=()
[[ "$CORRECTION_POLICY" == "oracle" ]] && VERIFY_ARGS+=(--oracle)
python3 "${REPO_ROOT}/scripts/verify_cum_estop_labels.py" \
    --labels-dir "$LABELS" --pool-path "${POOL_DIR}/${ENV_NAME}/pool.npz" \
    "${VERIFY_ARGS[@]+"${VERIFY_ARGS[@]}"}"

if [[ -f "$MIX_NPZ" ]]; then
    echo "Mix already exists: ${MIX_NPZ}"
else
    # build to a temp name, then rename, so no reader ever sees a partial mix
    python3 "${REPO_ROOT}/scripts/build_cum_estop_mix.py" \
        --stop-correction "${LABELS}/stop_correction_labels.npz" \
        --no-stop-demo "${LABELS}/no_stop_demo_labels.npz" \
        --n-per-branch "$N_PER_BRANCH" --seed "$MIX_SEED" --output "${MIX_NPZ}.tmp"
    mv "${MIX_NPZ}.tmp" "$MIX_NPZ"
fi
echo "Done — ${MIX_NPZ}"
