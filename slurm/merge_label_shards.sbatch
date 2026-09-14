#!/bin/bash
# Merge sharded corr/demo feedback-label files (from mw_de_generate_labels_sharded.sbatch)
# into single consolidated npz files, as a compute-node batch job instead of running
# scripts/merge_label_shards.py directly on a CHPC login node.
#
# For each env in ENVS: merges corr_labels_shard*of<NUM_SHARDS>.npz -> corr_labels.npz,
# and demo_labels_K<K>_shard*of<NUM_SHARDS>.npz -> demo_labels_K<K>.npz (K is
# auto-detected per env from an existing shard filename, since it varies by env).
# Envs with no matching shards for a type are skipped (not an error) so this is
# safe to run blindly across a mixed set of sharded/non-sharded envs.
#
# Usage:
#   sbatch --account=<account> --partition=<partition> slurm/merge_label_shards.sbatch
#
#   # Merge only specific envs:
#   ENVS="mw_button-press-v2 mw_door-open-v2" \
#     sbatch --account=<account> --partition=<partition> slurm/merge_label_shards.sbatch
#
# Resource overrides (command-line flags override #SBATCH defaults below):
#   --time=HH:MM:SS     default: 01:00:00
#   --cpus-per-task=N   default: 2
#   --mem=XG            default: 16G
#
# Optional env overrides:
#   ENVS          default: mw_button-press-v2 mw_door-open-v2 mw_drawer-open-v2 mw_plate-slide-v2
#   LABELS_DIR    default: /scratch/general/vast/u1472210/mw_de_labels
#   NUM_SHARDS    default: 8

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
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

ENVS="${ENVS:-mw_button-press-v2 mw_door-open-v2 mw_drawer-open-v2 mw_plate-slide-v2}"
LABELS_DIR="${LABELS_DIR:-/scratch/general/vast/u1472210/mw_de_labels}"
NUM_SHARDS="${NUM_SHARDS:-8}"

[[ "$LABELS_DIR" != /* ]] && LABELS_DIR="${REPO_ROOT}/${LABELS_DIR}"

echo "=============================="
echo "SLURM_JOB_ID = ${SLURM_JOB_ID:-}"
echo "ENVS         = ${ENVS}"
echo "LABELS_DIR   = ${LABELS_DIR}"
echo "NUM_SHARDS   = ${NUM_SHARDS}"
echo "=============================="

. "${REPO_ROOT}/setup_shell.sh"

shopt -s nullglob

for ENV_NAME in $ENVS; do
    ENV_DIR="${LABELS_DIR}/${ENV_NAME}"
    echo ""
    echo "== ${ENV_NAME} =="

    # ---- corr -------------------------------------------------------------
    corr_files=( "${ENV_DIR}"/corr_labels_shard*"of${NUM_SHARDS}.npz" )
    if [[ ${#corr_files[@]} -gt 0 ]]; then
        if [[ -f "${ENV_DIR}/corr_labels.npz" ]]; then
            echo "  [corr] output already exists — skipping"
        else
            python3 "${REPO_ROOT}/scripts/merge_label_shards.py" \
                --shards "${corr_files[@]}" \
                --output "${ENV_DIR}/corr_labels.npz"
        fi
    else
        echo "  [skip corr] no shards matching corr_labels_shard*of${NUM_SHARDS}.npz"
    fi

    # ---- demo (K varies per env — detect it from an existing shard0 file) -
    demo_shard0=( "${ENV_DIR}"/demo_labels_K*"_shard0of${NUM_SHARDS}.npz" )
    if [[ ${#demo_shard0[@]} -gt 0 ]]; then
        K="$(basename "${demo_shard0[0]}" | sed -E 's/demo_labels_K([0-9]+)_shard.*/\1/')"
        demo_files=( "${ENV_DIR}"/demo_labels_K"${K}"_shard*"of${NUM_SHARDS}.npz" )
        if [[ -f "${ENV_DIR}/demo_labels_K${K}.npz" ]]; then
            echo "  [demo] output already exists — skipping"
        else
            python3 "${REPO_ROOT}/scripts/merge_label_shards.py" \
                --shards "${demo_files[@]}" \
                --output "${ENV_DIR}/demo_labels_K${K}.npz"
        fi
    else
        echo "  [skip demo] no shards matching demo_labels_K*_shard0of${NUM_SHARDS}.npz"
    fi
done

shopt -u nullglob

echo ""
echo "Done."
