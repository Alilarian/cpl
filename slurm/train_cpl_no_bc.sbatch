#!/bin/bash
# Train CPL (any feedback type) without BC warmup — one array task per run.
#
# Uses slurm/manifests/cpl_no_bc_all/manifest.tsv which covers all 7 feedback
# types × 5 environments = 35 total training runs.
#
# Submit with --array=1-35 to train all at once, or use specific ranges:
#   --array=1-5    pref      (tasks 1–5)
#   --array=6-10   corr      (tasks 6–10)
#   --array=11-15  demo      (tasks 11–15)
#   --array=16-20  estop     (tasks 16–20)
#   --array=21-25  seq_estop (tasks 21–25)
#   --array=26-30  scalar    (tasks 26–30)
#   --array=31-35  credit    (tasks 31–35)
#
# Usage:
#   sbatch --array=1-35 \
#          --job-name=cpl_no_bc \
#          --time=12:00:00 \
#          --cpus-per-task=4 \
#          --mem=32G \
#          --gres=gpu:1 \
#          --account=<account> \
#          --partition=<partition> \
#          slurm/train_cpl_no_bc.sbatch

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=slurm/logs/%x_%A_%a.out
#SBATCH --error=slurm/logs/%x_%A_%a.err

set -euo pipefail

MANIFEST="slurm/manifests/cpl_no_bc_all/manifest.tsv"

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "SLURM_ARRAY_TASK_ID is not set. Submit with sbatch --array=1-35 ..."
    exit 1
fi

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"

if [[ -f "${SUBMIT_DIR}/setup_shell.sh" ]]; then
    REPO_ROOT="$SUBMIT_DIR"
else
    echo "Could not locate repo root (no setup_shell.sh in ${SUBMIT_DIR})"
    exit 1
fi

MANIFEST_PATH="${REPO_ROOT}/${MANIFEST}"

if [[ ! -f "$MANIFEST_PATH" ]]; then
    echo "Manifest not found: $MANIFEST_PATH"
    exit 1
fi

TASK_LINE="$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$MANIFEST_PATH")"
if [[ -z "$TASK_LINE" ]]; then
    echo "No manifest entry for SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
    exit 1
fi

IFS=$'\t' read -r CONFIG_PATH RUN_PATH <<< "$TASK_LINE"
[[ "$CONFIG_PATH" != /* ]] && CONFIG_PATH="${REPO_ROOT}/${CONFIG_PATH}"
[[ "$RUN_PATH"    != /* ]] && RUN_PATH="${REPO_ROOT}/${RUN_PATH}"

echo "=============================="
echo "SLURM_JOB_ID  = ${SLURM_JOB_ID:-}"
echo "ARRAY_TASK_ID = ${SLURM_ARRAY_TASK_ID}"
echo "CONFIG_PATH   = ${CONFIG_PATH}"
echo "RUN_PATH      = ${RUN_PATH}"
echo "=============================="

. "${REPO_ROOT}/setup_shell.sh"
mkdir -p "$(dirname "$RUN_PATH")"

export WANDB_PROJECT="cpl-no-bc"

python3 "${REPO_ROOT}/scripts/train.py" --config "$CONFIG_PATH" --path "$RUN_PATH"
