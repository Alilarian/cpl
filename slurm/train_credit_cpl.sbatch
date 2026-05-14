#!/bin/bash
# Train CPL with credit assignment feedback — one array task per environment (no BC warmup).
#
# Uses credit_labels.npz from generate_credit_labels.sbatch.
# Trains CreditAssignmentCPL (multi-way cross-entropy over C candidate windows)
# with bc_steps=0.
#
# Usage:
#   sbatch --array=1-5 \
#          --job-name=train_credit_cpl \
#          --time=12:00:00 \
#          --cpus-per-task=4 \
#          --mem=32G \
#          --gres=gpu:1 \
#          --account=<account> \
#          --partition=<partition> \
#          slurm/train_credit_cpl.sbatch

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=slurm/logs/%x_%A_%a.out
#SBATCH --error=slurm/logs/%x_%A_%a.err

set -euo pipefail

MANIFEST="slurm/manifests/credit_cpl_all/manifest.tsv"

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "SLURM_ARRAY_TASK_ID is not set. Submit with sbatch --array=1-5 ..."
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

export WANDB_PROJECT="cpl-credit"

python3 "${REPO_ROOT}/scripts/train.py" --config "$CONFIG_PATH" --path "$RUN_PATH"
