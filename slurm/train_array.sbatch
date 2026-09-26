#!/bin/bash
# Usage:
#   sbatch --array=1-N --time=24:00:00 slurm/train_array.sbatch /path/to/manifest.tsv

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=slurm/logs/%x_%A_%a.out
#SBATCH --error=slurm/logs/%x_%A_%a.err

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: sbatch [sbatch args] slurm/train_array.sbatch /path/to/manifest.tsv"
    exit 1
fi

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "SLURM_ARRAY_TASK_ID is not set. Submit this script with sbatch --array=..."
    exit 1
fi

RAW_MANIFEST_PATH="$1"

resolve_path() {
    local target_path="$1"
    local base_path="$2"
    if [[ "$target_path" = /* ]]; then
        printf '%s\n' "$target_path"
    else
        printf '%s\n' "${base_path}/${target_path}"
    fi
}

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
MANIFEST_PATH="$(resolve_path "$RAW_MANIFEST_PATH" "$SUBMIT_DIR")"
MANIFEST_PATH="$(cd "$(dirname "$MANIFEST_PATH")" && pwd)/$(basename "$MANIFEST_PATH")"

if [[ ! -f "$MANIFEST_PATH" ]]; then
    echo "Manifest not found: $MANIFEST_PATH"
    exit 1
fi

REPO_ROOT_FROM_MANIFEST="$(cd "$(dirname "$MANIFEST_PATH")/../../.." && pwd)"
if [[ -f "${SUBMIT_DIR}/setup_shell.sh" ]]; then
    REPO_ROOT="$SUBMIT_DIR"
elif [[ -f "${REPO_ROOT_FROM_MANIFEST}/setup_shell.sh" ]]; then
    REPO_ROOT="$REPO_ROOT_FROM_MANIFEST"
else
    echo "Could not locate repo root from SLURM_SUBMIT_DIR=${SUBMIT_DIR} or MANIFEST_PATH=${MANIFEST_PATH}"
    exit 1
fi

TASK_LINE="$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$MANIFEST_PATH")"
if [[ -z "$TASK_LINE" ]]; then
    echo "No manifest entry for SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
    exit 1
fi

IFS=$'\t' read -r CONFIG_PATH RUN_PATH_RAW <<< "$TASK_LINE"
CONFIG_PATH="$(resolve_path "$CONFIG_PATH" "$REPO_ROOT")"
RUN_PATH="$(resolve_path "$RUN_PATH_RAW" "$REPO_ROOT")"

# Default WandB run name/group from the manifest's own RUN_PATH -- e.g.
# "runs/mw_b10k_mixed/mw_drawer-open-v2/credit_assignment+demo_10pct/cpl_s0"
# becomes name "mw_b10k_mixed/mw_drawer-open-v2/credit_assignment+demo_10pct/cpl_s0",
# group "mw_b10k_mixed/mw_drawer-open-v2/credit_assignment+demo_10pct" (groups
# the 3 seeds of one combo together in the WandB UI). RUN_PATH already
# uniquely identifies every task in a manifest, so this is unique and
# descriptive by construction -- without it every task falls back to
# WANDB_RUN_NAME=os.path.basename(RUN_PATH) (train.py), which collides across
# every task that happens to share a basename like "cpl_s0". Only fills in
# when the caller hasn't already exported these, so submissions that set
# WANDB_RUN_NAME/WANDB_GROUP/WANDB_PROJECT themselves are unaffected.
WANDB_NAME_DEFAULT="${RUN_PATH_RAW#runs/}"
: "${WANDB_RUN_NAME:=$WANDB_NAME_DEFAULT}"
: "${WANDB_GROUP:=$(dirname "$WANDB_NAME_DEFAULT")}"
export WANDB_RUN_NAME WANDB_GROUP

echo "SLURM_JOBID=${SLURM_JOB_ID:-}"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "SLURM_SUBMIT_DIR=${SUBMIT_DIR}"
echo "REPO_ROOT=${REPO_ROOT}"
echo "MANIFEST_PATH=${MANIFEST_PATH}"
echo "CONFIG_PATH=${CONFIG_PATH}"
echo "RUN_PATH=${RUN_PATH}"
echo "WANDB_PROJECT=${WANDB_PROJECT:-<default: repo dir name>}"
echo "WANDB_GROUP=${WANDB_GROUP}"
echo "WANDB_RUN_NAME=${WANDB_RUN_NAME}"

. "${REPO_ROOT}/setup_shell.sh"
mkdir -p "$(dirname "$RUN_PATH")"

python3 "${REPO_ROOT}/scripts/train.py" --config "$CONFIG_PATH" --path "$RUN_PATH"
