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

MANIFEST_PATH="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

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

echo "SLURM_JOBID=${SLURM_JOB_ID:-}"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "CONFIG_PATH=${CONFIG_PATH}"
echo "RUN_PATH=${RUN_PATH}"

. "${REPO_ROOT}/setup_shell.sh"
mkdir -p "$(dirname "$RUN_PATH")"

python3 "${REPO_ROOT}/scripts/train.py" --config "$CONFIG_PATH" --path "$RUN_PATH"
