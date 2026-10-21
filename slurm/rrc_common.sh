# Shared setup for the rrc_*.sbatch scripts (sourced, not submitted).
#   REPO_ROOT      = submit dir (must contain setup_shell.sh)
#   RUNS_ROOT      default: /scratch/general/vast/u1472210/rrc_sac   (SAC checkpoints are large; keep off $HOME)
#   CONDA_ENV_NAME default: cpl_gpu      DEVICE default: cuda
set -euo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
if [[ ! -f "${SUBMIT_DIR}/setup_shell.sh" ]]; then
    echo "ERROR: submit from the repo root (no setup_shell.sh in ${SUBMIT_DIR})"; exit 1
fi
REPO_ROOT="$SUBMIT_DIR"
export CONDA_ENV_NAME="${CONDA_ENV_NAME:-cpl_gpu}"
. "${REPO_ROOT}/setup_shell.sh"
cd "$REPO_ROOT"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export RUNS_ROOT="${RUNS_ROOT:-/scratch/general/vast/u1472210/rrc_sac}"
export DEVICE="${DEVICE:-cuda}"
# Optional yaml overlay for CHPC paths that differ from configs/rrc_sac/base.yaml (e.g. the pool location):
#   RRC_OVERLAY=configs/rrc_sac/chpc_paths.yaml sbatch ...
OVERLAY_ARGS=()
if [[ -n "${RRC_OVERLAY:-}" ]]; then OVERLAY_ARGS=(--overlay "$RRC_OVERLAY"); fi
mkdir -p "$RUNS_ROOT" "${REPO_ROOT}/slurm/logs"
if [[ "$DEVICE" == "cuda" ]] && ! python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
    echo "ERROR: DEVICE=cuda but torch.cuda.is_available() is False in ${CONDA_ENV_NAME}"; exit 1
fi
echo "REPO_ROOT=${REPO_ROOT} RUNS_ROOT=${RUNS_ROOT} RRC_OVERLAY=${RRC_OVERLAY:-none} DEVICE=${DEVICE} CONDA_ENV_NAME=${CONDA_ENV_NAME} JOB=${SLURM_JOB_ID:-} TASK=${SLURM_ARRAY_TASK_ID:-}"
manifest_line() {  # $1 = manifest path -> the line for this array element
    local line
    line="$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$1")"
    if [[ -z "$line" ]]; then echo "ERROR: no line ${SLURM_ARRAY_TASK_ID} in $1" >&2; exit 1; fi
    printf '%s\n' "$line"
}
