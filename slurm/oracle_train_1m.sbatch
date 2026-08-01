#!/bin/bash
# Train an oracle SAC policy per MetaWorld env for 1,000,000 steps.
#
# One task per env (array index = manifest.tsv line):
#   1  mw_bin-picking-v2
#   2  mw_button-press-v2
#   3  mw_door-open-v2
#   4  mw_drawer-open-v2
#   5  mw_plate-slide-v2
#   6  mw_sweep-into-v2
#
# Checkpoints are written straight to scratch (not the home-dir repo):
#   /scratch/general/vast/u1472210/oracle_sac_1m/<env>/
#     model_<step>.pt   every 20,000 steps  (trainer_kwargs.checkpoint_freq)
#     final_model.pt    every 10,000 steps  (trainer_kwargs.eval_freq) — resume point
#     best_model.pt     whenever eval reward improves
#
# Manifest + per-run configs: slurm/manifests/oracle_sac_1m/
#   (regenerate via: PYTHONPATH=tools python3 tools/create_sweep_manifest.py
#    --sweep sweeps/oracle_sac_1m.json --output-root /scratch/general/vast/u1472210
#    --manifest-dir slurm/manifests/oracle_sac_1m)
#
# Usage:
#   sbatch --array=1-6 --time=72:00:00 --account=<acct> --partition=<part> \
#          slurm/oracle_train_1m.sbatch
#
# Example — override mem and wandb project on the command line:
#   sbatch --array=1-6 --time=72:00:00 --mem=32G \
#          --export=ALL,WANDB_PROJECT=my-oracle-project \
#          --account=<acct> --partition=<part> slurm/oracle_train_1m.sbatch
#
# Resume: if a job is killed by walltime, just resubmit the same command.
#   The trainer auto-resumes from final_model.pt on scratch; the wandb run
#   continues via wandb_run_id.txt stored in the run directory.
#   Already-finished envs (training_complete sentinel present) are skipped,
#   so it's safe to resubmit the full --array=1-6 blindly.
#
# WandB setup (do once on CHPC before submitting):
#   export WANDB_API_KEY=<your-key>          # or: wandb login
#   echo "export WANDB_API_KEY=<your-key>" >> ~/.bashrc
#
# Resource overrides (command-line flags override #SBATCH defaults below):
#   --time=HH:MM:SS     default: 72:00:00
#   --cpus-per-task=N   default: 8
#   --mem=XG            default: 16G
#
# Optional env overrides — pass via --export=ALL,VAR=value on the sbatch
# command line (or export before calling sbatch):
#   MANIFEST       default: slurm/manifests/oracle_sac_1m/manifest.tsv
#   WANDB_PROJECT  default: oracle-sac-1m

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=72:00:00
#SBATCH --output=/uufs/chpc.utah.edu/common/home/u1472210/cpl/slurm/logs/%x_%A_%a.out
#SBATCH --error=/uufs/chpc.utah.edu/common/home/u1472210/cpl/slurm/logs/%x_%A_%a.err

set -euo pipefail

mkdir -p "${SLURM_SUBMIT_DIR:-$PWD}/slurm/logs"

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "ERROR: submit with --array=1-6"
    exit 1
fi

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
if [[ ! -f "${SUBMIT_DIR}/setup_shell.sh" ]]; then
    echo "ERROR: no setup_shell.sh in ${SUBMIT_DIR}"
    exit 1
fi
REPO_ROOT="$SUBMIT_DIR"

MANIFEST="${MANIFEST:-slurm/manifests/oracle_sac_1m/manifest.tsv}"
[[ "$MANIFEST" != /* ]] && MANIFEST="${REPO_ROOT}/${MANIFEST}"

if [[ ! -f "$MANIFEST" ]]; then
    echo "ERROR: manifest not found: ${MANIFEST}"
    exit 1
fi

TASK_LINE="$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$MANIFEST")"
if [[ -z "$TASK_LINE" ]]; then
    echo "No manifest entry for SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
    exit 1
fi

IFS=$'\t' read -r CONFIG_PATH RUN_PATH <<< "$TASK_LINE"
[[ "$CONFIG_PATH" != /* ]] && CONFIG_PATH="${REPO_ROOT}/${CONFIG_PATH}"
[[ "$RUN_PATH"    != /* ]] && RUN_PATH="${REPO_ROOT}/${RUN_PATH}"

ENV_NAME="$(basename "$RUN_PATH")"

echo "=============================="
echo "SLURM_JOB_ID        = ${SLURM_JOB_ID:-}"
echo "ARRAY_TASK_ID        = ${SLURM_ARRAY_TASK_ID}"
echo "ENV_NAME             = ${ENV_NAME}"
echo "CONFIG_PATH          = ${CONFIG_PATH}"
echo "RUN_PATH             = ${RUN_PATH}"
echo "=============================="

# ── Skip if already completed ─────────────────────────────────────────────────
# The sentinel is written only after train.py exits cleanly at 1,000,000 steps.
# If a job was killed mid-training (no sentinel), the trainer resumes from
# final_model.pt, which is saved every eval_freq steps (every 10k steps).
if [[ -f "${RUN_PATH}/training_complete" ]]; then
    echo "Already completed (training_complete sentinel exists) — skipping."
    exit 0
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "ERROR: config not found: ${CONFIG_PATH}"
    exit 1
fi

. "${REPO_ROOT}/setup_shell.sh"
mkdir -p "$RUN_PATH"

# ── WandB ─────────────────────────────────────────────────────────────────────
# Project: oracle-sac-1m  (set WANDB_PROJECT to override, e.g. via
#          sbatch --export=ALL,WANDB_PROJECT=my-oracle-project ...)
# Run name / group: "<env>" — one run per env, all grouped together
# Resume: wandb_run_id.txt persists the run ID across walltime restarts
export WANDB_PROJECT="${WANDB_PROJECT:-oracle-sac-1m}"
export WANDB_RUN_NAME="${ENV_NAME}"
export WANDB_GROUP="oracle_sac_1m"

# ── Train ─────────────────────────────────────────────────────────────────────
python3 "${REPO_ROOT}/scripts/train.py" \
    --config "$CONFIG_PATH" \
    --path   "$RUN_PATH"

# Write sentinel only on clean exit (all steps reached).
# A walltime kill (SIGTERM) prevents this, so the next resubmit resumes training.
touch "${RUN_PATH}/training_complete"
echo "Training complete — sentinel written to ${RUN_PATH}/training_complete"
