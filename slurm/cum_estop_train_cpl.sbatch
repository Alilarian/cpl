#!/bin/bash
# Train the Cumulative E-stop model (MixedCPL on its two branches,
# stop_correction + no_stop_demo, equal-weighted) for one MetaWorld env,
# across N_SEEDS seeds via a SLURM array -- one task per seed.
#
# Assumes stop_correction_labels.npz / no_stop_demo_labels.npz already exist
# per env under LABELS_DIR -- generate with
# slurm/cum_estop_generate_labels_sharded.sbatch + scripts/merge_label_shards.py.
#
# Usage (GPU):
#   ENV_NAME=mw_button-press-v2 SEEDS="0 1 2" DEVICE=cuda CONDA_ENV_NAME=<env> \
#     sbatch --array=0-2 --gres=gpu:1 \
#            --account=<acct> --partition=<gpu-partition> \
#            [--time=D-HH:MM:SS] [--cpus-per-task=N] [--mem=XG] \
#            slurm/cum_estop_train_cpl.sbatch
#
# SEEDS and --array must agree: --array=0-(N_SEEDS-1).
#
# Resume: if killed by walltime, resubmit the same command -- the trainer
# auto-resumes from final_model.pt; completed runs are skipped.
#
# Optional env overrides:
#   ENV_NAME           default: mw_drawer-open-v2
#   SEEDS              default: "0 1 2"
#   LABELS_DIR         default: /scratch/general/vast/u1472210/cum_estop_labels
#   RUNS_DIR           default: runs/mw_cum_estop_cpl
#   CONFIG_DIR         default: configs/mw_state_dense
#   CONFIG_FILE        default: cum_estop_cpl.yaml
#   WANDB_PROJECT      default: mw-cpl-cum-estop
#   TOTAL_STEPS        override trainer_kwargs.total_steps (config default: 250000)
#   SUBSAMPLE_N        if set, draw this many rows uniformly at random (reproducible via
#                      SUBSAMPLE_SEED) from EACH branch independently -- e.g. SUBSAMPLE_N=5000
#                      (applied per-branch, not to the 90/10-mixed total)
#   SUBSAMPLE_SEED     default: 0
#   DEVICE             default: cpu (set to "cuda" to run on GPU, with --gres=gpu:1)
#   CONDA_ENV_NAME     default: cpl (set to your GPU-capable env)

#SBATCH --job-name=cum_estop_train_cpl
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
    echo "ERROR: submit with --array=0-(N_SEEDS-1), matching SEEDS"
    exit 1
fi

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
if [[ ! -f "${SUBMIT_DIR}/setup_shell.sh" ]]; then
    echo "ERROR: no setup_shell.sh in ${SUBMIT_DIR}"
    exit 1
fi
REPO_ROOT="$SUBMIT_DIR"

ENV_NAME="${ENV_NAME:-mw_drawer-open-v2}"
SEEDS="${SEEDS:-0 1 2}"
LABELS_DIR="${LABELS_DIR:-/scratch/general/vast/u1472210/cum_estop_labels}"
RUNS_DIR="${RUNS_DIR:-runs/mw_cum_estop_cpl}"
DEVICE="${DEVICE:-cpu}"

read -r -a SEED_ARR <<< "$SEEDS"
SEED="${SEED_ARR[$SLURM_ARRAY_TASK_ID]:-}"
if [[ -z "$SEED" ]]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} out of range for SEEDS=(${SEEDS})"
    exit 1
fi

[[ "$LABELS_DIR" != /* ]] && LABELS_DIR="${REPO_ROOT}/${LABELS_DIR}"
[[ "$RUNS_DIR"   != /* ]] && RUNS_DIR="${REPO_ROOT}/${RUNS_DIR}"

STOP_NPZ="${LABELS_DIR}/${ENV_NAME}/stop_correction_labels.npz"
DEMO_NPZ="${LABELS_DIR}/${ENV_NAME}/no_stop_demo_labels.npz"
RUN_TAG="cum_estop_s${SEED}"
[[ -n "${SUBSAMPLE_N:-}" ]] && RUN_TAG="${RUN_TAG}_n${SUBSAMPLE_N}"
RUN_PATH="${RUNS_DIR}/${ENV_NAME}/${RUN_TAG}"

echo "=============================="
echo "SLURM_JOB_ID   = ${SLURM_JOB_ID:-}"
echo "ENV_NAME       = ${ENV_NAME}"
echo "SEED           = ${SEED}"
echo "STOP_NPZ       = ${STOP_NPZ}"
echo "DEMO_NPZ       = ${DEMO_NPZ}"
echo "RUN_PATH       = ${RUN_PATH}"
echo "SUBSAMPLE_N    = ${SUBSAMPLE_N:-<unset, full dataset>}"
echo "DEVICE         = ${DEVICE}"
echo "CONDA_ENV_NAME = ${CONDA_ENV_NAME:-cpl (default)}"
echo "=============================="

if [[ -f "${RUN_PATH}/training_complete" ]]; then
    echo "Already completed — skipping."
    exit 0
fi
if [[ ! -f "$STOP_NPZ" || ! -f "$DEMO_NPZ" ]]; then
    echo "ERROR: label file(s) not found: ${STOP_NPZ} / ${DEMO_NPZ}"
    echo "Generate first with slurm/cum_estop_generate_labels_sharded.sbatch + scripts/merge_label_shards.py."
    exit 1
fi

. "${REPO_ROOT}/setup_shell.sh"
mkdir -p "$RUN_PATH"

if [[ "$DEVICE" == "cuda" ]]; then
    if ! python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
        echo "ERROR: DEVICE=cuda requested, but torch.cuda.is_available() is False"
        echo "in the active conda env (CONDA_ENV_NAME=${CONDA_ENV_NAME:-cpl (default)})."
        exit 1
    fi
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

# --set can't index into dataset_kwargs.components (a YAML list) -- generate
# a concrete config instead (see scripts/generate_cum_estop_standalone_config.py).
GENERATED_CONFIG="${RUN_PATH}/generated_config.yaml"
GEN_ARGS=(--env-name "$ENV_NAME" --stop-correction-path "$STOP_NPZ" --no-stop-demo-path "$DEMO_NPZ"
          --output "$GENERATED_CONFIG")
[[ -n "${SUBSAMPLE_N:-}" ]] && GEN_ARGS+=(--subsample-n "$SUBSAMPLE_N" --subsample-seed "${SUBSAMPLE_SEED:-0}")
[[ -n "${TOTAL_STEPS:-}" ]] && GEN_ARGS+=(--total-steps "$TOTAL_STEPS")
python3 "${REPO_ROOT}/scripts/generate_cum_estop_standalone_config.py" "${GEN_ARGS[@]}"

export WANDB_PROJECT="${WANDB_PROJECT:-mw-cpl-cum-estop}"
export WANDB_RUN_NAME="${ENV_NAME}/${RUN_TAG}"
export WANDB_GROUP="${ENV_NAME}"

python3 "${REPO_ROOT}/scripts/train.py" \
    --config "$GENERATED_CONFIG" \
    --path "$RUN_PATH" \
    --seed "$SEED" \
    --device "$DEVICE"

touch "${RUN_PATH}/training_complete"
echo "Training complete — sentinel written to ${RUN_PATH}/training_complete"
