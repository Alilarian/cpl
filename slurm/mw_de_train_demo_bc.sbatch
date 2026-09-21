#!/bin/bash
# Train the BC baseline (multi-type-feedback/multi_type_feedback/train_bc.py) on
# the demo track (K-index 0, no counterfactuals -- see
# scripts/extract_demo_track_for_bc.py) of demo_labels_K<K>.npz, for one
# MetaWorld env across N_SEEDS seeds via a SLURM array.
#
# Prerequisite (run once BEFORE submitting this, not part of this script): for
# EVERY seed you submit here, produce the demo pickle train_bc.py expects at
#   multi-type-feedback/feedback_regen/<ALGORITHM>_<metaworld-env>_<seed>.pkl
# via scripts/extract_demo_track_for_bc.py + convert_research_pool_to_demos.py.
# The demo DATA doesn't depend on the BC seed (it's the same K=0 track for every
# seed), but train_bc.py's own filename convention embeds --seed regardless
# (see TrainingUtils.get_model_ids), so the same converted pickle needs to be
# copied to each seed-suffixed filename you plan to submit.
#
# Runs on CPU: train_bc.py used to hardcode device="cuda:0" for imitation's BC
# trainer via a duplicate bc.BC(...) call that shadowed an earlier, correct
# device=TrainingUtils.get_device() call -- fixed (dead code removed) so it now
# picks CPU automatically when no GPU is allocated. Pass --gres=gpu:1 and set
# DEVICE=cuda below to go back to GPU.
# Requires multi-type-feedback's own dependencies (imitation, stable_baselines3,
# gymnasium) in the active conda env -- this script does not install anything;
# run multi-type-feedback/setup.sh (pip install -e .) first if imports fail.
#
# Usage (CPU):
#   ENV_NAME=mw_drawer-open-v2 SEEDS="0 1 2" \
#     sbatch --array=0-2 \
#            --account=dbrown --partition=coe-class-grn --qos=coe-students-grn \
#            slurm/mw_de_train_demo_bc.sbatch
#
# SEEDS and --array must agree: --array=0-(N_SEEDS-1) (0-indexed position into
# SEEDS, not the seed value itself).
#
# Optional env overrides:
#   ENV_NAME       default: mw_drawer-open-v2   (this repo's mw_<name>-v2 form)
#   SEEDS          default: "0 1 2"
#   ALGORITHM      default: research_demo   (must match what you passed to
#                                            convert_research_pool_to_demos.py's
#                                            --out when building the pkl -- it's
#                                            embedded in the pkl filename AND the
#                                            saved policy dir name)
#   N_EPOCHS       default: 20
#   BATCH_SIZE     default: 32
#   MTF_DIR        default: multi-type-feedback  (relative to repo root)

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00
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
ALGORITHM="${ALGORITHM:-research_demo}"
N_EPOCHS="${N_EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-32}"
MTF_DIR="${MTF_DIR:-multi-type-feedback}"

read -r -a SEED_ARR <<< "$SEEDS"
SEED="${SEED_ARR[$SLURM_ARRAY_TASK_ID]:-}"
if [[ -z "$SEED" ]]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} out of range for SEEDS=(${SEEDS})"
    exit 1
fi

MW_NAME="${ENV_NAME#mw_}"
MTF_ENV="metaworld-${MW_NAME}"

[[ "$MTF_DIR" != /* ]] && MTF_DIR="${REPO_ROOT}/${MTF_DIR}"

DEMO_PKL="${MTF_DIR}/feedback_regen/${ALGORITHM}_${MTF_ENV}_${SEED}.pkl"
MODEL_ID="${ALGORITHM}_${MTF_ENV}_${SEED}_default_${SEED}"

echo "=============================="
echo "SLURM_JOB_ID  = ${SLURM_JOB_ID:-}"
echo "ARRAY_TASK_ID = ${SLURM_ARRAY_TASK_ID}"
echo "ENV_NAME      = ${ENV_NAME}"
echo "MTF_ENV       = ${MTF_ENV}"
echo "SEED          = ${SEED}"
echo "DEMO_PKL      = ${DEMO_PKL}"
echo "SAVE_PATH     = ${MTF_DIR}/agents/BC_${MODEL_ID}"
echo "=============================="

if [[ ! -f "$DEMO_PKL" ]]; then
    echo "ERROR: demo pickle not found: ${DEMO_PKL}"
    echo "Run scripts/extract_demo_track_for_bc.py + convert_research_pool_to_demos.py first"
    echo "(--out must match this exact path -- ALGORITHM=${ALGORITHM}, seed=${SEED})."
    exit 1
fi

. "${REPO_ROOT}/setup_shell.sh"
cd "$MTF_DIR"

python3 multi_type_feedback/train_bc.py \
    --algorithm "$ALGORITHM" \
    --environment "$MTF_ENV" \
    --seed "$SEED" \
    --n-epochs "$N_EPOCHS" \
    --batch-size "$BATCH_SIZE"

echo "Done — BC policy saved to ${MTF_DIR}/agents/BC_${MODEL_ID}/bc_policy.zip"
