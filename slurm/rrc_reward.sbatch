#!/bin/bash
# RRC + SAC stage C1b: train reward ensembles + reports, one array element per (task, type).
# Manifest line: <task>\t<config>\t<seeds>\t<flags>   (scripts/make_rrc_manifest.py --stage reward)
#   sbatch --partition=rai-gpu-grn --qos=rai-gpu-grn --account=rai --gres=gpu:1 --array=1-N \
#          --cpus-per-task=8 --mem=64G --time=4:00:00 slurm/rrc_reward.sbatch slurm/manifests/rrc_sac/reward_c1.tsv
# Resubmitting is safe: seeds with a `complete` sentinel are skipped.
#SBATCH --job-name=rrc_reward
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=slurm/logs/%x_%A_%a.out
#SBATCH --error=slurm/logs/%x_%A_%a.err
. "${SLURM_SUBMIT_DIR:-$PWD}/slurm/rrc_common.sh"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
# WandB: one run per (task, type, seed), name rrc_reward/<task>/<type>/s<seed>, same project as the SAC runs.
export WANDB_PROJECT="${WANDB_PROJECT:-rrc-sac-2m}"
export WANDB_DIR="${WANDB_DIR:-${RUNS_ROOT}/_wandb}"
mkdir -p "$WANDB_DIR"
IFS=$'\t' read -r TASK CONFIG SEEDS FLAGS <<< "$(manifest_line "$1")"
echo "TASK=${TASK} CONFIG=${CONFIG} SEEDS=${SEEDS} FLAGS=${FLAGS}"
# shellcheck disable=SC2086  # SEEDS / FLAGS are intentionally word-split
python scripts/train_rrc_reward.py --config "$CONFIG" --task "$TASK" --seeds $SEEDS --runs_root "$RUNS_ROOT" \
    --device "$DEVICE" ${OVERLAY_ARGS[@]+"${OVERLAY_ARGS[@]}"} ${FLAGS:-}
