#!/bin/bash
# RRC + SAC stage A: check every input before any compute (docs/rrc_sac_plan.md section 5).
#   sbatch --partition=rai-gpu-grn --qos=rai-gpu-grn --account=rai --gres=gpu:1 \
#          --cpus-per-task=4 --mem=64G --time=1:00:00 slurm/rrc_preflight.sbatch [types...]
#SBATCH --job-name=rrc_preflight
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
. "${SLURM_SUBMIT_DIR:-$PWD}/slurm/rrc_common.sh"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
# Feedback types to check: positional args (e.g. `slurm/rrc_preflight.sbatch pref corr demo`), else
# $RRC_TYPES, else all six.
TYPES="${*:-${RRC_TYPES:-all}}"
# shellcheck disable=SC2086
python scripts/rrc_preflight.py --tasks all --types $TYPES --runs_root "$RUNS_ROOT" ${OVERLAY_ARGS[@]+"${OVERLAY_ARGS[@]}"}
