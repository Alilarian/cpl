#!/bin/bash
# Submits the window-choice scalar CPL runs (4 envs x 3 seeds = 12 tasks) on
# rai-gpu-grn through slurm/mw_b10k_scalar_window.sbatch.
#
# Usage (from the repo root on CHPC):
#   bash slurm/submit_scalar_window.sh
#
# Upload the labels first (from the machine that generated them):
#   rsync -avPR datasets/mw_de_labels/./mw_*/scalar_window_labels.npz \
#       chpc-notch:/scratch/general/vast/u1472210/scalar_window_labels/
#
# Walltime: 8h, same as the stride-11 credit CPL runs; rows here are 3
# subsegments of 10 steps, smaller than credit's 5 windows of 20.
#
# Resume: a task killed by its walltime resumes from its run dir's
# final_model.pt -- resubmit just that task, e.g.
#   ARRAY=7 bash slurm/submit_scalar_window.sh
#
# Optional env overrides:
#   ARRAY           override the --array spec (default: 1-12)
#   CONDA_ENV_NAME  default: cpl_gpu
#   (plus any override documented in slurm/mw_b10k_scalar_window.sbatch)

set -euo pipefail

if [[ ! -f setup_shell.sh ]]; then
    echo "ERROR: run from the repo root (no setup_shell.sh in $PWD)"
    exit 1
fi
mkdir -p slurm/logs

export DEVICE=cuda
export CONDA_ENV_NAME="${CONDA_ENV_NAME:-cpl_gpu}"

sbatch --job-name=scalar_window_cpl --array="${ARRAY:-1-12}" --time=08:00:00 \
    --partition=rai-gpu-grn \
    --qos=rai-gpu-grn \
    --account=rai \
    --gres=gpu:1 \
    --cpus-per-task=8 \
    --mem=64G \
    slurm/mw_b10k_scalar_window.sbatch
