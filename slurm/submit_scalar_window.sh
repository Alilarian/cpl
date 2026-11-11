#!/bin/bash
# Submits the window-choice scalar CPL and PIQL runs (4 envs x 3 seeds each)
# on rai-gpu-grn through slurm/mw_b10k_scalar_window.sbatch.
#
#   cpl  : tasks 1-12
#   piql : tasks 13-24
#
# Usage (from the repo root on CHPC):
#   bash slurm/submit_scalar_window.sh [cpl|piql|all]      (default: all)
#
# Upload the labels first (from the machine that generated them):
#   rsync -avPR datasets/mw_de_labels/./mw_*/scalar_window_labels.npz \
#       chpc-notch:/scratch/general/vast/u1472210/scalar_window_labels/
#
# Walltime: 4h for CPL, 8h for PIQL. A batch is 32 rows x 3 subsegments x 10
# steps (960 transitions), far smaller than the cum_estop PIQL runs (~12k) that
# took 0.010-0.030 s/step on rai-gpu-grn, i.e. ~1.5-4h for 500k steps.
#
# Resume: a task killed by its walltime resumes from its run dir's
# final_model.pt -- resubmit just that task, e.g.
#   ARRAY=19 bash slurm/submit_scalar_window.sh piql
#
# Optional env overrides:
#   ARRAY           override the --array spec for the chosen family
#   CONDA_ENV_NAME  default: cpl_gpu
#   (plus any override documented in slurm/mw_b10k_scalar_window.sbatch)

set -euo pipefail

WHICH="${1:-all}"
if [[ "$WHICH" != "cpl" && "$WHICH" != "piql" && "$WHICH" != "all" ]]; then
    echo "Usage: bash slurm/submit_scalar_window.sh [cpl|piql|all]"
    exit 1
fi
if [[ ! -f setup_shell.sh ]]; then
    echo "ERROR: run from the repo root (no setup_shell.sh in $PWD)"
    exit 1
fi
mkdir -p slurm/logs

COMMON=(
    --partition=rai-gpu-grn
    --qos=rai-gpu-grn
    --account=rai
    --gres=gpu:1
    --cpus-per-task=8
    --mem=64G
)
export DEVICE=cuda
export CONDA_ENV_NAME="${CONDA_ENV_NAME:-cpl_gpu}"

if [[ "$WHICH" == "cpl" || "$WHICH" == "all" ]]; then
    sbatch --job-name=scalar_window_cpl --array="${ARRAY:-1-12}" --time=04:00:00 "${COMMON[@]}" \
        slurm/mw_b10k_scalar_window.sbatch
fi

if [[ "$WHICH" == "piql" || "$WHICH" == "all" ]]; then
    sbatch --job-name=scalar_window_piql --array="${ARRAY:-13-24}" --time=08:00:00 "${COMMON[@]}" \
        slurm/mw_b10k_scalar_window.sbatch
fi
