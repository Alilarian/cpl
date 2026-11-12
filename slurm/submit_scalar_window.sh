#!/bin/bash
# Submits the window-choice scalar CPL and PIQL runs (4 envs x 3 seeds each)
# through slurm/mw_b10k_scalar_window.sbatch, PACK runs sharing each GPU
# (slurm/pack_gpu.sbatch; default 4 runs per GPU = 3 GPU jobs per family).
#
#   cpl  : tasks 1-12
#   piql : tasks 13-24
#
# Usage (from the repo root on CHPC):
#   bash slurm/submit_scalar_window.sh [cpl|piql|all]      (default: all)
#   PARTITION=granite-gpu bash slurm/submit_scalar_window.sh cpl
#   PARTITION=soc-gpu-class-grn bash slurm/submit_scalar_window.sh all   (Blackwell: needs cpl_gpu_bw,
#       built once with slurm/setup_cpl_gpu_bw.sh)
#
# Upload the labels first (from the machine that generated them):
#   rsync -avPR datasets/mw_de_labels/./mw_*/scalar_window_labels.npz \
#       chpc-notch:/scratch/general/vast/u1472210/scalar_window_labels/
#
# Walltime per GPU job: 6h for CPL, 12h for PIQL. A batch is 32 rows x 3
# subsegments x 10 steps (960 transitions), far smaller than the cum_estop PIQL
# runs (~12k) that took 0.010-0.030 s/step alone on rai-gpu-grn, i.e. ~1.5-4h
# for 500k steps; the margin covers sharing the GPU with 3 other runs.
#
# Logs: slurm/logs/scalar_window_<alg>_<jobid>_<task>.{out,err} per run, and
# ..._g<i>.out / ..._g<i>.gpu.csv per GPU job.
#
# Resume: a run killed by its walltime resumes from its run dir's
# final_model.pt -- resubmit just that task, e.g.
#   ARRAY=19 bash slurm/submit_scalar_window.sh piql
#
# Optional env overrides:
#   ARRAY           tasks to run (default 1-12 / 13-24), e.g. 19 or 13-15,22
#   CPL_TIME        default: 06:00:00
#   PIQL_TIME       default: 12:00:00
#   WANDB_MODE      default: offline  (runs are written to ~/wandb/wandb/offline-run-*;
#                   upload them later with `wandb sync`). WANDB_MODE=online logs live.
#   CONDA_ENV_NAME  default: cpl_gpu (cpl_gpu_bw on soc-gpu-class-grn)
#   PACK, CPUS, MEM, PARTITION, QOS, ACCOUNT  -- see slurm/pack_gpu_submit.sh
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
. slurm/pack_gpu_submit.sh

export DEVICE=cuda
export CONDA_ENV_NAME="${CONDA_ENV_NAME:-cpl_gpu}"
export WANDB_MODE="${WANDB_MODE:-offline}"

if [[ "$WHICH" == "cpl" || "$WHICH" == "all" ]]; then
    submit_packed scalar_window_cpl "${ARRAY:-1-12}" "${CPL_TIME:-06:00:00}" slurm/mw_b10k_scalar_window.sbatch
fi

if [[ "$WHICH" == "piql" || "$WHICH" == "all" ]]; then
    submit_packed scalar_window_piql "${ARRAY:-13-24}" "${PIQL_TIME:-12:00:00}" slurm/mw_b10k_scalar_window.sbatch
fi
