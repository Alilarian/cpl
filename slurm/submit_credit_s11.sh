#!/bin/bash
# Submits the stride-11 credit assignment CPL and PIQL runs (4 envs x 3 seeds
# each) on soc-gpu-class-grn through slurm/mw_b10k_credit_s11.sbatch.
#
#   cpl  : tasks 1-12
#   piql : tasks 13-24
#
# Usage (from the repo root on CHPC):
#   bash slurm/submit_credit_s11.sh [cpl|piql|all]      (default: all)
#
# Walltime: 4h for CPL, 8h for PIQL. Stride-11 rows are 5 windows instead of
# 45, so a batch is ~9x smaller than the stride-1 credit runs.
#
# Resume: a task killed by its walltime resumes from its run dir's
# final_model.pt -- resubmit just that task, e.g.
#   ARRAY=14 bash slurm/submit_credit_s11.sh piql
#
# Optional env overrides:
#   ARRAY           override the --array spec for the chosen family
#   CONDA_ENV_NAME  default: cpl_gpu
#   (plus any override documented in slurm/mw_b10k_credit_s11.sbatch)

set -euo pipefail

WHICH="${1:-all}"
if [[ ! -f setup_shell.sh ]]; then
    echo "ERROR: run from the repo root (no setup_shell.sh in $PWD)"
    exit 1
fi
mkdir -p slurm/logs

COMMON=(
    --partition=soc-gpu-class-grn
    --qos=soc-gpu-students-grn
    --account=dbrown
    --gres=gpu:1
    --cpus-per-task=8
    --mem=64G
)
export DEVICE=cuda
export CONDA_ENV_NAME="${CONDA_ENV_NAME:-cpl_gpu}"

if [[ "$WHICH" == "cpl" || "$WHICH" == "all" ]]; then
    sbatch --job-name=credit_s11_cpl --array="${ARRAY:-1-12}" --time=04:00:00 "${COMMON[@]}" \
        slurm/mw_b10k_credit_s11.sbatch
fi

if [[ "$WHICH" == "piql" || "$WHICH" == "all" ]]; then
    sbatch --job-name=credit_s11_piql --array="${ARRAY:-13-24}" --time=08:00:00 "${COMMON[@]}" \
        slurm/mw_b10k_credit_s11.sbatch
fi
