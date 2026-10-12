#!/bin/bash
# Submits the mixed-feedback (90% base / 10% secondary) CPL and PIQL runs on
# rai-gpu-grn through the generic slurm/train_array.sbatch. Both families run
# 500k steps (configs/mw_state_dense/mixed_cpl_500k.yaml, mixed_piql.yaml).
#
#   cpl  : all 6 combos on drawer-open x 3 seeds -> 18 tasks of
#          slurm/manifests/mixed_cpl_all
#   piql : 6 combos x 4 envs x 3 seeds -> all 72 tasks of
#          slurm/manifests/mixed_piql_all
#
# Usage (from the repo root on CHPC):
#   bash slurm/submit_mixed_b10k.sh [cpl|piql|all]      (default: all)
#
# GPU only: credit-containing batches are ~5 s/step on CPU (~29 days for 500k).
#
# Walltime: 8h for CPL, 16h for PIQL. Expected on rai-gpu-grn (H200), 500k
# steps: CPL ~1-5h; PIQL ~2-5h (scalar mixes) / ~8-10h (credit mixes, up to
# ~13h on a busy node).
#
# Resume: a task killed by its walltime resumes from its run dir's
# final_model.pt (saved every eval, 5k steps) -- resubmit just that task, e.g.
#   ARRAY=7 bash slurm/submit_mixed_b10k.sh piql
#
# Optional env overrides:
#   ARRAY             override the --array spec for the chosen family
#   CONDA_ENV_NAME    default: cpl_gpu
#   CPL_WANDB_PROJECT default: mw-cpl-b10k-mixed
#   PIQL_WANDB_PROJECT default: mw-piql-b10k-mixed

set -euo pipefail

WHICH="${1:-all}"
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
CPL_TIME="08:00:00"
PIQL_TIME="16:00:00"
export DEVICE=cuda
export CONDA_ENV_NAME="${CONDA_ENV_NAME:-cpl_gpu}"

# mixed_cpl_all task IDs for drawer-open (3 seeds each): 7-9 credit+demo,
# 19-21 credit+pref, 31-33 credit+corr, 43-45 scalar+demo, 55-57 scalar+pref,
# 67-69 scalar+corr. (credit+demo on the other envs: 1-6, 10-12.)
CPL_ARRAY="7-9,19-21,31-33,43-45,55-57,67-69"
PIQL_ARRAY="1-72"

if [[ "$WHICH" == "cpl" || "$WHICH" == "all" ]]; then
    WANDB_PROJECT="${CPL_WANDB_PROJECT:-mw-cpl-b10k-mixed}" \
        sbatch --job-name=mixed_cpl --array="${ARRAY:-$CPL_ARRAY}" --time="$CPL_TIME" "${COMMON[@]}" \
        slurm/train_array.sbatch slurm/manifests/mixed_cpl_all/manifest.tsv
fi

if [[ "$WHICH" == "piql" || "$WHICH" == "all" ]]; then
    WANDB_PROJECT="${PIQL_WANDB_PROJECT:-mw-piql-b10k-mixed}" \
        sbatch --job-name=mixed_piql --array="${ARRAY:-$PIQL_ARRAY}" --time="$PIQL_TIME" "${COMMON[@]}" \
        slurm/train_array.sbatch slurm/manifests/mixed_piql_all/manifest.tsv
fi
