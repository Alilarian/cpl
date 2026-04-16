#!/bin/bash
#SBATCH --account=owner-gpu-guest
#SBATCH --partition=owner-guest
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --job-name=ll_sac_%a
#SBATCH --output=logs/ll_sac_%a_%j.out
#SBATCH --error=logs/ll_sac_%a_%j.err

# ---------------------------------------------------------------
# Submit all three seeds as an array job:
#   sbatch --array=0,1,2 scripts/train_ll_sac_chpc.sh
#
# SLURM_ARRAY_TASK_ID (0, 1, 2) is used directly as the seed,
# so each run gets a unique seed and a unique output directory.
# All three runs appear in the same W&B project "ll-sac".
# ---------------------------------------------------------------

SEED=${SLURM_ARRAY_TASK_ID}

# ---------------------------------------------------------------
# Environment setup  (mirrors setup_shell.sh)
# ---------------------------------------------------------------
REPO_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"

if module load miniconda3 2>/dev/null; then
    conda activate cpl
else
    source /uufs/chpc.utah.edu/sys/installdir/r8/miniconda3/25.9.1/miniconda3/bin/activate
    conda activate cpl
fi

cd "$REPO_ROOT"
unset DISPLAY

# MuJoCo libraries
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco200/bin
if [ -d "/usr/lib/nvidia" ]; then
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
fi

# CUDA
if nvidia-smi | grep -q "CUDA Version"; then
    for cuda_ver in 11.8 11.7; do
        if [ -d "/usr/local/cuda-${cuda_ver}" ]; then
            export PATH=/usr/local/cuda-${cuda_ver}/bin:$PATH
            break
        fi
    done
    export MUJOCO_GL="egl"
else
    export MUJOCO_GL="osmesa"
fi

# W&B — all seeds go into the same project; run name = "seed_N"
export WANDB_PROJECT="ll-sac"
export WANDB_API_KEY="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"

# ---------------------------------------------------------------
# Run
# ---------------------------------------------------------------
OUTPUT_DIR="results/ll_sac/seed_${SEED}"
mkdir -p logs "$OUTPUT_DIR"

echo "LunarLander SAC | seed=${SEED} | output=${OUTPUT_DIR} | job=${SLURM_JOB_ID}"

python scripts/train.py \
    --config configs/ll_sac/sac.yaml \
    --path   "$OUTPUT_DIR" \
    --device cuda \
    --seed   "$SEED"
