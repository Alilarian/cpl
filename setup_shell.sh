# Make sure we have the conda environment set up.
CONDA_PATH=/uufs/chpc.utah.edu/sys/installdir/r8/miniconda3/25.9.1/miniconda3/bin/activate
CONDA_ENV_NAME="${CONDA_ENV_NAME:-cpl}"
REPO_PATH="${REPO_ROOT:-path/to/your/repo}"
USE_MUJOCO_PY=true # For using mujoco py
WANDB_API_KEY="${WANDB_API_KEY:-}" # Set via `export WANDB_API_KEY=...` in your shell profile before sourcing this script, or run `wandb login` instead and leave this unset.
WANDB_ENTITY="${WANDB_ENTITY:-a7a7}" # WandB entity to write runs to. Must be passed explicitly (see scripts/train.py) -- wandb.init()'s implicit default-entity resolution does not reliably match this account's actual default entity on CHPC. Override via `export WANDB_ENTITY=...`.

# Setup Conda
# In non-interactive SLURM jobs, module load miniconda3 adds conda to PATH
# but does NOT initialize the shell functions that conda activate requires.
# We must always source conda.sh explicitly before calling conda activate.
if module load miniconda3 2>/dev/null; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate $CONDA_ENV_NAME
else
    source $CONDA_PATH
    conda activate $CONDA_ENV_NAME
fi
cd $REPO_PATH
unset DISPLAY # Make sure display is not set or it will prevent scripts from running in headless mode.

# Print which env actually got activated -- CONDA_ENV_NAME is easy to omit or
# drop when copy-pasting a submit command, and the only other symptom is a
# traceback deep inside training (e.g. "Torch not compiled with CUDA enabled"
# if cpl_gpu was intended but the plain cpl env got used instead). Surfacing
# it here makes that immediately visible in every job's .out log.
echo "Active conda env: ${CONDA_ENV_NAME} ($(python -c 'import sys; print(sys.executable)' 2>/dev/null))"

# Install gymnasium if not present — required for LunarLander.
# gym 0.23's box2d is incompatible with numpy 2.x; gymnasium's box2d fixes this.
if ! python -c "import gymnasium" 2>/dev/null; then
    echo "Installing gymnasium[box2d] and pygame..."
    pip install -q "gymnasium[box2d]" pygame
fi


if [[ -n "$WANDB_API_KEY" ]]; then
    export WANDB_API_KEY="$WANDB_API_KEY"
fi
export WANDB_ENTITY="$WANDB_ENTITY"

if $USE_MUJOCO_PY; then
    echo "Using mujoco_py"
    if [ -d "/usr/lib/nvidia" ]; then
        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
    fi
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco200/bin
fi

# First check if we have a GPU available
if nvidia-smi | grep "CUDA Version"; then
    if [ -d "/usr/local/cuda-11.8" ]; then # This is the only GPU version supported by compile.
        export PATH=/usr/local/cuda-11.8/bin:$PATH
    elif [ -d "/usr/local/cuda-11.7" ]; then # This is the only GPU version supported by compile.
        export PATH=/usr/local/cuda-11.7/bin:$PATH
    elif [ -d "/usr/local/cuda" ]; then
        export PATH=/usr/local/cuda/bin:$PATH
        echo "Using default CUDA. Compatibility should be verified. torch.compile requires >= 11.7"
    else
        echo "Warning: Could not find a CUDA version but GPU was found."
    fi
    export MUJOCO_GL="egl"
    # Setup any GPU specific flags
else
    echo "GPU was not found, assuming CPU setup."
    export MUJOCO_GL="osmesa" # glfw doesn't support headless rendering
fi
