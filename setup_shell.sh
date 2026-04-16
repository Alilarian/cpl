# Make sure we have the conda environment set up.
CONDA_PATH=/uufs/chpc.utah.edu/sys/installdir/r8/miniconda3/25.9.1/miniconda3/bin/activate
ENV_NAME=cpl
REPO_PATH="${REPO_ROOT:-path/to/your/repo}"
USE_MUJOCO_PY=true # For using mujoco py
WANDB_API_KEY="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" # If you want to use wandb, set this to your API key.

# Setup Conda
# In non-interactive SLURM jobs, module load miniconda3 adds conda to PATH
# but does NOT initialize the shell functions that conda activate requires.
# We must always source conda.sh explicitly before calling conda activate.
if module load miniconda3 2>/dev/null; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate $ENV_NAME
else
    source $CONDA_PATH
    conda activate $ENV_NAME
fi
cd $REPO_PATH
unset DISPLAY # Make sure display is not set or it will prevent scripts from running in headless mode.

# Install gymnasium if not present — required for LunarLander.
# gym 0.23's box2d is incompatible with numpy 2.x; gymnasium's box2d fixes this.
if ! python -c "import gymnasium" 2>/dev/null; then
    echo "Installing gymnasium[box2d] and pygame..."
    pip install -q "gymnasium[box2d]" pygame
fi


if [[ -n "$WANDB_API_KEY" ]]; then
    export WANDB_API_KEY="$WANDB_API_KEY"
fi

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
