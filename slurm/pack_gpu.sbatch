#!/bin/bash
# Runs several tasks of an array-style launcher at once on one GPU.
#
# Slurm hands a --gres=gpu:1 job a whole GPU, and these partitions offer no GPU
# sharing between jobs (no shard/mps gres). The b10k runs use a small part of
# a GPU (a few 512x512 MLPs, ~1k-3k transitions per step), so one job runs
# several of them side by side instead.
#
# Submitted by slurm/pack_gpu_submit.sh::submit_packed, which sets:
#   INNER        launcher to run, e.g. slurm/mw_b10k_scalar_window.sbatch; it
#                reads SLURM_ARRAY_TASK_ID like a regular array task
#   TASK_GROUPS  space-separated groups of comma-separated task IDs; array
#                index i runs group i, e.g. "1,2,3,4 5,6,7,8 9,10,11,12"
#
# Each task logs to slurm/logs/<job-name>_<array-job-id>_<task>.{out,err},
# the same names as one task per job. This job's own log is ..._g<index>.out,
# and GPU utilization/memory is sampled every minute into ..._g<index>.gpu.csv.
# The job fails if any of its tasks failed; the others keep running.
# OMP/MKL threads are capped so the tasks share the job's cores evenly.

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/uufs/chpc.utah.edu/common/home/u1472210/cpl/slurm/logs/%x_%A_g%a.out
#SBATCH --error=/uufs/chpc.utah.edu/common/home/u1472210/cpl/slurm/logs/%x_%A_g%a.err

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
LOG_DIR="${SUBMIT_DIR}/slurm/logs"
mkdir -p "$LOG_DIR"

if [[ -z "${INNER:-}" || -z "${TASK_GROUPS:-}" || -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "ERROR: submit through slurm/pack_gpu_submit.sh (needs INNER, TASK_GROUPS and --array)"
    exit 1
fi

read -r -a PACKED <<< "$TASK_GROUPS"
GROUP="${PACKED[$(( SLURM_ARRAY_TASK_ID - 1 ))]}"
IFS=, read -r -a TASKS <<< "$GROUP"
N=${#TASKS[@]}

THREADS=$(( ${SLURM_CPUS_PER_TASK:-$N} / N ))
if (( THREADS < 1 )); then
    THREADS=1
fi
export OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS

PREFIX="${LOG_DIR}/${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID}"
echo "=============================="
echo "INNER       = ${INNER}"
echo "TASKS       = ${TASKS[*]}  (group ${SLURM_ARRAY_TASK_ID} of ${#PACKED[@]})"
echo "NODE        = $(hostname)"
echo "CPUS        = ${SLURM_CPUS_PER_TASK:-?}  (${THREADS} thread(s) per task)"
echo "GPU         = $(nvidia-smi -L 2>/dev/null | head -1 || echo none)"
echo "=============================="

# Fail fast with one clear message if this env's PyTorch has no kernels for this
# GPU (cpl_gpu's torch 2.5 on Blackwell), instead of N identical tracebacks.
if [[ "${DEVICE:-cuda}" == cuda ]] && ! ( set +eu; REPO_ROOT="$SUBMIT_DIR"
        . "${SUBMIT_DIR}/setup_shell.sh" >/dev/null 2>&1 </dev/null
        python -c "import torch; print(torch.nn.functional.dropout(torch.ones(8, device='cuda'), 0.5).sum().item())" >/dev/null ); then
    echo "ERROR: PyTorch in conda env '${CONDA_ENV_NAME:-?}' can't run on this GPU (see the .err log)."
    echo "       Blackwell GPUs (soc-gpu-class-grn) need CONDA_ENV_NAME=cpl_gpu_bw -- slurm/setup_cpl_gpu_bw.sh"
    exit 1
fi

nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total --format=csv -l 60 \
    > "${PREFIX}_g${SLURM_ARRAY_TASK_ID}.gpu.csv" 2>/dev/null &
SMI_PID=$!

PIDS=()
for t in "${TASKS[@]}"; do
    SLURM_ARRAY_TASK_ID="$t" bash "${SUBMIT_DIR}/${INNER}" > "${PREFIX}_${t}.out" 2> "${PREFIX}_${t}.err" &
    PIDS+=($!)
    echo "task ${t}: started (pid ${PIDS[-1]}), log ${PREFIX}_${t}.out"
    sleep 5  # stagger conda activation and env/mujoco startup
done

FAILED=()
for i in "${!PIDS[@]}"; do
    if wait "${PIDS[$i]}"; then
        echo "task ${TASKS[$i]}: OK"
    else
        echo "task ${TASKS[$i]}: FAILED (exit $?) -- see ${PREFIX}_${TASKS[$i]}.err"
        FAILED+=("${TASKS[$i]}")
    fi
done
kill "$SMI_PID" 2>/dev/null || true

if (( ${#FAILED[@]} )); then
    echo "Failed tasks: ${FAILED[*]}"
    exit 1
fi
echo "All ${N} tasks finished."
