#!/bin/bash
# Builds the cpl_gpu_bw conda env: a clone of cpl_gpu with a PyTorch that has
# kernels for NVIDIA Blackwell GPUs (sm_120: soc-gpu-class-grn's RTX PRO 6000 /
# RTX PRO 4000). cpl_gpu's PyTorch 2.5 stops at sm_90 and fails there with
# "CUDA error: no kernel image is available for execution on the device".
#
# torch 2.7.1 + torchvision 0.22.1 (CUDA 12.8 wheels) is the first release with
# Blackwell kernels, and it still runs on the H200 / H100 / L40S partitions.
# torchvision is upgraded with it because `import research` loads it
# (research/processors/image_augmentation.py); torchaudio/torchtext are unused
# and pinned to the old torch, so they are removed from the clone.
#
# Run once on a CHPC login node (pip needs internet), from the repo root:
#   bash slurm/setup_cpl_gpu_bw.sh
# Then test it on a Blackwell GPU:
#   srun --partition=soc-gpu-class-grn --qos=soc-gpu-students-grn --account=dbrown \
#       --gres=gpu:1 --time=00:10:00 --mem=8G bash slurm/setup_cpl_gpu_bw.sh --check
#
# Optional env overrides:
#   BW_ENV               default: cpl_gpu_bw
#   BASE_ENV             default: cpl_gpu
#   TORCH_VERSION        default: 2.7.1
#   TORCHVISION_VERSION  default: 0.22.1

set -euo pipefail

BW_ENV="${BW_ENV:-cpl_gpu_bw}"
BASE_ENV="${BASE_ENV:-cpl_gpu}"
TORCH_VERSION="${TORCH_VERSION:-2.7.1}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.22.1}"

if [[ ! -f setup_shell.sh ]]; then
    echo "ERROR: run from the repo root (no setup_shell.sh in $PWD)"
    exit 1
fi
module load miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"

check() {
    python - <<'EOF'
import torch, torchvision
print("torch", torch.__version__, "| torchvision", torchvision.__version__, "| CUDA", torch.version.cuda)
print("arch list:", torch.cuda.get_arch_list())
if torch.cuda.is_available():
    major, minor = torch.cuda.get_device_capability()
    x = torch.randn(256, 256, device="cuda", requires_grad=True)
    torch.nn.functional.dropout(x @ x, 0.25).sum().backward()
    torch.cuda.synchronize()
    print(f"GPU {torch.cuda.get_device_name()} (sm_{major}{minor}): matmul + dropout + backward OK")
else:
    print("no GPU visible -- run with --check through srun on a GPU node to test one")
import research  # loads torchvision through research.processors
print("import research OK")
EOF
}

if [[ "${1:-}" == "--check" ]]; then
    conda activate "$BW_ENV"
    check
    exit 0
fi

if conda env list | awk '{print $1}' | grep -qx "$BW_ENV"; then
    echo "${BW_ENV} already exists -- updating its torch in place"
else
    conda create -y -n "$BW_ENV" --clone "$BASE_ENV"
fi
conda activate "$BW_ENV"

# A conda-installed torch can't be replaced by pip, so drop it through conda first.
INSTALLER=$(python -c "import importlib.metadata as m; print((m.distribution('torch').read_text('INSTALLER') or '').strip())" 2>/dev/null || true)
if [[ "$INSTALLER" == conda ]]; then
    for pkg in pytorch torchvision torchaudio torchtext pytorch-cuda; do
        conda remove -y --force "$pkg" >/dev/null 2>&1 || true
    done
fi
pip uninstall -y torchaudio torchtext >/dev/null 2>&1 || true
pip install --upgrade "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" \
    --index-url https://download.pytorch.org/whl/cu128

check
echo "Done. Submit Blackwell runs with CONDA_ENV_NAME=${BW_ENV} (the default on soc-gpu-class-grn)."
