#!/bin/bash
# RRC + SAC stage C1a: cache off-distribution rollouts, one array element per task (manifest line = task).
#   python scripts/make_rrc_manifest.py --stage offdist --tasks all --out slurm/manifests/rrc_sac/offdist.tsv
#   sbatch --partition=rai-gpu-grn --qos=rai-gpu-grn --account=rai --gres=gpu:1 --array=1-4 \
#          --cpus-per-task=4 --mem=32G --time=2:00:00 slurm/rrc_offdist.sbatch slurm/manifests/rrc_sac/offdist.tsv
#SBATCH --job-name=rrc_offdist
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=slurm/logs/%x_%A_%a.out
#SBATCH --error=slurm/logs/%x_%A_%a.err
. "${SLURM_SUBMIT_DIR:-$PWD}/slurm/rrc_common.sh"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
TASK="$(manifest_line "$1")"
python scripts/collect_rrc_offdist.py --task "$TASK" --runs_root "$RUNS_ROOT" --device cpu ${OVERLAY_ARGS[@]+"${OVERLAY_ARGS[@]}"}
