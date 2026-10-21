#!/bin/bash
# RRC + SAC stage C2 / sweep: SAC with the exact oracle config on a learned / true / untrained reward.
# Manifest line: up to --pack runs "task:arm:seed" (scripts/make_rrc_manifest.py --stage sac); they run in
# parallel on one GPU, each with its own AsyncEnv subprocess.
#   sbatch --partition=rai-gpu-grn --qos=rai-gpu-grn --account=rai --gres=gpu:1 --array=1-N \
#          --cpus-per-task=16 --mem=64G --time=24:00:00 slurm/rrc_sac.sbatch slurm/manifests/rrc_sac/pilot.tsv
# Resubmitting resumes each run from its final_model.pt; finished runs (training_complete) are skipped.
#   TOTAL_STEPS    default: 2000000  (SAC env steps per run; every other setting is the oracle config's)
#   WANDB_PROJECT  default: rrc-sac-2m (entity: WANDB_ENTITY from setup_shell.sh, default a7a7)
#SBATCH --job-name=rrc_sac
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=slurm/logs/%x_%A_%a.out
#SBATCH --error=slurm/logs/%x_%A_%a.err
. "${SLURM_SUBMIT_DIR:-$PWD}/slurm/rrc_common.sh"
read -r -a RUNS <<< "$(manifest_line "$1")"
N=${#RUNS[@]}
THREADS=$(( ${SLURM_CPUS_PER_TASK:-16} / (2 * N) ))   # main process + AsyncEnv worker per run
(( THREADS < 1 )) && THREADS=1
export OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS
export WANDB_PROJECT="${WANDB_PROJECT:-rrc-sac-2m}"   # needs a WandB login on CHPC (wandb login / WANDB_API_KEY)
TOTAL_STEPS="${TOTAL_STEPS:-2000000}"
echo "TOTAL_STEPS=${TOTAL_STEPS} WANDB_PROJECT=${WANDB_PROJECT} WANDB_ENTITY=${WANDB_ENTITY:-}"
LOGDIR="${REPO_ROOT}/slurm/logs/rrc_sac_${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$LOGDIR"
PIDS=()
for spec in "${RUNS[@]}"; do
    IFS=':' read -r TASK ARM SEED <<< "$spec"
    LOG="${LOGDIR}/${TASK}_${ARM//@/_}_s${SEED}.log"
    echo "launch ${TASK} ${ARM} seed ${SEED} -> ${LOG}"
    WANDB_RUN_NAME="rrc_sac/${TASK}/${ARM}/s${SEED}" WANDB_GROUP="rrc_sac/${TASK}/${ARM}" \
        python scripts/train_rrc_sac.py --task "$TASK" --arm "$ARM" --seed "$SEED" --runs_root "$RUNS_ROOT" --total_steps "$TOTAL_STEPS" \
        --device "$DEVICE" ${OVERLAY_ARGS[@]+"${OVERLAY_ARGS[@]}"} > "$LOG" 2>&1 &
    PIDS+=($!)
done
FAIL=0
for i in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$i]}"; then echo "FAILED: ${RUNS[$i]} (see ${LOGDIR})"; FAIL=1; fi
done
exit $FAIL
