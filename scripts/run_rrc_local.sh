#!/bin/bash
# Run the whole RRC + SAC experiment on this machine (no SLURM): pref / corr / demo x 4 envs x 3 seeds,
# reward ensembles first, then SAC (oracle config, 2M steps) with $PARALLEL runs at a time.
# Uses the same manifests as the CHPC jobs (slurm/manifests/rrc_sac/) and the local data mirrors
# (configs/rrc_sac/local.yaml). Everything logs to WandB project $WANDB_PROJECT.
#
#   mkdir -p runs/rrc_sac/_logs
#   nohup setsid bash scripts/run_rrc_local.sh > runs/rrc_sac/_logs/runner.log 2>&1 &
#
# Re-running is safe: finished rollouts, reward seeds and SAC runs are skipped, unfinished SAC runs
# resume from their final_model.pt. Progress: tail -f runs/rrc_sac/_logs/runner.log
set -uo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-/home/alilarian/enter/envs/mujoco_cap/bin/python}"
export MUJOCO_PY_MUJOCO_PATH="$HOME/.mujoco/mujoco210"
export LD_LIBRARY_PATH="$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export WANDB_ENTITY="${WANDB_ENTITY:-a7a7}" WANDB_PROJECT="${WANDB_PROJECT:-rrc-sac-2m}"
export RUNS_ROOT="${RUNS_ROOT:-runs/rrc_sac}" TOTAL_STEPS="${TOTAL_STEPS:-2000000}" PY
PARALLEL="${PARALLEL:-6}"
OVERLAY=configs/rrc_sac/local.yaml
M=slurm/manifests/rrc_sac
OFFDIST_MANIFEST="${OFFDIST_MANIFEST:-$M/offdist.tsv}"
REWARD_MANIFEST="${REWARD_MANIFEST:-$M/reward_pref_corr_demo.tsv}"
SAC_MANIFEST="${SAC_MANIFEST:-$M/sac_pref_corr_demo.tsv}"
REWARD_SET="${REWARD_SET:-}"   # optional single "key=value" override for reward training (smoke tests)
export LOGS="$RUNS_ROOT/_logs" RESULTS="$RUNS_ROOT/_results"
mkdir -p "$LOGS" "$RESULTS" "$RUNS_ROOT/_wandb"
stage() { echo "[$(date '+%F %T')] $*"; }
stage "RRC + SAC local run: PARALLEL=$PARALLEL TOTAL_STEPS=$TOTAL_STEPS RUNS_ROOT=$RUNS_ROOT WANDB=$WANDB_ENTITY/$WANDB_PROJECT"

stage "1/4 off-distribution rollouts"
while read -r task; do
    [[ -z "$task" ]] && continue
    "$PY" -W ignore scripts/collect_rrc_offdist.py --task "$task" --overlay $OVERLAY --runs_root "$RUNS_ROOT" \
        < /dev/null > "$LOGS/offdist_${task}.log" 2>&1 || { stage "FAILED offdist $task (see $LOGS/offdist_${task}.log)"; exit 1; }
    stage "  offdist $task done"
done < "$OFFDIST_MANIFEST"

stage "2/4 reward models"
while IFS=$'\t' read -r task config seeds flags; do
    [[ -z "$task" ]] && continue
    t=$(basename "$config" .yaml)
    # shellcheck disable=SC2086
    WANDB_DIR="$RUNS_ROOT/_wandb" "$PY" -W ignore scripts/train_rrc_reward.py --config "$config" --task "$task" --seeds $seeds \
        --overlay $OVERLAY --runs_root "$RUNS_ROOT" --device cuda ${flags:-} ${REWARD_SET:+--set "$REWARD_SET"} \
        < /dev/null > "$LOGS/reward_${task}_${t}.log" 2>&1 || { stage "FAILED reward $task $t (see $LOGS/reward_${task}_${t}.log)"; exit 1; }
    stage "  reward $task $t done"
done < "$REWARD_MANIFEST"
"$PY" scripts/rrc_reward_table.py --runs_root "$RUNS_ROOT" --out "$RESULTS/rrc_reward_diagnostics.md" > /dev/null
"$PY" scripts/rrc_wandb_sync.py --stage reward --runs_root "$RUNS_ROOT" --results_dir "$RESULTS" > "$LOGS/wandb_sync_reward.log" 2>&1 \
    || stage "  (WandB reward summary sync failed, see $LOGS/wandb_sync_reward.log)"

run_sac() {  # one "task:arm:seed" spec; after it finishes, compute its hacking metrics and push them to WandB
    local task arm seed run
    IFS=':' read -r task arm seed <<< "$1"
    run="$RUNS_ROOT/$task/${arm%@*}/seed$seed/sac_${arm#*@}"
    echo "[$(date '+%F %T')]   start $task $arm s$seed"
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 WANDB_RUN_NAME="rrc_sac/$task/$arm/s$seed" WANDB_GROUP="rrc_sac/$task/$arm" \
        "$PY" -W ignore scripts/train_rrc_sac.py --task "$task" --arm "$arm" --seed "$seed" --runs_root "$RUNS_ROOT" \
        --overlay configs/rrc_sac/local.yaml --device cuda --total_steps "$TOTAL_STEPS" \
        < /dev/null > "$LOGS/sac_${task}_${arm//@/_}_s${seed}.log" 2>&1
    local rc=$?
    if [[ $rc -ne 0 ]]; then echo "[$(date '+%F %T')]   FAILED $task $arm s$seed (exit $rc)"; return 0; fi
    "$PY" -W ignore scripts/rrc_hacking.py --runs_root "$RUNS_ROOT" --overlay configs/rrc_sac/local.yaml --run_dirs "$run" \
        >> "$LOGS/hacking.log" 2>&1
    "$PY" scripts/rrc_wandb_sync.py --stage sac_run --run_dir "$run" >> "$LOGS/wandb_sync_sac.log" 2>&1
    echo "[$(date '+%F %T')]   done $task $arm s$seed: $(grep -F "$task $arm s$seed " "$LOGS/hacking.log" | tail -1 | sed 's/  */ /g')"
}
export -f run_sac

stage "3/4 SAC: $(tr ' ' '\n' < "$SAC_MANIFEST" | grep -c .) runs, $PARALLEL at a time"
tr ' ' '\n' < "$SAC_MANIFEST" | grep . | xargs -P "$PARALLEL" -I{} bash -c 'run_sac "$1"' _ {}

stage "4/4 summaries"
"$PY" -W ignore scripts/rrc_hacking.py --runs_root "$RUNS_ROOT" --overlay $OVERLAY > "$LOGS/hacking_all.log" 2>&1
"$PY" scripts/rrc_reward_table.py --runs_root "$RUNS_ROOT" --out "$RESULTS/rrc_reward_diagnostics.md" > /dev/null
"$PY" -W ignore scripts/aggregate_rrc_sac.py --runs_root "$RUNS_ROOT" --out "$RESULTS/rrc_sac_results.md" \
    --oracle_dir "datasets/mw_de_labels/{task}/oracle" > /dev/null
"$PY" scripts/rrc_wandb_sync.py --stage final --runs_root "$RUNS_ROOT" --results_dir "$RESULTS" > "$LOGS/wandb_sync_final.log" 2>&1
n_done=$(ls "$RUNS_ROOT"/*/*/seed*/sac_*/training_complete 2>/dev/null | wc -l)
stage "finished: $n_done / $(tr ' ' '\n' < "$SAC_MANIFEST" | grep -c .) SAC runs complete. Results: $RESULTS/rrc_sac_results.md, $RESULTS/rrc_reward_diagnostics.md"
