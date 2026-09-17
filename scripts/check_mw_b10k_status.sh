#!/bin/bash
# check_mw_b10k_status.sh — status audit for every CPL/PIQL mw_b10k experiment.
#
# For each (env, feedback_type, alg, seed) combo, reports one of:
#   COMPLETE        training_complete sentinel exists (all total_steps reached)
#   RUNNING         a Slurm task is actively R and its log was touched recently
#   STALE_RUNNING   Slurm shows R but the log hasn't moved in --stale-minutes (possibly hung)
#   PENDING         a Slurm task is queued (PD) for this run
#   NEEDS_RESUME    training stopped short of total_steps and nothing is queued for it
#                   (walltime hit, crashed, preempted, or manually cancelled -- resubmitting
#                   the same array/task will resume from final_model.pt / wandb_run_id.txt)
#   NO_LOG_YET      run directory exists but no log.csv was ever written (crashed at
#                   startup, e.g. before the first training step) and nothing is queued
#   MISSING         run directory doesn't exist at all -- never submitted, or lost
#
# Also reports, where available: current step / % of total_steps, training speed
# (steps/hour) and a rough ETA to completion, the last logged eval/success, and for
# NEEDS_RESUME/NO_LOG_YET a one-line tail of the matching .err file as a crash hint.
#
# Usage (run from the repo root on CHPC, wherever slurm/logs/ and runs/mw_b10k/ live):
#   bash scripts/check_mw_b10k_status.sh
#   bash scripts/check_mw_b10k_status.sh --only-issues
#   bash scripts/check_mw_b10k_status.sh --env mw_plate-slide-v2 --alg piql
#   bash scripts/check_mw_b10k_status.sh --no-color
#
# Env overrides: REPO_ROOT, RUNS_DIR, LOGS_DIR, TOTAL_STEPS, STALE_MINUTES

set -uo pipefail

REPO_ROOT="${REPO_ROOT:-$PWD}"
RUNS_DIR="${RUNS_DIR:-$REPO_ROOT/runs/mw_b10k}"
LOGS_DIR="${LOGS_DIR:-$REPO_ROOT/slurm/logs}"
TOTAL_STEPS="${TOTAL_STEPS:-500000}"
STALE_MINUTES="${STALE_MINUTES:-60}"

ENVS=(mw_button-press-v2 mw_door-open-v2 mw_drawer-open-v2 mw_plate-slide-v2)
TYPES=(pref corr seq_estop scalar credit_assignment)
SEEDS=(0 1 2)
ALGS=(cpl piql)

ONLY_ISSUES=0
FILTER_ENV=""
FILTER_ALG=""
USE_COLOR=1
[[ -t 1 ]] || USE_COLOR=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --only-issues) ONLY_ISSUES=1; shift ;;
        --env) FILTER_ENV="$2"; shift 2 ;;
        --alg) FILTER_ALG="$2"; shift 2 ;;
        --no-color) USE_COLOR=0; shift ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

if [[ "$USE_COLOR" == "1" ]]; then
    C_GREEN=$'\033[32m'; C_YELLOW=$'\033[33m'; C_RED=$'\033[31m'
    C_CYAN=$'\033[36m'; C_DIM=$'\033[2m'; C_RESET=$'\033[0m'
else
    C_GREEN=""; C_YELLOW=""; C_RED=""; C_CYAN=""; C_DIM=""; C_RESET=""
fi

color_for_status() {
    case "$1" in
        COMPLETE) echo -n "$C_GREEN" ;;
        RUNNING) echo -n "$C_CYAN" ;;
        PENDING) echo -n "$C_DIM" ;;
        STALE_RUNNING|NEEDS_RESUME) echo -n "$C_YELLOW" ;;
        MISSING|NO_LOG_YET) echo -n "$C_RED" ;;
        *) echo -n "" ;;
    esac
}

# ── Build RUN_PATH -> "jobid_taskid" map by scanning every .out log's header once ──
declare -A RUNPATH_TO_TASK
if [[ -d "$LOGS_DIR" ]]; then
    for f in "$LOGS_DIR"/mw_b10k_*.sbatch_*.out; do
        [[ -f "$f" ]] || continue
        rp=$(grep -m1 "^RUN_PATH" "$f" | sed 's/^RUN_PATH *= *//' | xargs)
        [[ -z "$rp" ]] && continue
        base=$(basename "$f" .out)
        jobtask="${base##*.sbatch_}"
        # Keep the highest job ID seen for a given run_path (most recent attempt).
        prev="${RUNPATH_TO_TASK[$rp]:-}"
        prev_job="${prev%%|*}"
        this_job="${jobtask%%_*}"
        if [[ -z "$prev" || "$this_job" -ge "${prev_job:-0}" ]]; then
            RUNPATH_TO_TASK["$rp"]="$jobtask|$f"
        fi
    done
fi

# ── Current queue snapshot (only shows what's live right now) ──────────────────
declare -A JOBSTATE
while IFS='|' read -r jobid state; do
    [[ -z "$jobid" ]] && continue
    JOBSTATE["$jobid"]="$state"
done < <(squeue -u "$USER" -h -o "%i|%t" 2>/dev/null)

now_epoch=$(date +%s)

header_fmt="%-22s %-19s %-5s %-4s %-14s %-9s %-6s %-9s %-9s %-10s %s\n"
printf "$header_fmt" "ENV" "TYPE" "ALG" "SEED" "STATUS" "STEP" "PCT" "STEP/HR" "ETA" "EVAL_SUCC" "DETAIL"
printf '%.0s-' $(seq 1 140); echo

declare -A COUNTS
declare -A COUNTS_BY_ENV
declare -A COUNTS_BY_TYPE
declare -A RESUBMIT_TASKS   # "env|alg" -> comma-separated array task IDs to resubmit
total=0

type_idx_of() {
    local t="$1" i
    for i in "${!TYPES[@]}"; do [[ "${TYPES[$i]}" == "$t" ]] && { echo "$i"; return; }; done
}

for env in "${ENVS[@]}"; do
  [[ -n "$FILTER_ENV" && "$env" != "$FILTER_ENV" ]] && continue
  for type in "${TYPES[@]}"; do
    for alg in "${ALGS[@]}"; do
      [[ -n "$FILTER_ALG" && "$alg" != "$FILTER_ALG" ]] && continue
      for seed in "${SEEDS[@]}"; do
        total=$((total+1))
        run_path="$RUNS_DIR/$env/$type/${alg}_s${seed}"
        status="MISSING"; step="-"; pct="-"; rate="-"; eta="-"; eval_succ="-"; detail=""

        if [[ -f "$run_path/training_complete" ]]; then
            status="COMPLETE"
            step="$TOTAL_STEPS"; pct="100%"
            log_csv="$run_path/log.csv"
            if [[ -f "$log_csv" ]]; then
                header=$(head -n1 "$log_csv")
                succ_col=$(echo "$header" | awk -F',' '{for(i=1;i<=NF;i++) if($i=="eval/success") print i}')
                if [[ -n "$succ_col" ]]; then
                    eval_succ=$(awk -F',' -v c="$succ_col" 'NF>=c && $c!="" {v=$c} END{if(v!="") printf "%.3f", v}' "$log_csv")
                    [[ -z "$eval_succ" ]] && eval_succ="-"
                fi
            fi

        elif [[ -d "$run_path" ]]; then
            log_csv="$run_path/log.csv"

            if [[ -f "$log_csv" ]]; then
                header=$(head -n1 "$log_csv")
                step_col=$(echo "$header" | awk -F',' '{for(i=1;i<=NF;i++) if($i=="step") print i}')
                succ_col=$(echo "$header" | awk -F',' '{for(i=1;i<=NF;i++) if($i=="eval/success") print i}')
                last_line=$(tail -n1 "$log_csv")
                step=$(echo "$last_line" | awk -F',' -v c="$step_col" '{print $c+0}')
                [[ -z "$step" ]] && step=0
                pct=$(awk -v s="$step" -v t="$TOTAL_STEPS" 'BEGIN{printf "%.1f%%", (s/t)*100}')
                if [[ -n "$succ_col" ]]; then
                    eval_succ=$(awk -F',' -v c="$succ_col" 'NF>=c && $c!="" {v=$c} END{if(v!="") printf "%.3f", v}' "$log_csv")
                    [[ -z "$eval_succ" ]] && eval_succ="-"
                fi

                log_mtime=$(stat -c %Y "$log_csv" 2>/dev/null || echo 0)
                start_mtime=$(stat -c %Y "$run_path/config.yaml" 2>/dev/null || echo "$log_mtime")
                age_min=$(( (now_epoch - log_mtime) / 60 ))
                elapsed_hr_num=$(( (log_mtime - start_mtime) ))
                if [[ "$elapsed_hr_num" -gt 60 && "$step" -gt 0 ]]; then
                    rate=$(awk -v s="$step" -v e="$elapsed_hr_num" 'BEGIN{printf "%.0f", s/(e/3600)}')
                    remaining=$(( TOTAL_STEPS - step ))
                    if [[ "$rate" -gt 0 ]]; then
                        eta_hr=$(awk -v r="$remaining" -v rt="$rate" 'BEGIN{printf "%.1f", r/rt}')
                        eta="${eta_hr}h"
                    fi
                fi
            else
                step=0; pct="0.0%"; age_min=999999
            fi

            entry="${RUNPATH_TO_TASK[$run_path]:-}"
            jobtask="${entry%%|*}"
            errfile="${entry#*|}"; errfile="${errfile%.out}.err"
            jobid="${jobtask%%_*}"
            qstate=""
            if [[ -n "$jobtask" && -n "${JOBSTATE[$jobtask]:-}" ]]; then
                qstate="${JOBSTATE[$jobtask]}"
            elif [[ -n "$jobid" && -n "${JOBSTATE[$jobid]:-}" ]]; then
                qstate="${JOBSTATE[$jobid]}"
            fi

            if [[ "$step" -ge "$TOTAL_STEPS" ]]; then
                status="COMPLETE"
                detail="reached $TOTAL_STEPS steps (no training_complete sentinel found, inferred from log)"
            elif [[ -n "$qstate" ]]; then
                case "$qstate" in
                    R|CG)
                        if [[ -f "$log_csv" && "$age_min" -gt "$STALE_MINUTES" ]]; then
                            status="STALE_RUNNING"
                            detail="job $jobtask is R but log untouched ${age_min}m"
                        else
                            status="RUNNING"
                            detail="job $jobtask"
                        fi
                        ;;
                    PD)
                        status="PENDING"
                        detail="job $jobtask queued"
                        ;;
                    *)
                        status="NEEDS_RESUME"
                        detail="job $jobtask state=$qstate"
                        ;;
                esac
            else
                if [[ -f "$log_csv" ]]; then
                    status="NEEDS_RESUME"
                    detail="no active Slurm task; log last written ${age_min}m ago"
                else
                    status="NO_LOG_YET"
                    detail="run dir exists, no log.csv, nothing queued"
                fi
                if [[ ( "$status" == "NEEDS_RESUME" || "$status" == "NO_LOG_YET" ) && -f "$errfile" ]]; then
                    hint=$(tail -n 5 "$errfile" 2>/dev/null | grep -iE "error|traceback|cancelled|time limit|oom|killed" | tail -n1)
                    [[ -n "$hint" ]] && detail="$detail | ${hint:0:60}"
                fi
            fi
        else
            status="MISSING"
            detail="no run directory"
        fi

        COUNTS["$status"]=$(( ${COUNTS["$status"]:-0} + 1 ))
        COUNTS_BY_ENV["$env|$status"]=$(( ${COUNTS_BY_ENV["$env|$status"]:-0} + 1 ))
        COUNTS_BY_TYPE["$type|$status"]=$(( ${COUNTS_BY_TYPE["$type|$status"]:-0} + 1 ))

        case "$status" in
            NEEDS_RESUME|MISSING|NO_LOG_YET)
                tidx=$(type_idx_of "$type")
                task_id=$(( tidx * ${#SEEDS[@]} + seed + 1 ))
                key="$env|$alg"
                RESUBMIT_TASKS["$key"]="${RESUBMIT_TASKS[$key]:-}${RESUBMIT_TASKS[$key]:+,}${task_id}"
                ;;
        esac

        if [[ "$ONLY_ISSUES" == "1" ]]; then
            case "$status" in
                COMPLETE|RUNNING|PENDING) continue ;;
            esac
        fi

        color=$(color_for_status "$status")
        printf "%s${header_fmt}%s" "$color" "$env" "$type" "$alg" "$seed" "$status" "$step" "$pct" "$rate" "$eta" "$eval_succ" "$detail" "$C_RESET"
      done
    done
  done
done

echo
echo "=== Summary ($total combos) ==="
for k in COMPLETE RUNNING STALE_RUNNING PENDING NEEDS_RESUME NO_LOG_YET MISSING; do
    n="${COUNTS[$k]:-0}"
    [[ "$n" == "0" ]] && continue
    color=$(color_for_status "$k")
    printf "  %s%-14s %d%s\n" "$color" "$k" "$n" "$C_RESET"
done

echo
echo "=== By environment ==="
for env in "${ENVS[@]}"; do
    [[ -n "$FILTER_ENV" && "$env" != "$FILTER_ENV" ]] && continue
    line="  $env: "
    for k in COMPLETE RUNNING STALE_RUNNING PENDING NEEDS_RESUME NO_LOG_YET MISSING; do
        n="${COUNTS_BY_ENV[$env|$k]:-0}"
        [[ "$n" == "0" ]] && continue
        line+="$k=$n  "
    done
    echo "$line"
done

echo
echo "=== By feedback type ==="
for type in "${TYPES[@]}"; do
    line="  $type: "
    for k in COMPLETE RUNNING STALE_RUNNING PENDING NEEDS_RESUME NO_LOG_YET MISSING; do
        n="${COUNTS_BY_TYPE[$type|$k]:-0}"
        [[ "$n" == "0" ]] && continue
        line+="$k=$n  "
    done
    echo "$line"
done

if [[ ${#RESUBMIT_TASKS[@]} -gt 0 ]]; then
    echo
    echo "=== Suggested resubmit commands ==="
    echo "(fill in --account/--partition/--qos for your allocation; task IDs sorted ascending)"
    for key in "${!RESUBMIT_TASKS[@]}"; do
        env="${key%%|*}"; alg="${key##*|}"
        tasks=$(echo "${RESUBMIT_TASKS[$key]}" | tr ',' '\n' | sort -n -u | paste -sd,)
        echo "ENV_NAME=$env sbatch --array=$tasks --account=<acct> --partition=<part> slurm/mw_b10k_${alg}.sbatch"
    done
fi

# Nonzero exit if anything needs attention -- lets you use this in a cron/watch loop.
if [[ "${COUNTS[NEEDS_RESUME]:-0}" -gt 0 || "${COUNTS[MISSING]:-0}" -gt 0 || "${COUNTS[NO_LOG_YET]:-0}" -gt 0 || "${COUNTS[STALE_RUNNING]:-0}" -gt 0 ]]; then
    exit 1
fi
exit 0
