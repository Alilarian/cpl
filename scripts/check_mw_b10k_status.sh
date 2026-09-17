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
#   bash scripts/check_mw_b10k_status.sh --cluster kingspeak   # scope the sacct
#       query to just this cluster instead of every registered one ("all")
#   bash scripts/check_mw_b10k_status.sh --no-color
#
# The output's CLUSTER column shows which cluster each matched job is actually
# running on, straight from sacct -- use this to see how work is spread across
# kingspeak/notchpeak/granite2 without having to log into each one separately.
#
# Env overrides: REPO_ROOT, RUNS_DIR, LOGS_DIR, TOTAL_STEPS,
#   STALE_MINUTES (default: 60 -- floor only; the effective per-row threshold is
#   max(STALE_MINUTES, 2x that run's own expected eval_freq interval, computed
#   from its measured STEP/HR), so slow feedback types like credit_assignment
#   don't get flagged STALE_RUNNING just for behaving normally),
#   EVAL_FREQ (default: 5000, matching every mw_state_dense config),
#   SACCT_START (default: 30 days ago), SACCT_CLUSTERS (default: "all" -- queries
#   every cluster registered with the shared slurmdbd in one call; same as
#   --cluster above, an explicit comma list like "kingspeak,notchpeak,granite"
#   works too if "all" errors on your site -- run `sacctmgr show clusters -p`
#   to see the exact registered names)

set -uo pipefail

REPO_ROOT="${REPO_ROOT:-$PWD}"
RUNS_DIR="${RUNS_DIR:-$REPO_ROOT/runs/mw_b10k}"
LOGS_DIR="${LOGS_DIR:-$REPO_ROOT/slurm/logs}"
TOTAL_STEPS="${TOTAL_STEPS:-500000}"
STALE_MINUTES="${STALE_MINUTES:-60}"
# log.csv is only written once per eval_freq steps (5000, fixed across every
# mw_state_dense config), and throughput varies hugely by feedback type --
# seq_estop/scalar write every few minutes, pref/corr roughly hourly,
# credit_assignment every several hours. A single fixed STALE_MINUTES flags
# slow-but-healthy types as "possibly hung" purely from timing coincidence.
# The effective threshold per row is max(STALE_MINUTES, 2x that run's own
# expected eval_freq interval, from its measured STEP/HR).
EVAL_FREQ="${EVAL_FREQ:-5000}"

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
        --cluster) SACCT_CLUSTERS="$2"; shift 2 ;;
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
# Keyed by file MTIME, not job-ID magnitude: job IDs are only comparable within one
# cluster (kingspeak/granite2/notchpeak each have independent counters), so "biggest
# ID = most recent" silently picks a dead job from the wrong cluster over the real,
# currently-running one. Wall-clock mtime is cluster-agnostic and always correct.
declare -A RUNPATH_TO_TASK
declare -A RUNPATH_TO_MTIME
if [[ -d "$LOGS_DIR" ]]; then
    for f in "$LOGS_DIR"/mw_b10k_*.sbatch_*.out; do
        [[ -f "$f" ]] || continue
        rp=$(grep -m1 "^RUN_PATH" "$f" | sed 's/^RUN_PATH *= *//' | xargs)
        [[ -z "$rp" ]] && continue
        base=$(basename "$f" .out)
        jobtask="${base##*.sbatch_}"
        this_mtime=$(stat -c %Y "$f" 2>/dev/null || echo 0)
        prev_mtime="${RUNPATH_TO_MTIME[$rp]:-0}"
        if [[ "$this_mtime" -ge "$prev_mtime" ]]; then
            RUNPATH_TO_TASK["$rp"]="$jobtask|$f"
            RUNPATH_TO_MTIME["$rp"]="$this_mtime"
        fi
    done
fi

# ── Historical + live job state via sacct ───────────────────────────────────────
# squeue only shows jobs that are CURRENTLY queued/running -- the instant a job
# ends for any reason (TIMEOUT, PREEMPTED, FAILED, CANCELLED, or a clean finish)
# it disappears from squeue, making "just hit its walltime 2 minutes ago" look
# identical to "no job ever existed." sacct keeps the historical record -- state,
# elapsed, and the walltime it was given -- so we can tell those apart.
#
# Both sacct and squeue are also scoped to whichever cluster's login node you
# run them from by default -- a job on granite2 is invisible to sacct when run
# from kingspeak/notchpeak, and vice versa, even though CHPC's home filesystem
# (and this repo checkout) is shared across all three. -M/--clusters asks the
# shared slurmdbd for every registered cluster in one query. If "all" isn't a
# valid target on your Slurm setup, override with an explicit comma list, e.g.:
#   SACCT_CLUSTERS=kingspeak,notchpeak,granite bash scripts/check_mw_b10k_status.sh
# (run `sacctmgr show clusters -p` once to see the exact registered names).
SACCT_START="${SACCT_START:-$(date -d '-30 days' +%Y-%m-%d 2>/dev/null || date -v-30d +%Y-%m-%d 2>/dev/null || echo 2024-01-01)}"
SACCT_CLUSTERS="${SACCT_CLUSTERS:-all}"

declare -A JOBSTATE JOBELAPSED JOBLIMIT JOBCLUSTER
sacct_n=0
while IFS='|' read -r jobid state elapsed tlimit cluster; do
    [[ -z "$jobid" ]] && continue
    state_short="${state%% *}"   # strip " by <uid>" off e.g. "CANCELLED by 123"
    JOBSTATE["$jobid"]="$state_short"
    JOBELAPSED["$jobid"]="$elapsed"
    JOBLIMIT["$jobid"]="$tlimit"
    JOBCLUSTER["$jobid"]="$cluster"
    sacct_n=$((sacct_n+1))
done < <(sacct -M "$SACCT_CLUSTERS" -u "$USER" -S "$SACCT_START" -E now \
             --format=JobID,State,Elapsed,Timelimit,Cluster --parsable2 -X --noheader 2>/dev/null)

if [[ "$sacct_n" -eq 0 ]]; then
    echo "WARNING: sacct -M $SACCT_CLUSTERS returned 0 job records -- either you have no" >&2
    echo "  jobs in the last $SACCT_START..now, or -M $SACCT_CLUSTERS isn't valid here." >&2
    echo "  Run 'sacctmgr show clusters -p' and retry with SACCT_CLUSTERS=<comma list>." >&2
fi

now_epoch=$(date +%s)

header_fmt="%-22s %-19s %-5s %-4s %-14s %-9s %-6s %-9s %-9s %-10s %-11s %-12s %-16s %s\n"
printf "$header_fmt" "ENV" "TYPE" "ALG" "SEED" "STATUS" "STEP" "PCT" "STEP/HR" "ETA" "EVAL_SUCC" "CLUSTER" "SLURM_STATE" "ELAPSED/LIMIT" "DETAIL"
printf '%.0s-' $(seq 1 170); echo

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
        slurm_state="-"; elapsed_limit="-"; cluster="-"

        if [[ -f "$run_path/training_complete" ]]; then
            status="COMPLETE"
            step="$TOTAL_STEPS"; pct="100%"
            log_csv="$run_path/log.csv"
            if [[ -f "$log_csv" ]]; then
                eval_succ=$(awk -F',' '
                    { gsub(/\r$/, "") }
                    NR==1 { for(i=1;i<=NF;i++) if($i=="eval/success") c=i; next }
                    c && NF>=c && $c!="" { v=$c }
                    END { if (v!="") printf "%.3f", v }
                ' "$log_csv")
                [[ -z "$eval_succ" ]] && eval_succ="-"
            fi

        elif [[ -d "$run_path" ]]; then
            log_csv="$run_path/log.csv"

            if [[ -f "$log_csv" ]]; then
                header=$(head -n1 "$log_csv")
                # Look up each column's index AND extract its value in one
                # single awk pass over the file, rather than reading the header
                # separately (`head -n1`) and re-scanning afterward. The trainer's
                # logger destructively truncates-and-rewrites log.csv with a new
                # header whenever a metric key first appears (CSVWriter's
                # _reset_csv_handler opens the file in "w" mode); if that rewrite
                # happens in the gap between a separate header-read and a
                # separate data-scan, column positions shift and we silently
                # grab the wrong column's value (observed: pulled eval/reward
                # instead of step, e.g. "9.18665", which then crashed bash's
                # integer arithmetic downstream). A single pass has no such gap.
                step=$(awk -F',' '
                    { gsub(/\r$/, "") }
                    NR==1 { for(i=1;i<=NF;i++) if($i=="step") c=i; next }
                    c && NF>=c && $c!="" { v=$c }
                    END { print v+0 }
                ' "$log_csv")
                [[ -z "$step" ]] && step=0
                # Hard type check: a valid step is always a plain non-negative
                # integer. Anything else (a float from a mis-attributed column,
                # an empty read, etc.) is a parsing artifact, not real data --
                # reset to 0 rather than feeding a non-integer into bash's `((.))`
                # arithmetic later, which errors out and kills the whole script.
                if ! [[ "$step" =~ ^[0-9]+$ ]]; then
                    step=0
                fi
                # Sanity clamp: a step count above total_steps is never real
                # (parsing artifact) -- treat it as unknown instead of nonsense.
                if [[ "$step" -gt "$TOTAL_STEPS" ]]; then
                    step=0
                fi
                pct=$(awk -v s="$step" -v t="$TOTAL_STEPS" 'BEGIN{printf "%.1f%%", (s/t)*100}')
                eval_succ=$(awk -F',' '
                    { gsub(/\r$/, "") }
                    NR==1 { for(i=1;i<=NF;i++) if($i=="eval/success") c=i; next }
                    c && NF>=c && $c!="" { v=$c }
                    END { if (v!="") printf "%.3f", v }
                ' "$log_csv")
                [[ -z "$eval_succ" ]] && eval_succ="-"

                # Rate comes straight from the trainer's own instantaneous
                # steps/sec log, not wall-clock/step arithmetic here -- that
                # breaks across resumes (config.yaml's mtime resets on every
                # relaunch while `step` keeps the full cumulative count from
                # every prior attempt, wildly inflating a locally-computed rate).
                sps=$(awk -F',' '
                    { gsub(/\r$/, "") }
                    NR==1 { for(i=1;i<=NF;i++) if($i=="time/steps_per_second") c=i; next }
                    c && NF>=c && $c!="" { v=$c }
                    END { print v+0 }
                ' "$log_csv")
                if [[ -n "$sps" ]] && awk -v s="$sps" 'BEGIN{exit !(s>0)}'; then
                    rate=$(awk -v s="$sps" 'BEGIN{printf "%.0f", s*3600}')
                    remaining=$(( TOTAL_STEPS - step ))
                    eta_hr=$(awk -v r="$remaining" -v rt="$rate" 'BEGIN{printf "%.1f", r/rt}')
                    eta="${eta_hr}h"
                fi

                log_mtime=$(stat -c %Y "$log_csv" 2>/dev/null || echo 0)
                age_min=$(( (now_epoch - log_mtime) / 60 ))
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
                slurm_state="$qstate"
                elapsed_limit="${JOBELAPSED[$jobtask]:--}/${JOBLIMIT[$jobtask]:--}"
                cluster="${JOBCLUSTER[$jobtask]:--}"
            elif [[ -n "$jobid" && -n "${JOBSTATE[$jobid]:-}" ]]; then
                qstate="${JOBSTATE[$jobid]}"
                slurm_state="$qstate"
                elapsed_limit="${JOBELAPSED[$jobid]:--}/${JOBLIMIT[$jobid]:--}"
                cluster="${JOBCLUSTER[$jobid]:--}"
            fi

            if [[ "$step" -ge "$TOTAL_STEPS" ]]; then
                status="COMPLETE"
                detail="reached $TOTAL_STEPS steps (no training_complete sentinel found, inferred from log)"
            elif [[ -n "$qstate" ]]; then
                effective_stale_min="$STALE_MINUTES"
                if [[ "$rate" != "-" ]] && awk -v r="$rate" 'BEGIN{exit !(r>0)}'; then
                    expected_interval_min=$(awk -v ef="$EVAL_FREQ" -v r="$rate" 'BEGIN{printf "%.0f", (ef/r)*60}')
                    dynamic_min=$(( expected_interval_min * 2 ))
                    if [[ "$dynamic_min" -gt "$effective_stale_min" ]]; then
                        effective_stale_min="$dynamic_min"
                    fi
                fi
                case "$qstate" in
                    RUNNING|CONFIGURING|COMPLETING)
                        if [[ -f "$log_csv" && "$age_min" -gt "$effective_stale_min" ]]; then
                            status="STALE_RUNNING"
                            detail="job $jobtask is $qstate but log untouched ${age_min}m (expected ~${expected_interval_min:-?}m/eval) -- may be hung"
                        else
                            status="RUNNING"
                            detail="job $jobtask, log updated ${age_min}m ago"
                        fi
                        ;;
                    PENDING)
                        status="PENDING"
                        detail="job $jobtask queued"
                        ;;
                    TIMEOUT)
                        status="NEEDS_RESUME"
                        detail="job $jobtask HIT ITS WALLTIME (ran full $elapsed_limit) -- resubmit, will resume from checkpoint"
                        ;;
                    PREEMPTED)
                        status="NEEDS_RESUME"
                        detail="job $jobtask PREEMPTED by scheduler at $elapsed_limit -- did not hit walltime, resubmit"
                        ;;
                    FAILED|OUT_OF_MEMORY|NODE_FAIL)
                        status="NEEDS_RESUME"
                        detail="job $jobtask $qstate after $elapsed_limit -- check .err, likely a real crash"
                        ;;
                    CANCELLED)
                        status="NEEDS_RESUME"
                        detail="job $jobtask CANCELLED after $elapsed_limit (manual scancel)"
                        ;;
                    COMPLETED)
                        status="NEEDS_RESUME"
                        detail="job $jobtask ended cleanly (Slurm COMPLETED) but log shows only $step/$TOTAL_STEPS steps -- check script logic, not an infra issue"
                        ;;
                    *)
                        status="NEEDS_RESUME"
                        detail="job $jobtask state=$qstate ($elapsed_limit)"
                        ;;
                esac
            else
                if [[ -f "$log_csv" ]]; then
                    status="NEEDS_RESUME"
                    detail="no Slurm record in last $SACCT_START..now; log last written ${age_min}m ago"
                else
                    status="NO_LOG_YET"
                    detail="run dir exists, no log.csv, no Slurm record found"
                fi
            fi
            if [[ ( "$status" == "NEEDS_RESUME" || "$status" == "NO_LOG_YET" ) && -f "$errfile" ]]; then
                hint=$(tail -n 5 "$errfile" 2>/dev/null | grep -iE "error|traceback|cancelled|time limit|oom|killed" | tail -n1)
                [[ -n "$hint" ]] && detail="$detail | ${hint:0:60}"
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
        printf "%s${header_fmt}%s" "$color" "$env" "$type" "$alg" "$seed" "$status" "$step" "$pct" "$rate" "$eta" "$eval_succ" "$cluster" "$slurm_state" "$elapsed_limit" "$detail" "$C_RESET"
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
