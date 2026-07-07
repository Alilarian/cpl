#!/bin/bash
# Phase 2: Generate full feedback labels for the MetaWorld DE sweep.
#
# One array task per feedback type (6 types, ignoring plain estop):
#   Task 1 — pref             fast: pool scoring only, no env rollouts   (~2h)
#   Task 2 — corr             expert rollout per segment                  (~8h)
#   Task 3 — demo             K rollouts per segment (most expensive)    (~16h)
#   Task 4 — seq_estop        V(s) scoring + estop simulation             (~4h)
#   Task 5 — scalar           V(s) scoring + sliding window               (~4h)
#   Task 6 — credit_assignment V(s) scoring + window extraction          (~4h)
#
# Each type produces ONE full label file sorted by oracle quality DESC.
# Budget subsets are applied at training time via dataset_kwargs.capacity=B,
# so capacity=50 ⊂ capacity=100 ⊂ … is guaranteed (nestedness).
#
# Pool size: 41 ckpts × 1000 segs = ~41k segments from mw_de_build_pool.sbatch.
# Generated label counts:
#   pref   : 100k pairs  (sampled from pool; oversampled, then filtered)
#   corr   : ~41k pairs  (one expert rollout per pool segment)
#   demo   : ~41k demos  (K=7 rollouts per pool segment)
#   seq_estop : ~100k–1M decision-point pairs (depends on kappa/rho/lam)
#   scalar : 100k pairs  (capped; sliding window over pool)
#   credit : ~41k examples (one per pool segment, C=31 candidate windows)
#
# Memory note: credit_assignment is the heaviest — the output array
#   (N, C, k, obs_dim) = (41k, 31, 20, 35) ≈ 3.5 GB obs alone; np.stack
#   peaks at ~2× during construction.  128G mem requested covers this.
#
# Usage:
#   sbatch --array=1-6 \
#          --account=<account> --partition=<partition> \
#          slurm/mw_de_generate_labels.sbatch
#
#   # Single type (e.g. pref only):
#   sbatch --array=1 --account=<account> --partition=<partition> \
#          slurm/mw_de_generate_labels.sbatch
#
# Optional env overrides:
#   ENV_NAME          default: mw_drawer-open-v2
#   ORACLE_RUNS_BASE  default: /scratch/general/vast/u1472210/oracle_sac_all
#   POOL_DIR          default: /scratch/general/vast/u1472210/mw_de_pool
#   OUTPUT_DIR        default: /scratch/general/vast/u1472210/mw_de_labels

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=slurm/logs/%x_%A_%a.out
#SBATCH --error=slurm/logs/%x_%A_%a.err

set -euo pipefail

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "ERROR: submit with --array=1-6"
    exit 1
fi

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
if [[ ! -f "${SUBMIT_DIR}/setup_shell.sh" ]]; then
    echo "ERROR: no setup_shell.sh in ${SUBMIT_DIR}"
    exit 1
fi
REPO_ROOT="$SUBMIT_DIR"

ENV_NAME="${ENV_NAME:-mw_drawer-open-v2}"
ORACLE_RUNS_BASE="${ORACLE_RUNS_BASE:-/scratch/general/vast/u1472210/oracle_sac_all}"
POOL_DIR="${POOL_DIR:-/scratch/general/vast/u1472210/mw_de_pool}"
OUTPUT_DIR="${OUTPUT_DIR:-/scratch/general/vast/u1472210/mw_de_labels}"

[[ "$ORACLE_RUNS_BASE" != /* ]] && ORACLE_RUNS_BASE="${REPO_ROOT}/${ORACLE_RUNS_BASE}"
[[ "$POOL_DIR"         != /* ]] && POOL_DIR="${REPO_ROOT}/${POOL_DIR}"
[[ "$OUTPUT_DIR"       != /* ]] && OUTPUT_DIR="${REPO_ROOT}/${OUTPUT_DIR}"

POOL_PATH="${POOL_DIR}/${ENV_NAME}/pool.npz"
ORACLE_DIR="${ORACLE_RUNS_BASE}/${ENV_NAME}"

TYPES=(pref corr demo seq_estop scalar credit_assignment)
FEEDBACK_TYPE="${TYPES[$(( SLURM_ARRAY_TASK_ID - 1 ))]}"

echo "=============================="
echo "SLURM_JOB_ID    = ${SLURM_JOB_ID:-}"
echo "ARRAY_TASK_ID   = ${SLURM_ARRAY_TASK_ID}"
echo "FEEDBACK_TYPE   = ${FEEDBACK_TYPE}"
echo "ENV_NAME        = ${ENV_NAME}"
echo "POOL_PATH       = ${POOL_PATH}"
echo "ORACLE_DIR      = ${ORACLE_DIR}"
echo "OUTPUT_DIR      = ${OUTPUT_DIR}"
echo "=============================="

if [[ ! -f "$POOL_PATH" ]]; then
    echo "ERROR: pool.npz not found: ${POOL_PATH}"
    echo "Run mw_de_build_pool.sbatch first."
    exit 1
fi

. "${REPO_ROOT}/setup_shell.sh"
mkdir -p "${OUTPUT_DIR}/${ENV_NAME}"

case "$FEEDBACK_TYPE" in

    pref)
        # Score all pool segments once; sample 100k pairs — no env rollouts.
        # Sorted by gap desc → capacity=B always takes top-B most informative.
        python3 "${REPO_ROOT}/scripts/generate_pref_labels.py" \
            --pool-path           "$POOL_PATH" \
            --run-dir             "$ORACLE_DIR" \
            --n-pairs             100000 \
            --min-gap-fraction    0.0 \
            --oversampling-factor 5.0 \
            --mcmc-samples        32 \
            --output-dir          "$OUTPUT_DIR" \
            --device cpu
        ;;

    corr)
        # Expert rollout from s_0 for every pool segment → ~41k corr pairs.
        # Sorted by improvement desc.
        python3 "${REPO_ROOT}/scripts/generate_corr_labels.py" \
            --pool-path    "$POOL_PATH" \
            --run-dir      "$ORACLE_DIR" \
            --min-adv-gap  0.0 \
            --mcmc-samples 32 \
            --output-dir   "$OUTPUT_DIR" \
            --device cpu
        ;;

    demo)
        # K=7: 1 expert + 5 stratified-tier rollouts + 1 original pool segment.
        # One demo per pool segment → ~41k demo examples.  Resumable.
        python3 "${REPO_ROOT}/scripts/generate_demo_labels.py" \
            --pool-path            "$POOL_PATH" \
            --run-dir              "$ORACLE_DIR" \
            --n-counterfactuals    6 \
            --checkpoints-per-tier 2 \
            --min-adv-gap          0.0 \
            --mcmc-samples         32 \
            --output-dir           "$OUTPUT_DIR" \
            --device cpu
        ;;

    seq_estop)
        # Simulate oracle e-stop; extract h-step windows at each decision point.
        python3 "${REPO_ROOT}/scripts/generate_seq_estop_labels.py" \
            --pool-path    "$POOL_PATH" \
            --run-dir      "$ORACLE_DIR" \
            --rho          0.8 \
            --lam          2.0 \
            --kappa        1.5 \
            --horizon      10 \
            --mcmc-samples 32 \
            --output-dir   "$OUTPUT_DIR" \
            --device cpu
        ;;

    scalar)
        # Sliding window over pool; oracle scores each segment.  Cap at 100k pairs.
        python3 "${REPO_ROOT}/scripts/generate_scalar_labels.py" \
            --pool-path    "$POOL_PATH" \
            --run-dir      "$ORACLE_DIR" \
            --segment-len  20 \
            --window-size  10 \
            --noise-std    0.1 \
            --n-pairs      100000 \
            --mcmc-samples 32 \
            --output-dir   "$OUTPUT_DIR" \
            --device cpu
        ;;

    credit_assignment)
        # Extract T-k+1 overlapping windows per trajectory; oracle picks best.
        # Memory note: output is (N, C, k, obs_dim).  For N=41k, C=31, k=20,
        # obs_dim=35 the obs array is ~3.5 GB; np.stack peaks at ~2× during
        # construction — hence the 128G mem request above.
        # --mcmc-samples 32 halves peak GPU/CPU tensor size vs default 64.
        # --batch-size 128 keeps per-batch V(s) tensor within ~2 GB.
        python3 "${REPO_ROOT}/scripts/generate_credit_labels.py" \
            --pool-path      "$POOL_PATH" \
            --run-dir        "$ORACLE_DIR" \
            --subsegment-len 20 \
            --batch-size     128 \
            --mcmc-samples   32 \
            --output-dir     "$OUTPUT_DIR" \
            --device cpu
        ;;

    *)
        echo "ERROR: unknown FEEDBACK_TYPE=${FEEDBACK_TYPE}"
        exit 1
        ;;
esac

echo "Done — ${FEEDBACK_TYPE} labels written to ${OUTPUT_DIR}/${ENV_NAME}/"
