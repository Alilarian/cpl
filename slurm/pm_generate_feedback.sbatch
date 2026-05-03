#!/bin/bash
# Generate PointMass feedback labels for all 5 feedback types (array job).
#
# Each array task generates one feedback type:
#   1 = pref       (pairwise preference)
#   2 = corr       (corrective)
#   3 = demo       (demonstrative, K-way)
#   4 = estop      (non-sequential e-stop)
#   5 = seq_estop  (sequential e-stop)
#
# Depends on:
#   - datasets/pm/pool.npz          from pm_generate_pool.sbatch
#   - runs/pm_sac_oracle/advantage_vi_N100.npz  from pm_compute_vi.sbatch
#
# Pipeline order on CHPC:
#   1. sbatch slurm/pm_generate_pool.sbatch
#   2. sbatch slurm/pm_compute_vi.sbatch
#   3. sbatch --array=1-5 slurm/pm_generate_feedback.sbatch   (this script)
#   4. sbatch --array=1-5 slurm/pm_train_cpl.sbatch
#
# Usage:
#   sbatch --array=1-5 \
#          --job-name=pm_feedback \
#          --time=1:00:00 \
#          --cpus-per-task=2 \
#          --mem=16G \
#          --account=<account> \
#          --partition=<partition> \
#          slurm/pm_generate_feedback.sbatch
#
# Optional overrides (set before sbatch or via --export):
#   POOL_NPZ        default: datasets/pm/pool.npz
#   ADVANTAGE_NPZ   default: runs/pm_sac_oracle/advantage_vi_N100.npz
#   OUTPUT_DIR      default: datasets/pm
#   N_PAIRS         default: 60000   (all types; estop/seq_estop stop early once reached)
#   N_COUNTERFACTUALS default: 4    (demo: K = 1 expert + N_CF + 1 original)
#   RHO             default: 0.8    (estop / seq_estop)
#   LAM             default: 2.0
#   KAPPA           default: 1.9101 (tuned; p10 of peak-H with rho=0.8)
#   HORIZON         default: 4      (seq_estop window h; [h-1, T-h]=[3,46] covers early stops)

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=slurm/logs/%x_%A_%a.out
#SBATCH --error=slurm/logs/%x_%A_%a.err

set -euo pipefail

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "SLURM_ARRAY_TASK_ID is not set. Submit with sbatch --array=1-5 ..."
    exit 1
fi

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
if [[ ! -f "${SUBMIT_DIR}/setup_shell.sh" ]]; then
    echo "ERROR: could not locate repo root (no setup_shell.sh in ${SUBMIT_DIR})"
    exit 1
fi
REPO_ROOT="$SUBMIT_DIR"

# Map task ID → feedback type
TYPES=(pref corr demo estop seq_estop)
FEEDBACK_TYPE="${TYPES[$((SLURM_ARRAY_TASK_ID - 1))]}"

# Paths
POOL_NPZ="${POOL_NPZ:-datasets/pm/pool.npz}"
ADVANTAGE_NPZ="${ADVANTAGE_NPZ:-runs/pm_sac_oracle/advantage_vi_N100.npz}"
OUTPUT_DIR="${OUTPUT_DIR:-datasets/pm}"

[[ "$POOL_NPZ"      != /* ]] && POOL_NPZ="${REPO_ROOT}/${POOL_NPZ}"
[[ "$ADVANTAGE_NPZ" != /* ]] && ADVANTAGE_NPZ="${REPO_ROOT}/${ADVANTAGE_NPZ}"
[[ "$OUTPUT_DIR"    != /* ]] && OUTPUT_DIR="${REPO_ROOT}/${OUTPUT_DIR}"

# Hyperparameters
N_PAIRS="${N_PAIRS:-60000}"
N_COUNTERFACTUALS="${N_COUNTERFACTUALS:-4}"
RHO="${RHO:-0.8}"
LAM="${LAM:-2.0}"
KAPPA="${KAPPA:-1.9101}"
HORIZON="${HORIZON:-4}"

OUTPUT_NPZ="${OUTPUT_DIR}/${FEEDBACK_TYPE}_labels.npz"

echo "=============================="
echo "SLURM_JOB_ID    = ${SLURM_JOB_ID:-}"
echo "ARRAY_TASK_ID   = ${SLURM_ARRAY_TASK_ID}"
echo "FEEDBACK_TYPE   = ${FEEDBACK_TYPE}"
echo "POOL_NPZ        = ${POOL_NPZ}"
echo "ADVANTAGE_NPZ   = ${ADVANTAGE_NPZ}"
echo "OUTPUT_NPZ      = ${OUTPUT_NPZ}"
echo "=============================="

if [[ ! -f "$POOL_NPZ" ]]; then
    echo "ERROR: pool.npz not found at ${POOL_NPZ}"
    echo "Run pm_generate_pool.sbatch first."
    exit 1
fi

if [[ ! -f "$ADVANTAGE_NPZ" ]]; then
    echo "ERROR: advantage_vi.npz not found at ${ADVANTAGE_NPZ}"
    echo "Run pm_compute_vi.sbatch first."
    exit 1
fi

. "${REPO_ROOT}/setup_shell.sh"
mkdir -p "$OUTPUT_DIR"

# Build the type-specific argument list
EXTRA_ARGS=()
case "$FEEDBACK_TYPE" in
    pref)
        EXTRA_ARGS=(--n-pairs "$N_PAIRS" --skip-expert)
        ;;
    corr)
        EXTRA_ARGS=(--n-pairs "$N_PAIRS" --skip-expert)
        ;;
    demo)
        EXTRA_ARGS=(--n-pairs "$N_PAIRS" --n-counterfactuals "$N_COUNTERFACTUALS" --skip-expert)
        ;;
    estop)
        EXTRA_ARGS=(--rho "$RHO" --lam "$LAM" --kappa "$KAPPA" --n-pairs "$N_PAIRS" --skip-expert)
        ;;
    seq_estop)
        EXTRA_ARGS=(--rho "$RHO" --lam "$LAM" --kappa "$KAPPA" --horizon "$HORIZON" --n-pairs "$N_PAIRS" --skip-expert)
        ;;
esac

python3 "${REPO_ROOT}/scripts/generate_pm_feedback.py" \
    --type          "$FEEDBACK_TYPE" \
    --pool          "$POOL_NPZ"      \
    --advantage-npz "$ADVANTAGE_NPZ" \
    --out           "$OUTPUT_NPZ"    \
    "${EXTRA_ARGS[@]}"
