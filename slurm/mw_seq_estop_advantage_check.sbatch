#!/bin/bash
# Check whether seq-estop's "preferred" label (index 0) actually has higher
# oracle return-based advantage than the "non-preferred" segment (index 1).
#
# generate_seq_estop_labels.py assigns preferred purely by position relative
# to the simulated stop time tau (t == tau -> stop_seg preferred, t < tau ->
# cont_seg preferred) -- it never compares a scalar return/advantage between
# the two candidate segments, unlike generate_pref_labels.py which explicitly
# picks+asserts the higher-rl_sum segment as preferred. This job re-scores
# every stored pair with the SAME oracle used at label-generation time and
# reports what fraction of pairs actually have segment_score[preferred] >
# segment_score[non-preferred] -- see scripts/analyze_seq_estop_advantage.py.
#
# Usage:
#   sbatch --account=<account> --partition=<partition> \
#          [--time=HH:MM:SS] [--cpus-per-task=N] [--mem=XG] \
#          slurm/mw_seq_estop_advantage_check.sbatch
#
# Resource overrides (command-line flags override #SBATCH defaults below):
#   --time=HH:MM:SS     default: 4:00:00
#   --cpus-per-task=N   default: 8
#   --mem=XG            default: 64G
#
# Optional env overrides:
#   ENV_NAME          default: mw_drawer-open-v2
#   ORACLE_RUNS_BASE  default: runs/runs/chpc/oracle_sac_seeds  (relative to repo root)
#   ORACLE_SEED       default: seed-1
#   LABELS_DIR        default: /scratch/general/vast/u1472210/mw_de_labels
#   OUTPUT_DIR        default: results/seq_estop_advantage  (relative to repo root)
#   MAX_PAIRS         default: 20000   (random subsample scored per env; unset/0 = score all)
#   DISCOUNT          default: 0.99    (must match generate_seq_estop_labels.py's value)
#   MCMC_SAMPLES      default: 32      (must match generate_seq_estop_labels.py's value)

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=/uufs/chpc.utah.edu/common/home/u1472210/cpl/slurm/logs/%x_%j.out
#SBATCH --error=/uufs/chpc.utah.edu/common/home/u1472210/cpl/slurm/logs/%x_%j.err

set -euo pipefail

mkdir -p "${SLURM_SUBMIT_DIR:-$PWD}/slurm/logs"

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
if [[ ! -f "${SUBMIT_DIR}/setup_shell.sh" ]]; then
    echo "ERROR: no setup_shell.sh in ${SUBMIT_DIR}"
    exit 1
fi
REPO_ROOT="$SUBMIT_DIR"

ENV_NAME="${ENV_NAME:-mw_drawer-open-v2}"
ORACLE_RUNS_BASE="${ORACLE_RUNS_BASE:-runs/runs/chpc/oracle_sac_seeds}"
ORACLE_SEED="${ORACLE_SEED:-seed-1}"
LABELS_DIR="${LABELS_DIR:-/scratch/general/vast/u1472210/mw_de_labels}"
OUTPUT_DIR="${OUTPUT_DIR:-results/seq_estop_advantage}"
MAX_PAIRS="${MAX_PAIRS:-20000}"
DISCOUNT="${DISCOUNT:-0.99}"
MCMC_SAMPLES="${MCMC_SAMPLES:-32}"

[[ "$ORACLE_RUNS_BASE" != /* ]] && ORACLE_RUNS_BASE="${REPO_ROOT}/${ORACLE_RUNS_BASE}"
[[ "$OUTPUT_DIR"       != /* ]] && OUTPUT_DIR="${REPO_ROOT}/${OUTPUT_DIR}"

LABELS_PATH="${LABELS_DIR}/${ENV_NAME}/seq_estop_labels.npz"
ORACLE_DIR="${ORACLE_RUNS_BASE}/${ENV_NAME}/${ORACLE_SEED}"

echo "=============================="
echo "SLURM_JOB_ID  = ${SLURM_JOB_ID:-}"
echo "ENV_NAME      = ${ENV_NAME}"
echo "ORACLE_DIR    = ${ORACLE_DIR}"
echo "LABELS_PATH   = ${LABELS_PATH}"
echo "OUTPUT_DIR    = ${OUTPUT_DIR}"
echo "MAX_PAIRS     = ${MAX_PAIRS}"
echo "DISCOUNT      = ${DISCOUNT}"
echo "MCMC_SAMPLES  = ${MCMC_SAMPLES}"
echo "=============================="

if [[ ! -f "$LABELS_PATH" ]]; then
    echo "ERROR: labels not found: ${LABELS_PATH}"
    echo "Run mw_de_generate_labels.sbatch (seq_estop task) first."
    exit 1
fi

if [[ ! -d "$ORACLE_DIR" ]]; then
    echo "ERROR: oracle dir not found: ${ORACLE_DIR}"
    exit 1
fi

. "${REPO_ROOT}/setup_shell.sh"

MAX_PAIRS_ARG=()
if [[ "$MAX_PAIRS" != "0" ]]; then
    MAX_PAIRS_ARG=(--max-pairs "$MAX_PAIRS")
fi

python3 "${REPO_ROOT}/scripts/analyze_seq_estop_advantage.py" \
    --labels-path  "$LABELS_PATH"  \
    --run-dir      "$ORACLE_DIR"   \
    --discount     "$DISCOUNT"     \
    --mcmc-samples "$MCMC_SAMPLES" \
    --output-dir   "$OUTPUT_DIR"   \
    "${MAX_PAIRS_ARG[@]}" \
    --device cpu
