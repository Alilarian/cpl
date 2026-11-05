#!/bin/bash
# Regenerates the cum_estop labels for the 4 MetaWorld envs with the oracle
# correction (generate_cum_estop_labels.py --correction-policy oracle) and the
# 2026-11-05 no-stop fixes, then trains CPL and PIQL on them, one run per seed.
# Per env, chained with SLURM afterok dependencies:
#
#   gen    NUM_SHARDS-task array of slurm/cum_estop_generate_labels_sharded.sbatch
#   merge  slurm/cum_estop_merge_labels.sbatch (merge, verify, build the 5k/5k mix)
#   cpl    slurm/cum_estop_train_cpl.sbatch,  one task per seed
#   piql   slurm/cum_estop_train_piql.sbatch, one task per seed
#
# Each env's stop threshold H, pool subsample and checkpoint_bands.json are
# reused from its old labels in OLD_LABELS_DIR, so the new labels cover the
# same pool segments under the same stopping rule -- only the correction and
# the no-stop fixes differ. This is checked for every env before anything is
# submitted. The old labels and runs are left untouched.
#
# Usage (from the repo root on CHPC):
#   bash slurm/submit_cum_estop_oracle.sh          # labels + training, all 4 envs
#   bash slurm/submit_cum_estop_oracle.sh train    # merge (if needed) + training on labels already generated
#   ENVS="mw_door-open-v2" bash slurm/submit_cum_estop_oracle.sh
#
# Walltime: generation measured ~1 s per pool segment per worker locally
# (OMP_NUM_THREADS=1), i.e. minutes per shard. Training is 250k steps;
# cum_estop PIQL ran 0.01-0.03 s/step on rai-gpu-grn (H200) before.
#
# Resume: a training task killed by its walltime resumes from final_model.pt --
# rerun with `train`; finished runs (training_complete) are skipped.
#
# Optional env overrides:
#   ENVS                default: the 4 envs
#   OLD_LABELS_DIR      default: /scratch/general/vast/u1472210/cum_estop_labels
#   NEW_LABELS_DIR      default: /scratch/general/vast/u1472210/cum_estop_labels_oracle
#   POOL_DIR            default: /scratch/general/vast/u1472210/mw_de_pool
#   NUM_SHARDS          default: 4
#   SEEDS               default: "0 1 2"
#   CONDA_ENV_NAME      default: cpl_gpu
#   CPL_RUNS_DIR        default: runs/mw_cum_estop_oracle_cpl
#   PIQL_RUNS_DIR       default: runs/mw_cum_estop_oracle_piql
#   CPL_WANDB_PROJECT   default: mw-cpl-cum-estop-oracle
#   PIQL_WANDB_PROJECT  default: mw-piql-cum-estop-oracle
#   CPU_SBATCH_ARGS     allocation for gen + merge (CPU work), default: the rai GPU allocation below

set -euo pipefail

MODE="${1:-all}"
if [[ "$MODE" != all && "$MODE" != train ]]; then
    echo "usage: bash slurm/submit_cum_estop_oracle.sh [all|train]"
    exit 1
fi
if [[ ! -f setup_shell.sh ]]; then
    echo "ERROR: run from the repo root (no setup_shell.sh in $PWD)"
    exit 1
fi
if ! grep -q -- "--correction-policy" scripts/generate_cum_estop_labels.py; then
    echo "ERROR: this checkout predates the oracle correction -- pull the latest code first"
    exit 1
fi
mkdir -p slurm/logs

ENVS="${ENVS:-mw_button-press-v2 mw_door-open-v2 mw_drawer-open-v2 mw_plate-slide-v2}"
OLD_LABELS_DIR="${OLD_LABELS_DIR:-/scratch/general/vast/u1472210/cum_estop_labels}"
NEW_LABELS_DIR="${NEW_LABELS_DIR:-/scratch/general/vast/u1472210/cum_estop_labels_oracle}"
POOL_DIR="${POOL_DIR:-/scratch/general/vast/u1472210/mw_de_pool}"
NUM_SHARDS="${NUM_SHARDS:-4}"
SEEDS="${SEEDS:-0 1 2}"
CPL_RUNS_DIR="${CPL_RUNS_DIR:-runs/mw_cum_estop_oracle_cpl}"
PIQL_RUNS_DIR="${PIQL_RUNS_DIR:-runs/mw_cum_estop_oracle_piql}"
CPL_WANDB_PROJECT="${CPL_WANDB_PROJECT:-mw-cpl-cum-estop-oracle}"
PIQL_WANDB_PROJECT="${PIQL_WANDB_PROJECT:-mw-piql-cum-estop-oracle}"
export CONDA_ENV_NAME="${CONDA_ENV_NAME:-cpl_gpu}"

GPU=(--partition=rai-gpu-grn --qos=rai-gpu-grn --account=rai --gres=gpu:1)
read -r -a CPU_ALLOC <<< "${CPU_SBATCH_ARGS:-${GPU[*]}}"
GEN_CPUS=16
N_PER_BRANCH=5000  # the training sbatch default: a 5k + 5k mix
read -r -a SEED_ARR <<< "$SEEDS"
SEED_ARRAY="0-$(( ${#SEED_ARR[@]} - 1 ))"

# Prints "<H> <subsample_n|full> <n_stop> <n_no_stop> <n_unresolved>" for one env's old labels,
# after checking they came from this pool and cover exactly the seed-0 subsample of it.
read -r -d '' OLD_SETTINGS_PY <<'EOF' || true
import os, sys
import numpy as np

old, pool_path = sys.argv[1], sys.argv[2]
idx, n, H = [], {}, set()
for name in ("stop_correction_labels", "no_stop_demo_labels", "unresolved_events"):
    path = os.path.join(old, name + ".npz")
    if not os.path.exists(path):
        sys.exit(f"missing {path}")
    with np.load(path) as f:
        idx.append(f["pool_index"])
        n[name] = len(f["pool_index"])
        if "H" in f.files:
            H.update(np.unique(f["H"]).tolist())
if len(H) != 1:
    sys.exit(f"{old}: expected one H value, found {sorted(H)}")
with np.load(pool_path) as f:
    pool_reward = f["reward"]
with np.load(os.path.join(old, "no_stop_demo_labels.npz")) as f:
    r, p = f["reward"][:50], f["pool_index"][:50]
if not all(any(np.array_equal(r[j, k], pool_reward[p[j]]) for k in range(2)) for j in range(len(p))):
    sys.exit(f"{old}: no-stop originals don't match {pool_path} -- the old labels came from another pool")
idx, N = np.sort(np.concatenate(idx)), len(pool_reward)
if len(np.unique(idx)) != len(idx):
    sys.exit(f"{old}: some pool segment is labeled twice")
if len(idx) == N:
    subsample = "full"
else:
    keep = np.sort(np.random.default_rng(0).choice(N, size=len(idx), replace=False))
    if not np.array_equal(keep, idx):
        sys.exit(f"{old}: labels cover {len(idx)} of {N} pool segments, but not the seed-0 subsample")
    subsample = str(len(idx))
print(np.format_float_positional(np.float32(H.pop()), trim="-"), subsample,
      n["stop_correction_labels"], n["no_stop_demo_labels"], n["unresolved_events"])
EOF

old_settings() {
    ( set +u; export REPO_ROOT="$PWD"; . ./setup_shell.sh >/dev/null 2>&1; python3 -c "$OLD_SETTINGS_PY" "$@" )
}

declare -A H_OF SUB_OF
if [[ "$MODE" == all ]]; then
    echo "Old settings (${OLD_LABELS_DIR}):"
    for ENV_NAME in $ENVS; do
        # checkpoint_bands.json is copied in at submission, so it also catches a rerun before any shard finishes
        if [[ -e "${NEW_LABELS_DIR}/${ENV_NAME}/checkpoint_bands.json" ]] \
           || compgen -G "${NEW_LABELS_DIR}/${ENV_NAME}/*.npz" >/dev/null; then
            echo "ERROR: ${NEW_LABELS_DIR}/${ENV_NAME} was already submitted -- use 'train', or move it aside"
            exit 1
        fi
        if [[ ! -f "${OLD_LABELS_DIR}/${ENV_NAME}/checkpoint_bands.json" ]]; then
            echo "ERROR: missing ${OLD_LABELS_DIR}/${ENV_NAME}/checkpoint_bands.json"
            exit 1
        fi
        settings=$(old_settings "${OLD_LABELS_DIR}/${ENV_NAME}" "${POOL_DIR}/${ENV_NAME}/pool.npz")
        read -r H SUB N_STOP N_NOSTOP N_UNRES <<< "$settings"
        H_OF[$ENV_NAME]="$H"
        SUB_OF[$ENV_NAME]="$SUB"
        printf "  %-20s H=%-8s segments=%-6s old rows: stop %s, no-stop %s, unresolved %s\n" \
            "$ENV_NAME" "$H" "$SUB" "$N_STOP" "$N_NOSTOP" "$N_UNRES"
        if (( N_NOSTOP < N_PER_BRANCH * 11 / 10 )); then
            echo "  WARNING: only ${N_NOSTOP} old no-stop rows; the new run may fall short of ${N_PER_BRANCH} for the mix"
        fi
    done
fi

echo "Submitting (${MODE}):"
for ENV_NAME in $ENVS; do
    DEP=()
    if [[ "$MODE" == all ]]; then
        mkdir -p "${NEW_LABELS_DIR}/${ENV_NAME}"
        cp "${OLD_LABELS_DIR}/${ENV_NAME}/checkpoint_bands.json" "${NEW_LABELS_DIR}/${ENV_NAME}/"
        SUB_VARS=()
        [[ "${SUB_OF[$ENV_NAME]}" != full ]] && SUB_VARS=(SUBSAMPLE_N="${SUB_OF[$ENV_NAME]}" SUBSAMPLE_SEED=0)
        gen=$(env ENV_NAME="$ENV_NAME" H="${H_OF[$ENV_NAME]}" NUM_SHARDS="$NUM_SHARDS" N_WORKERS="$GEN_CPUS" \
                  OUTPUT_DIR="$NEW_LABELS_DIR" POOL_DIR="$POOL_DIR" CORRECTION_POLICY=oracle \
                  "${SUB_VARS[@]+"${SUB_VARS[@]}"}" \
              sbatch --parsable --job-name="cum_estop_gen_${ENV_NAME}" --array="0-$((NUM_SHARDS - 1))" \
                  --cpus-per-task="$GEN_CPUS" --mem=32G --time=02:00:00 "${CPU_ALLOC[@]}" \
                  slurm/cum_estop_generate_labels_sharded.sbatch)
        gen="${gen%%;*}"
        DEP=(--dependency="afterok:${gen}")
    fi

    merge=$(env ENV_NAME="$ENV_NAME" OUTPUT_DIR="$NEW_LABELS_DIR" NUM_SHARDS="$NUM_SHARDS" POOL_DIR="$POOL_DIR" \
                CORRECTION_POLICY=oracle N_PER_BRANCH="$N_PER_BRANCH" \
            sbatch --parsable --job-name="cum_estop_merge_${ENV_NAME}" "${DEP[@]+"${DEP[@]}"}" \
                --cpus-per-task=2 --mem=32G --time=01:00:00 "${CPU_ALLOC[@]}" \
                slurm/cum_estop_merge_labels.sbatch)
    merge="${merge%%;*}"

    cpl=$(env ENV_NAME="$ENV_NAME" SEEDS="$SEEDS" LABELS_DIR="$NEW_LABELS_DIR" RUNS_DIR="$CPL_RUNS_DIR" \
              N_PER_BRANCH="$N_PER_BRANCH" WANDB_PROJECT="$CPL_WANDB_PROJECT" DEVICE=cuda \
          sbatch --parsable --job-name="cum_estop_cpl_${ENV_NAME}" --dependency="afterok:${merge}" \
              --array="$SEED_ARRAY" --cpus-per-task=8 --mem=64G --time=06:00:00 "${GPU[@]}" \
              slurm/cum_estop_train_cpl.sbatch)
    piql=$(env ENV_NAME="$ENV_NAME" SEEDS="$SEEDS" LABELS_DIR="$NEW_LABELS_DIR" RUNS_DIR="$PIQL_RUNS_DIR" \
               N_PER_BRANCH="$N_PER_BRANCH" WANDB_PROJECT="$PIQL_WANDB_PROJECT" DEVICE=cuda \
           sbatch --parsable --job-name="cum_estop_piql_${ENV_NAME}" --dependency="afterok:${merge}" \
               --array="$SEED_ARRAY" --cpus-per-task=8 --mem=64G --time=12:00:00 "${GPU[@]}" \
               slurm/cum_estop_train_piql.sbatch)
    printf "  %-20s gen=%-10s merge=%-10s cpl=%-10s piql=%s\n" \
        "$ENV_NAME" "${gen:--}" "$merge" "${cpl%%;*}" "${piql%%;*}"
    unset gen
done
echo "Labels: ${NEW_LABELS_DIR}/<env>/  Runs: ${CPL_RUNS_DIR}/, ${PIQL_RUNS_DIR}/  WandB: ${CPL_WANDB_PROJECT}, ${PIQL_WANDB_PROJECT}"
