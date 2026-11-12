# Packed submission for the array-style b10k launchers (sourced by the
# submit_*.sh scripts, not run on its own). submit_packed splits a task list
# into groups of PACK and submits one slurm/pack_gpu.sbatch array job, one GPU
# per group, that runs a group's tasks at once.
#
# Optional env overrides:
#   PACK       tasks per GPU          default: 4  (1 = one task per GPU, as before)
#   CPUS       cores per GPU job, shared by its tasks   default: 4
#   MEM        memory per GPU job     default: 64G
#   PARTITION  default: rai-gpu-grn. QOS, ACCOUNT and (on Blackwell) CONDA_ENV_NAME
#              follow from it unless set:
#                rai-gpu-grn        H200             rai-gpu-grn / rai
#                granite-gpu        H100 NVL, L40S   granite-gpu / dbrown
#                soc-gpu-class-grn  RTX PRO 6000/4000 Blackwell (sm_120)
#                                   soc-gpu-students-grn / dbrown, env cpl_gpu_bw
#              Blackwell needs cpl_gpu_bw (slurm/setup_cpl_gpu_bw.sh): cpl_gpu's
#              PyTorch 2.5 fails there with "no kernel image is available".
#   QOS, ACCOUNT  required for any other partition

PACK="${PACK:-4}"
CPUS="${CPUS:-4}"
MEM="${MEM:-64G}"
PARTITION="${PARTITION:-rai-gpu-grn}"
case "$PARTITION" in
    rai-gpu-grn)
        QOS="${QOS:-rai-gpu-grn}"; ACCOUNT="${ACCOUNT:-rai}" ;;
    granite-gpu)
        QOS="${QOS:-granite-gpu}"; ACCOUNT="${ACCOUNT:-dbrown}" ;;
    soc-gpu-class-grn)
        QOS="${QOS:-soc-gpu-students-grn}"; ACCOUNT="${ACCOUNT:-dbrown}"
        export CONDA_ENV_NAME="${CONDA_ENV_NAME:-cpl_gpu_bw}" ;;
esac
if [[ -z "${QOS:-}" || -z "${ACCOUNT:-}" ]]; then
    echo "ERROR: set QOS and ACCOUNT for partition ${PARTITION}"
    exit 1
fi

# "13-15,19" -> "13 14 15 19"
expand_tasks() {
    local part parts out=()
    IFS=, read -r -a parts <<< "$1"
    for part in "${parts[@]}"; do
        if [[ "$part" =~ ^([0-9]+)-([0-9]+)$ ]]; then
            out+=($(seq "${BASH_REMATCH[1]}" "${BASH_REMATCH[2]}"))
        elif [[ "$part" =~ ^[0-9]+$ ]]; then
            out+=("$part")
        else
            echo "ERROR: can't read task spec '${part}' (use e.g. 1-12 or 13-15,19)" >&2
            return 1
        fi
    done
    echo "${out[@]}"
}

# submit_packed JOB_NAME TASK_SPEC WALLTIME INNER_SBATCH
submit_packed() {
    local name="$1" spec="$2" time="$3" inner="$4"
    local list tasks groups=() i
    list=$(expand_tasks "$spec") || exit 1
    read -r -a tasks <<< "$list"
    for (( i = 0; i < ${#tasks[@]}; i += PACK )); do
        groups+=("$(IFS=,; echo "${tasks[*]:i:PACK}")")
    done
    echo "${name}: ${#tasks[@]} task(s) on ${#groups[@]} GPU(s) [${groups[*]}]" \
         "${PARTITION}/${QOS}/${ACCOUNT}, ${CPUS} CPUs, ${MEM}, ${time}, env ${CONDA_ENV_NAME:-?}"
    TASK_GROUPS="${groups[*]}" INNER="$inner" \
        sbatch --job-name="$name" --array="1-${#groups[@]}" --time="$time" \
            --partition="$PARTITION" --qos="$QOS" --account="$ACCOUNT" \
            --gres=gpu:1 --cpus-per-task="$CPUS" --mem="$MEM" \
            slurm/pack_gpu.sbatch
}
