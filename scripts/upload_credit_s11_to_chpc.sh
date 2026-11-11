#!/bin/bash
# Upload the stride-11 credit assignment labels to CHPC (the launchers come
# with the repo: git pull on CHPC).
# Run from your own terminal on this machine (it needs your Duo login; one
# prompt, the connection is reused for every step).
#
#   1. Check that CHPC's stride-1 source (mw_de_labels/<env>/credit_labels.npz)
#      is byte-identical to the local copy the stride-11 files were derived
#      from -- so the runs use the same segments as the b10k baseline.
#   2. rsync credit_stride11_labels.npz for the 4 envs next to it.
#   3. Verify the uploaded files' sha256 against the local ones.
#
# Usage:
#   bash scripts/upload_credit_s11_to_chpc.sh
#
# Optional env overrides:
#   CHPC_HOST       default: chpc-notch   (ssh alias)
#   CHPC_LABELS_DIR default: /scratch/general/vast/u1472210/mw_de_labels
#   CHPC_REPO       default: /uufs/chpc.utah.edu/common/home/u1472210/cpl

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHPC_HOST="${CHPC_HOST:-chpc-notch}"
CHPC_LABELS_DIR="${CHPC_LABELS_DIR:-/scratch/general/vast/u1472210/mw_de_labels}"
CHPC_REPO="${CHPC_REPO:-/uufs/chpc.utah.edu/common/home/u1472210/cpl}"
ENVS=(mw_button-press-v2 mw_door-open-v2 mw_drawer-open-v2 mw_plate-slide-v2)

CP="$HOME/.ssh/cm-credit-s11-%r@%h:%p"
SSH=(ssh -o ControlPath="$CP")

echo "Opening one SSH connection to ${CHPC_HOST} (Duo prompt) ..."
ssh -o ControlMaster=yes -o ControlPath="$CP" -o ControlPersist=60m -fN "$CHPC_HOST"
trap 'ssh -o ControlPath="$CP" -O exit "$CHPC_HOST" 2>/dev/null || true' EXIT

# ── 1. Same stride-1 source on both sides ─────────────────────────────────────
echo; echo "== 1/3  Checking stride-1 source files match (sha256) ..."
for e in "${ENVS[@]}"; do
    src="${REPO_ROOT}/datasets/mw_de_labels/${e}/credit_labels.npz"
    new="${REPO_ROOT}/datasets/mw_de_labels/${e}/credit_stride11_labels.npz"
    [[ -f "$src" && -f "$new" ]] || { echo "ERROR: missing local file for ${e}"; exit 1; }
    l=$(sha256sum "$src" | cut -d' ' -f1)
    r=$("${SSH[@]}" "$CHPC_HOST" "sha256sum ${CHPC_LABELS_DIR}/${e}/credit_labels.npz" | cut -d' ' -f1)
    if [[ "$l" != "$r" ]]; then
        echo "ERROR: ${e}: CHPC credit_labels.npz differs from the local copy"
        echo "       local  ${l}"
        echo "       chpc   ${r:-<missing>}"
        echo "       The stride-11 file would not match the baseline's segments -- not uploading."
        exit 1
    fi
    echo "  ${e}: match"
done

# ── 2. Upload labels ──────────────────────────────────────────────────────────
echo; echo "== 2/3  Uploading credit_stride11_labels.npz ..."
for e in "${ENVS[@]}"; do
    rsync -av --progress -e "ssh -o ControlPath=$CP" \
        "${REPO_ROOT}/datasets/mw_de_labels/${e}/credit_stride11_labels.npz" \
        "${CHPC_HOST}:${CHPC_LABELS_DIR}/${e}/"
done

# ── 3. Verify uploads ─────────────────────────────────────────────────────────
echo; echo "== 3/3  Verifying uploaded files (sha256) ..."
for e in "${ENVS[@]}"; do
    l=$(sha256sum "${REPO_ROOT}/datasets/mw_de_labels/${e}/credit_stride11_labels.npz" | cut -d' ' -f1)
    r=$("${SSH[@]}" "$CHPC_HOST" "sha256sum ${CHPC_LABELS_DIR}/${e}/credit_stride11_labels.npz" | cut -d' ' -f1)
    [[ "$l" == "$r" ]] || { echo "ERROR: ${e}: uploaded file checksum mismatch"; exit 1; }
    echo "  ${e}: ok"
done

echo
echo "Done. To train, on CHPC:"
echo "  cd ${CHPC_REPO} && git pull && bash slurm/submit_credit_s11.sh all"
