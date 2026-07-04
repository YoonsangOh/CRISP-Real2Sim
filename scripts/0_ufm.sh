#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

ROOT="$(realpath "${1%/}")"
cd "$REPO_ROOT/prep/MogeSAM"

###############################################################################
# 2) Paths
###############################################################################
DATA_PATH="${ROOT%/}_img"              # append “_img” if not already
[[ -d "$DATA_PATH" ]] || { echo "❌  '$DATA_PATH' not found"; exit 1; }

###############################################################################
# 3) GPUs
###############################################################################
if [[ -n "${CRISP_GPU_IDS:-}" ]]; then
    IFS=',' read -r -a GPU_IDS <<< "$CRISP_GPU_IDS"
    GPU_COUNT=${#GPU_IDS[@]}
else
    GPU_COUNT=$(nvidia-smi -L | wc -l)
    GPU_IDS=($(seq 0 $((GPU_COUNT-1))))
fi

echo "🖥️  Found $GPU_COUNT GPUs → ${GPU_IDS[*]}"
echo "📂  Scanning '$DATA_PATH' …"

###############################################################################
# 4) List all immediate sub‑folders (one job per folder)
###############################################################################
mapfile -d '' DIRS < <(find "$DATA_PATH" -mindepth 1 -maxdepth 1 -type d -print0)
NUM_DIRS=${#DIRS[@]}
echo "📄  ${NUM_DIRS} folders to process"

###############################################################################
# 5) Define worker (runs on a *single* GPU)
###############################################################################
worker() {
    local gpu_id="$1"
    shift
    local folders=("$@")

    for cam_folder in "${folders[@]}"; do
        seq=$(basename "$cam_folder")            # e.g. cam_06
        parent_dir=$(dirname  "$cam_folder")     # e.g. …/rich_07_img
        video_dir="${parent_dir}"             

        echo "→ GPU $gpu_id │ $seq"
        CUDA_VISIBLE_DEVICES="$gpu_id" \
            python ufm.py \
                --images "$cam_folder" \
                --stride 7 \
                --out "../../results/init/flows/$seq" \
                --mode window --window 4
    done
}

###############################################################################
# 6) Dispatch jobs: split DIRS array round‑robin by modulo GPU_COUNT
###############################################################################
pids=()
for gpu_slot in "${!GPU_IDS[@]}"; do
    gpu_id="${GPU_IDS[$gpu_slot]}"
    # build slice for this GPU
    gpu_dirs=()
    for (( idx=gpu_slot; idx<NUM_DIRS; idx+=GPU_COUNT )); do
        gpu_dirs+=("${DIRS[idx]}")
    done
    # start worker in background
    worker "$gpu_id" "${gpu_dirs[@]}" &
    pids+=("$!")
done

for pid in "${pids[@]}"; do
    wait "$pid"
done
echo "🏁  All jobs finished."
