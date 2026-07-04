#!/usr/bin/env bash
#set -eo pipefail 
#eval "$(conda shell.bash hook)"
# conda activate crisp

cd ../prep/MogeSAM

ROOT="$1"
DATA_PATH="${ROOT%/}_videos"  # Append "_video" suffix

DIRS=("$DATA_PATH"/*)
NUM_DIRS=${#DIRS[@]}

if [[ -n "${CRISP_GPU_IDS:-}" ]]; then
    IFS=',' read -r -a GPU_IDS <<< "$CRISP_GPU_IDS"
    GPU_COUNT=${#GPU_IDS[@]}
else
    GPU_COUNT=$(nvidia-smi -L | wc -l)
    GPU_IDS=($(seq 0 $((GPU_COUNT-1))))
fi

worker() {
    local gpu_id="$1"
    shift
    local folders=("$@")

    for cam_folder in "${folders[@]}"; do
        echo "→ GPU $gpu_id │ $cam_folder"
        CUDA_VISIBLE_DEVICES="$gpu_id" \
        python inference.py \
            --input_path "$cam_folder" \
            --checkpoint checkpoints/tapip3d_final.pth \
            --resolution_factor 1
    done
}

for gpu_idx in "${!GPU_IDS[@]}"; do
    gpu_id="${GPU_IDS[$gpu_idx]}"
    gpu_dirs=()
    for (( idx=gpu_idx; idx<NUM_DIRS; idx+=GPU_COUNT )); do
        gpu_dirs+=("${DIRS[idx]}")
    done
    worker "$gpu_id" "${gpu_dirs[@]}" &
done

wait
# ModuleNotFoundError: No module named 'timm.layers'
echo "🏁  All jobs finished."
