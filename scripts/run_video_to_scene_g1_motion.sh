#!/usr/bin/env bash
set -u -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ $# -lt 1 ]]; then
  echo "Usage: GPU_ID=5 bash scripts/run_video_to_scene_g1_motion.sh /path/to/video.mp4" >&2
  exit 1
fi

VIDEO_PATH="$(realpath "$1")"
SEQ_NAME="${SEQ_NAME:-$(basename "${VIDEO_PATH%.*}")}"
HMR_TYPE="${HMR_TYPE:-gv}"
GPU_ID="${GPU_ID:-}"
CRISP_ENV="${CRISP_ENV:-crisp}"
PYROKI_ENV="${PYROKI_ENV:-pyroki}"
WORK_ROOT="${WORK_ROOT:-$REPO_ROOT/batch_work/single_video}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/logs/video_to_scene_g1_motion}"
ALLOW_POINTCLOUD_MESH_FALLBACK="${ALLOW_POINTCLOUD_MESH_FALLBACK:-0}"

pick_gpu() {
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits \
    | awk -F, '{gsub(/ /, "", $1); gsub(/ /, "", $2); gsub(/ /, "", $3); if ($2 < 1000 && $3 < 10) {print $1; exit}}'
}

require_empty_gpu() {
  local gpu="$1"
  local used
  used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu" | tr -d ' ')"
  if [[ -z "$used" || "$used" -ge 1000 ]]; then
    echo "GPU $gpu is not empty enough: memory.used=${used:-unknown} MiB" >&2
    return 1
  fi
}

post_scene_ready() {
  local seq="$1"
  local root="$REPO_ROOT/results/output/post_scene/$seq/$HMR_TYPE"
  [[ -f "$root/scene_mesh_sqs/scene_mesh_sqs.obj" \
    && -f "$root/hmr/human_motion.npz" \
    && -f "$root/hmr/$seq.npz" ]]
}

if [[ ! -f "$VIDEO_PATH" ]]; then
  echo "Missing video: $VIDEO_PATH" >&2
  exit 1
fi

if [[ -z "$GPU_ID" ]]; then
  GPU_ID="$(pick_gpu)"
fi
if [[ -z "$GPU_ID" ]]; then
  echo "No empty GPU found. Set GPU_ID explicitly after checking gpustat." >&2
  exit 1
fi
require_empty_gpu "$GPU_ID" || exit 1

mkdir -p "$WORK_ROOT" "$LOG_DIR"
WORK_ROOT="$(realpath -m "$WORK_ROOT")"
LOG_DIR="$(realpath -m "$LOG_DIR")"
mkdir -p "$WORK_ROOT" "$LOG_DIR"

ROOT="$WORK_ROOT/$SEQ_NAME"
VIDEOS_ROOT="${ROOT}_videos"
mkdir -p "$VIDEOS_ROOT"
ln -sfn "$VIDEO_PATH" "$VIDEOS_ROOT/$SEQ_NAME.mp4"

SCENE_ROOT="$REPO_ROOT/results/output/scene/$SEQ_NAME/$HMR_TYPE"
POST_ROOT="$REPO_ROOT/results/output/post_scene/$SEQ_NAME/$HMR_TYPE"
POST_MESH="$POST_ROOT/scene_mesh_sqs/scene_mesh_sqs.obj"
POINTCLOUD="$SCENE_ROOT/nksr_input/pointcloud_world.npz"
RETARGET="$REPO_ROOT/results/output/g1_pyroki/$SEQ_NAME/$HMR_TYPE/${SEQ_NAME}_scene_g1_pyroki_simple.npz"
DATASET="$REPO_ROOT/results/output/scene_g1_motion_dataset/$SEQ_NAME/$HMR_TYPE/${SEQ_NAME}_scene_g1_motion.npz"
LOG_FILE="$LOG_DIR/$SEQ_NAME.log"

export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_ROOT/runtime_shims${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_HOME="${TORCH_HOME:-$HOME/.cache/torch}"
export CRISP_GPU_IDS="$GPU_ID"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

{
  set -e
  echo "===== $(date +'%F %T') start ====="
  echo "seq=$SEQ_NAME"
  echo "video=$VIDEO_PATH"
  echo "gpu_id=$GPU_ID"

  if ! post_scene_ready "$SEQ_NAME"; then
    echo "===== $(date +'%F %T') CRISP real2sim ====="
    set +e
    conda run --no-capture-output -n "$CRISP_ENV" bash "$REPO_ROOT/run_crisp_video.sh" "$ROOT"
    crisp_status=$?
    set -e
    echo "crisp_status=$crisp_status"
  else
    echo "===== $(date +'%F %T') CRISP skipped: post_scene ready ====="
  fi

  if ! post_scene_ready "$SEQ_NAME"; then
    if [[ "$ALLOW_POINTCLOUD_MESH_FALLBACK" != "1" ]]; then
      echo "post_scene is not ready and fallback is disabled"
      exit 11
    fi
    if [[ ! -f "$POINTCLOUD" ]]; then
      echo "post_scene is not ready and pointcloud is missing: $POINTCLOUD"
      exit 12
    fi

    echo "===== $(date +'%F %T') fallback scene mesh from pointcloud ====="
    conda run --no-capture-output -n "$CRISP_ENV" python "$REPO_ROOT/scripts/build_fallback_scene_mesh_from_pointcloud.py" \
      --seq-name "$SEQ_NAME" \
      --hmr-type "$HMR_TYPE" \
      --force

    echo "===== $(date +'%F %T') CRISP post_scene from fallback mesh ====="
    HMR_TYPE="$HMR_TYPE" conda run --no-capture-output -n "$CRISP_ENV" bash "$REPO_ROOT/vis_scripts/viser_m/rot.sh" "$SEQ_NAME"
    conda run --no-capture-output -n "$CRISP_ENV" python "$REPO_ROOT/vis_scripts/viser_m/rotate_scene_sqs_only.py" \
      --sequence-name "$SEQ_NAME" \
      --hmr-type "$HMR_TYPE"
  fi

  if ! post_scene_ready "$SEQ_NAME"; then
    echo "post_scene is still incomplete after CRISP/fallback"
    echo "expected mesh=$POST_MESH"
    exit 13
  fi

  echo "===== $(date +'%F %T') PyRoki simple retarget ====="
  conda run --no-capture-output -n "$PYROKI_ENV" python "$REPO_ROOT/scripts/retarget_crisp_scene_to_g1_pyroki.py" \
    --seq-name "$SEQ_NAME" \
    --hmr-type "$HMR_TYPE" \
    --mode simple \
    --force

  echo "===== $(date +'%F %T') export scene+g1 dataset ====="
  conda run --no-capture-output -n "$PYROKI_ENV" python "$REPO_ROOT/scripts/export_scene_g1_motion_dataset.py" \
    --seq-name "$SEQ_NAME" \
    --hmr-type "$HMR_TYPE" \
    --retarget-mode simple \
    --force

  echo "===== $(date +'%F %T') validate ====="
  conda run --no-capture-output -n "$PYROKI_ENV" python - "$DATASET" <<'PY'
import sys
import numpy as np

path = sys.argv[1]
d = np.load(path, allow_pickle=True)
z = d["g1_root_pos"][:, 2]
print(f"dataset={path}")
print(f"frames={int(d['frame_count'])} fps={float(d['fps']):.2f} dof={d['g1_dof_pos'].shape[1]}")
print(f"mesh_vertices={d['scene_mesh_vertices'].shape[0]} mesh_faces={d['scene_mesh_faces'].shape[0]}")
print(f"root_z_min_mean_max={float(z.min()):.4f},{float(z.mean()):.4f},{float(z.max()):.4f}")
print(f"root_dz_min_max={float(np.diff(z).min()):.4f},{float(np.diff(z).max()):.4f}")
PY
  echo "===== $(date +'%F %T') done ====="
} 2>&1 | tee "$LOG_FILE"
pipeline_status=${PIPESTATUS[0]}
if [[ $pipeline_status -ne 0 ]]; then
  exit "$pipeline_status"
fi

if [[ -f "$DATASET" ]]; then
  echo "$DATASET"
else
  exit 20
fi
