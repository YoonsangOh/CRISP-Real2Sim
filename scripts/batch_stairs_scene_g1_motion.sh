#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

VIDEO_DIR="${VIDEO_DIR:-$REPO_ROOT/videomimic_videos}"
VIDEO_PATTERN="${VIDEO_PATTERN:-*stairs*.mp4}"
HMR_TYPE="${HMR_TYPE:-gv}"
GPU_ID="${GPU_ID:-}"
WORK_ROOT="${WORK_ROOT:-$REPO_ROOT/batch_work/stairs_single}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/logs/batch_stairs_scene_g1_motion}"
SUMMARY_FILE="${SUMMARY_FILE:-$LOG_DIR/summary.tsv}"
SINGLE_SCRIPT="$REPO_ROOT/scripts/run_video_to_scene_g1_motion.sh"

pick_gpu() {
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits \
    | awk -F, '{gsub(/ /, "", $1); gsub(/ /, "", $2); gsub(/ /, "", $3); if ($2 < 1000 && $3 < 10) {print $1; exit}}'
}

failure_reason() {
  local log_file="$1"
  local reason=""
  if [[ -f "$log_file" ]]; then
    reason="$(
      grep -E 'OutOfMemoryError|CUDA out of memory|Killed|post_scene is not ready|pointcloud is missing|FileNotFoundError|No scene mesh found|Traceback|ERROR conda|GPU .*not empty' "$log_file" \
        | tail -1 \
        | tr '\t' ' ' \
        | sed 's/[[:space:]]\+/ /g; s/^ //; s/ $//'
    )"
  fi
  [[ -n "$reason" ]] || reason="see log: $log_file"
  printf "%s" "$reason"
}

if [[ -z "$GPU_ID" ]]; then
  GPU_ID="$(pick_gpu)"
fi

if [[ -z "$GPU_ID" ]]; then
  echo "No empty GPU found. Set GPU_ID explicitly after checking gpustat." >&2
  exit 1
fi

VIDEO_DIR="$(realpath -m "$VIDEO_DIR")"
WORK_ROOT="$(realpath -m "$WORK_ROOT")"
LOG_DIR="$(realpath -m "$LOG_DIR")"
SUMMARY_FILE="$(realpath -m "$SUMMARY_FILE")"
mkdir -p "$WORK_ROOT" "$LOG_DIR" "$(dirname "$SUMMARY_FILE")"

mapfile -t VIDEOS < <(find -L "$VIDEO_DIR" -maxdepth 1 -type f -iname "$VIDEO_PATTERN" | sort)
TOTAL=${#VIDEOS[@]}
if (( TOTAL == 0 )); then
  echo "No '$VIDEO_PATTERN' videos found under $VIDEO_DIR" >&2
  exit 1
fi

if [[ ! -f "$SUMMARY_FILE" ]]; then
  printf "timestamp\tseq\tstatus\tseconds\tmessage\n" > "$SUMMARY_FILE"
fi

echo "repo=$REPO_ROOT"
echo "video_dir=$VIDEO_DIR"
echo "video_pattern=$VIDEO_PATTERN"
echo "gpu_id=$GPU_ID"
echo "total_videos=$TOTAL"
echo "summary=$SUMMARY_FILE"

for idx in "${!VIDEOS[@]}"; do
  video="${VIDEOS[$idx]}"
  seq="$(basename "${video%.*}")"
  dataset="$REPO_ROOT/results/output/scene_g1_motion_dataset/$seq/$HMR_TYPE/${seq}_scene_g1_motion.npz"
  log_file="$LOG_DIR/$seq.log"

  if [[ -f "$dataset" ]]; then
    printf "%s\t%s\tSKIP\t0\t%s\n" "$(date +'%F %T')" "$seq" "dataset exists" >> "$SUMMARY_FILE"
    echo "[$((idx + 1))/$TOTAL] skip $seq"
    continue
  fi

  start_ts=$(date +%s)
  echo "[$((idx + 1))/$TOTAL] start $seq"

  GPU_ID="$GPU_ID" HMR_TYPE="$HMR_TYPE" WORK_ROOT="$WORK_ROOT" LOG_DIR="$LOG_DIR/single_logs" \
    bash "$SINGLE_SCRIPT" "$video" > "$log_file" 2>&1
  status=$?
  elapsed=$(( $(date +%s) - start_ts ))

  if [[ $status -eq 0 && -f "$dataset" ]]; then
    printf "%s\t%s\tOK\t%d\t%s\n" "$(date +'%F %T')" "$seq" "$elapsed" "$dataset" >> "$SUMMARY_FILE"
    echo "[$((idx + 1))/$TOTAL] ok $seq ${elapsed}s"
  else
    reason="$(failure_reason "$log_file")"
    printf "%s\t%s\tFAIL\t%d\t%s | log=%s\n" "$(date +'%F %T')" "$seq" "$elapsed" "$reason" "$log_file" >> "$SUMMARY_FILE"
    echo "[$((idx + 1))/$TOTAL] fail $seq ${elapsed}s reason=$reason log=$log_file"
  fi
done

echo "done summary=$SUMMARY_FILE"
