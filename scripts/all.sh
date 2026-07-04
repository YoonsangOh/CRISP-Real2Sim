#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/data}"
SCRIPT_REL="$SCRIPT_DIR/all_gv.sh"

shopt -s nullglob

for videos_dir in "$DATA_ROOT"/_emdb*_videos; do
  [[ -d "$videos_dir" ]] || continue

  base="$(basename "$videos_dir")"  # e.g. vmm_a_videos
  if [[ "$base" != _emdb* ]]; then
    continue
  fi

  base="${videos_dir%_videos}"   # strip trailing "_videos" -> /.../vmm_a
  echo "==> Processing: $videos_dir  ->  bash $SCRIPT_REL $base"
  bash "$SCRIPT_REL" "$base"
done
