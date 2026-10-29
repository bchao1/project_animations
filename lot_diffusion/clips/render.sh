#!/usr/bin/env bash
# Usage: ./render.sh [l|h] [script.py] [Clip ...]  → media/final/<res>/LoT_<Clip>.mp4 (one media dir per clip).
set -euo pipefail
cd "$(dirname "$0")"
Q="${1:-l}"; shift || true
SCRIPT="$(ls animation_lot_clips_v*.py | sort -V | tail -1)"
if [[ "${1:-}" == *.py ]]; then SCRIPT="$1"; shift; fi
ALL=($(command grep -oE "^class K[0-9]+_[A-Za-z]+" "$SCRIPT" | sed 's/class //'))
SCENES=("$@"); [ ${#SCENES[@]} -eq 0 ] && SCENES=("${ALL[@]}")
source ~/miniconda3/etc/profile.d/conda.sh
conda activate imp_tome
export LD_PRELOAD=/lib/x86_64-linux-gnu/libffi.so.7
NAME="$(basename "$SCRIPT" .py)"
if [ "$Q" = "h" ]; then FLAG=-qh; RES=1080p60; else FLAG=-ql; RES=480p15; fi
mkdir -p media/logs media/final/$RES
for S in "${SCENES[@]}"; do
  ( manim $FLAG --media_dir "media/scenes/$S" "$SCRIPT" "$S" > "media/logs/${NAME}_${RES}_${S}.txt" 2>&1 \
    && cp "media/scenes/$S/videos/${NAME}/${RES}/${S}.mp4" "media/final/$RES/LoT_${S}.mp4" ) &
done
FAIL=0
for job in $(jobs -p); do wait "$job" || FAIL=1; done
[ $FAIL -eq 0 ] || { echo "a clip failed; see media/logs/${NAME}_${RES}_*.txt"; exit 1; }
echo "wrote media/final/$RES/"
