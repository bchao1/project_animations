#!/usr/bin/env bash
# Usage: ./render.sh [l|h] [script.py] [Scene ...]
#   l = 480p15 test (default), h = 1080p60 final. Renders scenes in parallel (one media dir each: parallel manim
#   processes race on the shared text-SVG cache otherwise) and concatenates all scenes into media/final/.
set -euo pipefail
cd "$(dirname "$0")"
Q="${1:-l}"; shift || true
SCRIPT="$(ls animation_lot_v*.py | sort -V | tail -1)"
if [[ "${1:-}" == *.py ]]; then SCRIPT="$1"; shift; fi
ALL=(S00_Title S01_Problem S02_Layouts S03_Sources S04_Method S05_LoTDiT S06_Video S07_Control S08_Compare S09_Outlook)
SCENES=("$@"); [ ${#SCENES[@]} -eq 0 ] && SCENES=("${ALL[@]}")
source ~/miniconda3/etc/profile.d/conda.sh
conda activate imp_tome
export LD_PRELOAD=/lib/x86_64-linux-gnu/libffi.so.7   # imp_tome ships libffi 8; system pango needs libffi.so.7
NAME="$(basename "$SCRIPT" .py)"
if [ "$Q" = "h" ]; then FLAG=-qh; RES=1080p60; else FLAG=-ql; RES=480p15; fi
mkdir -p media/scenes
for S in "${SCENES[@]}"; do
  manim $FLAG --media_dir "media/scenes/$S" "$SCRIPT" "$S" > "media/log_${NAME}_${RES}_${S}.txt" 2>&1 &
done
FAIL=0
for job in $(jobs -p); do wait "$job" || FAIL=1; done
[ $FAIL -eq 0 ] || { echo "a scene failed; see media/log_${NAME}_${RES}_*.txt"; exit 1; }
if [ ${#SCENES[@]} -eq ${#ALL[@]} ]; then
  mkdir -p media/final
  LIST=media/final/list_${NAME}_${RES}.txt; : > "$LIST"
  for S in "${ALL[@]}"; do F="media/scenes/$S/videos/${NAME}/${RES}/${S}.mp4"; [ -f "$F" ] || { echo "missing $S"; exit 1; }
    echo "file '../../$F'" >> "$LIST"; done
  ffmpeg -y -loglevel error -f concat -safe 0 -i "$LIST" -c copy "media/final/LevelOfTokenDiffusion_5min_${NAME#animation_lot_}_${RES}.mp4"
  echo "wrote media/final/LevelOfTokenDiffusion_5min_${NAME#animation_lot_}_${RES}.mp4"
fi
