#!/usr/bin/env bash
# Usage: ./render.sh [l|h] [script] [Scene ...]
#   l = 480p15 test (default), h = 1080p60 final.
# Renders each scene into media/ (in parallel) and, when all scenes are rendered, concatenates them into
# media/final/RandomPhaseWaveSplatting_1min_vN_<res>.mp4 (N taken from the script name).
set -euo pipefail
cd "$(dirname "$0")"
Q="${1:-l}"; shift || true
SCRIPT="${1:-$(ls animation_rpws_1min_v*.py | sort -V | tail -1)}"; shift || true
ALL=(T1_Hook T2_Method T3_Payoff)
SCENES=("$@"); [ ${#SCENES[@]} -eq 0 ] && SCENES=("${ALL[@]}")
source ~/miniconda3/etc/profile.d/conda.sh
conda activate imp_tome
# imp_tome ships libffi 8 only; system pango/gobject needs libffi.so.7
export LD_PRELOAD=/lib/x86_64-linux-gnu/libffi.so.7
NAME="$(basename "$SCRIPT" .py)"
VER="${NAME##*_}"                      # e.g. v1
if [ "$Q" = "h" ]; then FLAG=-qh; RES=1080p60; else FLAG=-ql; RES=480p15; fi
# cached wave-optics assets come from ../data/generated (built by the 5-min script); the script only reads them
for S in "${SCENES[@]}"; do
  manim $FLAG --media_dir media "$SCRIPT" "$S" > "media/log_${NAME}_${RES}_${S}.txt" 2>&1 &
done
FAIL=0
for job in $(jobs -p); do wait "$job" || FAIL=1; done
[ $FAIL -eq 0 ] || { echo "a scene failed; see media/log_${NAME}_${RES}_*.txt"; exit 1; }
if [ ${#SCENES[@]} -eq ${#ALL[@]} ]; then
  mkdir -p media/final
  LIST=media/final/list_${VER}_${RES}.txt; : > "$LIST"
  for S in "${ALL[@]}"; do [ -f "media/videos/${NAME}/${RES}/${S}.mp4" ] || { echo "missing $S"; exit 1; }; echo "file '../videos/${NAME}/${RES}/${S}.mp4'" >> "$LIST"; done
  OUT="media/final/RandomPhaseWaveSplatting_1min_${VER}_${RES}.mp4"
  ffmpeg -y -loglevel error -f concat -safe 0 -i "$LIST" -c copy "$OUT"
  echo "wrote $OUT"
fi
