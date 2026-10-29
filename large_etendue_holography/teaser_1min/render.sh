#!/usr/bin/env bash
# Usage: ./render.sh [l|h] [script] [Scene ...]
#   l = 480p15 test (default), h = 1080p60 final.
# Renders each scene into media/ (in parallel) and concatenates all scenes into
# media/final/LargeEtendueHolography_1min_vN_<res>.mp4 (N taken from the script name).
set -euo pipefail
cd "$(dirname "$0")"
Q="${1:-l}"; shift || true
SCRIPT="${1:-$(ls animation_large_etendue_holography_1min_v*.py | sort -V | tail -1)}"; shift || true
ALL=(T1_Problem T2_Method T3_Payoff)
SCENES=("$@"); [ ${#SCENES[@]} -eq 0 ] && SCENES=("${ALL[@]}")
source ~/miniconda3/etc/profile.d/conda.sh
conda activate imp_tome
# imp_tome ships libffi 8 only; system pango/gobject needs libffi.so.7
export LD_PRELOAD=/lib/x86_64-linux-gnu/libffi.so.7
# assets are cached by the 5-min version (../data/generated); fail early if missing
[ -f ../data/generated/done_v1 ] || { echo "missing ../data/generated (run the 5-min script once)"; exit 1; }
NAME="$(basename "$SCRIPT" .py)"
VER="${NAME##*_}"
if [ "$Q" = "h" ]; then FLAG=-qh; RES=1080p60; else FLAG=-ql; RES=480p15; fi
mkdir -p scratch/logs
for S in "${SCENES[@]}"; do
  manim $FLAG --media_dir media "$SCRIPT" "$S" > "scratch/logs/${NAME}_${RES}_${S}.log" 2>&1 || echo "FAILED $S (see scratch/logs)" &
done
wait
if [ ${#SCENES[@]} -eq ${#ALL[@]} ]; then
  mkdir -p media/final
  LIST=media/final/list_${VER}_${RES}.txt; : > "$LIST"
  for S in "${ALL[@]}"; do [ -f "media/videos/${NAME}/${RES}/${S}.mp4" ] || { echo "missing $S"; exit 1; }; echo "file '../videos/${NAME}/${RES}/${S}.mp4'" >> "$LIST"; done
  OUT="media/final/LargeEtendueHolography_1min_${VER}_${RES}.mp4"
  ffmpeg -y -loglevel error -f concat -safe 0 -i "$LIST" -c copy "$OUT"
  echo "wrote $OUT"
fi
