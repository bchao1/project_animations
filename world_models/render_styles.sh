#!/usr/bin/env bash
# Usage: ./render_styles.sh [l|h] [NN ...]
#   Renders styles/animation_wm_sNN_*.py (all 10 if none given) in parallel; l = 480p15, h = 1080p60.
#   Finals are copied to media/final/WorldModels_sNN_<style>_<res>.mp4
set -uo pipefail
cd "$(dirname "$0")"
Q="${1:-l}"; shift || true
IDS=("$@"); [ ${#IDS[@]} -eq 0 ] && IDS=(01 02 03 04 05 06 07 08 09 10)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate imp_tome
# imp_tome ships libffi 8 only; system pango/gobject needs libffi.so.7
export LD_PRELOAD=/lib/x86_64-linux-gnu/libffi.so.7
if [ "$Q" = "h" ]; then FLAG=-qh; RES=1080p60; else FLAG=-ql; RES=480p15; fi
[ -d data/clips/spmem ] || python scratch/prep_assets.py
mkdir -p media/final media/logs
for N in "${IDS[@]}"; do
  F=$(ls styles/animation_wm_s${N}_*.py | head -1)
  (
    NAME=$(basename "$F" .py)
    SCENE=$(grep -oE '^class (S[0-9]+_[A-Za-z]+)' "$F" | head -1 | cut -d' ' -f2)
    # parallel jobs share media/texts; a job can lose a race on a cached text SVG, so retry
    for TRY in 1 2 3; do
      manim $FLAG --media_dir media "$F" "$SCENE" > "media/logs/${NAME}_${RES}.txt" 2>&1 && break
      sleep $((RANDOM % 5 + 1))
    done
    if [ -f "media/videos/${NAME}/${RES}/${SCENE}.mp4" ] && ! grep -q "Traceback\|Error" "media/logs/${NAME}_${RES}.txt"; then
      OUT="media/final/WorldModels_${NAME#animation_wm_}_${RES}.mp4"
      if [ "$Q" = "h" ]; then  # manim pads a few frames; cut to exactly 10.000 s (600 frames)
        ffmpeg -y -loglevel error -i "media/videos/${NAME}/${RES}/${SCENE}.mp4" -frames:v 600 -c:v libx264 -crf 14 \
          -preset slow -pix_fmt yuv420p -movflags +faststart "$OUT"
      else
        cp "media/videos/${NAME}/${RES}/${SCENE}.mp4" "$OUT"
      fi
      echo "ok   $NAME"
    else
      echo "FAIL $NAME (media/logs/${NAME}_${RES}.txt)"
    fi
  ) &
done
wait
