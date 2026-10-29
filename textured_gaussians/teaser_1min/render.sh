#!/usr/bin/env bash
# Usage: ./render.sh [l|h] [script] [Scene ...]
#   l = 480p15 test (default), h = 1080p60 final.
# Renders each scene into media/ (in parallel) and, when all scenes are rendered, concatenates them into
# media/final/TexturedGaussians_1min_vN_<res>.mp4 (vN taken from the script name).
set -euo pipefail
cd "$(dirname "$0")"
Q="${1:-l}"; shift || true
SCRIPT="${1:-animation_textured_gaussians_1min_v4.py}"; shift || true
ALL=(T1_Hook T2_Method T3_Payoff)
SCENES=("$@"); [ ${#SCENES[@]} -eq 0 ] && SCENES=("${ALL[@]}")
source ~/miniconda3/etc/profile.d/conda.sh
conda activate imp_tome
# imp_tome ships libffi 8 only; system pango/gobject needs libffi.so.7
export LD_PRELOAD=/lib/x86_64-linux-gnu/libffi.so.7
[ -f ../data/generated/toy_results.json ] || { echo "missing ../data/generated (run ../build_assets.py)"; exit 1; }
NAME="$(basename "$SCRIPT" .py)"
VN="${NAME##*_}"
if [ "$Q" = "h" ]; then FLAG=-qh; RES=1080p60; else FLAG=-ql; RES=480p15; fi
mkdir -p scratch/logs
PIDS=()
for S in "${SCENES[@]}"; do
  manim $FLAG --media_dir media "$SCRIPT" "$S" > "scratch/logs/${NAME}_${RES}_${S}.log" 2>&1 &
  PIDS+=($!)
done
FAIL=0
for i in "${!PIDS[@]}"; do wait "${PIDS[$i]}" || { echo "FAILED ${SCENES[$i]} (see scratch/logs/${NAME}_${RES}_${SCENES[$i]}.log)"; FAIL=1; }; done
[ $FAIL -eq 0 ] || exit 1
if [ ${#SCENES[@]} -eq ${#ALL[@]} ]; then
  mkdir -p media/final
  LIST=media/final/list_${VN}_${RES}.txt; : > "$LIST"
  for S in "${ALL[@]}"; do [ -f "media/videos/${NAME}/${RES}/${S}.mp4" ] || { echo "missing $S"; exit 1; }; echo "file '../videos/${NAME}/${RES}/${S}.mp4'" >> "$LIST"; done
  OUT="media/final/TexturedGaussians_1min_${VN}_${RES}.mp4"
  ffmpeg -y -loglevel error -f concat -safe 0 -i "$LIST" -c copy "$OUT"
  echo "wrote $OUT"
fi
