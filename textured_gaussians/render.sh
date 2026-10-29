#!/usr/bin/env bash
# Usage: ./render.sh [l|h] [script] [Scene ...]
#   l = 480p15 test (default), h = 1080p60 final.
# Renders each scene into media/ (in parallel) and concatenates all scenes into media/final/.
set -euo pipefail
cd "$(dirname "$0")"
Q="${1:-l}"; shift || true
SCRIPT="${1:-animation_textured_gaussians_v3.py}"; shift || true
ALL=(S00_Title S01_Hook S02_Splatting S03_Limitation S04_Textures S05_UVMapping S06_Appearance S07_ToyFit S08_Results S09_Ablations S10_Outlook)
SCENES=("$@"); [ ${#SCENES[@]} -eq 0 ] && SCENES=("${ALL[@]}")
source ~/miniconda3/etc/profile.d/conda.sh
conda activate imp_tome
# imp_tome ships libffi 8 only; system pango/gobject needs libffi.so.7
export LD_PRELOAD=/lib/x86_64-linux-gnu/libffi.so.7
[ -f data/generated/toy_results.json ] || python build_assets.py
NAME="$(basename "$SCRIPT" .py)"
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
  LIST=media/final/list_${RES}.txt; : > "$LIST"
  for S in "${ALL[@]}"; do [ -f "media/videos/${NAME}/${RES}/${S}.mp4" ] || { echo "missing $S"; exit 1; }; echo "file '../videos/${NAME}/${RES}/${S}.mp4'" >> "$LIST"; done
  ffmpeg -y -loglevel error -f concat -safe 0 -i "$LIST" -c copy "media/final/TexturedGaussians_${NAME}_${RES}.mp4"
  echo "wrote media/final/TexturedGaussians_${NAME}_${RES}.mp4"
fi
