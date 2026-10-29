#!/usr/bin/env bash
# Usage: ./render.sh [l|h] [script] [Scene ...]
#   l = 480p15 test (default), h = 1080p60 final.
# Renders each scene into media/ (in parallel) and, when all scenes are rendered,
# concatenates them into media/final/.
set -euo pipefail
cd "$(dirname "$0")"
Q="${1:-l}"; shift || true
SCRIPT="${1:-animation_spectral_progressive_diffusion_1min_v5.py}"; shift || true
ALL=(T1_Hook T2_Insight T3_Method T4_Payoff)
SCENES=("$@"); [ ${#SCENES[@]} -eq 0 ] && SCENES=("${ALL[@]}")
source ~/miniconda3/etc/profile.d/conda.sh
conda activate imp_tome
# imp_tome ships libffi 8 only; system pango/gobject needs libffi.so.7
export LD_PRELOAD=/lib/x86_64-linux-gnu/libffi.so.7
NAME="$(basename "$SCRIPT" .py)"; VER="${NAME##*_}"
if [ "$Q" = "h" ]; then FLAG=-qh; RES=1080p60; else FLAG=-ql; RES=480p15; fi
mkdir -p scratch/logs
for S in "${SCENES[@]}"; do
  manim $FLAG --media_dir media "$SCRIPT" "$S" > "scratch/logs/${NAME}_${RES}_${S}.log" 2>&1 &
done
wait
if [ ${#SCENES[@]} -eq ${#ALL[@]} ]; then
  mkdir -p media/final
  LIST=media/final/list_${RES}.txt; : > "$LIST"
  for S in "${ALL[@]}"; do [ -f "media/videos/${NAME}/${RES}/${S}.mp4" ] || { echo "missing $S (see scratch/logs)"; exit 1; }; echo "file '../videos/${NAME}/${RES}/${S}.mp4'" >> "$LIST"; done
  ffmpeg -y -loglevel error -f concat -safe 0 -i "$LIST" -c copy "media/final/SpectralProgressiveDiffusion_1min_${VER}_${RES}.mp4"
  echo "wrote media/final/SpectralProgressiveDiffusion_1min_${VER}_${RES}.mp4"
fi
