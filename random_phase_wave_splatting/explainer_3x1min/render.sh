#!/usr/bin/env bash
# Usage: ./render.sh [l|h] [script] [Scene ...]
#   l = 480p15 test (default), h = 1080p60 final.
# Renders scenes in parallel; with no scene list also writes one MP4 per part and the 3-min cut to media/final/.
set -euo pipefail
cd "$(dirname "$0")"
Q="${1:-l}"; shift || true
SCRIPT="${1:-$(ls animation_rpws_explainer_v*.py | sort -V | tail -1)}"; shift || true
ALL=(S00_Title P1_Difference P2_AlphaBlending P3_BackToFront)
SCENES=("$@"); [ ${#SCENES[@]} -eq 0 ] && SCENES=("${ALL[@]}")
source ~/miniconda3/etc/profile.d/conda.sh
conda activate imp_tome
export LD_PRELOAD=/lib/x86_64-linux-gnu/libffi.so.7   # imp_tome ships libffi 8 only
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3}
# build cached assets once before the parallel renders (runs sims.py on first use)
python -c "import importlib.util; s=importlib.util.spec_from_file_location('m','$SCRIPT'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)" >/dev/null
NAME="$(basename "$SCRIPT" .py)"
if [ "$Q" = "h" ]; then FLAG=-qh; RES=1080p60; else FLAG=-ql; RES=480p15; fi
mkdir -p scratch/logs
for S in "${SCENES[@]}"; do
  manim $FLAG --media_dir media "$SCRIPT" "$S" > "scratch/logs/${NAME}_${RES}_${S}.log" 2>&1 || echo "FAILED $S (see scratch/logs)" &
done
wait
if [ ${#SCENES[@]} -eq ${#ALL[@]} ]; then
  mkdir -p media/final
  V="media/videos/${NAME}/${RES}"
  for S in "${ALL[@]}"; do [ -f "$V/$S.mp4" ] || { echo "missing $S"; exit 1; }; done
  for k in 1 2 3; do
    S="${ALL[$k]}"; cp "$V/$S.mp4" "media/final/RPWS_Part${k}_${NAME}_${RES}.mp4"
  done
  LIST=media/final/list_${RES}.txt; : > "$LIST"
  for S in "${ALL[@]}"; do echo "file '../videos/${NAME}/${RES}/${S}.mp4'" >> "$LIST"; done
  ffmpeg -y -loglevel error -f concat -safe 0 -i "$LIST" -c copy "media/final/RPWS_3min_${NAME}_${RES}.mp4"
  for f in media/final/RPWS_*_${NAME}_${RES}.mp4; do
    printf "%s  %.1f s\n" "$f" "$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$f")"
  done
fi
