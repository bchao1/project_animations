#!/usr/bin/env bash
# Usage: ./render.sh [l|h] [script] [Scene ...]
#   l = 480p15 test (default), h = 1080p60 final.
# Renders each scene into media/ (in parallel) and concatenates all scenes into media/final/.
set -euo pipefail
cd "$(dirname "$0")"
Q="${1:-l}"; shift || true
SCRIPT="${1:-$(ls animation_fisheye_v*.py | sort -V | tail -1)}"; shift || true
ALL=(S00_Title S01_Hook S02_Projection S03_Flow S04_Data S05_GAN S06_Rotation S07_Radial S08_Results S09_Outlook)
SCENES=("$@"); [ ${#SCENES[@]} -eq 0 ] && SCENES=("${ALL[@]}")
source ~/miniconda3/etc/profile.d/conda.sh
conda activate imp_tome
# imp_tome ships libffi 8 only; system pango/gobject needs libffi.so.7
export LD_PRELOAD=/lib/x86_64-linux-gnu/libffi.so.7
NAME="$(basename "$SCRIPT" .py)"
if [ "$Q" = "h" ]; then FLAG=-qh; RES=1080p60; else FLAG=-ql; RES=480p15; fi
# build the cached toy assets (numpy warps) once, before the parallel renders
python -c "import sys; sys.argv=['x']; import importlib.util as u; s=u.spec_from_file_location('m', '$SCRIPT'); m=u.module_from_spec(s); s.loader.exec_module(m)"
for S in "${SCENES[@]}"; do
  manim $FLAG --media_dir media "$SCRIPT" "$S" > "media/log_${NAME}_${RES}_${S}.txt" 2>&1 &
done
FAIL=0
for job in $(jobs -p); do wait "$job" || FAIL=1; done
[ $FAIL -eq 0 ] || { echo "a scene failed; see media/log_${NAME}_${RES}_*.txt"; exit 1; }
if [ ${#SCENES[@]} -eq ${#ALL[@]} ]; then
  mkdir -p media/final
  LIST=media/final/list_${NAME}_${RES}.txt; : > "$LIST"
  for S in "${ALL[@]}"; do [ -f "media/videos/${NAME}/${RES}/${S}.mp4" ] || { echo "missing $S"; exit 1; }; echo "file '../videos/${NAME}/${RES}/${S}.mp4'" >> "$LIST"; done
  ffmpeg -y -loglevel error -f concat -safe 0 -i "$LIST" -c copy "media/final/FisheyeRectification_${NAME}_${RES}.mp4"
  echo "wrote media/final/FisheyeRectification_${NAME}_${RES}.mp4"
fi
