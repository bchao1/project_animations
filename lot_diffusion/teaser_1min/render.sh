#!/usr/bin/env bash
# Usage: ./render.sh [l|h] [script.py] [Scene ...]
#   l = 480p15 test (default), h = 1080p60 final. Renders scenes in parallel, one media dir per scene (parallel
#   manim processes race on a shared text-SVG cache otherwise), and concatenates all scenes into media/final/.
set -euo pipefail
cd "$(dirname "$0")"
Q="${1:-l}"; shift || true
SCRIPT="$(ls animation_lot_1min_v*.py | sort -V | tail -1)"
if [[ "${1:-}" == *.py ]]; then SCRIPT="$1"; shift; fi
ALL=(S1_Insight S2_Method S3_Payoff)
SCENES=("$@"); [ ${#SCENES[@]} -eq 0 ] && SCENES=("${ALL[@]}")
source ~/miniconda3/etc/profile.d/conda.sh
conda activate imp_tome
export LD_PRELOAD=/lib/x86_64-linux-gnu/libffi.so.7   # imp_tome ships libffi 8; system pango needs libffi.so.7
NAME="$(basename "$SCRIPT" .py)"
TAG="${NAME#animation_lot_1min_}"
if [ "$Q" = "h" ]; then FLAG=-qh; RES=1080p60; else FLAG=-ql; RES=480p15; fi
python -c "import importlib.util as u; s=u.spec_from_file_location('m', '$SCRIPT'); m=u.module_from_spec(s); s.loader.exec_module(m)" > /dev/null 2>&1
mkdir -p media/logs
for S in "${SCENES[@]}"; do
  manim $FLAG --media_dir "media/scenes/$S" "$SCRIPT" "$S" > "media/logs/${NAME}_${RES}_${S}.txt" 2>&1 &
done
FAIL=0
for job in $(jobs -p); do wait "$job" || FAIL=1; done
[ $FAIL -eq 0 ] || { echo "a scene failed; see media/logs/${NAME}_${RES}_*.txt"; exit 1; }
mkdir -p "media/videos/${NAME}/${RES}"
for S in "${SCENES[@]}"; do cp "media/scenes/$S/videos/${NAME}/${RES}/${S}.mp4" "media/videos/${NAME}/${RES}/"; done
if [ ${#SCENES[@]} -eq ${#ALL[@]} ]; then
  mkdir -p media/final
  LIST=media/final/list_${NAME}_${RES}.txt; : > "$LIST"
  for S in "${ALL[@]}"; do echo "file '../videos/${NAME}/${RES}/${S}.mp4'" >> "$LIST"; done
  ffmpeg -y -loglevel error -f concat -safe 0 -i "$LIST" -c copy "media/final/LevelOfTokenDiffusion_1min_${TAG}_${RES}.mp4"
  echo "wrote media/final/LevelOfTokenDiffusion_1min_${TAG}_${RES}.mp4"
fi
