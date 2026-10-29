#!/usr/bin/env bash
# Usage: ./render.sh [l|h] [script.py] [Scene ...]
#   l = 480p15 test (default), h = 1080p60 final. Renders the listed clips (default: all) in parallel and copies each
#   finished clip to media/final/<res>/LoT_<Scene>.mp4 (one file per thread post).
set -euo pipefail
cd "$(dirname "$0")"
Q="${1:-l}"; shift || true
SCRIPT="$(ls animation_lot_clips_v*.py | sort -V | tail -1)"
if [[ "${1:-}" == *.py ]]; then SCRIPT="$1"; shift; fi
ALL=(C01_EqualCompute C02_Lineage C03_AnySize C04_Sources C05_AsymFlow C06_ShapeRoPE C07_Budget C08_Numbers
     R01_PaintGUI R02_Agent R03_Scene3D R04_Games R05_Driving R06_Robotics R07_DepthOfField R08_MoveTheBox
     R09_Typography R10_Boxes R11_VideoBaselines)
SCENES=("$@"); [ ${#SCENES[@]} -eq 0 ] && SCENES=("${ALL[@]}")
source ~/miniconda3/etc/profile.d/conda.sh
conda activate imp_tome
export LD_PRELOAD=/lib/x86_64-linux-gnu/libffi.so.7   # imp_tome ships libffi 8; system pango needs libffi.so.7
NAME="$(basename "$SCRIPT" .py)"
if [ "$Q" = "h" ]; then FLAG=-qh; RES=1080p60; else FLAG=-ql; RES=480p15; fi
mkdir -p media/final/$RES
for S in "${SCENES[@]}"; do
  # one media dir per clip: parallel manim processes race on the shared text-SVG cache otherwise
  ( manim $FLAG --media_dir "media/scenes/$S" "$SCRIPT" "$S" > "media/log_${NAME}_${RES}_${S}.txt" 2>&1 \
    && cp "media/scenes/$S/videos/${NAME}/${RES}/${S}.mp4" "media/final/${RES}/LoT_${S}.mp4" ) &
done
FAIL=0
for job in $(jobs -p); do wait "$job" || FAIL=1; done
[ $FAIL -eq 0 ] || { echo "a clip failed; see media/log_${NAME}_${RES}_*.txt"; exit 1; }
echo "wrote media/final/$RES/"
