#!/usr/bin/env bash
# Usage: ./render.sh [l|h] [script] [Scene ...]
#   l = 480p15 test (default), h = 1080p60 final.
# Renders each clip into media/scenes/<Scene>/ in parallel (separate text caches) and copies it to media/final/<res>/LoT_budget_<Scene>.mp4.
set -euo pipefail
cd "$(dirname "$0")"
Q="${1:-l}"; shift || true
SCRIPT="${1:-animation_lot_budget_v4.py}"; shift || true
ALL=(F10_1_Deer F10_2_Mountain F10_3_Building F10_4_Mountain F11_1_Building F11_2_Mountain F11_3_Mountain F11_4_Car
     F12_1_Mountain F12_2_Tree F12_3_Mountain F12_4_Mountain F13_1_Mountain F13_2_Mountain F13_3_Mountain F13_4_Flower)
SCENES=("$@"); [ ${#SCENES[@]} -eq 0 ] && SCENES=("${ALL[@]}")
source ~/miniconda3/etc/profile.d/conda.sh
conda activate imp_tome
# imp_tome ships libffi 8 only; system pango/gobject needs libffi.so.7
export LD_PRELOAD=/lib/x86_64-linux-gnu/libffi.so.7
NAME="$(basename "$SCRIPT" .py)"
if [ "$Q" = "h" ]; then FLAG=-qh; RES=1080p60; else FLAG=-ql; RES=480p15; fi
# build the cached assets once before the parallel renders
python -c "import importlib.util as u; s=u.spec_from_file_location('m', '$SCRIPT'); s.loader.exec_module(u.module_from_spec(s))"
mkdir -p scratch/logs "media/final/${RES}"
for S in "${SCENES[@]}"; do
  ( manim $FLAG --media_dir "media/scenes/$S" "$SCRIPT" "$S" > "scratch/logs/${NAME}_${RES}_${S}.log" 2>&1 \
    && cp "media/scenes/${S}/videos/${NAME}/${RES}/${S}.mp4" "media/final/${RES}/LoT_budget_${S}.mp4" ) &
done
wait
ls media/final/${RES}/
