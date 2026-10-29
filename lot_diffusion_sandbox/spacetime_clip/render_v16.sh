#!/usr/bin/env bash
# Usage: ./render_v16.sh [l|h] [extent|image] [item]   → media/final/LoT_spacetime_v16[_color]_item<k>_<res>.mp4
# physics v3 (fluid from the first landing) → volume (render_volume_v11.py, 4 chunks) → manim v16.
set -euo pipefail
cd "$(dirname "$0")"
Q="${1:-l}"; export LOT_COLOR="${2:-extent}"; export LOT_ITEM="${3:-31}"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate imp_tome
export LD_PRELOAD=/lib/x86_64-linux-gnu/libffi.so.7
[ -f data/generated/physics3_${LOT_ITEM}.npz ] || CUDA_VISIBLE_DEVICES=${SIM_GPU:-5} python sim_physics_v3.py
SUF=$([ "$LOT_COLOR" = image ] && echo c || true)
VOLDIR=data/generated/vol11${SUF}_${LOT_ITEM}
if [ "$(ls "$VOLDIR" 2>/dev/null | wc -l)" -lt 828 ]; then
  for c in "0 206" "207 413" "414 620" "621 827"; do LOT_EGL_DEVICE=${EGL_GPU:-5} python render_volume_v11.py $c & done; wait
fi
if [ "$Q" = "h" ]; then FLAG=-qh; RES=1080p60; else FLAG=-ql; RES=480p15; fi
MD=media/v16/${LOT_COLOR}_${LOT_ITEM}
manim $FLAG --media_dir $MD animation_lot_spacetime_v16.py LoTSpaceTime
mkdir -p media/final
OUT=media/final/LoT_spacetime_v16$([ "$LOT_COLOR" = image ] && echo _color || true)_item${LOT_ITEM}_${RES}.mp4
cp $MD/videos/animation_lot_spacetime_v16/$RES/LoTSpaceTime.mp4 $OUT
echo "wrote $OUT"
