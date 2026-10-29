#!/usr/bin/env bash
# Usage: ./render.sh [l|h] [script]   l = 480p15 test (default), h = 1080p60 final → media/final/LoT_spacetime_<vN>_<res>.mp4
# The 3D volume frames (data/generated/vol9_<item>/) are rendered first on the GPU by render_volume_v9.py (EGL).
set -euo pipefail
cd "$(dirname "$0")"
Q="${1:-l}"; SCRIPT="${2:-animation_lot_spacetime_v14.py}"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate imp_tome
# imp_tome ships libffi 8 only; system pango/gobject needs libffi.so.7
export LD_PRELOAD=/lib/x86_64-linux-gnu/libffi.so.7
[ -f data/generated/physics2_${LOT_ITEM:-31}.npz ] || CUDA_VISIBLE_DEVICES=${SIM_GPU:-5} python sim_physics_v2.py
VOLDIR=data/generated/vol9_${LOT_ITEM:-31}
if [ "$(ls "$VOLDIR" 2>/dev/null | wc -l)" -lt 828 ]; then
  for c in "0 206" "207 413" "414 620" "621 827"; do python render_volume_v9.py $c & done; wait
fi
NAME="$(basename "$SCRIPT" .py)"; TAG="${NAME#animation_lot_spacetime_}"
if [ "$Q" = "h" ]; then FLAG=-qh; RES=1080p60; else FLAG=-ql; RES=480p15; fi
manim $FLAG --media_dir media "$SCRIPT" LoTSpaceTime
mkdir -p media/final
cp "media/videos/${NAME}/${RES}/LoTSpaceTime.mp4" "media/final/LoT_spacetime_${TAG}_${RES}.mp4"
echo "wrote media/final/LoT_spacetime_${TAG}_${RES}.mp4"
