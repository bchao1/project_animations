#!/usr/bin/env bash
# Usage: ./render.sh [l|h] [script]   l = 480p15 test (default), h = 1080p60 final → media/final/LoT_settle2d_<vN>_<res>.mp4
# Physics (sim2d_v3.py, GPU) and tile frames (render_tiles_v3.py, CPU pool) are built first if missing.
set -euo pipefail
cd "$(dirname "$0")"
Q="${1:-l}"; SCRIPT="${2:-animation_lot_settle2d_v3.py}"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate imp_tome
# imp_tome ships libffi 8 only; system pango/gobject needs libffi.so.7
export LD_PRELOAD=/lib/x86_64-linux-gnu/libffi.so.7
[ -f data/generated/physics2d_v3.npz ] || CUDA_VISIBLE_DEVICES=${SIM_GPU:-5} python sim2d_v3.py
[ "$(ls data/generated/tiles_v3 2>/dev/null | wc -l)" -ge 870 ] || python render_tiles_v3.py
NAME="$(basename "$SCRIPT" .py)"; TAG="${NAME#animation_lot_settle2d_}"
if [ "$Q" = "h" ]; then FLAG=-qh; RES=1080p60; else FLAG=-ql; RES=480p15; fi
manim $FLAG --media_dir media "$SCRIPT" LoTSettle2D
mkdir -p media/final
cp "media/videos/${NAME}/${RES}/LoTSettle2D.mp4" "media/final/LoT_settle2d_${TAG}_${RES}.mp4"
echo "wrote media/final/LoT_settle2d_${TAG}_${RES}.mp4"
