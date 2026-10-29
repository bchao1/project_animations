#!/usr/bin/env bash
# Usage: ./render.sh [l|h] [script]   l = 480p15 test (default), h = 1080p60 final → media/final/LoT_quadtree_<vN>_<res>.mp4
set -euo pipefail
cd "$(dirname "$0")"
Q="${1:-l}"; SCRIPT="${2:-animation_lot_quadtree_v2.py}"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate imp_tome
# imp_tome ships libffi 8 only; system pango/gobject needs libffi.so.7
export LD_PRELOAD=/lib/x86_64-linux-gnu/libffi.so.7
NAME="$(basename "$SCRIPT" .py)"; TAG="${NAME#animation_lot_quadtree_}"
if [ "$Q" = "h" ]; then FLAG=-qh; RES=1080p60; else FLAG=-ql; RES=480p15; fi
manim $FLAG --media_dir media "$SCRIPT" LoTQuadtree
mkdir -p media/final
cp "media/videos/${NAME}/${RES}/LoTQuadtree.mp4" "media/final/LoT_quadtree_${TAG}_${RES}.mp4"
echo "wrote media/final/LoT_quadtree_${TAG}_${RES}.mp4"
