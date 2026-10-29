#!/usr/bin/env bash
# Usage: ./render.sh [l|h] [script]   l = 480p15 test (default), h = 1080p60 → ../media/V2_quadtree_settle_<vN>_<res>.mp4
# Needs ../../spacetime_clip/data/generated/vol11c_31/ (render_v16.sh h image 31); corgi physics + frames built if missing.
set -euo pipefail
cd "$(dirname "$0")"
Q="${1:-l}"; SCRIPT="${2:-animation_quadtree_settle_v8.py}"; NAME="$(basename "$SCRIPT" .py)"; TAG="${NAME#animation_quadtree_settle_}"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate imp_tome
export LD_PRELOAD=/lib/x86_64-linux-gnu/libffi.so.7      # imp_tome ships libffi 8 only; pango needs .so.7
[ -f data/generated/physics_left_v1.npz ] || CUDA_VISIBLE_DEVICES=${SIM_GPU:-0} python sim_left_v1.py
FV=$(case "$TAG" in v3|v4|v5|v6|v7|v8) echo "$TAG";; v7) echo v6;; *) echo v1;; esac)   # v7 = v6 frames
[ -n "$(ls data/generated/frames_$FV 2>/dev/null | tail -1)" ] && [ "$(ls data/generated/frames_$FV | wc -l)" -ge 1008 ] || python render_frames_$FV.py
if [ "$Q" = "h" ]; then FLAG=-qh; RES=1080p60; else FLAG=-ql; RES=480p15; fi
manim $FLAG --media_dir media "$SCRIPT" LoTQuadtreeSettle
OUT="../media/V2_quadtree_settle_${TAG}_${RES}.mp4"
cp "media/videos/${NAME}/${RES}/LoTQuadtreeSettle.mp4" "$OUT"
echo "wrote $OUT"
