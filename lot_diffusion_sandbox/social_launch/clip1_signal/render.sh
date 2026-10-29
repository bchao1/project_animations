#!/usr/bin/env bash
# Usage: ./render.sh [l|h] [v1…v6]   l = 480p15 test (default), h = 1080p60 → ../media/V1_signal_settle_<vN>_<res>.mp4
# Reuses settle2d v3 physics (../../settle2d_clip/data/generated/physics2d_v3.npz); tiles are built first if missing.
set -euo pipefail
cd "$(dirname "$0")"
Q="${1:-l}"; V="${2:-v2}"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate imp_tome
export LD_PRELOAD=/lib/x86_64-linux-gnu/libffi.so.7      # imp_tome ships libffi 8 only; pango needs .so.7
PV=$(case "$V" in v6|v7) echo v6;; *) echo v2;; esac)    # v6 / v7 = new items, own physics
[ "$V" = v1 ] || [ -f data/generated/physics_$PV.npz ] || CUDA_VISIBLE_DEVICES=${SIM_GPU:-0} python sim_$PV.py
TV=$([ "$V" = v5 ] && echo v4 || echo "$V")                 # v5 = v4 tiles, new text only
[ "$(ls data/generated/tiles_$TV 2>/dev/null | wc -l)" -ge 1014 ] || python render_tiles_$TV.py
if [ "$Q" = "h" ]; then FLAG=-qh; RES=1080p60; else FLAG=-ql; RES=480p15; fi
manim $FLAG --media_dir media animation_signal_settle_$V.py LoTSignalSettle
cp "media/videos/animation_signal_settle_$V/${RES}/LoTSignalSettle.mp4" "../media/V1_signal_settle_${V}_${RES}.mp4"
echo "wrote ../media/V1_signal_settle_${V}_${RES}.mp4"
