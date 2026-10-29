#!/usr/bin/env bash
# Usage: ./render.sh [l|h] [v1|v2|v3]   l = 15-fps test, h = 60 fps → media/LoT_website_<vN>_<res>.mp4 (1024 × 1024)
set -euo pipefail
cd "$(dirname "$0")"
Q="${1:-l}"; V="${2:-v1}"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate imp_tome
export LD_PRELOAD=/lib/x86_64-linux-gnu/libffi.so.7      # imp_tome ships libffi 8 only; pango needs .so.7
DUR=$(case "$V" in v1|v2) echo 5;; *) echo 8;; esac)   # seconds
[ "$(ls data/generated/frames_$V 2>/dev/null | wc -l)" -ge $((DUR * 60)) ] || python render_frames_$V.py
if [ "$Q" = "h" ]; then FLAG=-qh; RES=60fps; else FLAG=-ql; RES=15fps; fi
manim $FLAG --media_dir media/manim animation_web_$V.py LoTWebsite
SRC=$(ls -t media/manim/videos/animation_web_$V/*/LoTWebsite.mp4 | head -1)
cp "$SRC" "media/LoT_website_${V}_${RES}.mp4"
echo "wrote media/LoT_website_${V}_${RES}.mp4"
# web copy: exactly DUR s, faststart for <video> streaming
[ "$Q" = "h" ] && ffmpeg -v error -y -i "media/LoT_website_${V}_${RES}.mp4" -t "$DUR" -c:v libx264 -crf 18 -preset slow \
  -pix_fmt yuv420p -movflags +faststart -an "media/LoT_website_${V}.mp4" && echo "wrote media/LoT_website_${V}.mp4"
