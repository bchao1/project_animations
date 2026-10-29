#!/usr/bin/env bash
# Usage: scratch/render_v7.sh [l|h] Scene ...   — renders v7 scenes in parallel with --save_sections.
set -uo pipefail
cd "$(dirname "$0")/.."
Q="$1"; shift
source ~/miniconda3/etc/profile.d/conda.sh
conda activate imp_tome
export LD_PRELOAD=/lib/x86_64-linux-gnu/libffi.so.7
if [ "$Q" = "h" ]; then FLAG=-qh; RES=1080p60; else FLAG=-ql; RES=480p15; fi
mkdir -p scratch/logs
for S in "$@"; do
  manim $FLAG --save_sections --media_dir media animation_spectral_progressive_v7.py "$S" > "scratch/logs/animation_spectral_progressive_v7_${RES}_${S}.log" 2>&1 &
done
wait
for S in "$@"; do tail -n 2 "scratch/logs/animation_spectral_progressive_v7_${RES}_${S}.log"; done
