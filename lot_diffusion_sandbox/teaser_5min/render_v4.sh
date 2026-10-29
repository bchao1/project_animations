#!/usr/bin/env bash
# Usage: ./render_v4.sh [l|h] [Scene ...]   (v4 slide clips: --save_sections, scenes S02–S08 by default)
#   → media/videos/animation_lot_5min_v4/<res>/sections/<Scene>_<NNNN>_<name>.mp4 (+ <Scene>.json)
set -euo pipefail
cd "$(dirname "$0")"
Q="${1:-l}"; shift || true
SCRIPT=animation_lot_5min_v4.py
ALL=(S02_Known S03_Layout S04_Naive S05_AsymFlow S06_DiT S07_Sources S08_Results)
SCENES=("$@"); [ ${#SCENES[@]} -eq 0 ] && SCENES=("${ALL[@]}")
source ~/miniconda3/etc/profile.d/conda.sh
conda activate imp_tome
export LD_PRELOAD=/lib/x86_64-linux-gnu/libffi.so.7
if [ "$Q" = "h" ]; then FLAG=-qh; RES=1080p60; else FLAG=-ql; RES=480p15; fi
python -c "import importlib.util as u; s=u.spec_from_file_location('m', '$SCRIPT'); s.loader.exec_module(u.module_from_spec(s))"
mkdir -p scratch/logs
for S in "${SCENES[@]}"; do
  manim $FLAG --save_sections --media_dir media "$SCRIPT" "$S" > "scratch/logs/animation_lot_5min_v4_${RES}_${S}.log" 2>&1 &
done
wait
ls media/videos/animation_lot_5min_v4/$RES/sections/
