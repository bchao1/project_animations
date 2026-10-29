#!/usr/bin/env bash
# Re-render the v6 scenes whose images changed (1080p60, one mp4 per section).
set -euo pipefail
cd "$(dirname "$0")/.."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate imp_tome
export LD_PRELOAD=/lib/x86_64-linux-gnu/libffi.so.7
for S in S01_TokenCost S02_Vision S05_Naive S08_Masks S09_Results; do
  manim -qh --save_sections --media_dir media animation_foveated_diffusion_v6.py "$S" > "slides_v6/logs/$S.log" 2>&1 &
done
wait
echo done
