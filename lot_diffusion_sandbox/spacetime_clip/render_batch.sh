#!/usr/bin/env bash
# Usage: ./render_batch.sh [l|h] [item ...]   default items: the 10-video batch; h = 1080p60 (default l = 480p15)
# Per item: physics (sim_physics_v2.py, GPUs round-robin) → volume (render_volume_v10.py, 4 chunks on one GPU)
# → manim (animation_lot_spacetime_v15.py, own media dir) → media/final/batch/LoT_spacetime_v15_item<k>_<res>.mp4
set -uo pipefail
cd "$(dirname "$0")"
Q="${1:-l}"; shift || true
ITEMS=("$@"); [ ${#ITEMS[@]} -eq 0 ] && ITEMS=(0 3 4 10 14 16 22 26 37 39)
GPUS=(${BATCH_GPUS:-1 5 7 8})
source ~/miniconda3/etc/profile.d/conda.sh
conda activate imp_tome
export LD_PRELOAD=/lib/x86_64-linux-gnu/libffi.so.7
if [ "$Q" = "h" ]; then FLAG=-qh; RES=1080p60; else FLAG=-ql; RES=480p15; fi
mkdir -p scratch/logs media/final/batch
i=0
for K in "${ITEMS[@]}"; do
  G=${GPUS[$((i % ${#GPUS[@]}))]}; i=$((i + 1))
  (
    [ -f data/generated/physics2_${K}.npz ] || \
      LOT_ITEM=$K CUDA_VISIBLE_DEVICES=$G python sim_physics_v2.py > scratch/logs/sim_${K}.log 2>&1
    if [ "$(ls data/generated/vol9_${K} 2>/dev/null | wc -l)" -lt 828 ]; then
      for c in "0 206" "207 413" "414 620" "621 827"; do
        LOT_ITEM=$K LOT_EGL_DEVICE=$G python render_volume_v10.py $c >> scratch/logs/vol_${K}.log 2>&1 &
      done; wait
    fi
    LOT_ITEM=$K manim $FLAG --media_dir media/batch/item$K animation_lot_spacetime_v15.py LoTSpaceTime \
      > scratch/logs/manim_${K}_${RES}.log 2>&1 && \
    cp media/batch/item$K/videos/animation_lot_spacetime_v15/$RES/LoTSpaceTime.mp4 \
       media/final/batch/LoT_spacetime_v15_item$(printf %02d $K)_$RES.mp4 && echo "item $K done" || echo "item $K FAILED"
  ) &
done
wait
ls media/final/batch/
