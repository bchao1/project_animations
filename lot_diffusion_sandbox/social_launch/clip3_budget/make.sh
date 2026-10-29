#!/usr/bin/env bash
# V3 (post 3): three budget_slider clips (mask car, texture-variance skyline, texture-variance tulip), 0.3-s crossfades.
# Usage: ./make.sh → ../media/V3_budget_1080p60.mp4
set -euo pipefail
cd "$(dirname "$0")"
B=../../budget_slider/media/final/1080p60
X=0.3; L=5.0
FF=${FF:-$HOME/miniconda3/envs/flex/bin/ffmpeg}   # system ffmpeg has no xfade, flex has no libx264: xfade there, encode with the system one
$FF -v error -y -i $B/LoT_budget_F11_4_Car.mp4 -i $B/LoT_budget_F12_2_Tree.mp4 -i $B/LoT_budget_F13_4_Flower.mp4 \
  -filter_complex "[0][1]xfade=transition=fade:duration=$X:offset=$(echo "$L-$X" | bc)[a];[a][2]xfade=transition=fade:duration=$X:offset=$(echo "2*$L-2*$X" | bc),format=yuv420p[v]" \
  -map "[v]" -c:v rawvideo -an -f nut - | ffmpeg -v error -y -i - -c:v libx264 -crf 16 -preset slow -pix_fmt yuv420p -r 60 ../media/V3_budget_1080p60.mp4
echo "wrote ../media/V3_budget_1080p60.mp4"
