#!/usr/bin/env bash
# V3 v3 (post 3): the user's four budget_slider v4 clips (v2 order: deer, mountain, tulip, skyline), each played
# backwards so the budget steps down 4,096 → 1,024 (the clips have no intro / outro fades and the morphs are
# symmetric crossfades, so reversal is clean), 0.3-s crossfades. Usage: ./make_v3.sh → ../media/V3_budget_v3_1080p60.mp4
set -euo pipefail
cd "$(dirname "$0")"
B=../../budget_slider/media/final/1080p60
FF=${FF:-$HOME/miniconda3/envs/flex/bin/ffmpeg}   # system ffmpeg has no xfade, flex has no libx264: xfade there, encode with the system one
X=0.3; L=5.0
$FF -v error -y -i $B/LoT_budget_F10_1_Deer.mp4 -i $B/LoT_budget_F10_4_Mountain.mp4 -i $B/LoT_budget_F13_4_Flower.mp4 \
  -i $B/LoT_budget_F12_2_Tree.mp4 -filter_complex \
  "[0]reverse[r0];[1]reverse[r1];[2]reverse[r2];[3]reverse[r3];[r0][r1]xfade=transition=fade:duration=$X:offset=$(echo "$L-$X" | bc)[a];[a][r2]xfade=transition=fade:duration=$X:offset=$(echo "2*$L-2*$X" | bc)[b];[b][r3]xfade=transition=fade:duration=$X:offset=$(echo "3*$L-3*$X" | bc),format=yuv420p[v]" \
  -map "[v]" -c:v rawvideo -an -f nut - | ffmpeg -v error -y -i - -c:v libx264 -crf 16 -preset slow -pix_fmt yuv420p -r 60 ../media/V3_budget_v3_1080p60.mp4
echo "wrote ../media/V3_budget_v3_1080p60.mp4"
