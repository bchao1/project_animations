#!/usr/bin/env bash
# Usage: ./render.sh [l|h] [teaser|method|Scene ...] [--script file.py]
#   l = 480p15 test (default), h = 1080p60 final.
# Renders the scenes in parallel into media/ and, for a full set (teaser or method), concatenates them
# into media/final/.
set -euo pipefail
cd "$(dirname "$0")"
Q="${1:-l}"; shift || true
SCRIPT=animation_sfs_v2.py
TEASER=(T00_Title T01_Spectrogram T02_Waste T03_PerSample T04_End)
METHOD=(M00_Title M01_Primer M02_Grid M03_Problem M04_Estimate M05_Rule M06_Tracking M07_Loop)
ARGS=(); for a in "$@"; do ARGS+=("$a"); done
SCENES=(); SET=""
i=0
while [ $i -lt ${#ARGS[@]} ]; do
  a="${ARGS[$i]}"
  case "$a" in
    --script) i=$((i+1)); SCRIPT="${ARGS[$i]}";;
    teaser) SCENES+=("${TEASER[@]}"); SET=teaser;;
    method) SCENES+=("${METHOD[@]}"); SET=method;;
    *) SCENES+=("$a");;
  esac
  i=$((i+1))
done
[ ${#SCENES[@]} -eq 0 ] && { SCENES=("${TEASER[@]}"); SET=teaser; }
source ~/miniconda3/etc/profile.d/conda.sh
conda activate imp_tome
# imp_tome ships libffi 8 only; system pango/gobject needs libffi.so.7
export LD_PRELOAD=/lib/x86_64-linux-gnu/libffi.so.7
NAME="$(basename "$SCRIPT" .py)"
if [ "$Q" = "h" ]; then FLAG=-qh; RES=1080p60; else FLAG=-ql; RES=480p15; fi
mkdir -p scratch/logs
for S in "${SCENES[@]}"; do
  manim $FLAG --media_dir media "$SCRIPT" "$S" > "scratch/logs/${NAME}_${RES}_${S}.log" 2>&1 &
done
FAIL=0; for job in $(jobs -p); do wait "$job" || FAIL=1; done
[ $FAIL -eq 1 ] && echo "some scenes failed (see scratch/logs)"
if [ -n "$SET" ]; then
  if [ "$SET" = teaser ]; then LISTED=("${TEASER[@]}"); TAG=Teaser; else LISTED=("${METHOD[@]}"); TAG=Method; fi
  mkdir -p media/final
  LIST=media/final/list_${SET}_${RES}.txt; : > "$LIST"
  for S in "${LISTED[@]}"; do [ -f "media/videos/${NAME}/${RES}/${S}.mp4" ] || { echo "missing $S (see scratch/logs)"; exit 1; }; echo "file '../videos/${NAME}/${RES}/${S}.mp4'" >> "$LIST"; done
  ffmpeg -y -loglevel error -f concat -safe 0 -i "$LIST" -c copy "media/final/SFS_${TAG}_${NAME}_${RES}.mp4"
  echo "wrote media/final/SFS_${TAG}_${NAME}_${RES}.mp4"
fi
