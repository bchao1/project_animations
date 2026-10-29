#!/usr/bin/env bash
# Usage: ./render.sh [l|h|s] [script] [Scene ...]
#   l = 480p15 test (default), h = 1080p60 final, s = 1080p60 with --save_sections (one mp4 per beat, for slides).
#   script defaults to the highest-numbered animation_*_vN.py in this folder.
#   With no scene list it renders every S* scene of the script in parallel (logs in scratch/logs/)
#   and, for l and h, concatenates them in file order into media/final/<PREFIX>_<script>_<res>.mp4.
# Environment:
#   MANIM_CONDA_ENV   conda env to activate (optional; else the current python)
#   MANIM_LD_PRELOAD  library to preload (optional; e.g. /lib/x86_64-linux-gnu/libffi.so.7, see references/pitfalls.md)
#   PREFIX            name of the final video (default: the folder name)
#   CAPTIONS=0        slide-deck mode for house.py (captions, headers and part cards off)
set -euo pipefail
cd "$(dirname "$0")"
Q="${1:-l}"; shift || true
SCRIPT="${1:-$(ls animation_*_v*.py | sort -V | tail -1)}"; shift || true
if [ -n "${MANIM_CONDA_ENV:-}" ]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$MANIM_CONDA_ENV"
fi
[ -n "${MANIM_LD_PRELOAD:-}" ] && export LD_PRELOAD="$MANIM_LD_PRELOAD"
mapfile -t ALL < <(grep -oP '^class \KS\w+(?=\()' "$SCRIPT")
SCENES=("$@"); [ ${#SCENES[@]} -eq 0 ] && SCENES=("${ALL[@]}")
NAME="$(basename "$SCRIPT" .py)"
PREFIX="${PREFIX:-$(basename "$PWD")}"
case "$Q" in
  h) FLAG=(-qh); RES=1080p60 ;;
  s) FLAG=(-qh --save_sections); RES=1080p60 ;;
  *) FLAG=(-ql); RES=480p15 ;;
esac
# build cached assets once, before the parallel renders race for them
python -c "import importlib.util; s=importlib.util.spec_from_file_location('m','$SCRIPT'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)" >/dev/null
mkdir -p scratch/logs
for S in "${SCENES[@]}"; do
  manim "${FLAG[@]}" --media_dir media "$SCRIPT" "$S" > "scratch/logs/${NAME}_${RES}_${S}.log" 2>&1 || echo "FAILED $S (see scratch/logs/${NAME}_${RES}_${S}.log)" &
done
wait
if [ "$Q" != "s" ] && [ ${#SCENES[@]} -eq ${#ALL[@]} ]; then
  mkdir -p media/final
  LIST="media/final/list_${NAME}_${RES}.txt"; : > "$LIST"
  for S in "${ALL[@]}"; do
    [ -f "media/videos/${NAME}/${RES}/${S}.mp4" ] || { echo "missing $S"; exit 1; }
    echo "file '../videos/${NAME}/${RES}/${S}.mp4'" >> "$LIST"
  done
  OUT="media/final/${PREFIX}_${NAME}_${RES}.mp4"
  ffmpeg -y -loglevel error -f concat -safe 0 -i "$LIST" -c copy -movflags +faststart "$OUT"
  echo "wrote $OUT"
fi
