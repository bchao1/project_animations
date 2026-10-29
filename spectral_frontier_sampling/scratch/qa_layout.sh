#!/usr/bin/env bash
# Layout check for every scene (no rendering): ./scratch/qa_layout.sh [Scene ...] -> scratch/qa/layout_<Scene>.txt
cd "$(dirname "$0")/.."
source ~/miniconda3/etc/profile.d/conda.sh; conda activate imp_tome
export LD_PRELOAD=/lib/x86_64-linux-gnu/libffi.so.7 SFS_QA=1
ALL=(T00_Title T01_Spectrogram T02_Waste T03_PerSample T04_End M00_Title M01_Primer M02_Grid M03_Problem M04_Estimate M05_Rule M06_Tracking M07_Loop)
S=("$@"); [ ${#S[@]} -eq 0 ] && S=("${ALL[@]}")
for s in "${S[@]}"; do rm -f scratch/qa/layout_$s.txt; (manim --dry_run --media_dir scratch/qa_media animation_sfs_v2.py $s > scratch/logs/qa_$s.log 2>&1 || echo "RUN FAIL $s") & done; wait
for s in "${S[@]}"; do n=$( [ -f scratch/qa/layout_$s.txt ] && wc -l < scratch/qa/layout_$s.txt || echo 0); echo "$s: $n issues"; done
