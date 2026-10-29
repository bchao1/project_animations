#!/bin/bash
while kill -0 1413893 2>/dev/null; do sleep 10; done
cd /home/brianchc/prompt-adaptive-speed-cvpr-2027
PY=/home/brianchc/miniconda3/envs/imp_tome/bin/python
S=/home/brianchc/project_animations/spectral_frontier_sampling/scratch
CUDA_VISIBLE_DEVICES=6 HF_HUB_OFFLINE=1 PYTHONPATH=$PWD $PY -W ignore $S/run_spec.py --prompt 2348 --src full > $S/spec_2348.log 2>&1
for job in "2348 cont" "2580 cont" "1026 cont"; do set -- $job
  CUDA_VISIBLE_DEVICES=6 HF_HUB_OFFLINE=1 PYTHONPATH=$PWD $PY -W ignore $S/capture.py --prompt $1 --src $2 > $S/capture_$1_$2.log 2>&1
done
echo ALLDONE > $S/q3.done
