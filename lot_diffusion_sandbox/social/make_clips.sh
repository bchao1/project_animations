#!/usr/bin/env bash
# Cuts the X-thread clips (one per master post in x_posts.md) from the finished 1080p60 teaser scenes.
# Usage: ./make_clips.sh   → clips/post{1..6}_*.mp4
set -euo pipefail
cd "$(dirname "$0")"
V1=../teaser_1min/media/videos/animation_lot_1min_v4/1080p60
V5=../teaser_5min/media/videos/animation_lot_5min_v3/1080p60
ENC="-c:v libx264 -preset slow -crf 18 -pix_fmt yuv420p -r 60 -movflags +faststart -an"
X=0.3   # crossfade between segments
mkdir -p clips

# one segment: in, start, end, out, with 0.2 s fade-in and 0.4 s fade-out
one() { local d; d=$(echo "$3 - $2" | bc)
  ffmpeg -v error -y -ss "$2" -to "$3" -i "$1" \
    -vf "setpts=PTS-STARTPTS,fade=t=in:st=0:d=0.2,fade=t=out:st=$(echo "$d - 0.4" | bc):d=0.4" $ENC "$4"; }

# two segments joined by a crossfade (alpha overlay: system ffmpeg 4.2 has no xfade): in1 s1 e1 in2 s2 e2 out
two() { local d1 d2 tot; d1=$(echo "$3 - $2" | bc); d2=$(echo "$6 - $5" | bc); tot=$(echo "$d1 + $d2 - $X" | bc)
  ffmpeg -v error -y -ss "$2" -to "$3" -i "$1" -ss "$5" -to "$6" -i "$4" -filter_complex \
    "[0:v]setpts=PTS-STARTPTS,tpad=stop_mode=clone:stop_duration=$d2[a];\
     [1:v]format=yuva420p,fade=t=in:st=0:d=$X:alpha=1,setpts=PTS-STARTPTS+($d1-$X)/TB[b];\
     [a][b]overlay=eof_action=pass,trim=duration=$tot,fade=t=in:st=0:d=0.2,fade=t=out:st=$(echo "$tot - 0.4" | bc):d=0.4[v]" -map "[v]" $ENC "$7"; }

# post 1 (hook): payoff first — layout → DiT → full-res corgi (1-min S2), then face vs sky at 4,096 tokens (1-min S1)
two $V1/S2_Method.mp4 13.0 15.2  $V1/S1_Insight.mp4 4.5 13.9   clips/post1_hook.mp4
# post 2 (key idea): painted detail map → merge passes → 1,546 of 4,096, 2.65× fewer (1-min S1)
one $V1/S1_Insight.mp4 13.8 27.4                                clips/post2_layout.mp4
# post 3 (naive fails / how): blocky toy vs LoT output (5-min S04), then DiT + size-specific head (1-min S2)
two $V5/S04_Naive.mp4 6.3 10.0   $V1/S2_Method.mp4 1.0 10.0      clips/post3_method.mp4
# post 4 (results): 2.04× / 3.53× with driving clip, then Fig. 7a at 1,269 tokens (1-min S3)
one $V1/S3_Payoff.mp4 0.4 13.3                                  clips/post4_results.mp4
# post 5 (neat property): one deer, four layout sources, compression · speedup (5-min S07)
one $V5/S07_Sources.mp4 2.5 15.0                                clips/post5_sources.mp4
# post 6 (applications): driving / robotics / game videos with their layouts (5-min S07)
one $V5/S07_Sources.mp4 15.8 21.8                               clips/post6_applications.mp4

# ---------------- quote threads: footage the master thread does not use ----------------
# site demo panel: agent-built detail map | LoT-Wan2.1 video, CMU Serif labels (text via files: no drawtext escaping)
CMU=/usr/share/texmf/fonts/opentype/public/lm/lmroman10-regular.otf   # Latin Modern = CMU design; this CMU Serif build lacks ×
demo() { local k=$1 dim=$2 out=$3 T=scratch/txt; mkdir -p $T
  printf '%s' "detail map from an agent-built Blender scene" > $T/l; printf '%s' "LoT-Wan2.1 14B output" > $T/r
  printf '%s' "An agent plans where the detail goes; the video model spends its tokens there." > $T/c; printf '%s' "$dim" > $T/d
  ffmpeg -v error -y -i ../web/video_demo_media/$k/detail-map.mp4 -i ../web/video_demo_media/$k/generated-video.mp4 -filter_complex \
    "color=black:s=1920x1080:r=60:d=5.4[bg];[0:v]fps=60,scale=880:-2[a];[1:v]fps=60,scale=880:-2[b];\
     [bg][a]overlay=60:250[t];[t][b]overlay=980:250:shortest=1,\
     drawtext=fontfile=$CMU:textfile=$T/l:fontsize=30:fontcolor=0xBBBBBB:x=60+(880-tw)/2:y=195,\
     drawtext=fontfile=$CMU:textfile=$T/r:fontsize=30:fontcolor=0xBBBBBB:x=980+(880-tw)/2:y=195,\
     drawtext=fontfile=$CMU:textfile=$T/c:fontsize=38:fontcolor=white:x=(w-tw)/2:y=880,\
     drawtext=fontfile=$CMU:textfile=$T/d:fontsize=26:fontcolor=0x888888:x=(w-tw)/2:y=790[v]" -map "[v]" $ENC "$out"; }
demo train "50,832 of 75,600 tokens · 1.49× fewer · supplementary website" scratch/demo_train.mp4
demo restaurant "32,058 of 75,600 tokens · 2.36× fewer · supplementary website" scratch/demo_restaurant.mp4

# Q1 @BrianCChao, origin → deep dive
two $V5/S02_Known.mp4 0.3 12.0     $V5/S02_Known.mp4 15.6 17.9     clips/q1_1_plan.mp4      # prompt → agent's detail map → "Can a diffusion model spend its tokens where the plan says?"
two $V5/S05_AsymFlow.mp4 6.8 10.9  $V5/S05_AsymFlow.mp4 17.0 26.5  clips/q1_2_flow.mp4      # DiT sees the patch through one token → Eq. 9 fills the rest, exact with a perfect head
one $V5/S06_DiT.mp4 12.0 26.0                                      clips/q1_3_dit.mp4       # shape embedding + patch-centre RoPE
one $V5/S09_Outlook.mp4 16.0 23.6                                  clips/q1_4_ideas.mp4     # "in three ideas" summary
# Q2 @<4th co-author>, skeptic → evidence
one $V5/S01_Cost.mp4 17.5 31.8                                     clips/q2_1_cost.mp4      # 16.8 M query–key pairs; 2× resolution → 16× attention
two $V5/S08_Results.mp4 1.5 6.8    $V5/S08_Results.mp4 20.6 27.4   clips/q2_2_baselines.mp4 # HPSv3-vs-speed plot → Fig. 7b video at the same budget
one $V5/S08_Results.mp4 28.3 34.7                                  clips/q2_3_ablation.mp4  # Tab. 3 ablation bars
one $V5/S09_Outlook.mp4 0.5 7.0                                    clips/q2_4_limit.mp4     # Fig. 22 sheet music, broken staff lines
# Q3 @YarivLior, question → vision
ffmpeg -v error -y -i $V5/S00_Title.mp4 -vf "drawbox=x=0:y=930:w=iw:h=150:color=black:t=fill,fade=t=out:st=5.9:d=0.4" $ENC clips/q3_1_emblem.mp4  # title emblem; anonymous-submission line masked
two scratch/demo_train.mp4 0 5.4   scratch/demo_restaurant.mp4 0 5.4 clips/q3_2_agent.mp4   # site demos: agent-built scenes → layouts → video
one $V5/S09_Outlook.mp4 7.5 15.8                                   clips/q3_3_outlook.mp4   # plan → compute → image, "editable plan"

for f in clips/*.mp4; do printf '%-32s %6.2f s\n' "$(basename "$f")" "$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$f")"; done
