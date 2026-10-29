#!/usr/bin/env bash
# qa.sh <video> <out.jpg> : 12-frame contact sheet (every 0.83 s)
ffmpeg -y -loglevel error -i "$1" -vf "fps=1.2,scale=640:-2,tile=3x4" -frames:v 1 "$2"
