"""Slide-deck export: after `./render.sh s`, write the last frame of every section, one contact
sheet per scene, and sections.json (the manifest the research-talk-slides build.py reads for
speaker-view cue labels and clip lengths).

Usage: python sections_manifest.py <script.py> <out dir> [descriptions.json]
  descriptions.json maps "Scene/section_name" to one line said in the speaker view, e.g.
  {"S01_Cost/tokens": "a 16 × 16 grid: every patch is one token"}.
Copy the section mp4s into the deck with faststart (moov atom at the front):
  ffmpeg -i in.mp4 -map 0 -c copy -movflags +faststart out.mp4
"""
import json
import os
import re
import subprocess
import sys

from PIL import Image, ImageDraw

SCRIPT, OUT = sys.argv[1], sys.argv[2]
DESC = json.load(open(sys.argv[3])) if len(sys.argv) > 3 else {}
NAME = os.path.basename(SCRIPT)[:-3]
SEC = os.path.join(os.path.dirname(os.path.abspath(SCRIPT)), "media", "videos", NAME, "1080p60", "sections")
SCENES = re.findall(r"^class (S\w+)\(", open(SCRIPT).read(), flags=re.M)
QA = os.path.join(OUT, "qa")
os.makedirs(QA, exist_ok=True)


def dur(p):
    return float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                          "-of", "csv=p=0", p]).decode())


manifest = []
for sc in SCENES:
    idx_path = os.path.join(SEC, f"{sc}.json")
    if not os.path.exists(idx_path):
        print("no sections for", sc, "(render with ./render.sh s)")
        continue
    thumbs = []
    for i, s in enumerate(json.load(open(idx_path))):
        mp4 = os.path.join(SEC, s["video"])
        png = os.path.join(QA, s["video"][:-4] + "_last.png")
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-sseof", "-0.05", "-i", mp4, "-update", "1",
                        "-frames:v", "1", png], check=True)
        d = dur(mp4)
        manifest.append(dict(scene=sc, index=i, name=s["name"], file=mp4, duration_s=round(d, 2),
                             last_frame=png, description=DESC.get(f"{sc}/{s['name']}", "")))
        thumbs.append((png, f"{i} {s['name']} ({d:.1f}s)"))
    W, H, cols = 640, 360, 3
    rows = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * (W + 6), rows * (H + 30)), (60, 60, 60))
    for k, (png, lab) in enumerate(thumbs):
        x, y = (k % cols) * (W + 6), (k // cols) * (H + 30)
        sheet.paste(Image.open(png).convert("RGB").resize((W, H)), (x + 3, y + 27))
        ImageDraw.Draw(sheet).text((x + 6, y + 6), lab, fill=(255, 255, 0))
    sheet.save(os.path.join(QA, f"sheet_{sc}.png"))
    print(sc, len(thumbs), "sections")
json.dump(manifest, open(os.path.join(OUT, "sections.json"), "w"), indent=2, ensure_ascii=False)
print("wrote", os.path.join(OUT, "sections.json"), len(manifest), "sections")
