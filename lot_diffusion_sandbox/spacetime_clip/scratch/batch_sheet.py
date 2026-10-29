import subprocess, numpy as np
from PIL import Image, ImageDraw
items = (0, 3, 4, 10, 14, 16, 22, 26, 37, 39)
ts = (0.5, 4.5, 8.0, 13.5)
rows = []
for k in items:
    f = f"media/final/batch/LoT_spacetime_v15_item{k:02d}_1080p60.mp4"
    row = Image.new("RGB", (427 * 4, 240))
    for j, t in enumerate(ts):
        png = f"scratch/_b{k}_{j}.png"
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", str(t), "-i", f, "-frames:v", "1",
                        "-vf", "scale=427:240", png], check=True)
        row.paste(Image.open(png), (427 * j, 0))
    ImageDraw.Draw(row).text((6, 6), f"item {k}", fill="white")
    rows.append(row)
for half in (0, 1):
    s = Image.new("RGB", (427 * 4, 240 * 5))
    for i, r in enumerate(rows[half * 5:(half + 1) * 5]):
        s.paste(r, (0, 240 * i))
    s.save(f"scratch/batch_sheet_{half}.png")
