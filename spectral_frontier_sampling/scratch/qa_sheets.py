"""Contact sheets: one frame every STEP seconds of every scene mp4, 4 columns, 5 rows per page.

Usage: python qa_sheets.py <video dir> <out dir> [step]
  e.g. python qa_sheets.py media/videos/animation_x_1min_v1/480p15 scratch/qa_v1 2
Look at every sheet before you show a render to the user. Needs ffmpeg/ffprobe and Pillow.
"""
import glob
import os
import subprocess
import sys

from PIL import Image, ImageDraw

V, OUT = sys.argv[1], sys.argv[2]
STEP = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0
os.makedirs(OUT, exist_ok=True)
for mp4 in sorted(glob.glob(os.path.join(V, "S*.mp4"))):
    name = os.path.basename(mp4)[:-4]
    dur = float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                         "-of", "csv=p=0", mp4]).decode())
    tdir = os.path.join(OUT, name)
    os.makedirs(tdir, exist_ok=True)
    for f in glob.glob(tdir + "/*.png"):
        os.remove(f)
    subprocess.run(["ffmpeg", "-v", "error", "-i", mp4, "-vf", f"fps=1/{STEP}", "-s", "427x240",
                    tdir + "/f_%03d.png"], check=True)
    fr = sorted(glob.glob(tdir + "/f_*.png"))
    cols = 4
    rows = (len(fr) + cols - 1) // cols
    for page in range(0, rows, 5):
        part = fr[page * cols:(page + 5) * cols]
        sheet = Image.new("RGB", (cols * 431, ((len(part) + cols - 1) // cols) * 244), (60, 60, 60))
        for k, f in enumerate(part):
            x, y = (k % cols) * 431, (k // cols) * 244
            sheet.paste(Image.open(f), (x + 2, y + 2))
            ImageDraw.Draw(sheet).text((x + 6, y + 4), f"{(page * cols + k) * STEP:.0f}s", fill=(255, 255, 0))
        sheet.save(os.path.join(OUT, f"{name}_p{page // 5}.png"))
    print(name, f"{dur:.1f}s", len(fr), "frames")
