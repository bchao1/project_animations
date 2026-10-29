"""Last frame of every section mp4 + one contact sheet per scene.
Usage: python scratch/qa_sections.py <sections dir> <out dir>"""
import subprocess, sys, os, glob
from PIL import Image, ImageDraw, ImageStat

SEC, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
scenes = {}
for mp4 in sorted(glob.glob(os.path.join(SEC, "S*_*.mp4"))):
    base = os.path.basename(mp4)[:-4]
    png = os.path.join(OUT, base + "_last.png")
    for off in ("-0.05", "-0.3"):          # -0.05 s can be shorter than one frame at 15 fps
        if os.path.exists(png):
            os.remove(png)
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-sseof", off, "-i", mp4, "-update", "1", png], check=True)
        if os.path.exists(png):
            break
    dur = float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                         "-of", "csv=p=0", mp4]).decode())
    mean = sum(ImageStat.Stat(Image.open(png).convert("L")).mean)
    scene = "_".join(base.split("_")[:2])
    scenes.setdefault(scene, []).append((base, png, dur, mean))
    print(f"{base:55s} {dur:6.2f}s  mean={mean:6.2f}{'  <-- DARK' if mean < 1.0 else ''}")
for scene, items in scenes.items():
    cols = 3
    rows = (len(items) + cols - 1) // cols
    W, H = 640, 360
    sheet = Image.new("RGB", (cols * (W + 4), rows * (H + 24)), (60, 60, 60))
    for k, (base, png, dur, _) in enumerate(items):
        im = Image.open(png).convert("RGB").resize((W, H))
        x, y = (k % cols) * (W + 4), (k // cols) * (H + 24)
        sheet.paste(im, (x + 2, y + 22))
        ImageDraw.Draw(sheet).text((x + 6, y + 4), f"{base[len(scene) + 1:]}  ({dur:.1f}s)", fill=(255, 255, 0))
    sheet.save(os.path.join(OUT, f"sheet_{scene}.png"))
