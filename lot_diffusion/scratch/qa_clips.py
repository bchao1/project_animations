"""QA contact sheets: 8 frames per clip -> scratch/<out>/<clip>.jpg.  usage: qa_clips.py <video dir> <out dir> [clip ...]"""
import glob, os, subprocess, sys
from PIL import Image
vd, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
names = sys.argv[3:]
tmp = os.path.join(out, "_f.png")
for f in sorted(glob.glob(os.path.join(vd, "*.mp4"))):
    n = os.path.basename(f)[:-4].replace("LoT_", "")
    if names and n not in names:
        continue
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", f],
                               capture_output=True, text=True).stdout)
    ims = []
    for k in range(8):
        t = 0.03 + k * (dur - 0.1) / 7
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(t), "-i", f, "-frames:v", "1", tmp])
        ims.append(Image.open(tmp).convert("RGB").copy())
    w, h = ims[0].size
    S = Image.new("RGB", (w * 4, h * 2))
    for k, im in enumerate(ims):
        S.paste(im, ((k % 4) * w, (k // 4) * h))
    S.save(os.path.join(out, n + ".jpg"))
os.remove(tmp)
