"""v5 section QA: last frame of every section mp4 + one contact sheet per scene + sections.json manifest.
Usage: python scratch/qa_sections_v5.py   (paths fixed to the v5 1080p60 render)"""
import glob, json, os, subprocess
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEC = os.path.join(HERE, "media/videos/animation_spectral_progressive_v5/1080p60/sections")
OUT = os.path.join(HERE, "slides_v5")
QA = os.path.join(OUT, "qa")
SCENES = ["S00_Title", "S01_Cost", "S02_Spectrum", "S03_Noise", "S04_Frontier", "S05_Expansion",
          "S06_Schedule", "S07_FineTune", "S08_Results"]
DESC = json.load(open(os.path.join(HERE, "scratch/section_desc_v5.json")))
os.makedirs(QA, exist_ok=True)


def dur(p):
    return float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                          "-of", "csv=p=0", p]).decode())


manifest = []
for sc in SCENES:
    idx = json.load(open(os.path.join(SEC, f"{sc}.json")))
    thumbs = []
    for i, s in enumerate(idx):
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
    print(sc, len(idx), "sections")
json.dump(manifest, open(os.path.join(OUT, "sections.json"), "w"), indent=2, ensure_ascii=False)
print("wrote", os.path.join(OUT, "sections.json"), len(manifest))
