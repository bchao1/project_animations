"""Crop supplement figures into data/ and extract video frames.
Sources: high-brightness-holography/pdfs/figs/*_hi.png (embedded rasters of the Optica supplement PDF, extracted with PyMuPDF
at native resolution; vector figures rendered at 400 dpi). Panel boxes were found from the white separators between panels
(run-length search over near-white rows/columns). Videos: data/video/OL2023.mp4 (bchao1.github.io) and
data/video/visualization1.mp4 (Optica Visualization 1, figshare 22290445) -> data/video/<name>/%04d.jpg at 30 fps.
Run from the project folder:  python scratch/crop_figs.py"""
import os
import subprocess
from PIL import Image
import fitz

P = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
F = P + "/high-brightness-holography/pdfs/figs/"
D = P + "/data/"
os.makedirs(D, exist_ok=True)


def crop(src, box, out, pad=2):
    im = Image.open(F + src).convert("RGB")
    x0, y0, x1, y1 = box
    im.crop((x0 + pad, y0 + pad, x1 - pad, y1 - pad)).save(D + out)


# Fig. S1 setup
crop("figS1_setup_hi.png", (0, 0, 1300, 1080), "s1_schematic.png", 0)
crop("figS1_setup_hi.png", (1321, 0, 2786, 1080), "s1_photo.png", 0)
# Fig. S6 conventional vs holographic (simulated, fixed power budget)
cols = [(0, 1190), (1200, 2390), (2400, 3590), (3600, 4790)]
for k, (a, b) in zip(["conv", "h017", "h033", "h067"], cols):
    crop("figS6_conventional_hi.png", (a, 150, b, 850), f"s6_{k}_monarch.png")
    crop("figS6_conventional_hi.png", (a, 860, b, 1560), f"s6_{k}_macaw.png")
# Fig. S7 long vs short propagation distance (captured, SGD-CITL)
crop("figS7_propdist_hi.png", (0, 100, 1425, 800), "s7_long.png")
crop("figS7_propdist_hi.png", (1430, 100, 2855, 800), "s7_short.png")
# Fig. S8 loss weight 0 / 1 / 100 / Inf (captured)
cols = [(0, 1280), (1285, 2565), (2570, 3850), (3855, 5135)]
rows = [(150, 950), (970, 1770), (1790, 2590)]
for k, (a, b) in zip(["w0", "w1", "w100", "winf"], cols):
    for r, (y0, y1) in zip(["monarch", "bunny", "eagle"], rows):
        crop("figS8_weights_hi.png", (a, y0, b, y1), f"s8_{k}_{r}.png")
# Fig. S9 ASM / CITL / Model, without (top) / with (bottom) the loss (captured)
cols = [(0, 1280), (1290, 2570), (2580, 3860)]
rows = {"castle": [(150, 950), (960, 1760)], "cat": [(1860, 2660), (2670, 3470)], "swallowtail": [(3570, 4370), (4380, 5180)]}
for k, (a, b) in zip(["asm", "citl", "model"], cols):
    for r, ((y0, y1), (y2, y3)) in rows.items():
        crop("figS9_methods_hi.png", (a, y0, b, y1), f"s9_{k}_{r}_noloss.png")
        crop("figS9_methods_hi.png", (a, y2, b, y3), f"s9_{k}_{r}_loss.png")
# Fig. S10 SGD-model results, without (top) / with (bottom)
names = [["bunny1", "bunny2", "castle"], ["parrot", "cat", "lake"], ["beach", "swallowtail", "glasses"]]
rows = [((0, 800), (810, 1611)), ((1710, 2510), (2520, 3320)), ((3420, 4220), (4230, 5030))]
for rr, ((y0, y1), (y2, y3)) in enumerate(rows):
    for k, (a, b) in enumerate(cols):
        crop("figS10_results_hi.png", (a, y0, b, y1), f"s10_{names[rr][k]}_noloss.png")
        crop("figS10_results_hi.png", (a, y2, b, y3), f"s10_{names[rr][k]}_loss.png")
# Fig. S12 (sinusoid) and S14 (natural image): plot + small / large ROI holograms
for f, tag in (("figS12_hi.png", "s12"), ("figS14_hi.png", "s14")):
    crop(f, (0, 0, 3560, 2278), f"{tag}_plot.png", 0)
    for nm, (y0, y1) in zip(["small_noloss", "small_loss", "large_noloss", "large_loss"],
                            [(0, 564), (567, 1131), (1147, 1711), (1714, 2278)]):
        crop(f, (3573, y0, 4474, y1), f"{tag}_{nm}.png", 3)
        im = Image.open(D + f"{tag}_{nm}.png")
        a = __import__("numpy").asarray(im).copy()
        if nm.endswith("noloss"):                 # paint out the (b) / (c) panel label
            a[:160, :170] = a[:160, -170:][:, ::-1]      # mirrored top-right corner (panels are near-symmetric)
        Image.fromarray(a).save(D + f"{tag}_{nm}.png")
# Fig. S5 model architecture (vector page, rendered)
d = fitz.open(P + "/high-brightness-holography/pdfs/chao2023high_supplement.pdf")
d[6].get_pixmap(dpi=400, clip=fitz.Rect(117, 97, 494, 194)).save(D + "s5_model.png")
d[4].get_pixmap(dpi=400, clip=fitz.Rect(205, 462, 425, 658)).save(D + "s4_plot.png")
# video frames
for name in ("OL2023", "visualization1"):
    out = D + "video/" + name
    os.makedirs(out, exist_ok=True)
    for f in os.listdir(out):
        os.remove(os.path.join(out, f))
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", D + f"video/{name}.mp4", "-vf", "fps=30", "-q:v", "2",
                    out + "/%04d.jpg"], check=True)
    print(name, len(os.listdir(out)), "frames")
print(len([f for f in os.listdir(D) if f.endswith(".png")]), "png crops in data/")
