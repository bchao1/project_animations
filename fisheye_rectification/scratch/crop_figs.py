"""Crop paper figures into data/. Sources: fisheye-rectification/pdfs/figs/*_hi.png = the images embedded in the IEEE
PDF at native resolution (extracted with PyMuPDF). Panel boxes were found programmatically (runs of non-white pixels
for Fig. 2; connected components of non-white pixels for Fig. 1).
Fig. 2: 5 rows (fisheye input, CycleGAN, UNIT, Aleman-Flores, ours) x 9 columns (3 MCindoor, 3 Wireframe, 3 LSUN Churches).
"""
from PIL import Image
import os
P = "/home/brianchc/project_animations/fisheye_rectification"
F = P + "/fisheye-rectification/pdfs/figs/"
D = P + "/data/"
os.makedirs(D, exist_ok=True)


def crop(src, box, out, pad=3, up=None):
    im = Image.open(F + src).convert("RGB")
    x0, y0, x1, y1 = box
    im = im.crop((x0 + pad, y0 + pad, x1 - pad, y1 - pad))
    if up:
        im = im.resize((up, up), Image.LANCZOS)
    im.save(D + out)


# ---- Fig. 2 qualitative comparison
cols = [(233, 420), (428, 615), (625, 812), (825, 1013), (1020, 1207), (1212, 1400), (1414, 1601), (1609, 1796), (1803, 1991)]
rows = dict(zip(["fisheye", "cyclegan", "unit", "aleman", "ours"],
                [(19, 206), (212, 399), (405, 592), (605, 792), (804, 991)]))
names = ["mc0", "mc1", "mc2", "wf0", "wf1", "wf2", "ch0", "ch1", "ch2"]
for n, (a, b) in zip(names, cols):
    for r, (y0, y1) in rows.items():
        crop("fig2_results_hi.png", (a, y0, b, y1), f"fig2_{r}_{n}.png", pad=2)
# ---- Fig. 1 architecture: input x, rotated input, flow f, output x~, and the discriminator's y
crop("fig1_architecture_hi.png", (33, 126, 185, 277), "fig1_x.png", pad=7)
crop("fig1_architecture_hi.png", (33, 488, 185, 639), "fig1_x_rot.png", pad=7)
crop("fig1_architecture_hi.png", (691, 127, 842, 278), "fig1_flow.png", pad=7)
crop("fig1_architecture_hi.png", (1086, 128, 1237, 279), "fig1_xtilde.png", pad=7)
crop("fig1_architecture_hi.png", (1352, 468, 1496, 612), "fig1_y.png", pad=1)
# ---- whole figures (for "paper Fig." references)
Image.open(F + "fig1_architecture_hi.png").convert("RGB").save(D + "fig1_full.png")
Image.open(F + "fig2_results_hi.png").convert("RGB").save(D + "fig2_full.png")
print("ok")
