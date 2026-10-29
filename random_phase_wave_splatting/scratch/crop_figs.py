"""Crop paper figures into data/. Sources: random-phase-wave-splatting/pdfs/figs/*_hi.png, rendered from the arXiv
LaTeX-source figure PDFs with PyMuPDF (long side 5200 px). Panel boxes were found programmatically from the white
separators between panels (see the run-length search used when this script was written)."""
from PIL import Image
import os
P = "/home/brianchc/project_animations/random_phase_wave_splatting"
F = P + "/random-phase-wave-splatting/pdfs/figs/"
D = P + "/data/"
os.makedirs(D, exist_ok=True)


def crop(src, box, out, pad=4):
    im = Image.open(F + src).convert("RGB")
    x0, y0, x1, y1 = box
    im.crop((x0 + pad, y0 + pad, x1 - pad, y1 - pad)).save(D + out)


# ---- Fig. 4 simulated focal stacks (materials) + light fields (bicycle)
cols = dict(zip(["gt", "poly1", "poly8", "gws", "rpws1", "rpws8", "rpws24"],
                [(394, 1077), (1080, 1764), (1767, 2450), (2454, 3137), (3141, 3824), (3827, 4511), (4514, 5197)]))
rows = {"front": (151, 832), "rear": (838, 1519), "zoomA": (1525, 1864), "zoomB": (1868, 2206),
        "center": (2214, 2670), "right": (2673, 3129), "epi": (3133, 3475)}
for c, (a, b) in cols.items():
    for r, (y0, y1) in rows.items():
        crop("sim_fs_lf_hi.png", (a, y0, b, y1), f"sim_{c}_{r}.png", pad=5)
# ---- Fig. 6 experimentally captured
cols = dict(zip(["gt", "poly1", "poly8", "gws", "rpws1", "rpws8", "rpws24", "tri24"],
                [(310, 846), (849, 1385), (1389, 1925), (1929, 2464), (2469, 3004), (3008, 3544), (3548, 4084), (4087, 4623)]))
rows = {"bike_front": (122, 480), "bike_back": (484, 841), "lego_front": (1386, 1743), "lego_back": (1747, 2105),
        "lego_zA": (2109, 2375), "lego_zB": (2379, 2644), "bike_left": (2672, 3030), "bike_right": (3034, 3391),
        "lego_left": (3936, 4293), "lego_right": (4298, 4655)}
for c, (a, b) in cols.items():
    for r, (y0, y1) in rows.items():
        crop("experimental_main_hi.png", (a, y0, b, y1), f"exp_{c}_{r}.png", pad=5)
# lego zoom crops are split into two squares per column (red | blue)
for c, (a, b) in cols.items():
    m = (a + b) // 2
    for r in ("lego_zA", "lego_zB"):
        y0, y1 = rows[r]
        crop("experimental_main_hi.png", (a, y0, m, y1), f"exp_{c}_{r}_L.png", pad=8)
        crop("experimental_main_hi.png", (m, y0, b, y1), f"exp_{c}_{r}_R.png", pad=8)
# ---- Fig. 3 ablation
cols = dict(zip(["gt", "nrp", "gwssqrt", "nosqrt", "full"],
                [(49, 1074), (1079, 2105), (2110, 3136), (3141, 4167), (4172, 5197)]))
for c, (a, b) in cols.items():
    crop("alpha_blending_ablation_hi.png", (a, 160, b, 843), f"abl_{c}_lego.png", pad=4)
    crop("alpha_blending_ablation_hi.png", (a, 848, b, 1531), f"abl_{c}_garden.png", pad=4)
# ---- Fig. 1 teaser (big panels + zoom insets)
crop("teaser_hi.png", (1303, 2, 2758, 977), "teaser_gws_front.png", pad=3)
crop("teaser_hi.png", (1303, 979, 2758, 1949), "teaser_rpws_back.png", pad=3)
crop("teaser_hi.png", (3255, 2, 4712, 977), "teaser_nrp_front.png", pad=3)
crop("teaser_hi.png", (3255, 979, 4712, 1949), "teaser_rpws_front.png", pad=3)
crop("teaser_hi.png", (0, 0, 1290, 1969), "teaser_left.png", pad=0)
# ---- Fig. 2 toy example
for k, (a, b) in zip(["gt", "nrp", "rpws"], [(31, 1649), (1674, 3241), (3266, 4884)]):
    crop("alpha_blending_toy_example_hi.png", (a, 79, b, 1697), f"toy_{k}.png", pad=3)
# ---- Fig. 5 triangle splats (RPWS-Δ, simulated)
for i, (a, b) in enumerate([(380, 1056), (1061, 1739), (1748, 2426), (2436, 3113)]):
    crop("triangle_splats_hi.png", (a, 6, b, 457), f"tri_s{i}_front.png", pad=4)
    crop("triangle_splats_hi.png", (a, 464, b, 916), f"tri_s{i}_rear.png", pad=4)
for i, (a, b) in enumerate([(3463, 4326), (4333, 5196)]):
    crop("triangle_splats_hi.png", (a, 6, b, 582), f"tri_lf{i}_center.png", pad=4)
    crop("triangle_splats_hi.png", (a, 586, b, 1162), f"tri_lf{i}_right.png", pad=4)
    crop("triangle_splats_hi.png", (a, 1167, b, 1599), f"tri_lf{i}_epi.png", pad=4)
print(len(os.listdir(D)), "files in data/")
