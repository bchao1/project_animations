"""Crop paper figures into data/. Sources: mpi-holography/pdfs/figs/*_hi.png, rendered from the arXiv LaTeX-source
figure PDFs with PyMuPDF (long side 5200 px). Panel boxes were found programmatically from the white separators
between panels (row / column white-run search)."""
from PIL import Image
import os
Image.MAX_IMAGE_PIXELS = None
P = "/home/brianchc/project_animations/mpi_holography"
F = P + "/mpi-holography/pdfs/figs/"
D = P + "/data/"
os.makedirs(D, exist_ok=True)
_cache = {}


def crop(src, box, out, pad=4, maxw=None):
    if src not in _cache:
        _cache.clear()
        _cache[src] = Image.open(F + src).convert("RGB")
    im = _cache[src]
    x0, y0, x1, y1 = box
    c = im.crop((x0 + pad, y0 + pad, x1 - pad, y1 - pad))
    if maxw and c.width > maxw:
        c = c.resize((maxw, round(c.height * maxw / c.width)), Image.LANCZOS)
    c.save(D + out)


METHODS = ["gt", "rpws", "rgbd4", "rgbd64", "rgbd512", "mpi4", "mpi64", "mpi512"]

# ---- Fig. 1 teaser
crop("teaser_hi.png", (0, 0, 1604, 1682), "teaser_mpi_stack.png", pad=0)
crop("teaser_hi.png", (1835, 0, 3092, 838), "teaser_rgbd_front.png", pad=3)
crop("teaser_hi.png", (3520, 0, 4779, 838), "teaser_rpws_front.png", pad=3)
crop("teaser_hi.png", (1835, 844, 3092, 1682), "teaser_mpi_back.png", pad=3)
crop("teaser_hi.png", (3520, 844, 4779, 1682), "teaser_mpi_front.png", pad=3)
for name, x0, x1 in (("rgbd", 3097, 3515), ("rpws", 4784, 5200)):
    crop("teaser_hi.png", (x0, 0, x1, 416), f"teaser_{name}_zoomA.png", pad=3)
    crop("teaser_hi.png", (x0, 421, x1, 838), f"teaser_{name}_zoomB.png", pad=3)
for name, x0, x1 in (("mpiback", 3097, 3515), ("mpifront", 4784, 5200)):
    crop("teaser_hi.png", (x0, 843, x1, 1261), f"teaser_{name}_zoomA.png", pad=3)
    crop("teaser_hi.png", (x0, 1266, x1, 1682), f"teaser_{name}_zoomB.png", pad=3)
# ---- Fig. 2 pipeline (whole panels a / b / c / d)
# (panel titles cropped off: their equation numbers refer to the paper, not the video)
crop("mpi_pipeline_hi.png", (0, 240, 2720, 1900), "pipe_a.png", pad=0)
crop("mpi_pipeline_hi.png", (3180, 240, 5200, 1900), "pipe_b.png", pad=0)
crop("mpi_pipeline_hi.png", (0, 2150, 3000, 3644), "pipe_c.png", pad=0)
crop("mpi_pipeline_hi.png", (3080, 2150, 5200, 3644), "pipe_d.png", pad=0)
# ---- Fig. 4 depth transitions (simulated zoom-ins)
cols = dict(zip(METHODS, [(407, 1002), (1005, 1601), (1605, 2200), (2203, 2799), (2802, 3397), (3401, 3996),
                          (4000, 4596), (4599, 5193)]))
rows = {"garden_fs": (126, 522), "garden_fs_z": (526, 821), "lego_fs": (825, 1222), "lego_fs_z": (1225, 1521),
        "bike_lf": (1523, 1920), "bike_lf_z": (1923, 2218), "garden_lf": (2223, 2620), "garden_lf_z": (2623, 2918)}
for c, (a, b) in cols.items():
    for r, (y0, y1) in rows.items():
        crop("depth_leakage_hi.png", (a, y0, b, y1), f"dl_{c}_{r}.png", pad=3)
# ---- Fig. 7 simulated focal stacks / light fields
cols = dict(zip(METHODS, [(315, 835), (837, 1358), (1361, 1882), (1884, 2405), (2407, 2928), (2931, 3451),
                          (3454, 3975), (3977, 4497)]))
rows = {"lego_front": (112, 631), "lego_back": (635, 1155), "lego_zA": (1159, 1417), "lego_zB": (1420, 1678),
        "kitchen_front": (1685, 2032), "kitchen_back": (2035, 2383), "kitchen_zA": (2385, 2644),
        "kitchen_zB": (2647, 2905), "gums_left": (2923, 3442), "gums_right": (3446, 3966), "gums_epi": (3970, 4230),
        "legolf_left": (4237, 4584), "legolf_right": (4586, 4935), "legolf_epi": (4937, 5198)}
for c, (a, b) in cols.items():
    for r, (y0, y1) in rows.items():
        crop("simulated_main_hi.png", (a, y0, b, y1), f"sim_{c}_{r}.png", pad=3)
# ---- Fig. 8 experimentally captured
cols = dict(zip(METHODS, [(304, 841), (844, 1381), (1384, 1922), (1925, 2462), (2465, 3002), (3005, 3543),
                          (3546, 4083), (4086, 4623)]))
rows = {"garden_front": (116, 474), "garden_back": (478, 836), "garden_zA": (839, 1107), "garden_zB": (1110, 1376),
        "lego_front": (1386, 1744), "lego_back": (1748, 2106), "lego_zA": (2109, 2376), "lego_zB": (2380, 2646),
        "bike_left": (2668, 3027), "bike_right": (3030, 3389), "legolf_left": (3938, 4297),
        "legolf_right": (4300, 4659)}
for c, (a, b) in cols.items():
    for r, (y0, y1) in rows.items():
        crop("experimental_main_hi.png", (a, y0, b, y1), f"exp_{c}_{r}.png", pad=3)
# ---- Fig. 5 captured light field at three focus depths
cols = {"left": (580, 1833), "center": (1840, 3093), "right": (3101, 4355)}
rows = {"front": (175, 1011), "middle": (1017, 1855), "back": (1861, 2697)}
for c, (a, b) in cols.items():
    for r, (y0, y1) in rows.items():
        crop("fs_lf_main_hi.png", (a, y0, b, y1), f"lf5_{r}_{c}.png", pad=3, maxw=1400)
# ---- Fig. 6 Stereo Magnification MPIs
for k, (a, b) in enumerate([(442, 1999), (2019, 3576), (3597, 5175)]):
    crop("stereo_mag_hi.png", (a, 28, b, 1059), f"sm{k}_back.png", pad=3, maxw=1400)
    crop("stereo_mag_hi.png", (a, 1080, b, 2132), f"sm{k}_front.png", pad=3, maxw=1400)
# ---- Table 1 / Fig. 3 are re-typeset in manim (numbers in the summary)
print(len([f for f in os.listdir(D) if f.endswith(".png")]), "png files in data/")
