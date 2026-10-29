"""Crop paper figures (rasterized at 2400 px max side) into data/. Coords measured on
2000-px-wide previews, scaled by s = 2400/2000 (or per-figure scale)."""
from PIL import Image
import os
P = "/home/brianchc/project_animations/gaussian_wave_splatting"
F = P + "/gaussian-wave-splatting/pdfs/figs/"
D = P + "/data/"
def crop(src, box, out, s=1.2, pad=4):
    im = Image.open(F + src).convert("RGB")
    x0, y0, x1, y1 = [round(v * s) for v in box]
    im.crop((x0 + pad, y0 + pad, x1 - pad, y1 - pad)).save(D + out)
# fig5: displayed 1782 wide -> 2138
s5 = 2138 / 1782
cols = {"point": (218, 604), "polygon": (609, 996), "fast": (1001, 1388), "exact": (1392, 1780)}
for k, (a, b) in cols.items():
    crop("fig5_results.png", (a, 42, b, 300), f"fig5_lego_target_{k}.png", s5)
    crop("fig5_results.png", (a, 305, b, 560), f"fig5_lego_captured_{k}.png", s5)
    crop("fig5_results.png", (a, 960, b, 1280), f"fig5_mat_target_{k}.png", s5)
    crop("fig5_results.png", (a, 1285, b, 1605), f"fig5_mat_captured_{k}.png", s5)
# fig3 awb
for k, (a, b) in zip(["naive", "silhouette", "ours", "target"], [(0, 494), (501, 996), (1001, 1496), (1503, 1998)]):
    crop("fig3_awb.png", (a, 80, b, 410), f"fig3_{k}.png")
# fig4 vde
for i, (a, b) in enumerate([(193, 634), (648, 1088), (1103, 1543), (1556, 1997)]):
    for j, (c, d) in enumerate([(140, 578), (594, 1032), (1048, 1487)]):
        crop("fig4_vde.png", (a, c, b, d), f"fig4_c{i}_r{j}.png", pad=3)
# fig6 nvs: separators found programmatically (orig px)
for i, (a, b) in enumerate([(128, 879), (886, 1638), (1645, 2396)]):
    for j, nm in enumerate(["rear", "mid", "front"]):
        c, d = [(1, 502), (509, 1010), (1018, 1519)][j]
        crop("fig6_nvs.png", (a, c, b, d), f"fig6_v{i}_{nm}.png", s=1, pad=3)
# teaser
crop("teaser.png", (1414, 5, 1850, 295), "teaser_front.png")
crop("teaser.png", (1414, 300, 1850, 590), "teaser_back.png")
# fig2 oit
crop("fig2_oit.png", (2, 100, 995, 755), "fig2_naive_oit.png")
crop("fig2_oit.png", (1008, 100, 1998, 755), "fig2_vd_oit.png")
# setup photo
Image.open(F + "fig_supp_setup.png").convert("RGB").save(D + "setup.png")
print(sorted(os.listdir(D)))
