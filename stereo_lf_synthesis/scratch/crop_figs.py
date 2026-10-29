"""Export paper figure panels (from the embedded PDF rasters in stereo-lf-synthesis/pdfs/figs/) into data/."""
import os
from PIL import Image
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
FG = os.path.join(R, "stereo-lf-synthesis", "pdfs", "figs")
D = os.path.join(R, "data", "fig3"); os.makedirs(D, exist_ok=True)
f3 = Image.open(os.path.join(FG, "p4_img0.png")).convert("RGB")
ov = {"L": (19, 255), "R": (733, 968)}
cols = {"L": [(257, 374), (376, 492), (494, 611), (614, 731)], "R": [(970, 1087), (1089, 1206), (1208, 1325), (1328, 1444)]}
names = ["gt", "kal", "sri", "ours"]
zrows = [(36, 152), (154, 271), (273, 390), (392, 509), (511, 628), (630, 744)]
for r in range(3):
    for s in "LR":
        f3.crop((ov[s][0], zrows[2 * r][0], ov[s][1], zrows[2 * r + 1][1])).save(os.path.join(D, f"r{r}{s}_ov.png"))
        for k in range(2):
            y0, y1 = zrows[2 * r + k]
            for (x0, x1), n in zip(cols[s], names):
                f3.crop((x0, y0, x1, y1)).save(os.path.join(D, f"r{r}{s}_{n}{k}.png"))
Image.open(os.path.join(FG, "p2_img0.png")).convert("RGB").save(os.path.join(R, "data", "fig1_pipeline.png"))
Image.open(os.path.join(FG, "p2_img1.png")).convert("RGB").save(os.path.join(R, "data", "fig2_error.png"))
Image.open(os.path.join(FG, "p4_img1.png")).convert("RGB").save(os.path.join(R, "data", "fig4_refocus.png"))
