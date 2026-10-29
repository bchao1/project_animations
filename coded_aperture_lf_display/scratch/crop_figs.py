"""Export the paper's figure panels into data/ at their native (embedded) resolution.
Source: coded-aperture-lf-display/pdfs/chao2023time.pdf (arXiv 2107.06205v3). The arXiv submission is PDF-only (no LaTeX
source), so panels are pulled out of the PDF by image xref (bounding boxes listed in pdfs/figs/image_bboxes.txt, mapped to
figure panels by page position and checked against the page renders in pdfs/pages/)."""
import fitz
import os
from PIL import Image
import io

P = "/home/brianchc/project_animations/coded_aperture_lf_display"
PDF = P + "/coded-aperture-lf-display/pdfs/chao2023time.pdf"
D = P + "/data/"
os.makedirs(D, exist_ok=True)
doc = fitz.open(PDF)


def save(xref, name):
    pix = fitz.Pixmap(doc, xref)
    if pix.n - pix.alpha >= 4 or pix.alpha:
        pix = fitz.Pixmap(fitz.csRGB, pix)
    im = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
    im.save(D + name + ".png")
    return im


# Fig. 1 optical configurations (a) generic AR, (b) SDM-LFD, (c) TDM-LFD, (d) CTDM-LFD
for x, k in zip([62, 63, 64, 65], "abcd"):
    save(x, f"fig1{k}")
# Fig. 2 refocused: (a) TDM-LFD, (b) CTDM-LFD
save(113, "fig2_tdm")
save(115, "fig2_ctdm")
# Fig. 3 pipeline, Fig. 4 focus measure
save(136, "fig3_pipeline")
save(138, "fig4_focus")
# Fig. 5/6 network, Fig. 8 Zemax lens, Fig. 9 binary apertures
save(157, "fig5_resblock")
save(158, "fig6_net")
save(173, "fig8_lens")
for i, x in enumerate([174, 175, 176, 177]):
    save(x, f"fig9_ap{i}")
# Fig. 10 INRIA-syn (n = 4): rows chairs / roses; baseline, ours, GT (+ zoom patches top/bottom)
for r, xs in enumerate([(187, 189, 191), (196, 198, 200)]):
    for m, x in zip(["base", "ours", "gt"], xs):
        save(x, f"fig10_r{r}_{m}")
for m, (a, b) in zip(["base", "ours", "gt"], [(188, 193), (190, 194), (192, 195)]):
    save(a, f"fig10_r0_{m}_zA")
    save(b, f"fig10_r0_{m}_zB")
for m, (a, b) in zip(["base", "ours", "gt"], [(197, 202), (199, 203), (201, 204)]):
    save(a, f"fig10_r1_{m}_zA")
    save(b, f"fig10_r1_{m}_zB")
# Fig. 11 INRIA Lytro (n = 4, k = 4): Totoro near/far, letters near/far
for name, xs in {"totoro_near": (209, 211, 213), "totoro_far": (218, 220, 221),
                 "mom_near": (226, 228, 230), "mom_far": (235, 237, 239)}.items():
    for m, x in zip(["base", "ours", "gt"], xs):
        save(x, f"fig11_{name}_{m}")
for name, xs in {"totoro_near": (210, 212, 214), "mom_near": (227, 229, 231), "mom_far": (236, 238, 240)}.items():
    for m, x in zip(["base", "ours", "gt"], xs):
        save(x, f"fig11_{name}_{m}_zA")
# Fig. 12 SIGGRAPH16 (n = 4, k = 4): poppy near/far, agapanthus near/far
for name, xs in {"poppy_near": (247, 249, 251), "poppy_far": (256, 258, 260),
                 "agap_near": (265, 267, 269), "agap_far": (274, 276, 278)}.items():
    for m, x in zip(["base", "ours", "gt"], xs):
        save(x, f"fig12_{name}_{m}")
for name, xs in {"poppy_near": (248, 250, 252)}.items():
    for m, x in zip(["base", "ours", "gt"], xs):
        save(x, f"fig12_{name}_{m}_zA")
# Fig. 13 Zemax optics simulation: near ours / near GT / far ours / far GT
for x, n in zip([294, 295, 296, 297], ["near_ours", "near_gt", "far_ours", "far_gt"]):
    save(x, f"fig13_{n}")
# Fig. 14 (baseline | learned P | learned f and P), Fig. 15 (binary | continuous), Fig. 16 (beta)
save(298, "fig14")
save(299, "fig15")
save(301, "fig16")
# Fig. 17 continuous focal stacks: 6 rows × 9 focal planes (160 px each)
rows = ["chairs", "roses", "totoro", "mom", "agap", "poppy"]
for x, r in zip([311, 312, 313, 314, 315, 316], rows):
    im = save(x, f"fig17_{r}")
    W, H = im.size
    w = W / 9
    for j in range(9):
        im.crop((round(j * w), 0, round((j + 1) * w), H)).save(D + f"fig17_{r}_{j}.png")
print(len(os.listdir(D)), "files in data/")
