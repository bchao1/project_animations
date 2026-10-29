"""Build every image asset used by the animation.

1. Crops of the paper's figures (rasters embedded in the arXiv LaTeX source PDFs) -> data/
2. A real 2D toy experiment -> data/generated/
   Fit a target image with N 2D Gaussians that are alpha-composited front to back
   (paper Eq. 3/4), then -- following the paper's two-stage recipe -- initialise from the
   plain fit and add per-Gaussian T x T texture maps (Eq. 6-9) with K = 1 (alpha), 3 (RGB)
   or 4 (RGBA) channels. UV = local coordinates in units of sigma, texture spans +-m sigma
   (m = 3, as in the paper's implementation), bilinear lookup (grid_sample, align_corners).
   All toy numbers shown in the video (PSNR of the toy fits) come from toy_results.json.

Run once:  conda activate imp_tome; python build_assets.py   (uses CUDA if available)
"""
import json
import os
import math

import fitz
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
GEN = os.path.join(DATA, "generated")
FIGS = os.path.join(HERE, "textured-gaussians", "pdfs", "src", "sec", "figures")
os.makedirs(GEN, exist_ok=True)


# ------------------------------------------------------------------ figure crops
def raster(name, k=0):
    d = fitz.open(os.path.join(FIGS, name + ".pdf"))
    p = d[0]
    x = p.get_images(full=True)[k][0]
    pix = fitz.Pixmap(d, x)
    if pix.n - pix.alpha > 3:
        pix = fitz.Pixmap(fitz.csRGB, pix)
    im = Image.frombytes("RGB" if pix.n == 3 else "RGBA", (pix.width, pix.height), pix.samples)
    rect = p.get_image_rects(x)[0]
    return im.convert("RGB"), rect


def crop_pt(name, boxes_pt, k=0):
    """Crop boxes given in PDF points of the figure page (as measured on the page)."""
    im, r = raster(name, k)
    sx = im.width / r.width
    sy = im.height / r.height
    out = {}
    for key, (x0, y0, x1, y1) in boxes_pt.items():
        b = (round((x0 - r.x0) * sx), round((y0 - r.y0) * sy),
             round((x1 - r.x0) * sx), round((y1 - r.y0) * sy))
        out[key] = im.crop(b)
    return out


def render_pdf(name, zoom=4.0):
    d = fitz.open(os.path.join(FIGS, name + ".pdf"))
    pix = d[0].get_pixmap(matrix=fitz.Matrix(zoom, zoom))
    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


def build_figures():
    # Teaser (Fig. 1): whole figure incl. baked-in 3DGS / Ours labels
    render_pdf("teaser", 3.4).save(os.path.join(DATA, "fig1_teaser.png"))
    # Teaser halves
    im, r = raster("teaser")
    W = im.width
    im.crop((0, 0, int(W * 0.3645), im.height)).save(os.path.join(DATA, "fig1_art_full.png"))
    im.crop((int(W * 0.5025), 0, int(W * 0.8645), im.height)).save(os.path.join(DATA, "fig1_flower_full.png"))
    # Main qualitative (Fig. 5): per scene, 3DGS* block and Ours block (main + 2 insets).
    # page coords (pt): columns of left block / right block, rows.
    cols = {"L3": (0.0, 150.6), "LO": (152.0, 302.7), "R3": (310.0, 460.4), "RO": (461.8, 612.0)}
    rows = [(23.0, 104.3), (106.0, 187.5), (188.4, 270.6), (271.5, 353.4), (354.3, 436.4)]
    scenes = {"ship": ("L", 0), "lego": ("R", 0), "drjohnson": ("L", 1), "playroom": ("R", 1),
              "counter": ("L", 2), "stump": ("R", 2), "truck": ("L", 3), "train": ("R", 3),
              "dtu_smurf": ("L", 4), "dtu_buddha": ("R", 4)}
    boxes = {}
    for s, (side, ri) in scenes.items():
        y0, y1 = rows[ri][0] + 0.4, rows[ri][1] - 0.4
        boxes[s + "_3dgs"] = (cols[side + "3"][0] + 0.3, y0, cols[side + "3"][1] - 0.3, y1)
        boxes[s + "_ours"] = (cols[side + "O"][0] + 0.3, y0, cols[side + "O"][1] - 0.3, y1)
    for k, v in crop_pt("main_qualitative", boxes).items():
        v.save(os.path.join(DATA, f"fig5_{k}.png"))
    # Custom captures (Fig. 6)
    rows = [(23.0, 104.3), (106.0, 187.7)]
    scenes = {"art1": ("L", 0), "art2": ("R", 0), "flower1": ("L", 1), "flower2": ("R", 1)}
    boxes = {}
    for s, (side, ri) in scenes.items():
        y0, y1 = rows[ri][0] + 0.4, rows[ri][1] - 0.4
        boxes[s + "_3dgs"] = (cols[side + "3"][0] + 0.3, y0, cols[side + "3"][1] - 0.3, y1)
        boxes[s + "_ours"] = (cols[side + "O"][0] + 0.3, y0, cols[side + "O"][1] - 0.3, y1)
    for k, v in crop_pt("custom_main_figure", boxes).items():
        v.save(os.path.join(DATA, f"fig6_{k}.png"))
    # Texture-variant ablation, Blender (Fig. A.x), 1% Gaussians: lego row + ship row
    im, r = raster("tex_ablation_blender")
    period_x, period_y = 567, 309.5
    for name, ri in (("lego", 4), ("ship", 7), ("chair", 0)):
        for ci, var in enumerate(("3dgs", "alpha", "rgb", "rgba")):
            x0 = ci * period_x
            y0 = int(round(ri * period_y)) + 1
            im.crop((x0 + 1, y0, x0 + 563, y0 + 305)).save(os.path.join(DATA, f"figA_{name}_{var}.png"))
    # Colour decomposition (Fig. 7), bicycle: image row only
    im, r = raster("color_contribution", 9)
    im.save(os.path.join(DATA, "fig7_decomp_row.png"))
    # Variants figure (Fig. 2), full render
    render_pdf("rgba_textures", 2.0).save(os.path.join(DATA, "fig2_variants.png"))
    # Pipeline figure (Fig. 3), full render
    render_pdf("model_pipeline", 2.6).save(os.path.join(DATA, "fig3_pipeline.png"))


# ------------------------------------------------------------------ toy experiment
DEV = "cuda" if torch.cuda.is_available() else "cpu"
RES = 128
M_EXT = 3.0      # texture spans +-3 sigma (paper: m = 3)


def target_image():
    im, r = raster("teaser")
    t = im.crop((1150, 380, 1590, 820)).resize((RES, RES), Image.LANCZOS)
    t.save(os.path.join(GEN, "toy_target.png"))
    im.crop((1150, 380, 1590, 820)).resize((384, 384), Image.LANCZOS).save(os.path.join(GEN, "toy_target_hi.png"))
    return torch.tensor(np.asarray(t) / 255.0, dtype=torch.float32, device=DEV)


class Splats(torch.nn.Module):
    def __init__(self, n, target, T=8, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        k = int(math.ceil(math.sqrt(n)))
        ys, xs = torch.meshgrid(torch.arange(k), torch.arange(k), indexing="ij")
        pts = torch.stack([xs.flatten(), ys.flatten()], -1)[:n].float()
        mu = (pts + 0.5) / k * RES + (torch.rand(n, 2, generator=g) - 0.5) * RES / k * 0.5
        self.mu = torch.nn.Parameter(mu.to(DEV))
        self.log_s = torch.nn.Parameter(torch.full((n, 2), math.log(RES / k * 0.45), device=DEV))
        self.theta = torch.nn.Parameter((torch.rand(n, generator=g) * math.pi).to(DEV))
        ij = mu.long().clamp(0, RES - 1)
        col = target[ij[:, 1].to(DEV), ij[:, 0].to(DEV)].clamp(0.02, 0.98)
        self.c = torch.nn.Parameter(torch.logit(col))
        self.o = torch.nn.Parameter(torch.full((n,), 1.5, device=DEV))
        self.T = T
        self.tex_rgb = torch.nn.Parameter(torch.zeros(n, 3, T, T, device=DEV))
        self.tex_a = torch.nn.Parameter(torch.full((n, 1, T, T), 4.0, device=DEV))
        self.mode = "none"     # none | alpha | rgb | rgba

    def forward(self, res=RES, return_parts=False, only=None):
        n = self.mu.shape[0]
        s = (torch.arange(res, device=DEV).float() + 0.5) * (RES / res)
        py, px = torch.meshgrid(s, s, indexing="ij")
        dx = px[None] - self.mu[:, 0, None, None]
        dy = py[None] - self.mu[:, 1, None, None]
        ct, st = torch.cos(self.theta)[:, None, None], torch.sin(self.theta)[:, None, None]
        sig = torch.exp(self.log_s)
        u = (ct * dx + st * dy) / sig[:, 0, None, None]
        v = (-st * dx + ct * dy) / sig[:, 1, None, None]
        G = torch.exp(-0.5 * (u * u + v * v))
        inside = ((u.abs() <= M_EXT) & (v.abs() <= M_EXT)).float()
        o = torch.sigmoid(self.o)[:, None, None]
        base = torch.sigmoid(self.c)[:, :, None, None]                     # n,3,1,1
        if self.mode != "none":
            grid = torch.stack([u / M_EXT, v / M_EXT], -1)                   # n,H,W,2
            tex = torch.cat([self.tex_rgb, torch.sigmoid(self.tex_a)], 1)
            smp = F.grid_sample(tex, grid, mode="bilinear", padding_mode="border", align_corners=True)
            a_tex = smp[:, 3] if self.mode in ("alpha", "rgba") else torch.ones_like(G)
            c_tex = smp[:, :3] if self.mode in ("rgb", "rgba") else torch.zeros_like(smp[:, :3])
        else:
            a_tex = torch.ones_like(G)
            c_tex = torch.zeros(n, 3, res, res, device=DEV)
        alpha = (a_tex * G * o * inside).clamp(max=0.99)                     # Eq. 8
        color = base + c_tex                                                 # Eq. 7
        if only is not None:
            keep = torch.zeros(n, device=DEV)
            keep[only] = 1
            alpha = alpha * keep[:, None, None]
        Tm = torch.cumprod(torch.cat([torch.ones_like(alpha[:1]), 1 - alpha[:-1]], 0), 0)
        w = alpha * Tm                                                       # Eq. 9 weights
        img = (w[:, None] * color).sum(0).permute(1, 2, 0)
        if return_parts:
            return img, alpha, color
        return img


def psnr(a, b):
    return float(-10 * torch.log10(((a.clamp(0, 1) - b) ** 2).mean()))


def optimise(model, target, iters, groups, snap_every=None, res_snap=256, tag=None, log=None):
    opt = torch.optim.Adam(groups)
    snaps = []
    for it in range(iters + 1):
        if snap_every and it % snap_every == 0:
            with torch.no_grad():
                im = model(res_snap).clamp(0, 1).cpu().numpy()
                p = psnr(model(), target)
            fn = f"{tag}_{it // snap_every:03d}.png"
            Image.fromarray((im * 255).astype(np.uint8)).save(os.path.join(GEN, fn))
            snaps.append({"iter": it, "psnr": round(p, 2), "file": fn})
        if it == iters:
            break
        opt.zero_grad()
        loss = (model() - target).abs().mean()
        loss.backward()
        opt.step()
    return snaps


def geo_groups(m):
    return [{"params": [m.mu], "lr": 0.25}, {"params": [m.log_s], "lr": 0.01},
            {"params": [m.theta], "lr": 0.01}, {"params": [m.c], "lr": 0.02},
            {"params": [m.o], "lr": 0.02}]


def save_state(m):
    return {k: v.detach().clone() for k, v in m.state_dict().items()}


def build_toy():
    torch.manual_seed(0)
    tgt = target_image()
    res = {"res": RES, "T": 8, "m": M_EXT, "iters_stage1": 1500, "iters_stage2": 1500}
    # --- how PSNR of plain Gaussians grows with N
    res["plain_sweep"] = {}
    for n in (16, 64, 256, 1024):
        m = Splats(n, tgt)
        snaps = optimise(m, tgt, 3000, geo_groups(m), snap_every=3000, tag=f"toy_plain_n{n}")
        res["plain_sweep"][n] = snaps[-1]
        print("plain", n, snaps[-1])
    # --- stage 1: plain N = 64
    N = 64
    m = Splats(N, tgt)
    res["stage1"] = optimise(m, tgt, 1500, geo_groups(m), snap_every=50, tag="toy_s1")
    st1 = save_state(m)
    with torch.no_grad():
        np.save(os.path.join(GEN, "toy_s1_params.npy"),
                np.concatenate([m.mu.cpu().numpy(), torch.exp(m.log_s).cpu().numpy(),
                                m.theta.cpu().numpy()[:, None]], 1))
    # --- stage 2: variants, same N, initialised from stage 1
    res["stage2"] = {}
    for mode in ("none", "alpha", "rgb", "rgba"):
        m2 = Splats(N, tgt)
        m2.load_state_dict(st1)
        m2.mode = mode
        groups = geo_groups(m2)
        if mode != "none":
            groups += [{"params": [m2.tex_rgb], "lr": 0.02}, {"params": [m2.tex_a], "lr": 0.1}]
        res["stage2"][mode] = optimise(m2, tgt, 1500, groups, snap_every=50, tag=f"toy_s2_{mode}")
        print(mode, res["stage2"][mode][-1])
        with torch.no_grad():
            np.save(os.path.join(GEN, f"toy_s2_{mode}_params.npy"),
                    np.concatenate([m2.mu.cpu().numpy(), torch.exp(m2.log_s).cpu().numpy(),
                                    m2.theta.cpu().numpy()[:, None]], 1))
            hi = m2(384).clamp(0, 1).cpu().numpy()
            Image.fromarray((hi * 255).astype(np.uint8)).save(os.path.join(GEN, f"toy_final_{mode}.png"))
            if mode == "rgba":
                # a few individual textured Gaussians: their texture maps and isolated renders
                img, alpha, color = m2(384, return_parts=True)
                area = torch.exp(m2.log_s).prod(1)
                picks = torch.argsort(area, descending=True)[:6].tolist()
                res["picks"] = picks
                for k, i in enumerate(picks):
                    rgb = (torch.sigmoid(m2.c[i])[:, None, None] + m2.tex_rgb[i]).clamp(0, 1)
                    a = torch.sigmoid(m2.tex_a[i])
                    t = torch.cat([rgb, a], 0).permute(1, 2, 0).cpu().numpy()
                    Image.fromarray((t * 255).astype(np.uint8), "RGBA").resize((256, 256), Image.NEAREST) \
                        .save(os.path.join(GEN, f"toy_pick{k}_tex.png"))
                    Image.fromarray((t[..., :3] * 255).astype(np.uint8)).resize((256, 256), Image.NEAREST) \
                        .save(os.path.join(GEN, f"toy_pick{k}_texrgb.png"))
                    Image.fromarray((t[..., 3] * 255).astype(np.uint8)).resize((256, 256), Image.NEAREST) \
                        .save(os.path.join(GEN, f"toy_pick{k}_texa.png"))
                    single = m2(384, only=[i]).clamp(0, 1)
                    al = alpha[i].clamp(0, 1)
                    rgba = torch.cat([color[i].clamp(0, 1).permute(1, 2, 0), al[..., None]], -1).cpu().numpy()
                    Image.fromarray((rgba * 255).astype(np.uint8), "RGBA").save(
                        os.path.join(GEN, f"toy_pick{k}_splat.png"))
                # same picks without textures (stage-1 state) for comparison
                m1 = Splats(N, tgt)
                m1.load_state_dict(st1)
                _, alpha1, color1 = m1(384, return_parts=True)
                for k, i in enumerate(picks):
                    rgba = torch.cat([color1[i].clamp(0, 1).permute(1, 2, 0), alpha1[i].clamp(0, 1)[..., None]], -1)
                    Image.fromarray((rgba.cpu().numpy() * 255).astype(np.uint8), "RGBA").save(
                        os.path.join(GEN, f"toy_pick{k}_splat_plain.png"))
    json.dump(res, open(os.path.join(GEN, "toy_results.json"), "w"), indent=1)


if __name__ == "__main__":
    build_figures()
    build_toy()
    print("done")
