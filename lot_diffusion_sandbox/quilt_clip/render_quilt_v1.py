"""LoT Diffusion, 20-s patchwork-quilt clip, v1 (numpy + PIL + cv2 frames → ffmpeg, synthesized sewing / music-box audio).

Tokens are fabric patches (gingham, polka dots, calico, denim, stripes, plaid) tinted by the colour underneath; token
borders are quilted seams with dashed stitches; a sewing needle runs along each sweep.
  0.0–1.9   embroidered title label
  2.0–3.6   the depth map, printed on fabric and pinned down; note "in focus" points at the giraffe
  3.6–5.6   sweep 1: cut into 4,096 tiny patches (one per token)
  5.9–7.5   sweep 2: blurry patches sewn into 2 × 2 (1,606)        → badge 1.9× faster
  8.3–9.7   sweep 3: bigger still, the real layout (1,027)          → badge 2.4× faster
 10.5–12.8  the quilt flips over: the generated image, quilted with the same layout
 13.0–18.2  platformer video left, its quilt right: 75,600 → 38,046 (2.2×) → 34,698 (2.5×); playback follows the speed
 18.2–20.0  title ribbon
Usage: python render_quilt_v1.py h|l      → media/final/LoT_quilt_v1_<res>.mp4
       python render_quilt_v1.py still t … → scratch/still_v1_<t>.png
"""
import json
import math
import os
import subprocess
import sys
from functools import lru_cache
from multiprocessing import Pool

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.io import wavfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TAG = "v1"
F_HAND = os.path.join(HERE, "data/PatrickHand-Regular.ttf")
F_BOLD = os.path.join(HERE, "data/Baloo2.ttf")
CUE = os.path.join(ROOT, "data/img/img_000000000025_dof_cue.png")
GEN = os.path.join(ROOT, "data/img/img_000000000025_dof_generated_4b.png")
LAYOUT = os.path.join(ROOT, "data/layouts/img_000000000025_dof.json")
VID = os.path.join(ROOT, "data/clips/vd_platformer_gen")
LEVELS = os.path.join(ROOT, "arcade_clip/data/platformer_levels.npy")

W, H = 1920, 1080
DUR = 20.0

LINEN = (238, 226, 205)
CREAM = (250, 242, 225)
BROWN = (84, 54, 38)
CRANBERRY = (172, 50, 60)
MUSTARD = (214, 158, 48)
SAGE = (118, 148, 108)
NAVY = (48, 62, 98)
THREAD = (252, 244, 226)

# ------------------------------------------------------------------ timeline (s)
T_TITLE1 = 1.9
T_DROP = 2.0
S1 = (3.6, 5.6)
S2 = (5.9, 7.5)
S3 = (8.3, 9.7)
T_B1, T_B2 = 7.6, 9.8          # image badges
T_FLIP = 10.5
FLIP_D = 0.8
T_OUT = 12.8                   # quilt slides out
T_V0 = 13.0                    # video panels in
V1 = (13.4, 14.3)
V2 = (14.7, 15.6)
V3 = (16.3, 17.1)
T_VB1, T_VB2 = 15.7, 17.2
T_END = 18.2

IMG_TOK = (4096, 1606, 1027)
VID_TOK = (75600, 38046, 34698)
VID_SPEED = (1.0, 2.2, 2.5)

P = 768
N = 64
QX0, QY0 = 300, 140
VW, VH = 880, 495
VX0L, VX0R, VY0 = 60, 980, 175
VN, VM = 45, 80
K_PAT = 7


def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def ease(x):
    x = clamp(x)
    return x * x * (3 - 2 * x)


def ease_out(x):
    x = clamp(x)
    return 1 - (1 - x) ** 3


def back_out(x, s=1.7):
    x = clamp(x) - 1
    return 1 + (s + 1) * x ** 3 + s * x ** 2


def sweep_frac(t, win):
    return clamp((t - win[0]) / (win[1] - win[0]))


# ------------------------------------------------------------------ fabric
def smooth_noise(h, w, scale, rng):
    g = rng.standard_normal((h // scale + 3, w // scale + 3)).astype(np.float32)
    g = cv2.resize(g, (w + 3 * scale, h + 3 * scale), interpolation=cv2.INTER_CUBIC)[:h, :w]
    return g / (g.std() + 1e-6)


def make_patterns(h, w, seed):
    """K_PAT multiplicative fabric textures (~0.75–1.25)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    fib = 0.035 * rng.standard_normal((h, w)).astype(np.float32)
    weave = 0.03 * (((xx.astype(int) + yy.astype(int)) % 2) * 2 - 1)
    pats = []
    pats.append(1 + weave + fib)                                                  # plain weave
    hs = ((yy // 8) % 2).astype(np.float32)
    vs = ((xx // 8) % 2).astype(np.float32)
    pats.append(1.12 - 0.13 * hs - 0.13 * vs + fib)                               # gingham
    cy = (yy % 14) - 7
    cx = ((xx + 7 * ((yy // 14) % 2)) % 14) - 7
    pats.append(np.where(cx * cx + cy * cy < 7.5, 1.3, 0.94).astype(np.float32) + fib)   # polka dots
    n = smooth_noise(h, w, 3, rng)
    n2 = smooth_noise(h, w, 2, rng)
    pats.append(np.where(n > 1.1, 1.25, np.where(n2 > 1.3, 0.82, 0.98)).astype(np.float32) + fib)  # calico
    pats.append(0.97 + 0.09 * np.sin(2 * np.pi * (xx + yy) / 5) + 1.5 * fib)        # denim twill
    pats.append(np.where((xx % 12) < 4, 1.18, 0.95).astype(np.float32) + fib)     # stripes
    pl = 0.09 * ((yy % 20) < 6) + 0.09 * ((xx % 20) < 6) + 0.06 * (((yy + 10) % 20) < 2) + 0.06 * (((xx + 10) % 20) < 2)
    pats.append(0.93 + pl.astype(np.float32) + fib)                               # plaid
    return np.stack(pats).astype(np.float32)


def pastel(c, k=0.3):
    return c * (1 - k) + np.array(CREAM, np.float32) * k


def up(arr, cell_px):
    return np.repeat(np.repeat(arr, cell_px, 0), cell_px, 1)


def block_ids(oi, oj, m):
    return oi * m + oj


def mosaic(cell, ids, n_tok):
    flat = ids.ravel()
    cnt = np.bincount(flat, minlength=n_tok)
    out = np.empty(cell.shape, np.float32)
    for ch in range(3):
        s = np.bincount(flat, weights=cell[..., ch].ravel(), minlength=n_tok)
        out[..., ch] = (s / np.maximum(cnt, 1))[flat].reshape(ids.shape)
    return out


def edge_thin(ids_px):
    e = np.zeros(ids_px.shape, bool)
    e[:, :-1] |= ids_px[:, 1:] != ids_px[:, :-1]
    e[:-1, :] |= ids_px[1:, :] != ids_px[:-1, :]
    e[0, :] = e[-1, :] = e[:, 0] = e[:, -1] = True
    return e


def dilate(e):
    d = e.copy()
    d[:, 1:] |= e[:, :-1]
    d[1:, :] |= e[:-1, :]
    return d


def quilt(cols, ids, sizes, cell_px, pats, grids, flash=None, ghost=None, puff=1.0, seed=0):
    """cols (n,m,3) per-cell colour; ids (n,m) block id; sizes (n,m) block size; returns float image."""
    pid = ((ids.astype(np.int64) * 2654435761 + sizes * 97 + seed) >> 7) % K_PAT
    pid = np.where(sizes == 1, pid % 3, pid)                     # tiny patches: plain / gingham / dots
    yy, xx = grids
    img = up(cols, cell_px)
    if pats is not None:
        img = img * pats[up(pid, cell_px), yy, xx][..., None]
    if flash is not None:
        img = img * (1 + 0.22 * up(flash, cell_px))[..., None]
    ids_px = up(ids, cell_px)
    e1 = edge_thin(ids_px)
    e2 = dilate(e1)
    dist = cv2.distanceTransform((~e2).astype(np.uint8), cv2.DIST_L2, 3)
    shade = 1.04 - 0.34 * puff * np.exp(-dist / 3.2)
    img = img * shade[..., None]
    img[e2] *= 1 - 0.45 * puff
    dash = e1 & (((xx + yy) // 4) % 2 == 0)
    img[dash] = img[dash] * (1 - puff) + np.array(THREAD, np.float32) * puff
    if ghost is not None:                                        # seams of just-merged patches fade out
        g_e, g_a = ghost
        g = (dilate(g_e) & ~e2)
        a = (g_a * g)[..., None] * puff
        img = img * (1 - a * 0.45)
        gd = g_e & ~e2 & (((xx + yy) // 4) % 2 == 0)
        a2 = (g_a * gd)[..., None] * puff
        img = img * (1 - a2) + np.array(THREAD, np.float32) * a2
    return img


# ------------------------------------------------------------------ assets
def load_assets():
    a = {}
    cue = np.asarray(Image.open(CUE).convert("RGB")).astype(np.float32)
    a["cue"] = pastel(cv2.resize(cue, (P, P), interpolation=cv2.INTER_AREA), 0.22)
    a["cue_cell"] = pastel(cue.reshape(N, 16, N, 16, 3).mean((1, 3)), 0.28)
    gen = np.asarray(Image.open(GEN).convert("RGB")).astype(np.float32)
    a["gen"] = cv2.resize(gen, (P, P), interpolation=cv2.INTER_AREA)
    a["gen_cell"] = gen.reshape(N, 16, N, 16, 3).mean((1, 3))
    rects = [tuple(r) for r in json.load(open(LAYOUT))["rects"]]
    size = np.zeros((N, N), int)
    r0m = np.zeros((N, N), int)
    c0m = np.zeros((N, N), int)
    for r0, r1, c0, c1 in rects:
        size[r0:r1, c0:c1] = r1 - r0
        r0m[r0:r1, c0:c1] = r0
        c0m[r0:r1, c0:c1] = c0
    a["rects"] = rects
    ii, jj = np.mgrid[0:N, 0:N]
    a["img_blocks"] = {}
    for cap in (1, 2, 8):
        b = np.minimum(size, cap)
        a["img_blocks"][cap] = (r0m + (ii - r0m) // b * b, c0m + (jj - c0m) // b * b, b)
    a["pats_img"] = make_patterns(P, P, 1)
    a["grid_img"] = tuple(np.mgrid[0:P, 0:P])
    a["pats_vid"] = make_patterns(VH, VW, 2)
    a["grid_vid"] = tuple(np.mgrid[0:VH, 0:VW])
    a["lv"] = np.load(LEVELS)
    a["vid"] = [cv2.resize(np.asarray(Image.open(os.path.join(VID, f"{f + 1:04d}.jpg")).convert("RGB")),
                           (VW, VH), interpolation=cv2.INTER_AREA).astype(np.float32) for f in range(81)]
    a["vid_cell"] = [pastel(cv2.resize(v, (VM, VN), interpolation=cv2.INTER_AREA), 0.12) for v in a["vid"]]
    # linen table with weave + warm vignette
    rng = np.random.default_rng(9)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    weave = 0.025 * np.sin(xx * 2.1) * np.sin(yy * 2.1) + 0.02 * rng.standard_normal((H, W))
    blot = 0.03 * smooth_noise(H, W, 120, rng)
    vig = 1 - 0.22 * (((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
    a["bg"] = np.clip(np.array(LINEN, np.float32) * (1 + weave + blot)[..., None] * vig[..., None], 0, 255)
    # wrinkle field for the quilt
    a["wr_dx"] = (2.2 * smooth_noise(P, P, 90, rng)).astype(np.float32)
    a["wr_dy"] = (2.2 * smooth_noise(P, P, 90, rng)).astype(np.float32)
    a["wr_sh"] = (1 + 0.035 * smooth_noise(P, P, 110, rng)).astype(np.float32)
    gy, gx = np.mgrid[0:P, 0:P].astype(np.float32)
    a["wr_map"] = (gx + a["wr_dx"], gy + a["wr_dy"])
    return a


A = None


def assets():
    global A
    if A is None:
        A = load_assets()
    return A


def wrinkle(img):
    a = assets()
    out = cv2.remap(img, a["wr_map"][0], a["wr_map"][1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return out * a["wr_sh"][..., None]


# ------------------------------------------------------------------ text
@lru_cache(maxsize=64)
def font(kind, size):
    if kind == "bold":
        f = ImageFont.truetype(F_BOLD, size)
        f.set_variation_by_name("ExtraBold")
        return f
    return ImageFont.truetype(F_HAND, size)


@lru_cache(maxsize=2048)
def text_rgba(s, kind, size, color, satin=False, shadow=True):
    f = font(kind, size)
    l, t0, r, b = f.getbbox(s)
    pad = 8
    w, h = r - l + 2 * pad, b - t0 + 2 * pad
    m = Image.new("L", (w, h), 0)
    ImageDraw.Draw(m).text((pad - l, pad - t0), s, font=f, fill=255)
    ma = np.asarray(m).astype(np.float32) / 255
    rgb = np.ones((h, w, 3), np.float32) * np.array(color, np.float32)
    if satin:                                            # satin-stitch sheen for embroidered text
        yy, xx = np.mgrid[0:h, 0:w]
        rgb *= (0.86 + 0.16 * (np.sin((xx - yy) * 2 * np.pi / 5) > 0))[..., None]
    out = np.zeros((h, w, 4), np.float32)
    if shadow:
        sh = np.roll(np.roll(ma, 2, 0), 2, 1) * 0.35
        out[..., 3] = sh
    a = ma
    out[..., :3] = rgb
    out[..., 3] = np.maximum(out[..., 3], a)
    # where only shadow: dark colour
    only_sh = a < out[..., 3]
    out[only_sh, :3] = (40, 25, 15)
    out[..., 3] *= 255
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGBA")


def paste_clip(canvas, im, x, y):
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + im.width), min(H, y + im.height)
    if x1 <= x0 or y1 <= y0:
        return
    canvas.alpha_composite(im.crop((x0 - x, y0 - y, x1 - x, y1 - y)), (x0, y0))


def put_text(canvas, s, kind, size, color, x, y, anchor="c", scale=1.0, alpha=1.0, satin=False, reveal=1.0):
    if alpha <= 0.01 or scale <= 0.02 or reveal <= 0:
        return
    im = text_rgba(s, kind, size, color, satin)
    if reveal < 1:
        im = im.copy()
        cut = int(im.width * reveal)
        im.paste((0, 0, 0, 0), (cut, 0, im.width, im.height))
    if abs(scale - 1) > 1e-3:
        im = im.resize((max(1, int(im.width * scale)), max(1, int(im.height * scale))), Image.BILINEAR)
    if alpha < 1:
        im = im.copy()
        im.putalpha(im.getchannel("A").point(lambda v: int(v * alpha)))
    if anchor == "c":
        x, y = x - im.width // 2, y - im.height // 2
    elif anchor == "r":
        x, y = x - im.width, y - im.height // 2
    else:
        y = y - im.height // 2
    paste_clip(canvas, im, int(x), int(y))


def dashed_rect(d, box, color, dash=10, gap=7, width=3, r=0):
    x0, y0, x1, y1 = box
    for (ax, ay, bx, by) in ((x0, y0, x1, y0), (x1, y0, x1, y1), (x1, y1, x0, y1), (x0, y1, x0, y0)):
        L = math.hypot(bx - ax, by - ay)
        k = 0.0
        while k < L:
            e = min(L, k + dash)
            d.line([(ax + (bx - ax) * k / L, ay + (by - ay) * k / L), (ax + (bx - ax) * e / L, ay + (by - ay) * e / L)],
                   fill=color, width=width)
            k += dash + gap


def ribbon(canvas, box, fill, border=THREAD, alpha=1.0):
    """fabric label: soft shadow, rounded rect, dashed stitched border inset."""
    x0, y0, x1, y1 = [int(v) for v in box]
    sh = Image.new("RGBA", (x1 - x0 + 60, y1 - y0 + 60), (0, 0, 0, 0))
    ImageDraw.Draw(sh).rounded_rectangle([30, 36, x1 - x0 + 30, y1 - y0 + 36], 18, fill=(60, 40, 20, int(90 * alpha)))
    sh_np = cv2.GaussianBlur(np.asarray(sh), (0, 0), 10)
    paste_clip(canvas, Image.fromarray(sh_np, "RGBA"), x0 - 30, y0 - 30)
    lab = Image.new("RGBA", (x1 - x0, y1 - y0), (0, 0, 0, 0))
    ld = ImageDraw.Draw(lab)
    ld.rounded_rectangle([0, 0, x1 - x0 - 1, y1 - y0 - 1], 18, fill=fill + (int(255 * alpha),))
    dashed_rect(ld, (10, 10, x1 - x0 - 11, y1 - y0 - 11), border + (int(255 * alpha),), width=3)
    paste_clip(canvas, lab, x0, y0)


# ------------------------------------------------------------------ stage 1: the giraffe quilt
def img_quilt(t):
    """front side; returns (img float P×P, tokens shown, bar fraction or None)."""
    a = assets()
    B = a["img_blocks"]
    cell_px = P // N
    pats, grids = a["pats_img"], a["grid_img"]
    if t < S1[0]:
        return a["cue"].copy(), 0, None
    if t < S2[0]:
        f = sweep_frac(t, S1)
        rows = int(f * N + 1e-6)
        oi, oj, b = B[1]
        ids = block_ids(oi, oj, N)
        q = quilt(a["cue_cell"], ids, b, cell_px, pats, grids)
        img = a["cue"].copy()
        ry = rows * cell_px
        img[:ry] = q[:ry]
        return img, rows * N, (f if f < 1 else None)
    if t < S3[0]:
        old, new, win = B[1], B[2], S2
    else:
        old, new, win = B[2], B[8], S3
    f = sweep_frac(t, win)
    t_sw = win[0] + (new[0] + new[2]) / N * (win[1] - win[0])
    use = t >= t_sw
    oi = np.where(use, new[0], old[0])
    oj = np.where(use, new[1], old[1])
    sz = np.where(use, new[2], old[2])
    ids = block_ids(oi, oj, N)
    merged = new[2] > old[2]
    dt = t - t_sw
    flash = np.where(merged & (dt >= 0), np.exp(-np.maximum(dt, 0) / 0.14), 0).astype(np.float32)
    cols = mosaic(a["cue_cell"], ids, N * N)
    old_ids_px = up(block_ids(old[0], old[1], N), cell_px)
    g_a = up(np.where(merged & (dt >= 0), np.exp(-np.maximum(dt, 0) / 0.22), 0).astype(np.float32), cell_px)
    q = quilt(cols, ids, sz, cell_px, pats, grids, flash=flash, ghost=(edge_thin(old_ids_px), g_a))
    return q, len(np.unique(ids)), (f if 0 < f < 1 else None)


def img_back(t):
    """back side: the generated image, quilted with the final layout (lighter puff)."""
    a = assets()
    oi, oj, b = a["img_blocks"][8]
    ids = block_ids(oi, oj, N)
    cell_px = P // N
    puff = 0.55 - 0.25 * ease((t - (T_FLIP + FLIP_D + 0.6)) / 0.8)
    # quilt() tints a per-cell colour by fabric; for the back use the image itself with seams + puff only
    ones = np.ones((N, N, 3), np.float32)
    shade = quilt(ones, ids, b, cell_px, None, a["grid_img"], puff=puff)
    return a["gen"] * shade


# ------------------------------------------------------------------ bonus: video quilt
def vid_V(tt):
    v = 0.0
    segs = [(T_V0, T_VB1, VID_SPEED[0]), (T_VB1, T_VB2, VID_SPEED[1]), (T_VB2, 99, VID_SPEED[2])]
    for s0, s1, sp in segs:
        if tt > s0:
            v += (min(tt, s1) - s0) * sp * 15
    return v


def vid_speed(t):
    if t < T_VB1:
        return VID_SPEED[0]
    return VID_SPEED[1] if t < T_VB2 else VID_SPEED[2]


def vid_frame(t):
    """frame blend so the 81-frame loop seam is a soft dissolve."""
    a = assets()
    v = vid_V(t)
    i = int(v) % 81
    k = v % 81
    if k > 77:
        u = (k - 77) / 4
        return a["vid"][i] * (1 - u) + a["vid"][0] * u, a["vid_cell"][i] * (1 - u) + a["vid_cell"][0] * u, i
    return a["vid"][i], a["vid_cell"][i], i


def vid_blocks(lv, cap):
    b = np.minimum(lv, cap).astype(int)
    ii, jj = np.mgrid[0:VN, 0:VM]
    return ii // b * b, jj // b * b, b


def vid_quilt(t):
    a = assets()
    frame, cell, idx = vid_frame(t)
    lv = a["lv"][idx]
    right = frame * 0.9 + np.array(LINEN, np.float32) * 0.1
    tok = 0
    bar = None
    pats, grids = a["pats_vid"], a["grid_vid"]
    if t < V1[0]:
        return frame, right, 0, None
    if t < V2[0]:
        f = sweep_frac(t, V1)
        rows = int(f * VN + 1e-6)
        oi, oj, b = vid_blocks(lv, 1)
        q = quilt(cell, block_ids(oi, oj, VM), b, 11, pats, grids, puff=0.8, seed=5)
        right = right.copy()
        right[:rows * 11] = q[:rows * 11]
        return frame, right, int(VID_TOK[0] * f), (f if f < 1 else None)
    if t < V3[0]:
        old, new, win, k = vid_blocks(lv, 1), vid_blocks(lv, 2), V2, 1
    else:
        old, new, win, k = vid_blocks(lv, 2), vid_blocks(lv, 4), V3, 2
    f = sweep_frac(t, win)
    t_sw = win[0] + (new[0] + new[2]) / VN * (win[1] - win[0])
    use = t >= t_sw
    oi = np.where(use, new[0], old[0])
    oj = np.where(use, new[1], old[1])
    sz = np.where(use, new[2], old[2])
    ids = block_ids(oi, oj, VM)
    merged = new[2] > old[2]
    dt = t - t_sw
    flash = np.where(merged & (dt >= 0), np.exp(-np.maximum(dt, 0) / 0.14), 0).astype(np.float32)
    cols = mosaic(cell, ids, VN * VM)
    old_px = up(block_ids(old[0], old[1], VM), 11)
    g_a = up(np.where(merged & (dt >= 0), np.exp(-np.maximum(dt, 0) / 0.22), 0).astype(np.float32), 11)
    q = quilt(cols, ids, sz, 11, pats, grids, flash=flash, ghost=(edge_thin(old_px), g_a), puff=0.8, seed=5)
    tok = int(VID_TOK[k - 1] + (VID_TOK[k] - VID_TOK[k - 1]) * f)
    return frame, q, tok, (f if 0 < f < 1 else None)


# ------------------------------------------------------------------ props
def shadow_under(canvas, x0, y0, w, h, lift=0.0):
    pad = 60
    sh = np.zeros((h + 2 * pad, w + 2 * pad), np.uint8)
    off = int(10 + 30 * lift)
    cv2.rectangle(sh, (pad + off // 2, pad + off), (pad + w + off // 2, pad + h + off), 255, -1)
    sh = cv2.GaussianBlur(sh, (0, 0), 14 + 16 * lift)
    rgba = np.zeros(sh.shape + (4,), np.uint8)
    rgba[..., :3] = (70, 45, 25)
    rgba[..., 3] = (sh * (0.45 - 0.15 * lift)).astype(np.uint8)
    paste_clip(canvas, Image.fromarray(rgba, "RGBA"), x0 - pad, y0 - pad)


def pin(d, x, y, col):
    d.line([(x, y), (x + 14, y + 20)], fill=(150, 150, 155), width=3)
    d.ellipse([x - 11, y - 11, x + 11, y + 11], fill=col, outline=(60, 30, 30), width=2)
    d.ellipse([x - 6, y - 7, x - 1, y - 2], fill=(255, 255, 255))


def needle(d, x, y):
    """sewing-machine needle, tip at (x, y), thread from the eye."""
    d.line([(x + 3, y - 92), (x + 50, y - 150), (x + 120, y - 175)], fill=CRANBERRY, width=3)
    d.polygon([(x - 6, y - 150), (x + 6, y - 150), (x + 4, y - 20), (x, y), (x - 4, y - 20)], fill=(196, 198, 205),
              outline=(90, 90, 100))
    d.line([(x - 2, y - 148), (x - 2, y - 30)], fill=(245, 245, 250), width=2)
    d.ellipse([x - 2, y - 98, x + 2, y - 86], fill=(80, 80, 90))
    # presser foot
    d.rounded_rectangle([x - 28, y - 10, x + 28, y + 4], 4, fill=(160, 162, 170), outline=(90, 90, 100))
    d.rectangle([x - 8, y - 160, x + 8, y - 145], fill=(120, 122, 130))


def badge(canvas, cx, cy, r, val, col, t0, t):
    dt = t - t0
    if dt < 0:
        return
    s = back_out(dt / 0.35)
    R = int(r * s)
    if R < 4:
        return
    sz = 2 * r + 40
    im = Image.new("RGBA", (sz, sz), (0, 0, 0, 0))
    c = sz // 2
    # fabric disc with gingham
    disc = np.zeros((sz, sz, 4), np.float32)
    yy, xx = np.mgrid[0:sz, 0:sz]
    inside = (xx - c) ** 2 + (yy - c) ** 2 <= r * r
    chk = 1.1 - 0.1 * ((yy // 9) % 2) - 0.1 * ((xx // 9) % 2)
    disc[..., :3] = np.array(col, np.float32) * chk[..., None]
    disc[..., 3] = inside * 255
    shadow = Image.new("RGBA", (sz, sz), (0, 0, 0, 0))
    ImageDraw.Draw(shadow).ellipse([c - r + 4, c - r + 8, c + r + 4, c + r + 8], fill=(60, 35, 20, 90))
    im.alpha_composite(Image.fromarray(cv2.GaussianBlur(np.asarray(shadow), (0, 0), 6), "RGBA"))
    im.alpha_composite(Image.fromarray(np.clip(disc, 0, 255).astype(np.uint8), "RGBA"))
    d = ImageDraw.Draw(im)
    prog = clamp(dt / 0.6)                                   # stitches run around the border
    n = 40
    for k in range(int(n * prog)):
        a0 = 2 * math.pi * k / n - math.pi / 2
        a1 = a0 + 2 * math.pi / n * 0.55
        rr = r - 12
        d.line([(c + rr * math.cos(a0), c + rr * math.sin(a0)), (c + rr * math.cos(a1), c + rr * math.sin(a1))],
               fill=THREAD, width=4)
    t1 = text_rgba(val, "bold", int(r * 0.62), THREAD, True)
    im.alpha_composite(t1, (c - t1.width // 2, c - t1.height // 2 - int(r * 0.16)))
    t2 = text_rgba("faster", "hand", int(r * 0.36), THREAD, False)
    im.alpha_composite(t2, (c - t2.width // 2, c + int(r * 0.30) - t2.height // 2))
    if abs(s - 1) > 1e-3:
        im = im.resize((max(2, int(sz * s)), max(2, int(sz * s))), Image.BILINEAR)
    paste_clip(canvas, im, cx - im.width // 2, cy - im.height // 2)


def caption(canvas, s, t, t0, t1, y=1012, size=58):
    if t0 <= t < t1:
        a = clamp((t - t0) / 0.2) * (1 - clamp((t - (t1 - 0.2)) / 0.2))
        put_text(canvas, s, "hand", size, BROWN, W // 2, y, alpha=a, scale=0.96 + 0.04 * ease_out((t - t0) / 0.3))


def count_tag(canvas, x0, y0, tokens, speed, alpha=1.0, w=520):
    ribbon(canvas, (x0, y0, x0 + w, y0 + 210), CREAM, border=CRANBERRY, alpha=alpha)
    put_text(canvas, "patches", "hand", 40, BROWN, x0 + w // 2, y0 + 42, alpha=alpha)
    put_text(canvas, f"{tokens:,}", "bold", 92, CRANBERRY, x0 + w // 2, y0 + 112, alpha=alpha, satin=True)
    put_text(canvas, f"= tokens  ·  speed {speed:.1f}×", "hand", 34, BROWN, x0 + w // 2, y0 + 176, alpha=alpha)


# ------------------------------------------------------------------ frame
def render(t):
    a = assets()
    canvas = Image.fromarray(a["bg"].astype(np.uint8)).convert("RGBA")
    d = ImageDraw.Draw(canvas)

    # ---------------------------------------------- title label
    if t < T_DROP + 0.3:
        out = clamp((t - T_TITLE1) / 0.35)
        al = 1 - out
        y_off = -int(500 * ease(out))
        ribbon(canvas, (360, 330 + y_off, 1560, 750 + y_off), CREAM, border=CRANBERRY, alpha=al)
        put_text(canvas, "Level-of-Token", "bold", 120, CRANBERRY, W // 2, 470 + y_off, satin=True, alpha=al,
                 reveal=clamp((t - 0.15) / 0.6))
        put_text(canvas, "Diffusion", "bold", 120, NAVY, W // 2, 590 + y_off, satin=True, alpha=al,
                 reveal=clamp((t - 0.6) / 0.5))
        put_text(canvas, "a patchwork of tokens", "hand", 48, BROWN, W // 2, 690 + y_off,
                 alpha=al * clamp((t - 1.1) / 0.3))
        if t < 1.15:
            nx = 760 + int(400 * clamp((t - 0.15) / 0.95))
            needle(d, nx, 520 + y_off)

    # ---------------------------------------------- stage 1: giraffe quilt
    if T_DROP <= t < T_V0:
        u = ease_out((t - T_DROP) / 0.45)
        slide = -int((W) * ease((t - T_OUT) / 0.25)) if t >= T_OUT else 0
        lift = 0.0
        if t < T_FLIP:
            img, tok, bar = img_quilt(t)
            img = wrinkle(img)
            pim_w = P
        else:
            v = clamp((t - T_FLIP) / FLIP_D)
            ang = math.pi * ease(v)
            lift = math.sin(ang)
            if ang < math.pi / 2:
                img, tok, bar = img_quilt(T_FLIP - 1e-3)
                img = wrinkle(img)
            else:
                img = wrinkle(img_back(t))
                tok, bar = 1027, None
            img = img * (0.75 + 0.25 * abs(math.cos(ang)))
            pim_w = max(2, int(P * abs(math.cos(ang))))
        sc = 1.12 - 0.12 * u
        sz_w, sz_h = int(pim_w * sc * (1 + 0.04 * lift)), int(P * sc * (1 + 0.04 * lift))
        pim = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
        if (sz_w, sz_h) != (P, P):
            pim = pim.resize((max(2, sz_w), max(2, sz_h)), Image.BILINEAR)
        cx, cy = QX0 + P // 2 + slide, QY0 + P // 2 - int(14 * lift)
        ox, oy = cx - pim.width // 2, cy - pim.height // 2
        shadow_under(canvas, ox, oy, pim.width, pim.height, lift=max(lift, 1 - u))
        al = clamp((t - T_DROP) / 0.2)
        if al < 1:
            pim = pim.convert("RGBA")
            pim.putalpha(int(255 * al))
            paste_clip(canvas, pim, ox, oy)
        else:
            canvas.paste(pim, (ox, oy))
        d = ImageDraw.Draw(canvas)
        if t < S1[0] + 0.3:                                  # pins on the printed depth fabric
            for (px, py), col in zip(((ox + 22, oy + 22), (ox + pim.width - 22, oy + 22),
                                      (ox + 22, oy + pim.height - 22), (ox + pim.width - 22, oy + pim.height - 22)),
                                     (CRANBERRY, MUSTARD, SAGE, NAVY)):
                pin(d, px, py, col)
        if bar is not None:
            by = oy + int(bar * P)
            nx = ox + int(P * (0.5 + 0.5 * math.sin(t * 9.0)))
            d.line([(ox, by), (ox + P, by)], fill=CRANBERRY + (160,), width=2)
            needle(d, nx, by)
        # note: in focus → tiny patches
        if T_DROP + 0.5 <= t < S2[0]:
            al = clamp((t - T_DROP - 0.5) / 0.25) * (1 - clamp((t - S2[0] + 0.25) / 0.25))
            gx, gy = QX0 + int(0.84 * P), QY0 + int(0.48 * P)
            nx0, ny0 = 1190, 360
            ribbon(canvas, (nx0, ny0, nx0 + 360, ny0 + 150), (255, 246, 196), border=MUSTARD, alpha=al)
            put_text(canvas, "in focus", "hand", 56, BROWN, nx0 + 180, ny0 + 58, alpha=al)
            put_text(canvas, "→ tiny patches", "hand", 40, BROWN, nx0 + 180, ny0 + 112, alpha=al)
            if al > 0.5:
                d.line([(nx0, ny0 + 75), (gx + 20, gy)], fill=BROWN, width=5)
                d.polygon([(gx + 6, gy - 2), (gx + 34, gy - 16), (gx + 30, gy + 14)], fill=BROWN)
                pin(d, nx0 + 180, ny0 + 4, CRANBERRY)
        if S1[0] <= t:
            al = clamp((t - S1[0]) / 0.25) * (1 - clamp((t - T_OUT) / 0.2))
            sp = 1.0 if t < T_B1 else (1.9 if t < T_B2 else 2.4)
            count_tag(canvas, 1230 + slide, 170, tok, sp, alpha=al)
        if t < T_OUT + 0.25:
            badge(canvas, 1365 + slide, 600, 120, "1.9×", MUSTARD, T_B1, t)
            badge(canvas, 1625 + slide, 600, 120, "2.4×", CRANBERRY, T_B2, t)
        caption(canvas, "Depth map: the giraffe is in focus", t, T_DROP + 0.4, S1[0])
        caption(canvas, "Cut it into tiny patches: one per token", t, S1[0], S2[0] - 0.1)
        caption(canvas, "Sew the blurry patches together", t, S2[0], S3[0] - 0.1)
        caption(canvas, "Bigger patches, fewer tokens", t, S3[0], T_FLIP)
        caption(canvas, "Flip it over: sharp where it's in focus", t, T_FLIP + 0.1, T_OUT)

    # ---------------------------------------------- bonus: video quilt
    if t >= T_V0:
        frame, right, tok, bar = vid_quilt(t)
        u = ease_out((t - T_V0) / 0.45)
        slide = int(W * (1 - u))
        for k, (x0, im) in enumerate(((VX0L, frame), (VX0R, wrinkle_v(right)))):
            pim = Image.fromarray(np.clip(im, 0, 255).astype(np.uint8))
            xx0 = x0 + slide
            shadow_under(canvas, xx0, VY0, VW, VH)
            if k == 0:                                         # wooden frame
                d.rounded_rectangle([xx0 - 14, VY0 - 14, xx0 + VW + 13, VY0 + VH + 13], 10, fill=(150, 98, 58),
                                    outline=(96, 60, 34), width=3)
            paste_clip(canvas, pim.convert("RGBA"), xx0, VY0)
        d = ImageDraw.Draw(canvas)
        if bar is not None:
            by = VY0 + int(bar * VH)
            nx = VX0R + slide + int(VW * (0.5 + 0.5 * math.sin(t * 9.0)))
            needle(d, nx, by)
        put_text(canvas, "video", "hand", 48, BROWN, VX0L + slide, VY0 - 50, anchor="l")
        put_text(canvas, "its quilt", "hand", 48, BROWN, VX0R + slide, VY0 - 50, anchor="l")
        sp = vid_speed(t)
        # playback indicator
        n_tri = 1 if sp < 1.5 else (2 if sp < 2.3 else 3)
        px0, py0 = VX0L + slide, VY0 + VH + 58
        for k in range(n_tri):
            d.polygon([(px0 + k * 24, py0 - 14), (px0 + k * 24 + 22, py0), (px0 + k * 24, py0 + 14)], fill=CRANBERRY)
        put_text(canvas, f"playing at {sp:.1f}×", "hand", 44, BROWN, px0 + 30 + n_tri * 24, py0, anchor="l")
        if t >= V1[0]:
            al = clamp((t - V1[0]) / 0.25)
            sp_tok = 1.0 if t < T_VB1 else (2.2 if t < T_VB2 else 2.5)
            count_tag(canvas, 700, 715, tok, sp_tok, alpha=al)
        badge(canvas, 1440, 818, 100, "2.2×", MUSTARD, T_VB1, t)
        badge(canvas, 1680, 818, 100, "2.5×", CRANBERRY, T_VB2, t)
        caption(canvas, "Now a video: every frame is a quilt", t, V1[0], V2[0] - 0.1, y=1030, size=54)
        caption(canvas, "Sew the ground into bigger patches", t, V2[0], V3[0] - 0.1, y=1030, size=54)
        caption(canvas, "…and the sky bigger still", t, V3[0], T_END, y=1030, size=54)
        if t >= T_END:
            dt = t - T_END
            s = back_out(dt / 0.4)
            bw, bh = int(1240 * s), int(150 * s)
            if bw > 20:
                ribbon(canvas, (W // 2 - bw // 2, 990 - bh // 2, W // 2 + bw // 2, 990 + bh // 2), CRANBERRY, border=THREAD)
                put_text(canvas, "Level-of-Token Diffusion", "bold", 76, THREAD, W // 2, 975, satin=True, scale=s,
                         reveal=clamp((dt - 0.25) / 0.6))
                put_text(canvas, "tiny patches only where it matters", "hand", 36, CREAM, W // 2, 1035, scale=s,
                         alpha=clamp((dt - 0.8) / 0.3))

    f = np.asarray(canvas.convert("RGB"))
    return f


def wrinkle_v(img):
    return img


# ------------------------------------------------------------------ audio
SR = 48000


def bell(freq, dur, vol=0.2, decay=0.6):
    n = int(dur * SR)
    tt = np.arange(n) / SR
    s = (np.sin(2 * np.pi * freq * tt) * np.exp(-tt / decay)
         + 0.45 * np.sin(2 * np.pi * freq * 2.76 * tt) * np.exp(-tt / (decay * 0.35))
         + 0.22 * np.sin(2 * np.pi * freq * 5.40 * tt) * np.exp(-tt / (decay * 0.18)))
    s[:30] *= np.linspace(0, 1, 30)
    return s * vol


def noise(dur, vol=0.3, decay=0.15, seed=0):
    n = int(dur * SR)
    rng = np.random.default_rng(seed)
    return rng.uniform(-1, 1, n) * np.exp(-np.arange(n) / SR / decay) * vol


def lowpass(x, k):
    c = np.cumsum(np.concatenate([np.zeros(k), x]))
    return (c[k:] - c[:-k]) / k


def note(name):
    names = {"C": -9, "C#": -8, "D": -7, "D#": -6, "E": -5, "F": -4, "F#": -3, "G": -2, "G#": -1, "A": 0, "A#": 1, "B": 2}
    return 440 * 2 ** ((names[name[:-1]] + 12 * (int(name[-1]) - 4)) / 12)


def music_speed(t):
    if t < T_B1:
        return 1.0
    if t < T_B2:
        return 1.9
    if t < T_V0:
        return 2.4
    return vid_speed(t)


def build_audio(path):
    out = np.zeros(int((DUR + 0.5) * SR))

    def add(sig, t0):
        i = int(t0 * SR)
        j = min(len(out), i + len(sig))
        if j > i:
            out[i:j] += sig[:j - i]

    # music box, tempo gently follows the speed-ups
    tune = ["E5", "G5", "A5", "G5", "E5", "D5", "C5", "D5", "E5", "G5", "C6", "B5", "A5", "G5", "E5", "G5"]
    t, k = 0.15, 0
    while t < DUR - 0.6:
        add(bell(note(tune[k % 16]), 1.2, vol=0.10, decay=0.5), t)
        if k % 4 == 0:
            add(bell(note(["C4", "A3", "F3", "G3"][(k // 4) % 4]), 1.6, vol=0.07, decay=0.8), t)
        t += 0.30 / math.sqrt(music_speed(t))
        k += 1
    # sewing machine during sweeps: clicks + motor hum
    for win in (S1, S2, S3, V1, V2, V3):
        tt = win[0]
        j = 0
        while tt < win[1]:
            add(noise(0.008, 0.22, 0.003, j + int(win[0] * 100)), tt)
            add(np.sin(2 * np.pi * 950 * np.arange(int(0.005 * SR)) / SR) * 0.06, tt)
            tt += 1 / 24
            j += 1
        n = int((win[1] - win[0]) * SR)
        ts = np.arange(n) / SR
        hum = 0.05 * (2 * ((ts * 110) % 1) - 1) * (0.7 + 0.3 * np.sin(2 * np.pi * 24 * ts))
        env = np.minimum(1, np.minimum(ts / 0.05, (ts[-1] - ts) / 0.08))
        add(lowpass(hum * env, 6), win[0])
    # soft pops when patches merge
    for win in (S2, S3, V2, V3):
        for j in range(14):
            tk = win[0] + (j + 0.5) / 14 * (win[1] - win[0])
            n = int(0.07 * SR)
            ts = np.arange(n) / SR
            f = 650 * np.exp(-ts * 14) + 260
            add(np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-ts / 0.025) * 0.22, tk)
    # badges: two-note chime
    for t0 in (T_B1, T_B2, T_VB1, T_VB2):
        add(bell(note("G5"), 1.4, 0.22, 0.6), t0)
        add(bell(note("C6"), 1.6, 0.24, 0.7), t0 + 0.12)
    # flip whoosh
    n = int(0.9 * SR)
    ts = np.arange(n) / SR
    add(lowpass(np.random.default_rng(4).uniform(-1, 1, n), 18) * np.sin(np.pi * ts / ts[-1]) ** 2 * 0.6, T_FLIP - 0.05)
    # title + end cadences
    for k, nm in enumerate(["C5", "E5", "G5", "C6"]):
        add(bell(note(nm), 1.6, 0.16, 0.7), 0.15 + k * 0.16)
        add(bell(note(nm), 2.0, 0.18, 0.9), T_END + k * 0.16)
    out = out / max(1e-6, np.abs(out).max()) * 0.89
    wavfile.write(path, SR, (out * 32767).astype(np.int16))


# ------------------------------------------------------------------ main
def render_scaled(args):
    t, w, h = args
    f = render(t)
    if w != W:
        f = cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA)
    return f.tobytes()


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "l"
    if mode == "still":
        os.makedirs(os.path.join(HERE, "scratch"), exist_ok=True)
        for s in sys.argv[2:]:
            Image.fromarray(render(float(s))).save(os.path.join(HERE, f"scratch/still_{TAG}_{float(s):05.2f}.png"))
        return
    w, h, fps = (1920, 1080, 60) if mode == "h" else (960, 540, 30)
    res = "1080p60" if mode == "h" else "540p30"
    os.makedirs(os.path.join(HERE, "media/final"), exist_ok=True)
    wav = os.path.join(HERE, f"media/final/LoT_quilt_{TAG}.wav")
    build_audio(wav)
    out = os.path.join(HERE, f"media/final/LoT_quilt_{TAG}_{res}.mp4")
    nf = int(DUR * fps)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps),
           "-i", "-", "-i", wav, "-c:v", "libx264", "-preset", "slow", "-crf", "16", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-ac", "2", "-ar", "48000", "-b:a", "192k", "-t", f"{DUR:.3f}", "-movflags", "+faststart", out]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    with Pool(10) as pool:
        for k, buf in enumerate(pool.imap(render_scaled, [(i / fps, w, h) for i in range(nf)], chunksize=2)):
            p.stdin.write(buf)
            if k % 120 == 0:
                print(f"frame {k}/{nf}", flush=True)
    p.stdin.close()
    p.wait()
    print("wrote", out)


if __name__ == "__main__":
    main()
