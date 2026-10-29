"""LoT Diffusion, 20-s cut-paper stop-motion clip, v1 (numpy + PIL + cv2 frames → ffmpeg, synthesized foley). No manim.

Animated on fours: 300 unique poses at 15 fps, each held 4 output frames at 60 fps, with per-pose jitter.
  0.0–1.5   a printed depth map slides onto the craft table (the giraffe is in focus)
  1.5–3.6   scissors cut it into 64 × 64 construction-paper squares, one colour each (4,096 tokens)
  4.4–5.8   a glue stick fuses blurry squares into 2 × 2 pieces → 1,606 pieces, stamp 1.9X FASTER
  6.9–8.2   second pass → 4 × 4 / 8 × 8 = the real LoT layout, 1,027 pieces, stamp 2.4X FASTER
  9.3–10.8  pieces flip over one by one: the generated image is printed on their backs
 12.3–13.0  table cleared; a flip-book of the platformer video and its paper tokens slide in
 13.2–17.3  cut every frame (75,600) → glue the ground (38,046, 2.2X) → glue the sky (34,698, 2.5X);
            the flip-book plays faster after each stamp
 18.4–20.0  torn-paper title banner

Speedups as in ../arcade_clip (image: fit of the paper's LoT-FLUX.2 4B timings; video: Tab. 2 LoT-Wan2.1 14B).
Usage: python render_cut_paper_v1.py h          → media/final/LoT_cut_paper_v1_1080p60.mp4
       python render_cut_paper_v1.py still t ... → scratch/still_v1_<t>.png
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
HAND = os.path.join(HERE, "data/GochiHand-Regular.ttf")
MARKER = os.path.join(HERE, "data/PermanentMarker-Regular.ttf")
CUE = os.path.join(ROOT, "data/img/img_000000000025_dof_cue.png")
GEN = os.path.join(ROOT, "data/img/img_000000000025_dof_generated_4b.png")
LAYOUT = os.path.join(ROOT, "data/layouts/img_000000000025_dof.json")
VID = os.path.join(ROOT, "data/clips/vd_platformer_gen")
LEVELS = os.path.join(ROOT, "arcade_clip/data/platformer_levels.npy")

W, H = 1920, 1080
FPS, POSE_FPS = 60, 15
DUR = 20.0
NPOSE = int(DUR * POSE_FPS)

INK = (40, 34, 30)
RED_INK = (196, 32, 40)
YELLOW_NOTE = (255, 226, 92)
PINK_NOTE = (255, 160, 190)
GREEN_NOTE = (160, 226, 150)
BLUE_NOTE = (150, 205, 245)
TAPE = (232, 222, 190)
PAPER_W = (247, 243, 232)

# ------------------------------------------------------------------ timeline (s)
T_IN = 0.0
CUT = (1.5, 3.4)
G1 = (4.4, 5.8)
G2 = (6.9, 8.2)
T_ST1, T_ST2 = 5.9, 8.3
T_FLIP = 9.3
T_OUT = 12.3
T_B0 = 12.65
BCUT = (13.2, 14.2)
BG1 = (14.4, 15.3)
BG2 = (16.4, 17.2)
T_BST1, T_BST2 = 15.4, 17.3
T_END = 18.4

IMG_TOK = (4096, 1606, 1027)
VID_TOK = (75600, 38046, 34698)
VID_SPEED = (1.0, 2.2, 2.5)

# ------------------------------------------------------------------ geometry
P, N, CELL = 768, 64, 12
BX, BY = (W - P) // 2, 140
VW, VH, VN, VM, VCELL = 800, 450, 45, 80, 10
VXL, VXR, VY = 110, 1010, 250


def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def ease(x):
    x = clamp(x)
    return x * x * (3 - 2 * x)


def sweep_frac(t, win):
    return clamp((t - win[0]) / (win[1] - win[0]))


# ------------------------------------------------------------------ procedural textures
def value_noise(h, w, cells, rng):
    g = rng.uniform(-1, 1, (cells[0] + 1, cells[1] + 1)).astype(np.float32)
    return cv2.resize(g, (w, h), interpolation=cv2.INTER_CUBIC)


def make_table(rng):
    base = np.array([178, 136, 94], np.float32)
    n = 10 * value_noise(H, W, (6, 10), rng) + 6 * value_noise(H, W, (30, 50), rng)
    img = base + n[..., None]
    im = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(im, "RGBA")
    for _ in range(3500):                                       # kraft fibres
        x, y = rng.uniform(0, W), rng.uniform(0, H)
        a = rng.uniform(0, np.pi)
        L = rng.uniform(6, 30)
        c = (230, 200, 160, 50) if rng.random() < 0.5 else (90, 60, 35, 45)
        d.line([x, y, x + L * math.cos(a), y + L * math.sin(a)], fill=c, width=1)
    img = np.asarray(im).astype(np.float32)
    img += rng.normal(0, 4, img.shape).astype(np.float32)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    v = 1 - 0.25 * (((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
    return np.clip(img * v[..., None], 0, 255)


def make_paper_tex(rng):
    t = 1 + 0.035 * value_noise(H, W, (40, 70), rng) + rng.normal(0, 0.018, (H, W)).astype(np.float32)
    im = Image.fromarray(np.full((H, W), 128, np.uint8))
    d = ImageDraw.Draw(im)
    for _ in range(9000):
        x, y = rng.uniform(0, W), rng.uniform(0, H)
        a = rng.uniform(0, np.pi)
        L = rng.uniform(3, 12)
        d.line([x, y, x + L * math.cos(a), y + L * math.sin(a)], fill=int(rng.choice([110, 146])), width=1)
    f = (np.asarray(im).astype(np.float32) - 128) / 128 * 0.25
    return (t + f).astype(np.float32)


def make_warp(h, w, cell, rng, amp=1.6):
    """hand-cut edges: token lookups at slightly wobbly coordinates (static per piece)."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dy = amp * value_noise(h, w, (h // 9, w // 9), rng)
    dx = amp * value_noise(h, w, (h // 9, w // 9), rng)
    wy = np.clip(((yy + dy) // cell).astype(int), 0, h // cell - 1)
    wx = np.clip(((xx + dx) // cell).astype(int), 0, w // cell - 1)
    return wy, wx


# ------------------------------------------------------------------ assets
def load_assets():
    rng = np.random.default_rng(1)
    a = {}
    a["table"] = make_table(rng)
    a["ptex"] = make_paper_tex(rng)
    cue = np.asarray(Image.open(CUE).convert("RGB")).astype(np.float32)
    a["cue"] = cv2.resize(cue, (P, P), interpolation=cv2.INTER_AREA)
    a["cue_cell"] = cue.reshape(N, 16, N, 16, 3).mean((1, 3))
    gen = np.asarray(Image.open(GEN).convert("RGB")).astype(np.float32)
    a["gen"] = cv2.resize(gen, (P, P), interpolation=cv2.INTER_AREA)
    rects = [tuple(r) for r in json.load(open(LAYOUT))["rects"]]
    size = np.zeros((N, N), int)
    r0m = np.zeros((N, N), int)
    c0m = np.zeros((N, N), int)
    for r0, r1, c0, c1 in rects:
        size[r0:r1, c0:c1] = r1 - r0
        r0m[r0:r1, c0:c1] = r0
        c0m[r0:r1, c0:c1] = c0
    ii, jj = np.mgrid[0:N, 0:N]
    a["img_blocks"] = {}
    for cap in (1, 2, 8):
        b = np.minimum(size, cap)
        a["img_blocks"][cap] = (r0m + (ii - r0m) // b * b, c0m + (jj - c0m) // b * b, b)
    a["warp_img"] = make_warp(P, P, CELL, rng)
    a["warp_vid"] = make_warp(VH, VW, VCELL, rng, amp=1.2)
    a["lum"] = rng.uniform(0.94, 1.05, N * N * 4).astype(np.float32)
    a["lum_v"] = rng.uniform(0.95, 1.05, VN * VM).astype(np.float32)
    a["flip_rand"] = rng.uniform(0, 0.3, N * N).astype(np.float32)
    a["lv"] = np.load(LEVELS)
    a["vid"] = [cv2.resize(np.asarray(Image.open(os.path.join(VID, f"{f + 1:04d}.jpg")).convert("RGB")),
                           (VW, VH), interpolation=cv2.INTER_AREA).astype(np.float32) for f in range(81)]
    a["vid_cell"] = [cv2.resize(v, (VM, VN), interpolation=cv2.INTER_AREA) for v in a["vid"]]
    return a


A = None


def assets():
    global A
    if A is None:
        A = load_assets()
    return A


# ------------------------------------------------------------------ token helpers (from arcade_clip)
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


def staged(old, new, n, win, t):
    oi_n, oj_n, b_n = new
    t_sw = win[0] + (oi_n + b_n) / n * (win[1] - win[0])
    use = t >= t_sw
    return np.where(use, oi_n, old[0]), np.where(use, oj_n, old[1]), np.where(use, b_n, old[2]), t_sw


# ------------------------------------------------------------------ paper compositing
def edges_of(ids_px):
    e = np.zeros(ids_px.shape, bool)
    d1 = ids_px[:, 1:] != ids_px[:, :-1]
    d2 = ids_px[1:, :] != ids_px[:-1, :]
    e[:, 1:] |= d1
    e[:, :-1] |= d1
    e[1:, :] |= d2
    e[:-1, :] |= d2
    return e


def composite(canvas, rgb, alpha, x0, y0, ptex_off=(0, 0), shadow=(5, 6), shade=0.42, blur=3.0):
    """paste a paper layer (h, w, 3) with alpha (h, w) onto canvas (H, W, 3 float) with a soft drop shadow."""
    h, w = alpha.shape
    pad = 20
    X0, Y0 = x0 - pad, y0 - pad
    al = np.zeros((h + 2 * pad, w + 2 * pad), np.float32)
    al[pad + shadow[1]:pad + shadow[1] + h, pad + shadow[0]:pad + shadow[0] + w] = alpha
    sh = cv2.GaussianBlur(al, (0, 0), blur)
    # clip to canvas
    cx0, cy0 = max(0, X0), max(0, Y0)
    cx1, cy1 = min(W, X0 + al.shape[1]), min(H, Y0 + al.shape[0])
    if cx1 <= cx0 or cy1 <= cy0:
        return
    s = sh[cy0 - Y0:cy1 - Y0, cx0 - X0:cx1 - X0]
    canvas[cy0:cy1, cx0:cx1] *= (1 - shade * s)[..., None]
    px0, py0 = max(0, x0), max(0, y0)
    px1, py1 = min(W, x0 + w), min(H, y0 + h)
    if px1 <= px0 or py1 <= py0:
        return
    a = assets()
    tx, ty = ptex_off
    tex = a["ptex"][(py0 + ty) % H:(py0 + ty) % H + (py1 - py0), (px0 + tx) % W:(px0 + tx) % W + (px1 - px0)]
    if tex.shape != (py1 - py0, px1 - px0):
        tex = np.ones((py1 - py0, px1 - px0), np.float32)
    src = rgb[py0 - y0:py1 - y0, px0 - x0:px1 - x0] * tex[..., None]
    al2 = alpha[py0 - y0:py1 - y0, px0 - x0:px1 - x0][..., None]
    canvas[py0:py1, px0:px1] = canvas[py0:py1, px0:px1] * (1 - al2) + src * al2


def paste_rgba(canvas, im, x, y, shadow=(4, 5), shade=0.35):
    arr = np.asarray(im).astype(np.float32)
    composite(canvas, arr[..., :3], arr[..., 3] / 255.0, int(x), int(y), shadow=shadow, shade=shade)


def paste_rgba_flat(canvas, im, x, y):
    """ink / tape with no shadow and no paper texture."""
    arr = np.asarray(im).astype(np.float32)
    h, w = arr.shape[:2]
    x, y = int(x), int(y)
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return
    sub = arr[y0 - y:y1 - y, x0 - x:x1 - x]
    al = sub[..., 3:] / 255.0
    canvas[y0:y1, x0:x1] = canvas[y0:y1, x0:x1] * (1 - al) + sub[..., :3] * al


# ------------------------------------------------------------------ text, notes, tape, stamps
@lru_cache(maxsize=64)
def font(path, size):
    return ImageFont.truetype(path, size)


def text_size(s, path, size):
    f = font(path, size)
    b = f.getbbox(s)
    return b[2] - b[0], b[3] - b[1], b


@lru_cache(maxsize=512)
def note_img(lines, w, h, color, rot, seed):
    """sticky note: lines = ((text, size, (r,g,b), reveal 0..1, strike bool, path), ...)."""
    rng = np.random.default_rng(seed)
    im = Image.new("RGBA", (w, h), color + (255,))
    d = ImageDraw.Draw(im)
    # sticky strip at the top, slightly darker
    d.rectangle([0, 0, w, int(h * 0.13)], fill=tuple(int(c * 0.93) for c in color) + (255,))
    y = int(h * 0.17)
    for text, size, col, rev, strike, path in lines:
        f = font(path, size)
        tw, th, b = text_size(text, path, size)
        x = (w - tw) // 2 - b[0]
        if rev > 0:
            layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
            ImageDraw.Draw(layer).text((x, y - b[1]), text, font=f, fill=col + (255,))
            if rev < 1:
                cut = int(x + b[0] + tw * rev)
                ImageDraw.Draw(layer).rectangle([cut, 0, w, h], fill=(0, 0, 0, 0))
            im.alpha_composite(layer)
            if strike:
                yy = y + th // 2
                d.line([x - 6, yy + rng.integers(-3, 3), x + tw + 10, yy + rng.integers(-6, 0)], fill=RED_INK + (255,), width=6)
        y += th + int(size * 0.45)
    # curl shading at the bottom
    g = np.linspace(0, 1, h)[:, None] ** 6 * 30
    arr = np.asarray(im).astype(np.float32)
    arr[..., :3] -= g[..., None]
    im = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    return im.rotate(rot, resample=Image.BICUBIC, expand=True)


@lru_cache(maxsize=256)
def tape_img(text, size, rot, seed):
    rng = np.random.default_rng(seed)
    tw, th, b = text_size(text, HAND, size)
    w, h = tw + 70, th + 34
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    # jagged short ends
    left = [(rng.integers(0, 8), y) for y in range(0, h + 1, 6)]
    right = [(w - 1 - rng.integers(0, 8), y) for y in range(h, -1, -6)]
    d.polygon(left + right, fill=TAPE + (225,))
    d.text(((w - tw) // 2 - b[0], (h - th) // 2 - b[1]), text, font=font(HAND, size), fill=INK + (255,))
    return im.rotate(rot, resample=Image.BICUBIC, expand=True)


@lru_cache(maxsize=64)
def stamp_img(text, size, rot, seed):
    rng = np.random.default_rng(seed)
    tw, th, b = text_size(text, MARKER, size)
    w, h = tw + 60, th + 50
    im = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([6, 6, w - 7, h - 7], radius=14, outline=255, width=7)
    d.text(((w - tw) // 2 - b[0], (h - th) // 2 - b[1]), text, font=font(MARKER, size), fill=255)
    m = np.asarray(im).astype(np.float32) / 255
    speck = cv2.GaussianBlur(rng.uniform(0, 1, m.shape).astype(np.float32), (0, 0), 1.2)
    m = m * (speck > 0.42) * rng.uniform(0.8, 1.0, m.shape)
    out = np.zeros((h, w, 4), np.uint8)
    out[..., :3] = RED_INK
    out[..., 3] = (m * 225).astype(np.uint8)
    return Image.fromarray(out).rotate(rot, resample=Image.BICUBIC, expand=True)


@lru_cache(maxsize=64)
def strip_img(text, path, size, color, rot, seed, pad=(60, 30)):
    """torn paper strip with text."""
    rng = np.random.default_rng(seed)
    tw, th, b = text_size(text, path, size)
    w, h = tw + 2 * pad[0], th + 2 * pad[1]
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    top = [(x, rng.integers(0, 9)) for x in range(0, w + 1, 9)]
    bot = [(x, h - 1 - rng.integers(0, 9)) for x in range(w, -1, -9)]
    d.polygon(top + bot, fill=PAPER_W + (255,))
    d.text(((w - tw) // 2 - b[0], (h - th) // 2 - b[1]), text, font=font(path, size), fill=color + (255,))
    return im.rotate(rot, resample=Image.BICUBIC, expand=True)


# ------------------------------------------------------------------ tools
@lru_cache(maxsize=32)
def scissors_img(open_deg, rot):
    S = 320
    im = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    cx, cy = S // 2, S // 2
    for sgn in (1, -1):
        a = math.radians(sgn * open_deg / 2)
        # blade towards -x (cutting direction left), handle towards +x
        tip = (cx - 150 * math.cos(a), cy - 150 * math.sin(a))
        nrm = (-math.sin(a) * sgn * 10, math.cos(a) * sgn * 10)
        d.polygon([(cx, cy - 4 * sgn), tip, (cx - 40 * math.cos(a) + nrm[0], cy - 40 * math.sin(a) + nrm[1] + 6 * sgn)],
                  fill=(205, 210, 215, 255), outline=(120, 125, 130, 255))
        hx, hy = cx + 70 * math.cos(-a), cy + 70 * math.sin(-a) + 18 * sgn
        d.line([(cx, cy), (hx - 20, hy)], fill=(230, 80, 40, 255), width=14)
        d.ellipse([hx - 34, hy - 24, hx + 34, hy + 24], outline=(230, 80, 40, 255), width=13)
    d.ellipse([cx - 8, cy - 8, cx + 8, cy + 8], fill=(90, 90, 95, 255))
    return im.rotate(rot, resample=Image.BICUBIC, expand=False)


@lru_cache(maxsize=8)
def glue_img(rot):
    im = Image.new("RGBA", (300, 300), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([60, 120, 250, 180], radius=14, fill=(120, 70, 190, 255))
    d.rectangle([60, 120, 90, 180], fill=(235, 235, 240, 255))
    d.rounded_rectangle([30, 128, 62, 172], radius=10, fill=(245, 245, 250, 255))
    d.text((120, 136), "GLUE", font=font(MARKER, 26), fill=(255, 255, 255, 255))
    return im.rotate(rot, resample=Image.BICUBIC, expand=False)


# ------------------------------------------------------------------ board (giraffe)
def board_layer(t):
    """returns rgb (P, P, 3), alpha (P, P) for the giraffe board at pose time t."""
    a = assets()
    wy, wx = a["warp_img"]
    B = a["img_blocks"]
    cue = a["cue"]
    shine = None
    if t < CUT[0]:
        rgb = cue.copy()
        alpha = np.ones((P, P), np.float32)
        return rgb, alpha, True
    if t < G1[0]:
        rows = int(sweep_frac(t, CUT) * N + 1e-6)
        oi, oj, b = B[1]
        ids = block_ids(oi, oj, N)
        cut_cell = oi < rows
        ids = np.where(cut_cell, ids, -1)
        cols = mosaic(a["cue_cell"], np.where(cut_cell, ids, N * N), N * N + 1)
        idp = ids[wy, wx]
        rgb = np.where((idp >= 0)[..., None], cols[wy, wx], cue)
    else:
        if t < G2[0]:
            old, new, win = B[1], B[2], G1
        else:
            old, new, win = B[2], B[8], G2
        oi, oj, b, t_sw = staged(old, new, N, win, t)
        ids = block_ids(oi, oj, N)
        merged = new[2] > old[2]
        dt = t - t_sw
        shine = np.where(merged & (dt >= 0), np.exp(-np.maximum(dt, 0) / 0.25), 0).astype(np.float32)
        cols = mosaic(a["cue_cell"], ids, N * N)
        idp = ids[wy, wx]
        rgb = cols[wy, wx]
    e = edges_of(idp)
    alpha = (~e).astype(np.float32)
    lum = a["lum"][np.maximum(idp, 0) % len(a["lum"])]
    rgb = rgb * lum[..., None]
    if shine is not None:
        rgb = rgb + (255 - rgb) * 0.35 * shine[wy, wx][..., None]
    return rgb, alpha, False


def flip_layer(t):
    """pieces of the final layout flip over to show the generated image on their backs."""
    a = assets()
    wy, wx = a["warp_img"]
    oi, oj, b = a["img_blocks"][8]
    ids = block_ids(oi, oj, N)
    cols = mosaic(a["cue_cell"], ids, N * N)
    front = cols[wy, wx] * a["lum"][ids[wy, wx]][..., None]
    t0 = T_FLIP + 0.85 * (oi + oj) / (2 * N) + a["flip_rand"][ids]
    p = np.clip((t - t0) / 0.3, 0, 1)
    pp = p[wy, wx]
    c = np.abs(np.cos(np.pi * pp))
    cx = ((oj + b / 2) * CELL)[wy, wx]
    hw = (b * CELL / 2)[wy, wx]
    xx = np.arange(P)[None, :].astype(np.float32)
    rel = xx - cx
    vis = np.abs(rel) <= hw * c + 0.5
    srcx = np.clip((cx + rel / np.maximum(c, 1e-3)).astype(int), 0, P - 1)
    yy = np.arange(P)[:, None].repeat(P, 1)
    back_side = pp > 0.5
    rgb = np.where(back_side[..., None], a["gen"][yy, srcx], front[yy, srcx])
    rgb = rgb * (0.7 + 0.3 * c)[..., None]
    idp = ids[wy, wx]
    e = edges_of(idp)
    line_fade = clamp((t - (T_FLIP + 1.4)) / 0.6)               # seams close up once all are flipped
    alpha = (vis & ~e).astype(np.float32)
    alpha = np.maximum(alpha, (vis & e).astype(np.float32) * line_fade * 0.0)
    if line_fade > 0:
        done = pp >= 1
        alpha = np.where(done & e, line_fade * 0.6, alpha)
    return rgb, alpha


# ------------------------------------------------------------------ bonus (video)
def vid_V(t):
    v = 0.0
    segs = [(T_B0, T_BST1, VID_SPEED[0]), (T_BST1, T_BST2, VID_SPEED[1]), (T_BST2, 99, VID_SPEED[2])]
    for s0, s1, sp in segs:
        if t > s0:
            v += (min(t, s1) - s0) * sp * 15
    return v


def vid_speed(t):
    return VID_SPEED[0] if t < T_BST1 else (VID_SPEED[1] if t < T_BST2 else VID_SPEED[2])


def vid_blocks(lv, cap):
    b = np.minimum(lv, cap).astype(int)
    ii, jj = np.mgrid[0:VN, 0:VM]
    return ii // b * b, jj // b * b, b


def bonus_layers(t):
    a = assets()
    idx = int(vid_V(t)) % 81
    frame = a["vid"][idx]
    cell = a["vid_cell"][idx]
    lv = a["lv"][idx]
    wy, wx = a["warp_vid"]
    tok = 0
    if t < BCUT[0]:
        return frame, frame.copy(), np.ones((VH, VW), np.float32), 0, idx
    shine = None
    if t < BG1[0]:
        f = sweep_frac(t, BCUT)
        rows = int(f * VN + 1e-6)
        oi, oj, b = vid_blocks(lv, 1)
        ids = block_ids(oi, oj, VM)
        cut = oi < rows
        ids = np.where(cut, ids, -1)
        cols = mosaic(cell, np.where(cut, ids, VN * VM), VN * VM + 1)
        idp = ids[wy, wx]
        right = np.where((idp >= 0)[..., None], cols[wy, wx], frame)
        tok = VID_TOK[0] * f
    else:
        if t < BG2[0]:
            old, new, win, k = vid_blocks(lv, 1), vid_blocks(lv, 2), BG1, 1
        else:
            old, new, win, k = vid_blocks(lv, 2), vid_blocks(lv, 4), BG2, 2
        oi, oj, b, t_sw = staged(old, new, VN, win, t)
        ids = block_ids(oi, oj, VM)
        dt = t - t_sw
        shine = np.where((new[2] > old[2]) & (dt >= 0), np.exp(-np.maximum(dt, 0) / 0.25), 0).astype(np.float32)
        cols = mosaic(cell, ids, VN * VM)
        idp = ids[wy, wx]
        right = cols[wy, wx]
        f = sweep_frac(t, win)
        tok = VID_TOK[k - 1] + (VID_TOK[k] - VID_TOK[k - 1]) * f
    e = edges_of(idp)
    alpha = (~e).astype(np.float32)
    right = right * a["lum_v"][np.maximum(idp, 0) % len(a["lum_v"])][..., None]
    if shine is not None:
        right = right + (255 - right) * 0.35 * shine[wy, wx][..., None]
    return frame, right, alpha, int(tok), idx


# ------------------------------------------------------------------ frame
def jitter(t, k, amp=1.5):
    rng = np.random.default_rng(int(round(t * POSE_FPS)) * 97 + k)
    return rng.uniform(-amp, amp, 2)


def captions(t):
    caps = [
        (0.5, 1.5, "a depth map: the giraffe is in focus"),
        (1.5, 2.6, "cut it into tokens..."),
        (2.6, 4.4, "the usual way: 4,096 tiny squares"),
        (4.4, 5.9, "LoT: glue the blurry ones together"),
        (5.9, 6.9, "fewer pieces = faster"),
        (6.9, 8.3, "bigger pieces where it's blurry"),
        (8.3, 9.3, "1,027 pieces"),
        (9.3, 10.8, "now flip them over..."),
        (10.8, 12.3, "same picture, 4X fewer pieces"),
        (13.0, 14.4, "video: cut every frame"),
        (14.4, 15.4, "glue the ground"),
        (15.4, 16.4, "2.2X faster, the flip-book speeds up"),
        (16.4, 17.3, "glue the sky"),
        (17.3, 18.4, "2.5X faster!"),
    ]
    for t0, t1, s in caps:
        if t0 <= t < t1:
            return s, t0
    return None, None


def draw_caption(canvas, t, y):
    s, t0 = captions(t)
    if s is None:
        return
    hs = sum(ord(c) * (i + 1) for i, c in enumerate(s)) % 1000          # stable across worker processes
    im = tape_img(s, 46, float(np.random.default_rng(hs).uniform(-1.5, 1.5)), hs)
    dy = 0 if t - t0 > 0.13 else 30                              # slaps down in two poses
    j = jitter(t, 5, 1.0)
    paste_rgba_flat(canvas, im, W // 2 - im.width // 2 + j[0], y - im.height // 2 + dy + j[1])


def count_lines(t):
    """left note: piece counts, old ones crossed out."""
    vals = []
    if t >= CUT[1]:
        vals.append(("4,096", CUT[1]))
    if t >= G1[1]:
        vals.append(("1,606", G1[1]))
    if t >= G2[1]:
        vals.append(("1,027", G2[1]))
    out = [("pieces", 44, INK, 1.0, False, HAND)]
    for k, (s, t0) in enumerate(vals):
        rev = clamp((t - t0) / 0.35)
        rev = round(rev * 4) / 4
        strike = k < len(vals) - 1
        out.append((s, 60, INK if not strike else (110, 100, 90), rev, strike, MARKER))
    return tuple(out)


def render(t):
    """t = pose time (multiple of 1/15)."""
    a = assets()
    canvas = a["table"].copy()

    # ---------------------------------------------- stage 1: giraffe board
    if t < T_OUT + 0.4:
        slide_in = (1 - ease(t / 0.5)) * 900
        slide_out = ease((t - T_OUT) / 0.35) * -1500 if t >= T_OUT else 0
        j = jitter(t, 1)
        x0 = int(BX + j[0] + slide_out)
        y0 = int(BY + j[1] + slide_in)
        if t < T_FLIP:
            rgb, alpha, whole = board_layer(t)
            if whole:                                            # uncut print with a white border
                bw = 22
                sheet = np.empty((P + 2 * bw, P + 2 * bw, 3), np.float32)
                sheet[:] = PAPER_W
                sheet[bw:-bw, bw:-bw] = rgb
                composite(canvas, sheet, np.ones(sheet.shape[:2], np.float32), x0 - bw, y0 - bw, shadow=(7, 9))
            else:
                composite(canvas, rgb, alpha, x0, y0, shadow=(4, 5), shade=0.5, blur=2.0)
        else:
            rgb, alpha = flip_layer(t)
            composite(canvas, rgb, alpha, x0, y0, shadow=(4, 5), shade=0.5, blur=2.0)

        # tools
        if CUT[0] <= t < CUT[1] + 0.15:
            f = sweep_frac(t, CUT)
            k = int(round(t * POSE_FPS))
            sc = scissors_img(26 if k % 2 else 6, 0)
            sy = y0 + f * P
            paste_rgba(canvas, sc, x0 + P - 40 - sc.width // 2 + 30 * math.sin(k * 1.3), sy - sc.height // 2, shadow=(10, 14))
        for win in (G1, G2):
            if win[0] <= t < win[1] + 0.1:
                f = sweep_frac(t, win)
                gx = x0 + P * (0.5 + 0.42 * math.sin(f * 2 * np.pi * 3.5))
                gy = y0 + f * P
                g = glue_img(35)
                paste_rgba(canvas, g, gx - 40, gy - g.height // 2 - 40, shadow=(12, 16))

        # notes
        if t >= CUT[1] - 0.1:
            n = note_img(count_lines(t), 300, 330, YELLOW_NOTE, 3.0, 11)
            paste_rgba(canvas, n, 150 + slide_out, 170)
        if t >= CUT[1]:
            spd = [("speed", 44, INK, 1.0, False, HAND)]
            spd.append(("1X", 64, INK, round(clamp((t - CUT[1] - 0.3) / 0.3) * 3) / 3, t >= T_ST1, MARKER))
            n = note_img(tuple(spd), 300, 250, PINK_NOTE, -2.5, 12)
            paste_rgba(canvas, n, W - 450 + slide_out, 170)
        if t >= G1[0] - 0.2:
            lines = (("blurry", 40, INK, 1.0, False, HAND), ("= big pieces", 40, INK, 1.0, False, HAND),
                     ("in focus", 40, INK, 1.0, False, HAND), ("= tiny pieces", 40, INK, 1.0, False, HAND))
            n = note_img(lines, 300, 330, GREEN_NOTE, -3.0, 13)
            dy = 0 if t - (G1[0] - 0.2) > 0.13 else -40
            paste_rgba(canvas, n, 150 + slide_out, 590 + dy)
        for t0, s, y, rot in ((T_ST1, "1.9X FASTER", 470, 8), (T_ST2, "2.4X FASTER", 680, -6)):
            if t >= t0:
                st = stamp_img(s, 58, rot, int(t0 * 10))
                sc = 1.25 if t - t0 < 0.07 else 1.0
                if sc != 1.0:
                    st = st.resize((int(st.width * sc), int(st.height * sc)), Image.BICUBIC)
                paste_rgba_flat(canvas, st, W - 300 - st.width // 2 + slide_out, y - st.height // 2)
                if t - t0 < 0.07:                                # the stamp block itself, lifting off
                    pass
        draw_caption(canvas, t, 1000)

    # ---------------------------------------------- bonus: flip-book
    if t >= T_B0:
        slide = (1 - ease((t - T_B0) / 0.35)) * 1400
        left, right, ralpha, tok, idx = bonus_layers(t)
        j = jitter(t, 2)
        # flip-book pad: stacked page edges under the left card
        for k in range(6, 0, -1):
            pg = np.full((VH + 24, VW + 24, 3), PAPER_W, np.float32) * (0.97 - 0.02 * k)
            composite(canvas, pg, np.ones(pg.shape[:2], np.float32), int(VXL - 12 + slide + 3 * k), int(VY - 12 + 3 * k),
                      shadow=(2, 3), shade=0.2)
        card = np.full((VH + 24, VW + 24, 3), PAPER_W, np.float32)
        card[12:-12, 12:-12] = left
        # page-flip highlight: a bright sliver sweeps across when the frame changes
        composite(canvas, card, np.ones(card.shape[:2], np.float32), int(VXL - 12 + slide + j[0]), int(VY - 12 + j[1]),
                  shadow=(5, 7))
        if t < BCUT[0]:
            card2 = np.full((VH + 24, VW + 24, 3), PAPER_W, np.float32)
            card2[12:-12, 12:-12] = right
            composite(canvas, card2, np.ones(card2.shape[:2], np.float32), int(VXR - 12 + slide - j[0]),
                      int(VY - 12 - j[1]), shadow=(5, 7))
        else:
            composite(canvas, right, ralpha, int(VXR + slide - j[0]), int(VY - j[1]), shadow=(3, 4), shade=0.5, blur=1.5)
        # tools
        if BCUT[0] <= t < BCUT[1] + 0.1:
            f = sweep_frac(t, BCUT)
            k = int(round(t * POSE_FPS))
            sc = scissors_img(26 if k % 2 else 6, 0)
            paste_rgba(canvas, sc, VXR + VW - 40 - sc.width // 2, VY + f * VH - sc.height // 2, shadow=(10, 14))
        for win in (BG1, BG2):
            if win[0] <= t < win[1] + 0.1:
                f = sweep_frac(t, win)
                g = glue_img(35)
                gx = VXR + VW * (0.5 + 0.42 * math.sin(f * 2 * np.pi * 3))
                paste_rgba(canvas, g, gx - 40, VY + f * VH - g.height // 2 - 40, shadow=(12, 16))
        # labels on tape
        for x, s in ((VXL + 30, "flip-book"), (VXR + 30, "paper tokens")):
            im = tape_img(s, 42, -2.0, len(s))
            paste_rgba_flat(canvas, im, x + slide, VY - 70)
        sp = vid_speed(t)
        sp_im = tape_img(f"{sp:.1f}X", 42, 2.0, int(sp * 10))
        paste_rgba_flat(canvas, sp_im, VXL + VW - sp_im.width - 10 + slide, VY - 70)
        if t < T_END:
            # count note (bottom left) and stamps (bottom right)
            if t >= BCUT[0]:
                vals = [("pieces", 40, INK, 1.0, False, HAND)]
                shown = [VID_TOK[0]] if t < BG1[1] else ([VID_TOK[0], VID_TOK[1]] if t < BG2[1] else list(VID_TOK))
                if t < BG1[0] and t < BCUT[1]:
                    shown = []
                for k, v in enumerate(shown):
                    vals.append((f"{v:,}", 46, INK if k == len(shown) - 1 else (110, 100, 90), 1.0, k < len(shown) - 1, MARKER))
                n = note_img(tuple(vals), 300, 290, YELLOW_NOTE, 2.5, 21)
                paste_rgba(canvas, n, 120 + slide, 755)
            for t0, s, x, rot in ((T_BST1, "2.2X FASTER", 1530, 7), (T_BST2, "2.5X FASTER", 1560, -5)):
                if t >= t0:
                    st = stamp_img(s, 54, rot, int(t0 * 10))
                    if t - t0 < 0.07:
                        st = st.resize((int(st.width * 1.25), int(st.height * 1.25)), Image.BICUBIC)
                    yy = 800 if s.startswith("2.2") else 940
                    paste_rgba_flat(canvas, st, x - st.width // 2, yy - st.height // 2)
            draw_caption(canvas, t, 905)
        else:
            dt = t - T_END
            s1 = strip_img("Level-of-Token Diffusion", MARKER, 76, INK, -1.5, 3)
            s2 = strip_img("tiny pieces where it matters, big ones where it doesn't", HAND, 44, (70, 60, 50), 1.0, 4, pad=(40, 18))
            d1 = 0 if dt > 0.13 else 60
            paste_rgba(canvas, s1, W // 2 - s1.width // 2, 760 + d1, shadow=(6, 8))
            if dt > 0.45:
                d2 = 0 if dt > 0.58 else 40
                paste_rgba(canvas, s2, W // 2 - s2.width // 2, 905 + d2, shadow=(5, 6))
            for t0, s, x, rot in ((T_BST1, "2.2X FASTER", 1650, 7), (T_BST2, "2.5X FASTER", 1650, -5)):
                pass
    return np.clip(canvas, 0, 255).astype(np.uint8)


# ------------------------------------------------------------------ audio (foley)
SR = 48000


def env_exp(n, tau):
    return np.exp(-np.arange(n) / SR / tau)


def bandnoise(dur, lo, hi, seed):
    n = int(dur * SR)
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n)
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, 1 / SR)
    X[(f < lo) | (f > hi)] = 0
    y = np.fft.irfft(X, n)
    return y / (np.abs(y).max() + 1e-9)


def snip(seed):
    s = bandnoise(0.05, 2500, 9000, seed) * env_exp(int(0.05 * SR), 0.008) * 0.5
    n = int(0.06 * SR)
    ring = np.sin(2 * np.pi * 3400 * np.arange(n) / SR) * env_exp(n, 0.012) * 0.15
    out = np.zeros(int(0.07 * SR))
    out[:len(s)] += s
    out[int(0.008 * SR):int(0.008 * SR) + n] += ring[:len(out) - int(0.008 * SR)]
    return out


def rustle(dur, seed, vol=0.25):
    n = int(dur * SR)
    x = bandnoise(dur, 800, 6000, seed)
    rng = np.random.default_rng(seed + 1)
    am = cv2.resize(rng.uniform(0.2, 1, (1, max(2, int(dur * 40)))).astype(np.float32), (n, 1))[0]
    win = np.sin(np.linspace(0, np.pi, n))
    return x * am * win * vol


def squeak(seed):
    n = int(0.11 * SR)
    tt = np.arange(n) / SR
    rng = np.random.default_rng(seed)
    f0 = rng.uniform(900, 1300)
    f = f0 + 400 * tt / tt[-1] + 40 * np.sin(2 * np.pi * 38 * tt)
    ph = np.cumsum(f) / SR
    return np.sin(2 * np.pi * ph) * np.sin(np.linspace(0, np.pi, n)) * 0.12


def thunk(seed):
    n = int(0.25 * SR)
    tt = np.arange(n) / SR
    body = np.sin(2 * np.pi * (70 + 60 * np.exp(-tt * 40)) * tt) * env_exp(n, 0.06) * 0.7
    click = np.zeros(n)
    c = bandnoise(0.02, 500, 4000, seed) * env_exp(int(0.02 * SR), 0.004) * 0.4
    click[:len(c)] = c
    return body + click


def pluck(freq, dur=1.2, vol=0.2, bright=1.0):
    n = int(dur * SR)
    tt = np.arange(n) / SR
    out = np.zeros(n)
    for h in range(1, 8):
        out += (bright ** (h - 1)) / h * np.sin(2 * np.pi * freq * h * tt) * np.exp(-tt * (3 + 2.5 * h))
    out[:60] *= np.linspace(0, 1, 60)
    return out * vol


def xylo(freq, vol=0.18):
    n = int(0.6 * SR)
    tt = np.arange(n) / SR
    return (np.sin(2 * np.pi * freq * tt) + 0.3 * np.sin(2 * np.pi * freq * 3.9 * tt) * np.exp(-tt * 30)) \
        * np.exp(-tt * 7) * vol


def note(name):
    names = {"C": -9, "C#": -8, "D": -7, "D#": -6, "E": -5, "F": -4, "F#": -3, "G": -2, "G#": -1, "A": 0, "A#": 1, "B": 2}
    return 440 * 2 ** ((names[name[:-1]] + 12 * (int(name[-1]) - 4)) / 12)


def build_audio(path):
    out = np.zeros((int((DUR + 1) * SR), 2))

    def add(sig, t0, pan=0.0):
        i = int(t0 * SR)
        j = min(len(out), i + len(sig))
        if j <= i:
            return
        out[i:j, 0] += sig[:j - i] * (1 - pan) * 0.5 * 2 ** 0.5
        out[i:j, 1] += sig[:j - i] * (1 + pan) * 0.5 * 2 ** 0.5

    rng = np.random.default_rng(9)
    add(rustle(0.6, 1, 0.35), 0.0)
    for win, nsn in ((CUT, 26), (BCUT, 14)):
        for k in range(nsn):
            add(snip(k + int(win[0] * 10)), win[0] + (k + 0.5) / nsn * (win[1] - win[0]), pan=0.3)
    for win in (G1, G2, BG1, BG2):
        n = 7
        for k in range(n):
            add(squeak(k + int(win[0] * 10)), win[0] + (k + 0.3) / n * (win[1] - win[0]), pan=float(rng.uniform(-0.4, 0.4)))
        add(rustle(win[1] - win[0], int(win[0] * 7), 0.06), win[0])
    for t0 in (T_ST1, T_ST2, T_BST1, T_BST2):
        add(thunk(int(t0 * 10)), t0)
        for k, nm in enumerate(["C5", "E5", "G5", "C6"]):
            add(xylo(note(nm), 0.14), t0 + 0.12 + k * 0.07)
    for t0 in (CUT[1], G1[1], G2[1]):                            # marker on the note
        add(rustle(0.35, int(t0 * 13), 0.12), t0)
    # flips: paper flicks + rising xylophone
    scale = ["C5", "D5", "E5", "G5", "A5", "C6", "D6", "E6", "G6", "A6", "C7"]
    for k in range(40):
        tk = T_FLIP + 0.05 + k * 0.03 + rng.uniform(0, 0.02)
        add(bandnoise(0.02, 1500, 7000, 100 + k) * env_exp(int(0.02 * SR), 0.005) * 0.25, tk, pan=float(rng.uniform(-0.6, 0.6)))
    for k, nm in enumerate(scale):
        add(xylo(note(nm), 0.12), T_FLIP + 0.1 + k * 0.12)
    for nm in ("C5", "E5", "G5", "C6"):
        add(pluck(note(nm), 1.6, 0.12), T_FLIP + 1.5)
    add(rustle(0.5, 77, 0.35), T_OUT)
    add(rustle(0.4, 78, 0.3), T_B0)
    add(rustle(0.3, 79, 0.3), T_END)
    add(rustle(0.3, 80, 0.25), T_END + 0.45)
    # flip-book page flicks follow the playback speed
    t = T_B0 + 0.3
    while t < DUR:
        add(bandnoise(0.015, 2000, 8000, int(t * 1000)) * env_exp(int(0.015 * SR), 0.004) * 0.08, t, pan=-0.5)
        t += 1 / (15 * vid_speed(t)) * 3
    # ukulele strums (GCEA voicings), 104 bpm, eighths after the first video speedup
    chords = {"C": ["G4", "C4", "E4", "C5"], "G": ["G4", "D4", "G4", "B4"], "Am": ["A4", "C4", "E4", "A4"],
              "F": ["A4", "C4", "F4", "A4"]}
    prog = ["C", "G", "Am", "F"]
    beat = 60 / 104
    t, k = 0.6, 0
    while t < DUR - 0.5:
        ch = chords[prog[(k // 4) % 4]]
        up_ = k % 2 == 1
        for s, nm in enumerate(ch[::-1] if up_ else ch):
            add(pluck(note(nm), 1.0, 0.045 if up_ else 0.06, 0.7), t + s * 0.012)
        step = beat / 2 if (T_BST1 <= t) else beat
        t += step
        k += 1
    for k, nm in enumerate(["C5", "E5", "G5", "C6", "E6"]):
        add(xylo(note(nm), 0.16), T_END + 0.1 + k * 0.1)
    add(pluck(note("C4"), 1.5, 0.15), T_END + 0.6)
    out = out[:int(DUR * SR)]
    out = out / max(1e-6, np.abs(out).max()) * 0.89
    wavfile.write(path, SR, (out * 32767).astype(np.int16))


# ------------------------------------------------------------------ main
def render_pose(k):
    return render(k / POSE_FPS).tobytes()


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "h"
    if mode == "still":
        for s in sys.argv[2:]:
            t = round(float(s) * POSE_FPS) / POSE_FPS
            Image.fromarray(render(t)).save(os.path.join(HERE, f"scratch/still_{TAG}_{float(s):05.2f}.png"))
        return
    if mode == "audio":
        build_audio(os.path.join(HERE, f"media/final/LoT_cut_paper_{TAG}.wav"))
        return
    wav = os.path.join(HERE, f"media/final/LoT_cut_paper_{TAG}.wav")
    build_audio(wav)
    out = os.path.join(HERE, f"media/final/LoT_cut_paper_{TAG}_1080p60.mp4")
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
           "-i", "-", "-i", wav, "-c:v", "libx264", "-preset", "slow", "-crf", "16", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-ac", "2", "-ar", "48000", "-b:a", "192k", "-t", f"{DUR}", "-movflags", "+faststart", out]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    hold = FPS // POSE_FPS
    with Pool(10) as pool:
        for k, buf in enumerate(pool.imap(render_pose, range(NPOSE), chunksize=2)):
            for _ in range(hold):
                p.stdin.write(buf)
            if k % 30 == 0:
                print(f"pose {k}/{NPOSE}", flush=True)
    p.stdin.close()
    p.wait()
    print("wrote", out)


if __name__ == "__main__":
    main()
