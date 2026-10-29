"""LoT Diffusion, 20-s silver-age comic-book clip, v1 (numpy + PIL + cv2 frames → ffmpeg, synthesized cartoon SFX). No manim.

Every panel is printed in CMYK halftone; inside the token layout the dot screen is as coarse as the token is big
(1×1 → 4-px dots, 8×8 → 16-px dots), and each token is a comic panel with an ink border (thicker for bigger tokens).
  0.0–1.45  cover: LEVEL-OF-TOKEN DIFFUSION, issue box, sunburst
  1.45–1.95 page flip → PAGE 1, the depth map panel ("I'M IN FOCUS!")
  3.4–4.8   ink 4,096 tiny panels (1 × 1 tokens)
  5.1–6.5   blurry panels merge to 2 × 2 (1,606)        → ZOOM! 1.9X FASTER
  7.9–9.0   merge to 4 × 4 / 8 × 8 = LoT layout (1,027) → KRAK! 2.4X FASTER
 10.1–11.4  panels ink-in to the generated image, halftone → full colour ("STILL SHARP!")
 11.9–12.4  page flip → PAGE 2: platformer video panel + its token panel; ground merge → WHOOSH! 2.2X,
            sky merge → POW! 2.5X, the video panel's playback speed follows
 18.1–20.0  finale banner, TO BE CONTINUED...
Usage: python render_comic_v1.py [l|h]       → media/final/LoT_comic_v1_<res>.mp4
       python render_comic_v1.py still t ...  → scratch/still_v1_<t>.png
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
from scipy.signal import butter, lfilter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TAG = "v1"
BANG = os.path.join(HERE, "data/Bangers-Regular.ttf")
NEUE = os.path.join(HERE, "data/ComicNeue-Bold.ttf")
CUE = os.path.join(ROOT, "data/img/img_000000000025_dof_cue.png")
GEN = os.path.join(ROOT, "data/img/img_000000000025_dof_generated_4b.png")
LAYOUT = os.path.join(ROOT, "data/layouts/img_000000000025_dof.json")
VID = os.path.join(ROOT, "data/clips/vd_platformer_gen")
LEVELS = os.path.join(ROOT, "arcade_clip/data/platformer_levels.npy")

W, H = 1920, 1080
DUR = 20.0

PAPER = (246, 236, 212)
INK = (22, 18, 20)
CAP_YEL = (255, 222, 70)
RED = (226, 36, 40)
BLUE = (30, 110, 200)
CYAN_I = np.array((0, 166, 230), np.float32)
MAG_I = np.array((230, 20, 130), np.float32)
YEL_I = np.array((255, 236, 0), np.float32)

# ------------------------------------------------------------------ timeline (s)
T_FLIP1 = (1.45, 1.95)
T_BUB1 = (2.2, 3.35)
S1 = (3.4, 4.8)
S2 = (5.1, 6.5)
S3 = (7.9, 9.0)
T_SP1, T_SP2 = 6.55, 9.05
SP_HOLD = 1.2
T_REV = (10.1, 10.8)          # panels ink-in to the image
T_COL = (10.8, 11.4)          # halftone → full colour
T_BUB2 = (11.0, 11.9)
T_FLIP2 = (11.9, 12.4)
T_B0 = 12.4
B1 = (12.9, 13.7)
B2 = (14.4, 15.2)
B3 = (16.3, 17.0)
T_BSP1, T_BSP2 = 15.25, 17.05
T_END = 18.1

IMG_TOK = (4096, 1606, 1027)
VID_TOK = (75600, 38046, 34698)
VID_SPEED = (1.0, 2.2, 2.5)

# ------------------------------------------------------------------ geometry
P = 768
PX0, PY0 = 150, 170
N = 64
CELL = P // N                       # 12 px per token
VW, VH = 880, 495
VXL, VXR, VY0 = 60, 980, 262
VN, VM = 45, 80
VC = 11
DOT = {1: 4.0, 2: 6.0, 4: 10.0, 8: 16.0}     # halftone cell per token size
TH = {1: 1, 2: 2, 4: 3, 8: 4}                # ink border per token side


def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def ease_out(x):
    x = clamp(x)
    return 1 - (1 - x) ** 3


def ease_io(x):
    x = clamp(x)
    return x * x * (3 - 2 * x)


def back_out(x, s=2.2):
    x = clamp(x) - 1
    return 1 + (s + 1) * x ** 3 + s * x ** 2


def win_alpha(t, t0, t1, fade=0.15):
    if t < t0 or t >= t1:
        return 0.0
    return min(clamp((t - t0) / fade), clamp((t1 - t) / fade))


# ------------------------------------------------------------------ assets
A = None


def assets():
    global A
    if A is None:
        A = load_assets()
    return A


def load_assets():
    a = {}
    cue = np.asarray(Image.open(CUE).convert("RGB")).astype(np.float32)
    a["cue"] = cv2.resize(cue, (P, P), interpolation=cv2.INTER_AREA)
    a["cue_cell"] = cue.reshape(N, 16, N, 16, 3).mean((1, 3))
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
    ii, jj = np.mgrid[0:N, 0:N]
    a["img_blocks"] = {}
    for cap in (1, 2, 8):
        b = np.minimum(size, cap)
        a["img_blocks"][cap] = (r0m + (ii - r0m) // b * b, c0m + (jj - c0m) // b * b, b)
    a["lv"] = np.load(LEVELS)
    a["vid"] = [cv2.resize(np.asarray(Image.open(os.path.join(VID, f"{f + 1:04d}.jpg")).convert("RGB")),
                           (VW, VH), interpolation=cv2.INTER_AREA).astype(np.float32) for f in range(81)]
    a["vid_cell"] = [cv2.resize(v, (VM, VN), interpolation=cv2.INTER_AREA) for v in a["vid"]]
    # paper: grain + pale Ben-Day dots
    rng = np.random.default_rng(1)
    grain = cv2.GaussianBlur(rng.normal(0, 1, (H, W)).astype(np.float32), (0, 0), 1.2)
    fib = cv2.GaussianBlur(rng.normal(0, 1, (H // 8, W // 8)).astype(np.float32), (0, 0), 3)
    fib = cv2.resize(fib, (W, H))
    paper = np.array(PAPER, np.float32)[None, None] * (1 + 0.025 * grain[..., None] + 0.03 * fib[..., None])
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    s = 14.0
    u = (xx * math.cos(0.26) + yy * math.sin(0.26)) / s
    v = (-xx * math.sin(0.26) + yy * math.cos(0.26)) / s
    d = np.sqrt((u - np.round(u)) ** 2 + (v - np.round(v)) ** 2) * s
    dots = np.clip(2.6 + 0.5 - d, 0, 1)[..., None]
    a["paper"] = paper * (1 - dots * 0.10 * (1 - MAG_I / 255)[None, None])
    a["paper_plain"] = paper
    a["yy"], a["xx"] = yy, xx
    return a


# ------------------------------------------------------------------ halftone
ANG = {"c": math.radians(15), "m": math.radians(75), "y": 0.0, "k": math.radians(45)}


def halftone(rgb, cellmap, x0=0, y0=0, misreg=True, paper=None):
    """CMYK halftone of rgb (h, w, 3) float 0..255; cellmap scalar or (h, w) dot pitch in px."""
    a = assets()
    h, w = rgb.shape[:2]
    yy = a["yy"][y0:y0 + h, x0:x0 + w]
    xx = a["xx"][y0:y0 + h, x0:x0 + w]
    f = np.clip(rgb / 255.0, 0, 1)
    k = 1 - f.max(2)
    den = np.maximum(1 - k, 1e-4)
    c = (1 - f[..., 0] - k) / den
    m = (1 - f[..., 1] - k) / den
    y = (1 - f[..., 2] - k) / den
    s = cellmap
    out = np.array(PAPER, np.float32)[None, None] * np.ones((h, w, 1), np.float32) if paper is None else paper.copy()
    for name, val, inkcol, shift in (("y", y, YEL_I, (0, 0)), ("c", c, CYAN_I, (1, 2)),
                                     ("m", m, MAG_I, (-2, 1)), ("k", k * 0.92, None, (0, 0))):
        th = ANG[name]
        u = (xx * math.cos(th) + yy * math.sin(th)) / s
        v = (-xx * math.sin(th) + yy * math.cos(th)) / s
        dd = np.sqrt((u - np.round(u)) ** 2 + (v - np.round(v)) ** 2) * s
        r = s * np.sqrt(np.clip(val, 0, 1) / math.pi) * 1.08
        cov = np.clip(r - dd + 0.5, 0, 1)
        if misreg and shift != (0, 0):
            cov = np.roll(cov, shift, (0, 1))
        if inkcol is None:
            out *= (1 - cov * 0.9)[..., None]
        else:
            out *= (1 - cov[..., None] * (1 - inkcol / 255)[None, None])
    return out


# ------------------------------------------------------------------ tokens
def mosaic(cell, ids, n_tok):
    flat = ids.ravel()
    cnt = np.bincount(flat, minlength=n_tok)
    out = np.empty(cell.shape, np.float32)
    for ch in range(3):
        sm = np.bincount(flat, weights=cell[..., ch].ravel(), minlength=n_tok)
        out[..., ch] = (sm / np.maximum(cnt, 1))[flat].reshape(ids.shape)
    return out


def up(arr, c):
    return np.repeat(np.repeat(arr, c, 0), c, 1)


def sweep_frac(t, win):
    return clamp((t - win[0]) / (win[1] - win[0]))


def staged(old, new, n, t0, dur, t):
    """per-token (oi, oj, b) where blocks the sweep has fully passed use the new cap; plus switch times."""
    t_sw = t0 + (new[0] + new[2]) / n * dur
    use = t >= t_sw
    return (np.where(use, new[0], old[0]), np.where(use, new[1], old[1]), np.where(use, new[2], old[2])), t_sw


def ink_borders(oi, oj, b, c):
    """per-pixel mask of each token panel's own inner border (thicker for bigger tokens)."""
    oiu, oju, bu = up(oi, c), up(oj, c), up(b, c)
    h, w = bu.shape
    yy, xx = np.mgrid[0:h, 0:w]
    ly = yy - oiu * c
    lx = xx - oju * c
    bw = bu * c
    th = np.choose(np.searchsorted([1, 2, 4, 8], bu), [1, 2, 3, 4])
    return (lx < th) | (lx >= bw - th) | (ly < th) | (ly >= bw - th), bu


def dotmap(bu):
    lut = np.zeros(9, np.float32)
    for k, v in DOT.items():
        lut[k] = v
    return lut[bu]


# ------------------------------------------------------------------ text / shapes
@lru_cache(maxsize=256)
def fnt(path, size):
    return ImageFont.truetype(path, size)


def text_layer(s, path, size, fill, stroke=0, stroke_fill=INK, shadow=None, shadow_off=(0, 0)):
    f = fnt(path, size)
    l, tp, r, b = f.getbbox(s, stroke_width=stroke)
    pad = stroke + max(abs(shadow_off[0]), abs(shadow_off[1])) + 4
    im = Image.new("RGBA", (r - l + 2 * pad, b - tp + 2 * pad), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    if shadow is not None:
        d.text((pad - l + shadow_off[0], pad - tp + shadow_off[1]), s, font=f, fill=shadow,
               stroke_width=stroke, stroke_fill=shadow)
    d.text((pad - l, pad - tp), s, font=f, fill=fill, stroke_width=stroke, stroke_fill=stroke_fill)
    return im


@lru_cache(maxsize=512)
def text_cached(s, path, size, fill, stroke=0, shadow=None, shadow_off=(0, 0)):
    return text_layer(s, path, size, fill, stroke, INK, shadow, shadow_off)


def paste(canvas, im, cx, cy, scale=1.0, rot=0.0, alpha=1.0, anchor="c"):
    if scale <= 0.01 or alpha <= 0:
        return
    if abs(scale - 1) > 1e-3:
        im = im.resize((max(1, int(im.width * scale)), max(1, int(im.height * scale))), Image.BICUBIC)
    if rot:
        im = im.rotate(rot, Image.BICUBIC, expand=True)
    if alpha < 1:
        im = im.copy()
        im.putalpha(im.getchannel("A").point(lambda v: int(v * alpha)))
    if anchor == "c":
        x, y = int(cx - im.width / 2), int(cy - im.height / 2)
    elif anchor == "l":
        x, y = int(cx), int(cy - im.height / 2)
    else:
        x, y = int(cx), int(cy)
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(canvas.width, x + im.width), min(canvas.height, y + im.height)
    if x1 > x0 and y1 > y0:
        canvas.alpha_composite(im.crop((x0 - x, y0 - y, x1 - x, y1 - y)), (x0, y0))


def wrap(s, f, maxw):
    words, lines, cur = s.split(), [], ""
    for wd in words:
        tst = (cur + " " + wd).strip()
        if f.getlength(tst) <= maxw:
            cur = tst
        else:
            lines.append(cur)
            cur = wd
    lines.append(cur)
    return lines


@lru_cache(maxsize=128)
def caption_box(s, w, size=40):
    f = fnt(NEUE, size)
    lines = wrap(s, f, w - 48)
    lh = int(size * 1.18)
    h = lh * len(lines) + 40
    im = Image.new("RGBA", (w + 10, h + 10), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rectangle([8, 8, w + 9, h + 9], fill=INK + (255,))                 # drop shadow
    d.rectangle([0, 0, w, h], fill=CAP_YEL + (255,), outline=INK + (255,), width=5)
    for k, ln in enumerate(lines):
        d.text((24, 20 + k * lh), ln, font=f, fill=INK + (255,))
    return im


@lru_cache(maxsize=64)
def bubble(s, tail_dx, tail_dy, size=44):
    """white speech balloon, text centred, tail tip at (tail_dx, tail_dy) relative to balloon centre."""
    f = fnt(NEUE, size)
    tw = f.getlength(s)
    bw, bh = int(tw + 110), int(size * 2.4)
    pad = max(abs(tail_dx), abs(tail_dy)) + 20
    Wb, Hb = bw + 2 * pad, bh + 2 * pad
    im = Image.new("RGBA", (Wb, Hb), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    cx, cy = Wb // 2, Hb // 2
    ang = math.atan2(tail_dy, tail_dx)
    nx, ny = -math.sin(ang), math.cos(ang)
    base = (cx + math.cos(ang) * bw * 0.3, cy + math.sin(ang) * bh * 0.3)
    tail = [(base[0] + nx * 26, base[1] + ny * 26), (cx + tail_dx, cy + tail_dy), (base[0] - nx * 26, base[1] - ny * 26)]
    d.polygon(tail, fill=INK + (255,))
    d.ellipse([cx - bw // 2 - 5, cy - bh // 2 - 5, cx + bw // 2 + 5, cy + bh // 2 + 5], fill=INK + (255,))
    inner = [(base[0] + nx * 19, base[1] + ny * 19), (cx + tail_dx * 0.93, cy + tail_dy * 0.93),
             (base[0] - nx * 19, base[1] - ny * 19)]
    d.polygon(inner, fill=(255, 255, 255, 255))
    d.ellipse([cx - bw // 2, cy - bh // 2, cx + bw // 2, cy + bh // 2], fill=(255, 255, 255, 255))
    d.text((cx - tw / 2, cy - size * 0.62), s, font=f, fill=INK + (255,))
    return im, cx, cy


@lru_cache(maxsize=16)
def starburst(word, sub, seed, r=290):
    rng = np.random.default_rng(seed)
    S = int(r * 2.4)
    im = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    c = S / 2
    n = 18
    pts_o, pts_i = [], []
    for k in range(2 * n):
        ang = k * math.pi / n + rng.uniform(-0.06, 0.06)
        rad = r * (1.0 if k % 2 == 0 else 0.68) * rng.uniform(0.9, 1.08)
        pts_o.append((c + math.cos(ang) * rad * 1.08, c + math.sin(ang) * rad * 0.82))
        pts_i.append((c + math.cos(ang) * rad * 0.82, c + math.sin(ang) * rad * 0.64))
    d.polygon(pts_o, fill=INK + (255,))
    shrink = [(c + (x - c) * 0.965, c + (y - c) * 0.955) for x, y in pts_o]
    d.polygon(shrink, fill=CAP_YEL + (255,))
    d.polygon(pts_i, fill=RED + (255,))
    t1 = text_layer(word, BANG, 150, CAP_YEL + (255,), stroke=7, shadow=INK + (255,), shadow_off=(8, 8))
    t2 = text_layer(sub, BANG, 78, (255, 255, 255, 255), stroke=5, shadow=INK + (255,), shadow_off=(5, 5))
    im.alpha_composite(t1, (int(c - t1.width / 2), int(c - t1.height / 2 - 48)))
    im.alpha_composite(t2, (int(c - t2.width / 2), int(c - t2.height / 2 + 72)))
    return im


def speed_lines(d, cx, cy, t, alpha, seed=0, n=56, r0=330):
    rng = np.random.default_rng(seed)
    for k in range(n):
        ang = k / n * 2 * math.pi + rng.uniform(-0.03, 0.03)
        wd = rng.uniform(0.006, 0.018)
        rr = r0 + rng.uniform(0, 120)
        p = [(cx + math.cos(ang - wd) * rr, cy + math.sin(ang - wd) * rr),
             (cx + math.cos(ang) * 2400, cy + math.sin(ang) * 2400),
             (cx + math.cos(ang + wd) * rr, cy + math.sin(ang + wd) * rr)]
        d.polygon(p, fill=INK + (int(150 * alpha),))


def panel_frame(d, x0, y0, x1, y1, w=6, fill=None):
    d.rectangle([x0 - w, y0 - w, x1 + w - 1, y1 + w - 1], fill=INK)
    if fill is not None:
        d.rectangle([x0, y0, x1 - 1, y1 - 1], fill=fill)


# ------------------------------------------------------------------ page 1 content
def img_panel(t):
    """(rgb float P×P, token count or None, sweep y px or None)."""
    a = assets()
    B = a["img_blocks"]
    if t < S1[0]:
        return halftone(a["cue"], 4.0, PX0, PY0), None, None
    if t < S2[0]:
        f = sweep_frac(t, S1)
        rows = int(f * N + 1e-6)
        oi, oj, b = B[1]
        ids = oi * N + oj
        mos = up(mosaic(a["cue_cell"], ids, N * N), CELL)
        img = a["cue"].copy()
        ry = rows * CELL
        img[:ry] = mos[:ry]
        ht = halftone(img, 4.0, PX0, PY0)
        e, _ = ink_borders(oi, oj, b, CELL)
        e[ry:] = False
        ht[e] = INK
        return ht, rows * N, (f * P if f < 1 else None)
    if t < T_REV[0]:
        if t < S3[0]:
            old, new, win = B[1], B[2], S2
        else:
            old, new, win = B[2], B[8], S3
        f = sweep_frac(t, win)
        (oi, oj, b), t_sw = staged(old, new, N, win[0], win[1] - win[0], t)
        ids = oi * N + oj
        mos = up(mosaic(a["cue_cell"], ids, N * N), CELL)
        e, bu = ink_borders(oi, oj, b, CELL)
        ht = halftone(mos, dotmap(bu), PX0, PY0)
        dt = t - t_sw
        fl = np.where((new[2] > old[2]) & (dt >= 0), np.exp(-np.maximum(dt, 0) / 0.15), 0).astype(np.float32)
        flu = up(fl, CELL)[..., None]
        ht = ht * (1 - flu * 0.7) + np.array(CAP_YEL, np.float32) * flu * 0.7
        ht[e] = INK
        return ht, len(np.unique(ids)), (f * P if 0 < f < 1 else None)
    # reveal: random panel order, the generated image printed with the same per-token dot pitch, then full colour
    oi, oj, b = B[8]
    ids = oi * N + oj
    rng = np.random.default_rng(3)
    t_rev = T_REV[0] + rng.uniform(0, T_REV[1] - T_REV[0] - 0.1, N * N)[ids]
    shown = up(t >= t_rev, CELL)
    e, bu = ink_borders(oi, oj, b, CELL)
    dm = dotmap(bu)
    mos = up(mosaic(a["cue_cell"], ids, N * N), CELL)
    src = np.where(shown[..., None], a["gen"], mos)
    ht = halftone(src, dm, PX0, PY0)
    col = ease_io((t - T_COL[0]) / (T_COL[1] - T_COL[0]))
    img = ht * (1 - col) + a["gen"] * col
    ea = 1 - ease_io((t - T_COL[0]) / (T_COL[1] - T_COL[0] + 0.2))
    if ea > 0:
        img[e] = img[e] * (1 - ea) + np.array(INK, np.float32) * ea
    return img, 1027, None


def counter_panel(canvas, d, x0, y0, x1, y1, label, value, color, pulse=0.0):
    panel_frame(d, x0, y0, x1, y1, fill=(255, 252, 240))
    lab = text_cached(label, BANG, 64, INK + (255,))
    paste(canvas, lab, (x0 + x1) / 2, y0 + 64)
    num = text_cached(value, BANG, 150 if len(value) <= 5 else 120, color + (255,), stroke=6, shadow=INK + (255,),
                      shadow_off=(7, 7))
    paste(canvas, num, (x0 + x1) / 2, (y0 + y1) / 2 + 40, scale=1 + 0.25 * pulse)


def caption(canvas, s, x, y, w, t, t0, t1, size=40):
    if not (t0 <= t < t1):
        return
    im = caption_box(s, w, size)
    k = ease_out((t - t0) / 0.18)
    paste(canvas, im, x, y - 30 * (1 - k), alpha=k, anchor="tl")


def burst_on(canvas, d, t, t0, word, sub, cx, cy, seed, rot):
    dt = t - t0
    if dt < 0 or dt > SP_HOLD + 0.2:
        return
    out = clamp((dt - SP_HOLD) / 0.2)
    s = back_out(dt / 0.22) * (1 - out)
    speed_lines(d, cx, cy, t, (1 - out) * clamp(dt / 0.1), seed)
    paste(canvas, starburst(word, sub, seed), cx, cy, scale=s * (1 + 0.025 * math.sin(dt * 30)), rot=rot)


def header(canvas, page):
    paste(canvas, text_cached("LEVEL-OF-TOKEN DIFFUSION", BANG, 64, INK + (255,), shadow=CAP_YEL + (255,),
                              shadow_off=(4, 4)), 150, 98, anchor="l")
    paste(canvas, text_cached(f"PAGE {page}", NEUE, 30, INK + (255,)), W - 120, 98, anchor="c")


def page1(t):
    a = assets()
    canvas = Image.fromarray(np.clip(a["paper"], 0, 255).astype(np.uint8)).convert("RGBA")
    d = ImageDraw.Draw(canvas)
    header(canvas, 1)
    img, tok, bar = img_panel(t)
    panel_frame(d, PX0, PY0, PX0 + P, PY0 + P, w=7)
    canvas.paste(Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)), (PX0, PY0))
    if bar is not None:                                  # inking pen nib riding the sweep
        by = PY0 + int(bar)
        d.rectangle([PX0, by - 2, PX0 + P - 1, by + 2], fill=INK)
        nx = PX0 + P + 14
        d.polygon([(nx, by), (nx + 40, by - 22), (nx + 120, by - 22), (nx + 120, by + 22), (nx + 40, by + 22)],
                  fill=INK)
        d.polygon([(nx + 8, by), (nx + 40, by - 14), (nx + 40, by + 14)], fill=(200, 200, 200))
    # right column
    RX0, RX1 = 1000, 1800
    sp = 1.0 if t < T_SP1 else (1.9 if t < T_SP2 else 2.4)
    pulse = max(math.exp(-(t - T_SP1) / 0.15) if t >= T_SP1 else 0, math.exp(-(t - T_SP2) / 0.15) if t >= T_SP2 else 0)
    counter_panel(canvas, d, RX0, 470, 1380, 938, "TOKENS", f"{tok:,}" if tok is not None else "?", BLUE)
    counter_panel(canvas, d, 1420, 470, RX1, 938, "SPEED", f"{sp:.1f}X", RED, pulse)
    caps = [
        ("A DEPTH MAP KNOWS WHAT'S IN FOCUS.", T_FLIP1[1], S1[0]),
        ("ONE TOKEN PER PATCH: 4,096 TINY PANELS!", S1[0], S2[0]),
        ("BLURRY? MERGE INTO BIGGER TOKENS!", S2[0], T_SP1),
        ("NOTHING IN FOCUS? EVEN BIGGER!", S3[0] - 0.3, T_SP2),
        ("4X FEWER TOKENS. THE HERO STAYS SHARP!", T_REV[0], T_FLIP2[0] + 0.6),
    ]
    for s, t0, t1 in caps:
        caption(canvas, s, RX0, 170, RX1 - RX0, t, t0, t1)
    if T_BUB1[0] <= t < T_BUB1[1]:
        im, bcx, bcy = bubble("I'M IN FOCUS!", 150, 70)
        k = back_out((t - T_BUB1[0]) / 0.25) * (1 - clamp((t - T_BUB1[1] + 0.15) / 0.15))
        paste(canvas, im, PX0 + 0.42 * P, PY0 + 0.12 * P, scale=k)
    if T_BUB2[0] <= t < T_BUB2[1]:
        im, bcx, bcy = bubble("STILL SHARP!", 150, 60)
        k = back_out((t - T_BUB2[0]) / 0.25) * (1 - clamp((t - T_BUB2[1] + 0.15) / 0.15))
        paste(canvas, im, PX0 + 0.58 * P, PY0 + 0.07 * P, scale=k)
    burst_on(canvas, d, t, T_SP1, "ZOOM!", "1.9X FASTER!", 1400, 330, 11, 6)
    burst_on(canvas, d, t, T_SP2, "KRAK!", "2.4X FASTER!", 1400, 330, 12, -7)
    return canvas


# ------------------------------------------------------------------ page 2 content
def vid_V(tt):
    v = 0.0
    for s0, s1, sp in ((T_B0, T_BSP1, VID_SPEED[0]), (T_BSP1, T_BSP2, VID_SPEED[1]), (T_BSP2, 99, VID_SPEED[2])):
        if tt > s0:
            v += (min(tt, s1) - s0) * sp * 15
    return v


def vid_index(t):
    v = vid_V(t)
    return int(v) % 81, int(v) // 81 != int(vid_V(t - 0.07)) // 81


def vid_speed(t):
    return VID_SPEED[0] if t < T_BSP1 else (VID_SPEED[1] if t < T_BSP2 else VID_SPEED[2])


def vid_blocks(lv, cap):
    b = np.minimum(lv, cap).astype(int)
    ii, jj = np.mgrid[0:VN, 0:VM]
    return ii // b * b, jj // b * b, b


def vid_panels(t):
    a = assets()
    idx, wrapped = vid_index(t)
    frame, cell, lv = a["vid"][idx], a["vid_cell"][idx], a["lv"][idx]
    left = halftone(cv2.GaussianBlur(frame, (0, 0), 0.8), 4.0, VXL, VY0) * 0.55 + frame * 0.45
    tok = None
    if t < B1[0]:
        right = halftone(frame, 4.0, VXR, VY0)
        return left, right, None, 0, wrapped
    if t < B2[0]:
        f = sweep_frac(t, B1)
        rows = int(f * VN + 1e-6)
        oi, oj, b = vid_blocks(lv, 1)
        mos = up(mosaic(cell, oi * VM + oj, VN * VM), VC)
        img = frame.copy()
        img[:rows * VC] = mos[:rows * VC]
        right = halftone(img, 4.0, VXR, VY0)
        e, _ = ink_borders(oi, oj, b, VC)
        e[rows * VC:] = False
        right[e] = INK
        return left, right, (f * VH if f < 1 else None), int(VID_TOK[0] * f), wrapped
    if t < B3[0]:
        old, new, win, k = vid_blocks(lv, 1), vid_blocks(lv, 2), B2, 1
    else:
        old, new, win, k = vid_blocks(lv, 2), vid_blocks(lv, 4), B3, 2
    f = sweep_frac(t, win)
    (oi, oj, b), t_sw = staged(old, new, VN, win[0], win[1] - win[0], t)
    mos = up(mosaic(cell, oi * VM + oj, VN * VM), VC)
    e, bu = ink_borders(oi, oj, b, VC)
    dm = np.where(bu == 1, 4.0, np.where(bu == 2, 6.0, 10.0)).astype(np.float32)
    right = halftone(mos, dm, VXR, VY0)
    dt = t - t_sw
    fl = np.where((new[2] > old[2]) & (dt >= 0), np.exp(-np.maximum(dt, 0) / 0.15), 0).astype(np.float32)
    flu = up(fl, VC)[..., None]
    right = right * (1 - flu * 0.7) + np.array(CAP_YEL, np.float32) * flu * 0.7
    right[e] = INK
    tok = int(round(VID_TOK[k - 1] + (VID_TOK[k] - VID_TOK[k - 1]) * f))
    return left, right, (f * VH if 0 < f < 1 else None), tok, wrapped


def page2(t):
    a = assets()
    canvas = Image.fromarray(np.clip(a["paper"], 0, 255).astype(np.uint8)).convert("RGBA")
    d = ImageDraw.Draw(canvas)
    header(canvas, 2)
    left, right, bar, tok, wrapped = vid_panels(t)
    for x0, im in ((VXL, left), (VXR, right)):
        panel_frame(d, x0, VY0, x0 + VW, VY0 + VH, w=7)
        canvas.paste(Image.fromarray(np.clip(im, 0, 255).astype(np.uint8)), (x0, VY0))
    sp = vid_speed(t)
    # action lines behind the runner once the video speeds up
    if sp > 1.2:
        rng = np.random.default_rng(int(t * 30))
        for k in range(int(10 * sp)):
            y = VY0 + rng.uniform(20, VH - 20)
            x = VXL + rng.uniform(0, VW * 0.5)
            ln = rng.uniform(80, 220)
            d.rectangle([x, y, x + ln, y + rng.choice([2, 3])], fill=(255, 255, 255, 200))
    if wrapped:                                        # loop seam hidden by a page-shake smear
        d.rectangle([VXL, VY0, VXL + VW, VY0 + VH], fill=(255, 252, 240, 170))
    if bar is not None:
        by = VY0 + int(bar)
        d.rectangle([VXR, by - 2, VXR + VW - 1, by + 2], fill=INK)
        nx = VXR + VW + 12
        d.polygon([(nx, by), (nx + 30, by - 16), (nx + 60, by - 16), (nx + 60, by + 16), (nx + 30, by + 16)], fill=INK)
    # play badge
    badge = caption_box(f"PLAY {sp:.1f}X", 250, 36)
    paste(canvas, badge, VXL + VW - 270, VY0 + 18, anchor="tl")
    n_tri = 1 if sp < 1.5 else (2 if sp < 2.3 else 3)
    for k in range(n_tri):
        bx = VXL + VW - 270 - 30 - (n_tri - k) * 30
        d.polygon([(bx, VY0 + 30), (bx + 26, VY0 + 48), (bx, VY0 + 66)], fill=INK)
    lab_l = caption_box("THE VIDEO", 260, 34)
    lab_r = caption_box("ITS TOKENS", 290, 34)
    paste(canvas, lab_l, VXL + 18, VY0 + 18, anchor="tl")
    paste(canvas, lab_r, VXR + 18, VY0 + 18, anchor="tl")
    pulse = max(math.exp(-(t - T_BSP1) / 0.15) if t >= T_BSP1 else 0,
                math.exp(-(t - T_BSP2) / 0.15) if t >= T_BSP2 else 0)
    panel_frame(d, 60, 812, 620, 1030, fill=(255, 252, 240))
    paste(canvas, text_cached("TOKENS", BANG, 52, INK + (255,)), 340, 850)
    paste(canvas, text_cached(f"{tok:,}" if tok else "?", BANG, 110, BLUE + (255,), stroke=5, shadow=INK + (255,),
                              shadow_off=(6, 6)), 340, 950)
    panel_frame(d, 660, 812, 1220, 1030, fill=(255, 252, 240))
    paste(canvas, text_cached("SPEED", BANG, 52, INK + (255,)), 940, 850)
    paste(canvas, text_cached(f"{sp:.1f}X", BANG, 110, RED + (255,), stroke=5, shadow=INK + (255,),
                              shadow_off=(6, 6)), 940, 950, scale=1 + 0.25 * pulse)
    caps = [
        ("MEANWHILE... IN VIDEO!", T_FLIP2[1], B1[0] + 0.2),
        ("EVERY PATCH OF EVERY FRAME!", B1[0] + 0.2, B2[0]),
        ("MERGE THE GROUND!", B2[0], T_BSP1),
        ("MERGE THE SKY!", B3[0] - 0.3, T_BSP2),
        ("THE VIDEO PLAYS FASTER, TOO!", T_BSP2 + SP_HOLD + 0.2, T_END),
    ]
    for s, t0, t1 in caps:
        caption(canvas, s, 1260, 812, 600, t, t0, t1, size=40)
    burst_on(canvas, d, t, T_BSP1, "WHOOSH!", "2.2X FASTER!", 1480, 760, 21, 5)
    burst_on(canvas, d, t, T_BSP2, "POW!", "2.5X FASTER!", 1480, 760, 22, -6)
    if t >= T_END:
        dt = t - T_END
        k = back_out(dt / 0.3)
        band = Image.new("RGBA", (W, H), (0, 0, 0, int(90 * clamp(dt / 0.3))))
        canvas.alpha_composite(band)
        box = finale_box()
        paste(canvas, box, W / 2, 500, scale=k, rot=2 * (1 - clamp(dt / 0.3)))
        if dt > 0.9:
            paste(canvas, caption_box("TO BE CONTINUED...", 420, 40), W - 520, 930, anchor="tl",
                  alpha=clamp((dt - 0.9) / 0.2))
    return canvas


@lru_cache(maxsize=2)
def finale_box():
    w, h = 1500, 380
    im = Image.new("RGBA", (w + 16, h + 16), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rectangle([14, 14, w + 15, h + 15], fill=INK + (255,))
    d.rectangle([0, 0, w, h], fill=CAP_YEL + (255,), outline=INK + (255,), width=8)
    t1 = text_layer("LEVEL-OF-TOKEN DIFFUSION!", BANG, 140, RED + (255,), stroke=6, shadow=INK + (255,),
                    shadow_off=(8, 8))
    t2 = text_layer("IMAGES 2.4X FASTER  •  VIDEO 2.5X FASTER", NEUE, 52, INK + (255,))
    im.alpha_composite(t1, (int(w / 2 - t1.width / 2), 50))
    im.alpha_composite(t2, (int(w / 2 - t2.width / 2), 250))
    return im


# ------------------------------------------------------------------ cover
def cover(t):
    a = assets()
    base = a["paper_plain"].copy()
    canvas = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8)).convert("RGBA")
    d = ImageDraw.Draw(canvas)
    cx, cy = W / 2, 600
    rot = t * 0.25
    n = 28
    for k in range(n):
        a0 = rot + k * 2 * math.pi / n
        a1 = a0 + math.pi / n
        col = (255, 200, 40) if k % 2 == 0 else (255, 140, 30)
        d.polygon([(cx, cy), (cx + math.cos(a0) * 2500, cy + math.sin(a0) * 2500),
                   (cx + math.cos(a1) * 2500, cy + math.sin(a1) * 2500)], fill=col)
    # halftone the sunburst for print feel
    arr = np.asarray(canvas.convert("RGB")).astype(np.float32)
    arr = halftone(arr, 9.0, 0, 0, misreg=True, paper=a["paper_plain"])
    canvas = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).convert("RGBA")
    d = ImageDraw.Draw(canvas)
    d.rectangle([0, 0, W - 1, 150], fill=RED)
    d.rectangle([0, 150, W - 1, 158], fill=INK)
    paste(canvas, text_cached("LoT COMICS", BANG, 110, (255, 255, 255, 255), stroke=5, shadow=INK + (255,),
                              shadow_off=(6, 6)), 60, 78, anchor="l")
    box = Image.new("RGBA", (200, 130), (255, 255, 255, 255))
    bd = ImageDraw.Draw(box)
    bd.rectangle([0, 0, 199, 129], outline=INK + (255,), width=5)
    bd.text((24, 12), "No. 1", font=fnt(BANG, 52), fill=INK + (255,))
    bd.text((24, 72), "20 SEC", font=fnt(NEUE, 34), fill=RED + (255,))
    paste(canvas, box, W - 270, 14, anchor="tl")
    s1 = back_out((t - 0.15) / 0.3)
    s2 = back_out((t - 0.35) / 0.3)
    t1 = text_cached("LEVEL-OF-TOKEN", BANG, 210, CAP_YEL + (255,), stroke=9, shadow=(140, 10, 20, 255),
                     shadow_off=(14, 14))
    t2 = text_cached("DIFFUSION!", BANG, 280, (255, 255, 255, 255), stroke=10, shadow=(140, 10, 20, 255),
                     shadow_off=(16, 16))
    paste(canvas, t1, cx, 420, scale=s1, rot=3)
    paste(canvas, t2, cx, 650, scale=s2, rot=-2)
    if t > 0.7:
        paste(canvas, caption_box("WHERE DETAIL GOES, TOKENS FOLLOW!", 900, 46), W / 2 - 450, 880, anchor="tl",
              alpha=clamp((t - 0.7) / 0.15))
    return canvas


# ------------------------------------------------------------------ page flip
def flip(old, new, p):
    """old page curls away to the left revealing new; p in [0, 1]."""
    p = ease_io(p)
    xf = int(W * (1 - p))
    out = new.copy()
    if xf > 0:
        out[:, :xf] = old[:, :xf]
    cw = int(160 * math.sin(math.pi * p))                 # back of the curling page
    if cw > 0:
        x0 = max(0, xf - cw)
        g = np.linspace(0.75, 1.0, xf - x0, dtype=np.float32)[None, :, None]
        out[:, x0:xf] = np.array(PAPER, np.float32) * g
        out[:, x0:x0 + 3] = INK
        sw = min(W - xf, 60)                             # shadow on the new page
        if sw > 0:
            sh = np.linspace(0.55, 1.0, sw, dtype=np.float32)[None, :, None]
            out[:, xf:xf + sw] = out[:, xf:xf + sw] * sh
    return out


def shake(t):
    for t0, amp in ((0.15, 10), (T_SP1, 10), (T_SP2, 14), (T_BSP1, 10), (T_BSP2, 14), (T_END, 8)):
        dt = t - t0
        if 0 <= dt < 0.3:
            k = amp * math.exp(-dt / 0.08)
            return int(k * math.sin(dt * 80)), int(k * math.cos(dt * 63))
    return 0, 0


def render(t):
    def page_at(tt):
        if tt < T_FLIP1[0]:
            return np.asarray(cover(tt).convert("RGB")).astype(np.float32)
        if tt < T_FLIP2[0]:
            return np.asarray(page1(tt).convert("RGB")).astype(np.float32)
        return np.asarray(page2(tt).convert("RGB")).astype(np.float32)

    if T_FLIP1[0] <= t < T_FLIP1[1]:
        f = flip(np.asarray(cover(T_FLIP1[0] - 1e-3).convert("RGB")).astype(np.float32),
                 np.asarray(page1(t).convert("RGB")).astype(np.float32), (t - T_FLIP1[0]) / (T_FLIP1[1] - T_FLIP1[0]))
    elif T_FLIP2[0] <= t < T_FLIP2[1]:
        f = flip(np.asarray(page1(T_FLIP2[0] - 1e-3).convert("RGB")).astype(np.float32),
                 np.asarray(page2(t).convert("RGB")).astype(np.float32), (t - T_FLIP2[0]) / (T_FLIP2[1] - T_FLIP2[0]))
    else:
        f = page_at(t)
    dx, dy = shake(t)
    if dx or dy:
        f = np.roll(f, (dy, dx), (0, 1))
    return np.clip(f, 0, 255).astype(np.uint8)


# ------------------------------------------------------------------ audio (cartoon SFX)
SR = 48000


def env_exp(n, decay):
    return np.exp(-np.arange(n) / SR / decay)


def lp(x, fc, order=2):
    b, a = butter(order, min(fc / (SR / 2), 0.99))
    return lfilter(b, a, x)


def bp(x, f0, f1, order=2):
    b, a = butter(order, [f0 / (SR / 2), min(f1 / (SR / 2), 0.99)], btype="band")
    return lfilter(b, a, x)


def whoosh(dur, vol, seed, up_=True):
    n = int(dur * SR)
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n)
    out = np.zeros(n)
    blk = int(0.01 * SR)
    for i in range(0, n, blk):
        fr = i / n
        fc = 300 + 3500 * (fr if up_ else 1 - fr)
        seg = x[max(0, i - 256):i + blk]
        out[i:i + blk] = bp(seg, fc * 0.6, fc * 1.6)[-len(out[i:i + blk]):]
    env = np.sin(np.linspace(0, np.pi, n)) ** 1.5
    return out * env * vol / (np.abs(out).max() + 1e-6)


def boing(dur=0.45, f0=180, vol=0.35):
    tt = np.arange(int(dur * SR)) / SR
    f = f0 * (1 + 0.5 * np.exp(-tt / 0.12)) + 40 * np.sin(2 * np.pi * 14 * tt) * np.exp(-tt / 0.2)
    ph = np.cumsum(f) / SR
    return np.sin(2 * np.pi * ph) * np.exp(-tt / 0.18) * vol


def brass(freqs, dur=0.55, vol=0.3):
    tt = np.arange(int(dur * SR)) / SR
    x = np.zeros_like(tt)
    for f in freqs:
        vib = f * (1 + 0.006 * np.sin(2 * np.pi * 5.5 * tt))
        ph = np.cumsum(vib) / SR
        x += 2 * (ph % 1) - 1
    x = lp(x, 2600)
    env = np.minimum(tt / 0.015, 1) * np.exp(-tt / (dur * 0.5))
    return x / len(freqs) * env * vol


def timpani(f=70, dur=0.6, vol=0.5):
    tt = np.arange(int(dur * SR)) / SR
    return np.sin(2 * np.pi * f * tt * (1 + 0.3 * np.exp(-tt / 0.03))) * np.exp(-tt / 0.2) * vol


def page_flip(seed, vol=0.3):
    rng = np.random.default_rng(seed)
    n = int(0.35 * SR)
    x = rng.normal(0, 1, n)
    x = bp(x, 1500, 7000)
    env = np.exp(-((np.arange(n) / SR - 0.12) / 0.07) ** 2) + 0.6 * np.exp(-((np.arange(n) / SR - 0.27) / 0.03) ** 2)
    return x / np.abs(x).max() * env * vol


def scratch(dur, vol, seed):
    rng = np.random.default_rng(seed)
    n = int(dur * SR)
    x = bp(rng.normal(0, 1, n), 2500, 9000)
    am = np.repeat(rng.uniform(0.2, 1, n // 1200 + 1), 1200)[:n]
    return x / np.abs(x).max() * am * vol


def plop(f0=600, vol=0.25):
    tt = np.arange(int(0.09 * SR)) / SR
    ph = np.cumsum(f0 * np.exp(-tt / 0.04)) / SR
    return np.sin(2 * np.pi * ph) * np.exp(-tt / 0.04) * vol


def note(name):
    names = {"C": -9, "C#": -8, "D": -7, "D#": -6, "E": -5, "F": -4, "F#": -3, "G": -2, "G#": -1, "A": 0, "A#": 1,
             "B": 2}
    return 440 * 2 ** ((names[name[:-1]] + 12 * (int(name[-1]) - 4)) / 12)


def pluck(f, dur=0.25, vol=0.3):
    tt = np.arange(int(dur * SR)) / SR
    x = np.sin(2 * np.pi * f * tt) + 0.4 * np.sin(4 * np.pi * f * tt)
    return x * np.exp(-tt / 0.12) * vol


def build_audio(path):
    out = np.zeros(int((DUR + 0.5) * SR))

    def add(sig, t0):
        i = int(t0 * SR)
        j = min(len(out), i + len(sig))
        out[i:j] += sig[:j - i]

    # cover slams
    add(timpani(60, 0.7, 0.6), 0.15)
    add(brass([note("C4"), note("E4"), note("G4")], 0.5, 0.35), 0.15)
    add(brass([note("F4"), note("A4"), note("C5")], 0.7, 0.35), 0.35)
    add(page_flip(1), T_FLIP1[0] + 0.05)
    add(boing(), T_BUB1[0])
    for win, sd in ((S1, 3), (B1, 4)):
        add(scratch(win[1] - win[0], 0.12, sd), win[0])
    for win, base in ((S2, 900), (S3, 700), (B2, 900), (B3, 700)):
        for k in range(12):
            add(plop(base * 2 ** (-k / 14), 0.22), win[0] + (k + 0.5) / 12 * (win[1] - win[0]))
    stabs = {T_SP1: ["C4", "E4", "G4"], T_SP2: ["D4", "F#4", "A4", "D5"], T_BSP1: ["C4", "E4", "G4"],
             T_BSP2: ["D4", "F#4", "A4", "D5"]}
    for k, (t0, chord) in enumerate(stabs.items()):
        add(whoosh(0.45, 0.45, 10 + k), t0 - 0.3)
        add(brass([note(c) for c in chord], 0.6, 0.42), t0)
        add(timpani(55, 0.6, 0.55), t0)
    for k in range(10):
        add(plop(1200 + 120 * k, 0.12), T_REV[0] + k * 0.06)
    add(whoosh(0.6, 0.25, 30, up_=False), T_COL[0])
    add(boing(0.4, 240, 0.3), T_BUB2[0])
    add(page_flip(2), T_FLIP2[0] + 0.05)
    add(brass([note("C4"), note("E4"), note("G4"), note("C5")], 1.2, 0.4), T_END)
    add(timpani(50, 0.9, 0.55), T_END)
    add(brass([note("G4"), note("C5")], 0.25, 0.25), T_END - 0.25)
    # walking bass, tempo × current speed
    walk = ["C2", "E2", "G2", "A2", "C3", "A2", "G2", "E2"]
    for seg, spf in (((T_FLIP1[1], T_FLIP2[0]), lambda tt: 1.0 if tt < T_SP1 else (1.9 if tt < T_SP2 else 2.4)),
                     ((T_FLIP2[1], T_END), vid_speed)):
        tt, k = seg[0], 0
        while tt < seg[1]:
            add(pluck(note(walk[k % 8]), 0.3, 0.32), tt)
            tt += 0.4 / spf(tt)
            k += 1
    out = out / max(1e-6, np.abs(out).max()) * 0.89
    st = np.stack([out, out], 1)
    wavfile.write(path, SR, (st * 32767).astype(np.int16))


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
        for s in sys.argv[2:]:
            Image.fromarray(render(float(s))).save(os.path.join(HERE, f"scratch/still_{TAG}_{float(s):05.2f}.png"))
        return
    w, h, fps = (1920, 1080, 60) if mode == "h" else (960, 540, 30)
    res = "1080p60" if mode == "h" else "540p30"
    wav = os.path.join(HERE, f"media/final/LoT_comic_{TAG}.wav")
    build_audio(wav)
    out = os.path.join(HERE, f"media/final/LoT_comic_{TAG}_{res}.mp4")
    nf = int(DUR * fps)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps),
           "-i", "-", "-i", wav, "-c:v", "libx264", "-preset", "slow", "-crf", "16", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-ac", "2", "-ar", "48000", "-b:a", "192k", "-t", f"{DUR}", "-movflags", "+faststart", out]
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
