"""LoT Diffusion, 20-s falling-block puzzle clip (Game Boy look), v2 (v1 + title layout fix, bigger captions). numpy + PIL + cv2 frames → ffmpeg, synthesized
Game Boy-style sound (pulse / wave / noise channels). No manim.

  0.0–1.3  title screen, PUSH START, wipe
  1.3–2.4  the depth map fills the well (in focus = detail), then fades to a ghost
  2.4–4.9  LEVEL 1: 4,096 1×1 blocks rain into the well (one block per patch)
  4.9–6.4  out-of-focus blocks blink like a line clear and lock into 2×2 (1,606)   → LEVEL UP, SPEED 1.9X
  7.5–8.7  again into 4×4 / 8×8 = the real LoT layout (1,027)                      → LEVEL UP, SPEED 2.4X
  9.9–12.2 every block flips over into the generated image → STAGE CLEAR
 12.2–13.4 curtain + BONUS STAGE
 13.4–18.5 platformer video (left screen) + its tokens (right screen): rain → 2×2 → 4×4, video playback 1.0 → 2.2 → 2.5X
 18.5–20.0 end card

Speedups as in ../arcade_clip (image: fit of the paper's LoT-FLUX.2 4B timings; video: Tab. 2 LoT-Wan2.1 14B).
Usage: python render_blocks_v2.py h            → media/final/LoT_blocks_v2_1080p60.mp4
       python render_blocks_v2.py still t ...  → scratch/still_v2_<t>.png
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
TAG = "v2"
FONT = os.path.join(HERE, "data/Silkscreen-Bold.ttf")
FONT_R = os.path.join(HERE, "data/Silkscreen-Regular.ttf")
CUE = os.path.join(ROOT, "data/img/img_000000000025_dof_cue.png")
GEN = os.path.join(ROOT, "data/img/img_000000000025_dof_generated_4b.png")
LAYOUT = os.path.join(ROOT, "data/layouts/img_000000000025_dof.json")
VID = os.path.join(ROOT, "data/clips/vd_platformer_gen")
LEVELS = os.path.join(ROOT, "arcade_clip/data/platformer_levels.npy")

W, H = 1920, 1080
DUR = 20.0

# classic 4-shade handheld LCD palette, darkest → lightest
G0 = (15, 56, 15)
G1 = (48, 98, 48)
G2 = (139, 172, 15)
G3 = (155, 188, 15)
WHITE = (255, 255, 255)

# ------------------------------------------------------------------ timeline (s)
T_START = 0.85
WIPE = (1.0, 1.3)
T_DEPTH = 1.3
RAIN = (2.45, 2.1, 0.3)        # start, spread of landings, fall time
M1 = (5.0, 6.3)
LV2 = 6.4
M2 = (7.5, 8.6)
LV3 = 8.7
FLIP = (9.9, 0.75, 0.22)       # start, wave spread, flip time
T_CLEAR = 10.95
CURTAIN = (12.2, 12.7)
T_BONUS_MSG = 12.7
T_B0 = 13.4
BRAIN = (13.6, 1.0, 0.3)
BM1 = (14.95, 15.65)
BLV2 = 15.75
BM2 = (16.75, 17.35)
BLV3 = 17.45
T_END = 18.5
LV_HOLD = 1.0

IMG_SPEED = (1.0, 1.9, 2.4)
VID_SPEED = (1.0, 2.2, 2.5)
VID_TOK = (75600, 38046, 34698)

BLINK = 0.24
DROP = 0.13

# ------------------------------------------------------------------ geometry
P = 768
PX0, PY0 = (W - P) // 2, 150
N = 64
CS = P // N                     # 12 px tokens
VW, VH = 880, 495
VX0L, VX0R, VY0 = 60, 980, 330
VN, VM = 45, 80
VCS = 11


def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def ease_out(x):
    x = clamp(x)
    return 1 - (1 - x) ** 3


def back_out(x, s=2.0):
    x = clamp(x) - 1
    return 1 + (s + 1) * x ** 3 + s * x ** 2


def blink(t, hz=2.0):
    return int(t * hz * 2) % 2 == 0


# ------------------------------------------------------------------ assets
A = None


def assets():
    global A
    if A is not None:
        return A
    a = {}
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
    a["rects"] = rects
    ii, jj = np.mgrid[0:N, 0:N]
    a["blocks"] = {}
    for cap in (1, 2, 8):
        b = np.minimum(size, cap)
        a["blocks"][cap] = (r0m + (ii - r0m) // b * b, c0m + (jj - c0m) // b * b, b)
    a["lv"] = np.load(LEVELS)
    a["vid"] = [cv2.resize(np.asarray(Image.open(os.path.join(VID, f"{f + 1:04d}.jpg")).convert("RGB")),
                           (VW, VH), interpolation=cv2.INTER_AREA).astype(np.float32) for f in range(81)]
    a["vid_cell"] = [cv2.resize(v, (VM, VN), interpolation=cv2.INTER_AREA) for v in a["vid"]]
    a["YX"] = np.mgrid[0:P, 0:P]
    a["VYX"] = np.mgrid[0:VH, 0:VW]
    lut = cv2.applyColorMap(np.arange(256, dtype=np.uint8)[:, None], cv2.COLORMAP_TURBO)[:, 0, ::-1].astype(int)
    a["turbo"] = lut
    a["focus_v"] = int(np.argmin(((lut - np.array([44, 208, 165])) ** 2).sum(1)))
    rng = np.random.default_rng(4)
    a["jit"] = rng.uniform(0, 1, (N, N))
    a["vjit"] = rng.uniform(0, 1, (VN, VM))
    # static renders of the image well
    a["R"] = {cap: render_blocks(a["cue_cell"], a["blocks"][cap], CS, a["YX"]) for cap in (1, 2, 8)}
    a["Rgen"] = render_blocks(None, a["blocks"][8], CS, a["YX"], base_px=a["gen"], strength=0.45)
    a["ghost"] = a["cue"] * 0.3 + np.array(G0, np.float32) * 0.7
    # lcd dot matrix
    lcd = np.ones((H, W, 1), np.float32)
    lcd[3::4, :] *= 0.9
    lcd[:, 3::4] *= 0.9
    a["lcd"] = lcd
    A = a
    return a


# ------------------------------------------------------------------ blocks
def ids_of(blocks, m):
    return blocks[0] * m + blocks[1]


def mosaic(cell, ids, n_tok):
    flat = ids.ravel()
    cnt = np.bincount(flat, minlength=n_tok)
    out = np.empty(cell.shape, np.float32)
    for ch in range(3):
        s = np.bincount(flat, weights=cell[..., ch].ravel(), minlength=n_tok)
        out[..., ch] = (s / np.maximum(cnt, 1))[flat].reshape(ids.shape)
    return out


def up(arr, c):
    return np.repeat(np.repeat(arr, c, 0), c, 1)


def render_blocks(cell, blocks, cs, YX, base_px=None, strength=1.0):
    """bevelled blocks: 1-px dark outline, light top/left bevel, dark bottom/right bevel, a shine pixel."""
    oi, oj, b = blocks
    n, m = oi.shape
    if base_px is None:
        base = up(mosaic(cell, ids_of(blocks, m), n * m), cs)
    else:
        base = base_px.copy()
    Y, X = YX
    top = up(oi, cs) * cs
    left = up(oj, cs) * cs
    sz = up(b, cs) * cs
    dy0, dx0 = Y - top, X - left
    dy1, dx1 = top + sz - 1 - Y, left + sz - 1 - X
    T = np.clip(sz // 6, 2, 6)
    hl = (dy0 < T) | (dx0 < T)
    sh = (dy1 < T) | (dx1 < T)
    both = hl & sh
    hw = np.minimum(dy0, dx0) < np.minimum(dy1, dx1)
    hl &= ~both | hw
    sh &= ~both | ~hw
    out = base
    out[hl] = out[hl] + (255 - out[hl]) * 0.45 * strength
    out[sh] = out[sh] * (1 - 0.45 * strength)
    shine = (dy0 >= T) & (dy0 < T + 2) & (dx0 >= T) & (dx0 < T + 2) & (sz >= 10)
    out[shine] = out[shine] + (255 - out[shine]) * 0.8 * strength
    ol = (dy0 == 0) | (dx0 == 0) | (dy1 == 0) | (dx1 == 0)
    out[ol] = out[ol] * (1 - strength) + np.array(G0, np.float32) * strength
    return out


def rain_frame(R1, ghost, cs, n, t, rain, jit):
    """1×1 blocks land bottom row first; falling blocks are returned as (tile, x, y)."""
    r0, spread, fd = rain
    ii = np.arange(n)[:, None]
    t_land = r0 + fd + (n - 1 - ii) / n * spread + jit * spread / n * 2.5
    landed = t >= t_land
    img = np.where(up(landed, cs)[..., None], R1, ghost)
    fall = []
    fi, fj = np.nonzero((t >= t_land - fd) & ~landed)
    for i, j in zip(fi, fj):
        p = 1 - (t_land[i, j] - t) / fd
        y_rest = i * cs
        y = y_rest - (1 - p * p) * (y_rest + 2 * cs)
        fall.append((R1[i * cs:(i + 1) * cs, j * cs:(j + 1) * cs], j * cs, int(y)))
    return img, fall, int(landed.sum()), t_land


def merge_frame(R_old, R_new, ghost, old, new, cs, n, t, win, speed):
    """groups of small blocks blink (line clear), vanish, and the merged block drops in; bottom rows first."""
    oi_n, oj_n, b_n = new
    merged = b_n > old[2]
    drop = DROP / speed
    span = (win[1] - win[0]) - BLINK - drop
    t_g = win[0] + (n - (oi_n + b_n)) / n * span
    st_blink = merged & (t >= t_g) & (t < t_g + BLINK)
    st_drop = merged & (t >= t_g + BLINK) & (t < t_g + BLINK + drop)
    st_new = merged & (t >= t_g + BLINK + drop)
    img = R_old.copy()
    m_new = up(st_new, cs)
    img[m_new] = R_new[m_new]
    flash = np.where(st_new, np.exp(-np.maximum(t - t_g - BLINK - drop, 0) / 0.12), 0)
    on = st_blink & (((t - t_g) / 0.06).astype(int) % 2 == 0)
    k = up(np.maximum(flash * 0.8, on * 0.75), cs)[..., None]
    img = img + (255 - img) * k
    m_drop = up(st_drop, cs)
    img[m_drop] = ghost[m_drop]
    fall = []
    di, dj = np.nonzero(st_drop & (oi_n == np.arange(n)[:, None]) & (oj_n == np.arange(oi_n.shape[1])[None, :]))
    for i, j in zip(di, dj):
        b = b_n[i, j]
        p = clamp((t - t_g[i, j] - BLINK) / drop)
        y0 = i * cs
        y = y0 - (1 - p * p) * b * cs * 1.2
        fall.append((R_new[y0:y0 + b * cs, j * cs:(j + b) * cs], j * cs, int(y)))
    use_new = merged & (t >= t_g + BLINK)
    ids = np.where(use_new, oi_n * 1000 + oj_n, old[0] * 1000 + old[1])
    bsz = np.where(use_new, b_n, old[2])
    counts = {s: int((bsz == s).sum()) // (s * s) for s in (1, 2, 4, 8)}
    return img, fall, len(np.unique(ids)), counts, t_g, merged


def blit(img, fall):
    h, w = img.shape[:2]
    for tile, x, y in fall:
        th, tw = tile.shape[:2]
        y0, y1 = max(0, y), min(h, y + th)
        if y1 <= y0:
            continue
        img[y0:y1, x:x + tw] = tile[y0 - y:y1 - y]
    return img


def flip_frame(t):
    """every block flips around its vertical axis: depth block → image block; diagonal wave from bottom-left."""
    a = assets()
    oi, oj, b = a["blocks"][8]
    f0, spread, ft = FLIP
    tf = f0 + ((N - oi - b) + oj) / (2 * N) * spread
    p = np.clip((t - tf) / ft, 0, 1)
    Y, X = a["YX"]
    left = up(oj, CS) * CS
    sz = up(b, CS) * CS
    pp = up(p, CS)
    s = np.abs(np.cos(np.pi * pp))
    cx = left + sz / 2
    rel = X + 0.5 - cx
    valid = np.abs(rel) <= sz / 2 * s
    sx = np.clip((cx + rel / np.maximum(s, 1e-3)).astype(int), 0, P - 1)
    src_new = pp >= 0.5
    out = np.where(src_new[..., None], a["Rgen"][Y, sx], a["R"][8][Y, sx])
    out = np.where(valid[..., None], out, np.array(G0, np.float32))
    k = clamp((t - (T_CLEAR + 0.5)) / 0.4)
    if k > 0:
        out = out * (1 - k) + a["gen"] * k
    return out


# ------------------------------------------------------------------ video
LOOP = 73                      # loop the 81-frame clip with an 8-frame crossfade


def vid_pos(t):
    segs = [(T_B0, BLV2, VID_SPEED[0]), (BLV2, BLV3, VID_SPEED[1]), (BLV3, 99, VID_SPEED[2])]
    v = 0.0
    for s0, s1, sp in segs:
        if t > s0:
            v += (min(t, s1) - s0) * sp * 15
    return v


def vid_frame(t):
    a = assets()
    v = vid_pos(t)
    if v < LOOP:
        i = int(v)
        return a["vid"][i], a["vid_cell"][i], a["lv"][i]
    p = (v - LOOP) % LOOP
    i = int(p)
    if i < 81 - LOOP:
        w = (p - 0) / (81 - LOOP)                 # tail (i + LOOP) fades into head (i)
        j = i + LOOP
        fr = a["vid"][j] * (1 - w) + a["vid"][i] * w
        ce = a["vid_cell"][j] * (1 - w) + a["vid_cell"][i] * w
        return fr, ce, a["lv"][i if w > 0.5 else j]
    return a["vid"][i], a["vid_cell"][i], a["lv"][i]


def vid_speed(t):
    return VID_SPEED[0] if t < BLV2 else (VID_SPEED[1] if t < BLV3 else VID_SPEED[2])


def vid_blocks(lv, cap):
    b = np.minimum(lv, cap).astype(int)
    ii, jj = np.mgrid[0:VN, 0:VM]
    return ii // b * b, jj // b * b, b


def bonus_right(t, frame, cell, lv):
    a = assets()
    ghost = frame * 0.3 + np.array(G0, np.float32) * 0.7
    if t < BRAIN[0]:
        return ghost, 0
    if t < BM1[0]:
        R1 = render_blocks(cell, vid_blocks(lv, 1), VCS, a["VYX"])
        img, fall, k, _ = rain_frame(R1, ghost, VCS, VN, t, BRAIN, a["vjit"])
        frac = k / (VN * VM)
        return blit(img, fall), int(VID_TOK[0] * frac)
    if t < BM2[0]:
        old, new, win, sp, kk = vid_blocks(lv, 1), vid_blocks(lv, 2), BM1, VID_SPEED[0], 1
    else:
        old, new, win, sp, kk = vid_blocks(lv, 2), vid_blocks(lv, 4), BM2, VID_SPEED[1], 2
    R_old = render_blocks(cell, old, VCS, a["VYX"])
    R_new = render_blocks(cell, new, VCS, a["VYX"])
    img, fall, _, _, _, _ = merge_frame(R_old, R_new, ghost, old, new, VCS, VN, t, win, sp)
    f = clamp((t - win[0]) / (win[1] - win[0]))
    tok = VID_TOK[kk - 1] + (VID_TOK[kk] - VID_TOK[kk - 1]) * f
    return blit(img, fall), int(round(tok))


# ------------------------------------------------------------------ text / ui
@lru_cache(maxsize=256)
def font(size, bold=True):
    return ImageFont.truetype(FONT if bold else FONT_R, size)


@lru_cache(maxsize=4096)
def text_rgba(s, size, color, shadow=None, bold=True):
    f = font(size, bold)
    off = max(2, size // 10) if shadow else 0
    bb = f.getbbox(s)
    w, h = bb[2] + off + 2, bb[3] + off + 2
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    if shadow:
        d.text((off, off), s, font=f, fill=shadow + (255,))
    d.text((0, 0), s, font=f, fill=color + (255,))
    return im


def put_text(canvas, s, size, color, x, y, anchor="c", scale=1.0, shadow=None, bold=True):
    if scale <= 0.01 or not s:
        return
    im = text_rgba(s, size, color, shadow, bold)
    if abs(scale - 1) > 1e-3:
        im = im.resize((max(1, int(im.width * scale)), max(1, int(im.height * scale))), Image.NEAREST)
    if anchor == "c":
        x, y = x - im.width // 2, y - im.height // 2
    elif anchor == "r":
        x = x - im.width
    paste_clip(canvas, im, int(x), int(y))


def paste_clip(canvas, im, x, y):
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + im.width), min(H, y + im.height)
    if x1 > x0 and y1 > y0:
        canvas.alpha_composite(im.crop((x0 - x, y0 - y, x1 - x, y1 - y)), (x0, y0))


def panel(d, x0, y0, x1, y1, fill=G2):
    """handheld-style box: dark outer border, mid inner line."""
    d.rectangle([x0, y0, x1, y1], fill=G0)
    d.rectangle([x0 + 5, y0 + 5, x1 - 5, y1 - 5], fill=G1)
    d.rectangle([x0 + 9, y0 + 9, x1 - 9, y1 - 9], fill=fill)


def rect(d, x0, y0, x1, y1, fill):
    if x1 >= x0 and y1 >= y0:
        d.rectangle([x0, y0, x1, y1], fill=fill)


def brick_frame(d, x0, y0, x1, y1, th=28):
    """brick wall around a well."""
    d.rectangle([x0 - th, y0 - th, x1 + th, y1 + th], fill=G0)
    bh, bw = th // 2, th
    for k, (yy0, yy1) in enumerate([(y0 - th, y0), (y1, y1 + th)]):
        for r in range(2):
            ry = yy0 + r * bh
            off = (r % 2) * bw // 2
            for x in range(x0 - th - off, x1 + th, bw):
                rect(d, max(x0 - th, x + 2), ry + 2, min(x1 + th, x + bw - 2), ry + bh - 2, G1)
    for xx0, xx1 in [(x0 - th, x0), (x1, x1 + th)]:
        for r, ry in enumerate(range(y0, y1, bh)):
            off = (r % 2) * bw // 2
            for x in (xx0 - off, xx0 - off + bw):
                rect(d, max(xx0, x + 2), ry + 2, min(xx1, x + bw - 2), min(y1, ry + bh - 2), G1)
    d.rectangle([x0 - 4, y0 - 4, x1 + 3, y1 + 3], outline=G0, width=4)


def dialog(canvas, lines, cx, cy, scale=1.0, pad=36):
    """lines = [(text, size, color)] in a handheld-style box, scaled with NEAREST for a pop."""
    if scale <= 0.02:
        return
    ims = [text_rgba(s, sz, c, None) for s, sz, c in lines]
    w = max(i.width for i in ims) + 2 * pad
    h = sum(i.height for i in ims) + 2 * pad + 16 * (len(ims) - 1)
    box = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(box)
    panel(d, 0, 0, w - 1, h - 1, fill=G3)
    y = pad
    for im in ims:
        box.alpha_composite(im, ((w - im.width) // 2, y))
        y += im.height + 16
    if abs(scale - 1) > 1e-3:
        box = box.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.NEAREST)
    paste_clip(canvas, box, int(cx - box.width / 2), int(cy - box.height / 2))


def mini_block(d, x, y, s, col=G1):
    t = max(2, s // 6)
    d.rectangle([x, y, x + s - 1, y + s - 1], fill=G0)
    d.rectangle([x + 1, y + 1, x + s - 2, y + s - 2], fill=col)
    d.rectangle([x + 1, y + 1, x + s - 2, y + t], fill=G3)
    d.rectangle([x + 1, y + 1, x + t, y + s - 2], fill=G3)
    d.rectangle([x + 1, y + s - 1 - t, x + s - 2, y + s - 2], fill=G0)
    d.rectangle([x + s - 1 - t, y + 1, x + s - 2, y + s - 2], fill=G0)


def caption(canvas, d, s, t, t0, t1, cx=W // 2, y=1012, w=1000):
    if not (t0 <= t < t1):
        return
    tw = font(40).getlength(s)
    w = max(w, int(tw) + 100)
    panel(d, cx - w // 2, y - 44, cx + w // 2, y + 44, fill=G3)
    k = int((t - t0) * 45)
    put_text(canvas, s[:max(1, k)], 40, G0, cx - tw / 2, y - 22, anchor="l")


def level_up(canvas, t, t0, level, speed, cx=W // 2, cy=540):
    dt = t - t0
    if 0 <= dt < LV_HOLD:
        s = back_out(dt / 0.2) * (1 - clamp((dt - LV_HOLD + 0.15) / 0.15))
        col = G0 if blink(t, 5) or dt > 0.5 else G1
        dialog(canvas, [("LEVEL UP!", 72, col), (f"LEVEL {level}", 40, G1), (f"SPEED {speed:.1f}X", 56, G0)], cx, cy, s)


# ------------------------------------------------------------------ HUD
def hud_image(canvas, d, t, tokens, level, speed, nxt, counts):
    x0, x1 = 150, 470
    panel(d, x0, 150, x1, 300)
    put_text(canvas, "TOKENS", 32, G0, x0 + 30, 172, anchor="l")
    put_text(canvas, f"{tokens:5d}", 64, G0, x1 - 30, 214, anchor="r")
    panel(d, x0, 330, x1, 480)
    put_text(canvas, "LEVEL", 32, G0, x0 + 30, 352, anchor="l")
    put_text(canvas, f"{level}", 64, G0, x1 - 30, 394, anchor="r")
    panel(d, x0, 510, x1, 660)
    put_text(canvas, "SPEED", 32, G0, x0 + 30, 532, anchor="l")
    put_text(canvas, f"{speed:.1f}X", 64, G0, x1 - 30, 574, anchor="r")
    # stats: count of each block size
    panel(d, x0, 690, x1, 918)
    put_text(canvas, "BLOCKS", 24, G0, x0 + 30, 708, anchor="l")
    for k, s in enumerate((1, 2, 4, 8)):
        y = 752 + k * 40
        px = (6, 12, 20, 28)[k]
        mini_block(d, x0 + 44 - px // 2, y + 14 - px // 2, px)
        put_text(canvas, f"{s}X{s}", 24, G1, x0 + 90, y + 2, anchor="l")
        put_text(canvas, f"{counts.get(s, 0)}", 24, G0, x1 - 30, y + 2, anchor="r")
    # right column: NEXT + depth legend
    rx0, rx1 = 1450, 1770
    panel(d, rx0, 150, rx1, 400)
    put_text(canvas, "NEXT", 32, G0, rx0 + 30, 172, anchor="l")
    if nxt:
        s = {1: 24, 2: 48, 4: 96, 8: 150}[nxt]
        mini_block(d, (rx0 + rx1) // 2 - s // 2, 300 - s // 2, s)
    panel(d, rx0, 430, rx1, 918)
    put_text(canvas, "DEPTH", 32, G0, rx0 + 30, 452, anchor="l")
    a = assets()
    bx, by, bw, bh = rx0 + 50, 540, 40, 320
    col = cv2.resize(a["turbo"][::-1][:, None, :].astype(np.uint8), (bw, bh), interpolation=cv2.INTER_NEAREST)
    canvas.paste(Image.fromarray(col), (bx, by))
    d.rectangle([bx - 4, by - 4, bx + bw + 3, by + bh + 3], outline=G0, width=4)
    put_text(canvas, "NEAR", 24, G1, bx + bw // 2 - 4, by - 32)
    put_text(canvas, "FAR", 24, G1, bx + bw // 2 - 4, by + bh + 30)
    fy = by + int((255 - a["focus_v"]) / 255 * bh)
    d.polygon([(bx + bw + 10, fy), (bx + bw + 30, fy - 14), (bx + bw + 30, fy + 14)], fill=G0)
    put_text(canvas, "IN", 32, G0, bx + bw + 42, fy - 36, anchor="l")
    put_text(canvas, "FOCUS", 32, G0, bx + bw + 42, fy + 2, anchor="l")
    put_text(canvas, "= DETAIL", 24, G1, bx + bw + 42, fy + 42, anchor="l")


def hud_bonus(canvas, d, t, tokens, level, speed):
    for k, (lab, val) in enumerate((("TOKENS", f"{tokens:6d}"), ("LEVEL", f"{level}"), ("SPEED", f"{speed:.1f}X"))):
        x0 = 60 + k * 620
        x1 = x0 + 560
        panel(d, x0, 60, x1, 190)
        put_text(canvas, lab, 32, G0, x0 + 30, 80, anchor="l")
        put_text(canvas, val, 64, G0, x1 - 30, 108, anchor="r")


# ------------------------------------------------------------------ frame
def image_level(t):
    if t < LV2:
        return 1, IMG_SPEED[0]
    if t < LV3:
        return 2, IMG_SPEED[1]
    return 3, IMG_SPEED[2]


def render(t):
    a = assets()
    canvas = Image.new("RGBA", (W, H), G3 + (255,))
    d = ImageDraw.Draw(canvas)

    # ---------------------------------------------- title
    if t < WIPE[1]:
        brick_frame(d, 60, 60, W - 61, H - 61, th=28)
        d.rectangle([60, 60, W - 61, H - 61], fill=G3)
        put_text(canvas, "LEVEL-OF-TOKEN", 120, G0, W // 2, 320, shadow=G1)
        put_text(canvas, "DIFFUSION", 168, G0, W // 2, 490, shadow=G1)
        # a few blocks stacked under the title: 1x1, 2x2, 4x4, 8x8
        for k, s in enumerate((16, 32, 64, 128)):
            mini_block(d, 795 + sum((16, 32, 64, 128)[:k]) + 30 * k, 960 - s, s)
        if t < T_START and blink(t, 2):
            put_text(canvas, "PUSH START", 48, G1, W // 2, 680)
        elif t >= T_START and blink(t, 8):
            put_text(canvas, "PUSH START", 48, G0, W // 2, 680)
        if t >= WIPE[0]:
            k = clamp((t - WIPE[0]) / (WIPE[1] - WIPE[0]))
            rows = int(k * 27)
            for r in range(rows):
                d.rectangle([0, r * 40, W, r * 40 + 39], fill=G0)
        f = np.asarray(canvas.convert("RGB")).astype(np.float32)
        return np.clip(f * a["lcd"], 0, 255).astype(np.uint8)

    # ---------------------------------------------- image well
    if t < CURTAIN[1] + 0.7:
        brick_frame(d, PX0, PY0, PX0 + P - 1, PY0 + P - 1)
        level, speed = image_level(t)
        counts = {}
        tok = 0
        nxt = None
        if t < RAIN[0]:
            k = clamp((t - (RAIN[0] - 0.35)) / 0.35)
            img = a["cue"] * (1 - k) + a["ghost"] * k
            # the depth map fills in row by row from the top
            rows = int(clamp((t - T_DEPTH) / 0.4) * P)
            img = img.copy()
            img[rows:] = np.array(G0, np.float32)
            nxt = 1
        elif t < M1[0]:
            img, fall, tok, _ = rain_frame(a["R"][1], a["ghost"], CS, N, t, RAIN, a["jit"])
            img = blit(img, fall)
            counts = {1: tok}
            nxt = 1
        elif t < FLIP[0]:
            if t < M2[0]:
                old, new, win, sp, cold, cnew = a["blocks"][1], a["blocks"][2], M1, IMG_SPEED[0], 1, 2
                nxt = 2
            else:
                old, new, win, sp, cold, cnew = a["blocks"][2], a["blocks"][8], M2, IMG_SPEED[1], 2, 8
                nxt = 8
            img, fall, tok, counts, _, _ = merge_frame(a["R"][cold], a["R"][cnew], a["ghost"], old, new, CS, N, t, win, sp)
            img = blit(img, fall)
        else:
            img = flip_frame(t)
            tok = 1027
            counts = {1: 776, 2: 174, 4: 48, 8: 29}
        if t >= CURTAIN[0]:
            k = clamp((t - CURTAIN[0]) / (CURTAIN[1] - CURTAIN[0]))
            rows = int(k * N)
            img = img.copy()
            if rows:
                cur = np.zeros((rows * CS, P, 3), np.float32) + np.array(G1, np.float32)
                cur[::CS] = np.array(G0, np.float32)
                cur[:, ::CS] = np.array(G0, np.float32)
                img[P - rows * CS:] = cur
        canvas.paste(Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)), (PX0, PY0))
        hud_image(canvas, d, t, tok, level, speed, nxt, counts)
        caption(canvas, d, "DEPTH MAP", t, T_DEPTH, RAIN[0], w=820)
        caption(canvas, d, "EVERY PATCH = 1 BLOCK", t, RAIN[0], M1[0] - 0.1, w=820)
        caption(canvas, d, "OUT OF FOCUS? MERGE!", t, M1[0] - 0.1, LV2, w=820)
        caption(canvas, d, "BIGGER BLOCKS!", t, M2[0], LV3, w=820)
        caption(canvas, d, "4X FEWER TOKENS", t, T_CLEAR + 0.7, CURTAIN[0], w=820)
        level_up(canvas, t, LV2, 2, IMG_SPEED[1])
        level_up(canvas, t, LV3, 3, IMG_SPEED[2])
        if T_CLEAR <= t < T_CLEAR + 0.75:
            dt = t - T_CLEAR
            s = back_out(dt / 0.2) * (1 - clamp((dt - 0.6) / 0.15))
            dialog(canvas, [("STAGE CLEAR!", 72, G0)], W // 2, 540, s)
        if t >= T_BONUS_MSG:
            dt = t - T_BONUS_MSG
            col = G0 if blink(t, 4) else G1
            dialog(canvas, [("BONUS", 96, col), ("STAGE", 96, col)], W // 2, PY0 + P // 2, back_out(dt / 0.2))

    # ---------------------------------------------- bonus
    else:
        frame, cell, lv = vid_frame(t)
        right, tok = bonus_right(t, frame, cell, lv)
        s = back_out((t - T_B0) / 0.25)
        for k, (x0, im) in enumerate(((VX0L, frame), (VX0R, right))):
            pim = Image.fromarray(np.clip(im, 0, 255).astype(np.uint8))
            if s < 0.999:
                pim = pim.resize((max(2, int(VW * s)), max(2, int(VH * s))), Image.NEAREST)
            cx, cy = x0 + VW // 2, VY0 + VH // 2
            if s > 0.999:
                brick_frame(d, x0, VY0, x0 + VW - 1, VY0 + VH - 1, th=20)
            canvas.paste(pim, (cx - pim.width // 2, cy - pim.height // 2))
        sp = vid_speed(t)
        level = 1 if t < BLV2 else (2 if t < BLV3 else 3)
        hud_bonus(canvas, d, t, tok, level, sp)
        put_text(canvas, "VIDEO", 32, G0, VX0L, VY0 - 62, anchor="l")
        put_text(canvas, "TOKENS", 32, G0, VX0R, VY0 - 62, anchor="l")
        # playback badge
        n_tri = 1 if sp < 1.5 else (2 if sp < 2.3 else 3)
        bx, by = VX0L + VW - 140 - n_tri * 30, VY0 - 64
        for k in range(n_tri):
            d.polygon([(bx + k * 30, by), (bx + k * 30 + 26, by + 15), (bx + k * 30, by + 30)], fill=G0)
        put_text(canvas, f"{sp:.1f}X", 32, G0, VX0L + VW, VY0 - 62, anchor="r")
        caption(canvas, d, "EVERY FRAME, EVERY PATCH", t, BRAIN[0], BM1[0] - 0.1, y=920, w=1000)
        caption(canvas, d, "MERGE THE GROUND!", t, BM1[0] - 0.1, BLV2 + 0.2, y=920, w=1000)
        caption(canvas, d, "MERGE THE SKY!", t, BM2[0] - 0.1, BLV3 + 0.3, y=920, w=1000)
        level_up(canvas, t, BLV2, 2, VID_SPEED[1], cy=VY0 + VH // 2)
        level_up(canvas, t, BLV3, 3, VID_SPEED[2], cy=VY0 + VH // 2)
        if t >= T_END:
            dt = t - T_END
            dialog(canvas, [("LEVEL-OF-TOKEN DIFFUSION", 64, G0), ("IMAGES 2.4X FASTER", 40, G1),
                            ("VIDEO 2.5X FASTER", 40, G1)], W // 2, VY0 + VH // 2, back_out(dt / 0.25))
            if dt > 0.6 and blink(t, 2):
                caption(canvas, d, "PUSH START", t, T_END, 99, y=920, w=1000)

    f = np.asarray(canvas.convert("RGB")).astype(np.float32)
    return np.clip(f * a["lcd"], 0, 255).astype(np.uint8)


# ------------------------------------------------------------------ audio (handheld channels)
SR = 48000


def pulse(freq, dur, duty=0.5, vol=0.25, decay=None):
    n = int(dur * SR)
    tt = np.arange(n) / SR
    ph = np.cumsum(freq(tt)) / SR if callable(freq) else freq * tt
    w = np.where((ph % 1) < duty, 1.0, -1.0)
    env = np.exp(-tt / decay) if decay else np.ones(n)
    env[:30] *= np.linspace(0, 1, 30)
    env[-60:] *= np.linspace(1, 0, 60)
    return w * env * vol


def wave(freq, dur, vol=0.35, decay=None):
    """4-bit stepped triangle (wave channel)."""
    n = int(dur * SR)
    tt = np.arange(n) / SR
    tri = 4 * np.abs((freq * tt % 1) - 0.5) - 1
    w = np.round(tri * 7.5) / 7.5
    env = np.exp(-tt / decay) if decay else np.ones(n)
    env[-60:] *= np.linspace(1, 0, 60)
    return w * env * vol


def noise(dur, vol=0.2, decay=0.05, seed=0, period=4):
    n = int(dur * SR)
    rng = np.random.default_rng(seed)
    w = np.repeat(rng.choice([-1.0, 1.0], n // period + 1), period)[:n]
    return w * np.exp(-np.arange(n) / SR / decay) * vol


def note(name):
    names = {"C": -9, "C#": -8, "D": -7, "D#": -6, "E": -5, "F": -4, "F#": -3, "G": -2, "G#": -1, "A": 0, "A#": 1, "B": 2}
    return 440 * 2 ** ((names[name[:-1]] + 12 * (int(name[-1]) - 4)) / 12)


def music_speed(t):
    if t < T_B0:
        return image_level(t)[1]
    return vid_speed(t)


def build_audio(path):
    out = np.zeros(int((DUR + 0.5) * SR))

    def add(sig, t0, pan=0.0):
        i = int(t0 * SR)
        j = min(len(out), i + len(sig))
        if j > i:
            out[i:j] += sig[:j - i]

    a = assets()
    # start
    add(pulse(note("A5"), 0.08, 0.5, 0.25), T_START)
    add(pulse(note("A6"), 0.3, 0.5, 0.25, 0.12), T_START + 0.09)
    # wipe whoosh
    add(noise(0.3, 0.12, 0.2, 1, period=12), WIPE[0])
    # depth fill: rising sweep
    add(pulse(lambda tt: 200 + 1600 * tt / 0.4, 0.4, 0.125, 0.1), T_DEPTH)
    # rain: one click per landed row
    for rain, n in ((RAIN, N), (BRAIN, VN)):
        r0, spread, fd = rain
        for i in range(n):
            tk = r0 + fd + (n - 1 - i) / n * spread
            add(noise(0.012, 0.18, 0.004, i, period=2), tk)
            add(pulse(note("C6") * 2 ** ((n - 1 - i) / n), 0.015, 0.125, 0.06), tk)
    # merges: sparkle on blink, thud on lock (binned per 80 ms)
    for win, sp, n, img in ((M1, IMG_SPEED[0], N, True), (M2, IMG_SPEED[1], N, True),
                            (BM1, VID_SPEED[0], VN, False), (BM2, VID_SPEED[1], VN, False)):
        drop = DROP / sp
        span = (win[1] - win[0]) - BLINK - drop
        bands = np.arange(0, 1.0001, 0.07)
        for k, f in enumerate(bands):
            tg = win[0] + f * span
            for q, nm in enumerate(("E7", "C7", "G6", "E6")):
                add(pulse(note(nm), 0.022, 0.25, 0.07), tg + q * 0.022)
            add(noise(0.06, 0.2, 0.025, 100 + k, period=16), tg + BLINK + drop)
    # level up jingle (original)
    for t0 in (LV2, LV3, BLV2, BLV3):
        seq = [("D5", .06), ("F#5", .06), ("A5", .06), ("D6", .12), ("C#6", .06), ("E6", .06), ("A6", .3)]
        tt = t0
        for nm, du in seq:
            add(pulse(note(nm), du, 0.25, 0.2), tt)
            add(pulse(note(nm) / 2, du, 0.5, 0.08), tt)
            tt += du
    # flip ticks
    for k in range(18):
        add(pulse(note("G6") * 2 ** ((k % 6) / 12), 0.02, 0.125, 0.08), FLIP[0] + k * 0.05)
    # stage clear
    tt = T_CLEAR
    for nm, du in [("C5", .09), ("E5", .09), ("G5", .09), ("E5", .09), ("F5", .09), ("A5", .09), ("C6", .5)]:
        add(pulse(note(nm), du, 0.5, 0.22, None if du < .2 else 0.3), tt)
        add(wave(note(nm) / 2, du, 0.25), tt)
        tt += du
    # curtain
    add(pulse(lambda tt: 900 - 1400 * tt, 0.5, 0.5, 0.12), CURTAIN[0])
    # bonus trill
    for k in range(10):
        add(pulse(note("E6" if k % 2 == 0 else "B5"), 0.05, 0.25, 0.18), T_BONUS_MSG + k * 0.05)
    # end
    tt = T_END
    for nm, du in [("G5", .1), ("C6", .1), ("E6", .1), ("D6", .1), ("G6", .55)]:
        add(pulse(note(nm), du, 0.5, 0.2, None if du < .2 else 0.35), tt)
        add(wave(note(nm) / 4, du, 0.3), tt)
        tt += du
    # bouncy octave bass + hats, tempo × speed
    bass = ["C2", "C3", "G2", "C3", "A#1", "A#2", "F2", "A#2", "G#1", "G#2", "D#2", "G#2", "G1", "G2", "D2", "B2"]
    for seg in ((T_DEPTH, CURTAIN[0]), (T_B0, T_END)):
        t, k = seg[0], 0
        while t < seg[1]:
            sp = music_speed(t)
            step = 0.17 / sp
            add(wave(note(bass[k % 16]), step * 0.9, 0.32, 0.12), t)
            if k % 2 == 1:
                add(noise(0.03, 0.06, 0.01, k, period=1), t)
            t += step
            k += 1
    out = out / max(1e-6, np.abs(out).max()) * 0.89
    st = np.stack([out, out], 1)
    wavfile.write(path, SR, (st * 32767).astype(np.int16))


# ------------------------------------------------------------------ main
def render_bytes(t):
    return render(t).tobytes()


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "h"
    if mode == "still":
        os.makedirs(os.path.join(HERE, "scratch"), exist_ok=True)
        for s in sys.argv[2:]:
            Image.fromarray(render(float(s))).save(os.path.join(HERE, f"scratch/still_{TAG}_{float(s):05.2f}.png"))
        return
    fps = 60
    os.makedirs(os.path.join(HERE, "media/final"), exist_ok=True)
    wav = os.path.join(HERE, f"media/final/LoT_blocks_{TAG}.wav")
    build_audio(wav)
    out = os.path.join(HERE, f"media/final/LoT_blocks_{TAG}_1080p60.mp4")
    nf = int(DUR * fps)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(fps),
           "-i", "-", "-i", wav, "-c:v", "libx264", "-preset", "slow", "-crf", "16", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-ac", "2", "-ar", "48000", "-b:a", "192k", "-t", f"{DUR}", "-movflags", "+faststart", out]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    with Pool(10) as pool:
        for k, buf in enumerate(pool.imap(render_bytes, [i / fps for i in range(nf)], chunksize=2)):
            p.stdin.write(buf)
            if k % 120 == 0:
                print(f"frame {k}/{nf}", flush=True)
    p.stdin.close()
    p.wait()
    print("wrote", out)


if __name__ == "__main__":
    main()
