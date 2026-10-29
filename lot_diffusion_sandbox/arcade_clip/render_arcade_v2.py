"""LoT Diffusion, 20-s 80s-arcade clip, v2 (v1 + HUD out of the vignette, TA-DA moves off the image, thinner video grid) (numpy + PIL + cv2 frames → ffmpeg, synthesized chiptune SFX). No manim.

  0.0–1.6  attract screen: INSERT COIN → coin → title slam
  1.6–2.3  STAGE 1: the depth map pops up (turbo, near = red; the in-focus band = the game's green)
  2.3–3.9  sweep 1: every 16-px patch becomes a 1 × 1 token (4,096), mosaic of the depth map
  4.2–5.6  sweep 2: out-of-focus tokens merge to 2 × 2 (1,606)  → SPEEDUP 1.9X!!
  6.9–8.2  sweep 3: merge to 4 × 4 / 8 × 8 = the real LoT layout (1,027) → SPEEDUP 2.4X!!
  9.5–11.9 tokens flip to the generated image → TA-DA!!!
 11.9–12.9 tokens fall away, BONUS ROUND!!
 12.9–18.0 platformer video left, its tokens right; sweeps 1 × 1 → 2 × 2 (SPEEDUP 2.2X) → 4 × 4 (SPEEDUP 2.5X),
           the video's playback speed follows the speedup
 18.0–20.0 title banner, INSERT COIN

Speedups: image = Amdahl fit of the paper's LoT-FLUX.2 4B timings (Fig. 21: 1,752 tok 1.8×, 1,111 2.2×, 765 2.9×);
video = interpolation of Tab. 2 LoT-Wan2.1 14B (2× fewer tokens 2.26×, 2.5× → 2.91×).
Usage: python render_arcade_v1.py [l|h]       → media/final/LoT_arcade_v1_<res>.mp4
       python render_arcade_v1.py still t ...  → scratch/still_v1_<t>.png (1080p)
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
FONT = os.path.join(HERE, "data/PressStart2P-Regular.ttf")
CUE = os.path.join(ROOT, "data/img/img_000000000025_dof_cue.png")
GEN = os.path.join(ROOT, "data/img/img_000000000025_dof_generated_4b.png")
LAYOUT = os.path.join(ROOT, "data/layouts/img_000000000025_dof.json")
VID = os.path.join(ROOT, "data/clips/vd_platformer_gen")
LEVELS = os.path.join(HERE, "data/platformer_levels.npy")

W, H = 1920, 1080
DUR = 20.0

# palette: GREEN = neon version of the in-focus band of the turbo depth map (44, 208, 165)
GREEN = (60, 255, 180)
GREEN_D = (20, 120, 85)
YELLOW = (255, 232, 60)
MAGENTA = (255, 60, 170)
CYAN = (60, 220, 255)
ORANGE = (255, 140, 30)
WHITE = (255, 255, 255)
SHADOW = (90, 0, 70)
RAINBOW = [YELLOW, ORANGE, MAGENTA, CYAN, GREEN]

# ------------------------------------------------------------------ timeline (s)
T_COIN = 0.45
T_TITLE = 0.6
T_POP = 1.6
S1 = (2.3, 3.9)              # sweep windows
S2 = (4.2, 5.6)
S3 = (6.9, 8.2)
T_SP1, T_SP2 = 5.65, 8.25    # SPEEDUP popups
T_FLIP = 9.5
T_TADA = 10.15
T_FALL = 11.9
T_BONUS = 12.05
T_B0 = 12.9                  # bonus layout on
B1 = (13.0, 13.9)
B2 = (14.1, 15.0)
B3 = (16.2, 17.0)
T_BSP1, T_BSP2 = 15.05, 17.05
T_END = 18.1
SP_HOLD = 1.25

IMG_SPEED = (1.0, 1.9, 2.4)          # 4,096 / 1,606 / 1,027 tokens
VID_SPEED = (1.0, 2.2, 2.5)          # 75,600 / 38,046 / 34,698 tokens
IMG_TOK = (4096, 1606, 1027)
VID_TOK = (75600, 38046, 34698)

# ------------------------------------------------------------------ geometry
P = 768                       # playfield (64 tokens × 12 px)
PX0, PY0 = (W - P) // 2, 150
N = 64
VW, VH = 880, 495             # video panels (80 × 45 tokens × 11 px)
VX0L, VX0R, VY0 = 60, 980, 300
VN, VM = 45, 80


def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def ease_out(x):
    x = clamp(x)
    return 1 - (1 - x) ** 3


def back_out(x, s=2.2):
    x = clamp(x) - 1
    return 1 + (s + 1) * x ** 3 + s * x ** 2


# ------------------------------------------------------------------ assets
def load_assets():
    a = {}
    cue = np.asarray(Image.open(CUE).convert("RGB")).astype(np.float32)
    a["cue"] = cv2.resize(cue, (P, P), interpolation=cv2.INTER_AREA)
    a["cue_cell"] = cue.reshape(N, 16, N, 16, 3).mean((1, 3))                  # 64 × 64 token means
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
        oi = r0m + (ii - r0m) // b * b
        oj = c0m + (jj - c0m) // b * b
        a["img_blocks"][cap] = (oi, oj, b)
    a["lv"] = np.load(LEVELS)
    a["vid"] = [cv2.resize(np.asarray(Image.open(os.path.join(VID, f"{f + 1:04d}.jpg")).convert("RGB")),
                           (VW, VH), interpolation=cv2.INTER_AREA).astype(np.float32) for f in range(81)]
    a["vid_cell"] = [cv2.resize(v, (VM, VN), interpolation=cv2.INTER_AREA) for v in a["vid"]]
    # turbo colour bar, near (top) → far (bottom)
    lut = cv2.applyColorMap(np.arange(256, dtype=np.uint8)[:, None], cv2.COLORMAP_TURBO)[:, 0, ::-1].astype(int)
    a["turbo"] = lut
    a["focus_v"] = int(np.argmin(((lut - np.array([44, 208, 165])) ** 2).sum(1)))
    # starfield
    rng = np.random.default_rng(7)
    a["stars"] = (rng.uniform(0, W, 260), rng.uniform(0, H, 260), rng.choice([1, 2, 3], 260, p=[.55, .3, .15]))
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    v = 1 - 0.28 * (((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2) ** 2
    a["vignette"] = np.clip(v, 0, 1)[..., None]
    scan = np.ones((H, 1, 1), np.float32)
    scan[2::3] = 0.72
    a["scan"] = scan
    return a


A = None


def assets():
    global A
    if A is None:
        A = load_assets()
    return A


# ------------------------------------------------------------------ text
@lru_cache(maxsize=4096)
def font(size):
    return ImageFont.truetype(FONT, size)


@lru_cache(maxsize=4096)
def text_rgba(s, size, colors, shadow=True):
    """colors: one RGB tuple, or a tuple of per-character RGB tuples."""
    f = font(size)
    off = max(2, size // 8) if shadow else 0
    w = int(f.getlength(s)) + off + 2
    h = size + off + 2
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    per = isinstance(colors[0], tuple)
    x = 0
    for k, ch in enumerate(s):
        c = colors[k % len(colors)] if per else colors
        if shadow:
            d.text((x + off, off), ch, font=f, fill=SHADOW + (255,))
        d.text((x, 0), ch, font=f, fill=tuple(c) + (255,))
        x += f.getlength(ch)
    return im


def rainbow(s, t, rate=10.0):
    k = int(t * rate)
    return tuple(RAINBOW[(i + k) % len(RAINBOW)] for i in range(len(s)))


def put_text(canvas, s, size, colors, x, y, anchor="c", scale=1.0, shadow=True, alpha=1.0):
    """anchor c = centre, l = left-top, r = right-top."""
    if alpha <= 0 or scale <= 0.01:
        return
    im = text_rgba(s, size, colors, shadow)
    if abs(scale - 1) > 1e-3:
        im = im.resize((max(1, int(im.width * scale)), max(1, int(im.height * scale))), Image.NEAREST)
    if alpha < 1:
        al = im.getchannel("A").point(lambda v: int(v * alpha))
        im.putalpha(al)
    if anchor == "c":
        x, y = x - im.width // 2, y - im.height // 2
    elif anchor == "r":
        x = x - im.width
    canvas.alpha_composite(im, (int(x), int(y))) if 0 <= x and 0 <= y and x + im.width <= W and y + im.height <= H \
        else paste_clip(canvas, im, int(x), int(y))


def paste_clip(canvas, im, x, y):
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + im.width), min(H, y + im.height)
    if x1 <= x0 or y1 <= y0:
        return
    canvas.alpha_composite(im.crop((x0 - x, y0 - y, x1 - x, y1 - y)), (x0, y0))


def blink(t, hz=3.0):
    return int(t * hz * 2) % 2 == 0


# ------------------------------------------------------------------ token mosaics
def block_ids(oi, oj, m):
    return oi * m + oj


def mosaic(cell, ids, n_tok):
    """cell (n, m, 3) token means → per-cell block mean colour."""
    flat = ids.ravel()
    cnt = np.bincount(flat, minlength=n_tok)
    out = np.empty(cell.shape, np.float32)
    for ch in range(3):
        s = np.bincount(flat, weights=cell[..., ch].ravel(), minlength=n_tok)
        out[..., ch] = (s / np.maximum(cnt, 1))[flat].reshape(ids.shape)
    return out


def edges(ids_px, thick=2):  # noqa: C901
    e = np.zeros(ids_px.shape, bool)
    e[:, 1:] |= ids_px[:, 1:] != ids_px[:, :-1]
    e[1:, :] |= ids_px[1:, :] != ids_px[:-1, :]
    if thick > 1:
        e[:, :-1] |= e[:, 1:]
        e[:-1, :] |= e[1:, :]
    e[:thick, :] = e[-thick:, :] = True
    e[:, :thick] = e[:, -thick:] = True
    return e


def staged_ids(blocks_old, blocks_new, n, m, bar_tok, t0, dur, t):
    """ids where blocks fully above the sweep bar use the new cap; also the per-cell switch time for flashes."""
    oi_o, oj_o, _ = blocks_old
    oi_n, oj_n, b_n = blocks_new
    t_sw = t0 + (oi_n + b_n) / n * dur
    new = t >= t_sw
    oi = np.where(new, oi_n, oi_o)
    oj = np.where(new, oj_n, oj_o)
    return block_ids(oi, oj, m), t_sw


def up(arr, cell_px):
    return np.repeat(np.repeat(arr, cell_px, 0), cell_px, 1)


def sweep_frac(t, win):
    return clamp((t - win[0]) / (win[1] - win[0]))


# ------------------------------------------------------------------ playfield (stage 1)
def img_stage(t):
    """returns (rgb P×P float, tokens shown, bar y in px or None, line alpha)."""
    a = assets()
    B = a["img_blocks"]
    cell_px = P // N
    bar = None
    flash = np.zeros((N, N), np.float32)
    if t < S1[0]:
        return a["cue"].copy(), 0, None, 0.0
    if t < S2[0]:
        f = sweep_frac(t, S1)
        rows = int(f * N + 1e-6)
        ids = block_ids(B[1][0], B[1][1], N)
        mos = up(mosaic(a["cue_cell"], ids, N * N), cell_px)
        img = a["cue"].copy()
        ry = rows * cell_px
        img[:ry] = mos[:ry]
        e = edges(up(ids, cell_px))
        e[ry:] = False
        if f < 1:
            bar = f * P
        return img, rows * N, bar, 1.0, e, flash
    if t < S3[0]:
        old, new, win = B[1], B[2], S2
    else:
        old, new, win = B[2], B[8], S3
    f = sweep_frac(t, win)
    ids, t_sw = staged_ids(old, new, N, N, f * N, win[0], win[1] - win[0], t)
    dt = t - t_sw
    merged = (new[2] > old[2])
    flash = np.where(merged & (dt >= 0), np.exp(-np.maximum(dt, 0) / 0.12), 0).astype(np.float32)
    mos = mosaic(a["cue_cell"], ids, N * N)
    mos = mos + flash[..., None] * (np.array(WHITE, np.float32) - mos) * 0.85
    img = up(mos, cell_px)
    e = edges(up(ids, cell_px))
    if 0 < f < 1:
        bar = f * P
    n_tok = len(np.unique(ids))
    return img, n_tok, bar, 1.0, e, flash


def reveal_stage(t):
    """tokens flip to the generated image in random order."""
    a = assets()
    cell_px = P // N
    ids = block_ids(a["img_blocks"][8][0], a["img_blocks"][8][1], N)
    rng = np.random.default_rng(3)
    order = rng.uniform(0, 0.55, N * N)
    t_rev = T_FLIP + 0.12 + order[ids]                     # one time per block
    mos_cue = mosaic(a["cue_cell"], ids, N * N)
    shown = t >= t_rev
    flip = np.clip((t - t_rev) / 0.08, 0, 1)
    img = up(mos_cue, cell_px)
    mask = up(shown, cell_px)
    img[mask] = a["gen"][mask]
    flash = np.where(shown, np.exp(-np.maximum(t - t_rev, 0) / 0.1), 0).astype(np.float32)
    img = img + up(flash, cell_px)[..., None] * (255 - img) * 0.6
    e = edges(up(ids, cell_px))
    line_a = 1 - clamp((t - (T_FLIP + 0.8)) / 0.5)
    return img, 1027, None, line_a, e


# ------------------------------------------------------------------ bonus round
def vid_index(t):
    """source frame (15 fps) given playback speed that follows the speedups; returns (idx, wrapped_recently)."""
    def V(tt):
        v = 0.0
        segs = [(T_B0, T_BSP1, VID_SPEED[0]), (T_BSP1, T_BSP2, VID_SPEED[1]), (T_BSP2, 99, VID_SPEED[2])]
        for s0, s1, sp in segs:
            if tt > s0:
                v += (min(tt, s1) - s0) * sp * 15
        return v
    v = V(t)
    idx = int(v) % 81
    wrapped = int(v) // 81 != int(V(t - 0.07)) // 81
    return idx, wrapped


def vid_speed(t):
    if t < T_BSP1:
        return VID_SPEED[0]
    return VID_SPEED[1] if t < T_BSP2 else VID_SPEED[2]


def vid_blocks(lv, cap):
    b = np.minimum(lv, cap).astype(int)
    ii, jj = np.mgrid[0:VN, 0:VM]
    return ii // b * b, jj // b * b, b


def bonus_panels(t):
    a = assets()
    idx, wrapped = vid_index(t)
    frame = a["vid"][idx]
    cell = a["vid_cell"][idx]
    lv = a["lv"][idx]
    left = frame.copy()
    right = frame * 0.35
    e = None
    bar = None
    flash = None
    tok = 0
    if t >= B1[0]:
        if t < B2[0]:
            f = sweep_frac(t, B1)
            rows = int(f * VN + 1e-6)
            ids = block_ids(*vid_blocks(lv, 1)[:2], VM)
            mos = up(mosaic(cell, ids, VN * VM), 11)
            ry = rows * 11
            right = right.copy()
            right[:ry] = mos[:ry]
            e = edges(up(ids, 11), 1)
            e[ry:] = False
            bar = f * VH if f < 1 else None
            tok = VID_TOK[0] * f
        else:
            if t < B3[0]:
                old, new, win, k = vid_blocks(lv, 1), vid_blocks(lv, 2), B2, 1
            else:
                old, new, win, k = vid_blocks(lv, 2), vid_blocks(lv, 4), B3, 2
            f = sweep_frac(t, win)
            ids, t_sw = staged_ids(old, new, VN, VM, f * VN, win[0], win[1] - win[0], t)
            dt = t - t_sw
            merged = new[2] > old[2]
            fl = np.where(merged & (dt >= 0), np.exp(-np.maximum(dt, 0) / 0.12), 0).astype(np.float32)
            mos = mosaic(cell, ids, VN * VM)
            mos = mos + fl[..., None] * (255 - mos) * 0.85
            right = up(mos, 11)
            e = edges(up(ids, 11), 1)
            bar = f * VH if 0 < f < 1 else None
            tok = VID_TOK[k - 1] + (VID_TOK[k] - VID_TOK[k - 1]) * f
    return left, right, e, bar, int(round(tok)) if t >= B1[0] else 0, wrapped


# ------------------------------------------------------------------ particles
def burst(seed, n, cx, cy, t0, t, speed=900, life=1.1, grav=1400):
    if t < t0 or t > t0 + life:
        return []
    rng = np.random.default_rng(seed)
    ang = rng.uniform(0, 2 * np.pi, n)
    sp = rng.uniform(0.3, 1.0, n) * speed
    sz = rng.choice([8, 12, 16], n)
    col = rng.integers(0, len(RAINBOW), n)
    dt = t - t0
    x = cx + np.cos(ang) * sp * dt
    y = cy + np.sin(ang) * sp * dt + 0.5 * grav * dt * dt
    return [(x[i], y[i], sz[i], RAINBOW[col[i]]) for i in range(n)]


def confetti(seed, n, t0, t, life=2.0):
    if t < t0 or t > t0 + life:
        return []
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(0, W, n)
    vy = rng.uniform(300, 700, n)
    delay = rng.uniform(0, 0.5, n)
    sz = rng.choice([8, 12, 16], n)
    col = rng.integers(0, len(RAINBOW), n)
    out = []
    for i in range(n):
        dt = t - t0 - delay[i]
        if dt < 0:
            continue
        out.append((x0[i] + 30 * math.sin(dt * 6 + i), -20 + vy[i] * dt, sz[i], RAINBOW[col[i]]))
    return out


def draw_particles(d, parts):
    for x, y, s, c in parts:
        if -20 < x < W + 20 and -20 < y < H + 20:
            d.rectangle([x - s / 2, y - s / 2, x + s / 2, y + s / 2], fill=c)


# ------------------------------------------------------------------ frame
def star_speed(t):
    if t < T_SP1:
        sp = 1.0
    elif t < T_SP2:
        sp = IMG_SPEED[1]
    elif t < T_B0:
        sp = IMG_SPEED[2]
    else:
        sp = vid_speed(t)
    return sp


def star_offset(t):
    # integral of 140 px/s × speed multiplier
    knots = [0, T_SP1, T_SP2, T_B0, T_BSP1, T_BSP2, 99]
    sps = [1.0, IMG_SPEED[1], IMG_SPEED[2], VID_SPEED[0], VID_SPEED[1], VID_SPEED[2]]
    v = 0.0
    for k in range(6):
        if t > knots[k]:
            v += (min(t, knots[k + 1]) - knots[k]) * sps[k]
    return 140 * v


def shake(t):
    for t0, amp in ((T_TITLE, 14), (T_SP1, 18), (T_SP2, 28), (T_TADA, 14), (T_BSP1, 18), (T_BSP2, 28)):
        dt = t - t0
        if 0 <= dt < 0.45:
            k = amp * math.exp(-dt / 0.12)
            return int(k * math.sin(dt * 90)), int(k * math.cos(dt * 71))
    return 0, 0


def speedup_popup(canvas, t, t0, val, big):
    dt = t - t0
    if dt < 0 or dt > SP_HOLD + 0.2:
        return
    out = clamp((dt - SP_HOLD) / 0.2)
    s = back_out(dt / 0.25) * (1 - out)
    band = Image.new("RGBA", (W, 280), (0, 0, 0, int(170 * (1 - out))))
    canvas.alpha_composite(band, (0, 540 - 140))
    wob = 1 + 0.04 * math.sin(dt * 25)
    put_text(canvas, "SPEEDUP", 80, rainbow("SPEEDUP", t), W // 2, 480, scale=s * wob)
    put_text(canvas, val + "!!", 128 if big else 112, rainbow(val + "!!", t, 14), W // 2, 600, scale=s * wob)


def hud(canvas, t, tokens, speed, stage):
    put_text(canvas, "TOKENS", 24, GREEN, 110, 44, anchor="l")
    put_text(canvas, f"{tokens:05d}", 48, YELLOW, 110, 80, anchor="l")
    put_text(canvas, "SPEED", 24, GREEN, W - 110, 44, anchor="r")
    sp_flash = any(0 <= t - t0 < 0.5 for t0 in (T_SP1, T_SP2, T_BSP1, T_BSP2))
    col = rainbow(f"{speed:.1f}X", t, 16) if sp_flash else YELLOW
    put_text(canvas, f"{speed:.1f}X", 48, col, W - 110, 80, anchor="r")
    put_text(canvas, stage, 24, GREEN, W // 2, 52)
    put_text(canvas, "LoT DIFFUSION", 16, GREEN_D, W // 2, 96, shadow=False)


def caption(canvas, s, t, t0, t1, color=YELLOW, y=1010, size=32):
    if t0 <= t < t1:
        k = int((t - t0) * 40)                              # typewriter, 40 chars/s
        put_text(canvas, s[:max(1, k)].ljust(len(s)), size, color, W // 2, y)


def colorbar(canvas, d, t, alpha):
    if alpha <= 0:
        return
    a = assets()
    x0, y0, bw, bh = 1460, 300, 36, 400
    bar = a["turbo"][::-1]                                  # top = near (high)
    col = cv2.resize(bar[:, None, :].astype(np.uint8), (bw, bh), interpolation=cv2.INTER_NEAREST)
    im = Image.fromarray(col).convert("RGBA")
    im.putalpha(int(255 * alpha))
    canvas.alpha_composite(im, (x0, y0))
    d.rectangle([x0 - 4, y0 - 4, x0 + bw + 3, y0 + bh + 3], outline=GREEN + (int(255 * alpha),), width=3)
    put_text(canvas, "NEAR", 16, GREEN, x0 + bw // 2, y0 - 26, alpha=alpha)
    put_text(canvas, "FAR", 16, GREEN, x0 + bw // 2, y0 + bh + 28, alpha=alpha)
    fy = y0 + int((255 - a["focus_v"]) / 255 * bh)
    d.polygon([(x0 + bw + 10, fy), (x0 + bw + 26, fy - 10), (x0 + bw + 26, fy + 10)], fill=GREEN + (int(255 * alpha),))
    if blink(t, 2.5) or t > S1[0]:
        put_text(canvas, "IN", 24, GREEN, x0 + bw + 36, fy - 30, anchor="l", alpha=alpha)
        put_text(canvas, "FOCUS", 24, GREEN, x0 + bw + 36, fy + 4, anchor="l", alpha=alpha)


def token_table(canvas, t, ids_sizes, alpha):
    """Space-Invaders style score table: token sizes and how many of each."""
    if alpha <= 0:
        return
    x0, y0 = 120, 330
    put_text(canvas, "TOKEN TABLE", 24, GREEN, x0, y0, anchor="l", alpha=alpha)
    for k, s in enumerate((1, 2, 4, 8)):
        y = y0 + 80 + k * 90
        d = ImageDraw.Draw(canvas)
        px = 6 * s if s < 8 else 44
        cx = x0 + 30
        d.rectangle([cx - px // 2, y - px // 2 + 10, cx + px // 2, y + px // 2 + 10], outline=GREEN + (int(255 * alpha),), width=3)
        put_text(canvas, f"{s}X{s}", 24, GREEN, x0 + 80, y, anchor="l", alpha=alpha)
        put_text(canvas, f"{ids_sizes.get(s, 0):4d}", 24, YELLOW, x0 + 210, y, anchor="l", alpha=alpha)


def size_counts(t):
    """number of tokens of each size currently on the playfield."""
    a = assets()
    if t < S1[0]:
        return {}
    if t < S2[0]:
        return {1: int(sweep_frac(t, S1) * N + 1e-6) * N}
    if t < S3[0]:
        old, new, win = a["img_blocks"][1], a["img_blocks"][2], S2
    else:
        old, new, win = a["img_blocks"][2], a["img_blocks"][8], S3
    f = sweep_frac(t, win)
    t_sw = win[0] + (new[0] + new[2]) / N * (win[1] - win[0])
    use_new = t >= t_sw
    b = np.where(use_new, new[2], old[2])
    out = {}
    for s in (1, 2, 4, 8):
        out[s] = int((b == s).sum()) // (s * s)
    return out


def render(t):
    a = assets()
    base = np.zeros((H, W, 3), np.float32)
    # stars
    sx, sy, sl = a["stars"]
    off = star_offset(t)
    yy = ((sy + off * sl * 0.6) % H).astype(int)
    for k in (1, 2, 3):
        m = sl == k
        s = k + 1
        v = 60 + 60 * k
        for x, y in zip(sx[m].astype(int), yy[m]):
            base[y:y + s * (1 + int(star_speed(t) > 1.5)), x:x + s] = v
    canvas = Image.fromarray(base.astype(np.uint8)).convert("RGBA")
    d = ImageDraw.Draw(canvas)
    parts = []

    # ---------------------------------------------- attract / title
    if t < T_POP:
        if t < T_COIN:
            if blink(t, 2.5):
                put_text(canvas, "INSERT COIN", 48, YELLOW, W // 2, 540)
        else:
            s = 1 + 2.0 * (1 - ease_out((t - T_TITLE) / 0.18)) if t >= T_TITLE else 0
            if t >= T_TITLE:
                put_text(canvas, "LEVEL-OF-TOKEN", 80, rainbow("LEVEL-OF-TOKEN", t, 8), W // 2, 420, scale=s)
                put_text(canvas, "DIFFUSION", 112, rainbow("DIFFUSION", t + 0.3, 8), W // 2, 560, scale=s)
            if t >= 1.0 and blink(t, 4):
                put_text(canvas, "PLAYER 1  READY!", 32, GREEN, W // 2, 720)
            parts += burst(11, 90, W // 2, 490, T_TITLE + 0.05, t, speed=1100)
    if t >= T_COIN and t < T_FALL + 1.2:
        put_text(canvas, "CREDIT 01", 16, GREEN, W - 110, H - 44, anchor="r", shadow=False)

    # ---------------------------------------------- stage 1: image
    if T_POP <= t < T_FALL:
        if t < T_FLIP:
            res = img_stage(t)
            if len(res) == 4:
                img, tok, bar, la = res
                e = None
            else:
                img, tok, bar, la, e, _ = res
        else:
            img, tok, bar, la, e = reveal_stage(t)
        if e is not None and la > 0:
            em = e[..., None].astype(np.float32) * la * 0.75
            img = img * (1 - em) + np.array(GREEN_D if t < T_FLIP else GREEN, np.float32) * em
        s = back_out((t - T_POP) / 0.35)
        pim = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
        frame_w = 6
        if s < 0.999:
            sz = max(2, int(P * s))
            pim = pim.resize((sz, sz), Image.NEAREST)
        cx, cy = PX0 + P // 2, PY0 + P // 2
        ox, oy = cx - pim.width // 2, cy - pim.height // 2
        d.rectangle([ox - frame_w - 4, oy - frame_w - 4, ox + pim.width + frame_w + 3, oy + pim.height + frame_w + 3],
                    outline=GREEN, width=frame_w)
        if ox >= 0 and oy >= 0:
            canvas.paste(pim, (ox, oy))
        else:
            paste_clip(canvas, pim.convert("RGBA"), ox, oy)
        if bar is not None:
            by = PY0 + int(bar)
            glow = Image.new("RGBA", (P + 40, 24), GREEN + (90,))
            canvas.alpha_composite(glow, (PX0 - 20, by - 12))
            d.rectangle([PX0 - 20, by - 3, PX0 + P + 19, by + 3], fill=WHITE)
            d.polygon([(PX0 - 40, by - 14), (PX0 - 20, by), (PX0 - 40, by + 14)], fill=YELLOW)
            d.polygon([(PX0 + P + 40, by - 14), (PX0 + P + 20, by), (PX0 + P + 40, by + 14)], fill=YELLOW)
        if t < S1[0]:
            tok_show = 0
        else:
            tok_show = tok
        speed = 1.0 if t < T_SP1 else (IMG_SPEED[1] if t < T_SP2 else IMG_SPEED[2])
        hud(canvas, t, tok_show, speed, "STAGE 1")
        colorbar(canvas, d, t, clamp((t - T_POP - 0.2) / 0.2) * (1 - clamp((t - T_FLIP) / 0.3)))
        sc = size_counts(t) if t < T_FLIP else {1: 776, 2: 174, 4: 48, 8: 29}
        token_table(canvas, t, sc, clamp((t - S1[0]) / 0.2))
        caption(canvas, "STAGE 1: DEPTH MAP", t, T_POP + 0.1, S1[0])
        caption(canvas, "EVERY PATCH = 1 TOKEN!", t, S1[0], S2[0])
        caption(canvas, "OUT OF FOCUS = BIG TOKENS", t, S2[0], T_SP1)
        caption(canvas, "BIGGER! BIGGER!", t, S3[0], T_SP2)
        speedup_popup(canvas, t, T_SP1, "1.9X", False)
        speedup_popup(canvas, t, T_SP2, "2.4X", True)
        parts += burst(21, 140, W // 2, 560, T_SP1, t)
        parts += burst(22, 220, W // 2, 560, T_SP2, t, speed=1300)
        if t >= T_FLIP and t < T_FLIP + 0.12:
            canvas.alpha_composite(Image.new("RGBA", (W, H), (255, 255, 255, int(255 * (1 - (t - T_FLIP) / 0.12)))))
        if t >= T_TADA:
            dt = t - T_TADA
            # pops big over the image, then slides down to the caption band so the image is clear
            s = back_out(dt / 0.3)
            m = ease_out((dt - 0.7) / 0.3)
            wob = 1 + 0.05 * math.sin(dt * 18) * (1 - m)
            y = 540 + (1005 - 540) * m
            put_text(canvas, "TA-DA!!!", 144, rainbow("TA-DA!!!", t, 12), W // 2, int(y), scale=s * wob * (1 - 0.55 * m))
            if dt > 0.9:
                k = back_out((dt - 0.9) / 0.25)
                for j, line in enumerate(("4X FEWER", "TOKENS", "", "IN FOCUS", "STAYS", "SHARP")):
                    if line:
                        put_text(canvas, line, 32, YELLOW if j < 2 else GREEN, 1600, 400 + j * 56, scale=k)
            parts += confetti(31, 260, T_TADA, t)
            parts += burst(32, 160, W // 2, 470, T_TADA, t, speed=1200)

    # ---------------------------------------------- transition: tokens fall
    if T_FALL <= t < T_B0:
        dt = t - T_FALL
        rng = np.random.default_rng(5)
        gc = a["gen_cell"]
        if dt < 0.9:
            delays = rng.uniform(0, 0.35, len(a["rects"]))
            vx = rng.uniform(-200, 200, len(a["rects"]))
            sc = P / N
            for k, (r0, r1, c0, c1) in enumerate(a["rects"]):
                tt = max(0.0, dt - delays[k])
                y = PY0 + r0 * sc + 0.5 * 3000 * tt * tt
                x = PX0 + c0 * sc + vx[k] * tt
                if y > H:
                    continue
                col = tuple(int(v) for v in gc[r0:r1, c0:c1].reshape(-1, 3).mean(0))
                s = (r1 - r0) * sc
                d.rectangle([x, y, x + s - 1, y + s - 1], fill=col, outline=GREEN)
        if t >= T_BONUS:
            dt2 = t - T_BONUS
            s = back_out(dt2 / 0.25)
            if blink(t, 6) or dt2 < 0.3:
                put_text(canvas, "BONUS", 144, rainbow("BONUS", t, 14), W // 2, 450, scale=s)
                put_text(canvas, "ROUND!!", 144, rainbow("ROUND!!", t + 0.2, 14), W // 2, 630, scale=s)
        hud(canvas, t, 0, 1.0, "BONUS")

    # ---------------------------------------------- bonus round: video
    if t >= T_B0:
        left, right, e, bar, tok, wrapped = bonus_panels(t)
        if e is not None:
            em = e[..., None].astype(np.float32) * 0.6
            right = right * (1 - em) + np.array(GREEN_D, np.float32) * em
        s = back_out((t - T_B0) / 0.3)
        for k, (x0, im) in enumerate(((VX0L, left), (VX0R, right))):
            if wrapped:
                rng = np.random.default_rng(int(t * 1000) + k)
                im = im * 0.5 + rng.uniform(0, 255, im.shape).astype(np.float32) * 0.5
            pim = Image.fromarray(np.clip(im, 0, 255).astype(np.uint8))
            if s < 0.999:
                pim = pim.resize((max(2, int(VW * s)), max(2, int(VH * s))), Image.NEAREST)
            cx, cy = x0 + VW // 2, VY0 + VH // 2
            ox, oy = cx - pim.width // 2, cy - pim.height // 2
            d.rectangle([ox - 10, oy - 10, ox + pim.width + 9, oy + pim.height + 9], outline=GREEN, width=6)
            canvas.paste(pim, (ox, oy))
        if bar is not None:
            by = VY0 + int(bar)
            glow = Image.new("RGBA", (VW + 40, 20), GREEN + (90,))
            canvas.alpha_composite(glow, (VX0R - 20, by - 10))
            d.rectangle([VX0R - 20, by - 3, VX0R + VW + 19, by + 3], fill=WHITE)
            d.polygon([(VX0R + VW + 40, by - 14), (VX0R + VW + 20, by), (VX0R + VW + 40, by + 14)], fill=YELLOW)
        sp = vid_speed(t)
        hud(canvas, t, tok, sp if t >= B1[0] else 1.0, "BONUS ROUND")
        put_text(canvas, "VIDEO", 32, GREEN, VX0L, VY0 - 56, anchor="l")
        put_text(canvas, "TOKENS", 32, GREEN, VX0R, VY0 - 56, anchor="l")
        # playback badge: one triangle per speed level
        n_tri = 1 if sp < 1.5 else (2 if sp < 2.3 else 3)
        bx = VX0L + VW - 240
        by = VY0 - 58
        pulse = 0 <= t - T_BSP1 < 0.5 or 0 <= t - T_BSP2 < 0.5
        for k in range(n_tri):
            d.polygon([(bx + k * 26, by), (bx + k * 26 + 24, by + 14), (bx + k * 26, by + 28)],
                      fill=MAGENTA if pulse and blink(t, 6) else YELLOW)
        put_text(canvas, f"{sp:.1f}X", 32, YELLOW, VX0L + VW, VY0 - 56, anchor="r")
        caption(canvas, "EVERY FRAME, EVERY PATCH!", t, B1[0], B2[0], y=900)
        caption(canvas, "MERGE THE GROUND!", t, B2[0], T_BSP1, y=900)
        caption(canvas, "MERGE THE SKY!", t, B3[0], T_BSP2, y=900)
        speedup_popup(canvas, t, T_BSP1, "2.2X", False)
        speedup_popup(canvas, t, T_BSP2, "2.5X", True)
        parts += burst(41, 140, W // 2, 560, T_BSP1, t)
        parts += burst(42, 220, W // 2, 560, T_BSP2, t, speed=1300)
        if t >= T_END:
            dt = t - T_END
            band = Image.new("RGBA", (W, 300), (0, 0, 0, int(200 * clamp(dt / 0.2))))
            canvas.alpha_composite(band, (0, 390))
            sc = back_out(dt / 0.3)
            put_text(canvas, "LEVEL-OF-TOKEN DIFFUSION", 64, rainbow("LEVEL-OF-TOKEN DIFFUSION", t, 8), W // 2, 490, scale=sc)
            if dt > 0.5:
                put_text(canvas, "IMAGES 2.4X FASTER   VIDEO 2.5X FASTER", 32, YELLOW, W // 2, 600)
            if dt > 1.0 and blink(t, 2.5):
                put_text(canvas, "INSERT COIN", 32, GREEN, W // 2, 1010)
            parts += confetti(51, 200, T_END, t, life=2.5)

    draw_particles(d, parts)

    # ---------------------------------------------- CRT post
    f = np.asarray(canvas.convert("RGB")).astype(np.float32)
    dx, dy = shake(t)
    if dx or dy:
        f = np.roll(f, (dy, dx), (0, 1))
    small = cv2.resize(f, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
    bright = np.clip(small - 120, 0, 255)
    glow = cv2.GaussianBlur(bright, (0, 0), 6)
    f = f + 0.55 * cv2.resize(glow, (W, H), interpolation=cv2.INTER_LINEAR)
    f[..., 0] = np.roll(f[..., 0], 2, 1)
    f[..., 2] = np.roll(f[..., 2], -2, 1)
    f = f * a["scan"] * a["vignette"]
    return np.clip(f, 0, 255).astype(np.uint8)


# ------------------------------------------------------------------ audio (chiptune)
SR = 44100


def sq(freq, dur, duty=0.5, vol=0.3, decay=None):
    n = int(dur * SR)
    tt = np.arange(n) / SR
    if callable(freq):
        ph = np.cumsum(freq(tt)) / SR
    else:
        ph = freq * tt
    w = np.where((ph % 1) < duty, 1.0, -1.0)
    env = np.exp(-tt / decay) if decay else np.ones(n)
    env[:40] *= np.linspace(0, 1, 40)
    env[-80:] *= np.linspace(1, 0, 80)
    return w * env * vol


def tri(freq, dur, vol=0.35, decay=None):
    n = int(dur * SR)
    tt = np.arange(n) / SR
    ph = freq * tt
    w = 4 * np.abs((ph % 1) - 0.5) - 1
    env = np.exp(-tt / decay) if decay else np.ones(n)
    env[-80:] *= np.linspace(1, 0, 80)
    return w * env * vol


def noise(dur, vol=0.3, decay=0.15, seed=0):
    n = int(dur * SR)
    rng = np.random.default_rng(seed)
    w = np.repeat(rng.uniform(-1, 1, n // 6 + 1), 6)[:n]       # crunchy 8-bit noise
    return w * np.exp(-np.arange(n) / SR / decay) * vol


def note(name):
    names = {"C": -9, "C#": -8, "D": -7, "D#": -6, "E": -5, "F": -4, "F#": -3, "G": -2, "G#": -1, "A": 0, "A#": 1, "B": 2}
    return 440 * 2 ** ((names[name[:-1]] + 12 * (int(name[-1]) - 4)) / 12)


def build_audio(path):
    out = np.zeros(int((DUR + 0.5) * SR))

    def add(sig, t0):
        i = int(t0 * SR)
        j = min(len(out), i + len(sig))
        out[i:j] += sig[:j - i]

    # coin
    add(sq(note("B5"), 0.07, vol=0.25), T_COIN)
    add(sq(note("E6"), 0.4, vol=0.25, decay=0.15), T_COIN + 0.07)
    # title slam
    add(noise(0.5, 0.45, 0.12, 1), T_TITLE)
    add(sq(lambda tt: 220 * np.exp(-tt * 6), 0.4, vol=0.3), T_TITLE)
    # pop
    add(sq(lambda tt: 300 + 4000 * tt, 0.15, duty=0.25, vol=0.25), T_POP)
    # sweep 1: blip every 2 rows, pitch climbing
    for k in range(32):
        tk = S1[0] + (k + 0.5) / 32 * (S1[1] - S1[0])
        add(sq(note("C5") * 2 ** (k / 16), 0.035, duty=0.125, vol=0.13), tk)
    # merge sweeps: descending bloops
    for win, base in ((S2, "G5"), (S3, "C5"), (B2, "G5"), (B3, "C5")):
        n = 16
        for k in range(n):
            tk = win[0] + (k + 0.5) / n * (win[1] - win[0])
            add(tri(note(base) * 2 ** (-k / 12), 0.06, vol=0.3, decay=0.04), tk)
    for win in (B1,):
        for k in range(24):
            tk = win[0] + (k + 0.5) / 24 * (win[1] - win[0])
            add(sq(note("C5") * 2 ** (k / 12), 0.03, duty=0.125, vol=0.12), tk)
    # speedup: rising arpeggio + boom
    arp = ["C5", "E5", "G5", "C6", "E6", "G6", "C7"]
    for t0, big in ((T_SP1, False), (T_SP2, True), (T_BSP1, False), (T_BSP2, True)):
        add(noise(0.6, 0.5 if big else 0.35, 0.18, int(t0 * 10)), t0)
        for k, nm in enumerate(arp + (["E7", "G7"] if big else [])):
            add(sq(note(nm), 0.06, duty=0.25, vol=0.2), t0 + 0.05 + k * 0.045)
        add(sq(note("C7"), 0.35, duty=0.25, vol=0.16, decay=0.12), t0 + 0.05 + len(arp) * 0.045)
    # flip sparkle
    for k in range(14):
        add(sq(note("C6") * 2 ** ((k % 7) / 7), 0.03, duty=0.125, vol=0.1), T_FLIP + 0.12 + k * 0.04)
    # ta-da fanfare
    for k, nm in enumerate(["G4", "C5", "E5"]):
        add(sq(note(nm), 0.1, vol=0.22), T_TADA - 0.32 + k * 0.1)
    for nm in ("C5", "E5", "G5", "C6"):
        vib = lambda tt, f=note(nm): f * (1 + 0.012 * np.sin(2 * np.pi * 6 * tt))
        add(sq(vib, 0.9, vol=0.12, decay=0.6), T_TADA)
    # bonus jingle
    for k in range(12):
        add(sq(note(["C6", "G5", "E6", "C6"][k % 4]), 0.05, duty=0.25, vol=0.18), T_BONUS + k * 0.055)
    # loop glitch
    tt = np.arange(T_B0, DUR, 1 / 60)
    prev = False
    for t in tt:
        w = vid_index(t)[1]
        if w and not prev:
            add(noise(0.08, 0.15, 0.05, int(t * 100)), t)
        prev = w
    # end jingle
    for k, nm in enumerate(["C5", "E5", "G5", "C6", "G5", "C6"]):
        add(sq(note(nm), 0.12 if k < 5 else 0.6, vol=0.2, decay=None if k < 5 else 0.3), T_END + k * 0.12)
    # space-invaders march, tempo × current speed
    march = ["A2", "G2", "F2", "E2"]
    for seg in ((T_POP, T_FALL), (T_B0, T_END)):
        t, k = seg[0], 0
        while t < seg[1]:
            sp = star_speed(t)
            add(tri(note(march[k % 4]), 0.12, vol=0.4, decay=0.08), t)
            add(sq(note(march[k % 4]), 0.08, duty=0.5, vol=0.06, decay=0.05), t)
            t += 0.42 / sp
            k += 1
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
    wav = os.path.join(HERE, f"media/final/LoT_arcade_{TAG}.wav")
    build_audio(wav)
    out = os.path.join(HERE, f"media/final/LoT_arcade_{TAG}_{res}.mp4")
    nf = int(DUR * fps)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps),
           "-i", "-", "-i", wav, "-c:v", "libx264", "-preset", "slow", "-crf", "16", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-ac", "2", "-ar", "48000", "-b:a", "192k", "-shortest", "-movflags", "+faststart", out]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    with Pool(48) as pool:
        for k, buf in enumerate(pool.imap(render_scaled, [(i / fps, w, h) for i in range(nf)], chunksize=2)):
            p.stdin.write(buf)
            if k % 120 == 0:
                print(f"frame {k}/{nf}", flush=True)
    p.stdin.close()
    p.wait()
    print("wrote", out)


if __name__ == "__main__":
    main()
