"""LoT Diffusion, 20-s Mondrian / De Stijl clip, v1 (numpy + PIL + cv2 frames → ffmpeg, synthesized marimba). No manim.

The token layout is a Mondrian painting: black bars = token borders, colour = token size
(1×1 white, 2×2 yellow, 4×4 blue, 8×8 red).
  0.0–1.6  title composition
  1.6–3.0  depth map (inset): what is in focus
  3.0–4.4  black lattice draws in: every patch is a token (4,096)
  4.6–6.0  roller: out-of-focus tokens merge to 2×2, painted yellow (1,606) → 1.9× faster
  7.0–8.3  roller: merge to 4×4 / 8×8, blue / red (1,027) → 2.4× faster
  9.0–10.4 blocks are painted over with the generated image
 10.9–11.7 black-bar wipe to video
 11.8–17.2 platformer video left, its living Mondrian layout right; 2×2 → 2.2×, 4×4 → 2.5×, playback speeds up
 17.4–20  title card
Speedups as in arcade_clip (paper timing fit / Tab. 2 interpolation).
Usage: python render_mondrian_v1.py h|l   → media/final/LoT_mondrian_v1_<res>.mp4
       python render_mondrian_v1.py still t ...  → scratch/still_v1_<t>.png
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
FONT = os.path.join(HERE, "data/Jost.ttf")
CUE = os.path.join(ROOT, "data/img/img_000000000025_dof_cue.png")
GEN = os.path.join(ROOT, "data/img/img_000000000025_dof_generated_4b.png")
LAYOUT = os.path.join(ROOT, "data/layouts/img_000000000025_dof.json")
VID = os.path.join(ROOT, "data/clips/vd_platformer_gen")
LEVELS = os.path.join(ROOT, "arcade_clip/data/platformer_levels.npy")

W, H = 1920, 1080
DUR = 20.0

CANVAS = (242, 238, 227)
WHITE = (250, 248, 242)
INK = (20, 20, 20)
GREY = (120, 116, 108)
RED = (214, 39, 24)
YELLOW = (247, 199, 0)
BLUE = (31, 63, 153)
SIZE_COL = {1: WHITE, 2: YELLOW, 4: BLUE, 8: RED}

# ------------------------------------------------------------------ timeline (s)
T_CUE = 1.6
L0 = (3.0, 4.4)              # lattice draws in
S2 = (4.6, 6.0)              # roller → 2×2
S3 = (7.0, 8.3)              # roller → 4×4 / 8×8
T_SP1, T_SP2 = 6.1, 8.4
R0 = (9.0, 10.4)             # image painted in
T_WIPE = (10.9, 11.7)
VL0 = (11.9, 12.6)           # video lattice
B2 = (12.9, 13.9)
B3 = (15.0, 16.0)
T_BSP1, T_BSP2 = 14.0, 16.1
T_END = 17.4
FILL = 0.28                  # brush time per block

IMG_TOK = (4096, 1606, 1027)
VID_TOK = (75600, 38046, 34698)
IMG_SPEED = (1.0, 1.9, 2.4)
VID_SPEED = (1.0, 2.2, 2.5)

# geometry
P = 768
PX0, PY0 = 170, 150
N = 64
CP = P // N                   # 12 px per token
VW, VH = 880, 495
VX0L, VX0R, VY0 = 60, 980, 250
VN, VM = 45, 80
VC = 11
COL_X = 1080                  # right column (image part)


def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def ease(x):
    x = clamp(x)
    return x * x * (3 - 2 * x)


def ease_out(x):
    x = clamp(x)
    return 1 - (1 - x) ** 3


# ------------------------------------------------------------------ assets
def load_assets():
    a = {}
    cue = np.asarray(Image.open(CUE).convert("RGB"))
    a["cue_small"] = Image.fromarray(cue).resize((300, 300), Image.LANCZOS)
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
    a["lv"] = np.load(LEVELS)
    a["vid"] = [cv2.resize(np.asarray(Image.open(os.path.join(VID, f"{f + 1:04d}.jpg")).convert("RGB")),
                           (VW, VH), interpolation=cv2.INTER_AREA) for f in range(81)]
    rng = np.random.default_rng(1)
    a["paper"] = (rng.normal(0, 3.0, (H, W, 1))).astype(np.float32)
    a["rev_order"] = rng.uniform(0, 1, (N, N))
    a["row_jit"] = rng.uniform(-0.06, 0.06, 2048)
    lut = cv2.applyColorMap(np.arange(256, dtype=np.uint8)[:, None], cv2.COLORMAP_TURBO)[:, 0, ::-1].astype(int)
    a["focus_v"] = int(np.argmin(((lut - np.array([44, 208, 165])) ** 2).sum(1)))
    return a


A = None


def assets():
    global A
    if A is None:
        A = load_assets()
    return A


# ------------------------------------------------------------------ text
@lru_cache(maxsize=64)
def font(size, weight="Bold"):
    f = ImageFont.truetype(FONT, size)
    f.set_variation_by_name(weight)
    return f


def text(d, s, size, xy, fill=INK, weight="Bold", anchor="la"):
    d.text(xy, s, font=font(size, weight), fill=fill, anchor=anchor)


# ------------------------------------------------------------------ painting core
def up(arr, c):
    return np.repeat(np.repeat(arr, c, 0), c, 1)


def edges(ids_px, thick):
    e = np.zeros(ids_px.shape, bool)
    e[:, 1:] |= ids_px[:, 1:] != ids_px[:, :-1]
    e[1:, :] |= ids_px[1:, :] != ids_px[:-1, :]
    k = thick - 1
    out = e.copy()
    for s in range(1, k + 1):
        out[:, :-s] |= e[:, s:]
        out[:-s, :] |= e[s:, :]
    out[:thick, :] = out[-thick:, :] = True
    out[:, :thick] = out[:, -thick:] = True
    return out


def palette_img(size_map):
    lut = np.zeros((9, 3), np.float32)
    for s, c in SIZE_COL.items():
        lut[s] = c
    return lut[size_map]


def paint(blocks_old, blocks_new, n, m, c, win, t, row_jit):
    """Mondrian state during a roller sweep from blocks_old to blocks_new.
    returns (rgb float n*c × m*c, ids at token res, new-size map at token res, roller y px or None)."""
    oi_o, oj_o, b_o = blocks_old
    oi_n, oj_n, b_n = blocks_new
    dur = win[1] - win[0]
    t_sw = win[0] + (oi_n + b_n) / n * dur
    switched = t >= t_sw
    oi = np.where(switched, oi_n, oi_o)
    oj = np.where(switched, oj_n, oj_o)
    b = np.where(switched, b_n, b_o)
    ids = oi * m + oj
    p = np.clip((t - t_sw) / FILL, 0, 1)
    p = np.where(b_n > b_o, p, 1.0)
    # pixel-level brush: block-local x fraction vs fill progress (+ ragged edge per row)
    hh, ww = n * c, m * c
    xx = np.arange(ww)[None, :].astype(np.float32)
    oj_px = up(oj, c).astype(np.float32) * c
    bw_px = up(b, c).astype(np.float32) * c
    frac = (xx - oj_px) / bw_px + row_jit[:hh, None]
    painted = frac < up(p, c)
    new_rgb = up(palette_img(b), c)
    old_rgb = up(palette_img(b_o), c)
    rgb = np.where(painted[..., None], new_rgb, old_rgb)
    roller = t_sw.min() if False else None
    f = clamp((t - win[0]) / dur)
    if 0 < f < 1:
        roller = f * hh
    return rgb, ids, b, roller


def draw_bars(rgb, ids, c, thick, alpha=1.0):
    e = edges(up(ids, c), thick)
    if alpha >= 1:
        rgb[e] = INK
    else:
        em = e[..., None].astype(np.float32) * alpha
        rgb[:] = rgb * (1 - em) + np.array(INK, np.float32) * em
    return rgb


def img_painting(t):
    """returns (rgb P×P, tokens, roller px, size map)."""
    a = assets()
    B = a["img_blocks"]
    jit = a["row_jit"]
    if t < L0[0]:
        return np.full((P, P, 3), WHITE, np.float32), 0, None, None
    if t < S2[0]:
        f = clamp((t - L0[0]) / (L0[1] - L0[0]))
        rgb = np.full((P, P, 3), WHITE, np.float32)
        ids = B[1][0] * N + B[1][1]
        e = edges(up(ids, CP), 3)
        yy, xx = np.mgrid[0:P, 0:P]
        vis = (xx + yy) / (2 * P) < f * 1.05
        rgb[e & vis] = INK
        rgb[:3], rgb[-3:], rgb[:, :3], rgb[:, -3:] = INK, INK, INK, INK
        # token count = cells whose top-left corner the wave has passed
        ii, jj = np.mgrid[0:N, 0:N]
        tok = int((((jj + ii + 2) * CP) / (2 * P) < f * 1.05).sum())
        return rgb, min(tok, 4096), None, np.ones((N, N), int)
    if t < S3[0]:
        rgb, ids, b, roller = paint(B[1], B[2], N, N, CP, S2, t, jit)
    else:
        rgb, ids, b, roller = paint(B[2], B[8], N, N, CP, S3, t, jit)
    rgb = draw_bars(rgb, ids, CP, 3)
    return rgb, len(np.unique(ids)), roller, b


def img_reveal(t):
    """blocks painted over with the generated image, largest first, then the bars fade."""
    a = assets()
    oi, oj, b = a["img_blocks"][8]
    ids = oi * N + oj
    order = a["rev_order"][oi, oj]
    t_b = R0[0] + (1 - np.log2(b) / 3) * 0.55 + order * 0.35
    p = np.clip((t - t_b) / FILL, 0, 1)
    xx = np.arange(P)[None, :].astype(np.float32)
    frac = (xx - up(oj, CP) * CP) / (up(b, CP) * CP) + a["row_jit"][:P, None]
    painted = frac < up(p, CP)
    rgb = np.where(painted[..., None], a["gen"], up(palette_img(b), CP))
    bar_a = 1 - clamp((t - (R0[1] - 0.3)) / 0.5)
    if bar_a > 0:
        rgb = draw_bars(rgb.copy(), ids, CP, 3, bar_a)
    return rgb


# ------------------------------------------------------------------ video
def vid_V(tt):
    v = 0.0
    segs = [(VL0[0], T_BSP1, VID_SPEED[0]), (T_BSP1, T_BSP2, VID_SPEED[1]), (T_BSP2, 99, VID_SPEED[2])]
    for s0, s1, sp in segs:
        if tt > s0:
            v += (min(tt, s1) - s0) * sp * 15
    return v


def vid_speed(t):
    if t < T_BSP1:
        return VID_SPEED[0]
    return VID_SPEED[1] if t < T_BSP2 else VID_SPEED[2]


def vid_blocks(lv, cap):
    b = np.minimum(lv, cap).astype(int)
    ii, jj = np.mgrid[0:VN, 0:VM]
    return ii // b * b, jj // b * b, b


def vid_painting(t, idx):
    a = assets()
    lv = a["lv"][idx]
    jit = a["row_jit"]
    if t < VL0[1]:
        f = clamp((t - VL0[0]) / (VL0[1] - VL0[0]))
        rgb = np.full((VH, VW, 3), WHITE, np.float32)
        ii, jj = np.mgrid[0:VN, 0:VM]
        ids = ii * VM + jj
        e = edges(up(ids, VC), 2)
        yy, xx = np.mgrid[0:VH, 0:VW]
        vis = (xx / VW + yy / VH) / 2 < f * 1.05
        rgb[e & vis] = INK
        return rgb, VID_TOK[0] * f, None
    if t < B3[0]:
        rgb, ids, b, roller = paint(vid_blocks(lv, 1), vid_blocks(lv, 2), VN, VM, VC, B2, t, jit)
        f = clamp((t - B2[0]) / (B2[1] - B2[0]))
        tok = VID_TOK[0] + (VID_TOK[1] - VID_TOK[0]) * f
    else:
        rgb, ids, b, roller = paint(vid_blocks(lv, 2), vid_blocks(lv, 4), VN, VM, VC, B3, t, jit)
        f = clamp((t - B3[0]) / (B3[1] - B3[0]))
        tok = VID_TOK[1] + (VID_TOK[2] - VID_TOK[1]) * f
    rgb = draw_bars(rgb, ids, VC, 2)
    return rgb, tok, roller


# ------------------------------------------------------------------ frame pieces
def roller_draw(d, x0, y0, w, ry, color):
    y = y0 + int(ry)
    d.rectangle([x0 - 14, y - 9, x0 + w + 13, y + 9], fill=color, outline=INK, width=3)
    d.rectangle([x0 + w + 13, y - 3, x0 + w + 60, y + 3], fill=INK)          # handle
    d.rectangle([x0 + w + 54, y - 3, x0 + w + 60, y + 60], fill=INK)
    d.rectangle([x0 + w + 48, y + 60, x0 + w + 66, y + 110], fill=GREY)


def speed_bar(d, x, y, speed, label, t_pop, t, max_w=640):
    """Mondrian bar: length ∝ speed; red once a speedup lands."""
    w0 = 230
    w = w0 * speed
    if t_pop is not None and t >= t_pop:
        k = ease_out((t - t_pop) / 0.35)
    else:
        k = 1.0
    col = INK if speed <= 1.0 else RED
    d.rectangle([x, y, x + int(w * k), y + 90], fill=col)
    d.rectangle([x, y, x + int(w * k), y + 90], outline=INK, width=6)
    text(d, label, 56, (x + int(w * k) - 22, y + 45), fill=WHITE, anchor="rm")


def legend(d, x, y, sizes=(1, 2, 4, 8)):
    for k, s in enumerate(sizes):
        xx = x + k * 150
        d.rectangle([xx, y, xx + 54, y + 54], fill=SIZE_COL[s], outline=INK, width=5)
        text(d, f"{s}×{s}", 30, (xx + 66, y + 27), anchor="lm", weight="SemiBold")


def caption(d, s, x, y, t, t0, t1, size=46, anchor="la"):
    if t0 <= t < t1:
        a = clamp((t - t0) / 0.2) * (1 - clamp((t - (t1 - 0.2)) / 0.2))
        col = tuple(int(CANVAS[i] + (INK[i] - CANVAS[i]) * a) for i in range(3))
        text(d, s, size, (x, y), fill=col, anchor=anchor, weight="Medium")


def img_scene(canvas, d, t):
    a = assets()
    # painting
    if t < R0[0]:
        rgb, tok, roller, b = img_painting(t)
    else:
        rgb = img_reveal(t)
        tok, roller = IMG_TOK[2], None
    s = ease_out((t - T_CUE + 0.6) / 0.5)
    if s > 0:
        pim = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))
        if s < 0.999:
            pim = pim.resize((max(2, int(P * s)), P), Image.BILINEAR)
        d.rectangle([PX0 - 8, PY0 - 8, PX0 + pim.width + 7, PY0 + P + 7], fill=INK)
        canvas.paste(pim, (PX0, PY0))
    if roller is not None:
        roller_draw(d, PX0, PY0, P, roller, YELLOW if t < S3[0] else BLUE)
    # right column: depth cue
    ca = clamp((t - T_CUE) / 0.4)
    if ca > 0:
        cx, cy = COL_X, PY0
        d.rectangle([cx - 6, cy - 6, cx + 305, cy + 305], fill=INK)
        cue = a["cue_small"]
        if ca < 1:
            cue = Image.blend(Image.new("RGB", cue.size, CANVAS), cue, ca)
        canvas.paste(cue, (cx, cy))
        text(d, "depth map", 34, (cx + 330, cy + 10), weight="SemiBold")
        text(d, "near = red, far = blue", 26, (cx + 330, cy + 58), fill=GREY, weight="Regular")
        text(d, "in focus = green", 26, (cx + 330, cy + 96), fill=GREY, weight="Regular")
        # in-focus band swatch
        d.rectangle([cx + 330, cy + 140, cx + 380, cy + 170], fill=(44, 208, 165), outline=INK, width=3)
        text(d, "gets the finest tokens", 26, (cx + 392, cy + 155), fill=INK, anchor="lm", weight="Regular")
    # counters
    if t >= L0[0]:
        text(d, "tokens", 30, (COL_X, 520), fill=GREY, weight="Medium")
        text(d, f"{tok:,}", 104, (COL_X, 552), weight="Bold")
        text(d, "speed", 30, (COL_X, 700), fill=GREY, weight="Medium")
        if t < T_SP1:
            speed_bar(d, COL_X, 735, 1.0, "1.0×", None, t)
        elif t < T_SP2:
            speed_bar(d, COL_X, 735, IMG_SPEED[1], "1.9× faster", T_SP1, t)
        else:
            speed_bar(d, COL_X, 735, IMG_SPEED[2], "2.4× faster", T_SP2, t)
        legend(d, COL_X, 880)
    caption(d, "Level-of-Token Diffusion", PX0, 960, t, T_CUE, L0[0], size=46)
    caption(d, "Every patch is a token", PX0, 960, t, L0[0], S2[0] - 0.05)
    caption(d, "Out of focus: merge into big tokens", PX0, 960, t, S2[0], S3[0] - 0.05)
    caption(d, "Bigger still where nothing happens", PX0, 960, t, S3[0], R0[0] - 0.05)
    caption(d, "Fine tokens keep the giraffe sharp", PX0, 960, t, R0[0], T_WIPE[0] + 0.3)


def vid_scene(canvas, d, t):
    a = assets()
    v = vid_V(t)
    idx = int(v) % 81
    left = a["vid"][idx]
    rgb, tok, roller = vid_painting(t, idx)
    for x0, im in ((VX0L, left), (VX0R, np.clip(rgb, 0, 255).astype(np.uint8))):
        d.rectangle([x0 - 8, VY0 - 8, x0 + VW + 7, VY0 + VH + 7], fill=INK)
        canvas.paste(Image.fromarray(im), (x0, VY0))
    if roller is not None:
        y = VY0 + int(roller)
        col = YELLOW if t < B3[0] else BLUE
        d.rectangle([VX0R - 14, y - 8, VX0R + VW + 13, y + 8], fill=col, outline=INK, width=3)
    sp = vid_speed(t)
    text(d, "video", 36, (VX0L, VY0 - 62), weight="SemiBold")
    text(d, "its token layout", 36, (VX0R, VY0 - 62), weight="SemiBold")
    # playback badge
    bx = VX0L + VW
    text(d, f"playing {sp:.1f}×", 36, (bx, VY0 - 62), anchor="ra", fill=RED if sp > 1 else INK, weight="SemiBold")
    text(d, "tokens", 30, (VX0L, 800), fill=GREY, weight="Medium")
    text(d, f"{int(round(tok)):,}", 84, (VX0L, 830), weight="Bold")
    text(d, "speed", 30, (VX0R, 800), fill=GREY, weight="Medium")
    if t < T_BSP1:
        speed_bar(d, VX0R, 840, 1.0, "1.0×", None, t)
    elif t < T_BSP2:
        speed_bar(d, VX0R, 840, VID_SPEED[1], "2.2× faster", T_BSP1, t)
    else:
        speed_bar(d, VX0R, 840, VID_SPEED[2], "2.5× faster", T_BSP2, t)
    legend(d, VX0L + 400, 838, (1, 2, 4))
    caption(d, "Now a video: every patch, every frame", VX0L, 975, t, T_WIPE[1], B2[0] - 0.05, size=42)
    caption(d, "Ground merges into 2×2", VX0L, 975, t, B2[0], B3[0] - 0.05, size=42)
    caption(d, "Sky merges into 4×4", VX0L, 975, t, B3[0], T_END, size=42)


def title_scene(canvas, d, t, alpha=1.0):
    """opening composition: bars draw in, blocks pop, title."""
    bars = [  # (x0, y0, x1, y1, t0) horizontal/vertical black bars
        (0, 300, W, 316, 0.05), (1380, 0, 1396, H, 0.15), (380, 316, 396, H, 0.25),
        (396, 820, 1380, 836, 0.35), (1396, 640, W, 656, 0.42), (900, 836, 916, H, 0.5)]
    for x0, y0, x1, y1, t0 in bars:
        k = ease_out((t - t0) / 0.35)
        if k <= 0:
            continue
        if x1 - x0 > y1 - y0:
            d.rectangle([x0, y0, x0 + (x1 - x0) * k, y1], fill=INK)
        else:
            d.rectangle([x0, y0, x1, y0 + (y1 - y0) * k], fill=INK)
    blocks = [((0, 0, 1380, 300), RED, 0.55), ((1396, 656, W, H), BLUE, 0.7), ((0, 316, 380, H), YELLOW, 0.8),
              ((916, 836, 1380, H), INK, 0.9)]
    for (x0, y0, x1, y1), c, t0 in blocks:
        k = ease_out((t - t0) / 0.3)
        if k > 0:
            d.rectangle([x0 + (2 if x0 else 0), y0 + (2 if y0 else 0), x0 + (x1 - x0) * k, y1 - 1], fill=c)
    if t > 0.75:
        k = clamp((t - 0.75) / 0.3)
        col = tuple(int(WHITE[i] + (INK[i] - WHITE[i]) * k) for i in range(3))
        text(d, "Level-of-Token", 96, (440, 400), fill=col, weight="Bold")
        text(d, "Diffusion", 96, (440, 520), fill=col, weight="Bold")
        text(d, "big tokens where nothing happens", 40, (444, 690), fill=col, weight="Regular")


T_TW = (T_CUE - 0.5, T_CUE)       # title → image wipe


def scene(t):
    canvas = Image.new("RGB", (W, H), CANVAS)
    d = ImageDraw.Draw(canvas)
    if t < T_CUE:
        title_scene(canvas, d, t)
    elif t < T_WIPE[1]:
        img_scene(canvas, d, t)
    else:
        vid_scene(canvas, d, t)
    return canvas


def wipe(t, win, before, after):
    """black bar sweeps left → right; left of it the next scene."""
    k = ease((t - win[0]) / (win[1] - win[0]))
    bx = int(-60 + (W + 120) * k)
    a = before(t)
    b = after(t)
    if bx - 30 > 0:
        a.paste(b.crop((0, 0, min(W, bx - 30), H)), (0, 0))
    ImageDraw.Draw(a).rectangle([bx - 30, 0, bx + 30, H], fill=INK)
    return a


def _title(t):
    c = Image.new("RGB", (W, H), CANVAS)
    title_scene(c, ImageDraw.Draw(c), t)
    return c


def _img(t):
    c = Image.new("RGB", (W, H), CANVAS)
    img_scene(c, ImageDraw.Draw(c), max(t, T_CUE))
    return c


def _vid(t):
    c = Image.new("RGB", (W, H), CANVAS)
    vid_scene(c, ImageDraw.Draw(c), t)
    return c


def render(t):
    a = assets()
    if T_TW[0] <= t < T_TW[1]:
        canvas = wipe(t, T_TW, _title, _img)
    elif T_WIPE[0] <= t < T_WIPE[1]:
        canvas = wipe(t, T_WIPE, _img, _vid)
    else:
        canvas = scene(t)
    d = ImageDraw.Draw(canvas)
    if t >= T_END:
        k = ease_out((t - T_END) / 0.45)
        y0 = int(H - 300 * k)
        d.rectangle([0, y0, W, H], fill=CANVAS)
        d.rectangle([0, y0, W, y0 + 14], fill=INK)
        d.rectangle([0, y0 + 14, 120, H], fill=RED)
        d.rectangle([120, y0 + 14, 134, H], fill=INK)
        d.rectangle([W - 260, y0 + 14, W - 246, H], fill=INK)
        d.rectangle([W - 246, y0 + 14, W, y0 + 150], fill=YELLOW)
        d.rectangle([W - 246, y0 + 150, W, y0 + 164], fill=INK)
        d.rectangle([W - 246, y0 + 164, W, H], fill=BLUE)
        if t > T_END + 0.3:
            text(d, "Level-of-Token Diffusion", 84, (200, y0 + 60), weight="Bold")
        if t > T_END + 0.7:
            text(d, "images 2.4× faster   ·   video 2.5× faster", 44, (204, y0 + 180), weight="Medium")
    f = np.asarray(canvas).astype(np.float32) + a["paper"]
    return np.clip(f, 0, 255).astype(np.uint8)


# ------------------------------------------------------------------ audio: marimba + brush
SR = 48000


def marimba(freq, dur=0.9, vol=0.25):
    n = int(dur * SR)
    tt = np.arange(n) / SR
    s = (np.sin(2 * np.pi * freq * tt) * np.exp(-tt / 0.35)
         + 0.35 * np.sin(2 * np.pi * freq * 3.93 * tt) * np.exp(-tt / 0.06)
         + 0.12 * np.sin(2 * np.pi * freq * 9.2 * tt) * np.exp(-tt / 0.02))
    s[:60] *= np.linspace(0, 1, 60)
    return s * vol


def piano(freq, dur=2.0, vol=0.18):
    n = int(dur * SR)
    tt = np.arange(n) / SR
    s = sum(np.sin(2 * np.pi * freq * k * (1 + 0.0004 * k * k) * tt) * np.exp(-tt * (0.9 + 0.9 * k)) / k for k in range(1, 7))
    s[:80] *= np.linspace(0, 1, 80)
    return s * vol


def brush(dur=0.35, vol=0.08, seed=0):
    n = int(dur * SR)
    rng = np.random.default_rng(seed)
    w = rng.normal(0, 1, n)
    w = np.convolve(w, np.ones(12) / 12, mode="same")          # soften
    w = w - np.convolve(w, np.ones(200) / 200, mode="same")     # remove rumble
    env = np.sin(np.pi * np.linspace(0, 1, n)) ** 1.5
    return w * env * vol


def note(name):
    names = {"C": -9, "C#": -8, "D": -7, "D#": -6, "E": -5, "F": -4, "F#": -3, "G": -2, "G#": -1, "A": 0, "A#": 1, "B": 2}
    return 440 * 2 ** ((names[name[:-1]] + 12 * (int(name[-1]) - 4)) / 12)


def build_audio(path):
    out = np.zeros((int((DUR + 1) * SR), 2))

    def add(sig, t0, pan=0.0):
        i = int(t0 * SR)
        j = min(len(out), i + len(sig))
        out[i:j, 0] += sig[:j - i] * (1 - pan) / 2 * 2 ** 0.5
        out[i:j, 1] += sig[:j - i] * (1 + pan) / 2 * 2 ** 0.5

    penta = ["C5", "D5", "E5", "G5", "A5", "C6", "D6", "E6"]
    # title: bars and blocks
    for k, t0 in enumerate((0.05, 0.15, 0.25, 0.35, 0.42, 0.5)):
        add(marimba(note(["C4", "G4", "E4", "C5", "G4", "E5"][k]), vol=0.22), t0, pan=-0.4 + 0.16 * k)
    for t0 in (0.55, 0.7, 0.8, 0.9):
        add(brush(0.3, 0.1, int(t0 * 100)), t0)
    add(piano(note("C4"), 2.5, 0.16), 0.8)
    add(piano(note("G4"), 2.5, 0.12), 0.8)
    # cue
    add(marimba(note("E5"), vol=0.2), T_CUE + 0.1)
    # lattice: soft ticks rising
    for k in range(10):
        add(marimba(note(penta[k % 8]) * (2 if k >= 8 else 1), 0.5, 0.12), L0[0] + k * (L0[1] - L0[0]) / 10, pan=-0.6 + 0.12 * k)
    # roller sweeps: brush strokes + plucks
    for win, seq in ((S2, ["G4", "A4", "C5", "D5", "E5", "G5"]), (S3, ["E4", "G4", "A4", "C5", "D5", "E5"]),
                     (B2, ["G4", "A4", "C5", "D5", "E5", "G5"]), (B3, ["E4", "G4", "A4", "C5", "D5", "E5"])):
        n = len(seq)
        for k in range(n):
            tk = win[0] + k * (win[1] - win[0]) / n
            add(brush(0.3, 0.09, int(tk * 100)), tk, pan=-0.3 + 0.1 * k)
            add(marimba(note(seq[k]), 0.7, 0.16), tk + 0.05)
    # speedups: chord
    for t0, chord in ((T_SP1, ["C4", "E4", "G4", "C5"]), (T_SP2, ["F4", "A4", "C5", "F5"]),
                      (T_BSP1, ["C4", "E4", "G4", "C5"]), (T_BSP2, ["F4", "A4", "C5", "F5"])):
        for k, nm in enumerate(chord):
            add(piano(note(nm), 2.0, 0.13), t0 + k * 0.03)
        add(marimba(note(chord[-1]) * 2, 0.8, 0.18), t0 + 0.12)
    # reveal: arpeggio up
    for k, nm in enumerate(["C5", "E5", "G5", "C6", "E6", "G6"]):
        add(marimba(note(nm), 0.8, 0.15), R0[0] + k * 0.18, pan=-0.5 + 0.2 * k)
    # wipe
    add(brush(0.6, 0.12, 9), T_WIPE[0])
    for k in range(6):
        add(marimba(note(penta[k]), 0.4, 0.1), VL0[0] + k * 0.11)
    # end chord
    for k, nm in enumerate(["C4", "G4", "C5", "E5", "G5"]):
        add(piano(note(nm), 2.6, 0.14), T_END + 0.1 + k * 0.06)
    out = out[:int(DUR * SR)]
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
        for s in sys.argv[2:]:
            Image.fromarray(render(float(s))).save(os.path.join(HERE, f"scratch/still_{TAG}_{float(s):05.2f}.png"))
        return
    w, h, fps = (1920, 1080, 60) if mode == "h" else (960, 540, 30)
    res = "1080p60" if mode == "h" else "540p30"
    os.makedirs(os.path.join(HERE, "media/final"), exist_ok=True)
    wav = os.path.join(HERE, f"media/final/LoT_mondrian_{TAG}.wav")
    build_audio(wav)
    out = os.path.join(HERE, f"media/final/LoT_mondrian_{TAG}_{res}.mp4")
    nf = int(DUR * fps)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps),
           "-i", "-", "-i", wav, "-c:v", "libx264", "-preset", "slow", "-crf", "16", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-ac", "2", "-ar", "48000", "-b:a", "192k", "-t", f"{DUR}", "-movflags", "+faststart", out]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    with Pool(10) as pool:
        for k, buf in enumerate(pool.imap(render_scaled, [(i / fps, w, h) for i in range(nf)], chunksize=2)):
            p.stdin.write(buf)
            if k % 240 == 0:
                print(f"frame {k}/{nf}", flush=True)
    p.stdin.close()
    p.wait()
    print("wrote", out)


if __name__ == "__main__":
    main()
