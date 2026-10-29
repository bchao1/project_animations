"""LoT Diffusion, 20-s retro-desktop clip, v1 (numpy + PIL + cv2 frames → ffmpeg, synthesized UI sounds). No manim.

Generic 90s desktop: teal background, bevelled grey windows, navy title bars, pixel cursor.
  0.0–1.3  startup chime, cursor double-clicks LoT.exe → "LoT Paint" opens the giraffe depth map
  1.9–3.8  rectangle tool draws every 16-px patch as a 1 × 1 token (4,096)
  3.9–8.1  "Token budget" dialog: the cursor drags the slider 1x1 → 2x2 (1,606, 1.9X) → 8x8 (1,027, 2.4X);
           out-of-focus tokens merge, the in-focus giraffe stays 1 × 1
  8.5–12.6 Render: "Full model" vs "LoT" progress dialogs race, LoT wins (Done! + tada), image viewer opens
 12.6–18.0 media player plays the platformer video, a Tokens window shows its tokens; clicking 2x2 / 4x4
           merges ground / sky and the player speeds up 2.2X / 2.5X
 18.0–20.0 About LoT dialog

Speedups as in ../arcade_clip (paper Fig. 21 fit for images, Tab. 2 interpolation for video).
Usage: python render_retro_desktop_v1.py h          → media/final/LoT_retro_desktop_v1_1080p60.mp4
       python render_retro_desktop_v1.py still t ...  → scratch/still_v1_<t>.png
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
FONT = os.path.join(HERE, "data/PixelifySans.ttf")
CUE = os.path.join(ROOT, "data/img/img_000000000025_dof_cue.png")
GEN = os.path.join(ROOT, "data/img/img_000000000025_dof_generated_4b.png")
LAYOUT = os.path.join(ROOT, "data/layouts/img_000000000025_dof.json")
VID = os.path.join(ROOT, "data/clips/vd_platformer_gen")
LEVELS = os.path.join(ROOT, "arcade_clip/data/platformer_levels.npy")

W, H = 1920, 1080
DUR = 20.0

TEAL = (0, 128, 128)
GREY = (192, 192, 192)
DGREY = (128, 128, 128)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
NAVY = (0, 0, 128)
NAVY2 = (16, 132, 208)
TIP = (255, 255, 225)
SEL = (0, 0, 128)

# ------------------------------------------------------------------ timeline (s)
T_CHIME = 0.1
T_DBL = (0.85, 1.0)
T_OPEN = 1.05
T_TOOL = 1.9
S1 = (2.2, 3.8)
T_DLG = 3.95
D1 = (4.6, 5.0)                 # slider drags
S2 = (5.05, 6.15)
T_SP1 = 6.2
D2 = (6.5, 7.0)
S3 = (7.05, 8.05)
T_SP2 = 8.1
T_RENDER = 8.55
R0 = 8.65
R_LOT = 1.6                     # LoT bar fill time; full model takes 2.4 × longer
R_FULL = R_LOT * 2.4
T_DONE = R0 + R_LOT + 0.05
T_OK = 11.0
T_VIEW = 11.05
T_MIN = 12.65
B_OPEN = 13.0
V0 = 13.1                       # video starts
B1 = (13.3, 14.0)
T_C2 = 14.5
B2 = (14.55, 15.35)
T_BSP1 = 15.4
T_C4 = 16.3
B3 = (16.35, 17.1)
T_BSP2 = 17.15
T_ABOUT = 18.0
ZOOM = 0.25

VID_SPEED = (1.0, 2.2, 2.5)
VID_TOK = (75600, 38046, 34698)

# ------------------------------------------------------------------ geometry
P = 768
N = 64
WX, WY, WW, WH = 260, 30, 1010, 976          # paint window
CX0, CY0 = WX + 120, WY + 92                  # canvas
BX, BY, BW, BH = 1310, 60, 570, 400           # budget dialog
TX0, TX1, TY = BX + 70, BX + BW - 70, BY + 170
PG1 = (1310, 490, 570, 210)                   # progress dialogs
PG2 = (1310, 730, 570, 210)
DN = (720, 400, 520, 250)                     # done dialog
IV = (340, 64, 836, 940)                      # image viewer
VW, VH = 880, 495
PL = (24, 110, 928, 720)                      # player window
TK = (968, 110, 928, 720)                     # tokens window
VN, VM = 45, 80
AB = (560, 290, 800, 440)                     # about dialog
TASK_Y = 1024


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
    cue = np.asarray(Image.open(CUE).convert("RGB")).astype(np.float32)
    a["cue"] = cv2.resize(cue, (P, P), interpolation=cv2.INTER_AREA)
    a["cue_cell"] = cue.reshape(N, 16, N, 16, 3).mean((1, 3))
    gen = np.asarray(Image.open(GEN).convert("RGB"))
    a["gen"] = Image.fromarray(cv2.resize(gen, (P, P), interpolation=cv2.INTER_AREA))
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
    a["desk"] = desktop_base()
    return a


A = None


def assets():
    global A
    if A is None:
        A = load_assets()
    return A


# ------------------------------------------------------------------ text
@lru_cache(maxsize=64)
def font(size, bold=False):
    f = ImageFont.truetype(FONT, size)
    f.set_variation_by_name("Bold" if bold else "Regular")
    return f


def text(d, xy, s, size=24, fill=BLACK, bold=False, anchor="la"):
    d.text(xy, s, font=font(size, bold), fill=fill, anchor=anchor)


def tw(s, size=24, bold=False):
    return font(size, bold).getlength(s)


# ------------------------------------------------------------------ widgets
def bevel(d, x, y, w, h, raised=True, fill=GREY):
    x1, y1 = x + w - 1, y + h - 1
    d.rectangle([x, y, x1, y1], fill=fill)
    lt, rb = (WHITE, BLACK) if raised else (DGREY, WHITE)
    lt2, rb2 = (GREY if raised else BLACK), (DGREY if raised else GREY)
    d.line([(x, y1), (x, y), (x1, y)], fill=lt, width=2)
    d.line([(x1, y), (x1, y1), (x, y1)], fill=rb, width=2)
    d.line([(x + 2, y1 - 2), (x + 2, y + 2), (x1 - 2, y + 2)], fill=lt2, width=2)
    d.line([(x1 - 2, y + 2), (x1 - 2, y1 - 2), (x + 2, y1 - 2)], fill=rb2, width=2)


def sunken(d, x, y, w, h, fill=WHITE):
    bevel(d, x, y, w, h, raised=False, fill=fill)


def button(d, x, y, w, h, label, pressed=False, size=24, bold=False, focus=False):
    bevel(d, x, y, w, h, raised=not pressed)
    o = 2 if pressed else 0
    text(d, (x + w / 2 + o, y + h / 2 + o), label, size, BLACK, bold, anchor="mm")
    if focus:
        for k in range(x + 8, x + w - 8, 4):
            d.point([(k, y + 7), (k, y + h - 8)], fill=BLACK)


@lru_cache(maxsize=32)
def title_grad(w, active):
    c0, c1 = (NAVY, NAVY2) if active else (DGREY, (180, 180, 180))
    g = np.linspace(0, 1, w)[None, :, None]
    arr = (np.array(c0)[None, None] * (1 - g) + np.array(c1)[None, None] * g)
    return Image.fromarray(np.repeat(arr, 36, 0).astype(np.uint8))


def window(canvas, d, rect, title, active=True, scale=1.0):
    x, y, w, h = rect
    bevel(d, x, y, w, h)
    canvas.paste(title_grad(w - 12, active), (x + 6, y + 6))
    text(d, (x + 16, y + 24), title, 24, WHITE, True, anchor="lm")
    for k, g in enumerate(("x", "max", "min")):
        bx = x + w - 44 - k * 36 - (6 if k else 0)
        bevel(d, bx, y + 10, 32, 28)
        if g == "x":
            d.line([(bx + 9, y + 17), (bx + 22, y + 30)], fill=BLACK, width=3)
            d.line([(bx + 22, y + 17), (bx + 9, y + 30)], fill=BLACK, width=3)
        elif g == "max":
            d.rectangle([bx + 8, y + 16, bx + 23, y + 30], outline=BLACK, width=2)
            d.line([(bx + 8, y + 17), (bx + 23, y + 17)], fill=BLACK, width=3)
        else:
            d.line([(bx + 9, y + 29), (bx + 20, y + 29)], fill=BLACK, width=3)


def zoom_rects(d, src, dst, f):
    """the classic expanding outline animation between two rects (x, y, w, h)."""
    for k in range(4):
        g = ease(f * 1.3 - k * 0.1)
        if 0 < g < 1:
            r = [src[i] + (dst[i] - src[i]) * g for i in range(4)]
            d.rectangle([r[0], r[1], r[0] + r[2], r[1] + r[3]], outline=DGREY, width=3)


def progress(d, x, y, w, h, frac):
    sunken(d, x, y, w, h)
    bw, gap = 22, 4
    n = int((w - 8) // (bw + gap))
    for k in range(int(n * clamp(frac) + 1e-6)):
        bx = x + 5 + k * (bw + gap)
        d.rectangle([bx, y + 5, bx + bw - 1, y + h - 6], fill=NAVY)


def tooltip(d, s, x, y, px=None, py=None):
    """yellow balloon, top-left at (x, y); optional pointer tip at (px, py)."""
    w = int(tw(s, 26)) + 28
    h = 44
    if px is not None:
        bx = clamp(px, x + 20, x + w - 40)
        if py > y:
            d.polygon([(bx, y + h - 2), (bx + 22, y + h - 2), (px, py)], fill=TIP, outline=BLACK)
        else:
            d.polygon([(bx, y + 2), (bx + 22, y + 2), (px, py)], fill=TIP, outline=BLACK)
    d.rectangle([x + 4, y + 4, x + w + 3, y + h + 3], fill=(60, 60, 60))
    d.rectangle([x, y, x + w - 1, y + h - 1], fill=TIP, outline=BLACK, width=2)
    if px is not None:
        bx = clamp(px, x + 20, x + w - 40)
        yy = y + h - 2 if py > y else y + 1
        d.line([(bx + 2, yy), (bx + 20, yy)], fill=TIP, width=3)
    text(d, (x + 14, y + h / 2), s, 26, BLACK, anchor="lm")


def arrow_cursor(d, x, y):
    pts = [(0, 0), (0, 34), (8, 27), (14, 41), (21, 38), (15, 24), (25, 24)]
    d.polygon([(x + px, y + py) for px, py in pts], fill=WHITE, outline=BLACK)
    d.line([(x + px, y + py) for px, py in pts + [pts[0]]], fill=BLACK, width=2)


def cross_cursor(d, x, y):
    for c, wd in ((WHITE, 5), (BLACK, 2)):
        d.line([(x - 16, y), (x + 16, y)], fill=c, width=wd)
        d.line([(x, y - 16), (x, y + 16)], fill=c, width=wd)


# ------------------------------------------------------------------ desktop
def icon_lot(d, x, y):
    d.rectangle([x, y, x + 63, y + 63], fill=WHITE, outline=BLACK, width=2)
    d.rectangle([x + 2, y + 2, x + 31, y + 31], fill=(0, 160, 160), outline=BLACK)
    for i in range(2):
        for j in range(2):
            d.rectangle([x + 32 + j * 15, y + 2 + i * 15, x + 46 + j * 15, y + 16 + i * 15], fill=(255, 220, 0), outline=BLACK)
    for i in range(4):
        for j in range(8):
            d.rectangle([x + 2 + j * 7.5, y + 32 + i * 7.5, x + 9 + j * 7.5, y + 39 + i * 7.5], fill=(220, 40, 40), outline=BLACK)


def icon_bmp(d, x, y):
    d.rectangle([x + 6, y, x + 57, y + 63], fill=WHITE, outline=BLACK, width=2)
    for k in range(40):
        c = cv2.applyColorMap(np.array([[int(k / 40 * 255)]], np.uint8), cv2.COLORMAP_TURBO)[0, 0, ::-1]
        d.line([(x + 12, y + 56 - k), (x + 51, y + 56 - k)], fill=tuple(int(v) for v in c))


def icon_avi(d, x, y):
    d.rectangle([x + 2, y + 10, x + 61, y + 53], fill=(40, 40, 40), outline=BLACK, width=2)
    for k in range(6):
        d.rectangle([x + 6 + k * 10, y + 13, x + 11 + k * 10, y + 18], fill=WHITE)
        d.rectangle([x + 6 + k * 10, y + 45, x + 11 + k * 10, y + 50], fill=WHITE)
    d.rectangle([x + 8, y + 22, x + 55, y + 41], fill=(90, 160, 230))


def icon_trash(d, x, y):
    d.polygon([(x + 12, y + 14), (x + 52, y + 14), (x + 47, y + 62), (x + 17, y + 62)], fill=(220, 220, 220), outline=BLACK)
    d.rectangle([x + 8, y + 6, x + 56, y + 14], fill=(200, 200, 200), outline=BLACK, width=2)
    for k in range(3):
        d.line([(x + 22 + k * 10, y + 20), (x + 22 + k * 10, y + 56)], fill=DGREY, width=2)


ICONS = [("LoT.exe", icon_lot), ("giraffe_depth.bmp", icon_bmp), ("platformer.avi", icon_avi), ("Trash", icon_trash)]
ICON_X, ICON_Y0, ICON_DY = 70, 40, 150


def desktop_base():
    im = Image.new("RGB", (W, H), TEAL)
    d = ImageDraw.Draw(im)
    for k, (name, fn) in enumerate(ICONS):
        fn(d, ICON_X, ICON_Y0 + k * ICON_DY)
    return np.asarray(im).copy()


def desktop_labels(d, t, selected):
    for k, (name, _) in enumerate(ICONS):
        cx, y = ICON_X + 32, ICON_Y0 + k * ICON_DY + 74
        w = tw(name, 22)
        if selected == k:
            d.rectangle([cx - w / 2 - 4, y - 2, cx + w / 2 + 4, y + 26], fill=SEL)
            text(d, (cx, y), name, 22, WHITE, anchor="ma")
        else:
            text(d, (cx + 1, y + 1), name, 22, BLACK, anchor="ma")
            text(d, (cx, y), name, 22, WHITE, anchor="ma")


def taskbar(canvas, d, t, items, active):
    bevel(d, 0, TASK_Y, W, H - TASK_Y)
    bevel(d, 6, TASK_Y + 8, 120, 42)
    icon_small = Image.new("RGB", (28, 28), WHITE)
    di = ImageDraw.Draw(icon_small)
    di.rectangle([0, 0, 13, 13], fill=(0, 160, 160))
    di.rectangle([14, 0, 27, 13], fill=(255, 220, 0))
    di.rectangle([0, 14, 27, 27], fill=(220, 40, 40))
    canvas.paste(icon_small, (16, TASK_Y + 15))
    text(d, (54, TASK_Y + 29), "LoT", 26, BLACK, True, anchor="lm")
    x = 140
    for k, it in enumerate(items):
        w = 250
        bevel(d, x, TASK_Y + 8, w, 42, raised=(k != active), fill=(220, 220, 220) if k == active else GREY)
        s = it if tw(it, 22) < w - 20 else it[:16] + "..."
        text(d, (x + 14, TASK_Y + 29), s, 22, BLACK, k == active, anchor="lm")
        x += w + 8
    sunken(d, W - 150, TASK_Y + 8, 144, 42, fill=GREY)
    text(d, (W - 78, TASK_Y + 29), "12:00 PM", 22, BLACK, anchor="mm")


# ------------------------------------------------------------------ token mosaics (from ../arcade_clip)
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


def edges(ids_px, thick=1):
    e = np.zeros(ids_px.shape, bool)
    e[:, 1:] |= ids_px[:, 1:] != ids_px[:, :-1]
    e[1:, :] |= ids_px[1:, :] != ids_px[:-1, :]
    if thick > 1:
        e[:, :-1] |= e[:, 1:]
        e[:-1, :] |= e[1:, :]
    return e


def staged_ids(blocks_old, blocks_new, n, m, t0, dur, t):
    oi_o, oj_o, _ = blocks_old
    oi_n, oj_n, b_n = blocks_new
    t_sw = t0 + (oi_n + b_n) / n * dur
    new = t >= t_sw
    return block_ids(np.where(new, oi_n, oi_o), np.where(new, oj_n, oj_o), m), t_sw


def up(arr, k):
    return np.repeat(np.repeat(arr, k, 0), k, 1)


def frac(t, win):
    return clamp((t - win[0]) / (win[1] - win[0]))


def invert_flash(mos, fl):
    return mos + fl[..., None] * ((255 - mos) - mos)


def paint_canvas(t):
    """returns (P × P × 3 float, tokens, bar y px or None, largest token size)."""
    a = assets()
    B = a["img_blocks"]
    k = P // N
    if t < S1[0]:
        return a["cue"], 0, None, 0
    if t < S2[0]:
        f = frac(t, S1)
        rows = int(f * N + 1e-6)
        ids = block_ids(B[1][0], B[1][1], N)
        img = a["cue"].copy()
        ry = rows * k
        img[:ry] = up(mosaic(a["cue_cell"], ids, N * N), k)[:ry]
        e = edges(up(ids, k))
        e[ry:] = False
        img = img * (1 - 0.6 * e[..., None])
        return img, rows * N, (f * P if f < 1 else None), 1
    old, new, win, cap = (B[1], B[2], S2, 2) if t < S3[0] else (B[2], B[8], S3, 8)
    f = frac(t, win)
    ids, t_sw = staged_ids(old, new, N, N, win[0], win[1] - win[0], t)
    dt = t - t_sw
    fl = np.where((new[2] > old[2]) & (dt >= 0) & (dt < 0.1), 1.0, 0.0).astype(np.float32)   # XOR-style flash
    mos = invert_flash(mosaic(a["cue_cell"], ids, N * N), fl)
    img = up(mos, k)
    e = edges(up(ids, k))
    img = img * (1 - 0.6 * e[..., None])
    return img, len(np.unique(ids)), (f * P if 0 < f < 1 else None), (cap if f > 0 else (1 if cap == 2 else 2))


# ------------------------------------------------------------------ video
def vid_speed(t):
    return VID_SPEED[0] if t < T_BSP1 else (VID_SPEED[1] if t < T_BSP2 else VID_SPEED[2])


def vid_pos(t):
    v = 0.0
    for s0, s1, sp in ((V0, T_BSP1, VID_SPEED[0]), (T_BSP1, T_BSP2, VID_SPEED[1]), (T_BSP2, 99, VID_SPEED[2])):
        if t > s0:
            v += (min(t, s1) - s0) * sp * 15
    return v


def vid_blocks(lv, cap):
    b = np.minimum(lv, cap).astype(int)
    ii, jj = np.mgrid[0:VN, 0:VM]
    return ii // b * b, jj // b * b, b


def token_panel(t):
    a = assets()
    idx = int(vid_pos(t)) % 81
    frame, cell, lv = a["vid"][idx], a["vid_cell"][idx], a["lv"][idx]
    right = frame * 0.45 + 255 * 0.55 * 0.6
    bar, tok = None, 0
    if t >= B1[0]:
        if t < B2[0]:
            f = frac(t, B1)
            ry = int(f * VN + 1e-6) * 11
            ids = block_ids(*vid_blocks(lv, 1)[:2], VM)
            mos = up(mosaic(cell, ids, VN * VM), 11)
            right = right.copy()
            right[:ry] = mos[:ry]
            e = edges(up(ids, 11))
            e[ry:] = False
            bar = f * VH if f < 1 else None
            tok = VID_TOK[0] * f
        else:
            old, new, win, k = (vid_blocks(lv, 1), vid_blocks(lv, 2), B2, 1) if t < B3[0] else \
                (vid_blocks(lv, 2), vid_blocks(lv, 4), B3, 2)
            f = frac(t, win)
            ids, t_sw = staged_ids(old, new, VN, VM, win[0], win[1] - win[0], t)
            dt = t - t_sw
            fl = np.where((new[2] > old[2]) & (dt >= 0) & (dt < 0.1), 1.0, 0.0).astype(np.float32)
            right = up(invert_flash(mosaic(cell, ids, VN * VM), fl), 11)
            e = edges(up(ids, 11))
            bar = f * VH if 0 < f < 1 else None
            tok = VID_TOK[k - 1] + (VID_TOK[k] - VID_TOK[k - 1]) * f
        right = right * (1 - 0.55 * e[..., None])
    return frame, right, bar, int(round(tok)), idx


# ------------------------------------------------------------------ cursor
def thumb_k(t):
    """slider position in tick units (0 = 1x1 … 3 = 8x8)."""
    if t < D1[0]:
        return 0.0
    if t < D1[1]:
        return ease(frac(t, D1))
    if t < D2[0]:
        return 1.0
    if t < D2[1]:
        return 1 + 2 * ease(frac(t, D2))
    return 3.0


def thumb_x(t):
    return TX0 + (TX1 - TX0) * thumb_k(t) / 3


RENDER_BTN = (BX + BW - 190, BY + BH - 66, 160, 46)
OK_BTN = (DN[0] + DN[2] // 2 - 70, DN[1] + DN[3] - 70, 140, 46)
TB_BTNS = {1: (TK[0] + 24, TK[1] + 636, 120, 50), 2: (TK[0] + 154, TK[1] + 636, 120, 50), 4: (TK[0] + 284, TK[1] + 636, 120, 50)}
ABOUT_OK = (AB[0] + AB[2] // 2 - 70, AB[1] + AB[3] - 72, 140, 46)
TOOL_RECT = (WX + 14, WY + 92 + 4 * 52, 44, 44)


def ctr(r):
    return r[0] + r[2] / 2, r[1] + r[3] / 2


def cursor_pos(t):
    """(x, y, kind) — kind 'arrow' or 'cross'."""
    if S1[0] <= t < S1[1]:
        f = frac(t, S1)
        row = f * N
        x = CX0 + (row % 1) * P
        return x, CY0 + f * P, "cross"
    if D1[0] <= t < D1[1] or D2[0] <= t < D2[1]:
        return thumb_x(t) + 4, TY + 6, "arrow"
    icon = (ICON_X + 30, ICON_Y0 + 30)
    tool = ctr(TOOL_RECT)
    keys = [
        (0.0, (1150, 700)), (0.75, icon), (1.3, icon), (1.8, tool), (2.15, (CX0, CY0)),
        (3.8, (CX0 + P, CY0 + P)), (4.5, (TX0 + 4, TY + 6)), (5.0, (TX0 + (TX1 - TX0) / 3 + 4, TY + 6)),
        (6.4, (TX0 + (TX1 - TX0) / 3 + 4, TY + 6)), (7.0, (TX1 + 4, TY + 6)), (8.4, ctr(RENDER_BTN)),
        (10.4, ctr(RENDER_BTN)), (10.9, ctr(OK_BTN)), (11.6, ctr(OK_BTN)), (12.5, (900, 990)),
        (13.6, (1500, 900)), (14.4, ctr(TB_BTNS[2])), (15.6, ctr(TB_BTNS[2])), (16.2, ctr(TB_BTNS[4])),
        (17.4, ctr(TB_BTNS[4])), (18.6, ctr(ABOUT_OK)), (20.0, ctr(ABOUT_OK)),
    ]
    for (t0, p0), (t1, p1) in zip(keys, keys[1:]):
        if t0 <= t <= t1:
            g = ease((t - t0) / (t1 - t0))
            return p0[0] + (p1[0] - p0[0]) * g, p0[1] + (p1[1] - p0[1]) * g, "arrow"
    return keys[-1][1][0], keys[-1][1][1], "arrow"


CLICKS = [T_DBL[0], T_DBL[1], T_TOOL, D1[0], D1[1], D2[0], D2[1], T_RENDER, T_OK, T_C2, T_C4]


def pressed(t, t0, dur=0.12):
    return t0 <= t < t0 + dur


# ------------------------------------------------------------------ frame
def fmt(n):
    return f"{n:,}"


def render(t):
    a = assets()
    canvas = Image.fromarray(a["desk"]).convert("RGB")
    d = ImageDraw.Draw(canvas)
    sel = 0 if T_DBL[0] <= t < T_MIN else None
    desktop_labels(d, t, sel)
    items, active = [], -1
    stage1 = T_OPEN <= t < T_MIN + ZOOM

    # ---------------------------------------------- stage 1 windows
    if t >= T_OPEN and t < T_OPEN + ZOOM:
        zoom_rects(d, (ICON_X, ICON_Y0, 64, 64), (WX, WY, WW, WH), (t - T_OPEN) / ZOOM)
    if T_OPEN + ZOOM <= t < T_MIN:
        items = ["LoT Paint - giraffe_depth.bmp"]
        active = 0
        img, tok, bar, cap = paint_canvas(t)
        paint_active = not (T_DLG <= t)
        window(canvas, d, (WX, WY, WW, WH), "giraffe_depth.bmp - LoT Paint", active=paint_active and t < T_VIEW)
        x0 = WX + 10
        for k, m in enumerate(("File", "Edit", "View", "Tokens", "Help")):
            text(d, (x0 + 14, WY + 62), m, 24, BLACK, anchor="lm")
            x0 += tw(m) + 30
        # toolbox
        for r in range(8):
            for c in range(2):
                bx, by = WX + 14 + c * 50, WY + 92 + r * 52
                is_rect = (r == 4 and c == 0)
                down = is_rect and t >= T_TOOL
                bevel(d, bx, by, 44, 44, raised=not down, fill=(225, 225, 225) if down else GREY)
                gx, gy = bx + 10, by + 10
                if is_rect:
                    d.rectangle([gx, gy + 4, gx + 24, gy + 20], outline=BLACK, width=2)
                elif (r, c) == (0, 0):
                    d.rectangle([gx + 2, gy + 2, gx + 22, gy + 22], outline=BLACK, width=1)
                elif (r, c) == (0, 1):
                    d.polygon([(gx + 4, gy + 22), (gx + 12, gy), (gx + 20, gy + 22)], outline=BLACK)
                elif (r, c) == (1, 0):
                    d.line([(gx + 2, gy + 22), (gx + 22, gy + 2)], fill=BLACK, width=3)
                elif (r, c) == (1, 1):
                    d.ellipse([gx + 2, gy + 4, gx + 22, gy + 20], outline=BLACK, width=2)
                elif (r, c) == (2, 0):
                    d.rectangle([gx + 4, gy + 2, gx + 10, gy + 22], fill=BLACK)
                elif (r, c) == (2, 1):
                    d.text((gx + 4, gy - 2), "A", font=font(26, True), fill=BLACK)
                elif (r, c) == (3, 0):
                    d.polygon([(gx + 2, gy + 14), (gx + 12, gy + 2), (gx + 22, gy + 14), (gx + 12, gy + 24)], fill=DGREY)
                elif (r, c) == (3, 1):
                    for q in range(5):
                        d.point([(gx + 4 + q * 4, gy + 6 + (q % 2) * 8)], fill=BLACK)
                elif (r, c) == (4, 1):
                    d.rectangle([gx, gy + 4, gx + 24, gy + 20], fill=DGREY)
                else:
                    d.line([(gx + 4, gy + 12), (gx + 20, gy + 12)], fill=DGREY, width=2)
        sunken(d, CX0 - 6, CY0 - 6, P + 12, P + 12, fill=DGREY)
        canvas.paste(Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)), (CX0, CY0))
        if bar is not None:
            by = CY0 + int(bar)
            for xx in range(CX0, CX0 + P, 12):
                d.line([(xx, by), (xx + 6, by)], fill=BLACK, width=2)
                d.line([(xx + 6, by), (xx + 12, by)], fill=WHITE, width=2)
        # palette strip: the four token sizes as "colours"
        py = CY0 + P + 12
        sunken(d, WX + 14, py, 90, 64, fill=GREY)
        d.rectangle([WX + 28, py + 12, WX + 88, py + 52], fill=BLACK)
        for k, s in enumerate((1, 2, 4, 8)):
            bx = CX0 + k * 96
            sunken(d, bx, py, 88, 64, fill=WHITE)
            q = 6 * s if s < 8 else 44
            d.rectangle([bx + 44 - q / 2, py + 32 - q / 2, bx + 44 + q / 2, py + 32 + q / 2], outline=BLACK, width=2)
        text(d, (CX0 + 4 * 96 + 12, py + 32), "token sizes", 24, BLACK, anchor="lm")
        # status bar
        sy = WY + WH - 44
        sunken(d, WX + 10, sy, 470, 36, fill=GREY)
        sunken(d, WX + 490, sy, 250, 36, fill=GREY)
        sunken(d, WX + 750, sy, 250, 36, fill=GREY)
        msg = "Ready" if t < T_TOOL else ("Rectangle: draw tokens" if t < S1[0] else f"Tokens: {fmt(tok)}")
        if t >= T_VIEW:
            msg = "Tokens: 1,027"
        text(d, (WX + 22, sy + 18), msg, 24, BLACK, anchor="lm")
        if cap:
            text(d, (WX + 502, sy + 18), f"Largest: {cap}x{cap}", 24, BLACK, anchor="lm")
        sp = 1.0 if t < T_SP1 else (1.9 if t < T_SP2 else 2.4)
        if t >= S1[1]:
            text(d, (WX + 762, sy + 18), f"Speed: {sp:.1f}X", 24, BLACK, anchor="lm")

        # captions as tooltips
        if 2.5 <= t < 3.9:
            tooltip(d, "Every patch = 1 token", CX0 + 200, CY0 + 30)
        if 5.15 <= t < 6.6:
            tooltip(d, "Blurry = bigger tokens", CX0 + 40, CY0 + 300, CX0 + 160, CY0 + 420)
        if 7.2 <= t < 8.5:
            tooltip(d, "In focus stays 1x1", CX0 + 280, CY0 + 200, CX0 + 610, CY0 + 360)

        # budget dialog
        if t >= T_DLG:
            items.append("Token budget")
            active = 1
            window(canvas, d, (BX, BY, BW, BH), "Token budget", active=t < T_RENDER + 0.1)
            text(d, (BX + 30, BY + 80), "Largest token size:", 26, BLACK, True)
            sunken(d, TX0 - 10, TY - 4, TX1 - TX0 + 20, 10, fill=WHITE)
            for k, lab in enumerate(("1x1", "2x2", "4x4", "8x8")):
                xk = TX0 + (TX1 - TX0) * k / 3
                d.line([(xk, TY + 32), (xk, TY + 42)], fill=BLACK, width=2)
                text(d, (xk, TY + 50), lab, 24, BLACK, anchor="ma")
            th = thumb_x(t)
            bevel(d, int(th - 12), TY - 22, 24, 48)
            d.polygon([(th - 12, TY + 26), (th + 12, TY + 26), (th, TY + 36)], fill=GREY, outline=BLACK)
            sunken(d, BX + 30, BY + 262, 330, 110, fill=GREY)
            n_tok = 4096 if t < S2[0] else (int(paint_canvas(t)[1]) if t < S3[1] else 1027)
            text(d, (BX + 46, BY + 290), f"Tokens: {fmt(n_tok)}", 30, BLACK, True, anchor="lm")
            flash = (T_SP1 <= t < T_SP1 + 0.5 or T_SP2 <= t < T_SP2 + 0.5) and int(t * 10) % 2 == 0
            if flash:
                d.rectangle([BX + 40, BY + 320, BX + 350, BY + 362], fill=SEL)
            text(d, (BX + 46, BY + 342), f"Speed: {sp:.1f}X", 30, WHITE if flash else BLACK, True, anchor="lm")
            button(d, *RENDER_BTN, "Render", pressed=pressed(t, T_RENDER), bold=True, focus=t > T_SP2)

        # progress race
        if t >= R0:
            items.append("Rendering")
            ff, fl = clamp((t - R0) / R_FULL), clamp((t - R0) / R_LOT)
            for rect, name, ntk, f, secs in ((PG1, "Full model", "4,096 tokens", ff, 58), (PG2, "LoT", "1,027 tokens", fl, 24)):
                x, y, w, h = rect
                window(canvas, d, rect, f"Rendering - {name}", active=t < T_DONE)
                text(d, (x + 24, y + 64), f"{name}: {ntk}", 26, BLACK, True)
                progress(d, x + 24, y + 102, w - 48, 40, f)
                rem = int(round(secs * (1 - f)))
                s = "Done!" if f >= 1 else f"{rem // 60}:{rem % 60:02d} remaining"
                text(d, (x + 24, y + 170), s, 24, BLACK, f >= 1, anchor="lm")
                text(d, (x + w - 24, y + 170), f"{int(f * 100)}%", 24, BLACK, anchor="rm")

        # image viewer
        if T_VIEW <= t < T_VIEW + ZOOM:
            zoom_rects(d, ctr(OK_BTN) + (2, 2), IV, (t - T_VIEW) / ZOOM)
        if t >= T_VIEW + ZOOM:
            items.append("giraffe.png")
            active = len(items) - 1
            x, y, w, h = IV
            window(canvas, d, IV, "giraffe.png - Image Viewer", active=True)
            sunken(d, x + 28, y + 52, P + 12, P + 12, fill=BLACK)
            canvas.paste(a["gen"], (x + 34, y + 58))
            sunken(d, x + 10, y + h - 50, w - 20, 38, fill=GREY)
            text(d, (x + 24, y + h - 31), "1,027 tokens   2.4X faster", 26, BLACK, True, anchor="lm")

        # done dialog
        if T_DONE <= t < T_OK + 0.1:
            x, y, w, h = DN
            window(canvas, d, DN, "LoT", active=True)
            d.ellipse([x + 30, y + 70, x + 90, y + 130], fill=(0, 0, 200), outline=BLACK, width=2)
            text(d, (x + 60, y + 100), "i", 40, WHITE, True, anchor="mm")
            text(d, (x + 116, y + 80), "LoT finished first!", 30, BLACK, True)
            text(d, (x + 116, y + 124), "2.4X faster", 28, BLACK)
            button(d, *OK_BTN, "OK", pressed=pressed(t, T_OK), bold=True, focus=True)

    if T_MIN <= t < T_MIN + ZOOM:
        f = (t - T_MIN) / ZOOM
        for r, k in ((IV, 0), ((WX, WY, WW, WH), 1), ((BX, BY, BW, BH), 2), (PG1, 3)):
            zoom_rects(d, r, (140 + k * 258, TASK_Y + 8, 250, 42), f)
        items = ["LoT Paint - giraffe_depth.bmp", "Token budget", "Rendering", "giraffe.png"]

    # ---------------------------------------------- bonus: player + tokens
    if B_OPEN <= t < B_OPEN + ZOOM:
        f = (t - B_OPEN) / ZOOM
        zoom_rects(d, (ICON_X, ICON_Y0 + 2 * ICON_DY, 64, 64), PL, f)
        zoom_rects(d, (ICON_X, ICON_Y0 + 2 * ICON_DY, 64, 64), TK, f)
        items = ["LoT Paint - giraffe_depth.bmp", "Token budget", "Rendering", "giraffe.png"]
    if t >= B_OPEN + ZOOM:
        items = ["platformer.avi - LoT Player", "Tokens - platformer.avi"]
        active = 1
        left, right, bar, tok, idx = token_panel(t)
        sp = vid_speed(t)
        # player
        x, y, w, h = PL
        window(canvas, d, PL, "platformer.avi - LoT Player", active=False)
        sunken(d, x + 18, y + 54, VW + 12, VH + 12, fill=BLACK)
        canvas.paste(Image.fromarray(left.astype(np.uint8)), (x + 24, y + 60))
        cy = y + 590
        bevel(d, x + 24, cy, 60, 50)
        d.polygon([(x + 44, cy + 12), (x + 44, cy + 38), (x + 66, cy + 25)], fill=BLACK)
        bevel(d, x + 92, cy, 60, 50)
        d.rectangle([x + 110, cy + 14, x + 134, cy + 36], fill=BLACK)
        sunken(d, x + 170, cy + 18, 400, 14, fill=WHITE)
        sx = x + 170 + 400 * (idx / 80)
        bevel(d, int(sx - 10), cy + 6, 20, 38)
        pulse = (T_BSP1 <= t < T_BSP1 + 0.6 or T_BSP2 <= t < T_BSP2 + 0.6) and int(t * 10) % 2 == 0
        sunken(d, x + 600, cy, 300, 50, fill=SEL if pulse else WHITE)
        n_tri = 1 if sp < 1.5 else (2 if sp < 2.3 else 3)
        for k in range(n_tri):
            tx = x + 616 + k * 22
            d.polygon([(tx, cy + 13), (tx, cy + 37), (tx + 20, cy + 25)], fill=WHITE if pulse else BLACK)
        text(d, (x + 700, cy + 25), f"Speed: {sp:.1f}X", 28, WHITE if pulse else BLACK, True, anchor="lm")
        # tokens window
        x, y, w, h = TK
        window(canvas, d, TK, "Tokens - platformer.avi", active=True)
        sunken(d, x + 18, y + 54, VW + 12, VH + 12, fill=BLACK)
        canvas.paste(Image.fromarray(np.clip(right, 0, 255).astype(np.uint8)), (x + 24, y + 60))
        if bar is not None:
            by = y + 60 + int(bar)
            for xx in range(x + 24, x + 24 + VW, 12):
                d.line([(xx, by), (xx + 6, by)], fill=BLACK, width=2)
                d.line([(xx + 6, by), (xx + 12, by)], fill=WHITE, width=2)
        text(d, (x + 24, y + 600), "Largest token:", 24, BLACK, True, anchor="lm")
        cur = 1 if t < T_C2 else (2 if t < T_C4 else 4)
        for s, r in TB_BTNS.items():
            down = (s == cur) or pressed(t, T_C2 if s == 2 else T_C4 if s == 4 else -1)
            button(d, *r, f"{s}x{s}", pressed=down, size=26, bold=True)
        sunken(d, x + 430, y + 636, 470, 50, fill=GREY)
        text(d, (x + 446, y + 661), f"Tokens: {fmt(tok)}" if t >= B1[0] else "Tokens: -", 28, BLACK, True, anchor="lm")
        # tooltips
        if 13.4 <= t < 14.5:
            tooltip(d, "Every frame, every patch", x + 220, y + 250)
        if 14.6 <= t < 16.2:
            tooltip(d, "Ground: 2x2 tokens", x + 420, y + 230, x + 600, y + 420)
        if 16.4 <= t < 17.9:
            tooltip(d, "Sky: 4x4 tokens", x + 300, y + 200, x + 420, y + 120)

        # about
        if t >= T_ABOUT:
            items.append("About LoT")
            active = 2
            x, y, w, h = AB
            s = ease_out((t - T_ABOUT) / 0.15)
            if s < 1:
                zoom_rects(d, (W // 2, H // 2, 2, 2), AB, s)
            else:
                window(canvas, d, AB, "About LoT", active=True)
                icon_lot(d, x + 40, y + 80)
                text(d, (x + 130, y + 76), "Level-of-Token", 44, BLACK, True)
                text(d, (x + 130, y + 128), "Diffusion", 44, BLACK, True)
                d.line([(x + 40, y + 200), (x + w - 40, y + 200)], fill=DGREY, width=2)
                d.line([(x + 40, y + 202), (x + w - 40, y + 202)], fill=WHITE, width=2)
                text(d, (x + 130, y + 226), "Images: 2.4X faster", 32, BLACK)
                text(d, (x + 130, y + 272), "Video:   2.5X faster", 32, BLACK)
                button(d, *ABOUT_OK, "OK", bold=True, focus=True)

    if T_OPEN + ZOOM <= t < T_MIN or T_MIN <= t < B_OPEN + ZOOM or t >= B_OPEN + ZOOM:
        taskbar(canvas, d, t, items, active)
    else:
        taskbar(canvas, d, t, [], -1)

    cx, cyy, kind = cursor_pos(t)
    (cross_cursor if kind == "cross" else arrow_cursor)(d, int(cx), int(cyy))
    return np.asarray(canvas)


# ------------------------------------------------------------------ audio
SR = 48000


def env_ad(n, a=0.002, dec=0.2):
    tt = np.arange(n) / SR
    e = np.exp(-tt / dec)
    na = int(a * SR)
    if na > 0:
        e[:na] *= np.linspace(0, 1, na)
    e[-64:] *= np.linspace(1, 0, 64)
    return e


def sine(f, dur, vol=0.3, dec=0.3, a=0.005):
    n = int(dur * SR)
    tt = np.arange(n) / SR
    return np.sin(2 * np.pi * f * tt) * env_ad(n, a, dec) * vol


def saw(f, dur, vol=0.2, dec=0.6, a=0.01):
    n = int(dur * SR)
    tt = np.arange(n) / SR
    w = 2 * ((f * tt) % 1) - 1
    w = cv2.blur(w[None, :].astype(np.float32), (9, 1))[0]          # soft low-pass
    return w * env_ad(n, a, dec) * vol


def click(vol=0.4, seed=0):
    n = int(0.012 * SR)
    rng = np.random.default_rng(seed)
    w = rng.uniform(-1, 1, n) * np.exp(-np.arange(n) / SR / 0.002)
    return w * vol + sine(1800, 0.012, vol * 0.5, 0.003)


def hdd(dur, seed, rate=28, vol=0.18):
    """hard-drive chatter: random band-limited clicks."""
    n = int(dur * SR)
    out = np.zeros(n)
    rng = np.random.default_rng(seed)
    tk = 0.0
    while tk < dur:
        tk += rng.exponential(1 / rate)
        i = int(tk * SR)
        m = int(0.006 * SR)
        if i + m >= n:
            break
        burst = rng.uniform(-1, 1, m)
        burst = burst - cv2.blur(burst[None, :].astype(np.float32), (7, 1))[0]
        out[i:i + m] += burst * np.exp(-np.arange(m) / SR / 0.0015) * vol * rng.uniform(0.4, 1)
    hum = np.sin(2 * np.pi * 120 * np.arange(n) / SR) * 0.02
    return out + hum


def build_audio(path):
    out = np.zeros(int((DUR + 0.5) * SR))

    def add(sig, t0):
        i = int(t0 * SR)
        j = min(len(out), i + len(sig))
        out[i:j] += sig[:j - i]

    def note(m):
        return 440 * 2 ** ((m - 69) / 12)

    # startup chime: soft rising arpeggio with echo
    for k, m in enumerate((68, 75, 80, 84, 87)):
        for e, g in ((0, 1.0), (0.18, 0.35), (0.36, 0.15)):
            add(sine(note(m), 1.6, 0.16 * g, 0.7, 0.02) + sine(note(m) * 2, 1.0, 0.04 * g, 0.4, 0.02), T_CHIME + k * 0.11 + e)
    for i, tc in enumerate(CLICKS):
        add(click(0.45, i), tc)
    for tc in (T_DLG, T_ABOUT):
        add(sine(1046.5, 0.8, 0.22, 0.25) + sine(1568, 0.8, 0.12, 0.2) + sine(2093, 0.5, 0.05, 0.1), tc)
    for win, sd in ((S1, 1), (S2, 2), (S3, 3), ((R0, R0 + R_FULL), 4), (B1, 5), (B2, 6), (B3, 7)):
        add(hdd(win[1] - win[0], sd), win[0])
    for tc in (T_SP1, T_SP2, T_BSP1, T_BSP2):
        add(sine(note(76), 0.25, 0.18, 0.12), tc)
        add(sine(note(83), 0.4, 0.18, 0.2), tc + 0.09)
    # tada: grace note + brassy chord
    add(saw(note(67), 0.12, 0.18, 0.1), T_DONE - 0.12)
    for m in (60, 64, 67, 72):
        add(saw(note(m), 1.4, 0.13, 0.7) + saw(note(m) * 1.003, 1.4, 0.08, 0.7), T_DONE)
    add(sine(note(84), 0.3, 0.08, 0.15), R0 + R_FULL)          # full model finally done
    # window open whooshes (filtered noise)
    rng = np.random.default_rng(9)
    for tc in (T_OPEN, T_VIEW, T_MIN, B_OPEN):
        n = int(0.25 * SR)
        w = rng.uniform(-1, 1, n)
        w = cv2.blur(w[None, :].astype(np.float32), (25, 1))[0] * np.sin(np.linspace(0, np.pi, n)) * 0.25
        add(w, tc)
    out = out / max(1e-6, np.abs(out).max()) * 0.89
    st = np.stack([out, out], 1)
    wavfile.write(path, SR, (st * 32767).astype(np.int16))


# ------------------------------------------------------------------ main
def render_bytes(t):
    return render(t).tobytes()


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "h"
    if mode == "still":
        for s in sys.argv[2:]:
            Image.fromarray(render(float(s))).save(os.path.join(HERE, f"scratch/still_{TAG}_{float(s):05.2f}.png"))
        return
    fps = 60
    os.makedirs(os.path.join(HERE, "media/final"), exist_ok=True)
    wav = os.path.join(HERE, f"media/final/LoT_retro_desktop_{TAG}.wav")
    build_audio(wav)
    out = os.path.join(HERE, f"media/final/LoT_retro_desktop_{TAG}_1080p60.mp4")
    nf = int(DUR * fps)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(fps),
           "-i", "-", "-i", wav, "-c:v", "libx264", "-preset", "slow", "-crf", "16", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-ac", "2", "-ar", "48000", "-b:a", "192k", "-t", f"{DUR}", "-movflags", "+faststart", out]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    with Pool(10) as pool:
        for k, buf in enumerate(pool.imap(render_bytes, [i / fps for i in range(nf)], chunksize=4)):
            p.stdin.write(buf)
            if k % 240 == 0:
                print(f"frame {k}/{nf}", flush=True)
    p.stdin.close()
    p.wait()
    print("wrote", out)


if __name__ == "__main__":
    main()
