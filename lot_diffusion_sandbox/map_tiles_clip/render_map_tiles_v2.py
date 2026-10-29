"""LoT Diffusion, 20-s web-map clip, v2 (v1 + auto-height place card, larger body text, stronger video seams) (numpy + PIL + cv2 frames → ffmpeg, synthesized UI sound). No manim.

LoT read as map level-of-detail: tokens are map tiles; fine tiles stream in only where the picture is in focus.
  0.0–1.7  zoom-in flourish: the depth map loads through a tile pyramid (1 tile → 4 → 16 … → full res)
  1.8–3.0  depth layer, pin drops on the giraffe: "In focus"
  3.0–6.3  Tiles layer: 4,096 grey placeholders stream in (full detail), load-time bar
  6.3–8.9  Max tile size → 2×2: re-tiled, 1,606 tiles, loads 1.9× faster
  8.9–11.1 Max tile size → 8×8: 1,027 tiles (the real layout), 2.4× faster
 11.1–13.2 Image layer streams into the same 1,027 tiles, borders fade
 13.2–18.6 fly to a live video layer (platformer): 75,600 → 38,046 (2.2×) → 34,698 (2.5×) tiles, playback speed follows
 18.6–20.0 end card
Usage: python render_map_tiles_v2.py [l|h]       → media/final/LoT_map_tiles_v2_<res>.mp4
       python render_map_tiles_v2.py still t ...  → scratch/still_v2_<t>.png
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
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from scipy.io import wavfile
from scipy.signal import butter, lfilter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TAG = "v2"
FONT = os.path.join(HERE, "data/Inter.ttf")
CUE = os.path.join(ROOT, "data/img/img_000000000025_dof_cue.png")
GEN = os.path.join(ROOT, "data/img/img_000000000025_dof_generated_4b.png")
LAYOUT = os.path.join(ROOT, "data/layouts/img_000000000025_dof.json")
VID = os.path.join(ROOT, "data/clips/vd_platformer_gen")
LEVELS = os.path.join(ROOT, "arcade_clip/data/platformer_levels.npy")

W, H = 1920, 1080
DUR = 20.0

# map-app palette
BG = (232, 234, 237)
BG_LINE = (218, 220, 224)
PH = (211, 214, 219)          # placeholder tile
PH_LINE = (240, 241, 243)
INK = (32, 33, 36)
INK2 = (95, 99, 104)
BLUE = (26, 115, 232)
RED = (234, 67, 53)
CARD = (255, 255, 255)
TEAL = (44, 208, 165)         # in-focus band of the turbo depth map

# ------------------------------------------------------------------ timeline (s)
Z0, Z1 = 0.0, 1.7             # zoom flourish
T_PIN = 1.9
T_TILES = 3.0                 # Tiles layer on
D_LOAD = (3.2, 2.4)           # (start, duration) full-detail streaming
T_L1 = 6.3
L1_LOAD = (6.5, 2.4 / 1.9)
T_L2 = 8.9
L2_LOAD = (9.1, 2.4 / 2.4)
T_IMG = 11.1
IMG_LOAD = (11.25, 0.9)
T_FLY = 13.2
T_V0 = 13.9                   # video on screen
V_LOAD = (14.0, 0.6)
T_VB1, T_VB2 = 15.0, 16.8     # re-tile waves (0.5 s)
WAVE = 0.5
T_END = 18.6

IMG_SPEED = (1.0, 1.9, 2.4)
VID_SPEED = (1.0, 2.2, 2.5)
IMG_TOK = (4096, 1606, 1027)
VID_TOK = (75600, 38046, 34698)

# ------------------------------------------------------------------ geometry
N = 64
CELL = 14
S = N * CELL                   # 896
X0, Y0 = (W - S) // 2, 92
VN, VM = 45, 80
VC = 16
VW, VH = VM * VC, VN * VC      # 1280 × 720
VX0, VY0 = 140, 130
PIN_UV = (0.86, 0.55)


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


# ------------------------------------------------------------------ token helpers (from arcade_clip)
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


def up(arr, c):
    return np.repeat(np.repeat(arr, c, 0), c, 1)


# ------------------------------------------------------------------ assets
def load_assets():
    a = {}
    cue = np.asarray(Image.open(CUE).convert("RGB")).astype(np.float32)
    a["cue"] = cv2.resize(cue, (S, S), interpolation=cv2.INTER_AREA)
    a["cue_cell"] = cue.reshape(N, 16, N, 16, 3).mean((1, 3))
    gen = np.asarray(Image.open(GEN).convert("RGB")).astype(np.float32)
    a["gen"] = cv2.resize(gen, (S, S), interpolation=cv2.INTER_AREA)
    rects = [tuple(r) for r in json.load(open(LAYOUT))["rects"]]
    size = np.zeros((N, N), int)
    r0m = np.zeros((N, N), int)
    c0m = np.zeros((N, N), int)
    for r0, r1, c0, c1 in rects:
        size[r0:r1, c0:c1] = r1 - r0
        r0m[r0:r1, c0:c1] = r0
        c0m[r0:r1, c0:c1] = c0
    ii, jj = np.mgrid[0:N, 0:N]
    a["lay"] = {}
    rng = np.random.default_rng(4)
    jit = rng.uniform(0, 1, (N, N))
    for cap in (1, 2, 8):
        b = np.minimum(size, cap)
        oi = r0m + (ii - r0m) // b * b
        oj = c0m + (jj - c0m) // b * b
        ids = oi * N + oj
        # load order: radial from the viewport centre with jitter, one rank per block
        cy, cx = oi + b / 2, oj + b / 2
        dist = np.hypot(cy - N / 2, cx - N / 2) / (N / 2 * 1.42)
        key = dist + 0.25 * jit[oi, oj]
        uniq, first = np.unique(ids.ravel(), return_index=True)
        kb = key.ravel()[first]
        rank = np.empty(len(uniq))
        rank[np.argsort(kb)] = np.arange(len(uniq)) / len(uniq)
        rank_map = rank[np.searchsorted(uniq, ids)]
        # image layer: big tiles first, then fine ones
        sz = b.ravel()[first]
        rank_img = np.empty(len(uniq))
        rank_img[np.lexsort((kb, -sz))] = np.arange(len(uniq)) / len(uniq)
        a["lay"][cap] = dict(ids=ids, b=b, rank=rank_map, rank_u=rank, rank_img=rank_img[np.searchsorted(uniq, ids)],
                             n=len(uniq), mos=mosaic(a["cue_cell"], ids, N * N),
                             edge=edges(up(ids, CELL)), oi=oi, oj=oj)
    # pyramid for the zoom flourish: block sizes 32 … 1 tokens
    a["pyr"] = {}
    for bs in (32, 16, 8, 4, 2, 1):
        ids = (ii // bs) * N + (jj // bs)
        a["pyr"][bs] = (up(mosaic(a["cue_cell"], ids, N * N), CELL), edges(up(ids, CELL)))
    a["lv"] = np.load(LEVELS)
    raw = [np.asarray(Image.open(os.path.join(VID, f"{f + 1:04d}.jpg")).convert("RGB")).astype(np.float32)
           for f in range(81)]
    # loop with an 8-frame crossfade so the seam is soft
    L, X = 73, 8
    vid = []
    for k in range(L):
        if k < X:
            w = (k + 0.5) / X
            vid.append(raw[k] * w + raw[L + k] * (1 - w))
        else:
            vid.append(raw[k])
    a["vid"] = vid
    a["vid_src"] = [k if k >= X or (k + 0.5) / X >= 0.5 else L + k for k in range(L)]
    lut = cv2.applyColorMap(np.arange(256, dtype=np.uint8)[:, None], cv2.COLORMAP_TURBO)[:, 0, ::-1].astype(int)
    a["turbo"] = lut
    a["focus_v"] = int(np.argmin(((lut - np.array(TEAL)) ** 2).sum(1)))
    return a


A = None


def assets():
    global A
    if A is None:
        A = load_assets()
    return A


# ------------------------------------------------------------------ text + UI primitives
@lru_cache(maxsize=256)
def font(size, weight=400):
    f = ImageFont.truetype(FONT, size)
    f.set_variation_by_axes([min(32, max(14, size)), weight])
    return f


def text(d, xy, s, size, color, weight=400, anchor="la"):
    d.text(xy, s, font=font(size, weight), fill=color, anchor=anchor)


def text_w(s, size, weight=400):
    return font(size, weight).getlength(s)


@lru_cache(maxsize=64)
def shadow_img(w, h, r, blur=14, alpha=70):
    pad = blur * 3
    im = Image.new("L", (w + 2 * pad, h + 2 * pad), 0)
    ImageDraw.Draw(im).rounded_rectangle([pad, pad + 4, pad + w, pad + h + 4], r, fill=alpha)
    return im.filter(ImageFilter.GaussianBlur(blur)), pad


def card(canvas, x, y, w, h, r=16, alpha=1.0):
    if alpha <= 0:
        return
    sh, pad = shadow_img(int(w), int(h), r)
    black = Image.new("RGBA", sh.size, (0, 0, 0, 0))
    black.putalpha(sh.point(lambda v: int(v * alpha)))
    canvas.alpha_composite(black, (int(x - pad), int(y - pad)))
    layer = Image.new("RGBA", (int(w), int(h)), (0, 0, 0, 0))
    ImageDraw.Draw(layer).rounded_rectangle([0, 0, w - 1, h - 1], r, fill=CARD + (int(255 * alpha),))
    canvas.alpha_composite(layer, (int(x), int(y)))


def spinner(d, cx, cy, r, t, color=BLUE, width=4):
    a0 = (t * 400) % 360
    span = 90 + 120 * (0.5 + 0.5 * math.sin(t * 6))
    d.arc([cx - r, cy - r, cx + r, cy + r], a0, a0 + span, fill=color, width=width)


def checkbox(d, x, y, on):
    if on:
        d.rounded_rectangle([x, y, x + 26, y + 26], 4, fill=BLUE)
        d.line([(x + 6, y + 13), (x + 11, y + 19), (x + 21, y + 7)], fill=CARD, width=4)
    else:
        d.rounded_rectangle([x, y, x + 26, y + 26], 4, outline=INK2, width=2)


def fmt(n):
    return f"{int(round(n)):,}"


# ------------------------------------------------------------------ map content: image
def stream_alpha(rank, start, dur, t, fade=0.12):
    tl = start + rank * dur
    return np.clip((t - tl) / fade, 0, 1)


def placeholder_layer(edge_px):
    ph = np.empty((S, S, 3), np.float32)
    ph[:] = PH
    ph[edge_px] = PH_LINE
    return ph


def image_content(t):
    """returns (rgb S×S, loaded count, total, edge alpha) for the token / image stages."""
    a = assets()
    if t < T_TILES:
        return a["cue"].copy(), 0, 0, 0.0
    if t < T_L1:
        cap, (st, du) = 1, D_LOAD
    elif t < T_L2:
        cap, (st, du) = 2, L1_LOAD
    else:
        cap, (st, du) = 8, L2_LOAD
    L = a["lay"][cap]
    al = stream_alpha(L["rank"], st, du, t)
    alp = up(al, CELL)[..., None]
    ph = placeholder_layer(L["edge"])
    tok = up(L["mos"], CELL)
    out = ph * (1 - alp) + tok * alp
    if t >= T_IMG:
        ai = up(stream_alpha(L["rank_img"], IMG_LOAD[0], IMG_LOAD[1], t, 0.15), CELL)[..., None]
        out = out * (1 - ai) + a["gen"] * ai
    # tile seams
    e_a = 1.0 if t < IMG_LOAD[0] + IMG_LOAD[1] + 0.1 else 1 - clamp((t - (IMG_LOAD[0] + IMG_LOAD[1] + 0.1)) / 0.5)
    if e_a > 0:
        em = L["edge"][..., None] * (0.75 * e_a)
        out = out * (1 - em) + 255 * em
    loaded = int((t >= st + L["rank_u"] * du).sum())
    return out, loaded, L["n"], e_a


def zoom_content(t):
    """tile pyramid during the zoom flourish: progressively finer mosaics with seams."""
    a = assets()
    p = clamp((t - Z0) / (Z1 - Z0 - 0.15))
    levels = (32, 16, 8, 4, 2, 1)
    k = int(p * (len(levels) + 1))
    if k >= len(levels):
        out = a["cue"].copy()
        if k == len(levels):
            em = a["pyr"][1][1][..., None] * 0.5
            out = out * (1 - em) + 255 * em
        return out
    mos, e = a["pyr"][levels[k]]
    em = e[..., None] * 0.8
    return mos * (1 - em) + 255 * em


# ------------------------------------------------------------------ map content: video
def vid_pos(t):
    def V(tt):
        v = 0.0
        segs = [(T_V0, T_VB1 + WAVE, VID_SPEED[0]), (T_VB1 + WAVE, T_VB2 + WAVE, VID_SPEED[1]),
                (T_VB2 + WAVE, 99, VID_SPEED[2])]
        for s0, s1, sp in segs:
            if tt > s0:
                v += (min(tt, s1) - s0) * sp * 15
        return v
    return int(V(t)) % 73


def vid_speed(t):
    if t < T_VB1 + WAVE:
        return VID_SPEED[0]
    return VID_SPEED[1] if t < T_VB2 + WAVE else VID_SPEED[2]


def vid_blocks(lv, cap):
    b = np.minimum(lv, cap).astype(int)
    ii, jj = np.mgrid[0:VN, 0:VM]
    return ii // b * b, jj // b * b, b


_vrank = None


def vid_jitter():
    global _vrank
    if _vrank is None:
        _vrank = np.random.default_rng(9).uniform(0, 1, (VN, VM))
    return _vrank


def video_content(t):
    """returns (rgb VH×VW, tiles per video shown, wave flash)."""
    a = assets()
    k = vid_pos(t)
    frame = a["vid"][k]
    lv = a["lv"][a["vid_src"][k]]
    ii, jj = np.mgrid[0:VN, 0:VM]
    jit = vid_jitter()
    if t < T_VB1:
        oi, oj, b = vid_blocks(lv, 1)
        ids = oi * VM + oj
        dist = np.hypot(ii + 0.5 - VN / 2, (jj + 0.5 - VM / 2) * 0.56) / 26
        rank = np.clip(0.8 * dist + 0.2 * jit, 0, 1)
        al = np.clip((t - (V_LOAD[0] + rank * V_LOAD[1])) / 0.1, 0, 1)
        alp = up(al, VC)[..., None]
        ph = np.empty_like(frame)
        ph[:] = PH
        e = edges(up(ids, VC))
        ph[e] = PH_LINE
        out = ph * (1 - alp) + frame * alp
        tok = VID_TOK[0]
        flash = None
    else:
        if t < T_VB2:
            old, new, t0, kk = vid_blocks(lv, 1), vid_blocks(lv, 2), T_VB1, 1
        else:
            old, new, t0, kk = vid_blocks(lv, 2), vid_blocks(lv, 4), T_VB2, 2
        cy, cx = new[0] + new[2] / 2, new[1] + new[2] / 2
        dist = np.clip(np.hypot(cy - VN / 2, (cx - VM / 2) * 0.56) / 26, 0, 1)
        t_sw = t0 + WAVE * dist
        sw = t >= t_sw
        oi = np.where(sw, new[0], old[0])
        oj = np.where(sw, new[1], old[1])
        ids = oi * VM + oj
        e = edges(up(ids, VC))
        changed = (new[2] > old[2]) & sw
        fl = np.where(changed, 1 - np.clip((t - t_sw) / 0.22, 0, 1), 0)
        flp = up(fl, VC)[..., None]
        out = frame * (1 - flp) + np.array(PH, np.float32) * flp
        f = clamp((t - t0) / WAVE)
        tok = VID_TOK[kk - 1] + (VID_TOK[kk] - VID_TOK[kk - 1]) * f
    em = e[..., None] * 0.72
    out = out * (1 - em) + 255 * em
    return out, tok


# ------------------------------------------------------------------ chrome
def search_bar(canvas, d, t, alpha):
    if alpha <= 0:
        return
    x, y, w, h = 24, 22, 470, 56
    card(canvas, x, y, w, h, r=28, alpha=alpha)
    d.ellipse([x + 22, y + 16, x + 40, y + 34], outline=INK2, width=3)
    d.line([(x + 38, y + 32), (x + 46, y + 40)], fill=INK2, width=3)
    if t < T_FLY:
        s, t0 = "giraffe", 0.35
    elif t < T_END:
        s, t0 = "platformer", T_FLY + 0.1
    else:
        s, t0 = "Level-of-Token Diffusion", T_END + 0.05
    n = int(clamp((t - t0) / (0.06 * len(s))) * len(s))
    text(d, (x + 62, y + h // 2), s[:n], 24, INK, 400, "lm")
    if int(t * 2.2) % 2 == 0 and n < len(s) + 1:
        cx = x + 62 + text_w(s[:n], 24) + 2
        d.line([(cx, y + 16), (cx, y + h - 16)], fill=BLUE, width=2)


def layers_card(canvas, d, t, video):
    x, y, w = 1452, 22, 444
    h = 370
    card(canvas, x, y, w, h)
    text(d, (x + 28, y + 26), "Layers", 26, INK, 600)
    rows = (("Video", True), ("Tiles", True)) if video else \
        (("Depth", t >= Z1), ("Tiles", t >= T_TILES), ("Image", t >= T_IMG))
    for k, (name, on) in enumerate(rows):
        yy = y + 80 + k * 48
        checkbox(d, x + 28, yy, on)
        text(d, (x + 70, yy + 13), name, 24, INK, 400, "lm")
    d.line([(x + 24, y + 238), (x + w - 24, y + 238)], fill=BG_LINE, width=2)
    text(d, (x + 28, y + 258), "Max tile size", 22, INK2, 500)
    labels = ("1×1", "2×2", "4×4") if video else ("1×1", "2×2", "4×4", "8×8")
    tx0, tx1, ty = x + 52, x + w - 52, y + 316
    xs = [tx0 + (tx1 - tx0) * k / (len(labels) - 1) for k in range(len(labels))]
    if video:
        pos = 0 + ease((t - T_VB1) / 0.3) + ease((t - T_VB2) / 0.3)
    else:
        pos = ease((t - T_L1) / 0.3) * 1 + ease((t - T_L2) / 0.3) * 2
    kx = tx0 + (tx1 - tx0) * pos / (len(labels) - 1)
    d.line([(tx0, ty), (tx1, ty)], fill=BG_LINE, width=6)
    d.line([(tx0, ty), (kx, ty)], fill=BLUE, width=6)
    for k, xx in enumerate(xs):
        d.ellipse([xx - 4, ty - 4, xx + 4, ty + 4], fill=BLUE if xx <= kx + 1 else BG_LINE)
        text(d, (xx, ty + 26), labels[k], 18, INK2, 500, "mm")
    d.ellipse([kx - 13, ty - 13, kx + 13, ty + 13], fill=BLUE, outline=CARD, width=3)


def zoom_buttons(canvas, d):
    x, y = 1840, 870
    card(canvas, x, y, 56, 112, r=10)
    d.line([(x + 18, y + 28), (x + 38, y + 28)], fill=INK2, width=3)
    d.line([(x + 28, y + 18), (x + 28, y + 38)], fill=INK2, width=3)
    d.line([(x + 10, y + 56), (x + 46, y + 56)], fill=BG_LINE, width=2)
    d.line([(x + 18, y + 84), (x + 38, y + 84)], fill=INK2, width=3)


def scale_bar(d, px, label):
    x, y = 32, 1040
    d.line([(x, y), (x + px, y)], fill=INK, width=3)
    d.line([(x, y - 8), (x, y)], fill=INK, width=3)
    d.line([(x + px, y - 8), (x + px, y)], fill=INK, width=3)
    text(d, (x + px + 10, y - 2), label, 18, INK, 500, "lb")


def status_pill(canvas, d, t, msg, busy, y=1000):
    if not msg:
        return
    w = int(text_w(msg, 24, 500)) + (100 if busy else 56)
    x = (W - w) // 2
    card(canvas, x, y, w, 52, r=26)
    tx = x + 28
    if busy:
        spinner(d, x + 36, y + 26, 11, t, width=4)
        tx = x + 64
    text(d, (tx, y + 26), msg, 24, INK, 500, "lm")


def bars(d, x, y, w, frac_full, frac_lod, show_lod, lod_done_label):
    """load-time comparison: full detail (grey) vs level of detail (blue)."""
    text(d, (x, y), "Load time", 20, INK2, 500)
    text(d, (x, y + 34), "Full detail", 18, INK2, 400)
    d.rounded_rectangle([x, y + 60, x + w, y + 76], 8, fill=(241, 243, 244))
    if frac_full > 0:
        d.rounded_rectangle([x, y + 60, x + max(16, w * frac_full), y + 76], 8, fill=(154, 160, 166))
    if show_lod:
        text(d, (x, y + 92), "Level of detail", 18, INK2, 400)
        d.rounded_rectangle([x, y + 118, x + w, y + 134], 8, fill=(241, 243, 244))
        if frac_lod > 0:
            d.rounded_rectangle([x, y + 118, x + max(16, w * frac_lod), y + 134], 8, fill=BLUE)
        if lod_done_label:
            text(d, (x, y + 160), lod_done_label, 44, BLUE, 700)


def colorbar_card(canvas, d, x, y, w, alpha):
    a = assets()
    bw, bh = w, 22
    lut = a["turbo"]
    col = cv2.resize(lut[None, :, :].astype(np.uint8), (bw, bh), interpolation=cv2.INTER_LINEAR)
    im = Image.fromarray(col).convert("RGBA")
    canvas.alpha_composite(im, (x, y))
    text(d, (x, y + bh + 10), "Far", 18, INK2, 500, "la")
    text(d, (x + bw, y + bh + 10), "Near", 18, INK2, 500, "ra")
    fx = x + int(a["focus_v"] / 255 * bw)
    d.polygon([(fx, y - 4), (fx - 9, y - 18), (fx + 9, y - 18)], fill=INK)
    text(d, (fx, y + bh + 10), "In focus", 18, INK, 700, "ma")


def place_card(canvas, d, t, info):
    if info is None:
        return
    title, body, kind = info
    x, y, w = 24, 104, 464
    h = 82 + 36 * len(body) + 28
    if kind == "depth":
        h += 96
    elif kind[0] == "tiles":
        h += 92 + (134 + 76 if kind[5] else 80)
    card(canvas, x, y, w, h)
    text(d, (x + 28, y + 28), title, 32, INK, 700)
    yy = y + 82
    for line in body:
        text(d, (x + 28, yy), line, 24, INK2, 400)
        yy += 36
    if kind == "depth":
        colorbar_card(canvas, d, x + 28, yy + 30, w - 56, 1)
    elif kind[0] == "tiles":
        _, total, loaded, fr_full, fr_lod, show_lod, lab = kind
        text(d, (x + 28, yy + 6), fmt(total), 56, INK, 700)
        text(d, (x + 28 + text_w(fmt(total), 56, 700) + 14, yy + 40), "tiles", 24, INK2, 500)
        bars(d, x + 28, yy + 92, w - 56, fr_full, fr_lod, show_lod, lab)


def pin(canvas, d, t):
    dt = t - T_PIN
    if dt < 0:
        return
    px = X0 + int(PIN_UV[0] * S)
    py = Y0 + int(PIN_UV[1] * S)
    drop = (1 - ease_out(dt / 0.35)) * 120
    bounce = 0 if dt < 0.35 else -10 * math.exp(-(dt - 0.35) / 0.08) * abs(math.sin((dt - 0.35) * 30))
    yy = py - drop + bounce
    sh_a = int(90 * clamp(dt / 0.35))
    d.ellipse([px - 12, py - 4, px + 12, py + 4], fill=(0, 0, 0, sh_a))
    r = 20
    cy = yy - 44
    d.polygon([(px, yy), (px - r * 0.86, cy + r * 0.5), (px + r * 0.86, cy + r * 0.5)], fill=RED)
    d.ellipse([px - r, cy - r, px + r, cy + r], fill=RED)
    d.ellipse([px - 7, cy - 7, px + 7, cy + 7], fill=(165, 14, 14))
    if dt > 0.3:
        al = clamp((dt - 0.3) / 0.2)
        lw = int(text_w("In focus", 24, 600)) + 36
        lx, ly = px - lw - 30, int(cy) - 22
        card(canvas, lx, ly, lw, 44, r=22, alpha=al)
        text(d, (lx + 18, ly + 22), "In focus", 24, INK, 600, "lm")


# ------------------------------------------------------------------ frame
def map_bg(canvas_np, scale, cx, cy):
    canvas_np[:] = BG
    step = 256 * scale
    if step < 16:
        return
    ox = cx % step
    oy = cy % step
    for x in np.arange(ox, W, step):
        canvas_np[:, int(x):int(x) + 2] = BG_LINE
    for y in np.arange(oy, H, step):
        canvas_np[int(y):int(y) + 2, :] = BG_LINE


def paste_np(dst, src, x, y):
    h, w = src.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return
    dst[y0:y1, x0:x1] = src[y0 - y:y1 - y, x0 - x:x1 - x]


def place_info(t):
    if T_PIN <= t < T_TILES:
        return "Depth layer", ["Teal = the part in focus"], "depth"
    if T_TILES <= t < T_L1:
        st, du = D_LOAD
        fr = clamp((t - st) / du)
        return "Full detail", ["Same small tile everywhere"], ("tiles", 4096, 0, fr, 0, False, None)
    if T_L1 <= t < T_L2:
        st, du = L1_LOAD
        fr = clamp((t - st) / D_LOAD[1])
        lab = "1.9× faster" if t >= st + du + 0.1 else None
        return "Level of detail", ["Bigger tiles where it's blurry"], ("tiles", 1606, 0, 1, fr, True, lab)
    if T_L2 <= t < T_IMG:
        st, du = L2_LOAD
        fr = clamp((t - st) / D_LOAD[1])
        lab = "2.4× faster" if t >= st + du + 0.1 else None
        return "Even bigger tiles", ["Up to 8×8 in the blur"], ("tiles", 1027, 0, 1, fr, True, lab)
    if T_IMG <= t < T_FLY:
        return "Image layer", ["Sharp where it's in focus", "4× fewer tiles"], \
            ("tiles", 1027, 0, 1, 1 / 2.4, True, "2.4× faster")
    return None


def status_msg(t, loaded, total):
    if t < Z1:
        return ("Loading map…", True)
    if t < T_TILES:
        return ("Depth layer", False)
    if t < T_L1:
        if t < D_LOAD[0] + D_LOAD[1] + 0.05:
            return (f"Loading tiles  {fmt(loaded)} / {fmt(total)}", True)
        return (f"{fmt(total)} tiles loaded", False)
    if t < T_L2:
        if t < L1_LOAD[0] + L1_LOAD[1] + 0.05:
            return (f"Re-tiling by focus  {fmt(loaded)} / {fmt(total)}", True)
        return ("Out of focus → bigger tiles", False)
    if t < T_IMG:
        if t < L2_LOAD[0] + L2_LOAD[1] + 0.05:
            return (f"Loading tiles  {fmt(loaded)} / {fmt(total)}", True)
        return ("In focus stays fine", False)
    if t < T_FLY:
        if t < IMG_LOAD[0] + IMG_LOAD[1] + 0.1:
            return ("Loading image…", True)
        return ("Same picture, 4× fewer tiles", False)
    return None


def video_strip(canvas, d, t, tok):
    x, y, w, h = VX0, VY0 + VH + 20, VW, 130
    card(canvas, x, y, w, h)
    text(d, (x + 28, y + 22), "Tiles per video", 20, INK2, 500)
    text(d, (x + 28, y + 50), fmt(tok), 52, INK, 700)
    bx = x + 380
    if t < T_VB1:
        frl = 0
    elif t < T_VB2:
        frl = 1 / VID_SPEED[1] * ease((t - T_VB1) / 0.5)
    else:
        frl = 1 / VID_SPEED[1] + (1 / VID_SPEED[2] - 1 / VID_SPEED[1]) * ease((t - T_VB2) / 0.5)
    text(d, (bx, y + 22), "Load time", 20, INK2, 500)
    d.rounded_rectangle([bx, y + 60, bx + 420, y + 74], 7, fill=(154, 160, 166))
    text(d, (bx + 430, y + 67), "Full", 18, INK2, 400, "lm")
    d.rounded_rectangle([bx, y + 92, bx + 420, y + 106], 7, fill=(241, 243, 244))
    if frl > 0:
        d.rounded_rectangle([bx, y + 92, bx + max(14, 420 * frl), y + 106], 7, fill=BLUE)
    text(d, (bx + 430, y + 99), "LoD", 18, INK2, 400, "lm")
    sp = vid_speed(t)
    rx = x + w - 28
    text(d, (rx, y + 22), "Playback", 20, INK2, 500, "ra")
    col = BLUE if sp > 1 else INK
    text(d, (rx, y + 50), f"{sp:.1f}×", 52, col, 700, "ra")
    # live dot
    if int(t * 2) % 2 == 0:
        d.ellipse([VX0 + 20, VY0 + 20, VX0 + 36, VY0 + 36], fill=RED)
    text(d, (VX0 + 46, VY0 + 28), "LIVE", 20, CARD, 700, "lm")


def video_msg(t):
    if t < T_V0 + 0.1:
        return ("Opening live layer…", True)
    if t < T_VB1:
        return ("Every frame, full detail", False)
    if t < T_VB2:
        return ("Ground → 2×2 tiles · 2.2× faster", False) if t > T_VB1 + 0.5 else ("Re-tiling…", True)
    if t < T_END:
        return ("Sky → 4×4 tiles · 2.5× faster", False) if t > T_VB2 + 0.5 else ("Re-tiling…", True)
    return None


def render(t):
    a = assets()
    base = np.zeros((H, W, 3), np.float32)
    loaded, total = 0, 0
    # -------- map content
    if t < T_FLY + 0.7:
        if t < Z1:
            p = ease(clamp((t - Z0) / Z1))
            scale = 0.16 + 0.84 * p
            cont = zoom_content(t)
        else:
            scale = 1.0
            cont, loaded, total, _ = image_content(t)
        fly = clamp((t - T_FLY) / 0.7)
        if fly > 0:
            scale = 1 - 0.75 * ease(fly)
        cx, cy = X0 + S / 2 - 900 * ease(fly), Y0 + S / 2
        map_bg(base, scale if fly == 0 else 1 - 0.75 * ease(fly), cx, cy)
        sz = max(2, int(S * scale))
        im = cv2.resize(cont, (sz, sz), interpolation=cv2.INTER_AREA if sz < S else cv2.INTER_NEAREST)
        paste_np(base, im, int(cx - sz / 2), int(cy - sz / 2))
    vtok = VID_TOK[0]
    if t >= T_FLY:
        fly = ease(clamp((t - T_FLY) / 0.7))
        if t >= T_FLY + 0.7:
            map_bg(base, 1.0, VX0 + VW / 2, VY0 + VH / 2)
        vimg, vtok = video_content(t) if t >= T_V0 else (None, VID_TOK[0])
        if vimg is None:
            vimg = np.empty((VH, VW, 3), np.float32)
            vimg[:] = PH
        vs = 0.3 + 0.7 * fly
        vw, vh = int(VW * vs), int(VH * vs)
        vcx = VX0 + VW / 2 + 900 * (1 - fly)
        vim = cv2.resize(vimg, (vw, vh), interpolation=cv2.INTER_AREA) if vs < 0.999 else vimg
        paste_np(base, vim, int(vcx - vw / 2), int(VY0 + VH / 2 - vh / 2))
    canvas = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8)).convert("RGBA")
    d = ImageDraw.Draw(canvas, "RGBA")

    # -------- chrome
    ui_a = clamp((t - 0.15) / 0.35)
    search_bar(canvas, d, t, ui_a)
    if t < T_FLY:
        if t >= 0.4:
            layers_card(canvas, d, t, False)
        if Z1 <= t:
            pin(canvas, d, t) if t < T_L1 else None
        place_card(canvas, d, t, place_info(t))
        msg = status_msg(t, loaded, total)
        if msg:
            status_pill(canvas, d, t, msg[0], msg[1])
        scale_bar(d, 56, "64 px")
    else:
        if t >= T_FLY + 0.35:
            layers_card(canvas, d, t, True)
        if t >= T_V0:
            video_strip(canvas, d, t, vtok)
        m = video_msg(t)
        if m:
            status_pill(canvas, d, t, m[0], m[1], y=1014)
        scale_bar(d, 64, "64 px")
    zoom_buttons(canvas, d)

    # -------- end card
    if t >= T_END:
        dt = t - T_END
        al = clamp(dt / 0.3)
        canvas.alpha_composite(Image.new("RGBA", (W, H), (255, 255, 255, int(150 * al))))
        s = back_out(dt / 0.35)
        cw, ch = int(1000 * s), int(300 * s)
        if cw > 10:
            card(canvas, (W - cw) // 2, (H - ch) // 2, cw, ch, r=24)
        if dt > 0.25:
            text(d, (W // 2, H // 2 - 70), "Level-of-Token Diffusion", 60, INK, 700, "mm")
            text(d, (W // 2, H // 2 + 6), "Fine tiles only where detail matters", 30, INK2, 400, "mm")
        if dt > 0.55:
            text(d, (W // 2, H // 2 + 76), "2.4× faster images  ·  2.5× faster video", 32, BLUE, 700, "mm")
    return np.asarray(canvas.convert("RGB"))


# ------------------------------------------------------------------ audio
SR = 48000


def env_exp(n, decay):
    return np.exp(-np.arange(n) / SR / decay)


def sine(f, dur, vol, decay=None, attack=0.004):
    n = int(dur * SR)
    tt = np.arange(n) / SR
    w = np.sin(2 * np.pi * f * tt)
    e = env_exp(n, decay) if decay else np.ones(n)
    na = max(1, int(attack * SR))
    e[:na] *= np.linspace(0, 1, na)
    e[-min(n, 200):] *= np.linspace(1, 0, min(n, 200))
    return w * e * vol


def click(vol=0.25, f=2400):
    n = int(0.03 * SR)
    rng = np.random.default_rng(1)
    w = np.sin(2 * np.pi * f * np.arange(n) / SR) * env_exp(n, 0.006) + rng.uniform(-1, 1, n) * env_exp(n, 0.002) * 0.4
    return w * vol


def whoosh(dur, vol, seed=0, rising=True):
    n = int(dur * SR)
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n)
    out = np.zeros(n)
    seg = 1200
    for k in range(0, n, seg):
        p = k / n
        fc = 300 + 3500 * (p if rising else 1 - p)
        b, a = butter(2, [max(60, fc * 0.5) / (SR / 2), min(0.95, fc * 1.6 / (SR / 2))], btype="band")
        out[k:k + seg] = lfilter(b, a, x[max(0, k - 2000):k + seg])[-len(out[k:k + seg]):]
    e = np.sin(np.pi * np.linspace(0, 1, n)) ** 1.5
    return out / (np.abs(out).max() + 1e-9) * e * vol


def build_audio(path):
    n = int((DUR + 0.5) * SR)
    out = np.zeros((n, 2))

    def add(sig, t0, pan=0.0):
        i = int(t0 * SR)
        j = min(n, i + len(sig))
        if j <= i:
            return
        l, r = math.cos((pan + 1) * math.pi / 4), math.sin((pan + 1) * math.pi / 4)
        out[i:j, 0] += sig[:j - i] * l
        out[i:j, 1] += sig[:j - i] * r

    rng = np.random.default_rng(12)
    # ambient pad: Fmaj9, slow swell
    tt = np.arange(n) / SR
    pad = np.zeros(n)
    for k, f in enumerate((174.6, 220.0, 261.6, 329.6, 392.0)):
        pad += np.sin(2 * np.pi * f * tt + k) * (0.5 + 0.5 * np.sin(2 * np.pi * (0.07 + 0.03 * k) * tt + k))
    b, a = butter(2, 1200 / (SR / 2))
    pad = lfilter(b, a, pad)
    fade = np.clip(tt / 1.5, 0, 1) * np.clip((DUR + 0.2 - tt) / 1.2, 0, 1)
    pad = pad / np.abs(pad).max() * 0.07 * fade
    out[:, 0] += pad
    out[:, 1] += np.roll(pad, 400)
    # zoom whoosh + fly whoosh
    add(whoosh(1.6, 0.35, 1), 0.05)
    add(whoosh(0.9, 0.3, 2), T_FLY - 0.05, pan=0.3)
    # clicks on UI changes
    for tc in (T_TILES, T_L1, T_L2, T_IMG, T_VB1, T_VB2):
        add(click(0.3), tc - 0.02)
    # pin drop
    add(sine(660, 0.25, 0.15, 0.08), T_PIN + 0.35)
    add(sine(990, 0.3, 0.1, 0.1), T_PIN + 0.4)
    # tile-loading patter: one soft tick per loaded tile (thinned to ≤ 90/s)
    def patter(start, dur, ntiles, seed):
        r = np.random.default_rng(seed)
        k = int(min(ntiles, dur * 90))
        times = np.sort(start + r.uniform(0, dur, k))
        for tk in times:
            f = r.uniform(2500, 5200)
            add(sine(f, 0.012, r.uniform(0.02, 0.05), 0.003, attack=0.0005), tk, pan=r.uniform(-0.8, 0.8))
    patter(D_LOAD[0], D_LOAD[1], 4096, 1)
    patter(L1_LOAD[0], L1_LOAD[1], 1606, 2)
    patter(L2_LOAD[0], L2_LOAD[1], 1027, 3)
    patter(IMG_LOAD[0], IMG_LOAD[1], 1027, 4)
    patter(V_LOAD[0], V_LOAD[1], 3600, 5)
    patter(T_VB1, WAVE, 900, 6)
    patter(T_VB2, WAVE, 900, 7)
    # success chimes (load finished faster)
    for t0, notes in ((L1_LOAD[0] + L1_LOAD[1] + 0.1, (659.3, 987.8)), (L2_LOAD[0] + L2_LOAD[1] + 0.1, (659.3, 987.8, 1318.5)),
                      (T_VB1 + WAVE, (659.3, 987.8)), (T_VB2 + WAVE, (659.3, 987.8, 1318.5))):
        for k, f in enumerate(notes):
            add(sine(f, 0.7, 0.13, 0.25), t0 + k * 0.08)
            add(sine(f * 2, 0.4, 0.03, 0.12), t0 + k * 0.08)
    # image done chime
    for k, f in enumerate((523.3, 659.3, 784.0)):
        add(sine(f, 0.8, 0.1, 0.3), IMG_LOAD[0] + IMG_LOAD[1] + 0.1 + k * 0.07)
    # end chord
    for k, f in enumerate((349.2, 440.0, 523.3, 659.3, 784.0)):
        add(sine(f, 1.4, 0.09, 0.6), T_END + 0.25 + k * 0.06)
    peak = np.abs(out).max()
    out = out / peak * 0.85
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
    wav = os.path.join(HERE, f"media/final/LoT_map_tiles_{TAG}.wav")
    build_audio(wav)
    out = os.path.join(HERE, f"media/final/LoT_map_tiles_{TAG}_{res}.mp4")
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
