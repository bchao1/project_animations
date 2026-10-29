"""LoT Diffusion, 20-s F1 race-broadcast clip, v1 (numpy + PIL + cv2 frames → ffmpeg, synthesized audio). No manim.

  0.0–1.4   title: LEVEL-OF-TOKEN GRAND PRIX
  1.4–7.0   SETUP (garage): both cars get the giraffe scene; every patch = 1 token (4,096 each);
            LoT merges out-of-focus tokens 2 × 2 (1,606, sector 1.9X green) then 4 × 4 / 8 × 8 (1,027, 2.4X purple)
  7.0–8.6   five red lights … lights out
  8.6–12.2  RACE: 10 laps = 10 compute sweeps; LoT (1,027 tokens) finishes 2.4× sooner → chequered flag,
            the generated giraffe, FASTEST LAP; FULL TOKENS finishes +3.37 s
 13.0–18.5  RACE REPLAY: platformer video left (playback rate = current speedup), its tokens right;
            ground merges 2 × 2 (2.2X), sky 4 × 4 (2.5X)
 18.5–20.0  podium card

Times: full-res FLUX.2 4B reference 5.78 s per image; LoT 5.78 / 2.4 = 2.41 s (broadcast clock runs 1.6× real time).
Usage: python render_f1_race_v1.py [l|h]       → media/final/LoT_f1_race_v1_<res>.mp4
       python render_f1_race_v1.py still t ...  → scratch/still_v1_<t>.png (1080p)
"""
import json
import math
import os
import subprocess
import sys
from multiprocessing import Pool

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.io import wavfile
from scipy.signal import butter, lfilter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TAG = "v1"
CUE = os.path.join(ROOT, "data/img/img_000000000025_dof_cue.png")
GEN = os.path.join(ROOT, "data/img/img_000000000025_dof_generated_4b.png")
LAYOUT = os.path.join(ROOT, "data/layouts/img_000000000025_dof.json")
VID = os.path.join(ROOT, "data/clips/vd_platformer_gen")
LEVELS = os.path.join(ROOT, "arcade_clip/data/platformer_levels.npy")
FD = os.path.join(HERE, "data")
F_XB = os.path.join(FD, "BarlowCondensed-ExtraBoldItalic.ttf")
F_BI = os.path.join(FD, "BarlowCondensed-BoldItalic.ttf")
F_B = os.path.join(FD, "BarlowCondensed-Bold.ttf")
F_SB = os.path.join(FD, "BarlowCondensed-SemiBold.ttf")
F_TB = os.path.join(FD, "TitilliumWeb-Bold.ttf")
F_TK = os.path.join(FD, "TitilliumWeb-Black.ttf")

W, H = 1920, 1080
DUR = 20.0

RED = (225, 6, 0)
RED_D = (130, 0, 0)
WHITE = (255, 255, 255)
OFFW = (235, 236, 238)
SILVER = (160, 168, 178)
PURPLE = (168, 52, 235)
GREEN = (0, 210, 95)
YELLOW = (255, 214, 0)
INK = (18, 20, 24)
PANEL = (14, 16, 20)
GRID = (12, 14, 18)

# ------------------------------------------------------------------ timeline (s)
T_SETUP = 1.4
S1 = (2.2, 3.3)              # 1 × 1 sweep, both cars
S2 = (4.3, 5.2)              # LoT merge to 2 × 2
S3 = (5.7, 6.5)              # LoT merge to the real layout
T_SEC1, T_SEC2 = 5.25, 6.55  # sector badges 1.9X / 2.4X
LIGHTS = [7.0, 7.3, 7.6, 7.9, 8.2]
R0 = 8.6                     # lights out
CLOCK = 1.6                  # real seconds per broadcast second
T_FULL_REAL = 5.78
T_LOT_REAL = 5.78 / 2.4
F_LOT = R0 + T_LOT_REAL / CLOCK          # 10.105
F_FULL = R0 + T_FULL_REAL / CLOCK        # 12.2125
T_SWAP = 8.85
LAPS = 10
T_STING = 12.95
T_RP = 13.35                 # replay layout on
B1 = (13.55, 14.25)
B2 = (14.6, 15.3)
B3 = (16.3, 17.0)
T_BSP1, T_BSP2 = 15.35, 17.05
T_END = 18.5

IMG_TOK = (4096, 1606, 1027)
VID_TOK = (75600, 38046, 34698)
VID_SPEED = (1.0, 2.2, 2.5)

# ------------------------------------------------------------------ geometry
P = 512                      # car panels, 64 tokens × 8 px
N = 64
FX0, LX0, PY0 = 480, 1040, 196
VW, VH = 880, 495            # replay panels, 80 × 45 tokens × 11 px
VX0L, VX0R, VY0 = 60, 980, 196
VN, VM = 45, 80
TRACK_Y = 790
TRACK_X0, TRACK_X1 = 480, 1860


def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def ease_out(x):
    x = clamp(x)
    return 1 - (1 - x) ** 3


def ease_io(x):
    x = clamp(x)
    return x * x * (3 - 2 * x)


# ------------------------------------------------------------------ assets
def carbon():
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    c = 10
    a = ((xx // c + yy // c) % 2)
    sheen = np.where(a > 0, (xx % c) / c, (yy % c) / c)
    tex = 16 + 5 * a + 6 * np.sin(sheen * np.pi)
    glow = 1 + 0.55 * np.exp(-(((xx - W * 0.55) / (W * 0.6)) ** 2 + ((yy - H * 0.2) / (H * 0.7)) ** 2))
    v = tex * glow
    img = np.stack([v * 0.95, v * 0.98, v * 1.08], -1)
    return np.clip(img, 0, 255).astype(np.uint8)


def load_assets():
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
    ii, jj = np.mgrid[0:N, 0:N]
    a["img_blocks"] = {}
    for cap in (1, 2, 8):
        b = np.minimum(size, cap)
        a["img_blocks"][cap] = (r0m + (ii - r0m) // b * b, c0m + (jj - c0m) // b * b, b)
    a["lv"] = np.load(LEVELS)
    a["vid"] = [cv2.resize(np.asarray(Image.open(os.path.join(VID, f"{f + 1:04d}.jpg")).convert("RGB")),
                           (VW, VH), interpolation=cv2.INTER_AREA).astype(np.float32) for f in range(81)]
    a["vid_cell"] = [cv2.resize(v, (VM, VN), interpolation=cv2.INTER_AREA) for v in a["vid"]]
    a["bg"] = carbon()
    return a


A = None


def assets():
    global A
    if A is None:
        A = load_assets()
    return A


_FONTS = {}


def font(path, size):
    k = (path, size)
    if k not in _FONTS:
        _FONTS[k] = ImageFont.truetype(path, size)
    return _FONTS[k]


def text(d, xy, s, path, size, fill, anchor="la"):
    d.text(xy, s, font=font(path, size), fill=fill, anchor=anchor)


def tw(s, path, size):
    return font(path, size).getlength(s)


def para(d, x, y, w, h, fill, slant=18):
    """slanted broadcast box (parallelogram leaning right)."""
    d.polygon([(x + slant, y), (x + w + slant, y), (x + w, y + h), (x, y + h)], fill=fill)


# ------------------------------------------------------------------ token mosaics (from arcade_clip)
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


def staged_ids(old, new, n, m, t0, dur, t):
    oi_o, oj_o, _ = old
    oi_n, oj_n, b_n = new
    t_sw = t0 + (oi_n + b_n) / n * dur
    use = t >= t_sw
    return block_ids(np.where(use, oi_n, oi_o), np.where(use, oj_n, oj_o), m), t_sw


def up(arr, px):
    return np.repeat(np.repeat(arr, px, 0), px, 1)


def frac(t, win):
    return clamp((t - win[0]) / (win[1] - win[0]))


def lines_over(img, e, col, alpha):
    em = e[..., None].astype(np.float32) * alpha
    return img * (1 - em) + np.array(col, np.float32) * em


# ------------------------------------------------------------------ car panels
def lot_ids(t):
    """LoT token ids (64 × 64) at time t + per-cell merge flash."""
    B = assets()["img_blocks"]
    if t < S2[0]:
        return block_ids(B[1][0], B[1][1], N), np.zeros((N, N), np.float32), 1
    if t < S3[0]:
        old, new, win = B[1], B[2], S2
    else:
        old, new, win = B[2], B[8], S3
    ids, t_sw = staged_ids(old, new, N, N, win[0], win[1] - win[0], t)
    dt = t - t_sw
    fl = np.where((new[2] > old[2]) & (dt >= 0), np.exp(-np.maximum(dt, 0) / 0.12), 0).astype(np.float32)
    return ids, fl, 2


def progress(t, t_real):
    return clamp((t - R0) * CLOCK / t_real)


def compute_band(p):
    """row band (in tokens) of the current compute sweep within a lap."""
    if p <= 0 or p >= 1:
        return None
    return (p * LAPS % 1) * N


def chequer(w, h, sq, phase=0):
    yy, xx = np.mgrid[0:h, 0:w]
    return (((xx + phase) // sq + yy // sq) % 2).astype(bool)


def car_panel(t, car):
    """returns P × P float image for car 'full' or 'lot'."""
    a = assets()
    B = a["img_blocks"]
    cell_px = P // N
    if t < S1[0]:
        return a["cue"].copy()
    if t < S1[1] + 0.05:
        f = frac(t, S1)
        rows = int(f * N + 1e-6)
        ids = block_ids(B[1][0], B[1][1], N)
        mos = up(mosaic(a["cue_cell"], ids, N * N), cell_px)
        img = a["cue"].copy()
        img[:rows * cell_px] = mos[:rows * cell_px]
        e = edges(up(ids, cell_px))
        e[rows * cell_px:] = False
        img = lines_over(img, e, GRID, 0.55)
        if 0 < f < 1:
            y = rows * cell_px
            img[max(0, y - 3):y + 3] = 255
        return img
    if car == "full":
        ids, fl = block_ids(B[1][0], B[1][1], N), np.zeros((N, N), np.float32)
        p, t_fin = progress(t, T_FULL_REAL), F_FULL
    else:
        ids, fl, _ = lot_ids(t)
        p, t_fin = progress(t, T_LOT_REAL), F_LOT
    mos = mosaic(a["cue_cell"], ids, N * N)
    band = compute_band(p)
    if band is not None:
        rows = np.arange(N)[:, None]
        g = np.exp(-((rows - band) / 2.5) ** 2)
        mos = mos + g[..., None] * (255 - mos) * 0.55
    mos = mos + fl[..., None] * (255 - mos) * 0.85
    img = up(mos, cell_px)
    e = edges(up(ids, cell_px))
    img = lines_over(img, e, GRID, 0.55)
    if t >= t_fin:
        dt = t - t_fin
        if car == "lot":
            # tokens flip to the generated image, then lines fade
            rng = np.random.default_rng(3)
            order = rng.uniform(0, 0.35, N * N)[ids]
            shown = up(dt >= 0.15 + order, cell_px)
            gen = lines_over(a["gen"], e, GRID, 0.55 * (1 - clamp((dt - 0.6) / 0.4)))
            img[shown] = gen[shown]
        else:
            img = img * 0.45
        # chequered flag sweeps across, 0.35 s
        k = clamp(dt / 0.35)
        if k < 1:
            x = int(P * (k * 1.4 - 0.2))
            ch = chequer(P, P, 32, phase=int(dt * 400))
            m = np.zeros((P, P), bool)
            m[:, max(0, x - 140):max(0, x)] = True
            img[m & ch] = 245
            img[m & ~ch] = 15
    return img


# ------------------------------------------------------------------ replay panels
def vid_V(tt):
    v = 0.0
    segs = [(T_RP, T_BSP1, VID_SPEED[0]), (T_BSP1, T_BSP2, VID_SPEED[1]), (T_BSP2, 99, VID_SPEED[2])]
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


def replay_panels(t):
    a = assets()
    v = vid_V(t)
    idx = int(v) % 81
    wrapped = int(v) // 81 != int(vid_V(t - 0.06)) // 81
    frame, cell, lv = a["vid"][idx], a["vid_cell"][idx], a["lv"][idx]
    left = frame.copy()
    right = frame * 0.4
    tok = None
    if t >= B1[0]:
        if t < B2[0]:
            f = frac(t, B1)
            rows = int(f * VN + 1e-6)
            ids = block_ids(*vid_blocks(lv, 1)[:2], VM)
            mos = up(mosaic(cell, ids, VN * VM), 11)
            right = right.copy()
            right[:rows * 11] = mos[:rows * 11]
            e = edges(up(ids, 11))
            e[rows * 11:] = False
            right = lines_over(right, e, GRID, 0.5)
            if 0 < f < 1:
                right[max(0, rows * 11 - 3):rows * 11 + 3] = 255
            tok = VID_TOK[0]
        else:
            if t < B3[0]:
                old, new, win, k = vid_blocks(lv, 1), vid_blocks(lv, 2), B2, 1
            else:
                old, new, win, k = vid_blocks(lv, 2), vid_blocks(lv, 4), B3, 2
            f = frac(t, win)
            ids, t_sw = staged_ids(old, new, VN, VM, win[0], win[1] - win[0], t)
            dt = t - t_sw
            fl = np.where((new[2] > old[2]) & (dt >= 0), np.exp(-np.maximum(dt, 0) / 0.12), 0).astype(np.float32)
            mos = mosaic(cell, ids, VN * VM)
            mos = mos + fl[..., None] * (255 - mos) * 0.85
            right = lines_over(up(mos, 11), edges(up(ids, 11)), GRID, 0.5)
            tok = VID_TOK[k - 1] + (VID_TOK[k] - VID_TOK[k - 1]) * ease_io(f)
    if wrapped:
        rng = np.random.default_rng(int(t * 1000))
        left = left * 0.6 + rng.uniform(0, 255, left.shape).astype(np.float32) * 0.4
    return left, right, tok


# ------------------------------------------------------------------ broadcast graphics
def lot_tokens(t):
    if t < S1[0]:
        return 0
    if t < S2[0]:
        return int(frac(t, S1) * N + 1e-6) * N
    ids, _, _ = lot_ids(t)
    return len(np.unique(ids))


def full_tokens(t):
    if t < S1[0]:
        return 0
    return int(frac(t, S1) * N + 1e-6) * N


def lower_third(d, ov, t, t0, t1, tag, msg, tag_col=RED, bar_col=OFFW, txt_col=INK):
    if not (t0 <= t < t1):
        return
    k_in = ease_out((t - t0) / 0.25)
    k_out = ease_out((t1 - t) / 0.2)
    k = min(k_in, k_out)
    if k <= 0:
        return
    x0, y0 = 120, 912
    tag_w = int(tw(tag, F_XB, 40)) + 50
    msg_w = int(tw(msg, F_B, 44)) + 70
    full = tag_w + msg_w
    shown = int(full * k)
    # clip by drawing on an overlay of limited width
    layer = Image.new("RGBA", (full + 40, 76), (0, 0, 0, 0))
    ld = ImageDraw.Draw(layer)
    para(ld, 0, 0, tag_w, 76, tag_col + (255,), 20)
    text(ld, (tag_w // 2 + 10, 38), tag, F_XB, 40, WHITE, "mm")
    para(ld, tag_w + 4, 0, msg_w, 76, bar_col + (255,), 20)
    text(ld, (tag_w + 40, 38), msg, F_B, 44, txt_col, "lm")
    ld.rectangle([tag_w + 4, 70, tag_w + 4 + msg_w + 20, 76], fill=tag_col + (255,))
    layer = layer.crop((0, 0, max(1, shown + 20), 76))
    ov.alpha_composite(layer, (x0, y0))


def tower(d, ov, t):
    """timing tower, top left."""
    x0, y0, w = 40, 150, 400
    od = ImageDraw.Draw(ov)
    od.rectangle([x0, y0, x0 + w, y0 + 52], fill=RED + (255,))
    if t < R0:
        head = "SETUP"
    else:
        lap = min(LAPS, int(progress(t, T_LOT_REAL) * LAPS) + 1)
        head = f"LAP {lap} / {LAPS}" if t < F_LOT else "FINISH"
    text(od, (x0 + 18, y0 + 26), head, F_XB, 36, WHITE, "lm")
    text(od, (x0 + w - 16, y0 + 26), "TOKENS", F_SB, 26, WHITE, "rm")
    # rows: FULL P1 (pole), LoT P2 until the swap
    k = ease_io((t - T_SWAP) / 0.3)
    rows = [("FULL", SILVER, full_tokens(t), 0 + k), ("LoT", RED, lot_tokens(t), 1 - k)]
    for name, col, tok, pos in rows:
        y = y0 + 56 + int(pos * 64)
        od.rectangle([x0, y, x0 + w, y + 60], fill=(10, 12, 16, 235))
        text(od, (x0 + 22, y + 30), str(int(round(pos)) + 1), F_XB, 34, WHITE, "mm")
        od.rectangle([x0 + 44, y + 10, x0 + 50, y + 50], fill=col + (255,))
        text(od, (x0 + 64, y + 30), name, F_XB, 36, WHITE, "lm")
        text(od, (x0 + 250, y + 30), f"{tok:,}" if tok else "—", F_B, 34, YELLOW if name == "LoT" else OFFW, "rm")
        # gap column
        g = ""
        gc = OFFW
        if t >= R0:
            if name == "LoT":
                g = "LEADER" if t < F_LOT else "WIN"
                gc = PURPLE if t >= F_LOT else OFFW
            else:
                r = min((t - R0) * CLOCK, T_LOT_REAL)
                g = f"+{r * (T_FULL_REAL / T_LOT_REAL - 1):.2f}"
        text(od, (x0 + w - 16, y + 30), g, F_B, 30, gc, "rm")


def clock_bug(d, ov, t):
    od = ImageDraw.Draw(ov)
    x0, y0 = 1560, 40
    para(od, x0, y0, 120, 56, RED + (255,), 14)
    text(od, (x0 + 66, y0 + 28), "LoT GP", F_XB, 34, WHITE, "mm")
    para(od, x0 + 128, y0, 196, 56, (10, 12, 16, 235), 14)
    if t >= R0:
        r = (min(t, F_FULL) - R0) * CLOCK
        s = f"{int(r // 60)}:{r % 60:06.3f}"
    elif t >= T_RP:
        s = "REPLAY"
    else:
        s = "0:00.000"
    if t >= T_RP:
        s = "REPLAY"
    text(od, (x0 + 236, y0 + 28), s, F_B, 36, WHITE, "mm")


def panel_header(od, x0, name, col, sub):
    para(od, x0, PY0 - 56, 200, 46, col + (255,), 14)
    text(od, (x0 + 107, PY0 - 33), name, F_XB, 34, WHITE if col != SILVER else INK, "mm")
    text(od, (x0 + 230, PY0 - 33), sub, F_SB, 28, OFFW, "lm")


def token_chip(od, x0, tok, col):
    od.rectangle([x0, PY0 + P + 10, x0 + P, PY0 + P + 58], fill=(10, 12, 16, 235))
    od.rectangle([x0, PY0 + P + 10, x0 + 8, PY0 + P + 58], fill=col + (255,))
    text(od, (x0 + 24, PY0 + P + 34), "TOKENS", F_SB, 28, SILVER, "lm")
    text(od, (x0 + P - 16, PY0 + P + 34), f"{tok:,}" if tok else "—", F_XB, 40, YELLOW if col == RED else WHITE, "rm")


def sector_badge(od, t, t0, val, col, label, x, y):
    if t < t0:
        return
    k = ease_out((t - t0) / 0.2)
    w = int(220 * k)
    od.rectangle([x, y, x + w, y + 64], fill=col + (255,))
    if k > 0.8:
        text(od, (x + 16, y + 32), label, F_SB, 26, WHITE, "lm")
        text(od, (x + 206, y + 32), val, F_XB, 44, WHITE, "rm")


def pit_board(od, t):
    """setup: LoT pit board on the right."""
    x0, y0 = 1600, PY0
    od.rectangle([x0 + 130, y0 + 380, x0 + 146, y0 + 512], fill=(80, 84, 90, 255))     # pole
    od.rectangle([x0, y0, x0 + 276, y0 + 390], fill=(8, 8, 8, 255))
    od.rectangle([x0, y0, x0 + 276, y0 + 390], outline=(60, 60, 60, 255), width=4)
    lt = lot_tokens(t)
    sp = "1.0X" if t < T_SEC1 else ("1.9X" if t < T_SEC2 else "2.4X")
    rows = [("LoT", WHITE), ("P2" if t < T_SWAP else "P1", YELLOW), (f"{lt:,}" if lt else "—", YELLOW), ("TOKENS", WHITE),
            (sp, GREEN if T_SEC1 <= t < T_SEC2 else (PURPLE if t >= T_SEC2 else WHITE))]
    for k, (s, c) in enumerate(rows):
        text(od, (x0 + 138, y0 + 46 + k * 74), s, F_XB, 64 if k != 3 else 40, c, "mm")


def telemetry(od, t):
    """race: speed trace on the right."""
    x0, y0, w, h = 1600, PY0, 276, 512
    od.rectangle([x0, y0, x0 + w, y0 + h], fill=(10, 12, 16, 235))
    od.rectangle([x0, y0, x0 + w, y0 + 46], fill=(30, 32, 38, 255))
    text(od, (x0 + 14, y0 + 23), "TELEMETRY · SPEED", F_SB, 26, OFFW, "lm")
    gx0, gy0, gw, gh = x0 + 20, y0 + 70, w - 40, 260
    for k in range(5):
        y = gy0 + gh * k / 4
        od.line([gx0, y, gx0 + gw, y], fill=(40, 44, 50, 255), width=1)
    tt = np.linspace(R0, min(t, F_FULL), 80)

    def sp(tq, top, t_fin):
        return np.where(tq < t_fin, top * np.clip((tq - R0) / 0.35, 0, 1), 0)
    for top, t_fin, col in ((1.0, F_FULL, SILVER), (2.4, F_LOT, RED)):
        v = sp(tt, top, t_fin)
        xs = gx0 + (tt - R0) / (F_FULL - R0) * gw
        ys = gy0 + gh - v / 2.6 * gh
        pts = list(zip(xs.tolist(), ys.tolist()))
        if len(pts) > 1:
            od.line(pts, fill=col + (255,), width=4)
    text(od, (x0 + 20, y0 + 380), "LoT", F_XB, 40, RED, "lm")
    text(od, (x0 + w - 20, y0 + 380), "2.4X", F_XB, 56, PURPLE if t >= F_LOT else WHITE, "rm")
    text(od, (x0 + 20, y0 + 450), "FULL", F_XB, 40, SILVER, "lm")
    text(od, (x0 + w - 20, y0 + 450), "1.0X", F_XB, 56, WHITE, "rm")


def car_shape(od, x, y, col, s=1.0):
    """top-down F1 car, nose pointing right, centred at (x, y)."""
    def P_(px, py):
        return (x + px * s, y + py * s)
    od.polygon([P_(-46, -6), P_(30, -6), P_(52, -2), P_(52, 2), P_(30, 6), P_(-46, 6)], fill=col + (255,))   # body
    od.rectangle([P_(-56, -16), P_(-46, 16)], fill=col + (255,))          # rear wing
    od.rectangle([P_(46, -14), P_(54, 14)], fill=col + (255,))            # front wing
    for wx in (-34, 30):
        od.rectangle([P_(wx - 8, -15), P_(wx + 8, -8)], fill=(20, 20, 20, 255))
        od.rectangle([P_(wx - 8, 8), P_(wx + 8, 15)], fill=(20, 20, 20, 255))
    od.ellipse([P_(-6, -4), P_(6, 4)], fill=(20, 20, 20, 255))            # cockpit


def track(od, t):
    x0, x1, y0 = TRACK_X0, TRACK_X1, TRACK_Y
    od.rectangle([x0 - 20, y0, x1 + 20, y0 + 92], fill=(40, 42, 46, 255))
    od.rectangle([x0 - 20, y0, x1 + 20, y0 + 4], fill=WHITE + (255,))
    od.rectangle([x0 - 20, y0 + 88, x1 + 20, y0 + 92], fill=WHITE + (255,))
    for k in range(0, x1 - x0 + 40, 40):                                  # kerbs
        c = RED if (k // 40) % 2 == 0 else WHITE
        od.rectangle([x0 - 20 + k, y0 - 10, x0 + 20 + k, y0], fill=c + (255,))
        od.rectangle([x0 - 20 + k, y0 + 92, x0 + 20 + k, y0 + 102], fill=c + (255,))
    for k in range(1, LAPS):                                              # lap marks
        x = x0 + (x1 - x0) * k / LAPS
        od.line([x, y0 + 44, x, y0 + 48], fill=(120, 120, 120, 255), width=2)
    sq = 11
    for i in range(8):                                                    # chequered finish
        for j in range(2):
            c = WHITE if (i + j) % 2 == 0 else (0, 0, 0)
            od.rectangle([x1 + j * sq, y0 + 4 + i * sq, x1 + (j + 1) * sq, y0 + 4 + (i + 1) * sq], fill=c + (255,))
    od.rectangle([x0 - 3, y0 + 4, x0 + 3, y0 + 88], fill=WHITE + (255,))  # start line
    for car, col, t_real, lane in (("LoT", RED, T_LOT_REAL, 26), ("FULL", SILVER, T_FULL_REAL, 66)):
        p = progress(t, t_real)
        x = x0 + 30 + p * (x1 - x0 - 60)
        if 0 < p < 1:
            for k in range(1, 6):                                         # speed streaks
                od.line([x - 60 - k * 30 * (2.4 if car == "LoT" else 1), y0 + lane - 6 + k * 2,
                         x - 60, y0 + lane - 6 + k * 2], fill=col + (60,), width=2)
        car_shape(od, x, y0 + lane, col, 0.8)
        text(od, (x - 70, y0 + lane), car, F_XB, 24, col, "rm")


def lights(od, t):
    if not (LIGHTS[0] - 0.25 <= t < R0 + 0.25):
        return
    k = min(ease_out((t - LIGHTS[0] + 0.25) / 0.2), ease_out((R0 + 0.25 - t) / 0.2))
    cx, y0 = W // 2 + 120, 330
    w = 760
    od.rectangle([cx - w // 2 - 40, y0 - 30, cx + w // 2 + 40, y0 + 250], fill=(0, 0, 0, int(200 * k)))
    od.rectangle([cx - w // 2, y0, cx + w // 2, y0 + 16], fill=(30, 30, 30, int(255 * k)))
    for i in range(5):
        x = cx - w // 2 + 76 + i * 152
        od.rectangle([x - 52, y0 + 16, x + 52, y0 + 220], fill=(18, 18, 18, int(255 * k)))
        on = LIGHTS[i] <= t < R0
        for j in range(2):
            yy = y0 + 70 + j * 100
            if on:
                od.ellipse([x - 44, yy - 44, x + 44, yy + 44], fill=(255, 30, 20, int(90 * k)))
                od.ellipse([x - 36, yy - 36, x + 36, yy + 36], fill=(255, 20, 10, int(255 * k)))
                od.ellipse([x - 14, yy - 22, x + 4, yy - 8], fill=(255, 160, 150, int(255 * k)))
            else:
                od.ellipse([x - 36, yy - 36, x + 36, yy + 36], fill=(45, 8, 8, int(255 * k)))


def fastest_lap(od, t):
    t0, t1 = F_LOT + 0.25, F_FULL - 0.05
    if not (t0 <= t < t1):
        return
    k = min(ease_out((t - t0) / 0.25), ease_out((t1 - t) / 0.2))
    x0, y0 = LX0, PY0 + P - 140
    w = int(P * k)
    od.rectangle([x0, y0, x0 + w, y0 + 120], fill=PURPLE + (240,))
    if k > 0.9:
        text(od, (x0 + 26, y0 + 36), "FASTEST LAP", F_XB, 46, WHITE, "lm")
        text(od, (x0 + 26, y0 + 86), "LoT · 1,027 TOKENS", F_B, 34, WHITE, "lm")
        text(od, (x0 + P - 24, y0 + 60), "2.4X", F_XB, 84, WHITE, "rm")


def finish_note(od, t):
    if t < F_FULL + 0.1 or t >= T_STING + 0.1:
        return
    x0, y0 = FX0, PY0 + P // 2 - 60
    od.rectangle([x0, y0, x0 + P, y0 + 120], fill=(10, 12, 16, 230))
    text(od, (x0 + P // 2, y0 + 40), "FINISHED", F_XB, 44, SILVER, "mm")
    text(od, (x0 + P // 2, y0 + 88), f"+{T_FULL_REAL - T_LOT_REAL:.2f} s", F_XB, 44, YELLOW, "mm")


def title(canvas, od, t):
    if t >= T_SETUP + 0.2:
        return
    k_in = ease_out(t / 0.35)
    k_out = ease_out((T_SETUP + 0.2 - t) / 0.3)
    k = min(k_in, k_out)
    off = int((1 - k) * 1400)
    od.polygon([(300 - off, 360), (1720 - off, 360), (1640 - off, 560), (220 - off, 560)], fill=RED + (255,))
    od.polygon([(1740 + off, 580), (1820 + off, 580), (1760 + off, 720), (1680 + off, 720)], fill=WHITE + (255,))
    od.polygon([(330 + off, 580), (1660 + off, 580), (1600 + off, 720), (270 + off, 720)], fill=(10, 12, 16, 255))
    text(od, (960 - off, 462), "LEVEL-OF-TOKEN GRAND PRIX", F_XB, 120, WHITE, "mm")
    text(od, (935 + off, 650), "FULL TOKENS  vs  LoT", F_XB, 80, OFFW, "mm")


def stinger(od, t):
    """replay stinger: red/white slanted bars sweep across."""
    dt = t - T_STING
    if not (0 <= dt < 0.6):
        return
    for i, (col, dl) in enumerate(((RED, 0.0), (WHITE, 0.05), (RED, 0.1))):
        x = -900 + (dt - dl) / 0.45 * 3600
        od.polygon([(x, 0), (x + 700, 0), (x + 400, H), (x - 300, H)], fill=col + (255,))
    if 0.15 < dt < 0.5:
        text(od, (W // 2, H // 2), "RACE REPLAY", F_XB, 160, WHITE, "mm")


def replay(canvas, od, t):
    left, right, tok = replay_panels(t)
    k = ease_out((t - T_RP) / 0.3)
    for x0, im in ((VX0L, left), (VX0R, right)):
        pim = Image.fromarray(np.clip(im, 0, 255).astype(np.uint8))
        if k < 0.999:
            pim = pim.resize((VW, max(2, int(VH * k))), Image.BILINEAR)
        canvas.paste(pim, (x0, VY0 + (VH - pim.height) // 2))
    # headers
    para(od, VX0L, VY0 - 56, 230, 46, RED + (255,), 14)
    text(od, (VX0L + 122, VY0 - 33), "LoT VIDEO", F_XB, 34, WHITE, "mm")
    para(od, VX0R, VY0 - 56, 270, 46, RED + (255,), 14)
    text(od, (VX0R + 142, VY0 - 33), "LoT TOKENS", F_XB, 34, WHITE, "mm")
    # REPLAY bug + playback rate on the video
    od.rectangle([VX0L + 16, VY0 + 16, VX0L + 150, VY0 + 58], fill=(10, 12, 16, 220))
    text(od, (VX0L + 83, VY0 + 37), "REPLAY", F_XB, 30, WHITE, "mm")
    sp = vid_speed(t)
    pulse = any(0 <= t - t0 < 0.6 for t0 in (T_BSP1, T_BSP2))
    col = PURPLE if (t >= T_BSP2 and pulse) else (GREEN if pulse else (10, 12, 16))
    od.rectangle([VX0L + VW - 190, VY0 + 16, VX0L + VW - 16, VY0 + 72], fill=col + (230,))
    n_tri = 1 if sp < 1.5 else 2
    for i in range(n_tri):
        bx = VX0L + VW - 176 + i * 22
        od.polygon([(bx, VY0 + 28), (bx + 20, VY0 + 44), (bx, VY0 + 60)], fill=WHITE + (255,))
    text(od, (VX0L + VW - 30, VY0 + 44), f"{sp:.1f}X", F_XB, 44, WHITE, "rm")
    # token bars (pit wall)
    y0 = VY0 + VH + 36
    full_w = W - 2 * VX0L - 260
    for k2, (name, col, val) in enumerate((("FULL", SILVER, VID_TOK[0]), ("LoT", RED, tok if tok else VID_TOK[0]))):
        y = y0 + k2 * 70
        od.rectangle([VX0L, y, VX0L + 240, y + 58], fill=(10, 12, 16, 235))
        od.rectangle([VX0L, y, VX0L + 8, y + 58], fill=col + (255,))
        text(od, (VX0L + 26, y + 29), name + " TOKENS", F_XB, 32, WHITE, "lm")
        bw = int(full_w * (val / VID_TOK[0]) * (ease_out((t - T_RP - 0.2) / 0.4)))
        od.rectangle([VX0L + 250, y + 6, VX0L + 250 + full_w, y + 52], fill=(10, 12, 16, 200))
        od.rectangle([VX0L + 250, y + 6, VX0L + 250 + bw, y + 52], fill=col + (255,))
        text(od, (VX0L + 250 + max(bw, 150) - 14, y + 29), f"{int(round(val)):,}", F_XB, 36,
             INK if bw > 160 else WHITE, "rm")
    # sector badges on the token panel
    sector_badge(od, t, T_BSP1, "2.2X", GREEN, "GROUND 2×2", VX0R + VW - 236, VY0 + 16)
    sector_badge(od, t, T_BSP2, "2.5X", PURPLE, "SKY 4×4", VX0R + VW - 236, VY0 + 90)


def end_card(od, t):
    if t < T_END:
        return
    k = ease_out((t - T_END) / 0.35)
    off = int((1 - k) * 1500)
    od.rectangle([0, 0, W, H], fill=(0, 0, 0, int(150 * k)))
    od.polygon([(260 - off, 380), (1700 - off, 380), (1640 - off, 560), (200 - off, 560)], fill=RED + (255,))
    od.polygon([(290 + off, 576), (1640 + off, 576), (1590 + off, 700), (240 + off, 700)], fill=(10, 12, 16, 255))
    text(od, (950 - off, 470), "LEVEL-OF-TOKEN DIFFUSION", F_XB, 116, WHITE, "mm")
    text(od, (940 + off, 638), "IMAGES 2.4X FASTER  ·  VIDEO 2.5X FASTER", F_XB, 64, YELLOW, "mm")
    sq = 30
    ph = int((t - T_END) * 120)
    for i in range(-1, W // sq + 2):
        for j in range(3):
            if (i + j) % 2 == 0:
                x = i * sq - ph % (2 * sq)
                od.rectangle([x, 760 + j * sq, x + sq, 760 + (j + 1) * sq], fill=WHITE + (int(255 * k),))


# ------------------------------------------------------------------ frame
def render(t):
    a = assets()
    canvas = Image.fromarray(a["bg"]).convert("RGBA")
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)

    if T_SETUP - 0.1 <= t < T_STING + 0.3:
        k = ease_out((t - T_SETUP + 0.1) / 0.35)
        for car, x0 in (("full", FX0), ("lot", LX0)):
            pim = Image.fromarray(np.clip(car_panel(t, car), 0, 255).astype(np.uint8))
            if k < 0.999:
                pim = pim.resize((P, max(2, int(P * k))), Image.BILINEAR)
            canvas.paste(pim, (x0, PY0 + (P - pim.height) // 2))
        panel_header(od, FX0, "FULL", SILVER, "every patch, always")
        panel_header(od, LX0, "LoT", RED, "detail where it's in focus")
        token_chip(od, FX0, full_tokens(t), SILVER)
        token_chip(od, LX0, lot_tokens(t), RED)
        sector_badge(od, t, T_SEC1, "1.9X", GREEN, "2×2", LX0 + P - 236, PY0 + 16)
        sector_badge(od, t, T_SEC2, "2.4X", PURPLE, "4×4 · 8×8", LX0 + P - 236, PY0 + 90)
        tower(None, ov, t)
        if t < R0 + 0.2:
            pit_board(od, t)
        else:
            telemetry(od, t)
        track(od, t)
        lights(od, t)
        fastest_lap(od, t)
        finish_note(od, t)
    if t >= T_RP:
        replay(canvas, od, t)

    lower_third(None, ov, t, 1.7, 3.0, "SETUP", "DEPTH MAP: THE GIRAFFE IS IN FOCUS")
    lower_third(None, ov, t, 3.0, 4.3, "SETUP", "EVERY PATCH = 1 TOKEN · 4,096 EACH")
    lower_third(None, ov, t, 4.3, 5.7, "LoT", "OUT OF FOCUS = BIGGER TOKENS")
    lower_third(None, ov, t, 5.7, 7.0, "LoT", "SAME SCENE · 1,027 TOKENS")
    lower_third(None, ov, t, 7.0, R0, "RACE", "LIGHTS OUT IN…")
    lower_third(None, ov, t, R0, F_LOT + 0.2, "RACE", "AND AWAY WE GO!")
    lower_third(None, ov, t, F_LOT + 0.2, F_FULL, "FLAG", "LoT WINS · FEWER TOKENS = FASTER", tag_col=PURPLE)
    lower_third(None, ov, t, F_FULL, T_STING, "FINISH", "FULL TOKENS: +3.37 s")
    lower_third(None, ov, t, T_RP + 0.1, B2[0], "REPLAY", "VIDEO · 75,600 TOKENS")
    lower_third(None, ov, t, B2[0], B3[0], "LoT", "GROUND MERGES 2×2 · 2.2X FASTER", tag_col=GREEN)
    lower_third(None, ov, t, B3[0], T_END, "LoT", "SKY MERGES 4×4 · 2.5X FASTER", tag_col=PURPLE)

    clock_bug(None, ov, t)
    title(canvas, od, t)
    end_card(od, t)
    stinger(od, t)
    canvas.alpha_composite(ov)
    return np.asarray(canvas.convert("RGB"))


# ------------------------------------------------------------------ audio
SR = 48000


def lp(x, fc, order=2):
    b, a_ = butter(order, fc / (SR / 2), "low")
    return lfilter(b, a_, x)


def bp(x, f0, f1):
    b, a_ = butter(2, [f0 / (SR / 2), f1 / (SR / 2)], "band")
    return lfilter(b, a_, x)


def saw_f(freq_arr):
    ph = np.cumsum(freq_arr) / SR
    return 2 * (ph % 1) - 1


def build_audio(path):
    n = int((DUR + 0.3) * SR)
    tt = np.arange(n) / SR
    L = np.zeros(n)
    R = np.zeros(n)
    rng = np.random.default_rng(0)

    def add(sig, t0, pan=0.0, vol=1.0):
        i = int(t0 * SR)
        j = min(n, i + len(sig))
        if j <= i:
            return
        gl, gr = math.cos((pan + 1) * math.pi / 4), math.sin((pan + 1) * math.pi / 4)
        L[i:j] += sig[:j - i] * gl * vol * 1.414
        R[i:j] += sig[:j - i] * gr * vol * 1.414

    def env_ad(m, a, d):
        e = np.ones(m)
        na = max(1, int(a * SR))
        e[:na] = np.linspace(0, 1, na)
        e[na:] = np.exp(-np.arange(m - na) / SR / d)
        return e

    # crowd wash with cheers at the flags
    crowd = bp(rng.uniform(-1, 1, n), 300, 2500)
    crowd_env = 0.05 + 0.0 * tt
    for t0, amp in ((F_LOT, 0.35), (F_FULL, 0.15), (T_END, 0.25)):
        crowd_env += amp * np.clip((tt - t0) / 0.15, 0, 1) * np.exp(-np.clip(tt - t0, 0, None) / 1.2)
    crowd_env *= np.clip(tt / 0.8, 0, 1)
    c2 = bp(rng.uniform(-1, 1, n), 300, 2500)
    L += crowd * crowd_env * 0.6
    R += c2 * crowd_env * 0.6

    # whooshes (title, replay stinger, end card)
    for t0, dur in ((0.0, 0.6), (T_STING, 0.6), (T_END, 0.5)):
        m = int(dur * SR)
        w = bp(rng.uniform(-1, 1, m), 400, 4000) * np.sin(np.linspace(0, np.pi, m)) ** 2
        tone = np.sin(2 * np.pi * np.cumsum(np.linspace(300, 1400, m) * np.sin(np.linspace(0, np.pi, m)) + 200) / SR)
        add(w * 0.9 + tone * 0.1 * np.sin(np.linspace(0, np.pi, m)), t0, 0, 0.6)

    # start-light beeps
    for t0 in LIGHTS:
        m = int(0.16 * SR)
        add(np.sin(2 * np.pi * 740 * np.arange(m) / SR) * env_ad(m, 0.005, 0.08), t0, 0, 0.35)

    # sector dings
    for t0, f in ((T_SEC1, 1320), (T_SEC2, 1760), (T_BSP1, 1320), (T_BSP2, 1760)):
        m = int(0.5 * SR)
        s = np.arange(m) / SR
        sig = (np.sin(2 * np.pi * f * s) + 0.5 * np.sin(2 * np.pi * f * 1.5 * s)) * env_ad(m, 0.003, 0.15)
        add(sig, t0, 0.3, 0.3)
        add(sig, t0 + 0.09, 0.3, 0.2)

    # engines: idle in the garage, race whine (pitch follows speed, gear shifts), replay
    def engine(t0, t1, f_of_t, vol, pan):
        i0, i1 = int(t0 * SR), int(t1 * SR)
        s = tt[i0:i1]
        f = f_of_t(s)
        sig = saw_f(f) * 0.6 + saw_f(f * 2.01) * 0.25 + np.sin(2 * np.pi * np.cumsum(f * 0.5) / SR) * 0.3
        sig = lp(sig, 3500)
        e = np.clip((s - t0) / 0.05, 0, 1) * np.clip((t1 - s) / 0.15, 0, 1)
        add(sig * e, t0, pan, vol)

    def idle(s):
        return 70 + 6 * np.sin(2 * np.pi * 7 * s) + 260 * np.clip(np.sin(np.pi * (s - 7.0) / 0.25), 0, 1) * (s > 7.0)

    engine(T_SETUP, R0, idle, 0.10, 0)

    def race_f(top, t_fin, shift):
        def f(s):
            r = np.clip((s - R0) / 0.35, 0, 1)
            gear = ((s - R0) / shift) % 1
            base = 220 + 380 * top / 2.4 * r
            return base * (0.82 + 0.3 * gear)
        return f
    engine(R0, F_LOT + 0.25, race_f(2.4, F_LOT, 0.28), 0.22, 0.45)
    engine(R0, F_FULL + 0.25, race_f(1.0, F_FULL, 0.5), 0.18, -0.45)

    def replay_f(s):
        sp = np.where(s < T_BSP1, 1.0, np.where(s < T_BSP2, 2.2, 2.5))
        gear = ((s - T_RP) / (0.5 / sp)) % 1
        return (140 + 110 * sp) * (0.85 + 0.25 * gear)
    engine(T_RP, T_END + 0.4, replay_f, 0.12, 0)

    # doppler pass at the overtake
    m = int(0.8 * SR)
    s = np.arange(m) / SR
    f = 520 * (1.25 - 0.5 / (1 + np.exp(-(s - 0.4) * 14)))
    sig = lp(saw_f(f), 3000) * np.exp(-((s - 0.4) / 0.2) ** 2)
    i = int((T_SWAP - 0.4) * SR)
    pan = np.linspace(-1, 1, m)
    L[i:i + m] += sig * (1 - pan) * 0.25
    R[i:i + m] += sig * (1 + pan) * 0.25

    # end sting: brass-ish chord
    for f0 in (261.6, 329.6, 392.0, 523.3):
        m = int(1.4 * SR)
        s = np.arange(m) / SR
        sig = lp(saw_f(np.full(m, f0) * (1 + 0.004 * np.sin(2 * np.pi * 5 * s))), 2200) * env_ad(m, 0.04, 0.9)
        add(sig, T_END + 0.05, 0, 0.12)

    out = np.stack([L, R], 1)
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
    wav = os.path.join(HERE, f"media/final/LoT_f1_race_{TAG}.wav")
    build_audio(wav)
    out = os.path.join(HERE, f"media/final/LoT_f1_race_{TAG}_{res}.mp4")
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
