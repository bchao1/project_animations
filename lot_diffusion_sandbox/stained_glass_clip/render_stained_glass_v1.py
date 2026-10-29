"""LoT Diffusion, 20-s stained-glass clip, v1 (numpy + PIL + cv2 frames → ffmpeg, synthesized organ / chimes / bell). No manim.

Tokens are glass panes, lead came = token borders. A dark stone nave; light through the window lights the panes.
  0.0–2.8  title; the depth map is the glazier's cartoon, faint in the empty window
  2.8–4.8  glazing: every 16-px patch becomes a pane (4,096)
  5.0–6.6  lead between out-of-focus panes melts away → 2 × 2 (1,606) → 1.9× faster
  7.9–9.3  bigger still → the real LoT layout (1,027) → 2.4× faster
 10.4–12.4 sunrise: the generated image glows through the same panes, god rays
 12.4–13.2 the light dims, a wide window: the platformer video
 13.2–20   75,600 panes; ground → 2 × 2 (2.2× faster), sky → 4 × 4 (2.5× faster); playback speed follows; title

Speedups as in ../arcade_clip (paper fits: image 1.9× / 2.4× for 1,606 / 1,027 tokens; video 2.2× / 2.5×).
Usage: python render_stained_glass_v1.py h          → media/final/LoT_stained_glass_v1_1080p60.mp4
       python render_stained_glass_v1.py still t …  → scratch/still_v1_<t>.png
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
from scipy.signal import fftconvolve

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TAG = "v1"
FONT = os.path.join(HERE, "data/Cinzel.ttf")
CUE = os.path.join(ROOT, "data/img/img_000000000025_dof_cue.png")
GEN = os.path.join(ROOT, "data/img/img_000000000025_dof_generated_4b.png")
LAYOUT = os.path.join(ROOT, "data/layouts/img_000000000025_dof.json")
VID = os.path.join(ROOT, "data/clips/vd_platformer_gen")
LEVELS = os.path.join(ROOT, "arcade_clip/data/platformer_levels.npy")

W, H = 1920, 1080
DUR = 20.0
GOLD = (236, 196, 112)
GOLD_D = (150, 104, 40)
IVORY = (238, 226, 200)

# ------------------------------------------------------------------ timeline (s)
T_TITLE = 0.2
T_CART = 1.0                 # cartoon (depth map) appears
G1 = (2.8, 4.8)              # glazing
M1 = (5.0, 6.6)              # merge → 2 × 2
M2 = (7.9, 9.3)              # merge → layout
T_SP1, T_SP2 = 6.7, 9.4
T_SUN = 10.4                 # sunrise
T_REV = (10.6, 11.7)         # panes turn into the image
T_OUT = 12.4                 # light dims
T_B0 = 13.2                  # wide window on
BM1 = (14.8, 15.6)
BM2 = (16.8, 17.5)
T_BSP1, T_BSP2 = 15.7, 17.6
T_END = 18.6

VID_SPEED = (1.0, 2.2, 2.5)
VID_TOK = (75600, 38046, 34698)

# ------------------------------------------------------------------ geometry
N = 64
CP = 11                       # px per token (main window)
P = N * CP                    # 704
PX0, PY0 = (W - P) // 2, 230
VN, VM = 45, 80
VC = 14                       # px per token (wide window)
VW, VH = VM * VC, VN * VC     # 1120 × 630
VX0, VY0 = (W - VW) // 2, 190
FR = 30                       # stone frame thickness


def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def smooth(x):
    x = clamp(x)
    return x * x * (3 - 2 * x)


def ease_out(x):
    x = clamp(x)
    return 1 - (1 - x) ** 3


# ------------------------------------------------------------------ assets
def saturate(rgb, s=1.3, v=1.0):
    hsv = cv2.cvtColor(np.clip(rgb, 0, 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * s, 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * v, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)


def stone_wall(rng):
    """dark ashlar wall: rows of blocks with mortar, low-frequency grime."""
    img = np.zeros((H, W, 3), np.float32)
    y = 0
    row = 0
    while y < H:
        bh = int(rng.integers(70, 96))
        x = -int(rng.integers(0, 120)) if row % 2 else 0
        while x < W:
            bw = int(rng.integers(130, 230))
            tone = rng.uniform(26, 40)
            img[y:y + bh, max(0, x):x + bw] = (tone * 1.05, tone, tone * 0.9)
            img[y:y + 3, max(0, x):x + bw] *= 1.25                # top edge catches light
            img[y + bh - 3:y + bh, max(0, x):x + bw] *= 0.55
            img[y:y + bh, max(0, x):max(0, x) + 3] *= 0.55
            x += bw
        y += bh
        row += 1
    grime = cv2.GaussianBlur(rng.uniform(0.7, 1.15, (H // 8, W // 8)).astype(np.float32), (0, 0), 3)
    img *= cv2.resize(grime, (W, H))[..., None]
    img *= (rng.uniform(0.9, 1.1, (H, W)).astype(np.float32))[..., None]
    return img


def glass_texture(rng, h, w):
    """seedy cathedral glass: blotches + vertical streaks, multiplier around 1."""
    a = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), 3)
    b = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), sigmaX=0.8, sigmaY=9)
    t = 1 + 0.10 * a / (a.std() + 1e-6) + 0.06 * b / (b.std() + 1e-6)
    seeds = rng.random((h, w)) < 0.0015
    t[cv2.dilate(seeds.astype(np.uint8), np.ones((2, 2), np.uint8)) > 0] *= 1.35
    return t.astype(np.float32)


def load_assets():
    a = {}
    rng = np.random.default_rng(11)
    cue = np.asarray(Image.open(CUE).convert("RGB")).astype(np.float32)
    cue = saturate(cue, 1.15, 0.95)
    a["cue"] = cv2.resize(cue, (P, P), interpolation=cv2.INTER_AREA)
    a["cue_cell"] = cue.reshape(N, 16, N, 16, 3).mean((1, 3))
    gen = np.asarray(Image.open(GEN).convert("RGB")).astype(np.float32)
    gen = saturate(gen, 1.3, 1.05)
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
    a["vid"] = [saturate(cv2.resize(np.asarray(Image.open(os.path.join(VID, f"{f + 1:04d}.jpg")).convert("RGB")),
                                    (VW, VH), interpolation=cv2.INTER_AREA).astype(np.float32), 1.25) for f in range(81)]
    a["vid_cell"] = [cv2.resize(v, (VM, VN), interpolation=cv2.INTER_AREA) for v in a["vid"]]
    a["wall"] = stone_wall(rng)
    a["tex_main"] = glass_texture(rng, P, P)
    a["tex_vid"] = glass_texture(rng, VH, VW)
    a["pane_rand"] = rng.uniform(0.88, 1.10, 8192).astype(np.float32)
    rng2 = np.random.default_rng(5)
    a["motes"] = (rng2.uniform(0, W, 260), rng2.uniform(0, H, 260), rng2.uniform(0, 2 * np.pi, 260), rng2.uniform(0.4, 1.0, 260))
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    a["vignette"] = np.clip(1 - 0.55 * (((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2), 0, 1)[..., None]
    return a


A = None


def assets():
    global A
    if A is None:
        A = load_assets()
    return A


# ------------------------------------------------------------------ text (engraved gold)
@lru_cache(maxsize=64)
def font(size, weight="Bold"):
    f = ImageFont.truetype(FONT, size)
    f.set_variation_by_name(weight)
    return f


@lru_cache(maxsize=512)
def text_rgba(s, size, weight="Bold", col=GOLD, col2=GOLD_D):
    f = font(size, weight)
    l, t, r, b = f.getbbox(s)
    w, h = r - l + 12, b - t + 14
    m = Image.new("L", (w, h), 0)
    ImageDraw.Draw(m).text((6 - l, 4 - t), s, font=f, fill=255)
    m = np.asarray(m).astype(np.float32) / 255
    g = np.linspace(0, 1, h)[:, None, None]
    fill = np.array(col, np.float32) * (1 - g) + np.array(col2, np.float32) * g
    sh = np.roll(np.roll(m, 3, 0), 2, 1)
    sh = cv2.GaussianBlur(sh, (0, 0), 1.5)
    out = np.zeros((h, w, 4), np.float32)
    out[..., :3] = fill * m[..., None]
    alpha = np.maximum(m, sh * 0.85)
    out[..., 3] = alpha
    out[..., :3] = fill * (m / np.maximum(alpha, 1e-6))[..., None]      # shadow is black
    return Image.fromarray(np.clip(np.dstack([out[..., :3], out[..., 3:] * 255]), 0, 255).astype(np.uint8), "RGBA")


def put_text(canvas, s, size, x, y, alpha=1.0, weight="Bold", col=GOLD, col2=GOLD_D, anchor="c", scale=1.0):
    if alpha <= 0.01 or not s:
        return
    im = text_rgba(s, size, weight, col, col2)
    if abs(scale - 1) > 1e-3:
        im = im.resize((max(1, int(im.width * scale)), max(1, int(im.height * scale))), Image.LANCZOS)
    if alpha < 0.999:
        im = im.copy()
        im.putalpha(im.getchannel("A").point(lambda v: int(v * alpha)))
    if anchor == "c":
        x, y = x - im.width // 2, y - im.height // 2
    elif anchor == "r":
        x, y = x - im.width, y - im.height // 2
    else:
        y = y - im.height // 2
    canvas.alpha_composite(im, (int(x), int(y)))


def caption(canvas, s, t, t0, t1, y=1022, size=40):
    a = smooth((t - t0) / 0.3) * (1 - smooth((t - (t1 - 0.3)) / 0.3))
    put_text(canvas, s, size, W // 2, y, alpha=a, col=IVORY, col2=(190, 170, 130))


# ------------------------------------------------------------------ token maps
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


def edges(ids_px):
    e = np.zeros(ids_px.shape, bool)
    e[:, 1:] |= ids_px[:, 1:] != ids_px[:, :-1]
    e[1:, :] |= ids_px[1:, :] != ids_px[:-1, :]
    e[:, :-1] |= e[:, 1:]
    e[:-1, :] |= e[1:, :]
    e[:2, :] = e[-2:, :] = True
    e[:, :2] = e[:, -2:] = True
    return e


def up(arr, c):
    return np.repeat(np.repeat(arr, c, 0), c, 1)


def frac(t, win):
    return clamp((t - win[0]) / (win[1] - win[0]))


def vid_blocks(lv, cap):
    b = np.minimum(lv, cap).astype(int)
    ii, jj = np.mgrid[0:VN, 0:VM]
    return ii // b * b, jj // b * b, b


def merge_state(old, new, n, m, win, t):
    """per-cell fade a (1 = old lead still there → 0 = melted) and the per-cell switch time."""
    oi_n, oj_n, b_n = new
    t_sw = win[0] + (oi_n + b_n) / n * (win[1] - win[0])
    a = 1 - np.clip((t - t_sw) / 0.28, 0, 1)
    merged = b_n > old[2]
    a = np.where(merged, a, 1.0)
    return a.astype(np.float32), t_sw, merged


# ------------------------------------------------------------------ glass rendering
def lead_render(rgb, lead, illum, tex, glint=None):
    """rgb: pane colours (h, w, 3) 0..255; lead: float (h, w) 0..1 coverage. Returns emitted light (h, w, 3)."""
    lead = np.clip(lead, 0, 1).astype(np.float32)
    hard = (lead > 0.5).astype(np.uint8)
    dt = cv2.distanceTransform(1 - hard, cv2.DIST_L2, 3)
    shade = 0.62 + 0.38 * np.clip(dt / 3.5, 0, 1)                   # panes darken towards the came
    glass = rgb / 255.0 * tex[..., None] * shade[..., None] * illum[..., None] * 255
    lb = cv2.GaussianBlur(lead, (0, 0), 0.9)
    bevel = np.clip((np.roll(np.roll(lb, 1, 0), 1, 1) - np.roll(np.roll(lb, -1, 0), -1, 1)) * 2.2, -1, 1)
    lead_col = 30 + 34 * np.clip(bevel, 0, 1) - 12 * np.clip(-bevel, 0, 1)
    lead_rgb = np.stack([lead_col * 1.02, lead_col, lead_col * 0.95], -1)
    out = glass * (1 - lead[..., None]) + lead_rgb * lead[..., None]
    if glint is not None:
        out = out + glint[..., None] * np.array([255, 214, 140], np.float32)
    return out


def sun_hotspot(h, w, sx, sy, r, k):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    return k * np.exp(-((xx - sx) ** 2 + (yy - sy) ** 2) / (2 * r * r))


def main_window(t):
    """emitted light of the giraffe window (P, P, 3) and the illumination level."""
    a = assets()
    B = a["img_blocks"]
    ambient = 0.30 + 0.45 * smooth((t - G1[0] + 0.5) / 0.8) + 0.35 * smooth((t - T_SUN) / 1.2)
    ambient *= 1 - 0.8 * smooth((t - T_OUT) / 0.6)
    sun_y = P * (1.25 - 1.0 * smooth((t - T_SUN) / 1.8))
    hot = sun_hotspot(P, P, P * 0.72, sun_y, P * 0.35, 0.9 * smooth((t - T_SUN) / 0.8) * (1 - smooth((t - T_OUT) / 0.6)))
    illum = ambient + hot
    tex = a["tex_main"]
    glint = None
    if t < G1[0]:
        # the cartoon: faint drawing, no lead yet
        cart = a["cue"] * smooth((t - T_CART) / 0.6)
        out = cart * 0.55 * tex[..., None] * illum[..., None]
        return out, ambient, None
    if t < M1[0]:
        f = frac(t, G1)
        rows = f * N
        ids = block_ids(B[1][0], B[1][1], N)
        mos = up(mosaic(a["cue_cell"], ids, N * N), CP)
        ry = int(rows) * CP
        rgb = a["cue"] * smooth((t - T_CART) / 0.6)
        rgb = rgb.copy()
        rgb[:ry] = mos[:ry]
        lead = edges(up(ids, CP)).astype(np.float32)
        lead[ry:] = 0
        lead[ry:, :2] = lead[ry:, -2:] = 1
        lead[-2:] = 1
        out = lead_render(rgb, lead, illum, tex)
        out[ry:] *= 0.55
        if f < 1:                                                  # molten seam at the glazing front
            yy = np.arange(P)[:, None].astype(np.float32)
            g = np.exp(-((yy - ry) ** 2) / (2 * 4.0 ** 2)) * 1.4
            out = out + g[..., None] * np.array([255, 190, 110], np.float32)
        return out, ambient, (ry if f < 1 else None)
    # merges / reveal
    if t < M2[0]:
        old, new, win = B[1], B[2], M1
    else:
        old, new, win = B[2], B[8], M2
    fa, t_sw, merged = merge_state(old, new, N, N, win, t)
    ids_o = block_ids(old[0], old[1], N)
    ids_n = block_ids(new[0], new[1], N)
    mos_o = mosaic(a["cue_cell"], ids_o, N * N)
    mos_n = mosaic(a["cue_cell"], ids_n, N * N)
    rgb_c = mos_o * fa[..., None] + mos_n * (1 - fa[..., None])
    rgb = up(rgb_c, CP)
    pr = a["pane_rand"][ids_n % 8192]
    rgb = rgb * up(pr, CP)[..., None]
    l_o = edges(up(ids_o, CP)).astype(np.float32)
    l_n = edges(up(ids_n, CP)).astype(np.float32)
    lead = np.maximum(l_n, l_o * up(fa, CP))
    dt = t - t_sw
    gl = np.where(merged & (dt >= 0), np.exp(-np.maximum(dt, 0) / 0.18), 0).astype(np.float32)
    glint = l_o * (1 - l_n) * up(gl, CP) * 0.9
    if t >= T_REV[0]:
        # sunrise: each pane turns from cartoon colour to the image, bottom → top following the sun
        rr = np.random.default_rng(4).uniform(0, 0.25, N * N)
        cell_t = T_REV[0] + (1 - (new[0] + new[2] / 2) / N) * (T_REV[1] - T_REV[0] - 0.35) + rr[ids_n]
        k = np.clip((t - cell_t) / 0.35, 0, 1).astype(np.float32)
        kp = up(k, CP)[..., None]
        rgb = rgb * (1 - kp) + a["gen"] * kp
        lead = lead * (1 - 0.45 * smooth((t - T_REV[1]) / 0.6))
    out = lead_render(rgb, lead, illum, tex, glint)
    return out, ambient, None


def vid_index(t):
    def V(tt):
        v = 0.0
        for s0, s1, sp in ((T_B0, T_BSP1, VID_SPEED[0]), (T_BSP1, T_BSP2, VID_SPEED[1]), (T_BSP2, 99, VID_SPEED[2])):
            if tt > s0:
                v += (min(tt, s1) - s0) * sp * 15
        return v
    v = V(t)
    return int(v) % 81, v - int(v)


def vid_speed(t):
    return VID_SPEED[0] if t < T_BSP1 else (VID_SPEED[1] if t < T_BSP2 else VID_SPEED[2])


def wide_window(t):
    a = assets()
    idx, _ = vid_index(max(t, T_B0))
    frame = a["vid"][idx]
    cell = a["vid_cell"][idx]
    lv = a["lv"][idx]
    ambient = 0.85 * smooth((t - T_B0 + 0.2) / 0.6)
    u = clamp((t - T_B0) / (DUR - T_B0))
    hot = sun_hotspot(VH, VW, VW * (0.1 + 0.8 * u), VH * (0.55 - 0.35 * math.sin(math.pi * u)), VW * 0.22, 0.75 * ambient)
    illum = ambient + hot
    if t < BM1[0]:
        old = new = vid_blocks(lv, 1)
        win = (BM1[0], BM1[1])
        fa = np.ones((VN, VM), np.float32)
        merged = np.zeros((VN, VM), bool)
        t_sw = np.full((VN, VM), 99.0)
    else:
        if t < BM2[0]:
            old, new, win = vid_blocks(lv, 1), vid_blocks(lv, 2), BM1
        else:
            old, new, win = vid_blocks(lv, 2), vid_blocks(lv, 4), BM2
        fa, t_sw, merged = merge_state(old, new, VN, VM, win, t)
    ids_o = block_ids(old[0], old[1], VM)
    ids_n = block_ids(new[0], new[1], VM)
    mos_o = mosaic(cell, ids_o, VN * VM)
    mos_n = mosaic(cell, ids_n, VN * VM)
    rgb_c = mos_o * fa[..., None] + mos_n * (1 - fa[..., None])
    rgb = up(rgb_c, VC) * 0.6 + frame * 0.4                         # painted glass: pane colour + a little detail
    rgb = rgb * up(a["pane_rand"][ids_n % 8192], VC)[..., None]
    l_o = edges(up(ids_o, VC)).astype(np.float32)
    l_n = edges(up(ids_n, VC)).astype(np.float32)
    lead = np.maximum(l_n, l_o * up(fa, VC))
    dt = t - t_sw
    gl = np.where(merged & (dt >= 0), np.exp(-np.maximum(dt, 0) / 0.18), 0).astype(np.float32)
    glint = l_o * (1 - l_n) * up(gl, VC) * 0.9
    out = lead_render(rgb, lead, illum, a["tex_vid"], glint)
    return out, ambient


# ------------------------------------------------------------------ compositing helpers
def stone_frame(img, x0, y0, w, h, light):
    """carved stone surround with bevel, lit by the window."""
    c = np.array([70, 64, 56], np.float32) * (0.55 + 0.6 * light)
    img[y0 - FR:y0, x0 - FR:x0 + w + FR] = c * 1.15
    img[y0 + h:y0 + h + FR, x0 - FR:x0 + w + FR] = c * 0.7
    img[y0:y0 + h, x0 - FR:x0] = c * 0.95
    img[y0:y0 + h, x0 + w:x0 + w + FR] = c * 0.85
    for k, f in ((0, 1.3), (FR - 4, 0.6)):
        img[y0 - FR + k:y0 - FR + k + 3, x0 - FR:x0 + w + FR] *= f
    img[y0 - 6:y0, x0 - 6:x0 + w + 6] *= 0.5                       # rebate shadow
    img[y0:y0 + h, x0 - 6:x0] *= 0.5
    # sill
    img[y0 + h + FR - 8:y0 + h + FR + 10, x0 - FR - 24:x0 + w + FR + 24] = c * 1.25


def god_rays(emit_full, sx, sy, strength):
    """radial blur of the window's light away from the sun (quarter res)."""
    if strength <= 0.01:
        return 0
    q = 4
    small = cv2.resize(emit_full, (W // q, H // q), interpolation=cv2.INTER_AREA)
    small = np.clip(small - 60, 0, None)
    cx, cy = sx / q, sy / q
    acc = np.zeros_like(small)
    wsum = 0
    for k in range(18):
        s = 1 + 0.045 * k
        M = np.float32([[s, 0, (1 - s) * cx], [0, s, (1 - s) * cy]])
        wk = 0.9 ** k
        acc += cv2.warpAffine(small, M, (W // q, H // q), flags=cv2.INTER_LINEAR) * wk
        wsum += wk
    acc = acc / wsum
    return cv2.resize(acc, (W, H), interpolation=cv2.INTER_LINEAR) * strength


def plaque(canvas, x, y, label, value, alpha, pulse=0.0):
    if alpha <= 0.01:
        return
    d = ImageDraw.Draw(canvas)
    w, h = 300, 150
    d.rectangle([x - w // 2, y - h // 2, x + w // 2, y + h // 2], fill=(18, 16, 14, int(200 * alpha)),
                outline=GOLD_D + (int(255 * alpha),), width=3)
    d.rectangle([x - w // 2 + 8, y - h // 2 + 8, x + w // 2 - 8, y + h // 2 - 8], outline=GOLD_D + (int(140 * alpha),), width=1)
    put_text(canvas, label, 26, x, y - 36, alpha=alpha, weight="Regular", col=(200, 180, 140), col2=(150, 130, 95))
    c1 = tuple(int(v + (255 - v) * pulse) for v in GOLD)
    put_text(canvas, value, 56, x, y + 22, alpha=alpha, col=c1)


def speed_banner(canvas, t, t0, val, y):
    dt = t - t0
    if dt < 0 or dt > 1.5:
        return
    a = smooth(dt / 0.25) * (1 - smooth((dt - 1.2) / 0.3))
    s = 0.85 + 0.15 * ease_out(dt / 0.35)
    put_text(canvas, f"{val} faster", 84, W // 2, y, alpha=a, scale=s, col=(255, 236, 170), col2=(214, 150, 50))


# ------------------------------------------------------------------ frame
def render(t):
    a = assets()
    img = a["wall"].copy()

    main = t < T_B0 - 0.4
    if main:
        emit, amb, glaze_y = main_window(t)
        x0, y0, w, h = PX0, PY0, P, P
        light = amb
        sx, sy = PX0 + P * 0.72, PY0 + P * (1.25 - 1.0 * smooth((t - T_SUN) / 1.8))
        ray_k = 0.25 * amb + 1.1 * smooth((t - T_SUN) / 1.0) * (1 - smooth((t - T_OUT) / 0.6))
    else:
        emit, amb = wide_window(t)
        x0, y0, w, h = VX0, VY0, VW, VH
        light = amb
        u = clamp((t - T_B0) / (DUR - T_B0))
        sx, sy = VX0 + VW * (0.1 + 0.8 * u), VY0 + VH * (0.55 - 0.35 * math.sin(math.pi * u))
        ray_k = 0.55 * amb
    # spill light on the wall around the window
    yy, xx = np.ogrid[0:H, 0:W]
    spill = np.exp(-(((xx - (x0 + w / 2)) / (w * 0.9)) ** 2 + ((yy - (y0 + h * 0.9)) / (h * 0.9)) ** 2))
    img *= (0.55 + 1.3 * light * spill)[..., None].astype(np.float32)
    stone_frame(img, x0, y0, w, h, light)
    img[y0:y0 + h, x0:x0 + w] = emit
    full = np.zeros((H, W, 3), np.float32)
    full[y0:y0 + h, x0:x0 + w] = emit
    rays = god_rays(full, sx, sy, ray_k)
    img = img + rays * np.array([1.0, 0.93, 0.8], np.float32) * 0.9
    # dust motes in the light
    mx, my, mph, mb = a["motes"]
    px = (mx + 18 * np.sin(t * 0.4 + mph)) % W
    py = (my - 14 * t + 10 * np.cos(t * 0.3 + mph)) % H
    if isinstance(rays, np.ndarray):
        lum = rays[py.astype(int), px.astype(int)].mean(1) / 255
        for k in range(len(px)):
            v = clamp(lum[k] * 2.2 * mb[k], 0, 1) * 200
            if v > 4:
                xi, yi = int(px[k]), int(py[k])
                img[max(0, yi - 1):yi + 2, max(0, xi - 1):xi + 2] += v * 0.5
                img[yi:yi + 1, xi:xi + 1] += v
    # bloom
    small = cv2.resize(img, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
    glow = cv2.GaussianBlur(np.clip(small - 150, 0, None), (0, 0), 7)
    img = img + 0.6 * cv2.resize(glow, (W, H), interpolation=cv2.INTER_LINEAR)
    # dim between windows
    img *= 1 - 0.85 * math.exp(-((t - (T_B0 - 0.4)) / 0.25) ** 2)   # dip to dark between the two windows
    img = img * a["vignette"]
    canvas = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).convert("RGBA")

    # ---------------------------------------------- text
    if main:
        ta = smooth((t - T_TITLE) / 0.8) * (1 - smooth((t - T_OUT) / 0.5))
        put_text(canvas, "Level-of-Token Diffusion", 60, W // 2, 112, alpha=ta)
        caption(canvas, "The design: a depth map. In focus: the giraffe", t, T_CART + 0.2, G1[0] + 0.1)
        caption(canvas, "Every patch gets its own pane", t, G1[0] + 0.2, M1[0])
        caption(canvas, "Out of focus: bigger panes", t, M1[0], T_SP1)
        caption(canvas, "Bigger still where it's blurred", t, M2[0], T_SP2)
        caption(canvas, "Sharp where it matters. 4× fewer tokens", t, T_REV[0], T_OUT + 0.3)
        speed_banner(canvas, t, T_SP1, "1.9×", 1012)
        speed_banner(canvas, t, T_SP2, "2.4×", 1012)
        # plaques
        pa = smooth((t - G1[0]) / 0.4) * (1 - smooth((t - T_OUT) / 0.4))
        if t < M1[0]:
            tok = int(frac(t, G1) * N) * N
        elif t < M2[0]:
            tok = int(round(4096 + (1606 - 4096) * frac(t, M1)))
        else:
            tok = int(round(1606 + (1027 - 1606) * frac(t, M2)))
        sp = "1.0×" if t < T_SP1 else ("1.9×" if t < T_SP2 else "2.4×")
        pulse = max(0.0, 1 - abs(t - T_SP1) / 0.4) if t >= T_SP1 else 0.0
        pulse = max(pulse, max(0.0, 1 - abs(t - T_SP2) / 0.4) if t >= T_SP2 else 0.0)
        plaque(canvas, 330, 560, "TOKENS", f"{tok:,}", pa)
        plaque(canvas, W - 330, 560, "SPEED", sp, pa, pulse)
    else:
        ba = smooth((t - T_B0 + 0.2) / 0.5)
        caption(canvas, "Now a video", t, T_B0 - 0.4, T_B0 + 1.0, y=1000)
        caption(canvas, "Every patch, every frame: 75,600 panes", t, T_B0 + 1.0, BM1[0], y=1000)
        caption(canvas, "The ground gets bigger panes", t, BM1[0], T_BSP1, y=1000)
        caption(canvas, "The sky gets the biggest", t, BM2[0], T_BSP2, y=1000)
        speed_banner(canvas, t, T_BSP1, "2.2×", 1000)
        speed_banner(canvas, t, T_BSP2, "2.5×", 1000)
        if t < BM1[0]:
            tok = VID_TOK[0]
        elif t < BM2[0]:
            tok = int(round(VID_TOK[0] + (VID_TOK[1] - VID_TOK[0]) * frac(t, BM1)))
        else:
            tok = int(round(VID_TOK[1] + (VID_TOK[2] - VID_TOK[1]) * frac(t, BM2)))
        sp = vid_speed(t)
        pulse = max([0.0] + [1 - abs(t - t0) / 0.4 for t0 in (T_BSP1, T_BSP2) if t >= t0])
        pa = ba * (1 - smooth((t - T_END) / 0.4))
        plaque(canvas, 190, 505, "TOKENS", f"{tok:,}", pa)
        plaque(canvas, W - 190, 505, "PLAYBACK", f"{sp:.1f}×", pa, pulse)
        if t >= T_END:
            e = smooth((t - T_END) / 0.5)
            band = Image.new("RGBA", (W, 230), (10, 8, 6, int(190 * e)))
            canvas.alpha_composite(band, (0, 395))
            put_text(canvas, "Level-of-Token Diffusion", 76, W // 2, 470, alpha=e)
            put_text(canvas, "Fine glass where it matters. Faster everywhere.", 36, W // 2, 560,
                     alpha=smooth((t - T_END - 0.4) / 0.5), weight="Regular", col=IVORY, col2=(190, 170, 130))
    return np.asarray(canvas.convert("RGB"))


# ------------------------------------------------------------------ audio
SR = 48000


def env_adsr(n, att, rel):
    e = np.ones(n)
    na, nr = int(att * SR), int(rel * SR)
    e[:na] = np.linspace(0, 1, na)
    e[n - nr:] *= np.linspace(1, 0, nr)
    return e


def organ(freqs, dur, vol=0.12, att=0.35, rel=0.5):
    n = int(dur * SR)
    tt = np.arange(n) / SR
    s = np.zeros(n)
    for f in freqs:
        for h, wgt in ((1, 1.0), (2, 0.5), (3, 0.25), (4, 0.18), (6, 0.08)):
            for det in (-0.6, 0.0, 0.6):                      # chorus = choir-ish pad
                s += wgt * np.sin(2 * np.pi * (f * h + det) * tt + 0.15 * np.sin(2 * np.pi * 5.2 * tt))
    s *= env_adsr(n, att, rel) * vol / (len(freqs) * 3)
    return s


def chime(f, dur=1.4, vol=0.12):
    n = int(dur * SR)
    tt = np.arange(n) / SR
    s = sum(a * np.sin(2 * np.pi * f * r * tt) * np.exp(-tt / (0.6 / r ** 0.5)) for r, a in ((1, 1), (2.76, 0.5), (5.4, 0.25), (8.93, 0.12)))
    s[:60] *= np.linspace(0, 1, 60)
    return s * vol


def bell(f, dur=4.0, vol=0.35):
    n = int(dur * SR)
    tt = np.arange(n) / SR
    parts = ((0.5, 1.0, 3.5), (1.0, 0.8, 2.5), (1.19, 0.6, 2.0), (1.56, 0.45, 1.6), (2.0, 0.5, 1.3), (2.74, 0.3, 1.0), (3.0, 0.25, 0.8), (4.07, 0.15, 0.5))
    s = sum(a * np.sin(2 * np.pi * f * r * tt) * np.exp(-tt / d) for r, a, d in parts)
    s[:80] *= np.linspace(0, 1, 80)
    return s * vol / 2


def tap(seed, vol=0.05):
    n = int(0.03 * SR)
    rng = np.random.default_rng(seed)
    s = rng.normal(0, 1, n) * np.exp(-np.arange(n) / SR / 0.006)
    return np.convolve(s, np.ones(6) / 6, "same") * vol


def hz(name):
    names = {"C": -9, "C#": -8, "D": -7, "D#": -6, "E": -5, "F": -4, "F#": -3, "G": -2, "G#": -1, "A": 0, "A#": 1, "B": 2}
    return 440 * 2 ** ((names[name[:-1]] + 12 * (int(name[-1]) - 4)) / 12)


def build_audio(path):
    n = int((DUR + 0.2) * SR)
    dry = np.zeros(n)

    def add(sig, t0):
        i = int(t0 * SR)
        j = min(n, i + len(sig))
        if j > i:
            dry[i:j] += sig[:j - i]

    chords = [(0.0, 2.9, ["C3", "G3", "C4", "E4"]), (2.8, 5.1, ["A2", "E3", "A3", "C4"]), (5.0, 8.0, ["F2", "C3", "F3", "A3"]),
              (7.9, 10.5, ["G2", "D3", "G3", "B3"]), (10.4, 12.9, ["C3", "G3", "C4", "E4", "G4", "C5"]),
              (12.8, 15.8, ["A2", "E3", "A3", "C4"]), (15.7, 17.7, ["F2", "C3", "F3", "A3"]), (17.6, 18.8, ["G2", "D3", "G3", "B3"]),
              (18.6, 20.2, ["C3", "G3", "C4", "E4", "G4"])]
    for t0, t1, ch in chords:
        add(organ([hz(c) for c in ch], t1 - t0 + 0.4, vol=0.16 if len(ch) > 4 else 0.12), t0)
    # glazing taps
    for k in range(32):
        add(tap(k), G1[0] + (k + 0.5) / 32 * (G1[1] - G1[0]))
    penta = ["C6", "D6", "E6", "G6", "A6", "C7", "D7", "E7"]
    rng = np.random.default_rng(2)
    for win in (M1, M2, BM1, BM2):
        for k in range(10):
            add(chime(hz(penta[rng.integers(0, len(penta))]), vol=0.07), win[0] + (k + rng.uniform(0, 0.8)) / 10 * (win[1] - win[0]))
    for t0 in (T_SP1, T_SP2, T_BSP1, T_BSP2):
        for k, nm in enumerate(["C6", "E6", "G6", "C7", "E7"]):
            add(chime(hz(nm), vol=0.09), t0 + k * 0.06)
    add(bell(hz("C3"), 4.5, 0.5), T_REV[0])
    add(bell(hz("G2"), 2.5, 0.35), T_B0 - 0.3)
    add(bell(hz("C3"), 2.0, 0.45), T_END)
    # cathedral reverb (stereo: two noise IRs)
    out = []
    for seed in (1, 2):
        r = np.random.default_rng(seed)
        ir_n = int(2.6 * SR)
        ir = r.normal(0, 1, ir_n) * np.exp(-np.arange(ir_n) / SR / 0.75)
        ir = np.convolve(ir, np.ones(8) / 8, "same")
        ir[0] = 0
        wet = fftconvolve(dry, ir)[:n]
        wet /= np.abs(wet).max() + 1e-9
        out.append(dry / (np.abs(dry).max() + 1e-9) * 0.75 + wet * 0.45)
    st = np.stack(out, 1)
    fade = np.ones(n)
    fade[-int(0.25 * SR):] = np.linspace(1, 0, int(0.25 * SR))
    st *= fade[:, None]
    st = st / np.abs(st).max() * 0.89
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
    wav = os.path.join(HERE, f"media/final/LoT_stained_glass_{TAG}.wav")
    build_audio(wav)
    out = os.path.join(HERE, f"media/final/LoT_stained_glass_{TAG}_1080p60.mp4")
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
