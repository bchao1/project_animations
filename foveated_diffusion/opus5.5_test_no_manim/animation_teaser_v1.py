#!/usr/bin/env python
"""Foveated Diffusion — 1-minute teaser (v1).

No Manim: frames are drawn with pycairo, text/math is typeset with matplotlib
mathtext (CMU Serif + Computer Modern), and frames are piped into ffmpeg.

Usage:
  python animation_teaser_v1.py --quality l            # 480p15 test
  python animation_teaser_v1.py --quality h            # 1080p60 final
  python animation_teaser_v1.py --stills 2,10,20       # PNG stills at given seconds
"""
import argparse
import math
import os
import subprocess
from multiprocessing import Pool

import cairo
import numpy as np
from PIL import Image, ImageFilter

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

HERE = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(HERE, "ref", "imgs")
FONT_DIR = "/home/brianchc/fonts/cmu-serif"
for _f in ("cmunrm", "cmunbx", "cmunti", "cmunbi"):
    font_manager.fontManager.addfont(os.path.join(FONT_DIR, _f + ".ttf"))
matplotlib.rcParams.update({"font.family": "CMU Serif", "mathtext.fontset": "cm"})

# ---------------------------------------------------------------- constants
W, H = 1920, 1080                      # design units (scaled to output res)
SCALE = 1.0                            # set in main()
FFMPEG = "/usr/bin/ffmpeg"             # has libx264

def hexc(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))

HR = hexc("#448AFF")
LR = hexc("#00C853")
WHITE = (1, 1, 1)
GRAY = hexc("#9E9E9E")
DARK = hexc("#1E1E1E")
RED = hexc("#FF5252")

# token grid used throughout: 16x16 HR tokens, 8x8 LR blocks, fovea = 12 blocks
G, B = 16, 8
_bc = np.arange(B) + 0.5 - B / 2
FOV_BLOCK = (_bc[None, :] ** 2 + _bc[:, None] ** 2) <= 2.5      # [by, bx]
M_HR = int(FOV_BLOCK.sum()) * 4                                  # m = 48
L_SEQ = M_HR + (G * G - M_HR) // 4                               # L = 100
FOV_R = 0.25                                                     # circle radius (image fraction)


# ---------------------------------------------------------------- helpers
def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))

def seg(t, a, b):
    return clamp((t - a) / (b - a))

def ease(x):
    x = clamp(x)
    return x * x * (3 - 2 * x)

def lerp(a, b, u):
    return a + (b - a) * u

def lerpc(c1, c2, u):
    return tuple(lerp(a, b, u) for a, b in zip(c1, c2))


# ---------------------------------------------------------------- text
_TEXT = {}

def _text_surface(s, size, weight, style):
    key = (s, size, weight, style, SCALE)
    if key in _TEXT:
        return _TEXT[key]
    px = size * SCALE
    fig = Figure(figsize=(max(4, len(s) * px / 72 * 0.9 + 1), px * 3 / 72), dpi=72)
    fig.patch.set_alpha(0)
    canvas = FigureCanvasAgg(fig)
    t = fig.text(0.02, 0.5, s, fontsize=px, color="white", va="center",
                 fontweight=weight, fontstyle=style)
    canvas.draw()
    bb = t.get_window_extent(renderer=canvas.get_renderer())
    a = np.asarray(canvas.buffer_rgba())[..., 3]
    fh = a.shape[0]
    pad = 3
    x0, x1 = int(bb.x0) - pad, int(math.ceil(bb.x1)) + pad
    y0, y1 = fh - int(math.ceil(bb.y1)) - pad, fh - int(bb.y0) + pad
    a = a[max(0, y0):y1, max(0, x0):x1]
    h, w = a.shape
    stride = cairo.ImageSurface.format_stride_for_width(cairo.FORMAT_A8, w)
    buf = np.zeros((h, stride), np.uint8)
    buf[:, :w] = a
    surf = cairo.ImageSurface.create_for_data(memoryview(buf), cairo.FORMAT_A8, w, h, stride)
    _TEXT[key] = (surf, w, h, buf)
    return _TEXT[key]

def text(ctx, s, x, y, size=36, color=WHITE, anchor="c", weight="normal", style="normal"):
    """Draw text centered vertically at y. anchor: c / l / r. Returns width (design units)."""
    surf, w, h, _ = _text_surface(s, size, weight, style)
    wd = w / SCALE
    if anchor == "c":
        x -= wd / 2
    elif anchor == "r":
        x -= wd
    ctx.save()
    ctx.scale(1 / SCALE, 1 / SCALE)
    ctx.set_source_rgb(*color)
    ctx.mask_surface(surf, round(x * SCALE), round(y * SCALE - h / 2))
    ctx.restore()
    return wd


# ---------------------------------------------------------------- images
_PIL = {}
_SURF = {}

def pil(name, crop=None):
    key = (name, crop)
    if key not in _PIL:
        im = Image.open(os.path.join(IMG_DIR, name)).convert("RGB")
        if crop:
            im = im.crop(crop)
        _PIL[key] = im
    return _PIL[key]

def surf_from_array(arr):
    """arr: HxWx3 uint8 RGB -> cairo RGB24 surface (keeps buffer alive)."""
    h, w, _ = arr.shape
    buf = np.empty((h, w, 4), np.uint8)
    buf[..., 0] = arr[..., 2]
    buf[..., 1] = arr[..., 1]
    buf[..., 2] = arr[..., 0]
    buf[..., 3] = 255
    s = cairo.ImageSurface.create_for_data(memoryview(buf), cairo.FORMAT_RGB24, w, h, w * 4)
    return s, buf

def img_surface(key, im, w, h):
    k = (key, round(w * SCALE), round(h * SCALE))
    if k not in _SURF:
        r = im.resize((max(1, k[1]), max(1, k[2])), Image.LANCZOS)
        _SURF[k] = surf_from_array(np.asarray(r))
    return _SURF[k][0]

def draw_surface(ctx, s, x, y, alpha=1.0, clip_circle=None):
    ctx.save()
    ctx.scale(1 / SCALE, 1 / SCALE)
    dx, dy = round(x * SCALE), round(y * SCALE)
    if clip_circle is not None:
        cx, cy, r = clip_circle
        ctx.arc(cx * SCALE, cy * SCALE, r * SCALE, 0, 2 * math.pi)
        ctx.clip()
    ctx.set_source_surface(s, dx, dy)
    ctx.paint_with_alpha(alpha)
    ctx.restore()

def draw_img(ctx, name, x, y, w, h, crop=None, alpha=1.0, key=None):
    im = pil(name, crop)
    s = img_surface(key or (name, crop), im, w, h)
    draw_surface(ctx, s, x, y, alpha)

def img_frame(ctx, x, y, w, h, color=WHITE, lw=2):
    ctx.set_source_rgb(*color)
    ctx.set_line_width(lw)
    ctx.rectangle(x, y, w, h)
    ctx.stroke()


# ---------------------------------------------------------------- shapes
def rrect(ctx, x, y, w, h, r):
    ctx.new_sub_path()
    ctx.arc(x + w - r, y + r, r, -math.pi / 2, 0)
    ctx.arc(x + w - r, y + h - r, r, 0, math.pi / 2)
    ctx.arc(x + r, y + h - r, r, math.pi / 2, math.pi)
    ctx.arc(x + r, y + r, r, math.pi, 3 * math.pi / 2)
    ctx.close_path()

def token(ctx, cx, cy, side, color, alpha=1.0):
    if side <= 0.2:
        return
    rrect(ctx, cx - side / 2, cy - side / 2, side, side, 0.1 * side)
    ctx.set_source_rgba(*color, alpha)
    ctx.fill()

def arrow(ctx, x0, y0, x1, y1, color=WHITE, lw=3, head=14, u=1.0):
    if u <= 0:
        return
    x1, y1 = lerp(x0, x1, u), lerp(y0, y1, u)
    ang = math.atan2(y1 - y0, x1 - x0)
    bx, by = x1 - head * math.cos(ang), y1 - head * math.sin(ang)
    ctx.set_source_rgb(*color)
    ctx.set_line_width(lw)
    ctx.move_to(x0, y0)
    ctx.line_to(bx, by)
    ctx.stroke()
    ctx.move_to(x1, y1)
    ctx.line_to(bx + head * 0.5 * math.sin(ang), by - head * 0.5 * math.cos(ang))
    ctx.line_to(bx - head * 0.5 * math.sin(ang), by + head * 0.5 * math.cos(ang))
    ctx.close_path()
    ctx.fill()

def box(ctx, x, y, w, h, label=None, size=34, fill=DARK, stroke=WHITE, lw=2.5):
    rrect(ctx, x, y, w, h, 12)
    ctx.set_source_rgb(*fill)
    ctx.fill_preserve()
    ctx.set_source_rgb(*stroke)
    ctx.set_line_width(lw)
    ctx.stroke()
    if label:
        text(ctx, label, x + w / 2, y + h / 2, size)

def trapezoid(ctx, x, y, w, h, label, wide_left=True):
    """VAE encoder/decoder symbol."""
    s = h * 0.22
    if wide_left:   # decoder: narrow -> wide reads left-to-right as expanding
        pts = [(x, y + s), (x + w, y), (x + w, y + h), (x, y + h - s)]
    else:
        pts = [(x, y), (x + w, y + s), (x + w, y + h - s), (x, y + h)]
    ctx.move_to(*pts[0])
    for p in pts[1:]:
        ctx.line_to(*p)
    ctx.close_path()
    ctx.set_source_rgb(*DARK)
    ctx.fill_preserve()
    ctx.set_source_rgb(*WHITE)
    ctx.set_line_width(2.5)
    ctx.stroke()
    text(ctx, label, x + w / 2, y + h / 2, 36, weight="bold")

def circle(ctx, cx, cy, r, color=WHITE, lw=3, u=1.0, dash=None):
    if u <= 0:
        return
    ctx.save()
    if dash:
        ctx.set_dash(dash)
    ctx.set_source_rgb(*color)
    ctx.set_line_width(lw)
    ctx.arc(cx, cy, r, -math.pi / 2, -math.pi / 2 + 2 * math.pi * u)
    ctx.stroke()
    ctx.restore()

def op_circle(ctx, cx, cy, sym, r=22):
    ctx.set_source_rgb(*DARK)
    ctx.arc(cx, cy, r, 0, 2 * math.pi)
    ctx.fill_preserve()
    ctx.set_source_rgb(*WHITE)
    ctx.set_line_width(2.5)
    ctx.stroke()
    text(ctx, sym, cx, cy, 34)

def header(ctx, s):
    text(ctx, s, W / 2, 80, 56, weight="bold")

def legend(ctx, y=148):
    side = 26
    x = W / 2 - 330
    token(ctx, x, y, side, HR)
    x += 28
    x += text(ctx, "high-res token (fovea)", x, y, 30, anchor="l") + 70
    token(ctx, x, y, side, LR)
    text(ctx, "low-res token (periphery)", x + 28, y, 30, anchor="l")


# ---------------------------------------------------------------- token layout
def grid_tokens():
    """List of HR tokens (gx, gy, block_is_fovea) and LR blocks (bx, by)."""
    hr = [(gx, gy, FOV_BLOCK[gy // 2, gx // 2]) for gy in range(G) for gx in range(G)]
    lr = [(bx, by) for by in range(B) for bx in range(B) if not FOV_BLOCK[by, bx]]
    return hr, lr

def seq_order():
    """Foveated sequence in block raster order: 4 HR tokens per fovea block, 1 LR token otherwise.
    Returns list of (kind, gx, gy) with gx, gy in HR-grid units for HR, block units for LR."""
    out = []
    for by in range(B):
        for bx in range(B):
            if FOV_BLOCK[by, bx]:
                for dy in range(2):
                    for dx in range(2):
                        out.append(("h", 2 * bx + dx, 2 * by + dy))
            else:
                out.append(("l", bx, by))
    return out

SEQ = seq_order()
assert len(SEQ) == L_SEQ

def seq_pos(i, n=L_SEQ, y=880, side=14):
    pitch = side * 1.2
    x0 = W / 2 - (n * pitch - side * 0.2) / 2 + side / 2
    return x0 + i * pitch, y

_rng = np.random.RandomState(3)
NOISE = _rng.uniform(0.2, 0.85, size=(64, L_SEQ))   # gray level per denoising step


# ---------------------------------------------------------------- foveated blur (scene 2)
_LEVELS = {}

def fov_levels(name, w):
    key = (name, round(w * SCALE))
    if key not in _LEVELS:
        n = key[1]
        im = pil(name).resize((n, n), Image.LANCZOS)
        lv = [np.asarray(im, np.float32)]
        for r in (2.0, 4.5, 9.0):
            lv.append(np.asarray(im.filter(ImageFilter.GaussianBlur(r * n / 640)), np.float32))
        yy, xx = np.mgrid[0:n, 0:n] / n
        _LEVELS[key] = (np.stack(lv), xx, yy)
    return _LEVELS[key]

def foveated_surface(name, w, gx, gy, strength):
    lv, xx, yy = fov_levels(name, w)
    d = np.sqrt((xx - gx) ** 2 + (yy - gy) ** 2)
    lvl = np.clip((d - 0.07) / 0.11, 0, 3) * strength
    i0 = np.floor(lvl).astype(int)
    i1 = np.minimum(i0 + 1, 3)
    f = (lvl - i0)[..., None]
    rows, cols = np.indices(d.shape)
    out = lv[i0, rows, cols] * (1 - f) + lv[i1, rows, cols] * f
    return surf_from_array(out.clip(0, 255).astype(np.uint8))


# ============================================================== scenes
def scene_title(ctx, t, dur):
    text(ctx, "Foveated Diffusion", W / 2, 380, 104, weight="bold")
    text(ctx, "Efficient Spatially Adaptive Image and Video Generation", W / 2, 490, 46)
    text(ctx, "Brian Chao*,  Lior Yariv*,  Howard Xiao,  Gordon Wetzstein", W / 2, 600, 34)
    text(ctx, "Stanford University", W / 2, 650, 32, style="italic")
    # token strip: dense (blue) at the center, sparse (green, larger) at the edges
    n_side = 7
    items = []
    x = W / 2 - 5 * 36 / 2 * 1.2
    for i in range(5):
        items.append((W / 2 + (i - 2) * 36 * 1.2, 36, HR))
    xl, xr = W / 2 - 2 * 43.2 - 18, W / 2 + 2 * 43.2 + 18
    for k in range(n_side):
        s = 60
        xl -= s * 0.2 + s / 2 if k == 0 else s * 1.2
        xr += s * 0.2 + s / 2 if k == 0 else s * 1.2
        items.append((xl, s, LR))
        items.append((xr, s, LR))
    for x, s, c in items:
        d = abs(x - W / 2) / (W / 2)
        u = ease(seg(t, 0.3 + d * 1.2, 0.8 + d * 1.2))
        token(ctx, x, 820, s * u, c)


def scene_motivation(ctx, t, dur):
    header(ctx, "Human vision is foveated")
    # left: lion with eccentricity-dependent blur around a moving gaze point
    ix, iy, iw = 150, 210, 660
    ph = max(0.0, t - 0.8)
    gx = 0.5 + 0.22 * math.sin(ph * 1.1)
    gy = 0.48 + 0.16 * math.sin(ph * 1.7 + 0.6)
    strength = ease(seg(t, 0.6, 1.6))
    s, _buf = foveated_surface("x993.png", iw, gx, gy, strength)
    draw_surface(ctx, s, ix, iy)
    if t > 0.6:
        cx, cy = ix + gx * iw, iy + gy * iw
        ctx.set_source_rgb(*RED)
        ctx.arc(cx, cy, 9, 0, 2 * math.pi)
        ctx.fill()
        circle(ctx, cx, cy, 0.16 * iw, WHITE, 2, dash=[8, 6])
    text(ctx, "Sharp only near the gaze point (fovea)", ix + iw / 2, iy + iw + 45, 34)

    # right: quadratic cost of attention
    rx = 1360
    if t > 2.4:
        text(ctx, "Diffusion transformers:  attention cost $\\propto N^2$", rx, 250, 38)
        n = int(round(lerp(4, 16, ease(seg(t, 2.6, 4.6)))))
        side, pitch = 18, 24
        x0 = rx - (16 * pitch) / 2 + 30
        y0 = 330
        for k in range(n):              # key tokens along the top
            token(ctx, x0 + k * pitch, y0, side, HR)
        for k in range(n):              # query tokens along the left
            token(ctx, x0 - 34, y0 + 34 + k * pitch, side, HR)
        for a in range(n):
            for b in range(n):
                token(ctx, x0 + b * pitch, y0 + 34 + a * pitch, side, GRAY, 0.55)
        text(ctx, f"$N = {n}$ tokens  $\\rightarrow$  ${n*n}$ pairs", rx, 780, 34)
    if t > 5.0:
        text(ctx, "Why generate every pixel at full resolution?", W / 2, 1000, 40, weight="bold")


def _grid_geom(x, y, size):
    cell = size / G
    return cell, lambda gx, gy: (x + (gx + 0.5) * cell, y + (gy + 0.5) * cell)

def scene_tokenize(ctx, t, dur):
    header(ctx, "Foveated tokenization")
    legend(ctx)
    ix, iy, sz = 150, 210, 560
    gx0, gy0 = 1210, 210
    cell, hpos = _grid_geom(gx0, gy0, sz)
    bcell = cell * 2
    t_fly = 7.4
    # image + token grid overlay lines
    if t < t_fly + 1.2:
        draw_img(ctx, "x993.png", ix, iy, sz, sz)
        merged = ease(seg(t, 4.0, 5.8))
        ctx.set_line_width(1)
        ctx.set_source_rgba(1, 1, 1, 0.35)
        for gy in range(G):
            for gx in range(G):
                fov = FOV_BLOCK[gy // 2, gx // 2]
                if t > 1.0 and (fov or merged < 1) and not (not fov and merged >= 0.5):
                    ctx.rectangle(ix + gx * cell, iy + gy * cell, cell, cell)
        ctx.stroke()
        if merged >= 0.5:
            ctx.set_source_rgba(1, 1, 1, 0.35)
            for by in range(B):
                for bx in range(B):
                    if not FOV_BLOCK[by, bx]:
                        ctx.rectangle(ix + bx * bcell, iy + by * bcell, bcell, bcell)
            ctx.stroke()
        if t > 2.5:
            circle(ctx, ix + sz / 2, iy + sz / 2, FOV_R * sz, WHITE, 3, u=ease(seg(t, 2.5, 3.3)))
        arrow(ctx, ix + sz + 40, iy + sz / 2, gx0 - 40, iy + sz / 2, u=ease(seg(t, 0.3, 0.9)))
        if t > 0.6:
            text(ctx, "VAE encode", (ix + sz + gx0) / 2, iy + sz / 2 - 36, 30)
            text(ctx, "+ patchify", (ix + sz + gx0) / 2, iy + sz / 2 + 36, 30)
        if t > 2.5:
            text(ctx, "Foveation mask $M$ (from gaze)", ix + sz / 2, iy + sz + 45, 34)

    # token grid -> merge -> fly into 1D sequence
    hr, _ = grid_tokens()
    seq_index = {}
    for i, (k, a, b) in enumerate(SEQ):
        seq_index[(k, a, b)] = i
    fly = ease(seg(t, t_fly, t_fly + 1.6))
    for gx, gy, fov in hr:
        u_in = ease(seg(t, 0.8 + (gx + gy) * 0.03, 1.2 + (gx + gy) * 0.03))
        px, py = hpos(gx, gy)
        side = cell * 0.8 * u_in
        col = HR
        if fov:
            idx = seq_index[("h", gx, gy)]
        else:
            bx, by = gx // 2, gy // 2
            dist = math.hypot(bx + 0.5 - B / 2, by + 0.5 - B / 2)
            u = ease(seg(t, 4.0 + dist * 0.25, 4.8 + dist * 0.25))
            bpx, bpy = gx0 + (bx + 0.5) * bcell, gy0 + (by + 0.5) * bcell
            px, py = lerp(px, bpx, u), lerp(py, bpy, u)
            side = lerp(side, bcell * 0.8, u)
            col = lerpc(HR, LR, u)
            idx = seq_index[("l", bx, by)]
            if u >= 1 and (gx % 2 or gy % 2):
                continue            # the four merged tokens are now one
        if fly > 0:
            sx, sy = seq_pos(idx)
            px, py = lerp(px, sx, fly), lerp(py, sy, fly)
            side = lerp(side, 14, fly)
        token(ctx, px, py, side, col)

    if t < t_fly:
        if 1.6 < t < 4.0:
            text(ctx, "Standard DiT:  $h \\times w = 256$ tokens", gx0 + sz / 2, gy0 + sz + 45, 34)
        elif 4.0 <= t < 6.0:
            text(ctx, "Periphery:  1 low-res token per $2 \\times 2$ block", gx0 + sz / 2, gy0 + sz + 45, 34)
        elif t >= 6.0:
            text(ctx, f"$L = m + (hw - m)/4 = {M_HR} + {L_SEQ - M_HR} = {L_SEQ}$",
                 gx0 + sz / 2, gy0 + sz + 45, 34)
    if t > t_fly + 1.6:
        text(ctx, f"Foveated token sequence $z^{{\\mathrm{{fov}}}}$:  $L = {L_SEQ}$ tokens instead of $256$",
             W / 2, 950, 36)
        text(ctx, "Mixed-resolution RoPE keeps positions consistent across both resolutions",
             W / 2, 1010, 30, color=GRAY)


def scene_generate(ctx, t, dur):
    header(ctx, "Foveated generation")
    legend(ctx)
    row_y = 250
    hr_order = [i for i, s in enumerate(SEQ) if s[0] == "h"]
    # denoising schedule: 8 discrete steps between 1.0s and 4.2s
    steps = 8
    k = int(clamp((t - 1.0) / 3.2) * steps)
    tval = 1 - k / steps
    clean = 1 - tval
    split = ease(seg(t, 4.8, 6.3))
    # --- layout for the lower half
    zg = 240                                   # z grid size
    zx, hy, ly = 150, 470, 750                 # left x, high row top, low row top
    cell = zg / G
    bcell = cell * 2
    for i, (kind, a, b) in enumerate(SEQ):
        mv = ease(seg(t, 0.0, 0.7))
        sx, sy = seq_pos(i, y=lerp(880, row_y, mv))
        base = HR if kind == "h" else LR
        g = NOISE[min(k, 63), i]
        col = lerpc((g, g, g), base, clean ** 0.8) if t > 0.7 else base
        if t <= 0.7:
            col = lerpc(base, (NOISE[0, i],) * 3, mv)
        side = 14
        if split > 0:
            if kind == "h":
                tx, ty = zx + (a + 0.5) * cell, hy + (b + 0.5) * cell
                tside = cell * 0.8
            else:
                tx, ty = zx + (a + 0.5) * bcell, ly + (b + 0.5) * bcell
                tside = bcell * 0.8
            d = (i / L_SEQ) * 0.3
            u = ease(seg(t, 4.8 + d, 6.0 + d))
            sx, sy, side = lerp(sx, tx, u), lerp(sy, ty, u), lerp(side, tside, u)
        token(ctx, sx, sy, side, col)
    # DiT box + step counter
    if 0.7 < t < 4.8:
        box(ctx, W / 2 - 330, 320, 660, 90, None)
        text(ctx, "DiT  (LoRA, mixed-resolution RoPE)", W / 2, 365, 34)
        arrow(ctx, W / 2 - 360, 272, W / 2 - 360, 358, lw=2.5, head=12)
        arrow(ctx, W / 2 + 360, 358, W / 2 + 360, 272, lw=2.5, head=12)
        text(ctx, f"denoise  $t = {tval:.2f}$", W / 2 + 390, 330, 32, anchor="l")
    if 0.7 < t < 4.3:
        text(ctx, "$z_1^{\\mathrm{fov}} \\sim \\mathcal{N}(0, I)$", 150, 330, 34, anchor="l")
    elif t >= 4.3 and t < 4.8:
        text(ctx, "$z_0^{\\mathrm{fov}}$  (clean)", 150, 330, 34, anchor="l")
    if t > 5.0:
        text(ctx, "Split$(z_0^{\\mathrm{fov}}, M)$", W / 2, 330, 36)
    # labels next to z grids
    if t > 6.0:
        text(ctx, "$z_0^{\\mathrm{high}}$", zx + zg / 2, hy - 28, 32)
        text(ctx, "$z_0^{\\mathrm{low}}$", zx + zg / 2, ly - 28, 32)
    # decoders and decoded images
    dx = zx + zg + 50
    imx = dx + 150
    isz = 240
    hc = hy + zg / 2
    lc = ly + zg / 2
    if t > 6.3:
        u = ease(seg(t, 6.3, 6.8))
        arrow(ctx, zx + zg + 10, hc, dx - 5, hc, u=u)
        arrow(ctx, zx + zg + 10, lc, dx - 5, lc, u=u)
        trapezoid(ctx, dx, hc - 60, 90, 120, "D")
        trapezoid(ctx, dx, lc - 60, 90, 120, "D")
        arrow(ctx, dx + 95, hc, imx - 8, hc, u=u)
        arrow(ctx, dx + 95, lc, imx + isz / 4 - 8, lc, u=u)
    if t > 6.8:
        # x_high: only the fovea is decoded at high resolution
        draw_img(ctx, "x993.png", imx, hy, isz, isz, alpha=0.18)
        s = img_surface(("x993.png", None), pil("x993.png"), isz, isz)
        draw_surface(ctx, s, imx, hy, clip_circle=(imx + isz / 2, hy + isz / 2, FOV_R * isz))
        text(ctx, "$x^{\\mathrm{high}}$", imx + isz / 2, hy - 28, 32)
        # x_low: half resolution image
        draw_img(ctx, "x993.png", imx + isz / 4, lc - isz / 4, isz / 2, isz / 2)
        text(ctx, "$x^{\\mathrm{low}}$", imx + isz / 2, lc - isz / 4 - 28, 32)
    # blend
    ox, osz = 1330, 440
    oy = (hc + lc) / 2 - osz / 2
    px = imx + isz + 110
    if t > 7.6:
        u = ease(seg(t, 7.6, 8.2))
        arrow(ctx, imx + isz + 10, hc, px - 26, hc, u=u)
        arrow(ctx, imx + isz * 3 / 4 + 10, lc, px - 26, lc, u=u)
        text(ctx, "Up", (imx + isz * 3 / 4 + px) / 2, lc - 24, 28)
        op_circle(ctx, px, hc, "$\\odot$")
        op_circle(ctx, px, lc, "$\\odot$")
        text(ctx, "$M'$", px, hc - 50, 30)
        text(ctx, "$1 - M'$", px, lc + 52, 30)
        pc = px + 110
        arrow(ctx, px + 24, hc, pc - 16, (hc + lc) / 2 - 18, u=u)
        arrow(ctx, px + 24, lc, pc - 16, (hc + lc) / 2 + 18, u=u)
        op_circle(ctx, pc, (hc + lc) / 2, "$+$")
        arrow(ctx, pc + 24, (hc + lc) / 2, ox - 12, (hc + lc) / 2, u=u)
    if t > 8.2:
        # composite: sharp fovea + upsampled half-res periphery
        low = pil("x993.png").resize((256, 256), Image.BICUBIC)
        s_low = img_surface(("x993_low",), low, osz / 2, osz / 2)
        ctx.save()
        ctx.translate(ox, oy)
        ctx.scale(2, 2)
        ctx.scale(1 / SCALE, 1 / SCALE)
        ctx.set_source_surface(s_low, 0, 0)
        ctx.get_source().set_filter(cairo.FILTER_BILINEAR)
        ctx.paint()
        ctx.restore()
        s = img_surface(("x993.png", None), pil("x993.png"), osz, osz)
        draw_surface(ctx, s, ox, oy, clip_circle=(ox + osz / 2, oy + osz / 2, FOV_R * osz))
        circle(ctx, ox + osz / 2, oy + osz / 2, FOV_R * osz, WHITE, 2, dash=[8, 6])
        text(ctx, "Foveated output $x^{\\mathrm{fov}}$", ox + osz / 2, oy - 30, 32)
    if t > 8.8:
        text(ctx, "$x^{\\mathrm{fov}} = M' \\odot x^{\\mathrm{high}} + (1 - M') \\odot \\mathrm{Up}(x^{\\mathrm{low}})$",
             ox + osz / 2, oy + osz + 55, 32)


def scene_train(ctx, t, dur):
    header(ctx, "Foveated training  (LoRA post-training)")
    legend(ctx)
    ix, iy, isz = 130, 290, 260
    zx, zg = 560, 240
    hy = iy + isz / 2 - zg / 2
    ly = 720
    lc = ly + zg / 2
    hc = iy + isz / 2
    cell = zg / G
    bcell = cell * 2
    draw_img(ctx, "x990.png", ix, iy, isz, isz)
    text(ctx, "Training image $x$", ix + isz / 2, iy - 30, 32)
    if t > 0.5:
        arrow(ctx, ix + isz / 2, iy + isz + 10, ix + isz / 2, lc - isz / 4 - 10, u=ease(seg(t, 0.5, 0.9)))
        text(ctx, "Down", ix + isz / 2 + 20, (iy + isz + lc - isz / 4) / 2, 28, anchor="l")
    if t > 0.9:
        draw_img(ctx, "x990.png", ix + isz / 4, lc - isz / 4, isz / 2, isz / 2)
        text(ctx, "$x^{\\mathrm{low}}$", ix + isz / 4 - 20, lc, 30, anchor="r")
    ex = 420
    if t > 1.2:
        u = ease(seg(t, 1.2, 1.6))
        arrow(ctx, ix + isz + 8, hc, ex - 6, hc, u=u)
        arrow(ctx, ix + isz * 3 / 4 + 8, lc, ex - 6, lc, u=u)
        trapezoid(ctx, ex, hc - 60, 90, 120, "E", wide_left=False)
        trapezoid(ctx, ex, lc - 60, 90, 120, "E", wide_left=False)
        arrow(ctx, ex + 96, hc, zx - 8, hc, u=u)
        arrow(ctx, ex + 96, lc, zx - 8, lc, u=u)
    mx, my = 1000, (hy + ly) / 2          # merged grid top-left
    merge = t > 3.0
    if t > 1.6:
        for gy in range(G):
            for gx in range(G):
                u_in = ease(seg(t, 1.6 + (gx + gy) * 0.02, 2.0 + (gx + gy) * 0.02))
                px, py = zx + (gx + 0.5) * cell, hy + (gy + 0.5) * cell
                token(ctx, px, py, cell * 0.8 * u_in, HR, 0.35 if merge and not FOV_BLOCK[gy // 2, gx // 2] else 1)
        for by in range(B):
            for bx in range(B):
                u_in = ease(seg(t, 1.6 + (bx + by) * 0.04, 2.0 + (bx + by) * 0.04))
                px, py = zx + (bx + 0.5) * bcell, ly + (by + 0.5) * bcell
                token(ctx, px, py, bcell * 0.8 * u_in, LR, 0.35 if merge and FOV_BLOCK[by, bx] else 1)
        text(ctx, "$z_0^{\\mathrm{high}} = E(x)$", zx + zg / 2, hy - 30, 32)
        text(ctx, "$z_0^{\\mathrm{low}} = E(\\mathrm{Down}(x))$", zx + zg / 2, ly - 30, 32)
    if merge:
        u = ease(seg(t, 3.0, 4.4))
        for gy in range(G):
            for gx in range(G):
                if FOV_BLOCK[gy // 2, gx // 2]:
                    sx, sy = zx + (gx + 0.5) * cell, hy + (gy + 0.5) * cell
                    tx, ty = mx + (gx + 0.5) * cell, my + (gy + 0.5) * cell
                    token(ctx, lerp(sx, tx, u), lerp(sy, ty, u), cell * 0.8, HR)
        for by in range(B):
            for bx in range(B):
                if not FOV_BLOCK[by, bx]:
                    sx, sy = zx + (bx + 0.5) * bcell, ly + (by + 0.5) * bcell
                    tx, ty = mx + (bx + 0.5) * bcell, my + (by + 0.5) * bcell
                    token(ctx, lerp(sx, tx, u), lerp(sy, ty, u), bcell * 0.8, LR)
        text(ctx, "Merge$(z_0^{\\mathrm{high}}, z_0^{\\mathrm{low}}, M)$", mx + zg / 2, my - 30, 32)
        text(ctx, "$= z_0^{\\mathrm{fov}}$", mx + zg / 2, my + zg + 34, 32)
    if t > 4.6:
        u = ease(seg(t, 4.6, 5.0))
        dxb = 1400
        arrow(ctx, mx + zg + 12, my + zg / 2, dxb - 10, my + zg / 2, u=u)
        text(ctx, "add noise", (mx + zg + dxb) / 2, my + zg / 2 - 30, 28)
        box(ctx, dxb, my + 20, 200, zg - 40, None)
        text(ctx, "DiT", dxb + 100, my + 80, 38, weight="bold")
        rrect(ctx, dxb + 40, my + 120, 120, 60, 10)
        ctx.set_source_rgb(*hexc("#FF9800"))
        ctx.fill()
        text(ctx, "LoRA", dxb + 100, my + 150, 30, weight="bold")
        arrow(ctx, dxb + 210, my + zg / 2, dxb + 290, my + zg / 2, u=u)
        text(ctx, "$v_\\theta$", dxb + 320, my + zg / 2, 38)
    if t > 5.3:
        text(ctx, "Flow matching:  $\\| v_\\theta(z_t^{\\mathrm{fov}}, t) - (z_1^{\\mathrm{fov}} - z_0^{\\mathrm{fov}}) \\|^2$",
             1500, my + zg + 110, 32)
    if t > 5.8:
        text(ctx, "Both resolutions come from the same image $\\rightarrow$ consistent content across fovea and periphery",
             W / 2, 1020, 32, color=GRAY)


# ------------------------------------------------------------- results
def _cell(r, c, starts=(0, 521, 1043, 1566, 2087, 2609), rows=(0, 522, 1044), s=512):
    return (starts[c], rows[r], starts[c] + s, rows[r] + s)

def scene_res_image(ctx, t, dur):
    header(ctx, "Results: image generation  (FLUX.2 Klein 4B)")
    labels = ["Full high-res", "Naïve mixed-res", "Ours"]
    sz, gap = 370, 30
    x0 = W / 2 - (3 * sz + 2 * gap) / 2
    for c in range(3):
        cx = x0 + c * (sz + gap)
        text(ctx, labels[c], cx + sz / 2, 170, 36, weight="bold" if c == 2 else "normal",
             color=HR if c == 2 else WHITE)
        for r, (row, col0) in enumerate([(0, 0), (0, 3)]):
            draw_img(ctx, "x1018.png", cx, 205 + r * (sz + 20), sz, sz, crop=_cell(row, col0 + c))
    text(ctx, "Naïve mixed-resolution denoising breaks scale and structure.  Ours matches full resolution.",
         W / 2, 1025, 32)


def scene_res_video(ctx, t, dur):
    header(ctx, "Results: video generation  (Wan2.1 1.3B)")
    labels = ["Full high-res", "Naïve mixed-res", "Ours"]
    fw, fh = 560, 322
    gap = 40
    x0 = W / 2 - (3 * fw + 2 * gap) / 2
    fr = int(t * 2.5) % 5
    for c in range(3):
        cx = x0 + c * (fw + gap)
        text(ctx, labels[c], cx + fw / 2, 250, 36, weight="bold" if c == 2 else "normal",
             color=HR if c == 2 else WHITE)
        y0 = c * 378
        crop = (fr * 640, y0, fr * 640 + 640, y0 + 368)
        draw_img(ctx, "x1050.png", cx, 290, fw, fh, crop=crop)
    text(ctx, "Naïve: duplicated entities near the fovea boundary", x0 + fw + gap + fw / 2, 290 + fh + 45, 30, color=GRAY)
    text(ctx, "3.5${\\times}$ faster", W / 2, 780, 64, weight="bold", color=HR)
    text(ctx, "at 38% of the tokens,  VBench quality on par with full high-res", W / 2, 860, 36)


def scene_res_numbers(ctx, t, dur):
    header(ctx, "Faster, and perceptually on par")
    # left: speedup bars
    lx, bw = 170, 460
    text(ctx, "Speedup over full high-res (25% tokens)", lx + 360, 250, 34, weight="bold")
    for i, (name, val) in enumerate([("Image", 2.0), ("Video", 4.0)]):
        y = 360 + i * 130
        text(ctx, name, lx + 110, y, 36, anchor="r")
        u = ease(seg(t, 0.3 + i * 0.4, 1.3 + i * 0.4))
        w = bw * val / 4.0 * u
        rrect(ctx, lx + 140, y - 35, max(w, 1), 70, 10)
        ctx.set_source_rgb(*(HR if i == 0 else LR))
        ctx.fill()
        if u > 0.99:
            text(ctx, f"> {val:.0f}" + "${\\times}$", lx + 160 + w, y, 40, anchor="l", weight="bold")
    text(ctx, "Image @ 30% tokens:  1.98${\\times}$,  HPSv2.1 0.279 vs. 0.280 (full high-res)", lx + 360, 640, 30)
    # right: 2AFC user study
    rx, rw = 1010, 560
    text(ctx, "User study (2AFC, gaze-contingent)", rx + rw / 2, 250, 34, weight="bold")
    rows = [("Ours vs. full high-res", 47.3, "full high-res"), ("Ours vs. naïve", 87.4, "naïve")]
    for i, (name, p, other) in enumerate(rows):
        y = 360 + i * 170
        text(ctx, name, rx, y - 55, 30, anchor="l")
        u = ease(seg(t, 1.0 + i * 0.5, 2.0 + i * 0.5))
        wo = rw * p / 100 * u
        rrect(ctx, rx, y - 30, rw, 60, 8)
        ctx.set_source_rgb(*hexc("#555555"))
        ctx.fill()
        rrect(ctx, rx, y - 30, max(wo, 1), 60, 8)
        ctx.set_source_rgb(*HR)
        ctx.fill()
        if u > 0.99:
            text(ctx, f"ours {p:.1f}%", rx + 14, y, 28, anchor="l", weight="bold")
            text(ctx, f"{100 - p:.1f}%  {other}", rx + rw + 14, y, 26, anchor="l", color=GRAY)
    text(ctx, "near parity with full resolution;  strongly preferred over naïve", rx + rw / 2, 720, 30, color=GRAY)
    # bottom row: applications
    if t > 2.6:
        apps = [("x822.png", "Saliency-guided"), ("x821.png", "Multiple foveae"), ("x820.png", "VR gaming"),
                ("x832.png", "Robotics"), ("x833.png", "Self-driving"), ("x831.png", "Aerial footage")]
        s, g = 200, 30
        ax0 = W / 2 - (len(apps) * s + (len(apps) - 1) * g) / 2
        for i, (f, lab) in enumerate(apps):
            if t > 2.6 + i * 0.12:
                draw_img(ctx, f, ax0 + i * (s + g), 800, s, s)
                text(ctx, lab, ax0 + i * (s + g) + s / 2, 1035, 26)


def scene_end(ctx, t, dur):
    text(ctx, "Foveated Diffusion", W / 2, 420, 100, weight="bold")
    text(ctx, "Spend tokens where the eye looks.", W / 2, 530, 44, style="italic")
    text(ctx, "Up to 2${\\times}$ faster images   ·   4${\\times}$ faster videos", W / 2, 620, 38)
    text(ctx, "bchao1.github.io/foveated-diffusion", W / 2, 740, 36, color=HR)


TIMELINE = [
    (scene_title, 4.0),
    (scene_motivation, 7.5),
    (scene_tokenize, 11.5),
    (scene_generate, 12.0),
    (scene_train, 8.0),
    (scene_res_image, 4.0),
    (scene_res_video, 4.0),
    (scene_res_numbers, 5.5),
    (scene_end, 3.5),
]
TOTAL = sum(d for _, d in TIMELINE)


def render(t):
    wp, hp = round(W * SCALE), round(H * SCALE)
    surf = cairo.ImageSurface(cairo.FORMAT_RGB24, wp, hp)
    ctx = cairo.Context(surf)
    ctx.set_source_rgb(0, 0, 0)
    ctx.paint()
    ctx.scale(SCALE, SCALE)
    start = 0.0
    for fn, dur in TIMELINE:
        if t < start + dur or fn is TIMELINE[-1][0]:
            fn(ctx, t - start, dur)
            break
        start += dur
    surf.flush()
    return surf


def _frame_bytes(args):
    i, fps = args
    s = render(i / fps)
    return bytes(s.get_data())


def main():
    global SCALE
    ap = argparse.ArgumentParser()
    ap.add_argument("--quality", choices=["l", "h"], default="l")
    ap.add_argument("--stills", default=None, help="comma-separated seconds")
    ap.add_argument("--workers", type=int, default=48)
    a = ap.parse_args()
    fps, hp = (15, 480) if a.quality == "l" else (60, 1080)
    wp = 854 if a.quality == "l" else 1920      # even width for yuv420p
    SCALE = wp / W
    media = os.path.join(HERE, "media")
    if a.stills:
        out = os.path.join(media, "stills")
        os.makedirs(out, exist_ok=True)
        for s in a.stills.split(","):
            render(float(s)).write_to_png(os.path.join(out, f"t{float(s):05.2f}.png"))
        print("stills ->", out)
        return
    wp, hp = round(W * SCALE), round(H * SCALE)
    out = os.path.join(media, "videos", f"{hp}p{fps}")
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "FoveatedDiffusionTeaser_v1.mp4")
    n = int(round(TOTAL * fps))
    cmd = [FFMPEG, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgra",
           "-s", f"{wp}x{hp}", "-r", str(fps), "-i", "-",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", "-preset", "slow",
           "-movflags", "+faststart", path]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    with Pool(a.workers, initializer=_init_worker, initargs=(SCALE,)) as pool:
        for k, b in enumerate(pool.imap(_frame_bytes, [(i, fps) for i in range(n)], chunksize=2)):
            proc.stdin.write(b)
            if k % (fps * 5) == 0:
                print(f"frame {k}/{n}", flush=True)
    proc.stdin.close()
    proc.wait()
    print("video ->", path, f"({TOTAL:.1f}s, {n} frames)")


def _init_worker(scale):
    global SCALE
    SCALE = scale


if __name__ == "__main__":
    main()
