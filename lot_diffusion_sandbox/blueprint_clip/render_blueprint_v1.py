"""LoT Diffusion, 20-s blueprint / architect clip, v1 (numpy + PIL + cv2 frames → ffmpeg, synthesized foley + lo-fi pad).

The token layout is drafted like an architectural plan before anything is rendered.
SHEET 1 OF 2 (giraffe, depth-of-field cue)
  0.0–0.9   blank cyanotype sheet, border and title block draw on
  0.9–2.3   depth survey: contour lines of the depth map, the in-focus band hatched
  2.4–4.3   plotter pens draw the full grid: one 1×1 token per 16-px patch (4,096)
  4.6–6.0   REV 1: eraser takes out lines in the blurry areas → 2×2 (1,606 · 1.9× faster), revision cloud
  6.6–8.0   REV 2: → 4×4 / 8×8 = the real layout (1,027 · 2.4× faster)
  8.4       red stamp APPROVED: 1,027 TOKENS
  9.2–10.8  the image is built token by token inside the plan
SHEET 2 OF 2 (platformer video)
 11.6–12.3  sheet change
 12.5–17.0  video left, its plan right: grid (75,600) → REV 1 ground 2×2 (2.2×) → REV 2 sky 4×4 (2.5×);
            playback speed follows the speedup
 17.6       stamp APPROVED: 34,698 TOKENS, closing line
Speedups: image = Amdahl fit of the paper's LoT-FLUX.2 4B timings; video = interpolation of Tab. 2 (LoT-Wan2.1 14B).
Usage: python render_blueprint_v1.py h|l          → media/final/LoT_blueprint_v1_<res>.mp4
       python render_blueprint_v1.py still t ...   → scratch/still_v1_<t>.png
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
from scipy import ndimage, signal
from scipy.io import wavfile
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TAG = "v1"
HAND = os.path.join(HERE, "data/ArchitectsDaughter-Regular.ttf")
MONO = os.path.join(HERE, "data/IBMPlexMono-Regular.ttf")
MONO_B = os.path.join(HERE, "data/IBMPlexMono-Bold.ttf")
CUE = os.path.join(ROOT, "data/img/img_000000000025_dof_cue.png")
GEN = os.path.join(ROOT, "data/img/img_000000000025_dof_generated_4b.png")
LAYOUT = os.path.join(ROOT, "data/layouts/img_000000000025_dof.json")
VID = os.path.join(ROOT, "data/clips/vd_platformer_gen")
LEVELS = os.path.join(ROOT, "arcade_clip/data/platformer_levels.npy")

W, H = 1920, 1080
DUR = 20.0

PAPER = np.array((18, 58, 112), np.float32)          # Prussian-blue cyanotype
INK = (228, 242, 255)
INK_F = np.array(INK, np.float32)
CYAN = (140, 215, 255)
RED = (214, 40, 40)

# ------------------------------------------------------------------ timeline (s)
T_SHEET = 0.0
T_SURVEY = (0.9, 2.3)
T_GRID = (2.4, 4.3)
T_REV1 = (4.6, 6.0)
T_REV2 = (6.6, 8.0)
T_STAMP1 = 8.4
T_BUILD = (9.2, 10.8)
T_SWAP = (11.6, 12.3)
T_VGRID = (12.5, 13.5)
T_VREV1 = (14.0, 15.0)
T_VREV2 = (15.8, 16.8)
T_STAMP2 = 17.6
T_CLOSE = 18.4

IMG_TOK = (4096, 1606, 1027)
IMG_SPEED = (1.0, 1.9, 2.4)
VID_TOK = (75600, 38046, 34698)
VID_SPEED = (1.0, 2.2, 2.5)

# ------------------------------------------------------------------ geometry
N, CELL = 64, 12
P = N * CELL                       # 768
PX0, PY0 = 200, 170                # sheet-1 plan
NOTE_X, NOTE_Y = 1170, 180         # sheet-1 notes column
TB = (1170, 790, 1830, 1010)       # title block
VN, VM, VC = 45, 80, 10            # video tokens, 10 px each → 800 × 450
VW, VH = VM * VC, VN * VC
VXL, VXR, VY0 = 110, 1010, 190


def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def ease(x):
    x = clamp(x)
    return x * x * (3 - 2 * x)


def ease_out(x):
    x = clamp(x)
    return 1 - (1 - x) ** 3


def frac(t, win):
    return clamp((t - win[0]) / (win[1] - win[0]))


# ------------------------------------------------------------------ assets
def load_assets():
    a = {}
    rng = np.random.default_rng(1)
    # paper: blue + low-frequency mottling + fine grain + faint drafting grid + one fold crease
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    mott = cv2.GaussianBlur(rng.normal(0, 1, (H // 8, W // 8)).astype(np.float32), (0, 0), 6)
    mott = cv2.resize(mott / (np.abs(mott).max() + 1e-6), (W, H))
    grain = rng.normal(0, 1, (H, W)).astype(np.float32)
    paper = PAPER[None, None] * (1 + 0.10 * mott[..., None]) + 5 * grain[..., None]
    g = ((xx % 30 < 1) | (yy % 30 < 1)).astype(np.float32) * 0.07 + ((xx % 150 < 1) | (yy % 150 < 1)) * 0.06
    paper = paper * (1 - g[..., None]) + INK_F * g[..., None]
    crease = np.exp(-((xx - 0.52 * W) ** 2) / (2 * 3.0 ** 2))
    paper += crease[..., None] * 10
    vig = 1 - 0.25 * (((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
    a["paper"] = np.clip(paper * vig[..., None], 0, 255).astype(np.float32)

    # depth cue → scalar via turbo inverse
    cue = np.asarray(Image.open(CUE).convert("RGB").resize((P, P), Image.BOX)).astype(np.float32)
    lut = cv2.applyColorMap(np.arange(256, dtype=np.uint8)[:, None], cv2.COLORMAP_TURBO)[:, 0, ::-1].astype(np.float32)
    _, s = cKDTree(lut).query(cue.reshape(-1, 3))
    s = cv2.GaussianBlur(s.reshape(P, P).astype(np.float32), (0, 0), 3)
    _, focus = cKDTree(lut).query(np.array([44, 208, 165], np.float32))
    lvl = np.floor(s / 18).astype(int)
    cont = np.zeros((P, P), bool)
    cont[:, 1:] |= lvl[:, 1:] != lvl[:, :-1]
    cont[1:, :] |= lvl[1:, :] != lvl[:-1, :]
    a["contour"] = cont
    yy2, xx2 = np.mgrid[0:P, 0:P]
    infocus = np.abs(s - focus) < 22
    infocus = ndimage.binary_opening(infocus, iterations=2)
    a["infocus"] = infocus
    a["hatch"] = infocus & (((xx2 + yy2) % 9) < 2)
    fe = infocus ^ ndimage.binary_erosion(infocus, iterations=2)
    a["focus_edge"] = fe

    gen = np.asarray(Image.open(GEN).convert("RGB")).astype(np.float32)
    a["gen"] = cv2.resize(gen, (P, P), interpolation=cv2.INTER_AREA)

    rects = [tuple(r) for r in json.load(open(LAYOUT))["rects"]]
    a["rects"] = rects
    size = np.zeros((N, N), int)
    r0m = np.zeros((N, N), int)
    c0m = np.zeros((N, N), int)
    for r0, r1, c0, c1 in rects:
        size[r0:r1, c0:c1] = r1 - r0
        r0m[r0:r1, c0:c1] = r0
        c0m[r0:r1, c0:c1] = c0
    ii, jj = np.mgrid[0:N, 0:N]
    a["blocks"] = {}
    for cap in (1, 2, 8):
        b = np.minimum(size, cap)
        a["blocks"][cap] = (r0m + (ii - r0m) // b * b, c0m + (jj - c0m) // b * b, b)
    # build order: bottom rows first, a little jitter
    rr = np.random.default_rng(4)
    a["build_t"] = {}
    for k, (r0, r1, c0, c1) in enumerate(rects):
        a["build_t"][(r0, c0)] = (1 - r1 / N) * 0.85 + rr.uniform(0, 0.15)

    a["lv"] = np.load(LEVELS)
    a["vid"] = [cv2.resize(np.asarray(Image.open(os.path.join(VID, f"{f + 1:04d}.jpg")).convert("RGB")),
                           (VW, VH), interpolation=cv2.INTER_AREA).astype(np.float32) for f in range(81)]
    return a


A = None


def assets():
    global A
    if A is None:
        A = load_assets()
    return A


# ------------------------------------------------------------------ text
@lru_cache(maxsize=64)
def font(path, size):
    return ImageFont.truetype(path, size)


def text(d, s, xy, size, fill=INK, path=HAND, anchor="la", reveal=1.0):
    """reveal < 1 writes the string on progressively (handwriting)."""
    if reveal <= 0:
        return
    k = max(1, int(round(len(s) * clamp(reveal))))
    f = font(path, size)
    if anchor == "la":
        d.text(xy, s[:k], font=f, fill=fill)
    else:
        # keep the final position fixed while writing: measure the full string
        w = f.getlength(s)
        x, y = xy
        if anchor[0] == "m":
            x -= w / 2
        elif anchor[0] == "r":
            x -= w
        d.text((x, y), s[:k], font=f, fill=fill, anchor="l" + anchor[1])


def write_frac(t, t0, chars, cps=26.0):
    return clamp((t - t0) * cps / max(1, chars))


# ------------------------------------------------------------------ token ids / edges
def block_ids(oi, oj, m):
    return oi * m + oj


def edges(ids_px, thick=2):
    e = np.zeros(ids_px.shape, bool)
    e[:, 1:] |= ids_px[:, 1:] != ids_px[:, :-1]
    e[1:, :] |= ids_px[1:, :] != ids_px[:-1, :]
    if thick > 1:
        e[:, :-1] |= e[:, 1:]
        e[:-1, :] |= e[1:, :]
    e[:thick, :] = e[-thick:, :] = True
    e[:, :thick] = e[:, -thick:] = True
    return e


def up(arr, c):
    return np.repeat(np.repeat(arr, c, 0), c, 1)


def revision_lines(old, new, n, m, c, win, t, thick):
    """line mask (float alpha) while an eraser sweep turns `old` blocks into `new` blocks top → bottom."""
    oi_o, oj_o, _ = old
    oi_n, oj_n, b_n = new
    e_old = edges(up(block_ids(oi_o, oj_o, m), c), thick)
    e_new = edges(up(block_ids(oi_n, oj_n, m), c), thick)
    t_sw = win[0] + (oi_n + b_n) / n * (win[1] - win[0])          # block erased once the sweep passes its bottom
    fade = np.clip((t - up(t_sw, c)) / 0.18, 0, 1)
    gone = e_old & ~e_new
    alpha = e_new.astype(np.float32)
    alpha[gone] = 1 - fade[gone] * 0.88                            # erased lines leave a faint ghost
    return alpha


# ------------------------------------------------------------------ drawing helpers
def blend(img, mask, color, alpha):
    if isinstance(mask, np.ndarray) and mask.dtype == bool:
        mask = mask.astype(np.float32)
    m = (mask * alpha)[..., None]
    return img * (1 - m) + np.asarray(color, np.float32) * m


def draw_pen(d, x, y, ang=-35):
    """technical pen: nib tip at (x, y)."""
    a = math.radians(ang)
    ux, uy = math.cos(a), math.sin(a)                # along the pen, away from the tip (up-right)
    px, py = -uy, ux
    pts_tip = [(x, y), (x + 18 * ux + 4 * px, y + 18 * uy + 4 * py), (x + 18 * ux - 4 * px, y + 18 * uy - 4 * py)]
    body = [(x + 18 * ux + 9 * px, y + 18 * uy + 9 * py), (x + 120 * ux + 9 * px, y + 120 * uy + 9 * py),
            (x + 120 * ux - 9 * px, y + 120 * uy - 9 * py), (x + 18 * ux - 9 * px, y + 18 * uy - 9 * py)]
    sh = 10
    d.polygon([(p[0] + sh, p[1] + sh) for p in body], fill=(8, 28, 60, 140))
    d.polygon(body, fill=(30, 34, 40, 255), outline=INK + (255,))
    band = [(x + 70 * ux + 9.5 * px, y + 70 * uy + 9.5 * py), (x + 80 * ux + 9.5 * px, y + 80 * uy + 9.5 * py),
            (x + 80 * ux - 9.5 * px, y + 80 * uy - 9.5 * py), (x + 70 * ux - 9.5 * px, y + 70 * uy - 9.5 * py)]
    d.polygon(band, fill=RED + (255,))
    d.polygon(pts_tip, fill=(200, 205, 215, 255), outline=INK + (255,))


def draw_eraser(d, x, y):
    w, h = 70, 34
    d.rounded_rectangle([x - w / 2 + 8, y - h / 2 + 8, x + w / 2 + 8, y + h / 2 + 8], 6, fill=(8, 28, 60, 140))
    d.rounded_rectangle([x - w / 2, y - h / 2, x + w / 2, y + h / 2], 6, fill=(238, 150, 160, 255), outline=INK + (255,), width=2)
    d.rectangle([x - w / 2 + 2, y - h / 2 + 2, x - w / 2 + 26, y + h / 2 - 2], fill=(80, 120, 200, 255))


def arrow_head(d, x, y, dx, dy, fill, s=10):
    n = math.hypot(dx, dy) or 1
    ux, uy = dx / n, dy / n
    d.polygon([(x, y), (x - s * ux + s * 0.45 * uy, y - s * uy - s * 0.45 * ux),
               (x - s * ux - s * 0.45 * uy, y - s * uy + s * 0.45 * ux)], fill=fill)


def dimension(d, x0, x1, y, label, alpha, size=24, label_dy=-34):
    """horizontal dimension line with arrows, extension ticks and a label."""
    if alpha <= 0:
        return
    c = INK + (int(255 * alpha),)
    xe = x0 + (x1 - x0) * ease_out(alpha * 1.4)
    d.line([(x0, y), (xe, y)], fill=c, width=2)
    d.line([(x0, y - 9), (x0, y + 9)], fill=c, width=2)
    if alpha > 0.7:
        d.line([(x1, y - 9), (x1, y + 9)], fill=c, width=2)
        arrow_head(d, x0, y, -1, 0, c)
        arrow_head(d, x1, y, 1, 0, c)
    f = font(HAND, size)
    w = f.getlength(label)
    cx = (x0 + x1) / 2
    bx0, by0 = cx - w / 2 - 8, y + label_dy - 4
    d.rectangle([bx0, by0, bx0 + w + 16, by0 + size + 12], fill=(18, 58, 112, int(235 * alpha)))
    text(d, label, (cx, y + label_dy + size // 2 + 2), size, fill=c, anchor="mm", reveal=alpha * 1.3)


def cloud(d, x0, y0, x1, y1, prog, tag, r=13):
    """revision cloud drawn on along its perimeter, with a revision triangle tag."""
    if prog <= 0:
        return
    per = 2 * ((x1 - x0) + (y1 - y0))
    n = int(per / (1.6 * r))
    step = per / n
    pts = []
    for k in range(n):
        s = k * step
        if s < x1 - x0:
            pts.append((x0 + s, y0, -90))
        elif s < (x1 - x0) + (y1 - y0):
            pts.append((x1, y0 + s - (x1 - x0), 0))
        elif s < 2 * (x1 - x0) + (y1 - y0):
            pts.append((x1 - (s - (x1 - x0) - (y1 - y0)), y1, 90))
        else:
            pts.append((x0, y1 - (s - 2 * (x1 - x0) - (y1 - y0)), 180))
    m = int(n * clamp(prog))
    for x, y, ang in pts[:m]:
        d.arc([x - r, y - r, x + r, y + r], ang - 100, ang + 100 + 0, fill=CYAN + (255,), width=3)
    if prog >= 1:
        tx, ty = x1 - 10, y0 - 34
        d.polygon([(tx, ty - 26), (tx + 30, ty + 26), (tx - 30, ty + 26)], fill=(18, 58, 112, 255), outline=CYAN + (255,), width=3)
        text(d, tag, (tx, ty + 10), 26, fill=CYAN, path=MONO_B, anchor="mm")


@lru_cache(maxsize=4)
def stamp_img(line1, line2):
    w, h = 560, 200
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([6, 6, w - 6, h - 6], 18, outline=RED + (255,), width=10)
    d.rounded_rectangle([22, 22, w - 22, h - 22], 10, outline=RED + (255,), width=4)
    d.text((w / 2, 82), line1, font=font(MONO_B, 76), fill=RED + (255,), anchor="mm")
    d.text((w / 2, 150), line2, font=font(MONO_B, 38), fill=RED + (255,), anchor="mm")
    # ink grunge: knock out speckles
    rng = np.random.default_rng(9)
    al = np.asarray(im.getchannel("A")).astype(np.float32)
    noise = cv2.GaussianBlur(rng.uniform(0, 1, (h, w)).astype(np.float32), (0, 0), 1.5)
    al *= np.clip((noise - 0.36) * 6, 0.25, 1)
    im.putalpha(Image.fromarray(al.astype(np.uint8)))
    return im.rotate(9, resample=Image.BICUBIC, expand=True)


def stamp(canvas, t, t0, cx, cy, line1, line2):
    dt = t - t0
    if dt < 0:
        return
    im = stamp_img(line1, line2)
    s = 1 + 0.6 * (1 - ease_out(dt / 0.12)) if dt < 0.12 else 1.0
    a = clamp(dt / 0.06)
    if abs(s - 1) > 1e-3:
        im = im.resize((int(im.width * s), int(im.height * s)), Image.BICUBIC)
    if a < 1:
        im = im.copy()
        im.putalpha(im.getchannel("A").point(lambda v: int(v * a)))
    canvas.alpha_composite(im, (int(cx - im.width / 2), int(cy - im.height / 2)))


def shake(t):
    for t0 in (T_STAMP1, T_STAMP2):
        dt = t - t0 - 0.1
        if 0 <= dt < 0.25:
            k = 7 * math.exp(-dt / 0.06)
            return int(k * math.sin(dt * 120)), int(k * math.cos(dt * 97))
    return 0, 0


def title_block(d, t, x0, sheet, drawing, tokens, speed, rev, alpha=1.0, extra=None):
    x1, y0, y1 = x0 + (TB[2] - TB[0]), TB[1], TB[3]
    c = INK + (int(255 * alpha),)
    d.rectangle([x0, y0, x1, y1], outline=c, width=3)
    rows = [y0, y0 + 64, y0 + 112, y0 + 160, y1]
    for y in rows[1:-1]:
        d.line([(x0, y), (x1, y)], fill=c, width=2)
    xm = x0 + 330
    d.line([(xm, rows[2]), (xm, y1)], fill=c, width=2)
    lab = font(MONO, 15)
    d.text((x0 + 12, y0 + 6), "PROJECT", font=lab, fill=CYAN + (int(255 * alpha),))
    d.text((x0 + 12, y0 + 24), "LEVEL-OF-TOKEN DIFFUSION", font=font(MONO_B, 30), fill=c)
    d.text((x0 + 12, rows[1] + 4), "DRAWING", font=lab, fill=CYAN + (int(255 * alpha),))
    d.text((x0 + 110, rows[1] + 10), drawing, font=font(MONO, 26), fill=c)
    d.text((x0 + 12, rows[2] + 4), "TOKENS", font=lab, fill=CYAN + (int(255 * alpha),))
    d.text((x0 + 110, rows[2] + 8), tokens, font=font(MONO_B, 30), fill=c)
    d.text((xm + 12, rows[2] + 4), "SPEED", font=lab, fill=CYAN + (int(255 * alpha),))
    d.text((xm + 100, rows[2] + 8), speed, font=font(MONO_B, 30), fill=c)
    d.text((x0 + 12, rows[3] + 4), "SHEET", font=lab, fill=CYAN + (int(255 * alpha),))
    d.text((x0 + 110, rows[3] + 10), sheet, font=font(MONO, 26), fill=c)
    d.text((xm + 12, rows[3] + 4), "REV", font=lab, fill=CYAN + (int(255 * alpha),))
    d.text((xm + 100, rows[3] + 10), rev, font=font(MONO, 26), fill=c)


def fmt(n):
    return f"{int(round(n)):,}"


# ------------------------------------------------------------------ sheet 1
def sheet1(t):
    """returns (rgb plan P×P, overlay draw callback list)."""
    a = assets()
    plan = np.zeros((P, P, 3), np.float32)        # composited later on paper; we work on a copy of paper crop
    return plan


def sheet1_layer(base, canvas, d, t, xoff):
    a = assets()
    x0, y0 = PX0 + xoff, PY0
    crop = base[y0:y0 + P, max(0, x0):x0 + P].copy() if x0 >= 0 else None
    if crop is None or crop.shape[1] != P:
        crop = np.repeat(np.repeat(PAPER[None, None], P, 0), P, 1).copy()
        full = False
    else:
        full = True
    img = crop
    # survey: contours wipe on top → bottom, hatch after
    sv = frac(t, T_SURVEY)
    build = frac(t, T_BUILD)
    if sv > 0:
        rows = int(P * ease(sv / 0.7))
        cm = a["contour"].copy()
        cm[rows:] = False
        fade_c = 0.55 if t > T_GRID[0] else 0.8
        img = blend(img, cm, INK_F, fade_c)
        hs = clamp((sv - 0.55) / 0.45)
        if hs > 0:
            img = blend(img, a["hatch"], CYAN, 0.55 * hs)
            img = blend(img, a["focus_edge"], CYAN, 0.9 * hs)
    # built image, token by token
    if build > 0:
        bt = np.zeros((N, N), np.float32)
        for (r0, c0), v in a["build_t"].items():
            pass
        oi, oj, b = a["blocks"][8]
        keyt = np.vectorize(lambda i, j: a["build_t"][(i, j)])(oi, oj) if "bt_map" not in a else a["bt_map"]
        a["bt_map"] = keyt
        al = np.clip((build - keyt) / 0.12, 0, 1)
        img = img * (1 - up(al, CELL)[..., None]) + a["gen"] * up(al, CELL)[..., None]
    # token linework
    line_a = None
    gw = frac(t, T_GRID)
    if gw > 0 and t < T_REV1[0]:
        # 65 horizontal then 65 vertical lines, each grows over 0.32 of the window... staggered
        line_a = np.zeros((P, P), np.float32)
        n_l = 2 * (N + 1)
        dur = 0.22
        tip = None
        for k in range(n_l):
            s0 = k / n_l * (1 - dur)
            f = clamp((gw - s0) / dur)
            if f <= 0:
                break
            pos = (k % (N + 1)) * CELL
            pos = min(P - 2, max(0, pos - 1))
            ln = int(P * f)
            if k < N + 1:
                line_a[pos:pos + 2, :ln] = 1
                if f < 1:
                    tip = (x0 + ln, y0 + pos)
            else:
                line_a[:ln, pos:pos + 2] = 1
                if f < 1:
                    tip = (x0 + pos, y0 + ln)
        a_tip = tip
    else:
        a_tip = None
    if T_REV1[0] <= t < T_REV2[0]:
        line_a = revision_lines(a["blocks"][1], a["blocks"][2], N, N, CELL, T_REV1, t, 2)
    elif t >= T_REV2[0]:
        line_a = revision_lines(a["blocks"][2], a["blocks"][8], N, N, CELL, T_REV2, t, 2)
    if line_a is not None:
        la = 0.92 * (1 - 0.6 * clamp((t - T_BUILD[1]) / 0.6))
        img = img * (1 - (line_a * la)[..., None]) + INK_F * (line_a * la)[..., None]
    if full:
        base[y0:y0 + P, x0:x0 + P] = img
    else:
        pim = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).convert("RGBA")
        canvas_paste(canvas, pim, x0, y0)
    return a_tip


def canvas_paste(canvas, im, x, y):
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + im.width), min(H, y + im.height)
    if x1 > x0 and y1 > y0:
        canvas.alpha_composite(im.crop((x0 - x, y0 - y, x1 - x, y1 - y)), (x0, y0))


def sheet1_overlay(canvas, d, t, xoff, tip):
    a = assets()
    x0, y0 = PX0 + xoff, PY0
    # plan frame + labels
    d.rectangle([x0 - 8, y0 - 8, x0 + P + 7, y0 + P + 7], outline=INK + (255,), width=2)
    text(d, "PLAN A — DEPTH-OF-FIELD LAYOUT", (x0, y0 - 52), 30, reveal=write_frac(t, 0.3, 30, 40))
    # overall dimension under the plan
    dim_a = clamp((t - 0.6) / 0.5)
    dimension(d, x0, x0 + P, y0 + P + 40, "1024 PX", dim_a, size=24, label_dy=10)
    # survey label
    if t >= T_SURVEY[0] + 1.0:
        ra = write_frac(t, T_SURVEY[0] + 1.0, 18)
        lx, ly = x0 + int(0.84 * P), y0 + int(0.47 * P)
        d.line([(lx, ly), (x0 + P + 60, ly - 120)], fill=CYAN + (255,), width=2)
        d.ellipse([lx - 5, ly - 5, lx + 5, ly + 5], fill=CYAN + (255,))
        text(d, "IN FOCUS", (x0 + P + 66, ly - 150), 28, fill=CYAN, reveal=ra)
        text(d, "(hatched)", (x0 + P + 66, ly - 116), 24, fill=CYAN, reveal=ra)
    # callouts on blocks
    callouts = [  # (r0, c0, s, label, t_on, t_off)
        (30, 50, 1, "1×1 · 16 PX", T_GRID[1] - 0.1, T_BUILD[0]),
        (16, 44, 2, "2×2 · 32 PX", T_REV1[1] + 0.05, T_BUILD[0]),
        (16, 40, 4, "4×4 · 64 PX", T_REV2[1] + 0.05, T_BUILD[0]),
        (24, 16, 8, "8×8 · 128 PX", T_REV2[1] + 0.25, T_BUILD[0]),
    ]
    for r0, c0, s, lab, t_on, t_off in callouts:
        al = clamp((t - t_on) / 0.4) * (1 - clamp((t - t_off) / 0.3))
        if al <= 0:
            continue
        bx0, by0 = x0 + c0 * CELL, y0 + r0 * CELL
        bx1, by1 = bx0 + s * CELL, by0 + s * CELL
        d.rectangle([bx0 - 1, by0 - 1, bx1, by1], outline=(255, 230, 120, int(255 * al)), width=3)
        if s >= 8:
            dimension(d, bx0 + 3, bx1 - 3, by0 + s * CELL // 2, lab, al, size=26, label_dy=-40)
        else:
            # leader to a label on the right of the plan
            ly = by0 + s * CELL // 2
            lx = x0 + P + 30
            if s == 1:
                tx, ty = lx, ly + 40
            else:
                tx, ty = lx, ly - 30
            d.line([(bx1, ly), (lx - 10, ty)], fill=(255, 230, 120, int(255 * al)), width=2)
            d.ellipse([bx1 - 4, ly - 4, bx1 + 4, ly + 4], fill=(255, 230, 120, int(255 * al)))
            text(d, lab, (lx, ty - 18), 28, fill=(255, 230, 120), reveal=al * 1.3)
    # revision clouds
    c1 = clamp((t - T_REV1[1]) / 0.45) * (1 - clamp((t - T_BUILD[0]) / 0.3))
    if c1 > 0 and t < T_BUILD[0] + 0.3:
        cloud(d, x0 + 4, y0 + 8 * CELL - 4, x0 + 40 * CELL - 4, y0 + 48 * CELL + 4, clamp((t - T_REV1[1]) / 0.45), "1")
    c2 = clamp((t - T_REV2[1]) / 0.45)
    if c2 > 0 and t < T_BUILD[0]:
        cloud(d, x0 + 48 * CELL + 4, y0 + 8 * CELL - 4, x0 + P - 4, y0 + 24 * CELL + 4, c2, "2")
    # pens / eraser
    if tip is not None:
        draw_pen(d, *tip)
    for win in (T_REV1, T_REV2):
        f = frac(t, win)
        if 0 < f < 1:
            ey = y0 + f * P
            ex = x0 + P / 2 + (P / 2 - 40) * math.sin(t * 2 * math.pi * 3.2)
            draw_eraser(d, ex, ey)
    sv = frac(t, T_SURVEY)
    if 0 < sv < 0.7:
        rows = P * ease(sv / 0.7)
        draw_pen(d, x0 + P / 2 + (P / 2 - 20) * math.sin(t * 2 * math.pi * 2.3), y0 + rows)


def sheet1_notes(d, t, xoff):
    x = NOTE_X + xoff
    y = NOTE_Y
    notes = [
        (T_SURVEY[0] + 0.1, "1. Survey the depth", None, None),
        (T_GRID[0] + 0.1, "2. Grid: one 1×1 token per patch", "4,096 tokens", T_GRID[1] - 0.3),
        (T_REV1[0] + 0.1, "3. Rev 1: blurry areas get 2×2", "1,606 tokens · 1.9× faster", T_REV1[1] + 0.1),
        (T_REV2[0] + 0.1, "4. Rev 2: blurrier, 4×4 and 8×8", "1,027 tokens · 2.4× faster", T_REV2[1] + 0.1),
        (T_BUILD[0] + 0.05, "5. Build. Sharp where it's in focus.", None, None),
    ]
    for k, (t0, s1, s2, t2) in enumerate(notes):
        if t < t0:
            continue
        text(d, s1, (x, y + k * 98), 34, reveal=write_frac(t, t0, len(s1), 30))
        if s2 and t >= t2:
            text(d, s2, (x + 38, y + k * 98 + 46), 30, fill=(255, 230, 120), reveal=write_frac(t, t2, len(s2), 30))


def sheet1_counts(t):
    if t < T_GRID[0]:
        return "—", "—", "0"
    if t < T_REV1[0]:
        gw = frac(t, T_GRID)
        return fmt(IMG_TOK[0] * clamp(gw / 0.95)), "1.0×", "0"
    if t < T_REV2[0]:
        f = frac(t, T_REV1)
        return fmt(IMG_TOK[0] + (IMG_TOK[1] - IMG_TOK[0]) * f), ("1.9×" if f >= 1 else "1.0×"), "1"
    f = frac(t, T_REV2)
    return fmt(IMG_TOK[1] + (IMG_TOK[2] - IMG_TOK[1]) * f), ("2.4×" if f >= 1 else "1.9×"), "2"


# ------------------------------------------------------------------ sheet 2 (video)
def vid_V(t):
    segs = [(T_SWAP[1], T_VREV1[1], VID_SPEED[0]), (T_VREV1[1], T_VREV2[1], VID_SPEED[1]), (T_VREV2[1], 99, VID_SPEED[2])]
    v = 0.0
    for s0, s1, sp in segs:
        if t > s0:
            v += (min(t, s1) - s0) * sp * 15
    return v


def vid_speed(t):
    if t < T_VREV1[1]:
        return VID_SPEED[0]
    return VID_SPEED[1] if t < T_VREV2[1] else VID_SPEED[2]


def vid_blocks(lv, cap):
    b = np.minimum(lv, cap).astype(int)
    ii, jj = np.mgrid[0:VN, 0:VM]
    return ii // b * b, jj // b * b, b


def sheet2_layer(base, canvas, d, t, xoff):
    a = assets()
    idx = int(vid_V(t)) % 81
    fr = a["vid"][idx]
    lv = a["lv"][idx]
    # left: the video, in a viewport
    xl = VXL + xoff
    xr = VXR + xoff
    imgs = []
    left = fr
    # right: cyanotype print of the frame + linework
    lum = fr.mean(2, keepdims=True) / 255
    paper = np.repeat(np.repeat(PAPER[None, None], VH, 0), VW, 1)
    right = paper * (1 - 0.35 * lum) + np.array(CYAN, np.float32) * 0.35 * lum
    tip = None
    line_a = None
    gw = frac(t, T_VGRID)
    if gw > 0 and t < T_VREV1[0]:
        line_a = np.zeros((VH, VW), np.float32)
        n_l = (VN + 1) + (VM + 1)
        dur = 0.22
        for k in range(n_l):
            s0 = k / n_l * (1 - dur)
            f = clamp((gw - s0) / dur)
            if f <= 0:
                break
            if k < VN + 1:
                pos = min(VH - 1, k * VC)
                ln = int(VW * f)
                line_a[max(0, pos - 1):pos + 1, :ln] = 1
                if f < 1:
                    tip = (xr + ln, VY0 + pos)
            else:
                pos = min(VW - 1, (k - VN - 1) * VC)
                ln = int(VH * f)
                line_a[:ln, max(0, pos - 1):pos + 1] = 1
                if f < 1:
                    tip = (xr + pos, VY0 + ln)
    elif T_VREV1[0] <= t < T_VREV2[0]:
        line_a = revision_lines(vid_blocks(lv, 1), vid_blocks(lv, 2), VN, VM, VC, T_VREV1, t, 1)
    elif t >= T_VREV2[0]:
        line_a = revision_lines(vid_blocks(lv, 2), vid_blocks(lv, 4), VN, VM, VC, T_VREV2, t, 1)
    if line_a is not None:
        right = right * (1 - 0.9 * line_a[..., None]) + INK_F * 0.9 * line_a[..., None]
    for x0, im in ((xl, left), (xr, right)):
        if 0 <= x0 and x0 + VW <= W:
            base[VY0:VY0 + VH, x0:x0 + VW] = im
        else:
            canvas_paste(canvas, Image.fromarray(np.clip(im, 0, 255).astype(np.uint8)).convert("RGBA"), x0, VY0)
    return tip


def sheet2_overlay(canvas, d, t, xoff, tip):
    xl, xr = VXL + xoff, VXR + xoff
    for x0, lab in ((xl, "VIEW B — VIDEO"), (xr, "PLAN B — TOKENS, EVERY FRAME")):
        d.rectangle([x0 - 8, VY0 - 8, x0 + VW + 7, VY0 + VH + 7], outline=INK + (255,), width=2)
        text(d, lab, (x0, VY0 - 56), 30, reveal=write_frac(t, T_SWAP[1], len(lab), 40))
    sp = vid_speed(t)
    pulse = any(0 <= t - t0 < 0.6 for t0 in (T_VREV1[1], T_VREV2[1]))
    col = (255, 230, 120) if pulse else INK
    text(d, f"playback {sp:.1f}×", (xl + VW, VY0 + VH + 20), 30, fill=col, anchor="ra")
    dimension(d, xr, xr + VW, VY0 + VH + 34, "1280 PX · 81 FRAMES", clamp((t - T_SWAP[1] - 0.2) / 0.5), size=24, label_dy=10)
    if tip is not None:
        draw_pen(d, *tip)
    for win in (T_VREV1, T_VREV2):
        f = frac(t, win)
        if 0 < f < 1:
            draw_eraser(d, xr + VW / 2 + (VW / 2 - 40) * math.sin(t * 2 * math.pi * 3.2), VY0 + f * VH)
    # notes
    x, y = 110 + xoff, 760
    notes = [
        (T_VGRID[0] + 0.1, "1. Grid: 75,600 tokens", None, None),
        (T_VREV1[0] + 0.1, "2. Rev 1: ground gets 2×2", "2.2× faster", T_VREV1[1]),
        (T_VREV2[0] + 0.1, "3. Rev 2: sky gets 4×4", "2.5× faster", T_VREV2[1]),
    ]
    for k, (t0, s1, s2, t2) in enumerate(notes):
        if t < t0:
            continue
        text(d, s1, (x, y + k * 72), 34, reveal=write_frac(t, t0, len(s1), 30))
        if s2 and t >= t2:
            text(d, s2, (x + 470, y + k * 72 + 2), 34, fill=(255, 230, 120), reveal=write_frac(t, t2, len(s2), 30))


def sheet2_counts(t):
    if t < T_VGRID[0]:
        return "—", "1.0×", "0"
    if t < T_VREV1[0]:
        return fmt(VID_TOK[0] * clamp(frac(t, T_VGRID) / 0.95)), "1.0×", "0"
    if t < T_VREV2[0]:
        f = frac(t, T_VREV1)
        return fmt(VID_TOK[0] + (VID_TOK[1] - VID_TOK[0]) * f), ("2.2×" if f >= 1 else "1.0×"), "1"
    f = frac(t, T_VREV2)
    return fmt(VID_TOK[1] + (VID_TOK[2] - VID_TOK[1]) * f), ("2.5×" if f >= 1 else "2.2×"), "2"


# ------------------------------------------------------------------ frame
def render(t):
    a = assets()
    base = a["paper"].copy()
    # sheet slide: sheet 1 leaves left, sheet 2 enters from the right
    sw = ease(frac(t, T_SWAP))
    off1 = int(-W * 1.05 * sw)
    off2 = int(W * 1.05 * (1 - sw))
    canvas = None
    tips = []
    # pixel layers first (into base), then RGBA overlay
    tip1 = tip2 = None
    canvas = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(canvas)
    if t < T_SWAP[1]:
        tip1 = sheet1_layer(base, canvas, d, t, off1)
    if t >= T_SWAP[0]:
        tip2 = sheet2_layer(base, canvas, d, t, off2)
    bg = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8)).convert("RGBA")
    bg.alpha_composite(canvas)
    canvas = bg
    d = ImageDraw.Draw(canvas)
    # sheet border (each sheet carries its own)
    for off, on in ((off1, t < T_SWAP[1]), (off2, t >= T_SWAP[0])):
        if on:
            d.rectangle([40 + off, 40, W - 41 + off, H - 41], outline=INK + (255,), width=3)
            d.rectangle([52 + off, 52, W - 53 + off, H - 53], outline=INK + (110,), width=1)
    bdraw = clamp(t / 0.6)
    if t < T_SWAP[1]:
        if bdraw < 1:   # cover the not-yet-drawn part of the border
            pass
        sheet1_overlay(canvas, d, t, off1, tip1)
        sheet1_notes(d, t, off1)
        tok, sp, rev = sheet1_counts(t)
        ta = clamp((t - 0.2) / 0.5)
        title_block(d, t, TB[0] + off1, "1 OF 2", "DEPTH-OF-FIELD LAYOUT", tok, sp, rev, alpha=ta)
        stamp(canvas, t, T_STAMP1, 1500 + off1, 690, "APPROVED", "1,027 TOKENS")
    if t >= T_SWAP[0]:
        sheet2_overlay(canvas, d, t, off2, tip2)
        tok, sp, rev = sheet2_counts(t)
        title_block(d, t, TB[0] + off2, "2 OF 2", "VIDEO LAYOUT", tok, sp, rev)
        stamp(canvas, t, T_STAMP2, 1500 + off2, 690, "APPROVED", "34,698 TOKENS")
        if t >= T_CLOSE:
            s = "Detail where it matters. Faster everywhere else."
            text(d, s, (110, 980), 34, fill=(255, 230, 120), reveal=write_frac(t, T_CLOSE, len(s), 34))
    # sheet 1 opening: border draws on (mask the border with paper that recedes)
    f = np.asarray(canvas.convert("RGB")).astype(np.float32)
    if t < 0.6:
        # reveal everything with a left→right wipe for the first 0.6 s (paper unrolling)
        xr = int(W * ease(t / 0.6))
        f[:, xr:] = a["paper"][:, xr:]
    dx, dy = shake(t)
    if dx or dy:
        f = np.roll(f, (dy, dx), (0, 1))
    return np.clip(f, 0, 255).astype(np.uint8)


# ------------------------------------------------------------------ audio
SR = 48000


def env_ad(n, a=0.005, dcy=0.1):
    tt = np.arange(n) / SR
    e = np.minimum(tt / a, 1) * np.exp(-np.maximum(tt - a, 0) / dcy)
    return e


def bandnoise(dur, lo, hi, seed):
    n = int(dur * SR)
    x = np.random.default_rng(seed).normal(0, 1, n)
    sos = signal.butter(4, [lo, hi], btype="band", fs=SR, output="sos")
    return signal.sosfilt(sos, x)


def build_audio(path):
    L = int((DUR + 0.3) * SR)
    out = np.zeros((L, 2))
    rng = np.random.default_rng(0)

    def add(sig, t0, pan=0.0, vol=1.0):
        i = int(t0 * SR)
        j = min(L, i + len(sig))
        if j <= i:
            return
        s = sig[:j - i] * vol
        out[i:j, 0] += s * math.sqrt(0.5 * (1 - pan))
        out[i:j, 1] += s * math.sqrt(0.5 * (1 + pan))

    # lo-fi pad: Am7 Fmaj7 Cmaj7 G6, 2.5 s each, sine + detuned triangle, lowpassed, with tape wobble
    chords = [[57, 60, 64, 67], [53, 57, 60, 64], [48, 55, 59, 64], [55, 59, 62, 64]]
    pad = np.zeros(L)
    tt = np.arange(L) / SR
    wob = 1 + 0.0025 * np.sin(2 * np.pi * 0.6 * tt)
    for k in range(int(DUR / 2.5) + 1):
        t0, t1 = k * 2.5, k * 2.5 + 2.7
        i0, i1 = int(t0 * SR), min(L, int(t1 * SR))
        if i0 >= L:
            break
        seg_t = tt[i0:i1] - t0
        e = np.minimum(seg_t / 0.5, 1) * np.minimum((t1 - t0 - seg_t) / 0.5, 1)
        for m in chords[k % 4]:
            f = 440 * 2 ** ((m - 69) / 12)
            ph = 2 * np.pi * f * np.cumsum(wob[i0:i1]) / SR
            pad[i0:i1] += e * (np.sin(ph) + 0.3 * np.sin(2 * ph + 0.3)) * 0.05
    sos = signal.butter(2, 1800, fs=SR, output="sos")
    pad = signal.sosfilt(sos, pad)
    out[:, 0] += pad
    out[:, 1] += pad
    # vinyl crackle
    cr = np.zeros(L)
    idx = rng.integers(0, L, 260)
    cr[idx] = rng.uniform(-1, 1, 260)
    cr = signal.sosfilt(signal.butter(2, [1500, 7000], btype="band", fs=SR, output="sos"), cr) * 0.25
    out[:, 0] += cr
    out[:, 1] += cr

    def scratch(t0, t1, seed, vol=0.25, pan=0.0):
        """pen on paper: band noise with stroke envelopes."""
        n = int((t1 - t0) * SR)
        x = bandnoise(t1 - t0, 2500, 7500, seed)
        r = np.random.default_rng(seed)
        e = np.zeros(n)
        p = 0
        while p < n:
            ln = int(r.uniform(0.06, 0.2) * SR)
            gap = int(r.uniform(0.0, 0.05) * SR)
            seg = np.sin(np.linspace(0, np.pi, max(2, min(ln, n - p)))) ** 0.6
            e[p:p + len(seg)] = seg * r.uniform(0.6, 1.0)
            p += ln + gap
        add(x * e * vol / (np.abs(x).max() + 1e-9), t0, pan)

    def click(t0, vol=0.35, pan=0.0):
        n = int(0.03 * SR)
        x = bandnoise(0.03, 1500, 5000, int(t0 * 1000)) * env_ad(n, 0.0005, 0.004)
        add(x / (np.abs(x).max() + 1e-9) * vol, t0, pan)

    def rustle(t0, dur, seed, vol=0.5):
        n = int(dur * SR)
        x = bandnoise(dur, 300, 6000, seed)
        r = np.random.default_rng(seed)
        e = np.repeat(r.uniform(0, 1, n // 800 + 1) ** 3, 800)[:n]
        e = cv2.GaussianBlur(e.reshape(1, -1).astype(np.float32), (0, 0), 200).ravel()
        e *= np.sin(np.linspace(0, np.pi, n))
        add(x * e * vol / (np.abs(x * e).max() + 1e-9), t0)

    def erase(t0, t1, seed, vol=0.3):
        n = int((t1 - t0) * SR)
        x = bandnoise(t1 - t0, 250, 1800, seed)
        tt2 = np.arange(n) / SR
        e = 0.5 + 0.5 * np.abs(np.sin(2 * np.pi * 3.2 * tt2))
        add(x * e * vol / (np.abs(x).max() + 1e-9), t0)

    def thump(t0, vol=0.9):
        n = int(0.4 * SR)
        tt2 = np.arange(n) / SR
        f = 75 * np.exp(-tt2 * 3) + 40
        x = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-tt2 / 0.09)
        x += bandnoise(0.4, 200, 3000, int(t0)) * np.exp(-tt2 / 0.02) * 0.08
        add(x * vol, t0)

    def chime(t0, f0=880, vol=0.22):
        n = int(1.4 * SR)
        tt2 = np.arange(n) / SR
        x = sum(a_ * np.sin(2 * np.pi * f0 * r_ * tt2) * np.exp(-tt2 / d_)
                for a_, r_, d_ in ((1, 1, 0.6), (0.5, 2.76, 0.3), (0.25, 5.4, 0.15)))
        add(x * vol, t0, pan=0.2)

    rustle(0.0, 0.7, 1)
    scratch(0.15, 0.85, 2, 0.18)
    scratch(T_SURVEY[0], T_SURVEY[0] + 1.0, 3, 0.22, -0.2)
    scratch(T_SURVEY[0] + 1.0, T_SURVEY[1], 4, 0.15)
    scratch(T_GRID[0], T_GRID[1], 5, 0.28, 0.1)
    for k in range(0, 130, 8):
        click(T_GRID[0] + k / 130 * (T_GRID[1] - T_GRID[0]) * 0.78, 0.25, -0.3 + 0.6 * (k % 16 == 0))
    erase(*T_REV1, 6)
    erase(*T_REV2, 7)
    scratch(T_REV1[1], T_REV1[1] + 0.45, 8, 0.15)
    scratch(T_REV2[1], T_REV2[1] + 0.45, 9, 0.15)
    for t0 in (T_REV1[1] + 0.1, T_REV2[1] + 0.1, T_VREV1[1], T_VREV2[1]):
        chime(t0, 880 if t0 < 10 else 988)
    thump(T_STAMP1 + 0.1)
    thump(T_STAMP2 + 0.1)
    for k in range(10):
        click(T_BUILD[0] + k * 0.16, 0.18, 0.4 * math.sin(k))
    chime(T_BUILD[1] - 0.1, 660, 0.25)
    rustle(T_SWAP[0] - 0.05, 0.8, 10, 0.6)
    scratch(T_VGRID[0], T_VGRID[1], 11, 0.28)
    erase(*T_VREV1, 12)
    erase(*T_VREV2, 13)
    scratch(T_CLOSE, T_CLOSE + 1.2, 14, 0.15)
    out = out[:int(DUR * SR)]
    out = out / (np.abs(out).max() + 1e-9) * 0.89
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
    wav = os.path.join(HERE, f"media/final/LoT_blueprint_{TAG}.wav")
    build_audio(wav)
    out = os.path.join(HERE, f"media/final/LoT_blueprint_{TAG}_{res}.mp4")
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
