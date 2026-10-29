"""Shared drawing for the edit-results clips (copied from ../edit_clip/render_edit_v3.py: style, Canvas, tokens).

MODE (argv[1]): h = 1080p60 (2× SSAA), l = 540p15, still = 1080p stills.
"""
import os
import subprocess
import sys
from contextlib import contextmanager
from functools import lru_cache
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
MODE = sys.argv[1] if len(sys.argv) > 1 else "l"
FPS, OUTS, SS = (60, 1.0, 2) if MODE in ("h", "still") else (15, 0.5, 1)
K = OUTS * SS                                  # 1080p px → canvas px
W, H = 1920, 1080
N = 64
# ------------------------------------------------------------------ style (STYLE_NOTES.md)
WHITE = (255, 255, 255)
DIM = (138, 138, 138)
GHOST = (58, 58, 58)
AMBER = (255, 202, 40)
RED = (255, 82, 82)
PURPLE = (224, 64, 251)
LORA = (255, 112, 67)
BOX = (30, 42, 51)
BOXS = (144, 164, 174)
FD = os.path.expanduser("~/.local/share/fonts")
FONTS = {"rm": f"{FD}/cmunrm.ttf", "bx": f"{FD}/cmunbx.ttf", "ti": f"{FD}/cmunti.ttf",
         "math": "/usr/share/texmf/fonts/opentype/public/lm-math/latinmodern-math.otf"}
MATHCH = set("×⊙→≈∼−∈≪≤‖↓÷")


@lru_cache(None)
def font(style, px):
    return ImageFont.truetype(FONTS[style], max(1, int(round(px))))

# ------------------------------------------------------------------ canvas
class Canvas:
    def __init__(self):
        self.im = Image.new("RGB", (int(W * K), int(H * K)), (0, 0, 0))
        self.d = ImageDraw.Draw(self.im)

    def _reset(self, im):
        self.im = im
        self.d = ImageDraw.Draw(self.im)

    @contextmanager
    def fade(self, a):
        """Everything drawn inside the block is blended in with opacity a."""
        if a <= 0:
            snap = self.im.copy()
            yield
            self._reset(snap)
            return
        if a >= 1:
            yield
            return
        snap = self.im.copy()
        yield
        self._reset(Image.blend(snap, self.im, a))

    def rect(self, x0, y0, x1, y1, fill=None, outline=None, width=1.0, r=0.0):
        box = [x0 * K, y0 * K, x1 * K, y1 * K]
        w = max(1, int(round(width * K))) if outline else 0
        if r * K >= 1.5:
            self.d.rounded_rectangle(box, radius=r * K, fill=fill, outline=outline, width=w)
        else:
            self.d.rectangle(box, fill=fill, outline=outline, width=w)

    def line(self, pts, fill, width=2.0):
        self.d.line([(x * K, y * K) for x, y in pts], fill=fill, width=max(1, int(round(width * K))))

    def poly(self, pts, fill=None, outline=None, width=2.0):
        self.d.polygon([(x * K, y * K) for x, y in pts], fill=fill, outline=outline,
                       width=max(1, int(round(width * K))))

    def arrow(self, x0, y0, x1, y1, fill=WHITE, width=3.0, tip=16):
        v = np.array([x1 - x0, y1 - y0], float)
        L = np.linalg.norm(v)
        u = v / L
        n = np.array([-u[1], u[0]])
        b = np.array([x1, y1]) - u * tip
        self.line([(x0, y0), tuple(b)], fill, width)
        self.poly([(x1, y1), tuple(b + n * tip * 0.45), tuple(b - n * tip * 0.45)], fill=fill)

    def paste(self, im, x0, y0, size, mask=None):
        px = max(1, int(round(size * K)))
        im = im.resize((px, px), Image.BILINEAR if px < im.width else Image.LANCZOS)
        if mask is not None:
            mask = mask.resize((px, px), Image.BILINEAR)
        self.im.paste(im, (int(round(x0 * K)), int(round(y0 * K))), mask)

    def text(self, x, y, s, size, fill=WHITE, style="rm", anchor="m", max_w=None):
        """Mixed-font line: CMU Serif runs + Latin Modern Math for MATHCH. anchor l / m / r; y = visual middle."""
        def runs(px):
            out = []
            for ch in s:
                f = "math" if ch in MATHCH else style
                if out and out[-1][0] == f:
                    out[-1][1] += ch
                else:
                    out.append([f, ch])
            return [(font(f, px), r) for f, r in out]
        px = size * K
        rs = runs(px)
        width = sum(f.getlength(r) for f, r in rs)
        if max_w is not None and width > max_w * K:
            px *= max_w * K / width
            rs = runs(px)
            width = sum(f.getlength(r) for f, r in rs)
        cap = font(style, px).getbbox("H", anchor="ls")
        yb = y * K - (cap[1] + cap[3]) / 2
        xc = x * K - {"l": 0, "m": width / 2, "r": width}[anchor]
        for f, r in rs:
            self.d.text((xc, yb), r, font=f, fill=fill, anchor="ls")
            xc += f.getlength(r)
        return width / K


def draw_tokens(c, panel, lv, cols, outline_idx=None, outline_col=WHITE, dy=0.0):
    x0, y0, S = panel
    y0 = y0 + dy
    cell = S / N
    cols = np.clip(cols, 0, 255).astype(np.uint8)
    for k, (r0, r1, c0, c1) in enumerate(lv):
        e = r1 - r0
        side = e * cell - 0.2 * cell
        px, py = x0 + c0 * cell + 0.1 * cell, y0 + r0 * cell + 0.1 * cell
        c.rect(px, py, px + side, py + side, fill=tuple(cols[k]), r=0.1 * side if e > 1 else 0)
    if outline_idx is not None:
        for k in np.atleast_1d(outline_idx):
            r0, r1, c0, c1 = lv[k]
            e = r1 - r0
            px, py = x0 + c0 * cell, y0 + r0 * cell
            c.rect(px - 1, py - 1, px + e * cell + 1, py + e * cell + 1, outline=outline_col, width=2.5)


def draw_region_outline(c, panel, region, col=WHITE, width=3.0):
    x0, y0, S = panel
    cell = S / N
    r = np.pad(region, 1)
    segs = []
    for i, j in zip(*np.nonzero(r[1:-1, 1:-1])):
        if not r[i, j + 1]:
            segs.append(((j, i), (j + 1, i)))           # top edge
        if not r[i + 2, j + 1]:
            segs.append(((j, i + 1), (j + 1, i + 1)))   # bottom
        if not r[i + 1, j]:
            segs.append(((j, i), (j, i + 1)))           # left
        if not r[i + 1, j + 2]:
            segs.append(((j + 1, i), (j + 1, i + 1)))   # right
    for (a, b) in segs:
        c.line([(x0 + a[0] * cell, y0 + a[1] * cell), (x0 + b[0] * cell, y0 + b[1] * cell)], col, width)


def ghost(c, panel):
    x0, y0, S = panel
    c.rect(x0, y0, x0 + S, y0 + S, outline=GHOST, width=1.5)


def frame_box(c, panel, col=(70, 70, 70)):
    x0, y0, S = panel
    c.rect(x0 - 1, y0 - 1, x0 + S + 1, y0 + S + 1, outline=col, width=1.5)

def tagbox(c, x, y, s, size=28, fill=WHITE, anchor="l"):
    """Dark rounded text box (as the website clip); (x, y) = left / centre of the box's left or middle edge."""
    w = font("rm", size * K).getlength(s) / K + 24
    x0 = x if anchor == "l" else x - w / 2
    c.rect(x0, y - size * 0.75, x0 + w, y + size * 0.75, fill=(10, 10, 10), r=8)
    c.text(x0 + 12, y, s, size, fill, anchor="l")


def seg(t, a, b):
    return float(np.clip((t - a) / (b - a), 0, 1))


def ease(x):
    return x * x * (3 - 2 * x)


def lerp(a, b, u):
    return a + (b - a) * u


def fmt(n):
    return f"{n:,}"


def pct(n, full=4096):
    return int(round(100 * n / full))


@lru_cache(24)
def load(path):
    return Image.open(path).convert("RGB")


def caption(c, t, caps, y=1012):
    c.text(W / 2, y, [s for s0, s in caps if t >= s0][-1], 40)


def finish(c):
    im = c.im
    if SS > 1:
        im = im.resize((int(W * OUTS), int(H * OUTS)), Image.LANCZOS)
    return im


def run(render, dur, name):
    """still t…  → scratch/<name>_<t>.png; h|l → media/<name>_<res>.mp4."""
    if MODE == "still":
        for s in sys.argv[2:]:
            render(float(s)).save(f"{HERE}/scratch/{name}_{s}.png")
        return
    nf = int(round(dur * FPS))
    res = "1080p60" if MODE == "h" else "540p15"
    out = f"{HERE}/media/{name}_{res}.mp4"
    w, h = int(W * OUTS), int(H * OUTS)
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                           "-s", f"{w}x{h}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow",
                           "-crf", "16", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out],
                          stdin=subprocess.PIPE)
    global _R
    _R = render
    with Pool(48) as p:
        for b in p.imap(_frame, [(i, FPS) for i in range(nf)], chunksize=4):
            ff.stdin.write(b)
    ff.stdin.close()
    ff.wait()
    print("wrote", out)


_R = None


def _frame(a):
    i, fps = a
    return np.asarray(_R(i / fps)).tobytes()
