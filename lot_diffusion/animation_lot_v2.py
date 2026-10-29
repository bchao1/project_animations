"""Level-of-Token Diffusion — ~5-minute teaser, v2 (house style: see STYLE.md; template foveated_diffusion/teaser_5min).

Source: anonymous ICLR 2027 submission 7014, "Level-of-Token Diffusion" (7014_Level_of_Token_Diffusion.pdf) and its
supplementary website (mirrored in web/). Numbers: level-of-token-diffusion/summaries/lot_diffusion.md.
Paper figure crops and website media live in data/ (built by scratch/prep_data.py); toy assets in data/generated/.

Every beat is a method of `Beats` so the 10-s thread clips (clips/animation_lot_clips_v2.py) render the same code.
Render: ./render.sh l|h.   Scenes, in order: S00_Title … S09_Outlook.
"""
from manim import *
import numpy as np
from PIL import Image
import glob
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "img")
CLIPS = os.path.join(HERE, "data", "clips")
GEN = os.path.join(HERE, "data", "generated")

config.background_color = BLACK

# Title / end-card credits. The paper is an anonymous submission: fill these in before the public render.
TITLE = "Level-of-Token Diffusion"
SUBTITLE = "Adaptive, Efficient Image and Video Generation"
AUTHORS = None      # e.g. "A. Author*, B. Author*, …  ·  Stanford University"
LINKS = None        # e.g. "arXiv 26xx.xxxxx  ·  project page"

# ------------------------------------------------------------------ style (foveated teaser values)
FONT = "CMU Serif"
MATHFONT = "Latin Modern Math"          # CMU Serif lacks × ⊙ − → ≈ …
SYMFONT = "DejaVu Sans"
HR = "#448AFF"                           # 1×1 tokens / ours
LR = "#00C853"                           # 2×2 tokens
LR3 = "#1B7F3B"                          # 4×4 tokens
LR4 = "#0E4A22"                          # 8×8 tokens
DIM = "#8A8A8A"
GHOST = "#3A3A3A"
RED = "#FF5252"
AMBER = "#FFCA28"
PURPLE = "#E040FB"
LORA = "#FF7043"
BOX = "#1E2A33"
TEAL = "#4DD0E1"

HEADER_Y = 3.62
CAPTION_Y = -3.5
MATHCH = set("×⊙→≈∼−∈𝒩ℒ𝔼′≪≤≥‖·↓ℝ⊤σ")
SYMCH = set("✗")


def mathify(s):
    out, inside = [], False
    for c in s:
        if c == "<":
            inside = True
        elif c == ">":
            inside = False
        if not inside and c in MATHCH:
            out.append(f'<span font_family="{MATHFONT}">{c}</span>')
        elif not inside and c in SYMCH:
            out.append(f'<span font_family="{SYMFONT}">{c}</span>')
        else:
            out.append(c)
    return "".join(out)


def sub(x):
    return f'<span rise="-3000"><small><small>{x}</small></small></span>'


def sup(x):
    return f'<span rise="4500"><small><small>{x}</small></small></span>'


def col(x, c):
    return f'<span foreground="{c}">{x}</span>'


def I(v):
    return f"<i>{v}</i>"


def tx(s, size=30, color=WHITE, bold=False, italic=False):
    if bold:
        s = f"<b>{s}</b>"
    if italic:
        s = f"<i>{s}</i>"
    return MarkupText(mathify(s), font=FONT, font_size=size, color=color)


def fit(m, w=13.0):
    if m.width > w:
        m.scale_to_fit_width(w)
    return m


def toy_note(s):
    return tx(s, 16, DIM)


# symbols
XT = f"{I('x')}{sub(I('t'))}"
XTI = f"{I('x')}{sup('(' + I('i') + ')')}{sub(I('t'))}"
AET = f"{I('A')}{sub(I('e'))}{sup('⊤')}"
PE = f"{I('P')}{sub(I('e'))}"
UA = f"û{sub('A')}"
EH = f"{I('e')}{sub('h')}"
EW = f"{I('e')}{sub('w')}"


# ------------------------------------------------------------------ tokens
def token(side, fill=HR, stroke=None, sw=0.0, op=1.0):
    return RoundedRectangle(
        width=side, height=side, corner_radius=0.1 * side,
        fill_color=fill, fill_opacity=op,
        stroke_color=stroke if stroke else fill, stroke_width=sw)


def ghost(side, dashed=False):
    r = RoundedRectangle(width=side, height=side, corner_radius=0.1 * side,
                         stroke_color=GHOST, stroke_width=1.5, fill_opacity=0)
    return DashedVMobject(r, num_dashes=16) if dashed else r


class Grid:
    """n×n token grid of side `size`. Cell margin 10% → a token covering eh×ew cells is 0.8·eh·c by 0.8·ew·c."""

    def __init__(self, n, size, center, m=None):
        self.n, self.size, self.c = n, size, size / n
        self.m = m or n                                     # columns (m ≠ n for wide grids)
        self.center = np.array([center[0], center[1], 0.0])

    def cell(self, i, j):
        return self.center + np.array(
            [(j + 0.5) * self.c - self.m * self.c / 2, self.n * self.c / 2 - (i + 0.5) * self.c, 0])

    def rect_center(self, r):
        y0, y1, x0, x1 = r
        return self.center + np.array(
            [(x0 + x1) / 2 * self.c - self.m * self.c / 2, self.n * self.c / 2 - (y0 + y1) / 2 * self.c, 0])

    @property
    def hs(self):
        return 0.8 * self.c


def level_color(area):
    """colour by token area: 1 HR blue, 4 green, 16 dark green, 64 darker; in-between areas interpolate"""
    stops = [(1, HR), (4, LR), (16, LR3), (64, LR4)]
    a = max(1, area)
    for (a0, c0), (a1, c1) in zip(stops, stops[1:]):
        if a <= a1:
            t = (np.log2(a) - np.log2(a0)) / (np.log2(a1) - np.log2(a0))
            return interpolate_color(ManimColor(c0), ManimColor(c1), t)
    return ManimColor(LR4)


def rect_token(g, r, fill=None, op=1.0, stroke=None, sw=0.0):
    y0, y1, x0, x1 = r
    w, h = 0.8 * (x1 - x0) * g.c + 0.2 * g.c * (x1 - x0 - 1), 0.8 * (y1 - y0) * g.c + 0.2 * g.c * (y1 - y0 - 1)
    # the 10% margin sits on the token's outer border only: a 2×2 token spans 1.8 c (foveated LR side is 1.6 c:
    # keep that look by shrinking multi-cell tokens a little more)
    w -= 0.2 * g.c * (x1 - x0 > 1)
    h -= 0.2 * g.c * (y1 - y0 > 1)
    area = (x1 - x0) * (y1 - y0)
    fc = fill if fill is not None else level_color(area)
    m = RoundedRectangle(width=w, height=h, corner_radius=0.1 * min(w, h), fill_color=fc, fill_opacity=op,
                         stroke_color=stroke if stroke else fc, stroke_width=sw)
    return m.move_to(g.rect_center(r))


def quadtree(D, t4=0.2, t2=0.8, top=4):
    """paper Eq. 23 on a small grid: split a block of extent b when its max detail ≥ t_b"""
    n, m = D.shape
    th = {4: t4, 2: t2}
    rects = []

    def rec(y, x, b):
        blk = D[y:y + b, x:x + b]
        if b > 1 and blk.max() >= th.get(b, 9):
            h = b // 2
            for dy in (0, h):
                for dx in (0, h):
                    rec(y + dy, x + dx, h)
        else:
            rects.append((y, y + b, x, x + b))

    for y in range(0, n, top):
        for x in range(0, m, top):
            rec(y, x, top)
    return rects


def uniform(n, m=None):
    m = m or n
    return [(i, i + 1, j, j + 1) for i in range(n) for j in range(m)]


def contains(a, b):
    return a[0] <= b[0] and a[1] >= b[1] and a[2] <= b[2] and a[3] >= b[3]


class LoTField:
    """Token field over a layout (list of rects on g). retarget() merges children into parents or splits them."""

    def __init__(self, g, rects, colors=None, op=1.0):
        self.g, self.colors, self.op = g, colors, op
        self.state = {r: self._make(r) for r in rects}

    def _fill(self, r):
        if self.colors is None:
            return None
        y0, y1, x0, x1 = r
        a = self.colors[y0:y1, x0:x1].reshape(-1, 3).mean(0)
        return rgb_to_color(a)

    def _make(self, r):
        return rect_token(self.g, r, self._fill(r), self.op)

    def mobs(self):
        return list(self.state.values())

    def group(self):
        return VGroup(*self.mobs())

    @property
    def rects(self):
        return list(self.state.keys())

    def count(self, area=None):
        if area is None:
            return len(self.state)
        return sum(1 for r in self.state if (r[1] - r[0]) * (r[3] - r[2]) == area)

    def steps(self, rects):
        """layout sequences for a general change: split to the common refinement, then merge into `rects`"""
        ref = []
        for a in self.rects:
            for b in rects:
                y0, y1, x0, x1 = max(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), min(a[3], b[3])
                if y0 < y1 and x0 < x1:
                    ref.append((y0, y1, x0, x1))
        return [ref, list(rects)]

    def retarget(self, rects, order_from=None):
        """animations turning the current layout into `rects` (each new rect contains or is contained by old ones)"""
        old = dict(self.state)
        new = {}
        anims = []
        done_old = set()
        for r in rects:
            if r in old:
                new[r] = old[r]
                done_old.add(r)
        for r in rects:
            if r in new:
                continue
            kids = [o for o in old if o not in done_old and contains(r, o)]
            if kids:                                       # merge
                m = self._make(r)
                anims.append((r, ReplacementTransform(VGroup(*[old[k] for k in kids]), m)))
                new[r] = m
                done_old.update(kids)
        for o in old:
            if o in done_old:
                continue
            parts = [r for r in rects if r not in new and contains(o, r)]
            if parts:                                      # split
                ms = [self._make(p) for p in parts]
                anims.append((o, ReplacementTransform(old[o], VGroup(*ms))))
                for p, m in zip(parts, ms):
                    new[p] = m
                done_old.add(o)
        self.state = {r: new[r] for r in rects if r in new}
        if order_from is not None:
            cy, cx = order_from
            anims.sort(key=lambda ra: ((ra[0][0] + ra[0][1]) / 2 - cy) ** 2 + ((ra[0][2] + ra[0][3]) / 2 - cx) ** 2)
        return [a for _, a in anims]


def patch_array(path, n, m=None):
    m = m or n
    return np.asarray(Image.open(path).convert("RGB").resize((m, n), Image.BOX)) / 255.0


def noise_array(n, seed):
    rng = np.random.default_rng(seed)
    v = np.clip(0.5 + 0.2 * rng.standard_normal((n, n, 3)), 0.08, 0.92)
    g = np.clip(0.5 + 0.2 * rng.standard_normal((n, n, 1)), 0.1, 0.9)
    return 0.75 * g + 0.25 * v


def grid_lines(g, color=WHITE, sw=1.0, op=0.6):
    L = VGroup()
    x0, x1 = g.center[0] - g.m * g.c / 2, g.center[0] + g.m * g.c / 2
    y0, y1 = g.center[1] - g.n * g.c / 2, g.center[1] + g.n * g.c / 2
    for k in range(g.m + 1):
        L.add(Line([x0 + k * g.c, y0, 0], [x0 + k * g.c, y1, 0]))
    for k in range(g.n + 1):
        L.add(Line([x0, y0 + k * g.c, 0], [x1, y0 + k * g.c, 0]))
    return L.set_stroke(color, sw, op)


def cell_box(g, r, color, sw=3):
    y0, y1, x0, x1 = r
    return Rectangle(width=(x1 - x0) * g.c, height=(y1 - y0) * g.c).set_stroke(color, sw).set_fill(opacity=0).move_to(
        g.rect_center(r))


# ------------------------------------------------------------------ assets
def gen(name):
    return os.path.join(GEN, name)


RESULT = os.path.join(DATA, "paint0_result.png")          # "a little girl riding an enormous friendly corgi"
CANVAS = os.path.join(DATA, "paint0_canvas.png")          # its painted detail map (site, agent-assisted demo)
D16 = np.load(os.path.join(DATA, "paint0_detail16.npy"))  # map scores area-averaged to 16 × 16
TOY = quadtree(D16)                                        # 104 × 1×1 + 26 × 2×2 + 3 × 4×4 = 133
TOY_B = quadtree(D16[:, ::-1])                             # mirrored plan (title hero only)
TOY_B = [(y0, y1, 16 - x1, 16 - x0) for (y0, y1, x0, x1) in TOY_B]
FACE = (3, 4, 7, 8)                                       # a 1×1 token on the girl's face
SKY = (1, 2, 13, 14)                                      # a token of empty sky


def rect_merge_toy():
    """TOY with directional merges: sky 4×4 pairs → 8×4, top 2×2 pairs → 2×4, grass 2×2 pairs → 2×4"""
    s = set(TOY)
    pairs = [((0, 4, 0, 4), (4, 8, 0, 4), (0, 8, 0, 4)), ((0, 2, 4, 6), (0, 2, 6, 8), (0, 2, 4, 8)),
             ((0, 2, 8, 10), (0, 2, 10, 12), (0, 2, 8, 12))]
    for c in range(0, 16, 4):
        pairs.append(((14, 16, c, c + 2), (14, 16, c + 2, c + 4), (14, 16, c, c + 4)))
    out = list(TOY)
    for a, b, m in pairs:
        if a in s and b in s:
            out = [r for r in out if r not in (a, b)] + [m]
    return out


def build_assets():
    os.makedirs(GEN, exist_ok=True)
    p = gen("attn_v2.png")
    if not os.path.exists(p):
        a = patch_array(RESULT, 16).reshape(-1, 3)
        d2 = ((a[:, None, :] - a[None, :, :]) ** 2).sum(-1)
        A = np.exp(-d2 / 0.02)
        A = A / A.max(1, keepdims=True)
        base = np.array([0x44, 0x8A, 0xFF]) / 255.0
        rgb = (0.06 + 0.94 * A[..., None]) * base
        tmp = p[:-4] + f".{os.getpid()}.png"
        Image.fromarray((rgb * 255).astype(np.uint8)).resize((512, 512), Image.NEAREST).save(tmp)
        os.replace(tmp, p)


build_assets()


def img(path, h=None, w=None):
    p = path if os.path.isabs(path) else os.path.join(DATA, path)
    m = ImageMobject(p)
    if h is not None:
        m.set(height=h)
    if w is not None:
        m.set(width=w)
    return m


def framed(m, color=WHITE, sw=1.5):
    return SurroundingRectangle(m, buff=0, color=color, stroke_width=sw)


def runs(mask, min_len=40):
    out, s = [], None
    for i, v in enumerate(mask):
        if v and s is None:
            s = i
        if not v and s is not None:
            if i - s >= min_len:
                out.append((s, i))
            s = None
    if s is not None and len(mask) - s >= min_len:
        out.append((s, len(mask)))
    return out


def panel(path, r, c, name, thr=245, frac=0.3):
    """crop panel (row r, col c) of a figure raster split by near-white gutters → data/generated/name"""
    out = gen(name)
    if not os.path.exists(out):
        a = np.asarray(Image.open(path).convert("L")).astype(int)
        rows, cols = runs((a < thr).mean(1) > frac), runs((a < thr).mean(0) > frac)
        (y0, y1), (x0, x1) = rows[r], cols[c]
        Image.open(path).convert("RGB").crop((x0 + 2, y0 + 2, x1 - 2, y1 - 2)).save(out)
    return out


class Clip:
    """website video frames (data/clips/<key>/*.jpg) as an ImageMobject that advances with scene time (HBH style)"""

    def __init__(self, key, h=None, w=None, speed=0.6, loop=True):
        self.paths = sorted(glob.glob(os.path.join(CLIPS, key, "*.jpg")))
        self.fps = json.load(open(os.path.join(CLIPS, "index.json")))[key]["fps"]
        self.speed, self.loop, self.t, self.i = speed, loop, 0.0, -1
        self.cache = {}
        self.mob = ImageMobject(self._frame(0))
        if h is not None:
            self.mob.set(height=h)
        if w is not None:
            self.mob.set(width=w)
        self.mob.add_updater(self._tick)

    def _frame(self, i):
        if i not in self.cache:
            self.cache[i] = np.asarray(Image.open(self.paths[i]).convert("RGBA"))
        return self.cache[i]

    def _tick(self, m, dt):
        self.t += dt * self.speed
        i = int(self.t * self.fps)
        i = i % len(self.paths) if self.loop else min(i, len(self.paths) - 1)
        if i != self.i:
            m.pixel_array = self._frame(i)
            self.i = i


# ------------------------------------------------------------------ scene base
class Base(Scene):
    clip = False                                          # thread-clip mode (clips/): beats end on a hold
    CPS = 16.0                                            # caption reading speed (characters / s)

    def setup(self):
        self.cap = None
        self.hdr = None

    def header(self, s):
        if self.hdr is not None:
            self.remove(self.hdr)
        self.hdr = tx(s, 24, DIM).to_corner(UL, buff=0.35).set_y(HEADER_Y)
        self.add(self.hdr)

    # Captions stay up long enough to read: ~16 characters per second + 0.8 s (teaser pacing).
    def _hold_caption(self):
        if self.cap is not None:
            left = self.cap_need - (self.renderer.time - self.cap_t0)
            if left > 0.05:
                self.wait(left)

    @staticmethod
    def _split(s):
        """Split at the space nearest the middle that lies outside every markup element."""
        opened, spaces, n, k = 0, [], 0, 0
        while k < len(s):
            if s[k] == "<":
                end = s.index(">", k)
                opened += -1 if s[k + 1] == "/" else 1
                k = end + 1
                continue
            if opened == 0 and s[k] == " ":
                spaces.append((k, n))
            n += 1
            k += 1
        if not spaces:
            return [s]
        k = min(spaces, key=lambda p: abs(p[1] - n / 2))[0]
        return [s[:k], s[k + 1:]]

    def caption(self, s, wait=None, size=28):
        self._hold_caption()
        s = re.sub(r"(\d) × (\d)", "\\1\u00a0×\u00a0\\2", s)   # never wrap inside “16 × 16”
        if self.cap is not None:
            self.remove(self.cap)
        c = tx(s, size)
        if c.width > 13.2:
            parts = self._split(s)
            c = VGroup(*[tx(p, size - 2) for p in parts]).arrange(DOWN, buff=0.12)
            if c.width > 13.2:
                c.scale_to_fit_width(13.2)
        c.move_to([0, CAPTION_Y, 0])
        if c.get_bottom()[1] < -3.85:
            c.shift(UP * (-3.85 - c.get_bottom()[1]))
        self.cap = c
        plain = re.sub(r"<[^>]+>", "", s)
        self.cap_t0 = self.renderer.time
        self.cap_need = len(plain) / self.CPS + 0.8
        self.add(c)
        if wait:
            self.wait(wait)
        return c

    def hold(self, extra=0.0):
        self._hold_caption()
        if extra:
            self.wait(extra)

    def uncaption(self):
        self._hold_caption()
        if self.cap is not None:
            self.remove(self.cap)
            self.cap = None

    def title_card(self, num, title, hold=1.6):
        a = tx(num, 30, DIM)
        b = tx(title, 50, bold=True)
        if b.width > 12.5:
            b.scale_to_fit_width(12.5)
        g = VGroup(a, b).arrange(DOWN, buff=0.3)
        self.add(g)
        self.wait(hold)
        self.remove(g)

    def clear_all(self, keep=(), run_time=0.7):
        self.uncaption()
        keep = set(keep) | {self.hdr}
        mobs = [m for m in self.mobjects if m not in keep]
        for m in mobs:
            m.clear_updaters()
        if mobs:
            self.play(*[FadeOut(m) for m in mobs], run_time=run_time)

    def end_beat(self, keep=(), run_time=0.7):
        """teaser: fade the beat out; thread clip: hold the finished visual"""
        if self.clip:
            self.hold(1.6)
        else:
            self.clear_all(keep=keep, run_time=run_time)


def arrow(a, b, buff=0.1, color=WHITE, sw=3):
    return Arrow(a, b, buff=buff, color=color, stroke_width=sw,
                 max_tip_length_to_length_ratio=0.18, tip_length=0.16)


def coder(label, center, w=0.62, h=0.95, widen_right=True):
    """VAE encoder/decoder trapezoid."""
    s = h / 2
    t = h / 4
    if widen_right:
        pts = [[-w / 2, t, 0], [w / 2, s, 0], [w / 2, -s, 0], [-w / 2, -t, 0]]
    else:
        pts = [[-w / 2, s, 0], [w / 2, t, 0], [w / 2, -t, 0], [-w / 2, -s, 0]]
    p = Polygon(*pts, color="#B0BEC5", stroke_width=2, fill_color="#37474F", fill_opacity=1)
    p.move_to(center)
    return VGroup(p, tx(label, 28).move_to(center))


def dit_box(center, w=1.5, h=2.0, lora=False, label="DiT"):
    r = RoundedRectangle(width=w, height=h, corner_radius=0.15, fill_color=BOX,
                         fill_opacity=1, stroke_color="#90A4AE", stroke_width=2).move_to(center)
    g = VGroup(r, tx(label, 34).move_to(r.get_center() + (UP * 0.2 if lora else 0)))
    if lora:
        tag = RoundedRectangle(width=w * 0.62, height=0.38, corner_radius=0.08,
                               fill_color=LORA, fill_opacity=1, stroke_width=0)
        tag.move_to(r.get_bottom() + UP * 0.4)
        g.add(tag, tx("LoRA", 20, BLACK, bold=True).move_to(tag))
    return g


def op_box(label, center, color=AMBER, w=0.95, h=0.62):
    r = RoundedRectangle(width=w, height=h, corner_radius=0.1, fill_color=BOX, fill_opacity=1,
                         stroke_color=color, stroke_width=2).move_to(center)
    return VGroup(r, tx(label, 26).move_to(center))


def count_lines(field, x, y_top, total=256):
    n1, n2, n4 = field.count(1), field.count(4), field.count(16)
    other = field.count() - n1 - n2 - n4
    rows = [tx(f"{col('1 × 1', HR)}: {n1} tokens", 28),
            tx(f"{col('2 × 2', LR)}: {n2} tokens", 28),
            tx(f"{col('4 × 4', LR3)}: {n4} tokens", 28)]
    if other:
        rows.append(tx(f"rectangles: {other} tokens", 28))
    L = field.count()
    eq = tx(f"{I('L')} = <b>{L}</b> tokens", 32)
    box = SurroundingRectangle(eq, color=WHITE, buff=0.14, stroke_width=2)
    pct = tx(f"{L} / {total} ≈ {round(100 * L / total)}% of the tokens", 24, DIM)
    g = VGroup(*rows, VGroup(eq, box), pct).arrange(DOWN, buff=0.28, aligned_edge=LEFT)
    return g.move_to([x, y_top - g.height / 2, 0], aligned_edge=LEFT).shift(LEFT * 0)


# ================================================================== beats
class Beats(Base):
    # ---------------------------------------------------------------- S00 hero
    def b_title(self):
        title = fit(tx(TITLE, 72, bold=True), 12.8).move_to([0, 2.75, 0])
        subt = tx(SUBTITLE, 32).move_to([0, 1.95, 0])
        g0 = Grid(16, 3.8, (0, -0.75))
        f0 = LoTField(g0, uniform(16))
        self.add(title, subt)
        if AUTHORS:
            self.add(fit(tx(AUTHORS, 22, DIM), 12.8).move_to([0, -3.3, 0]))
        self.play(LaggedStart(*[FadeIn(m, scale=0.6) for m in f0.mobs()], lag_ratio=0.004), run_time=1.2)
        for rects in (TOY, uniform(16), TOY_B, uniform(16), TOY):
            self.play(LaggedStart(*f0.retarget(rects, order_from=(8, 8)), lag_ratio=0.01), run_time=0.8)
            self.wait(0.3)
        self.wait(0.8)
        self.play(FadeOut(f0.group()), run_time=0.5)

    # ---------------------------------------------------------------- S01
    def b_tokencost(self):
        S = 4.6
        pic = img(RESULT, S).move_to([-3.5, 0.1, 0])
        g = Grid(16, S, (-3.5, 0.1))
        self.play(FadeIn(pic), run_time=0.5)
        self.caption("A diffusion transformer (DiT) generates an image as a grid of latent tokens.")
        lines = grid_lines(g)
        self.play(Create(lines), run_time=1.0)
        cols = patch_array(RESULT, 16)
        field = LoTField(g, uniform(16), colors=cols)
        self.play(LaggedStart(*[FadeIn(m) for m in field.mobs()], lag_ratio=0.003), FadeOut(lines), run_time=1.2)
        self.remove(pic)
        bf, bs = cell_box(g, FACE, AMBER), cell_box(g, SKY, PURPLE)
        self.play(Create(bf), Create(bs), run_time=0.5)
        leg = VGroup(VGroup(Square(0.26).set_stroke(AMBER, 3), tx("a token on her face", 26)).arrange(RIGHT, buff=0.2),
                     VGroup(Square(0.26).set_stroke(PURPLE, 3), tx("a token of empty sky", 26)).arrange(RIGHT, buff=0.2)
                     ).arrange(DOWN, buff=0.35, aligned_edge=LEFT).move_to([2.6, 0.6, 0])
        self.add(leg)
        self.caption("Both cost the same: every token passes every layer, at every step.")
        self.hold(0.3)
        self.play(*[m.animate.set_fill(HR, 1).set_stroke(HR) for m in field.mobs()], FadeOut(bf), FadeOut(bs),
                  run_time=0.8)
        self.remove(leg)
        cnt = tx(f"{I('h')} × {I('w')} = 16 × 16 = {col('256', HR)} tokens", 30).move_to([2.9, 0.6, 0])
        self.add(cnt)
        self.caption("Here 16 × 16 = 256 tokens, all at full resolution.")
        self.end_beat(keep=field.mobs() if not self.clip else ())
        return field

    def b_attention(self, field=None):
        if field is None:
            g = Grid(16, 4.6, (-3.5, 0.1))
            field = LoTField(g, uniform(16))
            for m in field.mobs():
                m.set_fill(HR, 1).set_stroke(HR)
            self.add(field.group())
        g = field.g
        A = img(gen("attn_v2.png"), 4.0).move_to([3.4, -0.1, 0])
        la = tx("self-attention: every token vs. every token", 22, DIM).next_to(A, UP, buff=0.15)
        lb = tx(f"256 × 256 = {col('65,536', HR)} query–key pairs", 26).next_to(A, DOWN, buff=0.2)
        self.play(FadeIn(A), run_time=0.5)
        self.add(la, lb)
        self.caption("Each step, self-attention compares every token with every other token.")
        q = Square(g.c).set_stroke(WHITE, 3).move_to(g.cell(5, 8))
        row = Rectangle(width=4.0, height=4.0 / 256).set_stroke(WHITE, 2).move_to(
            A.get_top() + DOWN * (5 * 16 + 8 + 0.5) * 4.0 / 256)
        self.play(Create(q), Create(row), run_time=0.4)
        for (i, j) in [(2, 3), (11, 12)]:
            k = i * 16 + j
            self.play(q.animate.move_to(g.cell(i, j)),
                      row.animate.move_to(A.get_top() + DOWN * (k + 0.5) * 4.0 / 256), run_time=0.7)
        self.hold()
        self.play(FadeOut(A), FadeOut(q), FadeOut(row), run_time=0.4)
        self.remove(la, lb)
        s1 = Square(0.9).set_fill(HR, 0.85).set_stroke(width=0).move_to([1.0, 1.2, 0])
        s2 = Square(1.8).set_fill(HR, 0.85).set_stroke(width=0).move_to([1.0, -1.1, 0])
        a1 = Square(0.9).set_fill(AMBER, 0.85).set_stroke(width=0).move_to([3.4, 1.2, 0])
        a2 = Square(3.6).set_fill(AMBER, 0.85).set_stroke(width=0).move_to([4.6, -1.1, 0]).scale(0.62)
        l1 = tx(f"{I('N')} tokens", 22, DIM).next_to(s1, UP, buff=0.12)
        l2 = tx(f"2× resolution: 4{I('N')}", 22, DIM).next_to(s2, DOWN, buff=0.12)
        l3 = tx(f"{I('N')}{sup('2')} pairs", 22, DIM).next_to(a1, UP, buff=0.12)
        l4 = tx(f"16 {I('N')}{sup('2')} pairs", 22, DIM).next_to(a2, DOWN, buff=0.12)
        self.play(FadeIn(s1), FadeIn(a1), run_time=0.4)
        self.add(l1, l3)
        self.caption("Double the resolution: 4× the tokens, 16× the attention. Video multiplies it by frames.")
        self.play(TransformFromCopy(s1, s2), TransformFromCopy(a1, a2), run_time=0.8)
        self.add(l2, l4)
        self.caption("But does every region of an image need the same detail?")
        self.end_beat()

    # ---------------------------------------------------------------- S02
    def b_graphics(self):
        # level of detail: a near object with many triangles, a far one with few
        def mesh(r, n, c, color):
            pts = [c + r * np.array([np.cos(2 * PI * k / n), np.sin(2 * PI * k / n), 0]) for k in range(n)]
            g = VGroup(Polygon(*pts).set_stroke(color, 2.5))
            inner = [c + 0.55 * (p - c) for p in pts]
            g.add(Polygon(*inner).set_stroke(color, 1.5, 0.8))
            for p, q in zip(pts, inner):
                g.add(Line(p, q).set_stroke(color, 1, 0.7), Line(q, c).set_stroke(color, 1, 0.5))
            return g
        near = mesh(1.25, 24, np.array([-4.9, 0.4, 0]), TEAL)
        far = mesh(0.45, 6, np.array([-2.3, 0.9, 0]), TEAL)
        ln = tx("near: many triangles", 22, DIM).next_to(near, DOWN, buff=0.25)
        lf = tx("far: few", 22, DIM).next_to(far, DOWN, buff=0.25)
        t1 = tx("level of detail", 26).move_to([-3.6, 2.45, 0])
        # variable-rate shading: shading-rate tiles, fine in the textured middle
        yy, xx = np.mgrid[0:8, 0:12]
        det = np.exp(-(((yy - 4) / 2.6) ** 2 + ((xx - 6) / 3.4) ** 2))
        gv = Grid(8, 3.2, (3.6, 0.4), m=12)
        vrs = LoTField(gv, quadtree(det, t4=0.25, t2=0.6))
        t2 = tx("variable-rate shading", 26).move_to([3.6, 2.45, 0])
        l2 = tx("shade flat regions once per 2 × 2 or 4 × 4 pixels", 22, DIM).next_to(gv.center + DOWN * 1.6, DOWN, buff=0.2)
        self.play(Create(near), Create(far), run_time=1.0)
        self.add(t1, ln, lf)
        self.caption("Graphics has long spent detail only where it is needed.")
        self.play(LaggedStart(*[FadeIn(m, scale=0.6) for m in vrs.mobs()], lag_ratio=0.02), run_time=1.0)
        self.add(t2, l2)
        self.caption("But rendering adapts to a scene that already exists, pixel by pixel.")
        self.end_beat()

    def b_plan(self):
        S = 4.6
        C = np.array([-3.4, 0.1, 0])
        canvas = Square(S).set_stroke(GHOST, 2).move_to(C)
        self.add(canvas, tx("the plan", 26, DIM).next_to(canvas, UP, buff=0.25))

        def box(u0, v0, u1, v1, name, colr):
            r = Rectangle(width=(u1 - u0) * S, height=(v1 - v0) * S).move_to(
                C + np.array([((u0 + u1) / 2 - 0.5) * S, (0.5 - (v0 + v1) / 2) * S, 0]))
            r = DashedVMobject(r.set_stroke(colr, 2.5), num_dashes=40)
            t = tx(name, 22, colr).next_to(r, UP, buff=0.06).align_to(r, LEFT).shift(RIGHT * 0.08)
            return VGroup(r, t)
        plans = [box(0.02, 0.02, 0.98, 0.56, "sky · coarse", DIM), box(0.34, 0.06, 0.58, 0.66, "girl · fine", WHITE),
                 box(0.06, 0.34, 0.95, 0.88, "corgi · fine", WHITE), box(0.02, 0.8, 0.98, 0.98, "grass · medium", AMBER)]
        self.play(LaggedStart(*[Create(p[0]) for p in plans], lag_ratio=0.25), run_time=1.2)
        self.add(*[p[1] for p in plans])
        self.caption("Image generators increasingly plan a layout first: what goes where.")
        pm = img(CANVAS, S).move_to(C)
        self.play(FadeIn(pm), *[FadeOut(p) for p in plans], run_time=0.8)
        self.add(tx("painted detail map: white = fine, black = coarse", 18, DIM).next_to(canvas, DOWN, buff=0.15))
        self.caption("An agent can even paint the plan as a detail map.")
        res = img(RESULT, S).move_to([3.4, 0.1, 0])
        self.play(FadeIn(res), GrowArrow(arrow([-0.95, 0.1, 0], [0.95, 0.1, 0])), run_time=0.6)
        self.add(tx("the generated image", 26, DIM).next_to(res, UP, buff=0.25))
        self.caption("Yet the plan only decides what goes where, never how much compute it gets.")
        self.end_beat()

    def b_bridge(self):
        q = fit(tx("Can a pretrained DiT spend its tokens where the plan asks for detail?", 40), 12.6)
        self.add(q)
        self.wait(2.4)
        self.remove(q)

    # ---------------------------------------------------------------- S03
    def b_merge(self):
        g = Grid(16, 5.0, (-3.4, 0.1))
        field = LoTField(g, uniform(16))
        self.play(LaggedStart(*[FadeIn(m) for m in field.mobs()], lag_ratio=0.003), run_time=1.0)
        dm = img(CANVAS, 2.4).move_to([3.6, 1.35, 0])
        self.play(FadeIn(dm), run_time=0.4)
        self.add(tx("detail map", 22, DIM).next_to(dm, UP, buff=0.15))
        self.caption("Start from the full grid: 256 tokens, all 1 × 1.")
        mid = [r for r in TOY if (r[1] - r[0]) <= 2]
        stage1 = mid + [(y, y + 2, x, x + 2) for r in TOY if (r[1] - r[0]) == 4
                        for y in (r[0], r[0] + 2) for x in (r[2], r[2] + 2)]
        self.caption("Where the map asks for less detail, each 2 × 2 block merges into one token.")
        self.play(LaggedStart(*field.retarget(stage1, order_from=(8, 8)), lag_ratio=0.02), run_time=2.2)
        self.caption("Flat sky merges further, into 4 × 4 tokens.")
        self.play(LaggedStart(*field.retarget(TOY, order_from=(8, 8)), lag_ratio=0.1), run_time=1.2)
        cl = count_lines(field, 1.6, 0.05)
        self.add(cl)
        self.caption(f"This toy layout: {field.count()} tokens instead of 256, and the same image size.")
        self.add(toy_note("toy layout on a 16 × 16 grid, from the painted map (paper Eq. 23)").move_to([-3.4, -2.65, 0]))
        self.end_beat(keep=field.mobs() if not self.clip else ())
        return field

    def b_shapes(self, field=None):
        if field is None:
            g = Grid(16, 5.0, (-3.4, 0.1))
            field = LoTField(g, TOY)
            self.add(field.group())
        rm = rect_merge_toy()
        self.caption("Tokens need not be square: any rectangle from 1 × 1 up to 8 × 8.")
        self.play(LaggedStart(*field.retarget(rm), lag_ratio=0.15), run_time=1.6)
        pick = (0, 2, 4, 8)
        hl = cell_box(field.g, pick, AMBER, 3)
        self.play(Create(hl), run_time=0.4)
        eq = tx(f"{I('R')}{sub(I('i'))} = [{I('u')}, {I('u')} + {EH}) × [{I('v')}, {I('v')} + {EW})", 34).move_to([3.4, 1.6, 0])
        ee = tx(f"{EH}, {EW} ∈ {{1, 2, 4, 8}}", 30).move_to([3.4, 0.75, 0])
        ex = tx(f"this one: {EH} = 2, {EW} = 4", 26, AMBER).move_to([3.4, -0.05, 0])
        self.add(eq, ee, ex)
        self.caption("A level-of-token layout tiles the grid with such rectangles, one token each.")
        L = field.count()
        self.add(tx(f"{I('L')} = <b>{L}</b> tokens", 32).move_to([3.4, -1.1, 0]))
        self.end_beat(keep=field.mobs() if not self.clip else ())
        return field

    def b_sequence(self, field=None):
        if field is None:
            g = Grid(16, 5.0, (-3.4, 0.1))
            field = LoTField(g, rect_merge_toy())
            self.add(field.group())
        rects = sorted(field.rects, key=lambda r: (r[0], r[2]))
        L = len(rects)
        side = min(0.11, 13.0 / L - 0.02)
        row = VGroup(*[Square(side).set_fill(level_color((r[1] - r[0]) * (r[3] - r[2])), 1).set_stroke(width=0)
                       for r in rects]).arrange(RIGHT, buff=0.02)
        if row.width > 13.4:
            row.scale_to_fit_width(13.4)
        row.move_to([0, 2.75, 0])
        mobs = [field.state[r] for r in rects]
        self.caption(f"The DiT sees one short sequence: {L} tokens instead of 256.")
        self.play(*[Transform(m, t) for m, t in zip(mobs, row)], run_time=2.0)
        big = Square(4.0).set_fill(GHOST, 1).set_stroke(width=0).move_to([-3.2, -0.55, 0])
        small = Square(4.0 * L / 256).set_fill(HR, 1).set_stroke(width=0)
        small.move_to(big.get_corner(UL), aligned_edge=UL)
        self.play(FadeIn(big), run_time=0.4)
        self.play(GrowFromEdge(small, UL), run_time=0.7)
        pct = round(100 * (L / 256) ** 2)
        info = VGroup(tx("full: 256 × 256 = 65,536 pairs", 26, DIM),
                      tx(f"level-of-token: {L} × {L} = {L * L:,} pairs", 26),
                      tx(f"≈ {pct}% of the attention work", 28, HR)).arrange(DOWN, buff=0.3, aligned_edge=LEFT)
        info.move_to([2.9, -0.55, 0])
        self.add(info)
        self.caption("Attention shrinks quadratically with the sequence; the MLPs linearly.")
        self.end_beat()

    # ---------------------------------------------------------------- S04
    def b_naive(self):
        fig = os.path.join(DATA, "fig21_full.png")
        S = 2.35
        xs = [-4.2, 0.0, 4.2]
        ys = [1.3, -1.15]
        cols_ = [0, 1, 4]
        names = [("token layout", DIM, False), ("direct full-res prediction", RED, False), ("LoT (ours)", HR, True)]
        stats = ["1,111 tokens · 2.2× faster", "1,752 tokens · 1.8× faster"]
        grid = Group()
        for r, y in enumerate(ys):
            for c, x in zip(cols_, xs):
                grid.add(img(panel(fig, r, c, f"fig21_r{r}_c{c}.png"), S).move_to([x, y, 0]))
        self.play(LaggedStart(*[FadeIn(m) for m in grid], lag_ratio=0.1), run_time=1.0)
        for (n, c, b), x in zip(names, xs):
            self.add(tx(n, 24, c, bold=b).move_to([x, ys[0] + S / 2 + 0.25, 0]))
        for r, y in enumerate(ys):
            self.add(tx(stats[r], 18, DIM).rotate(PI / 2).move_to([xs[0] - S / 2 - 0.3, y, 0]))
        self.add(tx("paper Fig. 21", 16, DIM).move_to([xs[2], ys[1] - S / 2 - 0.2, 0]))
        self.caption("Ask big tokens to predict the full-resolution flow directly: noise everywhere.")
        self.caption("Two things are missing: the right flow target, and each token's size.")
        self.end_beat()

    # ---------------------------------------------------------------- S05
    def b_patch(self):
        g = Grid(8, 3.0, (-4.9, 0.4))
        noise = noise_array(8, 3)
        cells = VGroup(*[token(g.hs, rgb_to_color(noise[i, j])).move_to(g.cell(i, j)) for i in range(8) for j in range(8)])
        lay = [(0, 4, 0, 4), (0, 2, 4, 6), (0, 2, 6, 8), (2, 4, 4, 6), (2, 4, 6, 8), (4, 8, 0, 2), (4, 6, 2, 4),
               (6, 8, 2, 4)] + [(y, y + 1, x, x + 1) for y in range(4, 8) for x in range(4, 8)]
        outl = VGroup(*[cell_box(g, r, level_color((r[1] - r[0]) * (r[3] - r[2])), 2.5) for r in lay])
        lx = tx(f"noisy latent {XT}", 24).next_to(cells, UP, buff=0.2)
        self.play(LaggedStart(*[FadeIn(c) for c in cells], lag_ratio=0.01), run_time=0.8)
        self.play(Create(outl), run_time=0.6)
        self.add(lx)
        self.caption("Take one 2 × 2 token: its patch holds 4 latent cells.")
        pr = (0, 2, 4, 6)
        sel = cell_box(g, pr, WHITE, 4)
        self.play(Create(sel), run_time=0.3)
        four = VGroup(*[token(0.42, rgb_to_color(noise[i, j]), WHITE, 1.2) for i in (0, 1) for j in (4, 5)])
        four.arrange(DOWN, buff=0.06).move_to([-2.3, 0.4, 0])
        l4 = tx(f"{XTI}: 4 × {I('D')}", 24).next_to(four, UP, buff=0.2)
        self.play(TransformFromCopy(VGroup(*[cells[i * 8 + j] for i in (0, 1) for j in (4, 5)]), four), run_time=0.8)
        self.add(l4)
        pj = op_box(AET, [-0.6, 0.4, 0])
        one = token(0.5, LR, WHITE, 1.2).move_to([1.1, 0.4, 0])
        l1 = tx(f"one token ∈ ℝ{sup(I('D'))}", 24).next_to(one, UP, buff=0.2)
        self.play(GrowArrow(arrow(four.get_right(), pj.get_left(), buff=0.12)), FadeIn(pj), run_time=0.4)
        self.play(GrowArrow(arrow(pj.get_right(), one.get_left(), buff=0.12)), FadeIn(one), run_time=0.4)
        self.add(l1)
        self.caption(f"A fixed basis {AET} per token size projects the patch to one ordinary-size token.")
        note = tx(f"{I('A')}{sub(I('e'))}: fitted to multi-scale VAE latents (orthogonal Procrustes)", 20, DIM)
        self.add(note.move_to([-1.3, -1.9, 0]))
        self.end_beat(keep=[one] if not self.clip else ())
        return one

    def b_flow(self, one=None):
        if one is None:
            one = token(0.5, LR, WHITE, 1.2).move_to([1.1, 0.4, 0])
            self.add(one)
        Y = 1.2
        self.play(one.animate.move_to([-6.1, Y + 0.95, 0]), run_time=0.6)
        seq = VGroup(*[token(0.3, c) for c in [HR, LR, LR3, HR, HR, LR, HR]]).arrange(DOWN, buff=0.06)
        seq.next_to(one, DOWN, buff=0.08)
        self.play(FadeIn(seq), run_time=0.3)
        dit = dit_box([-4.0, Y, 0], w=1.8, h=2.1, lora=True)
        self.play(GrowArrow(arrow([-5.75, Y, 0], dit.get_left())), FadeIn(dit), run_time=0.5)
        self.caption("The pretrained DiT (with LoRA) runs on the short sequence.")
        head = op_box(f"{I('h')}{sub('(2,2)')}", [-1.75, Y, 0], LR, w=1.1)
        self.play(GrowArrow(arrow(dit.get_right(), head.get_left())), FadeIn(head), run_time=0.4)
        ua = VGroup(*[token(0.36, HR, WHITE, 1.2) for _ in range(4)]).arrange(DOWN, buff=0.06).move_to([-0.2, Y, 0])
        lu = tx(f"{UA}: 4 cells", 24).next_to(ua, UP, buff=0.15)
        self.play(GrowArrow(arrow(head.get_right(), ua.get_left())), FadeIn(ua), run_time=0.4)
        self.add(lu)
        self.caption("A head per token size predicts a velocity for all 4 cells of the patch.")
        eqL = tx(f"û = {PE} {UA}", 40, HR)
        plus = tx("+", 40)
        eqR = tx(f"({I('I')} − {PE}) ({XT} + {UA}) / σ{sub(I('t'))}", 40, AMBER)
        eq = VGroup(eqL, plus, eqR).arrange(RIGHT, buff=0.25).move_to([0, -0.95, 0])
        la = tx("predicted, in the token's subspace", 20, HR).next_to(eqL, DOWN, buff=0.2)
        lb = tx("recovered in closed form from the noisy latent", 20, AMBER).next_to(eqR, DOWN, buff=0.2)
        self.add(eq, la, lb)
        self.caption("The part a token cannot hold is recovered from the noisy latent.")
        vel = VGroup(*[token(0.5, "#5C9DFF", WHITE, 1.2) for _ in range(4)]).arrange_in_grid(2, 2, buff=0.06)
        vel.move_to([3.4, Y, 0])
        lv = tx("full-resolution velocity", 24).next_to(vel, UP, buff=0.15)
        self.play(GrowArrow(arrow(ua.get_right(), vel.get_left(), buff=0.15)), TransformFromCopy(ua, vel), run_time=0.8)
        self.add(lv)
        vae = coder("D", [5.6, Y, 0])
        self.play(GrowArrow(arrow(vel.get_right(), vae.get_left())), FadeIn(vae), run_time=0.4)
        self.add(tx("VAE decodes as usual", 20, DIM).next_to(vae, DOWN, buff=0.2))
        self.caption("So every step still yields full-resolution flow, at the cost of a short sequence.")
        self.add(tx("paper Eq. 9 · patch-wise asymmetric flow, after AsymFlow", 16, DIM).move_to([0, -2.6, 0]))
        self.end_beat()

    # ---------------------------------------------------------------- S06
    def b_rope(self):
        n, S = 8, 3.6
        g = Grid(n, S, (-3.6, 0.3))
        cells = VGroup(*[Square(g.c * 0.92).set_stroke(GHOST, 1.2).set_fill(opacity=0).move_to(g.cell(i, j))
                         for i in range(n) for j in range(n)])
        idx = VGroup(*[tx(str(k), 16, DIM).move_to(g.cell(-0.8, k)) for k in range(n)] +
                     [tx(str(k), 16, DIM).move_to(g.cell(k, -0.8)) for k in range(n)])
        big = rect_token(g, (2, 4, 1, 5))
        one = rect_token(g, (6, 7, 6, 7))
        self.add(cells, idx)
        self.play(FadeIn(big), FadeIn(one), run_time=0.4)
        p = Dot(g.cell(2, 1), radius=0.08, color=AMBER).set_z_index(5)
        self.add(p)
        self.caption("Position: RoPE indexes each token at the centre of its patch.")
        self.play(p.animate.move_to(g.rect_center((2, 4, 1, 5))), run_time=0.8)
        d1 = Dot(g.cell(6, 6), radius=0.07, color=AMBER).set_z_index(5)
        self.add(d1)
        r1 = tx(f"centre = ({I('u')} + ({EH} − 1)/2, {I('v')} + ({EW} − 1)/2) = (2.5, 2.5)", 26).move_to([3.4, 2.45, 0])
        r2 = tx("a 1 × 1 token keeps its pretrained RoPE", 22, DIM).move_to([3.4, 1.95, 0])
        fit(r1, 6.6)
        self.add(r1, r2)
        self.caption("Size: height, width, area and aspect, embedded by a zero-initialised MLP.")
        s_eq = fit(tx(f"{I('s')}({I('e')}) = (log{sub('2')} {EH}, log{sub('2')} {EW}, log{sub('2')} {EH}{EW}, "
                      f"log{sub('2')} {EH}/{EW})", 26), 6.6).move_to([3.4, 1.15, 0])
        self.add(s_eq)
        vals = [1, 2, 3, -1]
        names = ["height", "width", "area", "aspect"]
        bars = VGroup()
        for k, (v, nm) in enumerate(zip(vals, names)):
            x = 2.0 + 0.95 * k
            b = Rectangle(width=0.5, height=abs(v) * 0.36).set_fill(level_color(8), 1).set_stroke(width=0)
            b.move_to([x, -0.75 + v * 0.18, 0])
            bars.add(VGroup(b, tx(str(v), 20).next_to(b, UP if v > 0 else DOWN, buff=0.06),
                            tx(nm, 18, DIM).move_to([x, -1.05 if v > 0 else -0.4, 0])))
        axis = Line([1.55, -0.75, 0], [5.3, -0.75, 0]).set_stroke(DIM, 1.5)
        self.play(Create(axis), LaggedStart(*[GrowFromEdge(b[0], DOWN if v > 0 else UP) for b, v in zip(bars, vals)],
                                            lag_ratio=0.15), run_time=0.8)
        self.add(*[VGroup(b[1], b[2]) for b in bars])
        self.add(tx(f"for the 2 × 4 token → MLP {I('g')}{sub('φ')} → added to its embedding", 20, DIM).move_to([3.4, -2.0, 0]))
        self.end_beat()

    def b_training(self):
        dit = dit_box([-2.2, 0.3, 0], w=2.4, h=2.8, lora=True)
        self.play(FadeIn(dit), run_time=0.4)
        heads = VGroup(*[op_box(f"{I('h')}{sub(e)}", [0.9, 1.3 - 0.8 * k, 0], level_color(a), w=1.2)
                         for k, (e, a) in enumerate([("(1,1)", 1), ("(2,2)", 4), ("(4,4)", 16)])])
        self.play(LaggedStart(*[FadeIn(h) for h in heads], lag_ratio=0.2), run_time=0.6)
        self.add(tx("one head per token size", 22, DIM).next_to(heads, DOWN, buff=0.2))
        base = VGroup(tx("pretrained base models:", 22, DIM), tx("FLUX.2 klein 4B and 9B (images)", 26),
                      tx("Wan2.1 14B (video, 81 frames at 720p)", 26)).arrange(DOWN, buff=0.18, aligned_edge=LEFT)
        base.move_to([4.6, 0.3, 0])
        self.add(fit(base, 4.6))
        self.caption("Post-train with LoRA; every size head starts from the pretrained head.")
        self.add(tx(f"{I('W')}{sub(I('e'))} = {I('A')}{sub(I('e'))} {I('W')}{sub('out')}", 30).move_to([0.9, 2.3, 0]))
        self.caption("Layouts in training come from masks, boxes, texture and depth, all mixed.")
        self.end_beat()

    # ---------------------------------------------------------------- S07
    def b_sources(self):
        fig = os.path.join(DATA, "fig4_full.png")
        names = ["bounding box", "semantic mask", "texture variance", "depth"]
        stats = [("1.73", "1.54"), ("2.18", "1.81"), ("1.36", "1.28"), ("3.11", "2.29")]
        xs = [-4.95, -1.65, 1.65, 4.95]
        S = 2.3
        tops, bots = Group(), Group()
        for k, x in enumerate(xs):
            tops.add(img(panel(fig, 0, 2 * k, f"fig4_r0_c{2 * k}.png"), S).move_to([x, 1.25, 0]))
            bots.add(img(panel(fig, 0, 2 * k + 1, f"fig4_r0_c{2 * k + 1}.png"), S).move_to([x, -1.2, 0]))
        self.play(LaggedStart(*[FadeIn(m) for m in tops], lag_ratio=0.15), run_time=0.9)
        for k, x in enumerate(xs):
            self.add(tx(names[k], 24).move_to([x, 1.25 + S / 2 + 0.25, 0]))
        self.add(tx("token layout", 20, DIM).move_to([-6.55, 1.25, 0]).rotate(PI / 2),
                 tx("generation", 20, DIM).move_to([-6.55, -1.2, 0]).rotate(PI / 2))
        self.caption("Boxes, masks, texture or depth: each becomes a token layout.")
        self.play(LaggedStart(*[FadeIn(m) for m in bots], lag_ratio=0.15), run_time=0.9)
        for k, x in enumerate(xs):
            self.add(VGroup(tx(f"{stats[k][0]}× fewer tokens", 18, DIM), tx(f"{stats[k][1]}× faster", 18, DIM))
                     .arrange(DOWN, buff=0.06).move_to([x, -1.2 - S / 2 - 0.35, 0]))
        self.caption("One LoT-FLUX.2 model follows all of them (paper Fig. 4).")
        self.end_beat()

    def b_budget(self):
        tags = ["B1024", "B1792", "B2560", "B3328", "B4096"]
        Bs = ["1,024", "1,792", "2,560", "3,328", "4,096"]
        xs = [-5.45 + 2.72 * k for k in range(5)]
        S = 2.3
        ims = Group(*[img(f"f10_deer_{t}_img.png", S).move_to([x, 1.0, 0]) for t, x in zip(tags, xs)])
        grs = Group(*[img(f"f10_deer_{t}_grid.png", S).move_to([x, -1.45, 0]) for t, x in zip(tags, xs)])
        for k, x in enumerate(xs):
            self.add(tx(f"{I('B')} = {Bs[k]}", 24, HR if k < 4 else WHITE).move_to([x, 1.0 + S / 2 + 0.25, 0]))
        self.play(LaggedStart(*[FadeIn(Group(a, b)) for a, b in zip(ims, grs)], lag_ratio=0.25), run_time=2.0)
        self.add(tx("full resolution", 18, DIM).move_to([xs[4], -1.45 - S / 2 - 0.2, 0]))
        self.caption("The token budget is one knob: same prompt, from 1,024 tokens up to all 4,096.")
        self.caption("Fewer tokens simplify textures; the deer and the scene stay (paper Fig. 10).")
        self.end_beat()

    def b_movebox(self):
        S = 3.1
        CL, CR, CD = np.array([-4.4, 0.55, 0]), np.array([0.0, 0.55, 0]), np.array([4.4, 0.55, 0])
        self.add(tx("token layout", 24, DIM).move_to(CL + UP * (S / 2 + 0.25)),
                 tx("LoT-FLUX.2 9B", 24, HR, bold=True).move_to(CR + UP * (S / 2 + 0.25)),
                 tx("FLUX.2 9B, no layout", 24, DIM).move_to(CD + UP * (S / 2 + 0.25)))
        prev = None
        first = True
        for r, name in (("a", "fire hydrant"), ("i", "cat")):
            ga, gb = img(f"f19_{r}_gridA.png", S).move_to(CL), img(f"f19_{r}_gridB.png", S).move_to(CL)
            ia, ib = img(f"f19_{r}_oursA.png", S).move_to(CR), img(f"f19_{r}_oursB.png", S).move_to(CR)
            dn = img(f"f19_{r}_dense.png", S).move_to(CD)
            lab = tx(f"“{name}”", 24).move_to([CR[0], CR[1] - S / 2 - 0.3, 0])
            grp = Group(ga, ia, dn, lab)
            if prev is None:
                self.play(FadeIn(grp), run_time=0.6)
            else:
                self.play(FadeOut(prev), FadeIn(grp), run_time=0.5)
            if first:
                self.caption("Same prompt, same noise, same token count: move the fine-token box…")
                first = False
            self.hold(0.2)
            self.play(FadeIn(gb), FadeIn(ib), run_time=0.9)
            self.remove(ga, ia)
            if r == "a":
                self.caption("…and the subject moves with it (paper Fig. 19).")
            self.wait(1.2)
            prev = Group(gb, ib, dn, lab)
        stat = VGroup(tx("subject inside the fine region, 2,096 COCO prompts:", 22, DIM),
                      tx(f"FLUX.2 9B 44.9%  →  {col('<b>LoT-FLUX.2 9B 66.8%</b>', HR)}", 28)).arrange(DOWN, buff=0.15)
        stat.move_to([0, -2.35, 0])
        self.add(stat)
        self.caption("Paper Tab. 4: the layout also steers where the subject appears.")
        self.end_beat()

    def b_driving(self):
        W_ = 6.2
        lay = Clip("show04_lay", w=W_)
        gen_ = Clip("show04_gen", w=W_)
        lay.mob.move_to([-3.3, 0.45, 0])
        gen_.mob.move_to([3.3, 0.45, 0])
        self.add(lay.mob, gen_.mob, framed(lay.mob, DIM, 1.0), framed(gen_.mob, DIM, 1.0))
        H = lay.mob.height
        self.add(tx("token layout (semantic classes)", 24, DIM).move_to([-3.3, 0.45 + H / 2 + 0.25, 0]),
                 tx("LoT-Wan2.1 14B", 24, HR, bold=True).move_to([3.3, 0.45 + H / 2 + 0.25, 0]))
        legend = VGroup(*[VGroup(Square(0.18).set_fill(c, 1).set_stroke(width=0), tx(t, 18, DIM)).arrange(RIGHT, buff=0.1)
                          for c, t in [("#EF4C62", "vehicles / people · 1×1"), ("#FFD640", "lanes · 1×1"),
                                       ("#A85CFF", "traffic controls · 1×1"), ("#3D7EFF", "road · 2×2"),
                                       ("#42C484", "sidewalk / buildings · 2×2"), ("#F8F8F8", "background · up to 4×4")]])
        legend.arrange_in_grid(rows=3, cols=2, buff=(0.35, 0.12), flow_order="dr")
        fit(legend, 6.2).move_to([-3.3, 0.45 - H / 2 - 0.2 - legend.height / 2, 0])
        self.add(legend)
        self.caption("In video, fine tokens follow cars, people and signs, frame by frame.")
        self.add(tx("1.97× fewer tokens", 34, AMBER).move_to([3.3, 0.45 - H / 2 - 0.55, 0]))
        self.caption("A rainy night drive, generated with 1.97× fewer tokens.")
        self.wait(1.0)
        self.end_beat()

    # ---------------------------------------------------------------- S08
    def b_speed(self):
        ax = Axes(x_range=[1, 3.2, 0.5], y_range=[1, 3.8, 0.5], x_length=5.4, y_length=4.2, tips=False,
                  axis_config={"include_numbers": False, "color": DIM, "stroke_width": 2}).move_to([-3.4, 0.2, 0])
        xt = VGroup(*[tx(f"{v:g}×", 20, DIM).next_to(ax.c2p(v, 1), DOWN, buff=0.15) for v in (1, 1.5, 2, 2.5, 3)])
        yt = VGroup(*[tx(f"{v:g}×", 20, DIM).next_to(ax.c2p(1, v), LEFT, buff=0.15) for v in (1, 2, 3)])
        xl = tx("fewer tokens (compression)", 22, DIM).next_to(ax, DOWN, buff=0.55)
        yl = tx("speed-up", 22, DIM).next_to(ax.c2p(1, 3.8), UP, buff=0.2)
        self.play(Create(ax), run_time=0.4)
        self.add(xt, yt, xl, yl)
        ip = [(1, 1), (1.893, 1.560), (2.406, 1.818), (2.935, 2.040)]
        vp = [(1, 1), (2.0, 2.262), (2.5, 2.909), (3.0, 3.530)]
        ci = VMobject(color=AMBER, stroke_width=4).set_points_as_corners([ax.c2p(*p) for p in ip])
        cv = VMobject(color=PURPLE, stroke_width=4).set_points_as_corners([ax.c2p(*p) for p in vp])
        dots = VGroup(*[Dot(ax.c2p(*p), radius=0.06, color=AMBER) for p in ip[1:]] +
                      [Dot(ax.c2p(*p), radius=0.06, color=PURPLE) for p in vp[1:]])
        self.caption("Fewer tokens per step, much faster generation (paper Tab. 1 and 2).")
        self.play(Create(ci), Create(cv), run_time=1.4)
        self.add(dots, tx("image", 22, AMBER).move_to(ax.c2p(1.35, 3.5)), tx("video", 22, PURPLE).move_to(ax.c2p(1.35, 3.15)))
        r1 = VGroup(tx("2.04×", 84, AMBER, bold=True), tx("faster images", 32))
        r2 = VGroup(tx("3.53×", 84, PURPLE, bold=True), tx("faster video", 32))
        for r in (r1, r2):
            r.arrange(RIGHT, buff=0.35, aligned_edge=DOWN)
        r1.move_to([3.6, 1.35, 0])
        r2.move_to([3.6, -0.1, 0]).align_to(r1, LEFT)
        r3 = tx("FLUX.2 4B · Wan2.1 14B, vs. the full-resolution model", 22, DIM).next_to(r2, DOWN, buff=0.45).align_to(r1, LEFT)
        self.add(r1, r2, fit(r3, 5.0))
        self.caption("Up to 2.04× faster images and 3.53× faster video.")
        self.end_beat()

    def b_quality(self):
        rows = [("Full-res. FLUX.2 4B", 10.43, "1.00×", "#FFFFFF"), ("ToMe-SD", 0.93, "2.00×", RED),
                ("DDiT", 7.41, "1.32×", DIM), ("Foveated Diffusion", 7.84, "1.18×", DIM), ("LoT (ours)", 9.55, "2.04×", HR)]
        head = tx("human-preference score HPSv3 ↑ at the same token budget (2.9× fewer tokens, paper Tab. 1)", 22, DIM)
        head.move_to([0, 2.55, 0])
        self.add(fit(head, 13))
        x0, W = -1.2, 6.0
        self.caption("Same budget: LoT scores closest to full resolution, and runs fastest.")
        for k, (n, v, sp, c) in enumerate(rows):
            y = 1.7 - 0.85 * k
            lab = tx(n, 26, c, bold=(k == 4))
            lab.shift(RIGHT * (x0 - 0.25 - lab.get_right()[0]) + UP * (y - lab.get_y()))
            bg = Rectangle(width=W, height=0.45, stroke_width=0).set_fill(GHOST, 1).move_to([x0 + W / 2, y, 0])
            fg = Rectangle(width=W * v / 11.0, height=0.45, stroke_width=0).set_fill(c if c != DIM else "#6E7B8B", 1)
            fg.move_to([x0 + W * v / 22.0, y, 0])
            self.add(lab, bg)
            self.play(GrowFromEdge(fg, LEFT), run_time=0.45)
            self.add(tx(f"{v:.2f}", 22).next_to(fg, RIGHT, buff=0.1) if v < 9 else tx(f"{v:.2f}", 22, BLACK).move_to(fg.get_right() + LEFT * 0.4),
                     tx(sp, 22, DIM).next_to(bg, RIGHT, buff=0.2))
        self.add(tx("speed-up", 18, DIM).move_to([x0 + W + 0.55, 2.1, 0]))
        self.caption("Token merging collapses; DDiT and Foveated Diffusion lose detail and speed.")
        self.end_beat()

    def b_imgcompare(self):
        fig = os.path.join(DATA, "fig7a_full.png")
        names = [("Full res.", "1.00×", "#FFFFFF"), ("ToMe-SD", "2.10×", RED), ("DDiT", "1.90×", RED),
                 ("Foveated Diffusion", "1.63×", RED), ("LoT (ours)", "2.29×", HR)]
        xs = [-5.45 + 2.72 * k for k in range(5)]
        S = 2.45
        ims = Group(*[img(panel(fig, 1, k, f"fig7a_r1_c{k}.png"), S).move_to([x, 0.45, 0]) for k, x in enumerate(xs)])
        self.play(LaggedStart(*[FadeIn(m) for m in ims], lag_ratio=0.12), run_time=1.0)
        for (n, sp, c), x in zip(names, xs):
            self.add(tx(n, 22, c if c != RED else "#FFFFFF", bold=(c == HR)).move_to([x, 0.45 + S / 2 + 0.45, 0]),
                     tx(sp + " faster", 20, c if c == HR else DIM).move_to([x, 0.45 + S / 2 + 0.15, 0]))
        self.add(tx("noisy low-res regions", 20, RED).move_to([(xs[1] + xs[2]) / 2, 0.45 - S / 2 - 0.25, 0]),
                 tx("seams at token borders", 20, RED).move_to([xs[3], 0.45 - S / 2 - 0.25, 0]))
        self.add(tx("same budget: 1,269 tokens · paper Fig. 7a", 18, DIM).move_to([0, -1.75, 0]))
        self.caption("At the same budget, baselines blur or break; LoT keeps the detail, and is fastest.")
        self.end_beat()

    def b_vidcompare(self):
        ci = json.load(open(os.path.join(CLIPS, "comparisons_index.json")))[2]      # the Fig. 7b example
        W_ = 3.2
        xs = [-5.1, -1.7, 1.7, 5.1]
        spec = [("Dense Wan2.1", "Full res.", "1.00×", "#FFFFFF"), ("ToMe-SD", "ToMe-SD", "1.60×", RED),
                ("Foveated Diffusion", "Foveated Diffusion", "1.59×", RED), ("Ours", "LoT (ours)", "1.62×", HR)]
        for (key, name, sp, c), x in zip(spec, xs):
            cl = Clip(ci["videos"][key], w=W_)
            cl.mob.move_to([x, 0.5, 0])
            self.add(cl.mob, framed(cl.mob, HR if c == HR else DIM, 2 if c == HR else 1.0))
            h = cl.mob.height
            self.add(tx(name, 22, c if c != RED else "#FFFFFF", bold=(c == HR)).move_to([x, 0.5 + h / 2 + 0.45, 0]),
                     tx(sp + " faster", 20, c if c == HR else DIM).move_to([x, 0.5 + h / 2 + 0.15, 0]))
        h = W_ * 9 / 16
        self.add(tx("noise", 20, RED).move_to([xs[1], 0.5 - h / 2 - 0.25, 0]),
                 tx("distortions", 20, RED).move_to([xs[2], 0.5 - h / 2 - 0.25, 0]))
        self.add(tx("same budget (1.51× fewer tokens) · Wan2.1 14B, 81 frames at 720p · paper Fig. 7b", 18, DIM).move_to([0, -1.3, 0]))
        self.caption("Video at the same budget: ToMe-SD turns to noise, Foveated distorts, LoT holds up.")
        self.hold(1.0)
        self.end_beat()

    # ---------------------------------------------------------------- S09
    def b_limit(self):
        S = 2.8
        ims = Group(*[img(f"fig22_{k}.png", S) for k in range(4)]).arrange(RIGHT, buff=0.25).move_to([0, 0.45, 0])
        self.play(LaggedStart(*[FadeIn(m) for m in ims], lag_ratio=0.12), run_time=0.8)
        self.add(tx("3,189 tokens (1.28× fewer)", 24).next_to(Group(ims[0], ims[1]), UP, buff=0.2),
                 tx("1,466 tokens (2.79× fewer)", 24).next_to(Group(ims[2], ims[3]), UP, buff=0.2),
                 tx("broken staff lines", 22, RED).next_to(ims[3], DOWN, buff=0.15))
        self.caption("Fine structure everywhere, like sheet music, leaves little to save (paper Fig. 22).")
        self.caption("Level-of-token layouts pay off when detail varies across the image.")
        self.end_beat()

    def b_ideas(self):
        t = tx("Level-of-Token Diffusion in three ideas", 40, bold=True).move_to([0, 2.45, 0])
        pts = ["1.  Plan the detail: a layout of tokens of any size and shape",
               "2.  Patch-wise asymmetric flow: full-resolution flow from a short sequence",
               "3.  Size-aware positions and embeddings: post-train a pretrained DiT"]
        self.add(t)
        for k, p in enumerate(pts):
            m = fit(tx(p, 27), 12.4)
            m.move_to([0, 1.3 - 0.85 * k, 0]).align_to([-6.0, 0, 0], LEFT)
            self.add(m)
            self.wait(1.8)
        self.add(fit(tx("up to 2.04× faster images  ·  up to 3.53× faster video  ·  layout-adaptive", 28, HR), 12.8)
                 .move_to([0, -1.5, 0]))
        self.wait(3.0)
        self.clear_all(run_time=0.6)
        if self.hdr is not None:
            self.remove(self.hdr)

    def b_end(self):
        self.add(tx(TITLE, 60, bold=True).move_to([0, 0.9, 0]))
        if LINKS:
            self.add(tx(LINKS, 28, DIM).move_to([0, -0.1, 0]))
        if AUTHORS:
            self.add(fit(tx(AUTHORS, 24, DIM), 12.8).move_to([0, -0.8, 0]))
        self.wait(4.0)


# ================================================================== scenes
class S00_Title(Beats):
    def construct(self):
        self.b_title()


class S01_TokenCost(Beats):
    def construct(self):
        self.title_card("Part 1 · The problem", "Every token costs the same")
        self.header("1 · Every token costs the same")
        field = self.b_tokencost()
        self.b_attention(field)


class S02_Plans(Beats):
    def construct(self):
        self.header("2 · Detail can be planned")
        self.b_graphics()
        self.b_plan()
        self.b_bridge()


class S03_Layouts(Beats):
    def construct(self):
        self.title_card("Part 2 · The method", "Level-of-token layouts")
        self.header("3 · Level-of-token layouts")
        f = self.b_merge()
        f = self.b_shapes(f)
        self.b_sequence(f)


class S04_Naive(Beats):
    def construct(self):
        self.header("4 · Just plug it in?  Predicting full-resolution flow directly")
        self.b_naive()


class S05_AsymFlow(Beats):
    def construct(self):
        self.header("5 · Patch-wise asymmetric flow")
        one = self.b_patch()
        self.b_flow(one)


class S06_SizeAware(Beats):
    def construct(self):
        self.header("6 · Telling the DiT each token's size")
        self.b_rope()
        self.b_training()


class S07_Layouts(Beats):
    def construct(self):
        self.header("7 · Layouts are just data")
        self.b_sources()
        self.b_budget()
        self.b_movebox()
        self.b_driving()


class S08_Results(Beats):
    def construct(self):
        self.title_card("Part 3", "Results", hold=1.4)
        self.header("8 · Results")
        self.b_speed()
        self.b_quality()
        self.b_imgcompare()
        self.b_vidcompare()


class S09_Outlook(Beats):
    def construct(self):
        self.header("9 · Limitations and outlook")
        self.b_limit()
        self.b_ideas()
        self.b_end()
