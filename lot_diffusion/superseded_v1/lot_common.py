"""Shared pieces for every Level-of-Token (LoT) Diffusion animation (clips/, the 5-min cut, teaser_1min/).

3b1b style, same family as foveated_diffusion/teaser_5min: CMU Serif, black background, Pango markup (no LaTeX),
text appears instantly. Token layouts are the paper's real ones (data/layouts/, built by scratch/prep_data.py).

Main pieces
  tx(), mathify(), sub(), sup(), col()      Pango text helpers
  LEVEL colours                            token colour by log2(area): 1x1 blue ... 8x8 purple
  Layout                                   a token layout (rects on an H x W lattice), load_layout()
  LayoutImage                              a layout rasterised into an ImageMobject; morph(a, b, t) between layouts
  VideoCard                                website video frames playing inside the scene (keeps playing while moving)
  Still                                    framed image with optional label
  Base                                     scene base: header, caption with reading-time hold, counters, pad_to()
"""
import glob
import json
import os
import re

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from manim import *

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
IMG = os.path.join(DATA, "img")
LAY = os.path.join(DATA, "layouts")
CLIPS = os.path.join(DATA, "clips")
GEN = os.path.join(DATA, "generated")
os.makedirs(GEN, exist_ok=True)

config.background_color = BLACK

# ------------------------------------------------------------------ style
FONT = "CMU Serif"
MATHFONT = "Latin Modern Math"
SANS = "Lato"
DIM = "#8A8A8A"
GHOST = "#3A3A3A"
RED = "#FF5252"
AMBER = "#FFCA28"
TEAL = "#4DD0E1"
PURPLE = "#B388FF"
GOLD = "#F0AC5F"
BOX = "#1E2A33"
HEADER_Y = 3.62
CAPTION_Y = -3.5

# token colour by level k = log2(area in lattice cells); 1x1 = HR blue and 2x2 = LR green as in Foveated Diffusion
LEVEL_HEX = ["#448AFF", "#29B6F6", "#00C853", "#9CCC65", "#FFCA28", "#FFA726", "#B388FF"]
LEVEL = [ManimColor(c) for c in LEVEL_HEX]
FINE, MID, COARSE = LEVEL_HEX[0], LEVEL_HEX[2], LEVEL_HEX[4]
METHOD_HEX = {"Full res.": "#BDBDBD", "ToMe-SD": "#FF7043", "DDiT": "#AB47BC", "Foveated Diffusion": "#00C853",
              "Ours": "#448AFF", "LoT": "#448AFF"}

MATHCH = set("×⊙→←≈∼−∈𝒩ℒ𝔼′≪≤≥‖·↓↑∑⊤ℝσεθφ⌈⌉⊕√∀")


def hex_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def level_of(eh, ew):
    return int(round(np.log2(max(eh * ew, 1))))


def level_hex(eh, ew):
    return LEVEL_HEX[min(level_of(eh, ew), 6)]


def mathify(s):
    out, inside = [], False
    for c in s:
        if c == "<":
            inside = True
        if c == ">":
            inside = False
        out.append(f'<span font_family="{MATHFONT}">{c}</span>' if (c in MATHCH and not inside) else c)
    return "".join(out)


def sub(x):
    return f'<span rise="-3000"><small><small>{x}</small></small></span>'


def sup(x):
    return f'<span rise="4500"><small><small>{x}</small></small></span>'


def col(x, c):
    return f'<span foreground="{c}">{x}</span>'


def it(x):
    return f"<i>{x}</i>"


def tx(s, size=30, color=WHITE, bold=False, italic=False, font=FONT):
    if bold:
        s = f"<b>{s}</b>"
    if italic:
        s = f"<i>{s}</i>"
    return MarkupText(mathify(s), font=font, font_size=size, color=color)


def fmt(n):
    return f"{int(round(n)):,}"


# ------------------------------------------------------------------ layouts
class Layout:
    """Token layout: rects [y0, y1, x0, x1] (half-open) on an H x W lattice of finest tokens."""

    def __init__(self, rects, H, W):
        self.rects = [tuple(int(v) for v in r) for r in rects]
        self.H, self.W = int(H), int(W)

    @property
    def L(self):
        return len(self.rects)

    @property
    def dense(self):
        return self.H * self.W

    @property
    def compression(self):
        return self.dense / max(self.L, 1)

    @staticmethod
    def uniform(H, W, e=1):
        return Layout([(y, min(y + e, H), x, min(x + e, W)) for y in range(0, H, e) for x in range(0, W, e)], H, W)

    def owner(self):
        """lattice cell -> index of the token covering it"""
        o = -np.ones((self.H, self.W), int)
        for k, (y0, y1, x0, x1) in enumerate(self.rects):
            o[y0:y1, x0:x1] = k
        return o

    def level_counts(self):
        c = {}
        for (y0, y1, x0, x1) in self.rects:
            key = (y1 - y0, x1 - x0)
            c[key] = c.get(key, 0) + 1
        return c


def load_layout(name, slice_k=None):
    d = json.load(open(os.path.join(LAY, name + ".json")))
    if "slices" in d:
        k = 0 if slice_k is None else slice_k
        return Layout(d["slices"][k], d["H"], d["W"])
    return Layout(d["rects"], d["H"], d["W"])


def load_video_layouts(name):
    d = json.load(open(os.path.join(LAY, name + ".json")))
    return [Layout(s, d["H"], d["W"]) for s in d["slices"]]


def quadtree_from_detail(detail, thresholds=(0.25, 0.5, 0.75), top=8):
    """Paper Eq. 23: start from top x top blocks, split a block of extent b when max detail >= t_b."""
    H, W = detail.shape
    tb = {8: thresholds[0], 4: thresholds[1], 2: thresholds[2]}
    rects = []

    def rec(y, x, b):
        blk = detail[y:y + b, x:x + b]
        if b > 1 and blk.size and blk.max() >= tb.get(b, 1.0):
            h = b // 2
            for dy in (0, h):
                for dx in (0, h):
                    if y + dy < H and x + dx < W:
                        rec(y + dy, x + dx, h)
        else:
            rects.append((y, min(y + b, H), x, min(x + b, W)))

    for y in range(0, H, top):
        for x in range(0, W, top):
            rec(y, x, top)
    return Layout(rects, H, W)


def mean_colors(path, layout):
    """per-token mean colour of an image (image is resized to the lattice first)"""
    a = np.asarray(Image.open(path).convert("RGB").resize((layout.W * 4, layout.H * 4), Image.BOX)).astype(float)
    out = []
    for (y0, y1, x0, x1) in layout.rects:
        out.append(tuple(int(v) for v in a[4 * y0:4 * y1, 4 * x0:4 * x1].reshape(-1, 3).mean(0)))
    return out


# ------------------------------------------------------------------ rasterised layouts
def draw_layout(layout, px_w, mode="level", image=None, alpha=1.0, gap=0.12, line=None, rect_override=None,
                fills=None, bg=(0, 0, 0, 0), radius=0.18, min_gap_px=1.0):
    """RGBA uint8 array of a token layout.

    mode: "level" (colour by token area), "image" (per-token mean colour of `image`), "outline" (lines only),
          "fills" (explicit per-token RGB list).
    gap:  inset per side as a fraction of one lattice cell (tokens read as separate rounded tiles).
    rect_override: optional float rects (same order as layout.rects) for morphs.
    """
    ph = int(round(px_w * layout.H / layout.W))
    S = 2  # supersample
    im = Image.new("RGBA", (px_w * S, ph * S), bg)
    d = ImageDraw.Draw(im)
    cw = px_w * S / layout.W
    if mode == "image" and fills is None:
        fills = mean_colors(image, layout)
    rects = rect_override if rect_override is not None else layout.rects
    g = max(gap * cw, min_gap_px * S)
    a255 = int(255 * alpha)
    for k, (y0, y1, x0, x1) in enumerate(rects):
        eh, ew = layout.rects[k][1] - layout.rects[k][0], layout.rects[k][3] - layout.rects[k][2]
        box = [x0 * cw + g, y0 * cw + g, x1 * cw - g, y1 * cw - g]
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        r = radius * min(box[2] - box[0], box[3] - box[1])
        if mode == "outline":
            c = hex_rgb(line or level_hex(eh, ew))
            d.rounded_rectangle(box, radius=r, outline=c + (a255,), width=max(1, int(S * 1.2)))
            continue
        if mode in ("image", "fills"):
            c = fills[k]
        else:
            c = hex_rgb(level_hex(eh, ew))
        d.rounded_rectangle(box, radius=r, fill=tuple(c) + (a255,))
        if line:
            d.rounded_rectangle(box, radius=r, outline=hex_rgb(line) + (a255,), width=S)
    im = im.resize((px_w, ph), Image.LANCZOS)
    return np.asarray(im)


def px_per_unit():
    return config.pixel_width / config.frame_width


def layout_mobject(layout, width, mode="level", image=None, **kw):
    px = int(min(max(width * px_per_unit(), 64), 2400))
    arr = draw_layout(layout, px, mode=mode, image=image, **kw)
    m = ImageMobject(arr)
    m.set_resampling_algorithm(RESAMPLING_ALGORITHMS["lanczos"])
    m.set_width(width)
    return m


def morph_rects(src, dst):
    """For each dst token, the union box of the src tokens whose centres fall inside it (its 'children')."""
    so = np.array(src.rects, float)
    cy = (so[:, 0] + so[:, 1]) / 2
    cx = (so[:, 2] + so[:, 3]) / 2
    starts = []
    for (y0, y1, x0, x1) in dst.rects:
        m = (cy >= y0) & (cy < y1) & (cx >= x0) & (cx < x1)
        if not m.any():
            starts.append((y0, y1, x0, x1))
            continue
        s = so[m]
        starts.append((s[:, 0].min(), s[:, 1].max(), s[:, 2].min(), s[:, 3].max()))
    return starts


class LayoutMorph:
    """ImageMobject showing a layout that can merge from a finer layout (uniform grid) into `dst`.

    t = 0: every dst token is drawn as the fine tokens it covers (looks like `src`), t = 1: `dst`.
    Implementation: fine tokens slide/scale toward the merged box, then the merged token fades in over them.
    """

    def __init__(self, src, dst, width, center=ORIGIN, mode="level", image=None, gap=0.12, src_mode=None):
        self.src, self.dst, self.width, self.mode, self.image, self.gap = src, dst, width, mode, image, gap
        self.src_mode = src_mode or mode
        self.px = int(min(max(width * px_per_unit(), 64), 2400))
        self.t = ValueTracker(0.0)
        # owner of each src token = dst token containing its centre
        so = np.array(src.rects, float)
        cy, cx = (so[:, 0] + so[:, 1]) / 2, (so[:, 2] + so[:, 3]) / 2
        own = dst.owner()
        self.parent = own[np.clip(cy.astype(int), 0, dst.H - 1), np.clip(cx.astype(int), 0, dst.W - 1)]
        self.src_fill = mean_colors(image, src) if (image and self.src_mode == "image") else None
        self.dst_fill = mean_colors(image, dst) if (image and mode == "image") else None
        self.img = ImageMobject(self.frame(0.0))
        self.img.set_resampling_algorithm(RESAMPLING_ALGORITHMS["lanczos"])
        self.img.set_width(width).move_to(center)
        self.last = None
        self.img.add_updater(self._tick)

    def frame(self, t):
        t = float(np.clip(t, 0, 1))
        s = t * t * (3 - 2 * t)
        # fine tokens pulled toward their parent token's box (shrink the gaps between siblings)
        rects = []
        for k, (y0, y1, x0, x1) in enumerate(self.src.rects):
            P = self.dst.rects[self.parent[k]]
            if (P[1] - P[0]) * (P[3] - P[2]) == 1:
                rects.append((y0, y1, x0, x1))
                continue
            # expand each child by s * (half the gap) so siblings touch at s = 1
            e = s * self.gap
            rects.append((max(y0 - e, P[0]), min(y1 + e, P[1]), max(x0 - e, P[2]), min(x1 + e, P[3])))
        if self.src_mode == "image":
            fills = self.src_fill
            if self.dst_fill is not None:
                fills = [tuple(int((1 - s) * a + s * b) for a, b in zip(self.src_fill[k], self.dst_fill[self.parent[k]]))
                         for k in range(len(self.src.rects))]
            base = draw_layout(self.src, self.px, mode="fills", fills=fills, gap=self.gap, rect_override=rects)
        else:
            if s > 0:  # children take the parent's level colour
                fills = []
                for k in range(len(self.src.rects)):
                    P = self.dst.rects[self.parent[k]]
                    c0 = np.array(hex_rgb(level_hex(1, 1)))
                    c1 = np.array(hex_rgb(level_hex(P[1] - P[0], P[3] - P[2])))
                    fills.append(tuple(int(v) for v in (1 - s) * c0 + s * c1))
                base = draw_layout(self.src, self.px, mode="fills", fills=fills, gap=self.gap, rect_override=rects)
            else:
                base = draw_layout(self.src, self.px, mode=self.src_mode, image=self.image, gap=self.gap)
        if s <= 0.55:
            return base
        a = (s - 0.55) / 0.45
        top = draw_layout(self.dst, self.px, mode=self.mode, image=self.image, gap=self.gap,
                          fills=self.dst_fill, alpha=a)
        B = base.astype(float)
        T = top.astype(float)
        ta = T[..., 3:4] / 255.0
        out = B.copy()
        out[..., :3] = B[..., :3] * (1 - ta) + T[..., :3] * ta
        out[..., 3] = np.maximum(B[..., 3] * (1 - a), T[..., 3])
        return out.astype(np.uint8)

    def _tick(self, m, dt=0):
        t = round(self.t.get_value(), 3)
        if t != self.last:
            c, w = m.get_center(), m.width
            m.pixel_array = self.frame(t)
            self.last = t


# ------------------------------------------------------------------ images and video
def img(path, h=None, w=None):
    p = path if os.path.isabs(path) else os.path.join(IMG, path)
    m = ImageMobject(p)
    m.set_resampling_algorithm(RESAMPLING_ALGORITHMS["lanczos"])
    if h:
        m.set_height(h)
    if w:
        m.set_width(w)
    return m


def framed(m, color=DIM, sw=1.5, buff=0.0):
    r = Rectangle(width=m.width + 2 * buff, height=m.height + 2 * buff).move_to(m)
    r.set_stroke(color, sw).set_fill(opacity=0)
    return r


def label_under(m, s, size=22, color=DIM, buff=0.12):
    return tx(s, size, color).next_to(m, DOWN, buff=buff)


def label_over(m, s, size=22, color=DIM, buff=0.12):
    return tx(s, size, color).next_to(m, UP, buff=buff)


_CLIP_INDEX = None


def clip_info(key):
    global _CLIP_INDEX
    if _CLIP_INDEX is None:
        _CLIP_INDEX = json.load(open(os.path.join(CLIPS, "index.json")))
    return _CLIP_INDEX[key]


class VideoCard:
    """A website video playing inside the scene: an ImageMobject whose pixels follow scene time.

    x, y, w, alpha, dim are ValueTrackers applied by an updater, so the clip keeps playing while it moves or fades.
    speed: playback rate (1 = real time). loop: wrap around; otherwise hold the last frame. start: offset (s).
    box: (w, h) the video is contained in. cover=True crops to fill the box instead.
    """

    def __init__(self, key, box=(6.4, 3.6), center=ORIGIN, speed=1.0, loop=True, start=0.0, radius=0.08,
                 border=DIM, border_w=1.2, cover=False, max_px=None, crop=None):
        """crop: optional fractional source crop (x0, y0, x1, y1) applied before fitting"""
        self.key = key
        self.crop = crop
        info = clip_info(key)
        self.fps, self.n = info["fps"], info["n"]
        self.paths = sorted(glob.glob(os.path.join(CLIPS, key, "*.jpg")))
        self.n = len(self.paths)
        self.speed, self.loop, self.clock, self.playing, self.last = speed, loop, start, False, None
        self.cache = {}
        W0, H0 = Image.open(self.paths[0]).size
        if crop:
            W0, H0 = W0 * (crop[2] - crop[0]), H0 * (crop[3] - crop[1])
        if cover:
            self.w_u, self.h_u = box
        else:
            s = min(box[0] / W0, box[1] / H0)
            self.w_u, self.h_u = W0 * s, H0 * s
        self.cover_aspect = self.w_u / self.h_u if cover else None
        self.pw = int(round(min(self.w_u * px_per_unit(), max_px or 1920)))
        self.pw = max(self.pw, 16)
        self.ph = max(int(round(self.pw * self.h_u / self.w_u)), 9)
        r = radius * self.pw / self.w_u
        yy, xx = np.mgrid[0:self.ph, 0:self.pw]
        if r > 0.5:
            dx = np.maximum(np.maximum(r - xx, xx - (self.pw - 1 - r)), 0)
            dy = np.maximum(np.maximum(r - yy, yy - (self.ph - 1 - r)), 0)
            self.mask = np.clip(r - np.sqrt(dx ** 2 + dy ** 2) + 0.5, 0, 1)
        else:
            self.mask = np.ones((self.ph, self.pw))
        c = np.array(center, dtype=float)
        self.x, self.y = ValueTracker(c[0]), ValueTracker(c[1])
        self.w, self.alpha, self.dim = ValueTracker(self.w_u), ValueTracker(1.0), ValueTracker(0.0)
        self.img = ImageMobject(np.zeros((self.ph, self.pw, 4), dtype=np.uint8))
        self.img.set_width(self.w_u)
        self.border = RoundedRectangle(width=self.w_u, height=self.h_u, corner_radius=max(radius, 1e-3))
        self.border.set_stroke(border or BLACK, border_w if border else 0, opacity=0).set_fill(opacity=0)
        self.border_op = 0.8 if border else 0
        self.img.add_updater(self._tick)
        self.border.add_updater(self._tick_border)
        self._tick(self.img, 0)

    def frame(self, i):
        if i not in self.cache:
            im = Image.open(self.paths[i]).convert("RGB")
            if self.crop:
                W0, H0 = im.size
                x0, y0, x1, y1 = self.crop
                im = im.crop((int(x0 * W0), int(y0 * H0), int(x1 * W0), int(y1 * H0)))
            if self.cover_aspect is not None:
                W0, H0 = im.size
                if W0 / H0 > self.cover_aspect:
                    nw = int(H0 * self.cover_aspect)
                    im = im.crop(((W0 - nw) // 2, 0, (W0 - nw) // 2 + nw, H0))
                else:
                    nh = int(W0 / self.cover_aspect)
                    im = im.crop((0, (H0 - nh) // 2, W0, (H0 - nh) // 2 + nh))
            im = im.resize((self.pw, self.ph), Image.LANCZOS)
            self.cache[i] = np.array(im)
            if len(self.cache) > 400:
                self.cache.pop(next(iter(self.cache)))
        return self.cache[i]

    def index(self):
        i = int(self.clock * self.fps)
        return i % self.n if self.loop else min(max(i, 0), self.n - 1)

    def pos(self):
        return np.array([self.x.get_value(), self.y.get_value(), 0])

    def _tick(self, m, dt):
        if self.playing:
            self.clock += dt * self.speed
        i = self.index()
        a, d = round(self.alpha.get_value(), 3), round(self.dim.get_value(), 3)
        if (i, a, d) != self.last:
            rgb = self.frame(i) if d == 0 else (self.frame(i) * (1 - d)).astype(np.uint8)
            m.pixel_array = np.dstack([rgb, (255 * self.mask * a).astype(np.uint8)])
            self.last = (i, a, d)
        w = self.w.get_value()
        if abs(m.width - w) > 1e-4:
            m.set_width(w)
        m.move_to(self.pos())

    def _tick_border(self, m):
        k = self.w.get_value() / self.w_u
        if abs(m.width - self.w_u * k) > 1e-4:
            m.set_width(self.w_u * k)
        m.move_to(self.pos()).set_stroke(opacity=self.border_op * self.alpha.get_value())

    @property
    def height(self):
        return self.h_u * self.w.get_value() / self.w_u

    @property
    def width(self):
        return self.w.get_value()

    def rect(self):
        """static Rectangle matching the card now (for next_to / labels)"""
        return Rectangle(width=self.width, height=self.height).move_to(self.pos())

    def mobs(self):
        return [self.img, self.border]

    def start(self, scene, alpha=None, play=True):
        self.playing = play
        if alpha is not None:
            self.alpha.set_value(alpha)
        scene.add(*self.mobs())
        return self

    def stop(self, scene):
        self.playing = False
        scene.remove(*self.mobs())


def hidden_driver(scene, key, box, center, speed=1.0):
    """an invisible, playing VideoCard used only as a clock (e.g. to drive a layout video)"""
    c = VideoCard(key, box=box, center=center, speed=speed, border=None)
    c.alpha.set_value(0.0)
    c.playing = True
    scene.add(c.img)
    return c


# ------------------------------------------------------------------ scene base
class Base(Scene):
    """header() top-left, caption() bottom with a reading-time hold, pad_to() for exact clip lengths."""

    CPS = 16.0  # caption reading speed (characters per second) + 0.8 s

    def setup(self):
        self.hdr = None
        self.cap = None
        self.cap_t0 = 0.0
        self.cap_need = 0.0

    @property
    def now(self):
        return self.renderer.time

    def header(self, s, size=24):
        if self.hdr is not None:
            self.remove(self.hdr)
        self.hdr = tx(s, size, DIM).to_corner(UL, buff=0.35).set_y(HEADER_Y)
        self.add(self.hdr)
        return self.hdr

    def _hold_caption(self):
        if self.cap is not None:
            left = self.cap_need - (self.now - self.cap_t0)
            if left > 0.05:
                self.wait(left)

    @staticmethod
    def _split(s):
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

    def caption(self, s, wait=None, size=28, hold=True, y=CAPTION_Y, maxw=13.2):
        if hold:
            self._hold_caption()
        s = re.sub(r"(\d) × (\d)", "\\1\u00a0×\u00a0\\2", s)
        if self.cap is not None:
            self.remove(self.cap)
        c = tx(s, size)
        if c.width > maxw:
            parts = self._split(s)
            c = VGroup(*[tx(p, size - 2) for p in parts]).arrange(DOWN, buff=0.12)
            if c.width > maxw:
                c.scale_to_fit_width(maxw)
        c.move_to([0, y, 0])
        if c.get_bottom()[1] < -3.7:
            c.shift(UP * (-3.7 - c.get_bottom()[1]))
        self.cap = c
        plain = re.sub(r"<[^>]+>", "", s)
        self.cap_t0 = self.now
        self.cap_need = len(plain) / self.CPS + 0.8
        self.add(c)
        if wait:
            self.wait(wait)
        return c

    def uncaption(self, hold=True):
        if hold:
            self._hold_caption()
        if self.cap is not None:
            self.remove(self.cap)
            self.cap = None

    def title_card(self, top, title, hold=2.0, size=48):
        a = tx(top, 26, DIM)
        b = tx(title, size, bold=True)
        g = VGroup(a, b).arrange(DOWN, buff=0.3)
        self.add(g)
        self.wait(hold)
        self.remove(g)

    def clear_all(self, keep=(), run_time=0.6):
        mobs = [m for m in self.mobjects if m not in keep]
        if mobs:
            self.play(*[FadeOut(m) for m in mobs], run_time=run_time)
        self.cap = None
        self.hdr = None if self.hdr not in keep else self.hdr

    def pad_to(self, t):
        """wait until scene time t (clips are cut to exactly this length)"""
        left = t - self.now
        if left > 1e-3:
            self.wait(left)

    def counter(self, value_tracker, fmt_fn, size=30, color=WHITE, pos=ORIGIN, anchor=None):
        """text that re-renders when the tracker changes (DecimalNumber needs LaTeX)"""
        state = {"v": None}

        def build():
            m = tx(fmt_fn(value_tracker.get_value()), size, color)
            if anchor is not None:
                m.next_to(anchor[0], anchor[1], buff=anchor[2] if len(anchor) > 2 else 0.2)
            else:
                m.move_to(pos)
            return m

        m = build()

        def upd(mob):
            v = fmt_fn(value_tracker.get_value())
            if v != state["v"]:
                state["v"] = v
                mob.become(build())

        m.add_updater(upd)
        return m


def arrow(a, b, buff=0.1, color=WHITE, sw=3, tip=0.18):
    return Arrow(a, b, buff=buff, color=color, stroke_width=sw, max_tip_length_to_length_ratio=0.5,
                 tip_length=tip)


def rbox(w, h, color=WHITE, fill=BOX, sw=2, r=0.12, op=1.0):
    return RoundedRectangle(width=w, height=h, corner_radius=r).set_stroke(color, sw).set_fill(fill, op)


def legend_levels(levels=((1, 1), (2, 2), (4, 4), (8, 8)), size=20, sw=0.24, direction=RIGHT, buff=0.35):
    items = []
    for (eh, ew) in levels:
        sq = RoundedRectangle(width=sw * max(1, ew / eh) ** 0.5, height=sw * max(1, eh / ew) ** 0.5,
                              corner_radius=0.04).set_fill(level_hex(eh, ew), 1).set_stroke(width=0)
        t = tx(f"{eh}×{ew}", size, DIM)
        items.append(VGroup(sq, t).arrange(RIGHT, buff=0.1))
    return VGroup(*items).arrange(direction, buff=buff)


def brand(size=14):
    """small corner tag used on every thread clip (bottom-right, below the caption band)"""
    b = tx("Level-of-Token Diffusion", size, DIM, italic=True)
    return b.move_to([7.05 - b.width / 2, -3.87, 0])
