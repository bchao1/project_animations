from manim import *
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = "/home/brianchc/project_animations/textured_gaussians"
DATA = os.path.join(SRC, "data")
GEN = os.path.join(DATA, "generated")

config.background_color = BLACK

# ------------------------------------------------------------------ style
FONT = "CMU Serif"
MATHFONT = "Latin Modern Math"          # CMU Serif lacks × ⊙ − → ≈ ∑ ∏ 𝒢 …
DIM = "#8A8A8A"
GS = "#90CAF9"        # 3DGS* / plain Gaussians
OURS = "#FFB74D"      # Textured Gaussians (RGBA)
C_A = "#87CEEB"       # alpha-only (paper's colour)
C_RGB = "#32CD32"     # RGB (paper's colour)
C_RGBA = "#FFA07A"    # RGBA (paper's colour)
RAY = "#FFEB3B"
R1C = "#FF5252"
R2C = "#69F0AE"
NC = "#448AFF"
BOX = "#1E2A33"

HEADER_Y = 3.62
CAPTION_Y = -3.5
MATHCH = set("×⊙→≈∼−∈≪≤‖·∑∏𝒢ℒ⊤′∞½⋯")


def mathify(s):
    return "".join(
        f'<span font_family="{MATHFONT}">{c}</span>' if c in MATHCH else c for c in s)


def sub(x):
    return f'<span rise="-3000"><small><small>{x}</small></small></span>'


def sup(x):
    return f'<span rise="4500"><small><small>{x}</small></small></span>'


def col(x, c):
    return f'<span foreground="{c}">{x}</span>'


def tx(s, size=30, color=WHITE, bold=False, italic=False):
    if bold:
        s = f"<b>{s}</b>"
    if italic:
        s = f"<i>{s}</i>"
    s = re.sub(r"&(?![a-zA-Z#][a-zA-Z0-9]*;)", "&amp;", s)
    return MarkupText(mathify(s), font=FONT, font_size=size, color=color)


# common symbols (bold = vectors, as in the paper)
def B(s):
    return f"<b>{s}</b>"


I_ = "<i>i</i>"
J_ = "<i>j</i>"
P_ = B("p")
X_ = B("x")
MU = B("<i>μ</i>")
ALPHA = "<i>α</i>"
SIG = "Σ"
G_ = "𝒢"
CI = B("c") + sub(I_)
CBASE = B("c") + sub(I_) + sup("base")
CTEX = B("c") + sub(I_) + sup("tex")
ATEX = ALPHA + sub(I_) + sup("tex")
UV = "(<i>u</i>, <i>v</i>)"


# ------------------------------------------------------------------ numpy splat renderer
def bilinear(tex, s, t):
    """tex (T,T,C); s = column coord, t = row coord in texel units (0..T-1)."""
    T = tex.shape[0]
    s = np.clip(s, 0, T - 1)
    t = np.clip(t, 0, T - 1)
    s0 = np.floor(s).astype(int).clip(0, T - 2)
    t0 = np.floor(t).astype(int).clip(0, T - 2)
    fs = (s - s0)[..., None]
    ft = (t - t0)[..., None]
    a = tex[t0, s0]
    b = tex[t0, s0 + 1]
    c = tex[t0 + 1, s0]
    d = tex[t0 + 1, s0 + 1]
    return (1 - ft) * ((1 - fs) * a + fs * b) + ft * ((1 - fs) * c + fs * d)


def splat_layers(res, E, s1, s2, th, base, o=1.0, a_tex=None, c_tex=None, m=3.0, cx=0.0, cy=0.0):
    """Evaluate one textured Gaussian on a res x res grid covering [-E, E]^2.
    Returns (alpha (H,W), colour (H,W,3)).  UV follows the paper: local coords in units of
    sigma, texture spans +-m sigma, bilinear lookup; outside +-m sigma alpha = 0."""
    g = (np.arange(res) + 0.5) / res * 2 * E - E
    X, Y = np.meshgrid(g, -g)
    dx, dy = X - cx, Y - cy
    c, s = np.cos(th), np.sin(th)
    u = (c * dx + s * dy) / s1
    v = (-s * dx + c * dy) / s2
    G = np.exp(-0.5 * (u * u + v * v))
    inside = (np.abs(u) <= m) & (np.abs(v) <= m)
    colr = np.ones((res, res, 3)) * np.asarray(base)[None, None, :]
    at = np.ones((res, res))
    if a_tex is not None or c_tex is not None:
        T = (a_tex if a_tex is not None else c_tex).shape[0]
        sc = (u / m + 1) / 2 * (T - 1)
        tr = (1 - (v / m + 1) / 2) * (T - 1)      # texture row 0 at +v (top)
        if a_tex is not None:
            at = bilinear(a_tex[..., None], sc, tr)[..., 0]
        if c_tex is not None:
            colr = colr + bilinear(c_tex, sc, tr)
    alpha = np.clip(o * G * at * inside, 0, 0.99)
    return alpha, np.clip(colr, 0, 1)


def rgba_arr(alpha, colr):
    a = np.dstack([colr, alpha[..., None]])
    return (np.clip(a, 0, 1) * 255).astype(np.uint8)


def over_black(alpha, colr):
    return (np.clip(colr * alpha[..., None], 0, 1) * 255).astype(np.uint8)


def composite(layers, bg=(0, 0, 0)):
    """layers front-to-back: list of (alpha, colour). Eq. 3 / Eq. 9."""
    Tm = np.ones(layers[0][0].shape)
    out = np.zeros(layers[0][1].shape)
    for a, c in layers:
        out += (a * Tm)[..., None] * c
        Tm = Tm * (1 - a)
    out += Tm[..., None] * np.asarray(bg)[None, None, :]
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def arr_img(a, h, nearest=False):
    m = ImageMobject(a)
    if nearest:
        m.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
    m.set(height=h)
    return m


def img(path, h):
    p = path if os.path.isabs(path) else os.path.join(DATA, path)
    m = ImageMobject(p)
    m.set(height=h)
    return m


def gen(name):
    return os.path.join(GEN, name)


def framed(m, color=WHITE, sw=1.5):
    return SurroundingRectangle(m, buff=0, color=color, stroke_width=sw)


def tex_tile(tex, h, alpha=False):
    """Show a texture map with visible texels (nearest)."""
    if alpha:
        a = (np.clip(tex, 0, 1) * 255).astype(np.uint8)
        a = np.dstack([a, a, a])
    else:
        a = (np.clip(tex, 0, 1) * 255).astype(np.uint8)
    T = a.shape[0]
    big = np.asarray(Image.fromarray(a).resize((T * 16, T * 16), Image.NEAREST))
    return arr_img(big, h)


# hand-made textures (deterministic)
def smooth_noise(T, sigma, seed):
    r = np.random.default_rng(seed).standard_normal((T, T))
    r = gaussian_filter(r, sigma, mode="wrap")
    r = (r - r.min()) / (r.max() - r.min())
    return r


def star_mask(T, k=5, r0=0.42, r1=0.95):
    g = (np.arange(T) + 0.5) / T * 2 - 1
    X, Y = np.meshgrid(g, -g)
    ang = np.arctan2(Y, X) + np.pi / 2
    rad = np.sqrt(X ** 2 + Y ** 2)
    f = (np.cos(k * ang) + 1) / 2
    edge = (r0 + (r1 - r0) * f ** 2) * 0.62
    return np.clip((edge - rad) / 0.04 + 0.5, 0, 1)


def crescent_mask(T):
    g = (np.arange(T) + 0.5) / T * 2 - 1
    X, Y = np.meshgrid(g, -g)
    a = (X ** 2 + Y ** 2) < 0.55 ** 2
    b = ((X - 0.28) ** 2 + (Y - 0.12) ** 2) < 0.45 ** 2
    return gaussian_filter((a & ~b).astype(float), 0.6)


def painting_tex(T):
    im = Image.open(gen("toy_target_hi.png")).convert("RGB").crop((190, 40, 340, 190)).resize((T, T), Image.BOX)
    return np.asarray(im) / 255.0


def load_toy():
    return json.load(open(gen("toy_results.json")))


# ------------------------------------------------------------------ scene base
class Base(Scene):
    def setup(self):
        self.cap = None
        self.hdr = None

    def header(self, s):
        if self.hdr is not None:
            self.remove(self.hdr)
        self.hdr = tx(s, 24, DIM).to_corner(UL, buff=0.35).set_y(HEADER_Y)
        self.add(self.hdr)

    # Captions stay up long enough to read: ~15 characters per second + 1 s.
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
        s = re.sub(r"(\d) × (\d)", "\\1\u00a0×\u00a0\\2", s)   # never wrap inside “4 × 4”
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
        self.cap_need = len(plain) / 15.0 + 1.0
        self.add(c)
        if wait:
            self.wait(wait)
        return c

    def uncaption(self):
        self._hold_caption()
        if self.cap is not None:
            self.remove(self.cap)
            self.cap = None

    def title_card(self, num, title, hold=1.8):
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
        if mobs:
            self.play(*[FadeOut(m) for m in mobs], run_time=run_time)


def arrow(a, b, buff=0.1, color=WHITE, sw=3):
    return Arrow(a, b, buff=buff, color=color, stroke_width=sw,
                 max_tip_length_to_length_ratio=0.18, tip_length=0.16)


def labeled(m, s, size=22, color=WHITE, d=UP, buff=0.15):
    return tx(s, size, color).next_to(m, d, buff=buff)


def fit_ellipses(params, center, h, res=128, color=WHITE, sw=1.2, k=2.0):
    """Outlines (k sigma) of fitted Gaussians drawn on top of a toy image of height h."""
    sc = h / res
    g = VGroup()
    for mx, my, s1, s2, th in params:
        e = Ellipse(width=2 * k * s1 * sc, height=2 * k * s2 * sc, stroke_color=color, stroke_width=sw)
        e.rotate(-th)
        e.move_to(center + np.array([(mx - res / 2) * sc, (res / 2 - my) * sc, 0]))
        g.add(e)
    return g

