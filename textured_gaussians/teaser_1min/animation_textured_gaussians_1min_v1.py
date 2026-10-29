"""Textured Gaussians -- 1-minute 3b1b-style teaser, v1.

Condensed from ../animation_textured_gaussians_v3.py (helpers copied, not imported).
Source: Chao et al., "Textured Gaussians for Enhanced 3D Scene Appearance Modeling", CVPR 2025, arXiv 2411.18625.
Paper figure crops and our 2D toy fit are read from ../data (absolute path, see SRC).
Render: ./render.sh l|h animation_textured_gaussians_1min_v1.py   Scenes: T1_Hook, T2_Method, T3_Payoff.
"""
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



# ------------------------------------------------------------------ 1-min teaser extras
def ellipse_overlay(params, res=384, fit_res=128, k=2.0):
    """RGBA image: white k-sigma outlines of the fitted Gaussians, clipped to the image."""
    sc = res / fit_res
    g = (np.arange(res) + 0.5) / sc
    X, Y = np.meshgrid(g, g)
    out = np.zeros((res, res), bool)
    for mx, my, s1, s2, th in params:
        c, s_ = np.cos(th), np.sin(th)
        dx, dy = X - mx, Y - my
        u = (c * dx + s_ * dy) / s1
        v = (-s_ * dx + c * dy) / s2
        r = np.sqrt(u * u + v * v)
        out |= np.abs(r - k) * min(s1, s2) * sc < 0.55
    a = np.zeros((res, res, 4), np.uint8)
    a[out] = (255, 255, 255, 200)
    return a


def painting_tex_T(T):
    im = Image.open(gen("toy_target_hi.png")).convert("RGB").crop((190, 40, 340, 190)).resize((T, T), Image.BOX)
    return np.asarray(im) / 255.0


# ================================================================== T1 Hook + key insight
class T1_Hook(Base):
    def construct(self):
        # ---- hook: same Gaussian budget, very different detail (paper Fig. 6, custom capture)
        title = tx("Textured Gaussians", 60, bold=True).move_to([0, 3.05, 0])
        self.add(title)
        a = img("fig6_art2_3dgs.png", 3.3).move_to([-3.5, 0.05, 0])
        b = img("fig6_art2_ours.png", 3.3).move_to([3.5, 0.05, 0])
        self.play(FadeIn(a), FadeIn(b), run_time=0.8)
        self.add(labeled(a, "3DGS*", 28, GS), labeled(b, "Textured Gaussians", 28, OURS))
        self.caption("About 100k Gaussians each. One is blurry.")
        self.caption("Why? Look at a single Gaussian.")
        self.uncaption()
        self.play(*[FadeOut(m) for m in self.mobjects if m is not title], run_time=0.5)
        self.remove(title)

        # ---- key insight: one color + ellipse  vs  RGBA-textured primitive
        self.header("The idea")
        T = 32
        E, H = 1.9, 3.4
        s1, s2, th = 0.62, 0.52, 0.3
        base = np.array([1.0, 0.6, 0.3])
        CL = np.array([-4.4, 0.45, 0])
        CR = np.array([1.1, 0.45, 0])
        a0, c0 = splat_layers(240, E, s1, s2, th, base)
        left = arr_img(over_black(a0, c0), H).move_to(CL)
        k = H / (2 * E)
        ell = DashedVMobject(Ellipse(width=4 * s1 * k, height=4 * s2 * k), num_dashes=36
                             ).set_stroke(WHITE, 1.6).rotate(th).move_to(CL)
        self.play(FadeIn(left), run_time=0.6)
        self.play(Create(ell), run_time=0.6)
        self.add(tx("3DGS: one color, one ellipse", 26, GS).next_to(left, DOWN, buff=0.1))
        self.caption("A plain Gaussian: one color, an elliptical footprint.")

        star = star_mask(T)
        paint = painting_tex_T(T)
        ta = ValueTracker(0.0)      # alpha texture: 1 -> star
        tc = ValueTracker(0.0)      # colour texture: 0 -> painting

        def a_tex():
            return (1 - ta.get_value()) + ta.get_value() * star

        def c_tex():
            return tc.get_value() * (paint - base[None, None, :])

        def spl():
            a_, c_ = splat_layers(240, E, s1, s2, th, base, 1.0, a_tex(), c_tex())
            return arr_img(over_black(a_, c_), H).move_to(CR)

        TX, TY1, TY2, TS = 5.15, 1.3, -0.5, 1.3

        def rgb_tile():
            return tex_tile(np.clip(base[None, None, :] + c_tex(), 0, 1), TS).move_to([TX, TY1, 0])

        def a_tile():
            return tex_tile(a_tex(), TS, alpha=True).move_to([TX, TY2, 0])

        right = always_redraw(spl)
        t1 = always_redraw(rgb_tile)
        t2 = always_redraw(a_tile)
        f1 = Square(TS, stroke_color=DIM, stroke_width=1).move_to([TX, TY1, 0])
        f2 = Square(TS, stroke_color=DIM, stroke_width=1).move_to([TX, TY2, 0])
        mid = arrow(CL + RIGHT * 1.85, CR + LEFT * 1.85, buff=0.05, color=DIM)
        self.play(GrowArrow(mid), FadeIn(right), run_time=0.6)
        self.play(FadeIn(t1), FadeIn(t2), Create(f1), Create(f2), run_time=0.5)
        ar1 = arrow([TX - TS / 2, TY1, 0], CR + RIGHT * 1.45 + UP * 0.45, buff=0.08, color=DIM)
        ar2 = arrow([TX - TS / 2, TY2, 0], CR + RIGHT * 1.45 + DOWN * 0.3, buff=0.08, color=DIM)
        self.add(ar1, ar2,
                 tx("RGB", 20, DIM).next_to(f1, RIGHT, buff=0.12),
                 tx("α", 22, DIM).next_to(f2, RIGHT, buff=0.12),
                 tx("per-Gaussian texture", 20, DIM).next_to(f1, UP, buff=0.12),
                 tx("Textured Gaussian", 26, OURS).next_to(CR + DOWN * H / 2, DOWN, buff=0.1))
        self.caption("Give every Gaussian its own small RGBA texture.")
        self.play(ta.animate.set_value(1.0), run_time=1.6)
        self.caption("Alpha reshapes it; RGB paints detail inside.")
        self.play(tc.animate.set_value(1.0), run_time=1.6)
        right.clear_updaters()
        t1.clear_updaters()
        t2.clear_updaters()
        self.wait(0.6)
        self.clear_all(run_time=0.5)


# ================================================================== T2 Method + 2D proof
class T2_Method(Base):
    def construct(self):
        self.header("How it works")
        T = 16
        base = np.array([0.55, 0.35, 0.25])
        ct = painting_tex(T)
        ct = ct - ct.mean((0, 1))
        at = star_mask(T, k=4, r0=0.6, r1=1.0) * 0.6 + 0.4 * smooth_noise(T, 1.4, 7)
        E, h = 1.8, 1.6
        s1, s2, th = 0.62, 0.5, 0.35
        eq = tx(f"{ALPHA}" + sub(I_) + f"({P_}) = {ATEX}{UV} · {G_}" + sub(I_) + f"({X_}) · <i>o</i>" + sub(I_), 34
                ).move_to([0, 2.65, 0])
        xs = [-5.6, -3.4, -1.2]
        y1, y2 = 0.95, -1.35
        cbase_t = arr_img((np.ones((8, 8, 3)) * base * 255).astype(np.uint8), h, True).move_to([xs[0], y1, 0])
        ctex_t = tex_tile(np.clip(ct + 0.5, 0, 1), h).move_to([xs[1], y1, 0])
        csum_t = tex_tile(np.clip(ct + base, 0, 1), h).move_to([xs[2], y1, 0])
        self.play(FadeIn(cbase_t), FadeIn(ctex_t), FadeIn(csum_t), run_time=0.5)
        self.add(tx("+", 34).move_to([(xs[0] + xs[1]) / 2, y1, 0]), tx("=", 34).move_to([(xs[1] + xs[2]) / 2, y1, 0]),
                 labeled(cbase_t, "base color", 20, DIM, DOWN, 0.1), labeled(ctex_t, "texture color", 20, DIM, DOWN, 0.1),
                 labeled(csum_t, "pixel color", 20, DIM, DOWN, 0.1))
        self.caption("Each ray hits the Gaussian and reads its texture.")
        g_only = splat_layers(200, E, s1, s2, th, base)[0]
        atex_t = tex_tile(at, h, alpha=True).move_to([xs[0], y2, 0])
        g_t = arr_img((np.dstack([g_only] * 3) * 255).astype(np.uint8), h).move_to([xs[1], y2, 0])
        a_full, c_full = splat_layers(200, E, s1, s2, th, base, 1.0, at, ct)
        a_t = arr_img((np.dstack([a_full] * 3) * 255).astype(np.uint8), h).move_to([xs[2], y2, 0])
        self.play(FadeIn(atex_t), FadeIn(g_t), FadeIn(a_t), run_time=0.5)
        self.add(eq, tx("×", 34).move_to([(xs[0] + xs[1]) / 2, y2, 0]), tx("=", 34).move_to([(xs[1] + xs[2]) / 2, y2, 0]),
                 labeled(atex_t, "texture alpha", 20, DIM, DOWN, 0.1),
                 labeled(g_t, "Gaussian falloff", 20, DIM, DOWN, 0.1),
                 labeled(a_t, "pixel opacity", 20, DIM, DOWN, 0.1))
        self.caption("Color adds; alpha multiplies the falloff.")
        fin = arr_img(over_black(a_full, c_full), 3.0).move_to([3.9, -0.2, 0])
        a1 = arrow([xs[2] + h / 2 + 0.1, y1, 0], fin.get_left() + UP * 0.55)
        a2 = arrow([xs[2] + h / 2 + 0.1, y2, 0], fin.get_left() + DOWN * 0.55)
        self.play(GrowArrow(a1), GrowArrow(a2), FadeIn(fin), run_time=0.6)
        self.add(labeled(fin, "one textured Gaussian", 24, OURS, DOWN, 0.1))
        self.wait(0.8)
        self.clear_all(run_time=0.5)

        # ---- our 2D toy: 64 Gaussians, plain vs RGBA textures, same start
        self.header("Our 2D toy: 64 Gaussians fit a photo")
        R = load_toy()
        hh = 3.1
        xs2 = [-4.3, 0.0, 4.3]
        y = 0.35
        tgt = img(gen("toy_target_hi.png"), hh).move_to([xs2[0], y, 0])
        modes = [("none", "plain", GS), ("rgba", "RGBA textured", OURS)]
        S2 = {md: R["stage2"][md] for md, _, _ in modes}
        fr = {md: [img(gen(f["file"]), hh).move_to([xs2[i + 1], y, 0]) for f in S2[md]]
              for i, (md, _, _) in enumerate(modes)}
        self.play(FadeIn(tgt), FadeIn(fr["none"][0]), FadeIn(fr["rgba"][0]), run_time=0.5)
        self.add(labeled(tgt, "target", 26))
        for md, name, cc in modes:
            self.add(labeled(fr[md][0], name, 26, cc))
        k2 = ValueTracker(0)

        def mk(md):
            return lambda: fr[md][int(round(k2.get_value()))]

        def mkl(md, cc):
            def f():
                e = S2[md][int(round(k2.get_value()))]
                return tx(f"PSNR {e['psnr']:.1f} dB", 28, cc).next_to(fr[md][0], DOWN, buff=0.15)
            return f

        dyn = []
        for md, _, cc in modes:
            dyn += [always_redraw(mk(md)), always_redraw(mkl(md, cc))]
        self.remove(fr["none"][0], fr["rgba"][0])
        self.add(*dyn)
        self.caption("Same 64 Gaussians, same starting fit.")
        self.play(k2.animate.set_value(len(S2["none"]) - 1), run_time=3.0, rate_func=linear)
        for d in dyn:
            d.clear_updaters()
        fin = {md: S2[md][-1]["psnr"] for md, _, _ in modes}
        ov_n = arr_img(ellipse_overlay(np.load(gen("toy_s2_none_params.npy"))), hh).move_to(fr["none"][0])
        ov_r = arr_img(ellipse_overlay(np.load(gen("toy_s2_rgba_params.npy"))), hh).move_to(fr["rgba"][0])
        self.caption(f"Still 64 Gaussians: {fin['none']:.1f} dB vs {fin['rgba']:.1f} dB.")
        self.play(FadeIn(ov_n), FadeIn(ov_r), run_time=0.5)
        self.wait(1.0)
        self.play(FadeOut(ov_n), FadeOut(ov_r), run_time=0.4)
        self.clear_all(run_time=0.5)


# ================================================================== T3 Payoff + end card
class T3_Payoff(Base):
    def construct(self):
        self.header("Paper results · equal Gaussian count")
        names = ["Blender", "Mip-NeRF 360", "DTU", "Tanks &|Temples", "Deep|Blending"]
        g1 = [(26.89, 28.11), (22.37, 23.73), (30.88, 32.43), (19.90, 21.10), (23.97, 24.83)]   # Table 1, 1 %
        ax = Axes(x_range=[0, 5, 1], y_range=[0, 1.8, 0.5], x_length=9.0, y_length=3.7, tips=False,
                  axis_config={"color": DIM, "stroke_width": 2}).move_to([0.2, 0.25, 0])
        self.play(Create(ax), run_time=0.4)
        self.add(tx("PSNR gain over 3DGS* with 1% of the Gaussians (dB)", 24, DIM).next_to(ax, UP, buff=0.2),
                 *[tx(f"{v:.1f}", 18, DIM).next_to(ax.c2p(0, v), LEFT, buff=0.1) for v in (0.5, 1.0, 1.5)])
        bars, labs = VGroup(), VGroup()
        for i, (a_, b_) in enumerate(g1):
            d = b_ - a_
            r = Rectangle(width=0.8, height=ax.c2p(0, d)[1] - ax.c2p(0, 0)[1], stroke_width=0,
                          fill_color=OURS, fill_opacity=0.9).move_to(ax.c2p(i + 0.5, 0), aligned_edge=DOWN)
            bars.add(r)
            labs.add(tx(f"+{d:.2f}", 24, OURS).next_to(r, UP, buff=0.08))
            nm = VGroup(*[tx(w, 22) for w in names[i].split("|")]).arrange(DOWN, buff=0.06)
            self.add(nm.next_to(ax.c2p(i + 0.5, 0), DOWN, buff=0.15))
        self.play(LaggedStart(*[GrowFromEdge(b, DOWN) for b in bars], lag_ratio=0.12), run_time=1.2)
        self.add(labs)
        self.caption("1% of the Gaussians: +0.86 to +1.55 dB PSNR.")
        self.caption("The fewer the Gaussians, the bigger the gain.")
        self.clear_all(run_time=0.5)

        # ---- qualitative (paper Fig. 5)
        rows_ = [("drjohnson", "Deep Blending"), ("dtu_buddha", "DTU")]
        ims = []
        for r_, (scn, dsn) in enumerate(rows_):
            y = 1.4 - r_ * 2.62
            a = img(f"fig5_{scn}_3dgs.png", 2.4).move_to([-3.3, y, 0])
            b = img(f"fig5_{scn}_ours.png", 2.4).move_to([3.3, y, 0])
            ims += [a, b]
            self.add(tx(dsn, 20, DIM).move_to([0, y, 0]))
        self.play(*[FadeIn(m) for m in ims], run_time=0.5)
        self.add(tx("3DGS*", 26, GS).next_to(ims[0], UP, buff=0.08),
                 tx("Textured Gaussians", 26, OURS).next_to(ims[1], UP, buff=0.08))
        self.caption("1% budget: ~39k Gaussians per scene, ~2.5k per object.")
        self.wait(0.5)
        self.clear_all(run_time=0.5)

        # ---- end card
        self.remove(self.hdr)
        self.hdr = None
        t = tx("Textured Gaussians", 60, bold=True).move_to([0, 2.1, 0])
        st = tx("for Enhanced 3D Scene Appearance Modeling", 34).move_to([0, 1.35, 0])
        v = tx("CVPR 2025", 28, DIM).move_to([0, 0.6, 0])
        u = tx("arXiv 2411.18625   ·   textured-gaussians.github.io", 30, OURS).move_to([0, -0.3, 0])
        a1 = tx("Brian Chao, Hung-Yu Tseng, Lorenzo Porzi, Chen Gao, Tuotuo Li, Qinbo Li,", 21, DIM)
        a2 = tx("Ayush Saraf, Jia-Bin Huang, Johannes Kopf, Gordon Wetzstein, Changil Kim", 21, DIM)
        a3 = tx("Stanford University  ·  Meta  ·  University of Maryland, College Park", 21, DIM)
        au = VGroup(a1, a2, a3).arrange(DOWN, buff=0.12).move_to([0, -1.9, 0])
        self.add(t, st, v, u, au)
        self.wait(5.0)
