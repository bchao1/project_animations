"""Textured Gaussians -- 3b1b-style teaser, v2.

Source: Chao, Tseng, Porzi, Gao, Li, Li, Saraf, Huang, Kopf, Wetzstein, Kim,
"Textured Gaussians for Enhanced 3D Scene Appearance Modeling", CVPR 2025, arXiv 2411.18625.
Paper figures in data/ are crops of the rasters in the arXiv LaTeX source (build_assets.py).
data/generated/ holds our own 2D toy experiment (build_assets.py); every toy number comes
from data/generated/toy_results.json. Every paper number is from the paper's tables.

Render (see render.sh):
    manim -ql --media_dir media animation_textured_gaussians_v2.py <Scene>
Scenes, in order: S00_Title ... S10_Outlook.
"""
from manim import *
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
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


# ================================================================== S00 Title
class S00_Title(Base):
    def construct(self):
        title = tx("Textured Gaussians", 66, bold=True).move_to([0, 3.0, 0])
        subt = tx("for Enhanced 3D Scene Appearance Modeling", 34).move_to([0, 2.3, 0])
        a1 = tx("Brian Chao, Hung-Yu Tseng, Lorenzo Porzi, Chen Gao, Tuotuo Li, Qinbo Li,", 21, DIM)
        a2 = tx("Ayush Saraf, Jia-Bin Huang, Johannes Kopf, Gordon Wetzstein, Changil Kim", 21, DIM)
        a3 = tx("Stanford University  ·  Meta  ·  University of Maryland, College Park  ·  CVPR 2025", 21, DIM)
        au = VGroup(a1, a2, a3).arrange(DOWN, buff=0.12).move_to([0, -2.95, 0])
        self.add(title, subt, au)
        plain = img(gen("toy_final_none.png"), 3.3).move_to([-1.9, -0.25, 0])
        texd = img(gen("toy_final_rgba.png"), 3.3).move_to([1.9, -0.25, 0])
        self.play(FadeIn(plain), run_time=1.0)
        lp = tx("64 Gaussians", 22, GS).next_to(plain, DOWN, buff=0.12)
        self.add(lp)
        self.wait(0.8)
        self.play(FadeIn(texd), run_time=1.0)
        lt = tx("64 textured Gaussians", 22, OURS).next_to(texd, DOWN, buff=0.12)
        self.add(lt)
        self.wait(3.2)
        self.play(FadeOut(Group(plain, texd)), run_time=0.7)
        self.remove(title, subt, au, lp, lt)
        self.wait(0.3)


# ================================================================== S01 Hook
class S01_Hook(Base):
    def construct(self):
        self.header("Textured Gaussians · CVPR 2025")
        tz = img("fig1_teaser.png", 3.55).move_to([0, 0.55, 0])
        if tz.width > 13.4:
            tz.set(width=13.4)
        self.play(FadeIn(tz), run_time=0.8)
        self.caption("3D Gaussian Splatting (3DGS) builds scenes out of Gaussian blobs.")
        self.caption("Same number of Gaussians: 3DGS blurs fine texture; Textured Gaussians keep it.")
        self.play(FadeOut(tz), run_time=0.5)
        a = img("fig6_art1_3dgs.png", 3.4).move_to([-3.45, 0.35, 0])
        b = img("fig6_art1_ours.png", 3.4).move_to([3.45, 0.35, 0])
        self.play(FadeIn(a), FadeIn(b), run_time=0.6)
        self.add(labeled(a, "3DGS*", 26, GS), labeled(b, "Textured Gaussians (RGBA)", 26, OURS))
        self.caption("Custom capture, about 100k Gaussians for both models.", 0.8)
        self.caption("The trick: give every Gaussian its own texture map.", 0.8)
        self.clear_all()


# ================================================================== S02 Splatting recap
class S02_Splatting(Base):
    def construct(self):
        self.title_card("Part 1", "Gaussian splatting in one minute")
        self.header("1 · Gaussian splatting recap")
        th = ValueTracker(0.5)
        s1 = ValueTracker(0.62)
        s2 = ValueTracker(0.30)
        E = 1.9
        H = 3.8
        C0 = np.array([-3.6, 0.15, 0])
        base = np.array([0.35, 0.65, 1.0])

        def g_img():
            a, c = splat_layers(200, E, s1.get_value(), s2.get_value(), th.get_value(), base)
            return arr_img(over_black(a, c), H).move_to(C0)

        def g_ell():
            k = H / (2 * E)
            e = Ellipse(width=2 * 2 * s1.get_value() * k, height=2 * 2 * s2.get_value() * k,
                        stroke_color=WHITE, stroke_width=1.5).rotate(th.get_value()).move_to(C0)
            d1 = np.array([np.cos(th.get_value()), np.sin(th.get_value()), 0])
            d2 = np.array([-np.sin(th.get_value()), np.cos(th.get_value()), 0])
            a1 = Arrow(C0, C0 + d1 * 2 * s1.get_value() * k, buff=0, color=R1C, stroke_width=3,
                       max_tip_length_to_length_ratio=0.15)
            a2 = Arrow(C0, C0 + d2 * 2 * s2.get_value() * k, buff=0, color=R2C, stroke_width=3,
                       max_tip_length_to_length_ratio=0.25)
            return VGroup(e, a1, a2)

        gi = always_redraw(g_img)
        ge = always_redraw(g_ell)
        frame = Square(H, stroke_color=DIM, stroke_width=1).move_to(C0)
        self.play(FadeIn(gi), Create(frame), run_time=0.8)
        self.add(ge)
        e1 = tx(f"{G_}({X_}) = exp(−½ ({X_} − {MU})" + sup("⊤") + f" {SIG}" + sup("−1") + f" ({X_} − {MU}))", 30)
        e2 = tx(f"{SIG} = <b>R</b> <b>S</b> <b>S</b>" + sup("⊤") + " <b>R</b>" + sup("⊤"), 30)
        e3 = tx(f"{SIG}′ = <b>J</b> <b>W</b> {SIG} <b>W</b>" + sup("⊤") + " <b>J</b>" + sup("⊤"), 30)
        e4 = tx(f"{SIG}" + sub("2D") + f" = upper-left 2 × 2 block of {SIG}′", 26)
        eqs = VGroup(e1, e2, e3, e4).arrange(DOWN, buff=0.42, aligned_edge=LEFT).move_to([2.95, 0.55, 0])
        self.add(e1)
        self.caption("Each primitive is a 3D Gaussian: center <b><i>μ</i></b>, covariance Σ.")
        self.add(e2)
        lr = tx("rotation <b>R</b>", 22, R1C).move_to([1.4, -1.55, 0])
        ls = tx("scales <b>S</b>", 22, R2C).next_to(lr, RIGHT, buff=0.5)
        self.add(lr, ls)
        self.caption("A rotation and three scales: an ellipsoid.")
        self.play(th.animate.set_value(1.6), run_time=1.2)
        self.play(s1.animate.set_value(0.4), s2.animate.set_value(0.55), run_time=1.0)
        self.play(th.animate.set_value(2.3), s1.animate.set_value(0.7), s2.animate.set_value(0.22), run_time=1.0)
        self.add(e3, e4)
        self.caption("Seen by a camera, it projects to a 2D elliptical splat.", 0.5)
        self.remove(lr, ls)
        self.play(FadeOut(gi), FadeOut(ge), FadeOut(frame), run_time=0.5)
        self.remove(*eqs)
        gi.clear_updaters()
        ge.clear_updaters()

        # ---------- front-to-back compositing
        E2 = 2.0
        H2 = 4.2
        C1 = np.array([-4.3, 0.25, 0])
        sp = [  # (cx, cy, s1, s2, th, colour, opacity) front-to-back
            (0.15, 0.25, 0.55, 0.28, 0.4, (1.0, 0.45, 0.35), 0.85),
            (-0.35, -0.05, 0.62, 0.35, -0.6, (0.35, 0.8, 0.45), 0.9),
            (0.35, -0.25, 0.75, 0.45, 1.2, (0.35, 0.55, 1.0), 0.95),
            (-0.9, 0.9, 0.4, 0.25, 0.9, (1.0, 0.85, 0.3), 0.9),
            (1.0, -1.05, 0.5, 0.3, -0.3, (0.8, 0.45, 1.0), 0.9),
        ]
        layers = [splat_layers(260, E2, s1_, s2_, th_, col_, o_, cx=cx, cy=cy)
                  for (cx, cy, s1_, s2_, th_, col_, o_) in sp]
        canvas = arr_img(composite(layers), H2).move_to(C1)
        cf = Square(H2, stroke_color=DIM, stroke_width=1).move_to(C1)
        self.play(FadeIn(canvas), Create(cf), run_time=0.8)
        self.caption("An image is many splats, sorted by depth, blended front to back.")
        pw = np.array([0.05, 0.0])        # pixel p (world coords of the canvas)
        pdot = C1 + np.array([pw[0], pw[1], 0]) * H2 / (2 * E2)
        cross = VGroup(Circle(0.11, color=WHITE, stroke_width=3).move_to(pdot),
                       Dot(pdot, radius=0.03, color=WHITE))
        self.add(cross, tx(P_, 26).next_to(cross, UR, buff=0.02))
        # alpha of every splat at p
        al = []
        for (cx, cy, s1_, s2_, th_, col_, o_) in sp:
            dx, dy = pw[0] - cx, pw[1] - cy
            c, s = np.cos(th_), np.sin(th_)
            u = (c * dx + s * dy) / s1_
            v = (-s * dx + c * dy) / s2_
            al.append(min(0.99, o_ * np.exp(-0.5 * (u * u + v * v))))
        hit = [k for k in range(len(sp)) if al[k] > 0.03][:3]
        # side view of the ray
        y0 = 1.55
        x_cam, x_end = -0.9, 6.3
        cam = Polygon([x_cam - 0.35, y0 + 0.22, 0], [x_cam - 0.35, y0 - 0.22, 0], [x_cam, y0, 0],
                      color=WHITE, fill_opacity=0.2, stroke_width=2)
        ray = Line([x_cam, y0, 0], [x_end, y0, 0], color=RAY, stroke_width=2.5)
        self.play(Create(cam), Create(ray), run_time=0.7)
        self.add(tx("camera", 20, DIM).next_to(cam, DOWN, buff=0.15))
        xs = [1.0, 2.9, 4.8]
        ells, labs = VGroup(), VGroup()
        Tm = 1.0
        wts = []
        for n, k in enumerate(hit):
            colr = rgb_to_color(sp[k][5])
            e = Ellipse(width=0.32, height=1.7, stroke_color=colr, stroke_width=2,
                        fill_color=colr, fill_opacity=al[k]).move_to([xs[n], y0, 0])
            ells.add(e)
            labs.add(tx(f"{ALPHA}" + sub(str(n + 1)) + f" = {al[k]:.2f}", 22).next_to(e, UP, buff=0.12))
            wts.append(al[k] * Tm)
            Tm *= 1 - al[k]
        self.play(LaggedStart(*[FadeIn(e) for e in ells], lag_ratio=0.3), run_time=1.0)
        self.add(labs)
        eq = tx(f"{B('c')}({P_}) = ∑" + sub(I_) + f" {CI} {ALPHA}" + sub(I_) + " ∏" + sub(f"{J_}&lt;{I_}")
                + f" (1 − {ALPHA}" + sub(J_) + ")", 34).move_to([2.7, -1.35, 0])
        eqa = tx(f"{ALPHA}" + sub(I_) + f" = {G_}" + sub(I_) + sup("2D") + f"({P_}) · <i>o</i>" + sub(I_), 28
                 ).next_to(eq, DOWN, buff=0.3)
        self.add(eq, eqa)
        self.caption("Each adds its color, weighted by opacity and remaining transmittance.")
        # travelling transmittance
        s = ValueTracker(0.0)

        def acc(sv):
            T_, col_acc = 1.0, np.zeros(3)
            for n, k in enumerate(hit):
                if sv >= xs[n]:
                    col_acc += np.array(sp[k][5]) * al[k] * T_
                    T_ *= 1 - al[k]
            return T_, col_acc

        def meter():
            T_, ca = acc(s.get_value())
            bar_bg = Rectangle(width=5.0, height=0.22, stroke_color=DIM, stroke_width=1).move_to([2.6, 0.05, 0])
            bar = Rectangle(width=max(0.001, 5.0 * T_), height=0.22, stroke_width=0, fill_color=WHITE,
                            fill_opacity=0.8).align_to(bar_bg, LEFT).set_y(0.05)
            lab = tx(f"transmittance <i>T</i> = {T_:.2f}", 22).next_to(bar_bg, UP, buff=0.12)
            sw = Square(0.62, stroke_color=WHITE, stroke_width=1.5, fill_color=rgb_to_color(np.clip(ca, 0, 1)),
                        fill_opacity=1).move_to([-0.6, 0.05, 0])
            sl = tx(f"{B('c')}({P_})", 22).next_to(sw, DOWN, buff=0.1)
            dot = Dot([min(s.get_value(), x_end), y0, 0], radius=0.07, color=RAY)
            return VGroup(bar_bg, bar, lab, sw, sl, dot)

        mt = always_redraw(meter)
        self.add(mt)
        self.play(s.animate.set_value(x_end), run_time=3.5, rate_func=linear)
        mt.clear_updaters()
        terms = "  +  ".join(f"{w:.2f}" for w in wts)
        self.add(tx(f"weights: {terms}", 22, DIM).move_to([2.6, -0.55, 0]))
        self.caption("Everything is fit by gradient descent against the training photos.")
        self.clear_all()


# ================================================================== S03 Limitation
class S03_Limitation(Base):
    def construct(self):
        self.title_card("Part 2", "One color, one ellipse")
        self.header("2 · The limitation")
        E, H = 1.8, 3.2
        C0 = np.array([-4.0, 1.0, 0])
        base = np.array([1.0, 0.6, 0.3])
        a, c = splat_layers(220, E, 0.7, 0.4, 0.0, base)
        sp = arr_img(over_black(a, c), H).move_to(C0)
        self.play(FadeIn(sp), run_time=0.6)
        scan = DashedLine(C0 + LEFT * 1.6, C0 + RIGHT * 1.6, color=WHITE, stroke_width=2)
        self.play(Create(scan), run_time=0.5)
        ax = Axes(x_range=[-1.8, 1.8, 0.9], y_range=[0, 1.1, 0.5], x_length=3.2, y_length=1.5, tips=False,
                  axis_config={"color": DIM, "stroke_width": 2}).move_to([-4.0, -1.55, 0])
        ca = ax.plot(lambda x: 0.8, x_range=[-1.8, 1.8], color=rgb_to_color(base), stroke_width=4)
        aa = ax.plot(lambda x: np.exp(-0.5 * (x / 0.7) ** 2), x_range=[-1.8, 1.8], color=WHITE, stroke_width=3)
        self.play(Create(ax), run_time=0.5)
        self.play(Create(ca), Create(aa), run_time=1.0)
        self.add(tx("color: constant", 20, rgb_to_color(base)).next_to(ax, RIGHT, buff=0.15).shift(UP * 0.35),
                 tx("opacity: Gaussian falloff", 20).next_to(ax, RIGHT, buff=0.15).shift(DOWN * 0.25))
        p1 = tx("1.  Pixels covered by one Gaussian share one", 26)
        p1b = tx("color, up to the Gaussian falloff.", 26)
        p2 = tx("2.  Its opacity footprint is always an ellipse.", 26)
        pts = VGroup(p1, p1b, p2).arrange(DOWN, buff=0.18, aligned_edge=LEFT)
        pts.move_to([0, 1.65, 0]).to_edge(RIGHT, buff=0.5)
        p1b.shift(RIGHT * 0.45)
        p2.next_to(p1b, DOWN, buff=0.35).align_to(p1, LEFT)
        self.add(p1, p1b)
        self.caption("Across a splat the color never changes; only opacity fades.")
        self.add(p2)
        # ellipse vs square painting footprint
        sq = Square(1.7, stroke_color=WHITE, stroke_width=2, fill_color="#C62828", fill_opacity=0.9).move_to([3.3, -1.1, 0])
        el = Ellipse(width=2.0, height=2.0, stroke_color=GS, stroke_width=3).move_to(sq)
        self.play(FadeIn(sq), run_time=0.4)
        self.play(Create(el), run_time=0.7)
        self.add(tx("a sharp square corner cannot be one ellipse", 20, DIM).next_to(sq, DOWN, buff=0.2))
        self.caption("And a sharp corner is not an ellipse.", 0.3)
        self.clear_all()

        # --- many Gaussians needed: toy sweep
        R = load_toy()
        ns = [16, 64, 256, 1024]
        h = 2.35
        xs = [-5.4, -2.7, 0.0, 2.7, 5.4]
        tgt = img(gen("toy_target_hi.png"), h).move_to([xs[0], 0.35, 0])
        self.play(FadeIn(tgt), run_time=0.5)
        self.add(labeled(tgt, "target", 24))
        self.caption("Our own 2D test: fit a photo crop with <i>N</i> plain, alpha-blended Gaussians.")
        for k, n in enumerate(ns):
            info = R["plain_sweep"][str(n)]
            im = img(gen(info["file"]), h).move_to([xs[k + 1], 0.35, 0])
            self.play(FadeIn(im), run_time=0.4)
            self.add(labeled(im, f"<i>N</i> = {n}", 24, GS),
                     labeled(im, f"PSNR {info['psnr']:.1f} dB", 22, DIM, DOWN))
            self.wait(0.3)
        self.caption("Detail needs many more, ever smaller Gaussians.")
        self.caption("Real scenes: 3DGS* averages 4.4 million per Mip-NeRF 360 scene.", 0.5)
        self.clear_all()


# ================================================================== S04 Textures
class S04_Textures(Base):
    def construct(self):
        self.title_card("Part 3", "Give every Gaussian a texture")
        self.header("3 · Textured Gaussians")
        T = 16
        base = np.array([0.35, 0.62, 0.95])
        aN = 0.15 + 0.85 * smooth_noise(T, 1.6, 3)
        cP = painting_tex(T)
        cRes = cP - cP.mean((0, 1))           # texture colour = residual on top of the base colour
        E, H = 1.75, 2.75
        cols = [(None, None, "no texture (3DGS)", WHITE),
                (aN, None, "alpha only (<i>K</i> = 1)", C_A),
                (None, cRes, "RGB (<i>K</i> = 3)", C_RGB),
                (aN, cRes, "RGBA (<i>K</i> = 4)", C_RGBA)]
        xs = [-5.1, -1.7, 1.7, 5.1]
        eq = tx("texture map per Gaussian:  <i>𝒯</i> × <i>𝒯</i> × <i>K</i>,   <i>K</i> ∈ {1, 3, 4}", 30).move_to([0, 2.75, 0])
        self.add(eq)
        self.caption("Textured Gaussians attach a small texture map to every single Gaussian.")
        for k, (at, ct, name, cc) in enumerate(cols):
            rgb_t = np.clip(base[None, None, :] + (ct if ct is not None else 0), 0, 1) * np.ones((T, T, 3))
            a_t = at if at is not None else np.ones((T, T))
            t1 = tex_tile(rgb_t, 1.2).move_to([xs[k] - 0.66, 1.4, 0])
            t2 = tex_tile(a_t, 1.2, alpha=True).move_to([xs[k] + 0.66, 1.4, 0])
            f1, f2 = framed(t1, DIM, 1), framed(t2, DIM, 1)
            a, c = splat_layers(240, E, 0.66, 0.58, 0.3, base, 1.0, at, ct)
            r = arr_img(over_black(a, c), H).move_to([xs[k], -0.95, 0])
            self.play(FadeIn(t1), FadeIn(t2), Create(f1), Create(f2), FadeIn(r), run_time=0.5)
            self.add(tx("RGB", 18, DIM).next_to(t1, UP, buff=0.06), tx("α", 18, DIM).next_to(t2, UP, buff=0.06))
            self.add(tx(name, 24, cc).next_to(r, DOWN, buff=0.02))
            if k == 0:
                self.caption("Every Gaussian gets its own small texture map.")
            elif k == 1:
                self.caption("Alpha texture: opacity varies across the splat.", 0.2)
            elif k == 2:
                self.caption("RGB texture: high-frequency color.", 0.2)
            else:
                self.caption("RGBA: both, the most expressive (after the paper's Fig. 2).", 0.5)
        self.clear_all()

        # --- alpha texture changes the shape
        Tm = 32
        star = star_mask(Tm)
        cres = crescent_mask(Tm)
        t = ValueTracker(0.0)
        which = {"m": star}
        C0 = np.array([-3.3, 0.2, 0])
        E2, H2 = 1.9, 4.2
        base2 = np.array([1.0, 0.72, 0.3])

        def blend():
            return (1 - t.get_value()) + t.get_value() * which["m"]

        def spl():
            a, c = splat_layers(240, E2, 0.6, 0.45, 0.25, base2, 1.0, blend(), None)
            return arr_img(over_black(a, c), H2).move_to(C0)

        def tile():
            return tex_tile(blend(), 2.2, alpha=True).move_to([3.6, 0.55, 0])

        sp_m = always_redraw(spl)
        tl = always_redraw(tile)
        tf = Square(2.2, stroke_color=DIM, stroke_width=1).move_to([3.6, 0.55, 0])
        self.play(FadeIn(sp_m), FadeIn(tl), Create(tf), run_time=0.6)
        self.add(tx(f"alpha texture  {ATEX}{UV}", 24).next_to(tf, UP, buff=0.18))
        form = tx(f"{ALPHA}" + sub(I_) + f"({P_}) = {ATEX}{UV} · {G_}" + sub(I_) + f"({X_}) · <i>o</i>" + sub(I_), 30
                  ).move_to([3.3, -1.55, 0])
        self.add(form)
        self.caption("Opacity = texture alpha × Gaussian falloff × opacity.")
        self.play(t.animate.set_value(1.0), run_time=2.0)
        self.caption("So one Gaussian can take shapes that are not ellipses.")
        self.play(t.animate.set_value(0.0), run_time=0.8)
        which["m"] = cres
        self.play(t.animate.set_value(1.0), run_time=1.4)
        sp_m.clear_updaters()
        tl.clear_updaters()
        self.clear_all()


# ================================================================== S05 UV mapping
class S05_UVMapping(Base):
    def construct(self):
        self.title_card("Part 4", "Ray, intersection, texture lookup")
        self.header("4 · Ray–Gaussian intersection and UV mapping")
        # world: X = depth (away from camera), Y = lateral, Z = up.  Parallel "cabinet" projection.
        OFF = np.array([-1.75, -0.05, 0])
        SC = 1.3

        def P(w):
            w = np.asarray(w, float)
            return OFF + SC * np.array([w[0] + 0.5 * w[1], w[2] + 0.28 * w[1], 0])

        o = np.array([-3.45, -0.5, 0.0])
        Xp = -2.15                     # image plane depth
        mu = np.array([0.9, 0.35, 0.25])
        n = np.array([-0.9, 0.3, 0.2])
        n = n / np.linalg.norm(n)
        r1 = np.array([0.1, 1.0, 0.35])
        r1 = r1 - r1.dot(n) * n
        r1 /= np.linalg.norm(r1)
        r2 = np.cross(n, r1)
        sg1, sg2, m = 0.42, 0.3, 3.0
        T = 16
        # texture: flower-wallpaper crop (paper's scene) as RGB, soft leaf-like alpha
        fl = Image.open(os.path.join(DATA, "fig1_flower_full.png")).convert("RGB")
        W_, H_ = fl.size
        texc = np.asarray(fl.crop((int(W_ * 0.47), int(H_ * 0.3), int(W_ * 0.47) + 150, int(H_ * 0.3) + 150))
                          .resize((T, T), Image.BOX)) / 255.0
        texc = np.clip(texc * 1.35, 0, 1)
        g = (np.arange(T) + 0.5) / T * 2 - 1
        GX, GY = np.meshgrid(g, -g)
        texa = np.clip(1.25 - (GX ** 2 / 0.8 + GY ** 2 / 0.55), 0, 1)
        # splat image in texture space (u right, v up): colour = texture, alpha = a_tex * G
        res = 240
        uu = (np.arange(res) + 0.5) / res * 2 * m - m
        U, V = np.meshgrid(uu, -uu)
        Gv = np.exp(-0.5 * (U ** 2 + V ** 2))
        sc = (U / m + 1) / 2 * (T - 1)
        tr = (1 - (V / m + 1) / 2) * (T - 1)
        ctex = bilinear(texc, sc, tr)
        atex = bilinear(texa[..., None], sc, tr)[..., 0]
        rgba = rgba_arr(np.clip(atex * Gv, 0, 1), ctex)      # alpha = a_tex * G (o = 1)
        splat = ImageMobject(rgba)
        support = tex_tile(texc, 1.0)
        support.set_opacity(0.22)
        ul = mu - m * sg1 * r1 + m * sg2 * r2
        ur = mu + m * sg1 * r1 + m * sg2 * r2
        dl = mu - m * sg1 * r1 - m * sg2 * r2
        dr = mu + m * sg1 * r1 - m * sg2 * r2
        splat.points = np.array([P(ul), P(ur), P(dl), P(dr)])
        support.points = np.array([P(ul), P(ur), P(dl), P(dr)])
        border = Polygon(P(ul), P(ur), P(dr), P(dl), stroke_color=DIM, stroke_width=1.2)
        # camera + image plane
        cam = Dot(P(o), radius=0.07, color=WHITE)
        plane_c = [np.array([Xp, y, z]) for (y, z) in ((-1.0, 0.8), (1.0, 0.8), (1.0, -0.8), (-1.0, -0.8))]
        ip = Polygon(*[P(c) for c in plane_c], stroke_color=WHITE, stroke_width=2, fill_color="#263238", fill_opacity=0.35)
        self.play(FadeIn(cam), Create(ip), run_time=0.7)
        self.add(tx(B("o"), 24).next_to(cam, DOWN, buff=0.1), tx("image plane", 20, DIM).next_to(ip, UP, buff=0.1))
        self.play(FadeIn(support), Create(border), run_time=0.6)
        self.play(FadeIn(splat), run_time=0.6)
        ax1 = Arrow(P(mu), P(mu + 2.2 * sg1 * r1), buff=0, color=R1C, stroke_width=3, max_tip_length_to_length_ratio=0.2)
        ax2 = Arrow(P(mu), P(mu + 2.4 * sg2 * r2), buff=0, color=R2C, stroke_width=3, max_tip_length_to_length_ratio=0.25)
        axn = Arrow(P(mu), P(mu + 0.9 * n), buff=0, color=NC, stroke_width=3, max_tip_length_to_length_ratio=0.25)
        self.play(GrowArrow(ax1), GrowArrow(ax2), GrowArrow(axn), run_time=0.8)
        self.add(tx(B("r") + sup("1"), 22, R1C).next_to(ax1.get_end(), UR, buff=0.04),
                 tx(B("r") + sup("2"), 22, R2C).next_to(ax2.get_end(), UP, buff=0.06),
                 tx(B("n"), 22, NC).next_to(axn.get_end(), LEFT, buff=0.06),
                 tx(MU, 22).next_to(P(mu), DOWN + RIGHT * 0.3, buff=0.08))
        self.caption("The texture lies on the plane of the two longest axes, spanning ±3σ.")
        # texture panel
        PC = np.array([5.1, 0.85, 0])
        PS = 2.6
        panel = tex_tile(texc, PS).move_to(PC)
        pf = Square(PS, stroke_color=WHITE, stroke_width=1.5).move_to(PC)
        self.play(FadeIn(panel), Create(pf), run_time=0.6)
        self.add(tx(f"texture map, <i>𝒯</i> = {T}", 22).next_to(pf, UP, buff=0.12),
                 tx("<i>u</i> →", 20, R1C).next_to(pf, DOWN, buff=0.08).align_to(pf, LEFT),
                 tx("<i>v</i> ↑", 20, R2C).next_to(pf, LEFT, buff=0.08).align_to(pf, DOWN))
        # moving pixel: choose points on the Gaussian plane, find their pixel on the image plane
        k = ValueTracker(0.0)

        def local(kv):
            return np.array([1.6 * sg1 * np.cos(2 * np.pi * kv) + 0.3 * sg1,
                             1.5 * sg2 * np.sin(4 * np.pi * kv)])

        def geom(kv):
            a, b = local(kv)
            x = mu + a * r1 + b * r2
            d = x - o
            p = o + d * (Xp - o[0]) / d[0]
            return p, x, a, b

        def uv_of(a, b):
            u = (m * sg1 + a) / (2 * m * sg1) * (T - 1)
            v = (m * sg2 + b) / (2 * m * sg2) * (T - 1)
            return u, v

        def dyn():
            p, x, a, b = geom(k.get_value())
            u, v = uv_of(a, b)
            g_ = VGroup()
            g_.add(Line(P(o), P(x), color=RAY, stroke_width=2.5))
            g_.add(Dot(P(p), radius=0.06, color=WHITE))
            g_.add(Dot(P(x), radius=0.07, color=RAY))
            g_.add(tx(P_, 22).next_to(P(p), UL, buff=0.05))
            g_.add(tx(X_, 22, RAY).next_to(P(x), DR, buff=0.04))
            q = PC + np.array([(u / (T - 1) - 0.5) * PS * (T - 1) / T, (v / (T - 1) - 0.5) * PS * (T - 1) / T, 0])
            cell = PS / T
            i0, j0 = min(int(np.floor(u)), T - 2), min(int(np.floor(v)), T - 2)
            c00 = PC + np.array([(i0 + 0.5) / T * PS - PS / 2, (j0 + 0.5) / T * PS - PS / 2, 0])
            g_.add(Rectangle(width=2 * cell, height=2 * cell, stroke_color=WHITE, stroke_width=2)
                   .move_to(c00 + np.array([cell / 2, cell / 2, 0])))
            g_.add(Dot(q, radius=0.06, color=RAY))
            g_.add(tx(f"(<i>u</i>, <i>v</i>) = ({u:.1f}, {v:.1f})", 22).move_to(PC + DOWN * (PS / 2 + 0.55)))
            return g_

        dm = always_redraw(dyn)
        self.add(dm)
        self.caption("For each pixel, cast a ray and intersect it with that plane.")
        e5 = tx(f"{X_} = {B('o')} + <i>t</i> {B('d')},    <i>t</i> = ({MU} − {B('o')}) · {B('n')} / ({B('d')} · {B('n')})", 28
                ).move_to([-2.3, 2.7, 0])
        self.add(e5)
        self.play(k.animate.set_value(0.5), run_time=3.5, rate_func=linear)
        e6 = tx(f"<i>u</i> = ( <i>m</i> σ" + sup("1") + f" + ({X_} − {MU}) · {B('r')}" + sup("1") + " ) / (2 <i>m</i> σ" + sup("1")
                + ") · (<i>𝒯</i> − 1)", 28).move_to([1.6, -2.3, 0])
        e6b = tx("<i>v</i>: same with σ" + sup("2") + ", " + B("r") + sup("2") + ";   <i>m</i> = 3", 22, DIM
                 ).next_to(e6, DOWN, buff=0.12)
        self.add(e6, e6b)
        self.caption("Offsets along the axes, in units of σ, give UV coordinates.")
        self.play(k.animate.set_value(1.0), run_time=3.5, rate_func=linear)
        self.caption("Bilinear lookup of 4 texels gives texture color and alpha.")
        self.play(k.animate.set_value(1.35), run_time=2.0, rate_func=linear)
        dm.clear_updaters()
        self.caption("The paper does this in custom CUDA kernels.", 0.3)
        self.clear_all()


# ================================================================== S06 Appearance model
class S06_Appearance(Base):
    def construct(self):
        self.title_card("Part 5", "A generalized appearance model")
        self.header("5 · Color, opacity, compositing")
        T = 16
        base = np.array([0.55, 0.35, 0.25])
        ct = painting_tex(T)
        ct = ct - ct.mean((0, 1))
        at = star_mask(T, k=4, r0=0.6, r1=1.0) * 0.6 + 0.4 * smooth_noise(T, 1.4, 7)
        E, h = 1.8, 1.75
        s1, s2, th = 0.62, 0.5, 0.35
        e7 = tx(f"{CI}({P_}) = {CBASE} + {CTEX}{UV}", 32).move_to([-2.9, 2.55, 0])
        e8 = tx(f"{ALPHA}" + sub(I_) + f"({P_}) = {ATEX}{UV} · {G_}" + sub(I_) + f"({X_}) · <i>o</i>" + sub(I_), 32
                ).move_to([3.1, 2.55, 0])
        self.add(e7)
        # colour row
        y1 = 0.75
        a0, cb = splat_layers(200, E, s1, s2, th, base)
        cbase_t = arr_img((np.ones((8, 8, 3)) * base * 255).astype(np.uint8), h, True).move_to([-5.4, y1, 0])
        ctex_t = tex_tile(np.clip(ct + 0.5, 0, 1), h).move_to([-3.1, y1, 0])
        csum_t = tex_tile(np.clip(ct + base, 0, 1), h).move_to([-0.8, y1, 0])
        row1 = [cbase_t, ctex_t, csum_t]
        self.play(*[FadeIn(r) for r in row1], run_time=0.6)
        self.add(tx("+", 36).move_to([-4.25, y1, 0]), tx("=", 36).move_to([-1.95, y1, 0]),
                 labeled(cbase_t, "base color (SH)", 20, DIM, DOWN), labeled(ctex_t, "texture color", 20, DIM, DOWN),
                 labeled(csum_t, "per-pixel color", 20, DIM, DOWN))
        self.caption("Color: constant base color (spherical harmonics) plus texture color.")
        # alpha row
        self.add(e8)
        y2 = -1.75
        g_only = splat_layers(200, E, s1, s2, th, base)[0]
        atex_t = tex_tile(at, h, alpha=True).move_to([-5.4, y2, 0])
        g_t = arr_img((np.dstack([g_only] * 3) * 255).astype(np.uint8), h).move_to([-3.1, y2, 0])
        a_full, c_full = splat_layers(200, E, s1, s2, th, base, 1.0, at, ct)
        a_t = arr_img((np.dstack([a_full] * 3) * 255).astype(np.uint8), h).move_to([-0.8, y2, 0])
        self.play(FadeIn(atex_t), FadeIn(g_t), FadeIn(a_t), run_time=0.6)
        self.add(tx("×", 36).move_to([-4.25, y2, 0]), tx("=", 36).move_to([-1.95, y2, 0]),
                 labeled(atex_t, "texture alpha", 20, DIM, DOWN), labeled(g_t, f"{G_}" + sub(I_) + " · <i>o</i>" + sub(I_), 20, DIM, DOWN),
                 labeled(a_t, "per-pixel opacity", 20, DIM, DOWN))
        self.caption("Opacity: texture alpha times the usual falloff and opacity.")
        fin = arr_img(over_black(a_full, c_full), 3.2).move_to([4.2, -0.5, 0])
        a1 = arrow([0.3, y1, 0], fin.get_left() + UP * 0.6)
        a2 = arrow([0.3, y2, 0], fin.get_left() + DOWN * 0.6)
        self.play(GrowArrow(a1), GrowArrow(a2), run_time=0.5)
        self.play(FadeIn(fin), run_time=0.5)
        self.add(labeled(fin, "one textured Gaussian", 22, OURS, DOWN))
        self.wait(0.8)
        self.clear_all()

        e9 = tx(f"{B('c')}" + sup("final") + f"({P_}) = ∑" + sub(I_) + f" {CI}({P_}) {ALPHA}" + sub(I_) + f"({P_}) ∏"
                + sub(f"{J_}&lt;{I_}") + f" (1 − {ALPHA}" + sub(J_) + f"({P_}))", 38).move_to([0, 2.2, 0])
        self.add(e9)
        self.caption("Then composite front to back, as in 3DGS.")
        sc = tx(f"{CTEX} = 0  and  {ATEX} = 1   ⟹   original 3DGS", 30).move_to([0, 1.15, 0])
        self.add(sc)
        self.caption("Zero texture color and unit alpha: exactly 3DGS again.")
        # two-stage optimisation timeline
        y = -0.6
        b1 = RoundedRectangle(width=5.2, height=1.1, corner_radius=0.12, fill_color=BOX, fill_opacity=1,
                              stroke_color=GS, stroke_width=2).move_to([-3.0, y, 0])
        b2 = RoundedRectangle(width=5.2, height=1.1, corner_radius=0.12, fill_color=BOX, fill_opacity=1,
                              stroke_color=OURS, stroke_width=2).move_to([3.0, y, 0])
        self.play(FadeIn(b1), run_time=0.4)
        self.add(tx("Stage 1 · plain 3DGS", 26, GS).move_to(b1.get_center() + UP * 0.2),
                 tx("30,000 iterations, densification on", 20, DIM).move_to(b1.get_center() + DOWN * 0.25))
        self.play(GrowArrow(arrow(b1.get_right(), b2.get_left(), buff=0.05)), FadeIn(b2), run_time=0.5)
        self.add(tx("Stage 2 · + textures", 26, OURS).move_to(b2.get_center() + UP * 0.2),
                 tx("30,000 more, densification off", 20, DIM).move_to(b2.get_center() + DOWN * 0.25))
        self.add(tx("ℒ = <i>λ</i> ℒ" + sub("1") + " + (1 − <i>λ</i>) ℒ" + sub("SSIM") + ",   <i>λ</i> = 0.8", 28).move_to([0, -1.95, 0]))
        self.caption("Training: fit plain 3DGS, then add textures and optimize jointly.")
        self.caption("Stage 2 keeps the number of Gaussians fixed.", 0.3)
        self.clear_all()


# ================================================================== S07 Toy experiment
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
        out |= np.abs(r - k) < 0.9 / (min(s1, s2) * sc) * k
    a = np.zeros((res, res, 4), np.uint8)
    a[out] = (255, 255, 255, 230)
    return a


class S07_ToyFit(Base):
    def construct(self):
        self.title_card("Part 6", "Seeing it work: a 2D experiment")
        self.header("6 · Our 2D toy: 64 Gaussians, with and without textures")
        R = load_toy()
        h = 2.6
        xs = [-5.55, -2.78, 0.0, 2.78, 5.55]
        y = 0.3
        tgt = img(gen("toy_target_hi.png"), h).move_to([xs[0], y, 0])
        self.play(FadeIn(tgt), run_time=0.5)
        tl = labeled(tgt, "target", 24)
        self.add(tl)
        # stage 1 progress (plain)
        s1 = R["stage1"]
        frames1 = [img(gen(f["file"]), h).move_to([xs[1], y, 0]) for f in s1]
        kk = ValueTracker(0)

        def f1():
            return frames1[int(round(kk.get_value()))]

        def l1():
            f = s1[int(round(kk.get_value()))]
            return VGroup(tx(f"iteration {f['iter']}", 20, DIM), tx(f"PSNR {f['psnr']:.2f} dB", 22)
                          ).arrange(DOWN, buff=0.08).next_to(frames1[0], DOWN, buff=0.15)

        fm = always_redraw(f1)
        lm = always_redraw(l1)
        s1l = labeled(frames1[0], "stage 1: plain", 24, GS)
        self.add(fm, lm, s1l)
        self.caption("Stage 1: fit 64 plain Gaussians by gradient descent.")
        self.play(kk.animate.set_value(len(s1) - 1), run_time=3.0, rate_func=linear)
        fm.clear_updaters()
        lm.clear_updaters()
        self.remove(fm, lm, s1l)
        # stage 2 variants
        modes = [("none", "no texture", GS), ("alpha", "alpha only", C_A), ("rgb", "RGB", C_RGB), ("rgba", "RGBA", C_RGBA)]
        S2 = {md: R["stage2"][md] for md, _, _ in modes}
        fr = {md: [img(gen(f["file"]), h).move_to([xs[i + 1], y, 0]) for f in S2[md]] for i, (md, _, _) in enumerate(modes)}
        k2 = ValueTracker(0)

        def mk(md):
            def f():
                return fr[md][int(round(k2.get_value()))]
            return f

        def mkl(md):
            def f():
                e = S2[md][int(round(k2.get_value()))]
                return tx(f"{e['psnr']:.2f} dB", 26).next_to(fr[md][0], DOWN, buff=0.18)
            return f

        dyn = []
        for md, name, cc in modes:
            dyn.append(always_redraw(mk(md)))
            dyn.append(always_redraw(mkl(md)))
            self.add(labeled(fr[md][0], name, 24, cc))
        self.add(*dyn)
        it_lab = tx("stage 2: 1,500 more iterations from the same start · 8 × 8 textures", 22, DIM).move_to([1.4, 2.3, 0])
        self.add(it_lab)
        self.caption("Stage 2: same start; three variants get 8 × 8 textures.")
        self.play(k2.animate.set_value(len(S2["none"]) - 1), run_time=4.5, rate_func=linear)
        for d in dyn:
            d.clear_updaters()
        fin = {md: S2[md][-1]["psnr"] for md, _, _ in modes}
        self.caption(f"Same 64 Gaussians: {fin['none']:.1f} dB plain, {fin['rgba']:.1f} dB with RGBA textures.")
        ov_n = arr_img(ellipse_overlay(np.load(gen("toy_s2_none_params.npy"))), h).move_to(fr["none"][0])
        ov_r = arr_img(ellipse_overlay(np.load(gen("toy_s2_rgba_params.npy"))), h).move_to(fr["rgba"][0])
        self.play(FadeIn(ov_n), FadeIn(ov_r), run_time=0.8)
        self.caption("Outlines: the 2σ footprints, 64 in both fits.", 0.3)
        self.play(FadeOut(ov_n), FadeOut(ov_r), run_time=0.4)
        self.caption("Here RGB beats alpha-only; in the paper's 3D scenes it is reversed.")
        self.caption("Toy numbers are ours; textures add parameters. Now the paper.", 0.3)
        self.clear_all()


# ================================================================== S08 Results
class S08_Results(Base):
    def construct(self):
        self.title_card("Part 7", "Results")
        self.header("7 · Novel-view synthesis results")
        ds = [("Blender", "8 synthetic objects"), ("Mip-NeRF 360", "9 scenes"), ("DTU", "5 objects"),
              ("Tanks and Temples", "2 scenes"), ("Deep Blending", "2 scenes"), ("Custom captures", "2 art-center rooms")]
        rows = VGroup(*[VGroup(tx(a, 28), tx(b, 24, DIM)).arrange(RIGHT, buff=0.4) for a, b in ds]
                      ).arrange(DOWN, buff=0.22, aligned_edge=LEFT).move_to([-2.6, 0.4, 0])
        self.add(rows)
        base = VGroup(tx("Baseline 3DGS*:", 26, GS),
                      tx("the authors' 3DGS, following", 22, DIM),
                      tx("Gaussian Opacity Fields", 22, DIM),
                      tx("(exact ray-based Gaussian values,", 22, DIM),
                      tx("revised densification)", 22, DIM)).arrange(DOWN, buff=0.1, aligned_edge=LEFT).move_to([4.0, 0.4, 0])
        self.add(base)
        self.caption("Five benchmarks plus two custom captures, against 3DGS*.", 0.5)
        self.remove(rows, base)

        # ---- PSNR vs % Gaussians (supplementary Table A.3)
        pct = [1, 2, 5, 10, 20, 50, 100]
        D = {"Mip-NeRF 360": ([23.75, 24.79, 25.87, 26.45, 26.83, 27.13, 27.35],
                              [22.37, 23.43, 24.88, 25.77, 26.45, 27.02, 27.28], (22, 28)),
             "Tanks and Temples": ([21.08, 22.14, 23.05, 23.42, 23.99, 24.18, 24.26],
                                   [19.90, 21.05, 22.25, 22.82, 23.58, 23.96, 24.18], (19.5, 24.5)),
             "Deep Blending": ([24.88, 26.29, 27.00, 27.98, 28.16, 28.27, 28.33],
                               [23.97, 25.49, 26.55, 27.64, 27.99, 28.11, 28.04], (23.5, 28.5))}
        cx = [-4.6, 0.0, 4.6]
        plots = []
        for i, (name, (o_, g_, yr)) in enumerate(D.items()):
            ax = Axes(x_range=[0, 2, 1], y_range=[yr[0], yr[1], 1], x_length=3.6, y_length=3.0, tips=False,
                      axis_config={"color": DIM, "stroke_width": 2}).move_to([cx[i], 0.4, 0])
            lx = [np.log10(p) for p in pct]
            po = VMobject(stroke_color=OURS, stroke_width=4).set_points_as_corners([ax.c2p(a, b) for a, b in zip(lx, o_)])
            pg = DashedVMobject(VMobject().set_points_as_corners([ax.c2p(a, b) for a, b in zip(lx, g_)]), num_dashes=40
                                ).set_stroke(GS, 3)
            dots_o = VGroup(*[Dot(ax.c2p(a, b), radius=0.05, color=OURS) for a, b in zip(lx, o_)])
            dots_g = VGroup(*[Dot(ax.c2p(a, b), radius=0.05, color=GS) for a, b in zip(lx, g_)])
            xt = VGroup(*[tx(f"{p}%", 18, DIM).next_to(ax.c2p(np.log10(p), yr[0]), DOWN, buff=0.1) for p in (1, 10, 100)])
            ytk = VGroup(*[tx(f"{v}", 18, DIM).next_to(ax.c2p(0, v), LEFT, buff=0.1)
                           for v in range(int(np.ceil(yr[0])), int(yr[1]) + 1, 2)])
            ttl = tx(name, 24).next_to(ax, UP, buff=0.2)
            self.play(Create(ax), run_time=0.4)
            self.add(xt, ytk, ttl)
            plots.append((ax, po, pg, dots_o, dots_g))
        self.add(tx("PSNR (dB)", 20, DIM).move_to([-6.25, 2.2, 0]),
                 tx("% of the default number of Gaussians", 22, DIM).move_to([0, -1.75, 0]))
        leg = VGroup(Line(ORIGIN, RIGHT * 0.5, color=OURS, stroke_width=4), tx("Textured Gaussians (RGBA)", 22, OURS),
                     DashedLine(ORIGIN, RIGHT * 0.5, color=GS, stroke_width=3), tx("3DGS*", 22, GS)
                     ).arrange(RIGHT, buff=0.2).move_to([0, 2.75, 0])
        self.add(leg)
        self.play(*[Create(p[2]) for p in plots], *[FadeIn(p[4]) for p in plots], run_time=1.0)
        self.play(*[Create(p[1]) for p in plots], *[FadeIn(p[3]) for p in plots], run_time=1.0)
        self.caption("The fewer the Gaussians, the bigger the lead.")
        self.caption("Total texels fixed: fewer Gaussians get larger textures.", 0.3)
        self.clear_all()

        # ---- Table 1 gains at 1 % and 100 %
        names = ["Blender", "Mip-NeRF 360", "DTU", "Tanks &|Temples", "Deep|Blending"]
        g1 = [(26.89, 28.11), (22.37, 23.73), (30.88, 32.43), (19.90, 21.10), (23.97, 24.83)]
        g100 = [(33.08, 33.24), (27.26, 27.35), (33.54, 33.61), (24.18, 24.26), (28.04, 28.33)]
        ax = Axes(x_range=[0, 5, 1], y_range=[0, 1.8, 0.5], x_length=9.0, y_length=3.6, tips=False,
                  axis_config={"color": DIM, "stroke_width": 2}).move_to([0.2, 0.25, 0])
        self.play(Create(ax), run_time=0.5)
        self.add(tx("PSNR gain over 3DGS* (dB)", 22, DIM).next_to(ax, UP, buff=0.15).align_to(ax, LEFT),
                 *[tx(f"{v:.1f}", 18, DIM).next_to(ax.c2p(0, v), LEFT, buff=0.1) for v in (0.5, 1.0, 1.5)])
        bars = VGroup()
        labs = VGroup()
        for i, ((a1, b1), (a2, b2)) in enumerate(zip(g1, g100)):
            for j, (d, c_, lab) in enumerate(((b2 - a2, GS, "100%"), (b1 - a1, OURS, "1%"))):
                x = i + 0.3 + 0.4 * j
                r = Rectangle(width=0.34, height=max(0.01, ax.c2p(0, d)[1] - ax.c2p(0, 0)[1]), stroke_width=0,
                              fill_color=c_, fill_opacity=0.9)
                r.move_to(ax.c2p(x + 0.2, 0), aligned_edge=DOWN)
                bars.add(r)
                labs.add(tx(f"+{d:.2f}", 18, c_).next_to(r, UP, buff=0.06))
            nm = VGroup(*[tx(w, 20) for w in names[i].split("|")]).arrange(DOWN, buff=0.06)
            self.add(nm.next_to(ax.c2p(i + 0.5, 0), DOWN, buff=0.15))
        self.play(LaggedStart(*[GrowFromEdge(b, DOWN) for b in bars], lag_ratio=0.1), run_time=1.6)
        self.add(labs)
        leg = VGroup(Square(0.22, fill_color=GS, fill_opacity=0.9, stroke_width=0), tx("100% of Gaussians", 20, GS),
                     Square(0.22, fill_color=OURS, fill_opacity=0.9, stroke_width=0), tx("1% of Gaussians", 20, OURS)
                     ).arrange(RIGHT, buff=0.15).move_to([3.4, 2.45, 0])
        self.add(leg)
        self.caption("Table 1, equal counts: ahead everywhere; +0.86 to +1.55 dB at 1%.")
        self.caption("DTU at 1%: 30.88 dB (3DGS*) vs 32.43 dB (ours).", 0.3)
        self.clear_all()

        # ---- qualitative
        sets = [(("lego", "Blender", "lego"), ("train", "Tanks & Temples", "train")),
                (("drjohnson", "Deep Blending", "drjohnson"), ("counter", "Mip-NeRF 360", "counter"))]
        for si, rows_ in enumerate(sets):
            ims = []
            for r_, (scn, dsn, sn) in enumerate(rows_):
                y = 1.4 - r_ * 2.62
                a = img(f"fig5_{scn}_3dgs.png", 2.4).move_to([-3.3, y, 0])
                b = img(f"fig5_{scn}_ours.png", 2.4).move_to([3.3, y, 0])
                ims += [a, b]
                self.add(VGroup(tx(dsn, 18, DIM), tx(sn, 18, DIM)).arrange(DOWN, buff=0.06).move_to([0, y, 0]))
            self.play(*[FadeIn(m) for m in ims], run_time=0.6)
            self.add(tx("3DGS*", 24, GS).next_to(ims[0], UP, buff=0.08), tx("Ours (RGBA)", 24, OURS).next_to(ims[1], UP, buff=0.08))
            if si == 0:
                self.caption("Equal, small counts: about 2.5k Gaussians per object, 39k per scene.")
                self.caption("Textures keep treads, lettering and fine structure.", 0.3)
            else:
                self.caption("Portraits and printed labels survive; plain Gaussians smear them.", 0.3)
            self.clear_all()


# ================================================================== S09 Ablations
class S09_Ablations(Base):
    def construct(self):
        self.title_card("Part 8", "Which channels matter?")
        self.header("8 · Alpha, RGB or RGBA?")
        # ΔPSNR at 1 % Gaussians, supplementary Table A.4 (T&T / DB columns read in the order of Table 1)
        dsn = ["Blender", "Tanks &|Temples", "Deep|Blending"]
        gs1 = [26.89, 19.90, 23.97]
        var = {"alpha": [27.64, 20.93, 24.68], "rgb": [27.60, 20.73, 24.64], "rgba": [28.11, 21.08, 24.83]}
        cc = {"alpha": C_A, "rgb": C_RGB, "rgba": C_RGBA}
        ax = Axes(x_range=[0, 3, 1], y_range=[0, 1.4, 0.5], x_length=6.2, y_length=3.4, tips=False,
                  axis_config={"color": DIM, "stroke_width": 2}).move_to([-3.2, 0.2, 0])
        self.play(Create(ax), run_time=0.5)
        self.add(tx("PSNR gain over 3DGS*, 1% of Gaussians (dB)", 20, DIM).next_to(ax, UP, buff=0.15).align_to(ax, LEFT),
                 *[tx(f"{v:.1f}", 18, DIM).next_to(ax.c2p(0, v), LEFT, buff=0.1) for v in (0.5, 1.0)])
        bars, labs = VGroup(), VGroup()
        for i in range(3):
            for j, k in enumerate(("alpha", "rgb", "rgba")):
                d = var[k][i] - gs1[i]
                r = Rectangle(width=0.4, height=ax.c2p(0, d)[1] - ax.c2p(0, 0)[1], stroke_width=0,
                              fill_color=cc[k], fill_opacity=0.9).move_to(ax.c2p(i + 0.25 + 0.25 * j, 0), aligned_edge=DOWN)
                bars.add(r)
                labs.add(tx(f"{d:.2f}", 16, cc[k]).next_to(r, UP, buff=0.05))
            nm = VGroup(*[tx(w, 20) for w in dsn[i].split("|")]).arrange(DOWN, buff=0.06)
            self.add(nm.next_to(ax.c2p(i + 0.5, 0), DOWN, buff=0.15))
        self.play(LaggedStart(*[GrowFromEdge(b, DOWN) for b in bars], lag_ratio=0.08), run_time=1.5)
        self.add(labs)
        leg = VGroup(*[VGroup(Square(0.22, fill_color=cc[k], fill_opacity=0.9, stroke_width=0), tx(n, 20, cc[k])).arrange(RIGHT, buff=0.12)
                       for k, n in (("alpha", "alpha only"), ("rgb", "RGB"), ("rgba", "RGBA"))]).arrange(RIGHT, buff=0.35
                                                                                                        ).next_to(ax, DOWN, buff=0.75)
        self.add(leg)
        side = VGroup(tx("texture channels per texel", 22, DIM),
                      tx("alpha only:  1", 26, C_A), tx("RGB:  3", 26, C_RGB), tx("RGBA:  4", 26, C_RGBA),
                      tx("model size vs 3DGS (same count)", 22, DIM),
                      tx("1.27×  ·  1.82×  ·  2.08×", 26)).arrange(DOWN, buff=0.2, aligned_edge=LEFT).move_to([3.8, 0.3, 0])
        side[4].shift(DOWN * 0.25)
        side[5].shift(DOWN * 0.25)
        self.add(side)
        self.caption("RGBA is best, yet alpha-only beats RGB with a third of the channels.")
        self.caption("Alpha reshapes Gaussians; RGB-textured ones stay ellipses.", 0.3)
        self.clear_all()

        # tex ablation crops (lego, 1% Gaussians)
        vs = [("3dgs", "3DGS*", GS), ("alpha", "alpha only", C_A), ("rgb", "RGB", C_RGB), ("rgba", "RGBA", C_RGBA)]
        xs = [-5.1, -1.7, 1.7, 5.1]
        ims = []
        for i, (k, n, c_) in enumerate(vs):
            for r_, scn in enumerate(("lego", "ship")):
                m_ = img(f"figA_{scn}_{k}.png", 2.3).move_to([xs[i], 1.15 - 2.65 * r_, 0])
                if m_.width > 3.3:
                    m_.set(width=3.3)
                ims.append(m_)
            self.add(tx(n, 24, c_).next_to(ims[-2], UP, buff=0.12))
        self.play(*[FadeIn(m) for m in ims], run_time=0.7)
        self.caption("Blender, 1% of the Gaussians: alpha variants recover the fine structure.", 0.3)
        self.clear_all()

        # same model size (Table 2)
        hdr = ["", "3DGS*", "alpha only", "RGBA"]
        data = [("Blender", (33.08, 0.27), (33.12, 0.19), (33.03, 0.21)),
                ("Mip-NeRF 360", (27.26, 4.4), (27.37, 3.1), (27.26, 3.5)),
                ("DTU", (33.53, 0.28), (33.45, 0.24), (33.41, 0.22)),
                ("Tanks and Temples", (24.17, 2.9), (24.38, 2.6), (24.28, 1.3)),
                ("Deep Blending", (28.04, 2.8), (28.36, 1.6), (28.52, 1.0))]
        xs = [-4.3, -0.9, 1.9, 4.7]
        y0 = 1.9
        self.add(tx("Same model size (Table 2): PSNR (dB) · #Gaussians (millions)", 26).move_to([0, 2.75, 0]))
        for j, hname in enumerate(hdr):
            if hname:
                self.add(tx(hname, 24, [WHITE, GS, C_A, C_RGBA][j]).move_to([xs[j], y0, 0]))
        for i, (name, *vals) in enumerate(data):
            y = y0 - 0.62 * (i + 1)
            self.add(tx(name, 24).move_to([xs[0], y, 0]))
            best = max(v[0] for v in vals)
            for j, (p, n) in enumerate(vals):
                s = f"{p:.2f} · {n}"
                t = tx(f"<b>{s}</b>" if p == best else s, 24, WHITE if p == best else DIM).move_to([xs[j + 1], y, 0])
                self.add(t)
        self.add(Line([-6.2, y0 - 0.3, 0], [6.2, y0 - 0.3, 0], stroke_width=1, color=DIM))
        self.caption("Equal memory: alpha-only wins on three of five datasets.")
        self.caption("A sweet spot between Gaussian count and texture size.", 0.3)
        self.clear_all()


# ================================================================== S10 Outlook
class S10_Outlook(Base):
    def construct(self):
        self.title_card("Part 9", "Costs, limits, outlook")
        self.header("9 · Costs and limitations")
        ttl = tx("Same 2.12 M Gaussians, 4 × 4 RGBA textures vs 3DGS (supplement)", 24, DIM).move_to([0, 2.6, 0])
        m = [("model size", "125.3 MB → 261.2 MB"), ("GPU memory", "2.4 GB → 4.0 GB"),
             ("time per frame", "0.073 s → 0.123 s"), ("training", "about 2× (two stages)")]
        cells = VGroup(*[VGroup(tx(a, 22, DIM), tx(b, 28)).arrange(DOWN, buff=0.1) for a, b in m])
        cells.arrange_in_grid(rows=2, cols=2, buff=(1.6, 0.35)).move_to([0, 1.2, 0])
        self.add(ttl, cells)
        self.caption("Textures about double model size, and training has two stages.")
        lim = VGroup(tx("Limitations and next steps", 24, DIM),
                     tx("2D diffuse textures on a local surface: no spatially varying specular color", 26),
                     tx("→ 3D volume textures, 5D radiance fields (e.g. TensoRF, triplanes)", 26),
                     tx("→ time-varying textures for dynamic scenes", 26)).arrange(DOWN, buff=0.18, aligned_edge=LEFT)
        lim.move_to([0, -1.55, 0])
        self.add(lim)
        self.caption("Diffuse 2D textures miss spatially varying specular color.", 0.5)
        self.clear_all()
        summ = VGroup(tx("1.  Attach an alpha, RGB or RGBA texture map to every Gaussian.", 28),
                      tx("2.  Ray–plane intersection gives UV; texture modulates color and opacity.", 28),
                      tx("3.  Sharper novel views than 3DGS with the same Gaussians;", 28),
                      tx("the biggest gains when Gaussians are few.", 28)).arrange(DOWN, buff=0.28, aligned_edge=LEFT).move_to([0, 0.6, 0])
        summ[3].shift(RIGHT * 0.45)
        self.add(summ)
        self.wait(4.0)
        self.remove(summ, self.hdr)
        self.hdr = None
        t = tx("Textured Gaussians for Enhanced 3D Scene Appearance Modeling", 38, bold=True).move_to([0, 1.6, 0])
        if t.width > 13:
            t.set(width=13)
        v = tx("CVPR 2025", 28, DIM).move_to([0, 0.95, 0])
        u = tx("arXiv 2411.18625   ·   textured-gaussians.github.io", 30, OURS).move_to([0, 0.0, 0])
        a1 = tx("Brian Chao, Hung-Yu Tseng, Lorenzo Porzi, Chen Gao, Tuotuo Li, Qinbo Li,", 21, DIM)
        a2 = tx("Ayush Saraf, Jia-Bin Huang, Johannes Kopf, Gordon Wetzstein, Changil Kim", 21, DIM)
        a3 = tx("Stanford University  ·  Meta  ·  University of Maryland, College Park", 21, DIM)
        au = VGroup(a1, a2, a3).arrange(DOWN, buff=0.12).move_to([0, -1.6, 0])
        self.add(t, v, u, au)
        self.wait(5.0)
