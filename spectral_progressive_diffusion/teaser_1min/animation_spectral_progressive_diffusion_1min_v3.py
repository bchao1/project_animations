"""Spectral Progressive Diffusion — 1-minute 3b1b-style teaser, v3.

Condensed from ../animation_spectral_progressive_v4.py (helpers copied, not imported).
Source: Xiao, Chao, Yariv, Wetzstein, "Spectral Progressive Diffusion for Efficient Image and
Video Generation", arXiv 2605.18736.  Paper figure crops and data/corgi_clean.png are read from
the parent project's data/ folder (absolute path).  Spectra / sweeps / DCT demo computed at import.

Render (see render.sh):  manim -ql --media_dir media <this file> <Scene>
Scenes, in order: T1_Hook, T2_Insight, T3_Method, T4_Payoff.
"""
from manim import *
import numpy as np
from PIL import Image
from scipy.fft import dctn, idctn
from scipy.ndimage import zoom
import matplotlib
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = "/home/brianchc/project_animations/spectral_progressive_diffusion/data"

config.background_color = BLACK

# ------------------------------------------------------------------ style
FONT = "CMU Serif"
MATHFONT = "Latin Modern Math"          # CMU Serif lacks × − → ≈ √ ∝ …
SIG = "#58C4DD"        # signal / image power (3b1b blue)
NOI = "#FC6255"        # noise (3b1b red)
FRONT = "#FFCA28"      # frequency frontier / ours
DIM = "#8A8A8A"
GHOST = "#3A3A3A"
ST = ["#5B8FD9", "#E8955A", "#6BBF6B", "#D9534F"]    # stage colours (as in paper Fig. 1)
OURS = "#FFCA28"
OURS3 = "#FF9100"
RALU_C = "#E040FB"
BOT_C = "#A1887F"
BASE_C = "#B0BEC5"
LORA = "#FF7043"

HEADER_Y = 3.62
CAPTION_Y = -3.5
MATHCH = set("×→≈∼−∈𝒩ℒ𝔼′≪≤≥‖·√∝≲⋯∖")


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
    return MarkupText(mathify(s), font=FONT, font_size=size, color=color)


# common symbols
def X(s=None, p=None):
    return "<i>x</i>" + (sup(p) if p else "") + (sub(s) if s is not None else "")


XT = X("<i>t</i>")
X0 = X("0")
EPS = "<i>ε</i>"
OM = "<i>ω</i>"
PW = "<i>P</i>" + sub("<i>ω</i>")
TI = "<i>t</i>" + sub("<i>i</i>")
TTI = "<i>t̃</i>" + sub("<i>i</i>")
DEL = "<i>δ</i>"


# ------------------------------------------------------------------ image helpers
def rgba(a):
    a = np.clip(a, 0.0, 1.0)
    if a.ndim == 2:
        a = np.stack([a] * 3, -1)
    out = np.empty(a.shape[:2] + (4,), np.uint8)
    out[..., :3] = (a * 255 + 0.5).astype(np.uint8)
    out[..., 3] = 255
    return out


def arr_img(a, h, nearest=False):
    m = ImageMobject(rgba(a))
    if nearest:
        m.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
    m.height = h
    return m


def img(path, h=None, w=None):
    p = path if os.path.isabs(path) else os.path.join(DATA, path)
    m = ImageMobject(p)
    if h is not None:
        m.height = h
    if w is not None:
        m.width = w
    return m


def framed(m, color=WHITE, sw=1.5):
    return SurroundingRectangle(m, buff=0, color=color, stroke_width=sw)


def arrow(a, b, buff=0.1, color=WHITE, sw=3):
    return Arrow(a, b, buff=buff, color=color, stroke_width=sw,
                 max_tip_length_to_length_ratio=0.2, tip_length=0.16)


CMAP = matplotlib.colormaps["magma"]


def cmap(v, lo, hi):
    return CMAP(np.clip((v - lo) / (hi - lo), 0, 1))[..., :3]


def disp(x):
    """[-1, 1] image → [0, 1] display."""
    return (x + 1.0) / 2.0


# ------------------------------------------------------------------ computed data
class SpecData:
    """Corgi (paper Fig. 1) at 512²: spectra, radial PSD, noisy states, progressive frames."""
    N = 512
    LEVELS = [32, 64, 128, 256, 512]

    def __init__(self):
        a = np.asarray(Image.open(os.path.join(DATA, "corgi_clean.png")).convert("RGB")) / 255.0
        self.x0 = a * 2 - 1                                   # [-1, 1]
        rng = np.random.default_rng(7)
        self.eps = rng.standard_normal(self.x0.shape)
        xc = self.x0 - self.x0.mean(axis=(0, 1))              # centred, as in the paper
        self.X0 = np.fft.fftshift(np.fft.fft2(xc, axes=(0, 1), norm="ortho"), axes=(0, 1))
        self.E = np.fft.fftshift(np.fft.fft2(self.eps, axes=(0, 1), norm="ortho"), axes=(0, 1))
        self.X0raw = np.fft.fft2(self.x0, axes=(0, 1), norm="ortho")
        N = self.N
        yy, xx = np.mgrid[-N // 2:N // 2, -N // 2:N // 2]
        self.r = np.sqrt(xx ** 2 + yy ** 2)
        self.cheb = np.maximum(np.abs(xx), np.abs(yy))
        P = (np.abs(self.X0) ** 2).mean(-1)
        rb = np.round(self.r).astype(int)
        cnt = np.bincount(rb.ravel())
        ps = np.bincount(rb.ravel(), P.ravel()) / np.maximum(cnt, 1)
        self.w = np.arange(1, N // 2 + 1).astype(float)
        self.pw = ps[1:N // 2 + 1]
        self.env = np.minimum.accumulate(self.pw)
        b, lA = np.polyfit(np.log10(self.w), np.log10(self.pw), 1)
        self.beta, self.A = -b, 10 ** lA
        # per-resolution copies for the progressive trajectory
        self.x0n, self.epsn = {}, {}
        for n in self.LEVELS:
            k = N // n
            self.x0n[n] = self.x0.reshape(n, k, n, k, 3).mean((1, 3))
            self.epsn[n] = self.eps.reshape(n, k, n, k, 3).mean((1, 3)) * k
        rr = np.fft.ifftshift(self.r)
        self.r_unshift = rr
        self._lp = {}

    # SNR = 1 crossing ("frequency frontier") for the demo image
    def wstar(self, t):
        t = float(np.clip(t, 1e-4, 1.0))
        k = np.where((1 - t) ** 2 * self.env < t ** 2)[0]
        return float(self.w[k[0]]) if len(k) else float(self.N // 2)

    def t_of_w(self, w):
        """Time at which the frontier sits at radius w: (1 − t)² env(w) = t²."""
        e = np.interp(np.log10(w), np.log10(self.w), np.log10(self.env))
        sq = np.sqrt(10 ** e)
        return float(sq / (1 + sq))

    def n_of(self, t):
        ws = self.wstar(t)
        for n in self.LEVELS:
            if n / 2 >= ws:
                return n
        return self.N

    def xt(self, t):
        return (1 - t) * self.x0 + t * self.eps

    def spec(self, t, band=None):
        F = (1 - t) * self.X0 + t * self.E
        L = np.log10((np.abs(F) ** 2).mean(-1) + 1e-9)
        c = cmap(L, -4.0, 3.5)
        if band is not None:
            c = np.where((self.cheb <= band / 2)[..., None], c, 0.22 * c)
        return c

    def spec_clean(self):
        return self.spec(0.0)

    def lowpass(self, R):
        R = int(round(R))
        if R not in self._lp:
            m = (self.r_unshift <= R)[..., None]
            self._lp[R] = np.real(np.fft.ifft2(self.X0raw * m, axes=(0, 1), norm="ortho"))
        return self._lp[R]

    def prog(self, t):
        n = self.n_of(t)
        x = (1 - t) * self.x0n[n] + t * self.epsn[n]
        return n, np.repeat(np.repeat(disp(x), self.N // n, 0), self.N // n, 1)


D = SpecData()


# ------------------------------------------------------------------ δ-optimal schedule (paper Eq. 9/10)
PIX_A, PIX_B = 3745.46, 2.4493        # paper Table 4: pixel 512² power-law fit


def t_omega(w, delta, A=PIX_A, beta=PIX_B):
    P = A * np.power(w, -beta)
    return 1.0 / (1.0 + np.sqrt(delta / (P * (1 + P - delta))))


def t_align(t, r=2.0):
    return r * t / (1 + (r - 1) * t)


# ------------------------------------------------------------------ plotting helpers
def log_axes(center, w, h, xr, yr, xt_vals, yt_vals, xlab, ylab, size=20):
    """Axes in log10 units with custom tick labels (values given in log10).
    The y range is shifted to start at 0 so the x-axis sits at the bottom; use ax.p(x, y)."""
    ax = Axes(x_range=[xr[0], xr[1], 1], y_range=[0, yr[1] - yr[0], 1], x_length=w, y_length=h,
              tips=False, axis_config={"stroke_width": 2, "color": GREY_B, "include_ticks": False})
    ax.move_to(center)
    ax.p = lambda x, y: ax.c2p(x, y - yr[0])
    labs = VGroup()
    for v, s in xt_vals:
        p = ax.p(v, yr[0])
        labs.add(Line(p + UP * 0.07, p + DOWN * 0.07, stroke_width=2, color=GREY_B))
        labs.add(tx(s, size, GREY_B).next_to(p, DOWN, buff=0.14))
    for v, s in yt_vals:
        p = ax.p(xr[0], v)
        labs.add(Line(p + LEFT * 0.07, p + RIGHT * 0.07, stroke_width=2, color=GREY_B))
        labs.add(tx(s, size, GREY_B).next_to(p, LEFT, buff=0.14))
    xl = tx(xlab, size + 4).next_to(ax.p(xr[1], yr[0]), DOWN, buff=0.14).shift(RIGHT * 0.3)
    yl = tx(ylab, size + 4).next_to(ax.p(xr[0], yr[1]), UP, buff=0.12)
    return ax, labs, VGroup(xl, yl)


def poly(ax, xs, ys, color, sw=3, ymin=None, ymax=None):
    f = ax.p if hasattr(ax, "p") else ax.c2p
    pts = []
    for x, y in zip(xs, ys):
        if (ymin is not None and y < ymin) or (ymax is not None and y > ymax):
            continue
        pts.append(f(x, y))
    m = VMobject(stroke_color=color, stroke_width=sw)
    if len(pts) >= 2:
        m.set_points_as_corners(pts)
    return m


def e10(k):
    return "10" + sup(("−" if k < 0 else "") + str(abs(k)))


PSD_XT = [(0, "1"), (1, "10"), (2, "100")]
PSD_YT = [(-4, e10(-4)), (0, e10(0)), (4, e10(4))]
PSD_XR = (0, np.log10(256))
PSD_YR = (-5, 4.5)


# ------------------------------------------------------------------ scene base
class Base(Scene):
    header_text = ""

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
        for m in mobs:
            m.clear_updaters()
        if mobs:
            self.play(*[FadeOut(m) for m in mobs], run_time=run_time)


def live_img(fn, h, nearest=False):
    """ImageMobject whose pixels are recomputed every frame by fn() → float image."""
    m = arr_img(fn(), h, nearest)
    m.add_updater(lambda mob: setattr(mob, "pixel_array", rgba(fn())))
    return m


def frontier_circle(get_w, spec, S):
    return always_redraw(lambda: Circle(radius=max(get_w(), 1.5) / 256 * S / 2,
                                        color=FRONT, stroke_width=3).move_to(spec))

def tokgrid(n, size, center, color=ST[0]):
    c = size / n
    return VGroup(*[RoundedRectangle(width=0.8 * c, height=0.8 * c, corner_radius=0.08 * c,
                                     fill_color=color, fill_opacity=0.85, stroke_width=0)
                    .move_to(np.array(center) + np.array([(j + 0.5) * c - size / 2, size / 2 - (i + 0.5) * c, 0]))
                    for i in range(n) for j in range(n)])



def expansion_demo(t_i=1 / 3, r=2):
    n = 128
    a = np.asarray(Image.open(os.path.join(DATA, "corgi_clean.png")).convert("RGB").resize((n, n), Image.BOX))
    x0 = a / 255.0 * 2 - 1
    Xc = dctn(x0, axes=(0, 1), norm="ortho")
    x0_low = idctn(Xc[:n // 2, :n // 2], axes=(0, 1), norm="ortho") / r
    rng = np.random.default_rng(3)
    e_low = rng.standard_normal(x0_low.shape)
    xt_low = (1 - t_i) * x0_low + t_i * e_low
    C_low = dctn(xt_low, axes=(0, 1), norm="ortho")
    C = t_i * rng.standard_normal((n, n, 3))
    C[:n // 2, :n // 2] = C_low
    x_exp = idctn(C, axes=(0, 1), norm="ortho")
    kappa = r / (1 + (r - 1) * t_i)
    t_new = r * t_i / (1 + (r - 1) * t_i)
    x_al = kappa * x_exp
    x_true = (1 - t_new) * x0 + t_new * rng.standard_normal((n, n, 3))
    x_bil = zoom(xt_low, (2, 2, 1), order=1)

    def cvis(Cf):
        return cmap(np.log10((Cf ** 2).mean(-1) + 1e-9), -1.4, 1.4)
    return dict(xt_low=xt_low, C_low=cvis(C_low), C=cvis(C), x_exp=x_exp, x_al=x_al, x_true=x_true,
                x_bil=x_bil, kappa=kappa, t_new=t_new)



def fade_all(scene, run_time=0.6):
    scene.uncaption()
    mobs = list(scene.mobjects)
    for m in mobs:
        m.clear_updaters()
    if mobs:
        scene.play(*[FadeOut(m) for m in mobs], run_time=run_time)


# ================================================================== T1 Hook
class T1_Hook(Base):
    def construct(self):
        title = tx("Spectral Progressive Diffusion", 54, bold=True).move_to([0, 3.05, 0])
        subt = tx("for Efficient Image and Video Generation", 28).move_to([0, 2.4, 0])
        self.add(title, subt)
        ts = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
        xs = [-5.25 + 2.1 * k for k in range(6)]
        thumbs = Group(*[arr_img(disp(D.xt(tv)), 1.6).move_to([xs[k], 0.55, 0]) for k, tv in enumerate(ts)])
        labs = VGroup(*[tx(f"<i>t</i> = {tv:.1f}", 22).next_to(thumbs[k], UP, buff=0.12) for k, tv in enumerate(ts)])
        grids = VGroup(*[tokgrid(12, 1.2, (xs[k], -1.5, 0)) for k in range(6)])
        dn = arrow([-6.2, -0.55, 0], [6.2, -0.55, 0], buff=0, color=GREY_B, sw=2)
        self.caption("Every denoising step pays for the full-resolution grid.")
        self.play(LaggedStart(*[FadeIn(m) for m in thumbs], lag_ratio=0.15), run_time=1.0)
        self.add(labs)
        self.play(GrowArrow(dn), LaggedStart(*[FadeIn(g) for g in grids], lag_ratio=0.12), run_time=1.0)
        self.add(tx("denoising", 20, GREY_B).next_to(dn, DOWN, buff=0.05).set_x(5.6))
        box = SurroundingRectangle(Group(thumbs[0], thumbs[1], labs[0], labs[1], grids[0], grids[1]),
                                   color=NOI, buff=0.12)
        self.caption("Early on, that compute is spent on pure noise.")
        self.play(Create(box), run_time=0.5)
        fade_all(self)


# ================================================================== T2 Insight
class T2_Insight(Base):
    def construct(self):
        S = 3.0
        Y = 0.4
        t = ValueTracker(0.0)
        pic = live_img(lambda: disp(D.xt(t.get_value())), S).move_to([-5.3, Y, 0])
        spec = live_img(lambda: D.spec(t.get_value()), S).move_to([-1.55, Y, 0])
        f1, f2 = framed(pic, GREY_B, 1), framed(spec, GREY_B, 1)
        tl = always_redraw(lambda: tx(f"<i>t</i> = {t.get_value():.2f}", 26).next_to(f1, DOWN, buff=0.2))
        top1 = tx("image " + XT, 22, DIM).next_to(f1, UP, buff=0.15)
        top2 = tx("its Fourier spectrum", 22, DIM).next_to(f2, UP, buff=0.15)
        ax, ticks, alab = log_axes([4.0, Y, 0], 4.6, 3.4, PSD_XR, PSD_YR, PSD_XT, PSD_YT, "|" + OM + "|", "power")
        lw, lp = np.log10(D.w), np.log10(D.pw)
        xr1 = PSD_XR[1]

        def tc():
            return float(np.clip(t.get_value(), 0.02, 0.995))

        sig = always_redraw(lambda: poly(ax, lw, lp + 2 * np.log10(1 - tc()), SIG, 3, ymin=PSD_YR[0]))
        noi = always_redraw(lambda: poly(ax, [0, xr1], [2 * np.log10(tc())] * 2, NOI, 3))
        shade = always_redraw(lambda: Rectangle(
            width=max(ax.p(xr1, 0)[0] - ax.p(np.log10(D.wstar(tc())), 0)[0], 0.001),
            height=ax.y_length, stroke_width=0, fill_color=NOI, fill_opacity=0.14)
            .align_to(ax.p(xr1, PSD_YR[0]), DR))
        dot = always_redraw(lambda: Dot(ax.p(np.log10(D.wstar(tc())), 2 * np.log10(tc())), radius=0.07,
                                        color=FRONT) if D.wstar(tc()) < 256 else VGroup())
        circ = frontier_circle(lambda: D.wstar(tc()), spec, S)
        sl = tx("signal", 22, SIG).move_to([4.6, Y + 1.95, 0])
        nl = tx("noise", 22, NOI).move_to([5.85, Y + 1.3, 0])
        self.play(FadeIn(pic), FadeIn(spec), Create(f1), Create(f2), Create(ax), run_time=0.6)
        self.add(tl, top1, top2, ticks, alab, shade, sig, noi, dot, circ, sl, nl)
        self.add(tx("yellow: signal = noise", 20, FRONT).next_to(f2, DOWN, buff=0.2))
        self.caption("Noise drowns the high frequencies first.")
        self.wait(0.6)
        self.play(t.animate.set_value(0.97), run_time=3.0, rate_func=smooth)
        self.wait(0.4)
        lwt = ValueTracker(np.log10(D.wstar(0.97)))
        t.add_updater(lambda m: m.set_value(D.t_of_w(10 ** lwt.get_value())))
        self.add(t)
        self.caption("Denoising resolves coarse structure first, fine detail last.")
        self.play(lwt.animate.set_value(np.log10(256)), run_time=6.0, rate_func=linear)
        t.clear_updaters()
        self.remove(t)
        self.wait(0.8)
        fade_all(self)


# ================================================================== T3 Method
class T3_Method(Base):
    def construct(self):
        # A. grow the grid with the frontier
        S = 3.4
        Y = 0.45
        lwt = ValueTracker(np.log10(D.wstar(0.9)))
        T = lambda: D.t_of_w(10 ** lwt.get_value())
        spec = live_img(lambda: D.spec(T(), band=D.n_of(T())), S).move_to([-3.3, Y, 0])
        f1 = framed(spec, GREY_B, 1)
        sq = VGroup(*[DashedVMobject(Square(n / 512 * S), num_dashes=int(8 + n / 12)).set_stroke(GREY_B, 1.5, 0.6)
                      .move_to(spec) for n in (32, 64, 128, 256)])
        circ = frontier_circle(lambda: 10 ** lwt.get_value(), spec, S)
        band = always_redraw(lambda: Square(D.n_of(T()) / 512 * S).set_stroke(WHITE, 3).move_to(spec))
        pix = live_img(lambda: D.prog(T())[1], S, nearest=True).move_to([2.6, Y, 0])
        f2 = framed(pix, GREY_B, 1)
        glab = always_redraw(lambda: tx(f"{D.n_of(T())} × {D.n_of(T())} grid", 24).next_to(f2, UP, buff=0.15))
        FR = {32: "1/256 of the", 64: "1/64 of the", 128: "1/16 of the", 256: "1/4 of the", 512: "all"}
        frac = always_redraw(lambda: tx(FR[D.n_of(T())] + " full-resolution pixels", 22, FRONT)
                             .next_to(f2, DOWN, buff=0.2))
        self.play(FadeIn(spec), Create(f1), Create(sq), FadeIn(pix), Create(f2), run_time=0.6)
        self.add(tx("spectrum + frontier", 22, DIM).next_to(f1, UP, buff=0.15), circ, band, glab, frac)
        self.caption("So grow the grid as the frontier expands.")
        self.wait(0.5)
        self.play(lwt.animate.set_value(np.log10(256)), run_time=6.5, rate_func=linear)
        self.wait(0.4)
        fade_all(self)

        # B. spectral noise expansion
        E = expansion_demo()
        YR = 1.35
        A = arr_img(disp(E["xt_low"]), 1.9, True).move_to([-5.0, YR, 0])
        fA = framed(A, GREY_B, 1)
        C_full = arr_img(E["C"], 2.4, True).move_to([-0.6, YR, 0])
        fc = framed(C_full, GREY_B, 1)
        Dm = arr_img(disp(E["x_exp"]), 2.4, True).move_to([4.2, YR, 0])
        fD = framed(Dm, GREY_B, 1)
        a1 = arrow(A.get_right(), C_full.get_left(), buff=0.1)
        a2 = arrow(C_full.get_right(), Dm.get_left(), buff=0.1)
        self.play(FadeIn(A), Create(fA), run_time=0.4)
        self.add(tx(X(TI) + ",  64 × 64", 22).next_to(fA, UP, buff=0.15))
        self.caption("Spectral noise expansion: pad new frequencies with noise.")
        self.play(GrowArrow(a1), Create(fc), run_time=0.4)
        self.add(tx("DCT", 22).next_to(a1, UP, buff=0.08))
        Bc = arr_img(E["C_low"], 1.9, True).move_to(A)
        self.play(FadeIn(Bc), run_time=0.3)
        self.play(Bc.animate.scale_to_fit_height(1.2).move_to(C_full.get_corner(UL) + np.array([0.6, -0.6, 0])),
                  run_time=0.8)
        self.add(tx("128 × 128 spectrum", 22).next_to(fc, UP, buff=0.15))
        self.play(FadeIn(C_full), run_time=0.6)
        self.remove(Bc)
        nlab = tx(TI + " " + EPS, 26, FRONT)
        self.add(VGroup(BackgroundRectangle(nlab, fill_opacity=0.75, buff=0.08), nlab)
                 .move_to(C_full.get_center() + np.array([0.6, -0.6, 0])))
        self.play(GrowArrow(a2), run_time=0.3)
        self.add(tx("inverse DCT", 22).next_to(a2, UP, buff=0.08))
        self.play(FadeIn(Dm), Create(fD), run_time=0.5)
        self.add(tx(X(TI) + ",  128 × 128", 22).next_to(fD, UP, buff=0.15))

        # C. the schedule
        t1, t2 = float(t_omega(64, 0.01)), float(t_omega(128, 0.01))
        X_ = lambda tv: -6.0 + (1 - tv) * 12.0
        YT = -1.35
        segs = [(1.0, t1, "<i>s</i>" + sub("1") + " = 1/4", "1/16 of the tokens"),
                (t1, t2, "<i>s</i>" + sub("2") + " = 1/2", "1/4"),
                (t2, 0.0, "<i>s</i>" + sub("3") + " = 1", "all tokens")]
        rects, labs = VGroup(), VGroup()
        for k, (a_, b_, s_, tk) in enumerate(segs):
            clr = [ST[0], ST[1], ST[3]][k]
            r_ = Rectangle(width=X_(b_) - X_(a_) - 0.03, height=0.42, fill_color=clr, fill_opacity=0.85,
                           stroke_width=0).move_to([(X_(a_) + X_(b_)) / 2, YT, 0])
            rects.add(r_)
            labs.add(tx(s_, 22, WHITE).move_to(r_), tx(tk, 20, clr).next_to(r_, DOWN, buff=0.14))
        axl = VGroup(*[tx(s, 22).move_to([X_(v), YT + 0.6, 0]) for v, s in
                       [(1.0, "<i>t</i> = 1"), (t1, "<i>t</i>" + sub("1") + "*"), (t2, "<i>t</i>" + sub("2") + "*"),
                        (0.0, "<i>t</i> = 0")]])
        self.caption("When to grow: read off the power spectrum.")
        self.play(LaggedStart(*[FadeIn(r) for r in rects], lag_ratio=0.3), run_time=0.9)
        self.add(labs, axl)
        self.add(tx("leave a scale once its highest frequency rises above the noise", 20, DIM)
                 .move_to([0, YT - 1.1, 0]))
        self._hold_caption()
        self.wait(1.2)
        fade_all(self)


# ================================================================== T4 Payoff
class T4_Payoff(Base):
    def construct(self):
        ax = Axes(x_range=[0, 8, 1], y_range=[0.75, 1.15, 0.05], x_length=8.0, y_length=4.5, tips=False,
                  axis_config={"stroke_width": 2, "color": GREY_B, "include_ticks": False}).move_to([-1.3, 0.55, 0])
        ticks = VGroup()
        for v in range(1, 8):
            p = ax.c2p(v, 0.75)
            ticks.add(Line(p + UP * 0.07, p + DOWN * 0.07, stroke_width=2, color=GREY_B),
                      tx(f"{v}×", 20, GREY_B).next_to(p, DOWN, buff=0.12))
        for v in (0.8, 0.9, 1.0, 1.1):
            p = ax.c2p(0, v)
            ticks.add(Line(p + LEFT * 0.07, p + RIGHT * 0.07, stroke_width=2, color=GREY_B),
                      tx(f"{v:.1f}", 20, GREY_B).next_to(p, LEFT, buff=0.12))
        xl = tx("wall-clock speedup vs 50-step FLUX.1-dev", 22).next_to(ax, DOWN, buff=0.45)
        yl = tx("ImageReward", 22).next_to(ax.c2p(0, 1.15), UP, buff=0.12)
        pts = {  # paper Table 1 (FLUX.1-dev, 1024², training-free)
            "flux": [(1.00, 1.095), (4.84, 0.981), (6.62, 0.920)],
            "bot": [(4.67, 0.889), (6.64, 0.792)],
            "ralu": [(1.58, 1.028), (4.98, 1.022), (6.69, 0.999)],
            "ours2": [(1.66, 1.049), (5.77, 1.059), (6.78, 1.039)],
            "ours3": [(6.09, 1.042), (7.09, 1.015)],
        }
        sty = {"flux": (BASE_C, "FLUX, fewer steps"), "bot": (BOT_C, "Bottleneck Sampling"), "ralu": (RALU_C, "RALU"),
               "ours2": (OURS, "Ours, <i>S</i> = 2"), "ours3": (OURS3, "Ours, <i>S</i> = 3")}
        leg = VGroup()
        for key, (c, nm) in sty.items():
            d = Dot(radius=0.08, color=c)
            leg.add(VGroup(d, tx(nm, 22, c).next_to(d, RIGHT, buff=0.15)))
        leg.arrange(DOWN, buff=0.22, aligned_edge=LEFT).move_to([5.3, 1.4, 0])
        dots = {k: VGroup(*[Dot(ax.c2p(x, y), radius=0.085, color=sty[k][0]) for x, y in v]) for k, v in pts.items()}
        fl = VMobject(stroke_color=BASE_C, stroke_width=2).set_points_as_corners([ax.c2p(x, y) for x, y in pts["flux"]])
        self.play(Create(ax), run_time=0.5)
        self.add(ticks, xl, yl, leg, tx("paper Table 1 · training-free", 18, DIM).move_to([5.3, -0.1, 0]))
        self.play(FadeIn(dots["flux"]), Create(DashedVMobject(fl, num_dashes=30)), FadeIn(dots["bot"]),
                  FadeIn(dots["ralu"]), run_time=0.8)
        self.caption("Beats all accelerated baselines, up to 7.09× faster.")
        ol = VMobject(stroke_color=OURS, stroke_width=3).set_points_as_corners([ax.c2p(x, y) for x, y in pts["ours2"]])
        self.play(FadeIn(dots["ours2"]), FadeIn(dots["ours3"]), Create(ol), run_time=0.8)
        self.add(tx("7.09×", 24, OURS3, bold=True).next_to(ax.c2p(7.09, 1.015), RIGHT, buff=0.12))
        self._hold_caption()
        self.wait(1.0)
        fade_all(self)

        keys = ["flux1x", "flux484", "ralu498", "ours577"]
        labs = [("FLUX, 50 steps", "1×"), ("FLUX, 10 steps", "4.84×"), ("RALU", "4.98×"), ("Ours", "5.77×")]
        xs = [-3.75, -1.25, 1.25, 3.75]
        grp = Group()
        for r, subj in enumerate(["buddha", "canyon"]):
            for k, key in enumerate(keys):
                grp.add(img(f"{subj}_{key}.png", 2.2).move_to([xs[k], 0.95 - 2.3 * r, 0]))
        for k, (nm, sp) in enumerate(labs):
            ours = nm == "Ours"
            self.add(tx(nm, 21, OURS if ours else WHITE, bold=ours).move_to([xs[k], 2.78, 0]),
                     tx(sp, 22, OURS if ours else GREY_B, bold=ours).move_to([xs[k], 2.35, 0]))
        self.play(LaggedStart(*[FadeIn(m) for m in grp], lag_ratio=0.06), run_time=1.0)
        self.add(SurroundingRectangle(Group(grp[3], grp[7]), color=OURS, buff=0.06, stroke_width=2.5))
        self.caption("Faster than the baselines, with quality preserved.")
        self.add(tx("paper Fig. 3 · FLUX.1-dev, 1024" + sup("2") + " · wall-clock speedup", 18, DIM).move_to([0, -2.8, 0]))
        fade_all(self)

        h1 = tx("up to 7× faster images   ·   2.5× faster video", 40, FRONT, bold=True).move_to([0, 0.6, 0])
        h2 = tx("training-free on pretrained models, or fine-tuned with a LoRA", 28).move_to([0, -0.25, 0])
        h3 = tx("FLUX.1-dev  ·  Z-Image  ·  PixelGen  ·  Wan 2.1", 24, DIM).move_to([0, -0.95, 0])
        self.add(h1, h2, h3)
        self.wait(3.8)
        self.remove(h1, h2, h3)
        self.wait(0.3)

        t = tx("Spectral Progressive Diffusion", 56, bold=True).move_to([0, 1.0, 0])
        u = tx("arXiv 2605.18736  ·  howardxiao.ca/speed", 28, DIM).move_to([0, 0.0, 0])
        v = tx("Howard Xiao, Brian Chao, Lior Yariv, Gordon Wetzstein · Stanford University", 24, DIM).move_to([0, -0.7, 0])
        self.add(t, u, v)
        self.wait(5.0)
