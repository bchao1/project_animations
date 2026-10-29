"""Spectral Progressive Diffusion — 3b1b-style teaser, v8 (slide-deck clips).

v8: framed() draws nothing (thin image outlines read as white crop borders on the slides).

v7: S06 and S08 load data/*_box.png. These are the clean source images with the paper's yellow and
orange inset boxes drawn back on. The boxes use the same selection, colors and relative width as
make_figure_grid.py in ~/wavelet-diffusion/figure_scripts.
v6: same as v5, but the result images in data/ are now the true source images. They come from the
generation runs and the Wan videos that built the paper figures, not from PDF crops. The old crops
left a thin white gutter line and carried the inset boxes. Backups and the source list are in
data/_pdfcrop_backup_2026-10-14/.
v5: no captions / headers / part cards (CAPTIONS = False); every visual step is its own manim section
(render with --save_sections); each section ends on the finished visual with a 0.3 s hold; S01 no longer
labels attention as quadratic.

Source: Xiao, Chao, Yariv, Wetzstein, "Spectral Progressive Diffusion for Efficient Image and
Video Generation", arXiv 2605.18736 (2026).  Result images in data/ are the source images of the
paper figures (see data/_pdfcrop_backup_2026-10-14/SOURCES.tsv).  All spectra, noise
sweeps and the spectral-expansion demo are computed here with numpy/scipy from data/corgi_clean.png
(the clean corgi of the paper's Fig. 1).

Render (see render.sh):
    manim -ql --save_sections --media_dir media animation_spectral_progressive_v7.py <Scene>
Scenes, in order: S00_Title ... S09_Outlook.
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
DATA = os.path.join(HERE, "data")

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

# v5: the slide deck supplies titles and narration, so captions, "k · title" headers and part cards are off.
CAPTIONS = False
HEADERS = CAPTIONS
PART_CARDS = CAPTIONS
SECTION_HOLD = 0.3     # hold at the end of every section (the slide freezes on the last frame)

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
    # No visible frame: thin outlines around images read as white crop borders on the slides.
    return SurroundingRectangle(m, buff=0, color=color, stroke_width=sw, stroke_opacity=0)


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

    def step(self, name, hold=SECTION_HOLD):
        """End the current visual step (short hold on the finished frame) and start section `name`.
        Called before any fade-out, so the fade belongs to the next section."""
        if self.mobjects and hold:          # nothing on screen yet (scene start) → no hold
            self.wait(hold)
        self.next_section(name)

    def header(self, s):
        if not HEADERS:
            return
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
        if not CAPTIONS:
            return None
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
        if not CAPTIONS:
            return
        self._hold_caption()
        if self.cap is not None:
            self.remove(self.cap)
            self.cap = None

    def title_card(self, num, title, hold=1.8):
        if not PART_CARDS:
            return
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


# ================================================================== S00 Title
class S00_Title(Base):
    def construct(self):
        title = tx("Spectral Progressive Diffusion", 60, bold=True).move_to([0, 2.8, 0])
        subt = tx("for Efficient Image and Video Generation", 32).move_to([0, 2.05, 0])
        auth = tx("Howard Xiao  ·  Brian Chao  ·  Lior Yariv  ·  Gordon Wetzstein", 24, DIM).move_to([0, -2.95, 0])
        inst = tx("Stanford University", 22, DIM).move_to([0, -3.4, 0])
        self.step("title")
        self.add(title, subt, auth, inst)

        lw = ValueTracker(np.log10(3))                     # log10 of the frontier radius
        T = lambda: D.t_of_w(10 ** lw.get_value())
        S = 3.0
        pic = live_img(lambda: D.prog(T())[1], S, nearest=True).move_to([-1.9, -0.45, 0])
        spec = live_img(lambda: D.spec(T(), band=D.n_of(T())), S).move_to([1.9, -0.45, 0])
        fr1, fr2 = framed(pic, GREY_B, 1), framed(spec, GREY_B, 1)
        circ = frontier_circle(lambda: 10 ** lw.get_value(), spec, S)
        self.play(FadeIn(pic), FadeIn(spec), Create(fr1), Create(fr2), run_time=0.8)
        self.add(circ)
        self.play(lw.animate.set_value(np.log10(256)), run_time=6.0, rate_func=linear)
        self.wait(SECTION_HOLD)      # v5: ends on the full-resolution frame (no fade-out)


# ================================================================== S01 Cost
def tokgrid(n, size, center, color=ST[0]):
    c = size / n
    return VGroup(*[RoundedRectangle(width=0.8 * c, height=0.8 * c, corner_radius=0.08 * c,
                                     fill_color=color, fill_opacity=0.85, stroke_width=0)
                    .move_to(np.array(center) + np.array([(j + 0.5) * c - size / 2, size / 2 - (i + 0.5) * c, 0]))
                    for i in range(n) for j in range(n)])


class S01_Cost(Base):
    def construct(self):
        self.title_card("Part 1", "Every step pays full price")
        self.header("1 · Every step pays full price")
        self.step("tokens")
        g1 = tokgrid(8, 2.0, (-4.6, 0.5, 0))
        self.play(LaggedStart(*[FadeIn(m) for m in g1], lag_ratio=0.01), run_time=0.8)
        self.add(tx("<i>N</i> tokens", 26).next_to(g1, DOWN, buff=0.25))
        self.caption("A diffusion transformer denoises a grid of <i>N</i> tokens.")
        self.step("double_res")
        g2 = tokgrid(16, 4.0, (0.3, 0.5, 0))
        a = arrow(g1.get_right() + RIGHT * 0.1, g2.get_left() + LEFT * 0.1)
        self.play(GrowArrow(a), LaggedStart(*[FadeIn(m) for m in g2], lag_ratio=0.002), run_time=1.2)
        self.add(tx("2× resolution  →  4<i>N</i> tokens", 26).next_to(g2, DOWN, buff=0.25))
        # v5: no quadratic-attention claim (the "N² pairs → 16 N² pairs" labels are gone)
        self.caption("Double the resolution: 4× the tokens.")
        self.step("every_step")
        self.clear_all()

        ts = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
        thumbs = Group(*[arr_img(disp(D.xt(tv)), 1.75).move_to([-5.25 + 2.1 * k, 1.2, 0])
                         for k, tv in enumerate(ts)])
        labs = VGroup(*[tx(f"<i>t</i> = {tv:.1f}", 22).next_to(thumbs[k], UP, buff=0.12)
                        for k, tv in enumerate(ts)])
        grids = VGroup(*[tokgrid(12, 1.3, (-5.25 + 2.1 * k, -1.15, 0)) for k in range(6)])
        dn = arrow([-6.2, -0.1, 0], [6.2, -0.1, 0], buff=0, color=GREY_B, sw=2)
        self.play(LaggedStart(*[FadeIn(m) for m in thumbs], lag_ratio=0.15), run_time=1.2)
        self.add(labs)
        self.play(GrowArrow(dn), LaggedStart(*[FadeIn(g) for g in grids], lag_ratio=0.12), run_time=1.2)
        self.add(tx("denoising", 20, GREY_B).next_to(dn, UP, buff=0.04).set_x(5.3),
                 tx("full token grid at every step", 22, ST[0]).next_to(grids, DOWN, buff=0.22))
        self.caption("Dozens of denoising steps, each paying for the full grid.")
        self.step("early_noise")
        box = SurroundingRectangle(Group(thumbs[0], thumbs[1], labs[0], labs[1]), color=NOI, buff=0.1)
        self.play(Create(box), run_time=0.5)
        self.caption("Yet early on it is mostly noise. What is all that resolution computing?")
        self.wait(SECTION_HOLD)


# ================================================================== S02 Spectrum
class S02_Spectrum(Base):
    def construct(self):
        self.title_card("Part 2", "Images are mostly low frequency")
        self.header("2 · Images are mostly low frequency")
        self.step("fft")
        S = 3.0
        pic = arr_img(disp(D.x0), S).move_to([-5.3, 0.5, 0])
        spec = arr_img(D.spec_clean(), S).move_to([-1.35, 0.5, 0])
        f1, f2 = framed(pic, GREY_B, 1), framed(spec, GREY_B, 1)
        self.play(FadeIn(pic), Create(f1), run_time=0.5)
        a = arrow(pic.get_right(), spec.get_left(), buff=0.08)
        self.play(GrowArrow(a), run_time=0.3)
        self.add(tx("FFT", 20).next_to(a, UP, buff=0.08))
        self.play(FadeIn(spec), Create(f2), run_time=0.7)
        self.add(tx("centre: low |" + OM + "|   ·   edges: high |" + OM + "|", 20, DIM).next_to(spec, DOWN, buff=0.2))
        self.caption("The 2D Fourier transform splits an image into frequencies <i>ω</i>.")
        self.step("radial_psd")

        ax, ticks, alab = log_axes([3.95, 0.45, 0], 4.8, 3.4, PSD_XR, PSD_YR, PSD_XT, PSD_YT,
                                   "|" + OM + "|", PW)
        self.play(Create(ax), run_time=0.5)
        self.add(ticks, alab)
        rho = ValueTracker(0.0)
        ring = frontier_circle(lambda: 10 ** rho.get_value(), spec, S)
        lw, lp = np.log10(D.w), np.log10(D.pw)
        curve = always_redraw(lambda: poly(ax, lw[lw <= rho.get_value() + 1e-9],
                                           lp[lw <= rho.get_value() + 1e-9], SIG, 3))
        self.add(ring, curve)
        self.caption("Average the power over rings of radius |<i>ω</i>|: the spectrum " + PW + ".")
        self.play(rho.animate.set_value(np.log10(256)), run_time=3.5, rate_func=linear)
        self.remove(ring)
        curve.clear_updaters()
        self.step("power_law")
        fit = DashedVMobject(poly(ax, [0, np.log10(256)], [np.log10(D.A), np.log10(D.A) - D.beta * np.log10(256)],
                                  WHITE, 2), num_dashes=30)
        self.play(Create(fit), run_time=0.5)
        fl = VGroup(tx(PW + " ∝ |" + OM + "|" + sup("−<i>β</i>"), 26),
                    tx(f"<i>β</i> ≈ {D.beta:.1f} for this image", 20)).arrange(DOWN, buff=0.12)
        fl.move_to(ax.p(1.62, 2.9))
        self.add(fl)
        self.add(VGroup(tx("paper’s fits:  <i>β</i> = 1.92 (FLUX latents),", 18, DIM),
                        tx("2.42 (Wan latents),  2.45 (pixels)", 18, DIM)).arrange(DOWN, buff=0.08)
                 .move_to([3.95, -2.05, 0]))
        self.caption("Natural images follow a power law: power falls steeply with frequency.")
        self.step("lowpass_low")

        R = ValueTracker(np.log10(3))
        lp_img = live_img(lambda: disp(D.lowpass(10 ** R.get_value())), S).move_to(pic)
        self.remove(pic)
        self.add(lp_img)
        self.add(frontier_circle(lambda: 10 ** R.get_value(), spec, S))
        self.add(always_redraw(lambda: DashedLine(ax.p(R.get_value(), PSD_YR[0]), ax.p(R.get_value(), PSD_YR[1]),
                                                  color=FRONT, stroke_width=2, dash_length=0.08)))
        self.add(tx("only |" + OM + "| inside the circle", 20, FRONT).next_to(lp_img, DOWN, buff=0.2))
        self.caption("Low frequencies carry the layout and colour...")
        self.play(R.animate.set_value(np.log10(24)), run_time=2.5, rate_func=smooth)
        self.step("lowpass_high")
        self.caption("...high frequencies add fur and edges, with very little power.")
        self.play(R.animate.set_value(np.log10(256)), run_time=3.0, rate_func=smooth)
        self.wait(SECTION_HOLD)


# ================================================================== S03 Noise
class S03_Noise(Base):
    def construct(self):
        self.title_card("Part 3", "Noise drowns the high frequencies first")
        self.header("3 · Noise drowns the high frequencies first")
        self.step("flow_matching")
        S = 3.0
        t = ValueTracker(0.0)
        eq = tx(XT + " = (1 − <i>t</i>) " + X0 + " + <i>t</i> " + EPS + ",     " + EPS + " ∼ 𝒩(0, <i>I</i>)", 34)
        eq.move_to([0, 2.85, 0])
        self.add(eq)
        pic = live_img(lambda: disp(D.xt(t.get_value())), S).move_to([-5.3, 0.3, 0])
        spec = live_img(lambda: D.spec(t.get_value()), S).move_to([-1.55, 0.3, 0])
        f1, f2 = framed(pic, GREY_B, 1), framed(spec, GREY_B, 1)
        tl = always_redraw(lambda: tx(f"<i>t</i> = {t.get_value():.2f}", 26).next_to(f1, DOWN, buff=0.2))
        self.play(FadeIn(pic), FadeIn(spec), Create(f1), Create(f2), run_time=0.6)
        self.add(tl)
        ax, ticks, alab = log_axes([4.0, 0.3, 0], 4.6, 3.4, PSD_XR, PSD_YR, PSD_XT, PSD_YT,
                                   "|" + OM + "|", "power")
        self.play(Create(ax), run_time=0.5)
        self.add(ticks, alab)
        lw, lp = np.log10(D.w), np.log10(D.pw)
        xr1 = PSD_XR[1]

        def tc():
            return float(np.clip(t.get_value(), 0.02, 0.995))

        sig = always_redraw(lambda: poly(ax, lw, lp + 2 * np.log10(1 - tc()), SIG, 3, ymin=PSD_YR[0]))
        sl = tx("signal  (1 − <i>t</i>)" + sup("2") + " " + PW, 22, SIG).move_to([4.45, 2.25, 0])
        self.add(sig, sl)
        self.caption("Flow matching: <i>t</i> = 0 is the image, <i>t</i> = 1 is pure Gaussian noise.")
        self.step("add_noise")       # v5: replaces the 2.5 s reading hold on the clean spectrum

        noi = always_redraw(lambda: poly(ax, [0, xr1], [2 * np.log10(tc())] * 2, NOI, 3))
        shade = always_redraw(lambda: Rectangle(
            width=max(ax.p(xr1, 0)[0] - ax.p(np.log10(D.wstar(tc())), 0)[0], 0.001),
            height=ax.y_length, stroke_width=0, fill_color=NOI, fill_opacity=0.14)
            .align_to(ax.p(xr1, PSD_YR[0]), DR))
        dot = always_redraw(lambda: Dot(ax.p(np.log10(D.wstar(tc())), 2 * np.log10(tc())), radius=0.07,
                                        color=FRONT) if D.wstar(tc()) < 256 else VGroup())
        circ = frontier_circle(lambda: D.wstar(tc()), spec, S)
        nl = tx("noise  <i>t</i>" + sup("2"), 22, NOI).move_to([5.75, 1.6, 0])
        self.add(noi, nl, shade, dot, circ)
        self.caption("White noise adds equal power <i>t</i>" + sup("2") + " at every frequency: a flat line.")
        self.play(t.animate.set_value(0.97), run_time=4.5, rate_func=smooth)
        self.step("snr")
        snr = tx("SNR" + sub(OM) + "(<i>t</i>) = (1 − <i>t</i>)" + sup("2") + " / <i>t</i>" + sup("2") + "  · " + PW, 26)
        snr.move_to([4.0, -2.2, 0])
        self.add(snr)
        self.caption("Where the signal sinks below the noise (red), it is lost. Highs drown first.")
        self.step("sampler_sweep")
        lwt = ValueTracker(np.log10(D.wstar(0.97)))
        t.add_updater(lambda m: m.set_value(D.t_of_w(10 ** lwt.get_value())))
        self.add(t)
        self.caption("Now run the sampler from <i>t</i> = 1 to 0: the frontier sweeps outward.")
        self.play(lwt.animate.set_value(np.log10(256)), run_time=7.0, rate_func=linear)
        t.clear_updaters()
        self.remove(t)
        self.step("autoregression")
        self.caption("Coarse first, fine last: diffusion is autoregression in frequency.")
        self.add(tx("Rissanen et al. 2023 · Dieleman 2024", 20, DIM).move_to([-3.4, -2.2, 0]))
        self.wait(SECTION_HOLD)


# ================================================================== S04 Frontier
class S04_Frontier(Base):
    def construct(self):
        self.title_card("Part 4", "Grow the grid with the frontier")
        self.header("4 · Grow the grid with the frontier")
        self.step("frontier")
        S = 3.6
        lwt = ValueTracker(np.log10(D.wstar(0.9)))
        T = lambda: D.t_of_w(10 ** lwt.get_value())
        spec = live_img(lambda: D.spec(T(), band=D.n_of(T())), S).move_to([-3.3, 0.35, 0])
        f1 = framed(spec, GREY_B, 1)
        sq = VGroup(*[DashedVMobject(Square(n / 512 * S), num_dashes=int(8 + n / 12)).set_stroke(GREY_B, 1.5, 0.6)
                      .move_to(spec) for n in (32, 64, 128, 256)])
        circ = frontier_circle(lambda: 10 ** lwt.get_value(), spec, S)
        band = always_redraw(lambda: Square(D.n_of(T()) / 512 * S).set_stroke(WHITE, 3).move_to(spec))
        self.play(FadeIn(spec), Create(f1), run_time=0.5)
        self.add(tx("Fourier plane (|" + OM + "| up to 256)", 22, DIM).next_to(f1, UP, buff=0.15), circ)
        self.caption("Early on (<i>t</i> ≈ 0.9), only frequencies inside the yellow frontier carry signal.")
        self.step("nyquist")
        self.play(Create(sq), run_time=0.8)
        self.add(band)
        self.caption("An <i>n</i> × <i>n</i> grid holds frequencies up to <i>n</i>/2 (Nyquist): the squares.")
        self.step("grow_grid")

        pix = live_img(lambda: D.prog(T())[1], S, nearest=True).move_to([2.6, 0.35, 0])
        f2 = framed(pix, GREY_B, 1)
        glab = always_redraw(lambda: tx(f"{D.n_of(T())} × {D.n_of(T())} grid,  <i>t</i> = {T():.2f}", 24)
                             .next_to(f2, UP, buff=0.15))
        FR = {32: "1/256 of the", 64: "1/64 of the", 128: "1/16 of the", 256: "1/4 of the", 512: "all"}
        frac = always_redraw(lambda: tx(FR[D.n_of(T())] + " full-resolution pixels", 22, FRONT)
                             .next_to(f2, DOWN, buff=0.2))
        self.play(FadeIn(pix), Create(f2), run_time=0.5)
        self.add(glab, frac)
        self.caption("So use the smallest grid that contains the frontier, and grow it.")
        self.play(lwt.animate.set_value(np.log10(256)), run_time=8.0, rate_func=linear)
        self.caption("High-resolution compute on noise-dominated frequencies is wasted.")
        self.wait(SECTION_HOLD)
        if PART_CARDS:               # v5: the "→ Part 5 / Part 6" closing card is a part card; skipped
            self.clear_all()
            q1 = tx("How do we grow without breaking the noise?  →  Part 5", 26, DIM)
            q2 = tx("When exactly should we grow?  →  Part 6", 26, DIM)
            VGroup(q1, q2).arrange(DOWN, buff=0.35, aligned_edge=LEFT).move_to([0, 0.3, 0])
            self.add(q1, q2)
            self.wait(2.6)
            self.remove(q1, q2)


# ================================================================== S05 Expansion
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


class S05_Expansion(Base):
    def construct(self):
        self.title_card("Part 5", "Spectral noise expansion")
        self.header("5 · Spectral noise expansion")
        self.step("dct")
        E = expansion_demo()
        A = arr_img(disp(E["xt_low"]), 1.9, True).move_to([-5.55, 0.8, 0])
        B = arr_img(E["C_low"], 1.9, True).move_to([-2.9, 0.8, 0])
        ab = arrow(A.get_right(), B.get_left(), buff=0.08)
        self.play(FadeIn(A), Create(framed(A, GREY_B, 1)), run_time=0.5)
        self.add(tx(X(TI) + " at 64 × 64", 22).next_to(A, UP, buff=0.15))
        self.play(GrowArrow(ab), run_time=0.3)
        self.add(tx("<i>T</i>" + sub("Φ"), 24).next_to(ab, UP, buff=0.06))
        self.play(FadeIn(B), Create(framed(B, GREY_B, 1)), run_time=0.5)
        self.add(tx("DCT coefficients", 22).next_to(B, UP, buff=0.15),
                 tx("(i) transform", 22, DIM).move_to([-4.2, -0.95, 0]))
        self.caption("To grow mid-trajectory, take the low-res state " + X(TI) + " into the DCT domain.")
        self.step("embed")

        C_full = arr_img(E["C"], 2.9, True).move_to([0.75, 0.8, 0])
        fc = framed(C_full, GREY_B, 1)
        ac = arrow(B.get_right(), C_full.get_left(), buff=0.08)
        self.play(GrowArrow(ac), Create(fc), run_time=0.4)
        Bc = B.copy()
        self.play(Bc.animate.scale_to_fit_height(1.45).move_to(C_full.get_corner(UL) + np.array([0.725, -0.725, 0])),
                  run_time=0.9)
        self.add(tx("128 × 128 coefficients", 22).next_to(fc, UP, buff=0.15))
        self.caption("Embed it as the low-frequency corner of a spectrum twice as large...")
        self.step("pad_noise")
        self.play(FadeIn(C_full), run_time=1.0)
        self.remove(Bc)
        noise_lab = tx(TI + " " + EPS, 26, FRONT)
        self.add(VGroup(BackgroundRectangle(noise_lab, fill_opacity=0.75, buff=0.08), noise_lab)
                 .move_to(C_full.get_center() + np.array([0.725, -0.725, 0])))
        self.add(tx("(ii) embed + pad with " + TI + " " + EPS, 22, DIM).move_to([0.75, -0.95, 0]))
        Dm = arr_img(disp(E["x_exp"]), 2.9, True).move_to([4.95, 0.8, 0])
        ad = arrow(C_full.get_right(), Dm.get_left(), buff=0.08)
        self.caption("...fill the new slots with noise at the current level " + TI + ", and invert.")
        self.step("invert")
        self.play(GrowArrow(ad), run_time=0.3)
        self.add(tx("<i>T</i>" + sub("Φ") + sup("−1"), 24).next_to(ad, UP, buff=0.06))
        self.play(FadeIn(Dm), Create(framed(Dm, GREY_B, 1)), run_time=0.6)
        self.add(tx(X(TI) + " at 128 × 128", 22).next_to(Dm, UP, buff=0.15),
                 tx("(iii) invert", 22, DIM).move_to([4.95, -0.95, 0]))
        self.caption("Twice the resolution; the partially denoised low frequencies are untouched.")
        self.step("align")
        self.clear_all()

        # timestep alignment
        e1 = tx("<i>x̃</i> = <i>κ</i> · " + X(TI) + ",       " + TTI + " = <i>κ</i> · " + TI, 34)
        e2 = tx("<i>κ</i> = <i>r</i> / (1 + (<i>r</i> − 1) " + TI + "),       <i>r</i> = <i>s</i>" + sub("<i>i</i>+1") +
                " / <i>s</i>" + sub("<i>i</i>"), 30)
        ex = tx("e.g.  <i>r</i> = 2,  " + TI + " = 1/3   →   <i>κ</i> = " + f"{E['kappa']:.2f}" + ",   " + TTI +
                f" = {E['t_new']:.2f}", 28, FRONT)
        VGroup(e1, e2, ex).arrange(DOWN, buff=0.32).move_to([0, 1.95, 0])
        self.add(e1, e2)
        self.caption("Padding raises the noise level, so rescale by <i>κ</i> and re-index the time.")
        self.add(ex)
        self.step("compare")
        ims = [(E["x_al"], "expanded + aligned", FRONT), (E["x_true"], "a true " + X(TTI) + " at 128 × 128", WHITE),
               (E["x_bil"], "bilinear upsampling instead", DIM)]
        grp = Group()
        for k, (a_, lab, c) in enumerate(ims):
            m = arr_img(disp(a_), 2.0, True).move_to([-3.6 + 3.6 * k, -1.05, 0])
            grp.add(Group(m, framed(m, GREY_B, 1), tx(lab, 22, c).next_to(m, DOWN, buff=0.15)))
        self.play(LaggedStart(*[FadeIn(g) for g in grp], lag_ratio=0.4), run_time=1.4)
        self.caption("Demo: it matches a genuine noisy state; bilinear upsampling smears the noise.")
        self.wait(SECTION_HOLD)


# ================================================================== S06 Schedule
class S06_Schedule(Base):
    def construct(self):
        self.title_card("Part 6", "When to grow: a schedule from the spectrum")
        self.header("6 · When to grow: a schedule from the spectrum")
        self.step("prop1")
        p1 = tx("Prop. 1:  frequency " + OM + " is noise-dominated while  𝔼 | <i>v</i>*(" + OM + ") − " + EPS + "(" + OM +
                ") |" + sup("2") + " ≤ " + DEL, 26)
        p1.move_to([0, 2.8, 0])
        f1 = tx("<i>t</i>" + sub(OM) + " = 1 / [ 1 + ( " + DEL + " / (" + PW + " (1 + " + PW + " − " + DEL + ")) )" +
                sup("1/2") + " ]", 32, FRONT)
        f1.move_to([0, 2.1, 0])
        self.add(p1)
        self.caption("Prop. 1: when does a frequency stop being pure noise for the ideal model?")
        self.add(f1)
        self.step("t_omega_curve")

        delta = ValueTracker(0.01)
        ax = Axes(x_range=[0, np.log10(256), 1], y_range=[0, 1, 0.25], x_length=5.6, y_length=2.9, tips=False,
                  axis_config={"stroke_width": 2, "color": GREY_B, "include_ticks": False}).move_to([-2.6, -0.55, 0])
        ticks = VGroup()
        for v, s in [(0, "1"), (1, "10"), (2, "100")]:
            p = ax.c2p(v, 0)
            ticks.add(Line(p + UP * 0.07, p + DOWN * 0.07, stroke_width=2, color=GREY_B),
                      tx(s, 20, GREY_B).next_to(p, DOWN, buff=0.12))
        for v in (0, 0.5, 1):
            p = ax.c2p(0, v)
            ticks.add(Line(p + LEFT * 0.07, p + RIGHT * 0.07, stroke_width=2, color=GREY_B),
                      tx(f"{v:g}", 20, GREY_B).next_to(p, LEFT, buff=0.12))
        xl = tx("|" + OM + "|", 24).next_to(ax.c2p(np.log10(256), 0), DOWN, buff=0.12).shift(RIGHT * 0.3)
        yl = tx("<i>t</i>" + sub(OM), 24).next_to(ax.c2p(0, 1), UP, buff=0.1)
        self.play(Create(ax), run_time=0.4)
        self.add(ticks, xl, yl)
        lws = np.linspace(0, np.log10(256), 200)
        curve = always_redraw(lambda: poly(ax, lws, t_omega(10 ** lws, delta.get_value()), FRONT, 3))
        self.play(Create(curve), run_time=0.8)
        self.add(tx("curve: paper’s pixel-space fit  " + PW + " ≈ 3745 |" + OM + "|" + sup("−2.45"), 18, DIM)
                 .next_to(ax, DOWN, buff=0.5))
        self.caption("Closed form from the power " + PW + ": weaker frequencies activate later.")
        self.step("prop2")

        marks = VGroup()
        for k, (w, sh) in enumerate(((64, LEFT * 0.28), (128, RIGHT * 0.28))):
            lx = np.log10(w)
            marks.add(DashedLine(ax.c2p(lx, 0), ax.c2p(lx, 1), color=ST[k], stroke_width=2, dash_length=0.07),
                      tx(f"<i>s</i>{sub(str(k + 1))}<i>ω</i>{sub('max')}", 20, ST[k])
                      .next_to(ax.c2p(lx, 1), UP, buff=0.08).shift(sh))
        self.play(Create(marks), run_time=0.5)

        def dotline(w, k):
            def f():
                y = float(t_omega(w, delta.get_value()))
                p = ax.c2p(np.log10(w), y)
                return VGroup(DashedLine(ax.c2p(0, y), p, color=ST[k], stroke_width=2, dash_length=0.06),
                              Dot(p, radius=0.07, color=ST[k]),
                              tx(f"<i>t</i>{sub(str(k + 1))}*", 22, ST[k]).next_to(ax.c2p(0, y), RIGHT, buff=0.08)
                              .shift(UP * 0.16))
            return always_redraw(f)
        self.add(dotline(64, 0), dotline(128, 1))
        p2 = VGroup(tx("Prop. 2:  leave scale <i>s</i>" + sub("<i>i</i>") + " when its", 24),
                    tx("highest frequency activates", 24),
                    tx("<i>t</i>" + sub("<i>i</i>") + "* = <i>t</i>" + sub(OM) + "  at  " + OM + " = <i>s</i>" +
                       sub("<i>i</i>") + " · <i>ω</i>" + sub("max"), 28, FRONT),
                    tx("<i>ω</i>" + sub("max") + " = min(<i>H</i>, <i>W</i>) / 2   (Nyquist)", 22, DIM))
        p2.arrange(DOWN, buff=0.2, aligned_edge=LEFT).move_to([4.2, -0.3, 0])
        self.add(p2)
        self.caption("Prop. 2: leave a scale once its highest representable frequency activates.")
        self.step("delta_knob")
        self.add(always_redraw(lambda: tx(DEL + f" = {delta.get_value():.3f}", 26, FRONT).move_to(ax.c2p(0.55, 0.22))))
        self.caption("One knob, the tolerance " + DEL + ": larger " + DEL + " means later growth, more speed.")
        self.play(delta.animate.set_value(0.1), run_time=1.8)
        self.play(delta.animate.set_value(0.001), run_time=1.8)
        self.play(delta.animate.set_value(0.01), run_time=1.0)
        self.step("timeline")
        self.clear_all()

        # timeline + evidence
        t1, t2 = float(t_omega(64, 0.01)), float(t_omega(128, 0.01))
        X_ = lambda tv: -6.0 + (1 - tv) * 12.0
        segs = [(1.0, t1, "<i>s</i>" + sub("1") + " = 1/4", "1/16 of the tokens"),
                (t1, t2, "<i>s</i>" + sub("2") + " = 1/2", "1/4"),
                (t2, 0.0, "<i>s</i>" + sub("3") + " = 1", "all tokens")]
        rects, labs = VGroup(), VGroup()
        for k, (a_, b_, s_, tk) in enumerate(segs):
            clr = [ST[0], ST[1], ST[3]][k]
            r_ = Rectangle(width=X_(b_) - X_(a_) - 0.03, height=0.42, fill_color=clr, fill_opacity=0.85,
                           stroke_width=0).move_to([(X_(a_) + X_(b_)) / 2, 1.9, 0])
            rects.add(r_)
            labs.add(tx(s_, 22, WHITE).move_to(r_), tx(tk, 20, clr).next_to(r_, DOWN, buff=0.14))
        axl = VGroup(*[tx(s, 22).move_to([X_(v), 2.55, 0]) for v, s in
                       [(1.0, "<i>t</i> = 1"), (t1, "<i>t</i>" + sub("1") + "*"), (t2, "<i>t</i>" + sub("2") + "*"),
                        (0.0, "<i>t</i> = 0")]])
        self.play(LaggedStart(*[FadeIn(r) for r in rects], lag_ratio=0.3), run_time=0.9)
        self.add(labs, axl)
        self.caption("A resolution schedule; " + DEL + " is independent of the number of stages (default 0.01).")
        self.step("passthrough")

        names = ["full", "d0001", "d001", "d01", "d05", "d1"]
        blabs = ["full res", DEL + " = 0.0001", DEL + " = 0.001", DEL + " = 0.01", DEL + " = 0.05", DEL + " = 0.1"]
        row = Group()
        for k, (nm, lb) in enumerate(zip(names, blabs)):
            m = img(f"pt_bus_{nm}_box.png", 1.95).move_to([-5.25 + 2.1 * k, -1.15, 0])
            row.add(m)
            self.add(tx(lb, 20, FRONT if nm == "d01" else WHITE).next_to(m, UP, buff=0.1))
        self.play(LaggedStart(*[FadeIn(m) for m in row], lag_ratio=0.12), run_time=1.0)
        self.caption("Paper Fig. 5: faking inactive frequencies with noise is harmless for small " + DEL +
                     ", blurry by 0.05.")
        self.wait(SECTION_HOLD)


# ================================================================== S07 Fine-tuning
class S07_FineTune(Base):
    def construct(self):
        self.title_card("Part 7", "Training-free, or fine-tuned")
        self.header("7 · Training-free, or fine-tuned")
        self.step("training_free")
        t1, t2 = float(t_omega(64, 0.01)), float(t_omega(128, 0.01))
        tt1, tt2 = t_align(t1), t_align(t2)
        X_ = lambda tv: -5.0 + (1 - tv) * 5.9
        ly = [1.8, 0.6, -0.6]
        cl = [ST[0], ST[1], ST[3]]
        lanes = VGroup(*[DashedLine([-5.2, y, 0], [1.1, y, 0], color=GHOST, stroke_width=1.5) for y in ly])
        lane_l = VGroup(*[tx(f"<i>s</i>{sub(str(k + 1))}", 24, cl[k]).move_to([-5.7, y, 0]) for k, y in enumerate(ly)])
        tax = VGroup(tx("<i>t</i> = 1", 22, GREY_B).move_to([X_(1), -1.25, 0]),
                     tx("<i>t</i> = 0", 22, GREY_B).move_to([X_(0), -1.25, 0]),
                     arrow([X_(1) + 0.5, -1.25, 0], [X_(0) - 0.5, -1.25, 0], buff=0, color=GREY_B, sw=2))
        self.add(lanes, lane_l, tax)
        segs = [(1.0, t1, 0), (tt1, t2, 1), (tt2, 0.0, 2)]
        paths = VGroup(*[Line([X_(a_), ly[k], 0], [X_(b_), ly[k], 0], color=cl[k], stroke_width=6) for a_, b_, k in segs])
        jumps = VGroup(arrow([X_(t1), ly[0], 0], [X_(tt1), ly[1], 0], buff=0.05, sw=2),
                       arrow([X_(t2), ly[1], 0], [X_(tt2), ly[2], 0], buff=0.05, sw=2))
        self.play(Create(paths[0]), run_time=0.5)
        self.play(Create(jumps[0]), run_time=0.3)
        self.play(Create(paths[1]), run_time=0.5)
        self.play(Create(jumps[1]), run_time=0.3)
        self.play(Create(paths[2]), run_time=0.7)
        self.add(tx("expand + align: time steps back to " + TTI, 20).move_to([X_(tt1) + 2.2, 2.4, 0]))
        self.caption("Training-free, the pretrained model must handle every intermediate scale itself.")
        self.step("finetune")
        dots = VGroup(Dot([X_(tt1), ly[1], 0], color=WHITE, radius=0.07), Dot([X_(t2), ly[1], 0], color=WHITE, radius=0.07))
        self.add(dots, tx("<i>x̃</i>" + sub("<i>t̃</i>" + sub("<i>i</i>−1")), 22).next_to(dots[0], DL, buff=0.06),
                 tx(X(TI), 22).next_to(dots[1], UR, buff=0.06))
        eqs = VGroup(tx("fine-tune each stage on a straight path:", 22, DIM),
                     tx(X(TI) + " = (1 − " + TI + ") " + X0 + " + " + TI + " " + EPS, 26),
                     tx("<i>v</i>" + sup("<i>s</i>" + sub("<i>i</i>")) + " = (<i>x̃</i>" + sub("<i>t̃</i>" + sub("<i>i</i>−1")) +
                        " − " + X(TI) + ") / (<i>t̃</i>" + sub("<i>i</i>−1") + " − " + TI + ")", 26, FRONT),
                     tx("ℒ = 𝔼‖<i>v</i>" + sub("<i>θ</i>") + "(" + XT + ", <i>t</i>) − <i>v</i>" + sup("<i>s</i>" + sub("<i>i</i>")) +
                        "‖" + sup("2"), 26))
        eqs.arrange(DOWN, buff=0.28, aligned_edge=LEFT)
        if eqs.width > 5.2:
            eqs.scale_to_fit_width(5.2)
        eqs.move_to([4.3, 0.55, 0])
        self.add(eqs)
        self.caption("Fine-tuning: each stage learns a straight path from the expanded state to " + X(TI) + ".")
        self.step("lora")
        self.add(tx("LoRA rank 32 · 2000 iterations · 5K synthetic images · ≈ 2 h on 8 H100s", 22, LORA)
                 .move_to([0, -2.2, 0]))
        self.caption("Cheap: a LoRA trained on images generated by the base model itself.")
        self.step("imagereward")
        self.clear_all()

        groups = [("Z-Image, 1024" + sup("2"), [("50 steps", "1.00×", 0.965, BASE_C), ("training-free", "1.65×", 0.904, OURS),
                                                ("+ LoRA", "1.65×", 0.982, LORA)]),
                  ("PixelGen, 512" + sup("2"), [("25 steps", "1.00×", 0.921, BASE_C), ("training-free", "1.60×", 0.799, OURS),
                                                ("+ LoRA", "1.55×", 0.913, LORA)])]
        base_y, k_h = -1.6, 11.0
        for gi, (gname, bars) in enumerate(groups):
            cx = -3.4 + 6.8 * gi
            self.add(tx(gname, 28, bold=True).move_to([cx, 2.55, 0]),
                     Line([cx - 2.6, base_y, 0], [cx + 2.6, base_y, 0], color=GREY_B, stroke_width=2))
            anims = []
            for bi, (nm, sp, v, c) in enumerate(bars):
                x = cx - 1.7 + 1.7 * bi
                h = (v - 0.7) * k_h
                r_ = Rectangle(width=1.1, height=h, fill_color=c, fill_opacity=0.9, stroke_width=0)
                r_.move_to([x, base_y + h / 2, 0])
                anims.append(GrowFromEdge(r_, DOWN))
                self.add(tx(f"{v:.3f}", 22).move_to([x, base_y + h + 0.22, 0]),
                         tx(nm, 20, DIM).move_to([x, base_y - 0.3, 0]), tx(sp, 22, c).move_to([x, base_y - 0.65, 0]))
            self.play(*anims, run_time=0.7)
        self.add(tx("ImageReward (axis starts at 0.70) · paper Table 2", 20, DIM).move_to([0, -2.75, 0]))
        self.caption("Same speedup, better quality: LoRA lifts ImageReward back to (or near) the full model.")
        self.wait(SECTION_HOLD)


# ================================================================== S08 Results
class S08_Results(Base):
    def construct(self):
        self.title_card("Part 8", "Results")
        self.header("8 · Results")
        self.step("flux_steps")
        ax = Axes(x_range=[0, 8, 1], y_range=[0.75, 1.15, 0.05], x_length=8.0, y_length=4.5, tips=False,
                  axis_config={"stroke_width": 2, "color": GREY_B, "include_ticks": False}).move_to([-1.3, 0.45, 0])
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
        self.play(Create(ax), run_time=0.5)
        self.add(ticks, xl, yl)
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
        leg.arrange(DOWN, buff=0.22, aligned_edge=LEFT).move_to([5.3, 1.3, 0])
        self.add(leg, tx("paper Table 1 · training-free", 18, DIM).move_to([5.3, -0.2, 0]))
        dots = {k: VGroup(*[Dot(ax.c2p(x, y), radius=0.085, color=sty[k][0]) for x, y in v]) for k, v in pts.items()}
        fl = VMobject(stroke_color=BASE_C, stroke_width=2).set_points_as_corners([ax.c2p(x, y) for x, y in pts["flux"]])
        self.play(FadeIn(dots["flux"]), Create(DashedVMobject(fl, num_dashes=30)), run_time=0.7)
        self.caption("FLUX.1-dev at 1024" + sup("2") + ": fewer sampling steps is faster, but worse.")
        self.step("baselines")
        self.play(FadeIn(dots["bot"]), FadeIn(dots["ralu"]), run_time=0.5)
        self.caption("Two training-free progressive-resolution baselines, in three speed tiers.")
        self.step("ours")
        ol = VMobject(stroke_color=OURS, stroke_width=3).set_points_as_corners([ax.c2p(x, y) for x, y in pts["ours2"]])
        self.play(FadeIn(dots["ours2"]), FadeIn(dots["ours3"]), Create(ol), run_time=0.7)
        for (x, y), s, d in zip(pts["ours2"], ["1.66×", "5.77×", "6.78×"], [UL, UP, UR]):
            self.add(tx(s, 20, OURS).next_to(ax.c2p(x, y), d, buff=0.1))
        self.add(tx("7.09×", 20, OURS3).next_to(ax.c2p(7.09, 1.015), RIGHT, buff=0.12))
        self.caption("Spectral Progressive Diffusion wins every accelerated tier, up to 7.09× faster.")
        self.step("fig3_flux")
        self.clear_all()

        keys = ["flux1x", "ralu158", "ours166", "flux484", "ralu498", "ours577"]
        labs = ["FLUX (1×)", "RALU (1.58×)", "Ours (1.66×)", "FLUX (4.84×)", "RALU (4.98×)", "Ours (5.77×)"]
        grp = Group()
        for r, subj in enumerate(["buddha", "canyon"]):
            for k, key in enumerate(keys):
                grp.add(img(f"{subj}_{key}_box.png", 2.1).move_to([-5.5 + 2.2 * k, 1.2 - 2.3 * r, 0]))
        for k, lb in enumerate(labs):
            self.add(tx(lb, 20, OURS if "Ours" in lb else WHITE, bold="Ours" in lb).move_to([-5.5 + 2.2 * k, 2.52, 0]))
        sep = DashedLine([0.0, 2.7, 0], [0.0, -2.3, 0], color=GREY_B, stroke_width=1.5)
        self.play(LaggedStart(*[FadeIn(m) for m in grp], lag_ratio=0.05), Create(sep), run_time=1.3)
        self.caption("Paper Fig. 3: few-step FLUX oversaturates, RALU adds noise artifacts.")
        self.step("fig3_pixelgen")
        self.clear_all()

        grp = Group()
        for r, subj in enumerate(["city", "aurora"]):
            for k, key in enumerate(["pixelgen1x", "ours16"]):
                grp.add(img(f"{subj}_{key}_box.png", 2.4).move_to([-1.25 + 2.5 * k, 1.05 - 2.5 * r, 0]))
        self.add(tx("PixelGen (1×)", 24).move_to([-1.25, 2.6, 0]),
                 tx("Ours (1.6×)", 24, OURS, bold=True).move_to([1.25, 2.6, 0]))
        self.play(LaggedStart(*[FadeIn(m) for m in grp], lag_ratio=0.1), run_time=0.8)
        self.caption("Pixel space works too: PixelGen at 512" + sup("2") + ", 1.6× faster, training-free.")
        self.step("fig3_video")
        self.clear_all()

        rows = [("wan1x", "Wan 2.1, 50 steps (1×)", WHITE), ("wan2x", "25 steps (2.0×)", WHITE),
                ("ours203", "Ours, <i>S</i> = 2 (2.03×)", OURS)]
        grp = Group()
        for r, (key, lb, c) in enumerate(rows):
            y = 1.75 - 1.35 * r
            for k in range(4):
                grp.add(img(f"coral_{key}_{k}.png", h=1.25).move_to([-2.0 + 2.22 * k, y, 0]))
            self.add(tx(lb, 22, c, bold=(key == "ours203")).move_to([-3.25, y, 0]).align_to([-3.3, 0, 0], RIGHT))
        self.play(LaggedStart(*[FadeIn(m) for m in grp], lag_ratio=0.04), run_time=1.2)
        self.add(tx("VBench image quality: 0.6133 (1×)  ·  0.5706 (2.0×)  ·  " + col("0.6114 (2.03×)", OURS) +
                    "   ·   <i>S</i> = 3: 2.54×", 22).move_to([0, -2.45, 0]))
        self.caption("Video, Wan 2.1 at 720P: halving the steps loses detail; the schedule keeps it.")
        self.wait(SECTION_HOLD)


# ================================================================== S09 Outlook
class S09_Outlook(Base):
    def construct(self):
        self.title_card("Part 9", "Takeaways")
        self.header("9 · Takeaways")
        self.step("editing")
        ed = img("fig4b_editing.png", w=10.6).move_to([0, 0.35, 0])
        self.play(FadeIn(ed), run_time=0.7)
        self.caption("Bonus: frequency-based editing re-noises only low frequencies; it beats SDEdit (Fig. 4b).")
        self.step("limitations")
        self.clear_all()

        lim = VGroup(tx("Limitations and outlook", 34, bold=True),
                     tx("•  Training-free use assumes the model generalizes across resolutions", 26),
                     tx("•  Fine-tuning closes that gap; pretraining with the schedule is a natural next step", 26),
                     tx("•  Only spatial frequencies so far; temporal frequencies could speed up video further", 26))
        lim.arrange(DOWN, buff=0.4, aligned_edge=LEFT).move_to([0, 0.5, 0])
        self.add(lim)
        self.step("three_ideas")
        self.remove(lim)

        self.add(tx("Spectral Progressive Diffusion in three ideas", 38, bold=True).move_to([0, 2.4, 0]))
        pts = ["1.  Diffusion resolves frequencies low → high: grow the resolution as they emerge",
               "2.  Spectral noise expansion + timestep alignment keep the trajectory valid",
               "3.  A " + DEL + "-optimal schedule read off the power spectrum, with one knob " + DEL]
        for k, p in enumerate(pts):
            self.add(tx(p, 25).move_to([0, 1.25 - 0.8 * k, 0]).align_to([-6.2, 0, 0], LEFT))
            self.wait(1.9)
        self.add(tx("up to 7× faster images  ·  2.5× faster video  ·  training-free or fine-tuned", 28, FRONT)
                 .move_to([0, -1.45, 0]))
        self.step("end_card")
        self.clear_all(run_time=0.6)
        if self.hdr is not None:
            self.remove(self.hdr)
        t = tx("Spectral Progressive Diffusion", 56, bold=True).move_to([0, 1.0, 0])
        u = tx("arXiv 2605.18736  ·  howardxiao.ca/speed", 28, DIM).move_to([0, 0.0, 0])
        v = tx("Howard Xiao, Brian Chao, Lior Yariv, Gordon Wetzstein · Stanford University", 24, DIM).move_to([0, -0.7, 0])
        self.add(t, u, v)
        self.wait(SECTION_HOLD)
