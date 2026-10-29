"""Time-Division Multiplexing Light Field Display With Learned Coded Aperture — 1-minute teaser, v2.

Condensed from ../animation_tdmlf_v3.py (5-min teaser). Source: Chao*, Liu*, Chen, IEEE TIP 32, 350-363 (2023),
arXiv 2107.06205. Reads ../data (paper figure crops, website GIF frames) and ../data/generated (toy-optics caches built by
the 5-min script) read-only. Helpers are copied from the 5-min script, not imported.
Scenes: T1_Hook, T2_Method, T3_Payoff.
"""
from manim import *
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import matplotlib
import fcntl
import glob
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, "..", "data"))   # read-only: figure crops, GIF frames
GEN = os.path.join(DATA, "generated")

config.background_color = BLACK

# ------------------------------------------------------------------ style (same as random_phase_wave_splatting)
FONT = "CMU Serif"
MATHFONT = "Latin Modern Math"          # CMU Serif lacks × − → ≈ ∑ ∏ ℱ 𝒫 𝔼 ∇ 𝒲 …
SYMFONT = "DejaVu Sans"                 # ✗ is in neither
DIM = "#8A8A8A"
RED = "#FF5252"
GREEN = "#69F0AE"
BLUE = "#448AFF"
AMBER = "#FFCA28"
CYAN = "#4DD0E1"
PURPLE = "#E040FB"
ORANGE = "#FF9E40"

HEADER_Y = 3.62
CAPTION_Y = -3.5
MATHCH = set("×⊙→≈∼−∈ℝ≪≤↓↔∑∏∬∫ℱ𝒫𝒢𝒩ℋ𝒪𝒰𝔼∗≠∞∂∘⊤●⇒∠✓∇𝒲ℒ")
SYMCH = set("✗")


def mathify(s):
    out = []
    for c in s:
        if c in MATHCH:
            out.append(f'<span font_family="{MATHFONT}">{c}</span>')
        elif c in SYMCH:
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


def tx(s, size=30, color=WHITE, bold=False, italic=False):
    if bold:
        s = f"<b>{s}</b>"
    if italic:
        s = f"<i>{s}</i>"
    return MarkupText(mathify(s), font=FONT, font_size=size, color=color)


def I(v):
    return f"<i>{v}</i>"


def B(v):
    return f"<b>{v}</b>"


SQ = sup("2")
PH = I("P\u0302")                     # P-hat (combining circumflex renders in CMU Serif italic)
EQ8 = (f"{I('R')}{sub(I('z'))} = ∑{sub(I('s') + '∈' + I('G'))} |{I('h')}{sub(I('s') + ',' + I('z'))}|{SQ} ∗ "
       f"{I('L')}{sub(I('s'))}")
EQ9 = f"{I('h')}{sub(PH + '<sub>' + I('i') + '</sub>' + ',' + I('z'))} = ℱ{{ {PH}{sub(I('i'))} · {I('W')}({I('z')}) }}"
EQ9s = f"PSF = |ℱ{{ {I('P')} · {I('W')}({I('z')}) }}|{SQ}"
EQ10 = (f"{I('R')}{sub(I('z'))} = ∑{sub(I('t') + ' = 1')}{sup(I('k'))} |{I('h')}{sub(PH + I('t') + ',' + I('z'))}|{SQ} ∗ "
        f"{I('I')}{sub(I('c') + ',' + I('t'))}")
EQ12 = (f"ℒ = 𝒲 ⊙ ‖ {I('F')}{sub(I('c'))} − {I('F')}{sub(I('g'))} ‖{sub('1')}")
EQ13 = (f"{I('U')}{sub(I('j'))} = ‖∇{sub(I('x'))}{I('R')}{sub(I('j'))}‖{sub('1')} + "
        f"‖∇{sub(I('y'))}{I('R')}{sub(I('j'))}‖{sub('1')}")
EQ14 = f"𝒲{sub(I('j'))} = exp(β {I('Ū')}{sub(I('j'))})"

# ------------------------------------------------------------------ toy constants (must match ../animation_tdmlf_v3.py)
L = 9
CORN = [(0, 0), (0, L - 1), (L - 1, 0), (L - 1, L - 1)]
D_NEAR, D_MID, D_FAR = 1.0, 0.0, -1.0
AS_S = np.linspace(-1.4, 1.4, 57)
W20S = np.linspace(0, 18, 37)
NIT = 3000


def cmap_rgb(a, cmap="inferno"):
    return (matplotlib.colormaps[cmap](np.clip(a, 0, 1))[..., :3] * 255).astype(np.uint8)


def rgba(a):
    if a.ndim == 2:
        a = np.stack([a] * 3, -1)
    if a.shape[2] == 3:
        a = np.concatenate([a, np.full(a.shape[:2] + (1,), 255, np.uint8)], -1)
    return np.ascontiguousarray(a.astype(np.uint8))


def gen(name):
    p = os.path.join(GEN, name)
    assert os.path.exists(p), f"{p} missing: render the 5-min script once to build ../data/generated"
    return p


# ------------------------------------------------------------------ mobject helpers
def img(path, h=None, w=None):
    p = path if os.path.isabs(path) else os.path.join(DATA, path)
    m = ImageMobject(p)
    if h is not None:
        m.set(height=h)
    if w is not None:
        m.set(width=w)
    return m


def arr_img(a, h=None, w=None, nearest=False):
    m = ImageMobject(rgba(a))
    if nearest:
        m.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
    if h is not None:
        m.set(height=h)
    if w is not None:
        m.set(width=w)
    return m


def framed(m, color=WHITE, sw=1.5):
    return SurroundingRectangle(m, buff=0, color=color, stroke_width=sw)


def live(m, fn):
    """Attach an updater that rewrites the image pixels every frame."""
    m.add_updater(lambda mob: setattr(mob, "pixel_array", rgba(fn())))
    return m


def interp(frames, v):
    v = float(np.clip(v, 0, len(frames) - 1))
    i0 = int(np.floor(v))
    i1 = min(i0 + 1, len(frames) - 1)
    w = v - i0
    return ((1 - w) * frames[i0] + w * frames[i1]).astype(np.uint8)


def arrow(a, b, buff=0.1, color=WHITE, sw=3):
    return Arrow(a, b, buff=buff, color=color, stroke_width=sw,
                 max_tip_length_to_length_ratio=0.18, tip_length=0.16)


def aperture_rgb(P, color=(77, 208, 225), grid=True, px=12):
    """9 × 9 aperture pattern as an image: open = cyan, closed = near-black, thin grid lines."""
    P = np.clip(np.asarray(P, float), 0, 1)
    c = np.array(color, float)
    base = np.array([18, 20, 26], float)
    cell = base + P[..., None] * (c - base)
    im = np.kron(cell, np.ones((px, px, 1)))
    if grid:
        im[::px, :, :] = 55
        im[:, ::px, :] = 55
        im[-1, :, :] = 55
        im[:, -1, :] = 55
    return im.astype(np.uint8)


# ------------------------------------------------------------------ scene base
class Base(Scene):
    def setup(self):
        self.cap = None
        self.hdr = None
        self.clock = ValueTracker(0)
        self.clock.add_updater(lambda m, dt: m.increment_value(dt))
        self.add(self.clock)

    def header(self, s):
        if self.hdr is not None:
            self.remove(self.hdr)
        self.hdr = tx(s, 24, DIM).to_corner(UL, buff=0.35).set_y(HEADER_Y)
        self.add(self.hdr)

    def _hold_caption(self):
        if self.cap is not None:
            left = self.cap_need - (self.renderer.time - self.cap_t0)
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

    def caption(self, s, wait=None, size=28):
        self._hold_caption()
        s = re.sub(r"(\d) × (\d)", "\\1\u00a0×\u00a0\\2", s)
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
        self.cap_need = len(plain) / 17.0 + 0.8
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

    def title_card(self, num, title, hold=1.5):
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
        keep = set(keep) | {self.hdr, self.clock}
        mobs = [m for m in self.mobjects if m not in keep]
        for m in mobs:
            m.clear_updaters()
        if mobs:
            self.play(*[FadeOut(m) for m in mobs], run_time=run_time)

    def run(self, tracker, target, t, rate=linear):
        self.play(tracker.animate.set_value(target), run_time=t, rate_func=rate)


def label(m, s, size=22, color=WHITE, d=UP, buff=0.15):
    """Label above/below m, placed on a common baseline so labels in a row line up."""
    t = tx(s, size, color)
    plain = re.sub(r"<[^>]+>", "", s)
    desc = any(c in "gjpqy(),;[]{}|µφμβγηρχψξζ" for c in plain) or 'rise="-' in s
    dd, asc = 0.0027 * size, 0.0098 * size
    base = m.get_top()[1] + buff + dd if d is UP else m.get_bottom()[1] - buff - asc
    t.shift(UP * (base - (t.get_bottom()[1] + (dd if desc else 0))))
    t.set_x(m.get_center()[0])
    return t


def toy_note(s):
    return tx(s, 16, DIM)


def star_shape(r=0.14, color=AMBER):
    return Star(5, outer_radius=r, inner_radius=r * 0.45, color=color, fill_opacity=1, stroke_width=0)


def focus_ruler(center, length, f_of_t, labels=True):
    """Depth ruler 'near ↔ far' with star / leaf / letters markers and a moving focus dot.
    f_of_t() returns the focus disparity a in [−1.4, 1.4]; near (a = +1) is on the left."""
    nl = Line(LEFT * length / 2, RIGHT * length / 2, color=DIM, stroke_width=2).move_to(center)
    a2x = lambda a: center[0] - (a / 1.4) * length / 2
    g = VGroup(nl)
    marks = [(D_NEAR, "near: star", AMBER), (D_MID, "middle: leaf", GREEN), (D_FAR, "far: letters", CYAN)]
    for a, s, c in marks:
        tri = Triangle(color=c, fill_opacity=1, stroke_width=0).scale(0.08).rotate(PI).move_to(
            [a2x(a), center[1] + 0.13, 0])
        g.add(tri)
        if labels:
            g.add(tx(s, 17, c).move_to([a2x(a), center[1] + 0.38, 0]))
    dot = always_redraw(lambda: Dot([a2x(float(np.clip(f_of_t(), -1.4, 1.4))), center[1], 0], radius=0.08, color=WHITE))
    lab = tx("focus", 18, DIM).next_to(nl, LEFT, buff=0.25)
    return VGroup(g, lab), dot


def sweep_index(a):
    return (a - AS_S[0]) / (AS_S[1] - AS_S[0])


def fitw(m, w=12.8):
    if m.width > w:
        m.scale_to_fit_width(w)
    return m


# ------------------------------------------------------------------ eye diagram (side view, thin-lens optics)
class Eye:
    """Side-view eye facing left. Lens at x_L; rays from an on-axis point at x_o are refracted by a thin lens with focal
    length f and stopped on the eyeball circle (the retina side)."""

    def __init__(self, center=(3.6, 0.35), R=1.35, pupil=0.42, scale=1.0):
        self.c = np.array([center[0], center[1], 0.0])
        self.R = R
        self.xL = center[0] - 0.78 * R
        self.pupil = pupil
        # axial distance lens → back of the eye
        self.dR = center[0] + R - self.xL

    def body(self):
        c, R = self.c, self.R
        ball = Circle(radius=R, color=DIM, stroke_width=2).move_to(c)
        retina = Arc(radius=R, start_angle=-0.95, angle=1.9, color=ORANGE, stroke_width=5).move_to(c, aligned_edge=ORIGIN)
        retina.shift(c - retina.get_arc_center())
        cornea = Arc(radius=0.62 * R, start_angle=PI - 0.9, angle=1.8, color=DIM, stroke_width=2)
        cornea.shift(np.array([self.xL + 0.28 * R, c[1], 0]) - cornea.get_arc_center())
        iris = VGroup(Line([self.xL, c[1] + self.pupil, 0], [self.xL, c[1] + 0.8 * R, 0], color=BLUE, stroke_width=5),
                      Line([self.xL, c[1] - self.pupil, 0], [self.xL, c[1] - 0.8 * R, 0], color=BLUE, stroke_width=5))
        return VGroup(ball, retina, cornea, iris)

    def lens(self, f):
        w = float(np.clip(0.16 + 0.95 * (1 / f - 0.45), 0.12, 0.62))
        return Ellipse(width=w, height=2 * self.pupil + 0.28, color=WHITE, fill_color="#B3E5FC", fill_opacity=0.35,
                       stroke_width=2).move_to([self.xL, self.c[1], 0])

    def hit(self, p0, d):
        """Far intersection of the ray p0 + s d with the eyeball circle."""
        c = self.c[:2]
        p0, d = np.asarray(p0[:2], float), np.asarray(d[:2], float) / np.linalg.norm(d[:2])
        b = 2 * np.dot(d, p0 - c)
        cc = np.dot(p0 - c, p0 - c) - self.R ** 2
        s = (-b + np.sqrt(max(b * b - 4 * cc, 0))) / 2
        return p0 + s * d

    def rays(self, xo, f, n=7, color=AMBER, op=0.85, yo=0.0):
        y0 = self.c[1]
        do = self.xL - xo
        inv = 1 / f - 1 / do
        g = VGroup()
        ends = []
        for h in np.linspace(-self.pupil * 0.92, self.pupil * 0.92, n):
            pL = np.array([self.xL, y0 + h])
            g.add(Line([xo, y0 + yo, 0], [pL[0], pL[1], 0], color=color, stroke_width=2, stroke_opacity=op))
            if abs(inv) < 1e-6:
                d = np.array([1.0, -yo / do])
            else:
                di = 1 / inv
                q = np.array([self.xL + di, y0 - yo * di / do])
                d = (q - pL) * np.sign(di)
            e = self.hit(pL, d)
            ends.append(e[1])
            g.add(Line([pL[0], pL[1], 0], [e[0], e[1], 0], color=color, stroke_width=2, stroke_opacity=op))
        return g, (min(ends), max(ends))

    def f_for(self, xo):
        return 1 / (1 / (self.xL - xo) + 1 / self.dR)




def gif_frames():
    return [np.asarray(Image.open(os.path.join(DATA, "gif_frames", f"{i:03d}.png")).convert("RGB")) for i in (0, 1)]


# ================================================================== T1 Hook
class T1_Hook(Base):
    def construct(self):
        title = fitw(tx("Time-Division Multiplexing Light Field Display", 46, bold=True)).move_to([0, 3.1, 0])
        subt = tx("with Learned Coded Aperture", 32).move_to([0, 2.5, 0])
        frames = gif_frames()
        gif = arr_img(frames[0], 3.0).move_to([0, -0.15, 0])
        live(gif, lambda: frames[int(self.clock.get_value()) % 2])
        lab = always_redraw(lambda: label(gif, ("baseline TDM-LFD: ghosted blur" if int(self.clock.get_value()) % 2 == 0
                                                else "CTDM-LFD (this paper): natural blur"), 24,
                                          RED if int(self.clock.get_value()) % 2 == 0 else GREEN))
        auth = tx("Brian Chao*, Chang-Le Liu*, Homer H. Chen  ·  National Taiwan University  ·  IEEE TIP 2023", 20, DIM
                  ).move_to([0, -2.35, 0])
        note = tx("near focus, SIGGRAPH16 light field (paper Fig. 12)", 16, DIM).next_to(gif, DOWN, buff=0.12)
        self.add(title, subt, auth)
        self.play(FadeIn(gif), Create(framed(gif, DIM, 1.2)), run_time=0.5)
        self.add(lab, note)
        self.caption("A light field display that gets defocus blur right.")
        self.hold(1.2)
        self.clear_all()

        # vergence-accommodation conflict, top view
        cx = -2.6
        eyes = [np.array([cx - 0.85, -2.1, 0]), np.array([cx + 0.85, -2.1, 0])]
        E = VGroup(*[VGroup(Circle(radius=0.3, color=WHITE, stroke_width=2).move_to(p),
                            Dot(p + UP * 0.22, radius=0.08, color=BLUE)) for p in eyes])
        scr_y = 2.0
        scr = Line([cx - 2.0, scr_y, 0], [cx + 2.0, scr_y, 0], color=WHITE, stroke_width=5)
        virt = star_shape(0.24).move_to([cx, 0.0, 0])
        verg = VGroup(*[Line(p + UP * 0.3, virt.get_center(), color=AMBER, stroke_width=3) for p in eyes])
        fl = DashedLine([cx - 1.7, scr_y - 0.22, 0], [cx + 1.7, scr_y - 0.22, 0], color=CYAN, stroke_width=4)
        bx = cx + 2.35
        br = VGroup(Line([bx, 0.0, 0], [bx, scr_y - 0.22, 0], color=RED, stroke_width=4),
                    Line([bx - 0.15, 0.0, 0], [bx + 0.15, 0.0, 0], color=RED, stroke_width=4),
                    Line([bx - 0.15, scr_y - 0.22, 0], [bx + 0.15, scr_y - 0.22, 0], color=RED, stroke_width=4))
        self.play(FadeIn(E), FadeIn(scr), FadeIn(virt), Create(verg), Create(fl), run_time=0.6)
        self.add(tx("screen", 20, DIM).next_to(scr, RIGHT, buff=0.15), br,
                 tx("eyes point here (vergence)", 24, AMBER).move_to([3.3, 0.0, 0]),
                 tx("but focus on the screen", 24, CYAN).move_to([3.3, 1.78, 0]),
                 tx("vergence-accommodation conflict", 26, RED).move_to([3.3, 0.9, 0]))
        self.caption("Stereo headsets make your eyes converge at one depth but focus at another.")
        self.clear_all()

        gt = np.load(gen("sweep_gt.npy"))
        base = np.load(gen("sweep_base.npy"))
        a = ValueTracker(D_NEAR)
        h = 3.4
        A = arr_img(gt[0], h).move_to([-2.2, 0.8, 0])
        Bm = arr_img(base[0], h).move_to([2.2, 0.8, 0])
        live(A, lambda: interp(gt, sweep_index(a.get_value())))
        live(Bm, lambda: interp(base, sweep_index(a.get_value())))
        self.play(FadeIn(A), FadeIn(Bm), run_time=0.4)
        self.add(label(A, "dense light field: 81 views", 26, GREEN), label(Bm, "sparse display: 4 views", 26, RED))
        rl, dot = focus_ruler(np.array([0, -1.6, 0]), 5.5, a.get_value)
        self.add(rl, dot, toy_note("toy simulation: refocusing a three-layer scene").move_to([0, -2.4, 0]))
        self.caption("Light field displays let the eye refocus. But with few views, blur breaks into ghosts.")
        self.run(a, D_FAR, 2.4, smooth)
        self.run(a, D_NEAR, 2.4, smooth)
        self.run(a, D_FAR, 2.4, smooth)
        self.clear_all()


# ================================================================== T2 Method
class T2_Method(Base):
    def construct(self):
        psfs = [np.load(gen(f"psf_{k}.npy")) for k in ("open", "corners", "coded")]
        ap9 = np.load(os.path.join(DATA, "fig9_apertures_9x9.npy")).astype(float)
        cm = np.zeros((L, L))
        for v, u in CORN:
            cm[v, u] = 1
        apert = [np.ones((L, L)), cm, ap9[0]]
        names = ["open aperture", "4 windows (TDM)", "coded aperture (paper)"]
        colsn = [WHITE, RED, GREEN]
        w = ValueTracker(0.0)
        wi = lambda: w.get_value() / (W20S[1] - W20S[0])
        xs = [-4.4, 0, 4.4]
        for x, st, nm, c, P in zip(xs, psfs, names, colsn, apert):
            m = arr_img(st[0], 3.0).move_to([x, 0.75, 0])
            live(m, (lambda s_: (lambda: interp(s_, wi())))(st))
            ap = arr_img(aperture_rgb(P, px=10), 0.95, nearest=True).move_to([x, -1.4, 0])
            self.add(m, framed(m, DIM, 1.0), label(m, nm, 24, c), ap)
        self.add(tx("blur of one point, growing defocus (toy Fourier optics); below: the aperture", 16, DIM).move_to([0, -2.25, 0]))
        self.caption("Blur takes the shape of the aperture: four tiny windows give four ghost dots.")
        self.run(w, 18.0, 3.5, smooth)
        self.caption("A coded aperture instead spreads each point into a pattern we can design.")
        self.run(w, 8.0, 1.6, smooth)
        self.run(w, 18.0, 1.6, smooth)
        self.clear_all()

        enc = np.load(gen("enc_images.npy"))
        P = np.load(gen("P_final.npy"))
        cum = np.load(gen("cycle_cum.npy"))
        ys = [1.9, 0.75, -0.4, -1.55]
        encm = Group(*[arr_img(enc[i], 1.05).move_to([-4.6, y, 0]) for i, y in enumerate(ys)])
        apm = Group(*[arr_img(aperture_rgb(P[i], px=10), 1.0, nearest=True).move_to([-3.2, y, 0]) for i, y in enumerate(ys)])
        tl = VGroup(*[tx(f"{I('t')} = {i + 1}", 18, DIM).next_to(encm[i], LEFT, buff=0.15) for i in range(4)])
        t = ValueTracker(0.0)
        ti = lambda: int(np.clip(t.get_value(), 0, 3.999))
        R = arr_img(cum[0], 3.2).move_to([2.9, 0.55, 0])
        live(R, lambda: cum[ti()])
        a3 = arrow([-2.4, 0.2, 0], [1.1, 0.2, 0])
        hl = always_redraw(lambda: SurroundingRectangle(Group(encm[ti()], apm[ti()]), color=AMBER, buff=0.08, stroke_width=3))
        self.play(FadeIn(encm), FadeIn(apm), FadeIn(R), GrowArrow(a3), run_time=0.5)
        self.add(tl, hl, framed(R, DIM, 1.0), VGroup(tx("encoded", 20), tx("images", 20)).arrange(DOWN, buff=0.05).next_to(encm, UP, buff=0.12),
                 VGroup(tx("coded", 20, CYAN), tx("apertures", 20, CYAN)).arrange(DOWN, buff=0.05).next_to(apm, UP, buff=0.12),
                 tx("eye sums the frames", 20, DIM).next_to(a3, UP, buff=0.1),
                 always_redraw(lambda: label(R, f"retina, {ti() + 1} of 4 frames", 22)),
                 tx(EQ10, 30).move_to([2.9, -1.65, 0]),
                 toy_note("toy: learned images and apertures, eye focused on the near star").move_to([2.9, -2.2, 0]))
        self.caption("Show k encoded images, each through its own coded aperture. The eye sums them.")
        self.run(t, 3.99, 3.2)
        t.set_value(0)
        self.run(t, 3.99, 2.4)
        self.clear_all()

        sP = np.load(gen("train_snapP.npy"))
        sI = np.load(gen("train_snapImg.npy"))
        sIt = np.load(gen("train_snapIt.npy"))
        it = ValueTracker(0.0)

        def sidx():
            v = it.get_value()
            j = int(np.clip(np.searchsorted(sIt, v, side="right") - 1, 0, len(sIt) - 2))
            return j, np.clip((v - sIt[j]) / max(sIt[j + 1] - sIt[j], 1), 0, 1)
        aps = []
        for i in range(4):
            m = arr_img(aperture_rgb(sP[0, i], px=10), 1.45, nearest=True).move_to([-4.9 + 1.65 * (i % 2), 1.35 - 1.65 * (i // 2), 0])
            live(m, (lambda k: (lambda: aperture_rgb((lambda j, w_: (1 - w_) * sP[j, k] + w_ * sP[j + 1, k])(*sidx()), px=10)))(i))
            aps.append(m)
        R = arr_img(sI[0], 3.3).move_to([2.6, 0.55, 0])
        live(R, lambda: (lambda j, w_: ((1 - w_) * sI[j] + w_ * sI[j + 1]).astype(np.uint8))(*sidx()))
        self.play(*[FadeIn(m) for m in aps], FadeIn(R), run_time=0.4)
        self.add(tx("coded apertures", 22, CYAN).move_to([-4.08, 2.6, 0]), label(R, "what the eye sees (near focus)", 22),
                 framed(R, DIM, 1.0),
                 always_redraw(lambda: tx(f"training step {int(it.get_value())}", 22, AMBER).move_to([-4.08, -1.75, 0])),
                 toy_note("toy optimisation through the display model; the paper trains an encoding CNN and the apertures jointly"
                          ).move_to([0, -2.4, 0]))
        self.caption("A network and the apertures are learned end to end through a Fourier-optics model.")
        self.run(it, 60, 1.6, rate=lambda x: x)
        self.run(it, NIT - 1, 3.6, rate=lambda x: x ** 2.2)
        for m in aps + [R]:
            m.clear_updaters()
        self.clear_all()


# ================================================================== T3 Payoff
class T3_Payoff(Base):
    def construct(self):
        w = 3.1
        xs = [-3.6, 0, 3.6]
        for x, m_, n_, c in zip(xs, ["base", "ours", "gt"], ["baseline TDM-LFD", "CTDM-LFD (ours)", "ground truth"],
                                [RED, GREEN, DIM]):
            a = img(f"fig11_mom_near_{m_}_zA.png", w=w).move_to([x, 1.05, 0])
            self.add(a, label(a, n_, 26, c))
        self.add(tx("real Lytro light field, near focus, zoom on the far letters (paper Fig. 11)", 16, DIM).move_to([0, -0.8, 0]))
        head = fitw(tx("+3.7 dB (INRIA-syn)  ·  +3.8 dB (INRIA-real)  ·  +4.7 dB (SIGGRAPH16)", 32, AMBER)).move_to([0, -1.55, 0])
        sub_ = tx("PSNR over TDM-LFD with the same 4 views (paper Table V)", 20, DIM).move_to([0, -2.15, 0])
        self.add(head, sub_)
        self.caption("Natural blur instead of ghosts, and about 4 dB better on three datasets.")
        self.hold(1.0)
        self.clear_all()
        rows17 = ["chairs", "mom", "poppy"]
        stacks = [[np.asarray(Image.open(os.path.join(DATA, f"fig17_{r}_{j}.png")).convert("RGB")) for j in range(9)]
                  for r in rows17]
        j = ValueTracker(0.0)
        for x, stck in zip([-4.3, 0, 4.3], stacks):
            m = arr_img(stck[0], 3.5).move_to([x, 0.7, 0])
            live(m, (lambda s_: (lambda: s_[int(np.clip(round(j.get_value()), 0, 8))]))(stck))
            self.add(m)
        self.add(always_redraw(lambda: tx(f"focal plane {int(np.clip(round(j.get_value()), 0, 8)) + 1} of 9", 24, AMBER
                                          ).move_to([0, -1.5, 0])),
                 tx("CTDM-LFD focal stacks, n = 4, k = 4 (paper Fig. 17)", 18, DIM).move_to([0, -2.0, 0]))
        self.caption("Sweep the focus: sharp where you look, smooth blur everywhere else.")
        self.run(j, 8, 3.0)
        self.run(j, 0, 2.4)
        self.clear_all()
        frames = gif_frames()
        gif = arr_img(frames[1], 2.3).move_to([0, 1.6, 0])
        t1 = fitw(tx("Time-Division Multiplexing Light Field Display", 40, bold=True)).move_to([0, -0.1, 0])
        t1b = tx("with Learned Coded Aperture", 30).move_to([0, -0.7, 0])
        t2 = tx("IEEE Transactions on Image Processing 32 (2023)  ·  arXiv 2107.06205", 22).move_to([0, -1.35, 0])
        t4 = tx("Brian Chao*, Chang-Le Liu*, Homer H. Chen", 22, DIM).move_to([0, -1.95, 0])
        t5 = tx("National Taiwan University  ·  * equal contribution", 20, DIM).move_to([0, -2.37, 0])
        self.add(gif, t1, t1b, t2, t4, t5)
        self.wait(5.0)
