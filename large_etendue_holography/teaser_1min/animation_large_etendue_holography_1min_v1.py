"""Large Étendue 3D Holographic Display with Content-adaptive Dynamic Fourier Modulation
— 1-minute 3Blue1Brown-style teaser, v1.

Condensed from ../animation_large_etendue_v3.py (helpers copied, not imported).
Source: Chao, Gopakumar, Choi, Kim, Shi, Wetzstein, SIGGRAPH Asia 2024, arXiv 2409.03143.
Assets are read from ../data (paper figure crops) and ../data/generated (cached toy simulation
built by the 5-min script); toy visuals are labelled "toy simulation", every number is from the paper.

Render (see render.sh):
    manim -ql --media_dir media animation_large_etendue_holography_1min_v1.py <Scene>
Scenes, in order: T1_Problem, T2_Method, T3_Payoff.
"""
from manim import *
import numpy as np
from PIL import Image
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = "/home/brianchc/project_animations/large_etendue_holography/data"
GEN = os.path.join(DATA, "generated")

config.background_color = BLACK

# ------------------------------------------------------------------ style (copied from the 5-min version)
FONT = "CMU Serif"
MATHFONT = "Latin Modern Math"
GRN = "#39E27D"
RED = "#FF5252"
BLUE = "#448AFF"
AMBER = "#FFCA28"
CYAN = "#4DD0E1"
PURPLE = "#E040FB"
DIM = "#8A8A8A"
GHOST = "#3A3A3A"
SLMC = "#90A4AE"
SRC3 = ["#FFB74D", GRN, "#4FC3F7"]

HEADER_Y = 3.62
CAPTION_Y = -3.5
MATHCH = set("×⊙→≈∼−∈↓′≤≥∝⋅∞ℱℋ𝒫𝒰ℳ⟶↔∘")


def mathify(s):
    return "".join(
        f'<span font_family="{MATHFONT}">{c}</span>' if c in MATHCH else c for c in s)


def sub(x):
    return f'<span rise="-3000"><small><small>{x}</small></small></span>'


def sup(x):
    return f'<span rise="4500"><small><small>{x}</small></small></span>'


def tx(s, size=30, color=WHITE, bold=False, italic=False):
    if bold:
        s = f"<b>{s}</b>"
    if italic:
        s = f"<i>{s}</i>"
    return MarkupText(mathify(s), font=FONT, font_size=size, color=color)


LAM = "<i>λ</i>"
PF = "𝒫"


def gmap(v):
    v = np.clip(v, 0, 1)
    r = 0.10 * v + 0.75 * v ** 3
    g = v ** 0.85
    b = 0.25 * v + 0.60 * v ** 3
    a = np.stack([r, g, b, np.ones_like(v)], -1)
    return (a * 255).astype(np.uint8)


def gen(name):
    return os.path.join(GEN, name)


TOY = np.load(gen("toy_cgh.npz"))
N_, S_, M_, CELL, EB, R_P, SPAN, NF = [int(v) for v in TOY["meta"]]
SRC_ORDER = [(0, 0), (0, 1), (0, -1), (-1, 0), (1, 0), (-1, 1), (1, -1), (-1, -1), (1, 1)]


def img(path, h=None, w=None):
    p = path if os.path.isabs(path) else os.path.join(DATA, path)
    m = ImageMobject(p)
    if h is not None:
        m.set(height=h)
    if w is not None:
        m.set(width=w)
    return m


def gimg(name, h=None, w=None, nearest=False):
    m = img(gen(name), h, w)
    if nearest:
        m.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
    return m


def framed(m, color=WHITE, sw=1.5):
    return SurroundingRectangle(m, buff=0, color=color, stroke_width=sw)


class FrameSeq:
    """ImageMobject whose pixels follow a ValueTracker index into a list of equal-size RGBA arrays."""

    def __init__(self, frames, h):
        self.frames = frames
        self.k = ValueTracker(0)
        self.mob = ImageMobject(frames[0]).set(height=h)
        self.mob.add_updater(lambda m: setattr(
            m, "pixel_array", self.frames[int(np.clip(round(self.k.get_value()), 0, len(self.frames) - 1))]))


def load_rgba(path):
    return np.asarray(Image.open(path).convert("RGBA"))


def build_seq(names, h):
    return FrameSeq([load_rgba(gen(n)) for n in names], h)


def lens(x, y, h, color=CYAN):
    return Ellipse(width=0.32, height=h, stroke_color=color, stroke_width=2.5,
                   fill_color=color, fill_opacity=0.12).move_to([x, y, 0])


def slm_bar(x, y, h, n=12, w=0.16, color=SLMC, seed=0, amp=False):
    rng = np.random.default_rng(seed)
    cells = VGroup()
    for i in range(n):
        v = rng.uniform(0.15, 0.95)
        c = interpolate_color(BLACK, ManimColor(AMBER) if amp else WHITE, v)
        r = Rectangle(width=w, height=h / n, stroke_width=0.6, stroke_color=color,
                      fill_color=c, fill_opacity=1)
        r.move_to([x, y + h / 2 - (i + 0.5) * h / n, 0])
        cells.add(r)
    return cells


def beam(p1a, p1b, p2a, p2b, color=GRN, op=0.22):
    return Polygon(p1a, p1b, p2b, p2a, stroke_width=0, fill_color=color, fill_opacity=op)


def eye(x, y, r=0.42):
    ball = Circle(radius=r, stroke_color=WHITE, stroke_width=2).move_to([x, y, 0])
    pupil = Ellipse(width=0.12, height=0.36, fill_color=WHITE, fill_opacity=1,
                    stroke_width=0).move_to([x - r * 0.93, y, 0])
    return VGroup(ball, pupil)


class Base(Scene):
    def setup(self):
        self.cap = None
        self.hdr = None

    def wait(self, duration=1.0, **kw):
        kw.setdefault("frozen_frame", False)
        return super().wait(duration, **kw)

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

    def caption(self, s, wait=None, size=30):
        self._hold_caption()
        s = re.sub(r"(\d) × (\d)", "\\1\u00a0×\u00a0\\2", s)
        if self.cap is not None:
            self.remove(self.cap)
        c = tx(s, size)
        if c.width > 13.2:
            c.scale_to_fit_width(13.2)
        c.move_to([0, CAPTION_Y, 0])
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

    def clear_all(self, keep=(), run_time=0.5):
        self.uncaption()
        keep = set(keep)
        mobs = [m for m in self.mobjects if m not in keep]
        for m in mobs:
            m.clear_updaters()
        if mobs:
            self.play(*[FadeOut(m) for m in mobs], run_time=run_time)
        self.hdr = None


# ================================================================== T1 Problem: hook + étendue
NLAM = 1000 * 632.8e-6        # N·λ in mm (paper Fig. 3: 1000 px, 632.8 nm, p = 10.8 µm)


def fov_deg(w, alpha=1):
    return np.degrees(2 * np.arctan(alpha * NLAM / (2 * w)))


class T1_Problem(Base):
    def construct(self):
        # --- hook: title + the eyebox problem in a real capture (paper Fig. 1 b-d)
        t1 = tx("Large Étendue 3D Holographic Display", 44, bold=True).move_to([0, 3.25, 0])
        t2 = tx("with Content-adaptive Dynamic Fourier Modulation", 26, DIM).move_to([0, 2.65, 0])
        spec = img("teaser_spec_single.png", w=4.6).move_to([-3.3, 0.05, 0])
        view_c = img("teaser_single_center.png", w=4.6).move_to([3.3, 0.05, 0])
        view_r = img("teaser_single_right.png", w=4.6).move_to([3.3, 0.05, 0])
        l1 = tx("Fourier plane (single source)", 22, DIM).next_to(spec, UP, buff=0.15)
        l2 = tx("what the eye sees", 22, DIM).next_to(view_c, UP, buff=0.15).set_y(l1.get_y())
        src = tx("experimental captures, paper Fig. 1", 18, DIM).next_to(view_c, DOWN, buff=0.12).align_to(view_c, RIGHT)
        W = spec.width
        c0 = spec.get_center() + RIGHT * 0.006 * W
        c1 = spec.get_center() + RIGHT * 0.248 * W
        pup = Circle(radius=0.188 * W, color=RED, stroke_width=4).move_to(c0)
        plab = tx("eye pupil", 20, RED).next_to(pup, DOWN, buff=0.08)
        self.add(t1, t2, spec, view_c, l1, l2, src, pup, plab, framed(view_c, GHOST, 1))
        self.caption("Holograms show true 3D, but to a tiny eyebox")
        self.wait(2.2)
        self.remove(plab)
        self.caption("Move the eye a few millimetres: the image vanishes")
        self.play(pup.animate.move_to(c1).set_color(BLUE), FadeIn(view_r), run_time=2.2)
        self.clear_all()

        # --- étendue: FoV x eyebox is capped by the SLM pixel count
        eq = tx(f"FoV{sub('<i>x</i>')} · FoV{sub('<i>y</i>')} · <i>w</i>{sup('2')}  ≈  "
                f"{LAM}{sup('2')} <i>N</i>{sub('<i>x</i>')}<i>N</i>{sub('<i>y</i>')}", 42).move_to([0, 2.85, 0])
        eqn = tx("field of view × eyebox = étendue, fixed by the SLM pixel count", 22, DIM).next_to(eq, DOWN, buff=0.2)
        w = ValueTracker(2.0)
        x0, cy, Ls = -6.4, -0.55, 1.8

        def Gs():
            return 0.8 + 0.22 * w.get_value()

        def diagram():
            G = Gs()
            xl = x0 + G
            xe = xl + G
            we = 0.24 * w.get_value()
            fov = np.radians(fov_deg(w.get_value()))
            slm = Rectangle(width=0.14, height=Ls, stroke_color=SLMC, stroke_width=2,
                            fill_color="#37474F", fill_opacity=1).move_to([x0, cy, 0])
            ln = lens(xl, cy, Ls + 0.5)
            lit = Polygon([x0, cy + Ls / 2, 0], [xl, cy + Ls / 2, 0], [xe, cy + we / 2, 0],
                          [xe, cy - we / 2, 0], [xl, cy - Ls / 2, 0], [x0, cy - Ls / 2, 0],
                          stroke_width=0, fill_color=GRN, fill_opacity=0.18)
            eb = Line([xe, cy - we / 2, 0], [xe, cy + we / 2, 0], color=AMBER, stroke_width=6)
            Lw = 1.25
            o = np.array([xe, cy, 0])
            f1 = DashedLine(o, o + Lw * np.array([np.cos(fov / 2), np.sin(fov / 2), 0]), color=GRN,
                            stroke_width=2.5, dash_length=0.07)
            f2 = DashedLine(o, o + Lw * np.array([np.cos(fov / 2), -np.sin(fov / 2), 0]), color=GRN,
                            stroke_width=2.5, dash_length=0.07)
            arc = Arc(radius=0.75, start_angle=-fov / 2, angle=fov, arc_center=o, color=GRN, stroke_width=2.5)
            lb1 = tx("eyebox <i>w</i>", 22, AMBER).next_to(eb, DOWN, buff=0.12).shift(DOWN * max(0, 0.9 - we / 2))
            lb2 = tx("FoV", 22, GRN).next_to(o + np.array([Lw, 0, 0]), RIGHT, buff=0.12)
            return VGroup(lit, slm, ln, eb, f1, f2, arc, lb1, lb2)

        dg = always_redraw(diagram)
        top = tx("SLM", 22).move_to([x0, cy + Ls / 2 + 0.3, 0])
        eyelab = always_redraw(lambda: tx("eyepiece", 22, CYAN).move_to([x0 + Gs(), cy + Ls / 2 + 0.55, 0]))
        ax = Axes(x_range=[0, 10, 2], y_range=[0, 40, 10], x_length=5.0, y_length=2.75,
                  axis_config={"color": DIM, "stroke_width": 2, "include_ticks": True, "tick_size": 0.05},
                  tips=False).move_to([3.85, -0.75, 0])
        xl_ = tx("eyebox (mm)", 20, DIM).next_to(ax.x_axis, DOWN, buff=0.32).align_to(ax, RIGHT)
        yl_ = tx("FoV (°)", 20, DIM).next_to(ax.y_axis, UP, buff=0.08).align_to(ax.y_axis, LEFT)
        nums = VGroup(*[tx(str(v), 16, DIM).next_to(ax.c2p(v, 0), DOWN, buff=0.08) for v in (2, 4, 6, 8, 10)],
                      *[tx(str(v), 16, DIM).next_to(ax.c2p(0, v), LEFT, buff=0.08) for v in (10, 20, 30, 40)])
        curve = ax.plot(lambda x: fov_deg(x), x_range=[0.85, 10, 0.02], color=WHITE, stroke_width=2.5)
        note = VGroup(tx("1000 × 1000 SLM, 10.8 µm pixels,", 16, DIM),
                      tx("632.8 nm (paper Fig. 3)", 16, DIM)).arrange(DOWN, buff=0.06, aligned_edge=RIGHT)
        note.next_to(ax.c2p(10, 38), DOWN + LEFT, buff=0).shift(DOWN * 0.05)

        def rect():
            ww = w.get_value()
            f = fov_deg(ww)
            r = Polygon(ax.c2p(0, 0), ax.c2p(ww, 0), ax.c2p(ww, f), ax.c2p(0, f), stroke_color=AMBER,
                        stroke_width=2, fill_color=AMBER, fill_opacity=0.25)
            return VGroup(r, Dot(ax.c2p(ww, f), radius=0.06, color=AMBER))

        rr = always_redraw(rect)
        self.add(eq, eqn, dg, top, eyelab, ax, xl_, yl_, nums, curve, note, rr)
        self.caption("SLM pixels cap the étendue: FoV × eyebox")
        self.play(w.animate.set_value(7.5), run_time=3.0)
        self.caption("A 1080p SLM: 80° FoV leaves under 1 mm of eyebox")
        self.play(w.animate.set_value(1.1), run_time=3.0)
        self.clear_all()


# ================================================================== T2 Method: multi-source + Fourier modulation
class T2_Method(Base):
    def construct(self):
        # --- key insight: tilted sources tile shifted copies of the spectrum
        xsrc, xcol, xslm, cy = -6.4, -5.0, -3.3, 0.5
        fcol = xcol - xsrc
        ys = [0.75, 0.0, -0.75]
        srcs = VGroup(*[Dot([xsrc, cy + y, 0], radius=0.08, color=SRC3[i]) for i, y in enumerate(ys)])
        cl = lens(xcol, cy, 2.6)
        slm = slm_bar(xslm, cy, 2.4, n=12, seed=5)
        beams = VGroup()
        for i, y in enumerate(ys):
            ang = -np.arctan(y / fcol)
            h = 1.1
            d = (xslm - xcol) * np.tan(ang)
            beams.add(beam([xcol, cy + h, 0], [xcol, cy - h, 0], [xslm, cy + h + d, 0], [xslm, cy - h + d, 0],
                           SRC3[i], 0.18))
        cones = VGroup(*[Polygon([xsrc, cy + y, 0], [xcol, cy + 1.1, 0], [xcol, cy - 1.1, 0], stroke_width=0,
                                 fill_color=SRC3[i], fill_opacity=0.12) for i, y in enumerate(ys)])
        ly = cy + 2.35
        labs = VGroup(tx("lasers", 22).move_to([xsrc, ly, 0]),
                      tx("collimator", 22, CYAN).move_to([xcol, ly, 0]),
                      tx("phase SLM", 22).move_to([xslm, ly, 0]))
        tilt = tx("a tilt in space = a shift in frequency", 22, DIM).move_to([-4.5, cy - 2.4, 0])
        seq = build_seq([f"ms_{k}.png" for k in range(1, 10)], 4.3)
        seq.mob.move_to([3.3, 0.3, 0])
        fr = framed(seq.mob, GHOST, 1)
        cw = seq.mob.height / 4
        ebox = Square(side_length=3 * cw, color=AMBER, stroke_width=3).move_to(seq.mob)
        c1 = Square(side_length=cw, color=WHITE, stroke_width=2).move_to(seq.mob)
        spl = tx("computed Fourier plane (toy simulation)", 20, DIM).next_to(seq.mob, UP, buff=0.12)
        c1l = tx("one source", 18).next_to(c1, UP, buff=0.06)
        icon = VGroup(*[Dot([-6.1 + 0.3 * dx, -2.55 - 0.3 * dy, 0], radius=0.06, color=GHOST)
                        for (dy, dx) in SRC_ORDER])
        icl = tx("3 × 3 lasers", 20, DIM).next_to(icon, RIGHT, buff=0.25)
        icon[0].set_color(GRN)
        self.add(cl, slm, labs, seq.mob, seq.k, fr, c1, spl, c1l, icon, icl)
        self.caption("Idea: light the SLM from many angles at once")
        self.play(FadeIn(srcs[1]), FadeIn(cones[1]), FadeIn(beams[1]), run_time=0.6)
        self.play(FadeIn(srcs[0]), FadeIn(srcs[2]), FadeIn(cones[0]), FadeIn(cones[2]),
                  FadeIn(beams[0]), FadeIn(beams[2]), run_time=0.7)
        self.add(tilt)
        self.remove(c1l)
        for k in range(1, 9):
            icon[k].set_color(GRN)
            self.play(seq.k.animate.set_value(k), run_time=0.33, rate_func=linear)
        ebl = tx("expanded eyebox", 20, AMBER).next_to(ebox, DOWN, buff=0.12)
        self.add(ebox, ebl)
        self.caption("Tilts tile spectrum copies: eyebox 3×, étendue 9×")
        self.clear_all()

        # --- the catch and the fix: a content-adaptive amplitude SLM in the Fourier plane
        cy = 2.1
        xs = [-6.4, -5.4, -4.2, -2.8, -1.2, 0.1]
        srcs = VGroup(*[Dot([xs[0], cy + y, 0], radius=0.06, color=SRC3[i]) for i, y in enumerate((0.35, 0, -0.35))])
        l1 = lens(xs[1], cy, 1.3)
        s1 = slm_bar(xs[2], cy, 1.2, n=10, w=0.13, seed=1)
        l2 = lens(xs[3], cy, 1.3)
        s2 = slm_bar(xs[4], cy, 1.2, n=10, w=0.13, seed=4, amp=True)
        ey = eye(xs[5] + 0.3, cy, 0.34)
        glow = VGroup(
            Polygon([xs[0], cy, 0], [xs[1], cy + 0.55, 0], [xs[1], cy - 0.55, 0], stroke_width=0, fill_color=GRN, fill_opacity=0.15),
            beam([xs[1], cy + 0.55, 0], [xs[1], cy - 0.55, 0], [xs[2], cy + 0.55, 0], [xs[2], cy - 0.55, 0], GRN, 0.15),
            Polygon([xs[2], cy + 0.55, 0], [xs[2], cy - 0.55, 0], [xs[3], cy - 0.6, 0], [xs[3], cy + 0.6, 0], stroke_width=0, fill_color=GRN, fill_opacity=0.15),
            Polygon([xs[3], cy + 0.6, 0], [xs[3], cy - 0.6, 0], [xs[4], cy - 0.25, 0], [xs[4], cy + 0.25, 0], stroke_width=0, fill_color=GRN, fill_opacity=0.15),
        )
        labs = VGroup(tx("lasers", 18).move_to([xs[0], cy + 0.8, 0]),
                      tx("phase SLM", 18).move_to([xs[2], cy + 0.8, 0]),
                      tx("Fourier lens", 18, CYAN).move_to([xs[3], cy - 0.9, 0]))
        s2l = tx("amplitude SLM", 18, AMBER).move_to([xs[4], cy + 0.8, 0])
        fpl = tx("Fourier plane = eyebox plane", 22, AMBER).move_to([4.2, cy, 0])
        py, ps = -0.65, 2.35
        xp = [-5.0, -1.75, 1.5, 4.9]
        sp = gimg("v6_nomask.png", h=ps).move_to([xp[0], py, 0])
        ly = py + ps / 2 + 0.25
        spl = tx("spectrum (all 9 lasers)", 20, DIM).move_to([xp[0], ly, 0])
        bad = gimg("view_III1_c.png", h=ps).move_to([xp[3], py, 0])
        badl = tx("no Fourier mask", 20, RED).move_to([xp[3], ly, 0])
        longarr = Arrow([xp[0] + ps / 2 + 0.2, py, 0], [xp[3] - ps / 2 - 0.2, py, 0], buff=0, color=DIM,
                        stroke_width=3, max_tip_length_to_length_ratio=0.05)
        nt = tx("toy simulation", 18, DIM).next_to(bad, DOWN, buff=0.1).align_to(bad, RIGHT)
        f_sp, f_view = framed(sp, GHOST, 1), framed(bad, GHOST, 1)
        self.add(glow, srcs, l1, s1, l2, ey, labs, fpl, sp, f_sp, spl, bad, f_view, badl, longarr, nt)
        self.caption("But the copies overlap: ghosts, low contrast")
        self.wait(0.3)
        # fix
        clock = ValueTracker(0)
        clock.add_updater(lambda m, dt: m.increment_value(dt))
        Pseq = build_seq([f"v6_P_{t}.png" for t in range(6)], ps)
        Pseq.mob.move_to([xp[1], py, 0])
        Pseq.mob.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
        Mseq = build_seq([f"v6_masked_{t}.png" for t in range(6)], ps)
        Mseq.mob.move_to([xp[2], py, 0])
        for s in (Pseq, Mseq):
            s.k.add_updater(lambda m: m.set_value(int(clock.get_value() / 0.4) % 6))
        rec = FrameSeq([gmap(TOY["V6_path"][i].astype(np.float32)) for i in range(25)], ps)
        rec.mob.move_to([xp[3], py, 0])
        rec.k.set_value(12)
        ops = VGroup(tx("×", 40).move_to([(xp[0] + xp[1]) / 2, py, 0]),
                     tx("=", 40).move_to([(xp[1] + xp[2]) / 2, py, 0]),
                     tx("→", 40).move_to([(xp[2] + xp[3]) / 2 - 0.08, py, 0]))
        pl = VGroup(always_redraw(lambda: tx(f"mask {PF}{sup('(<i>t</i>)')},  <i>t</i> = {int(round(Pseq.k.get_value())) + 1}",
                                             20, AMBER).move_to([xp[1], ly, 0])),
                    tx("masked spectrum", 20, DIM).move_to([xp[2], ly, 0]))
        goodl = tx("ours, 6 frames", 20, GRN).move_to([xp[3], ly, 0])
        self.caption("Fix: a second SLM applies a content-adaptive mask per frame")
        self.remove(longarr, bad, badl)
        self.add(clock, Pseq.k, Mseq.k, Pseq.mob, Mseq.mob, rec.mob, rec.k, ops, pl, goodl,
                 framed(Pseq.mob, GHOST, 1), framed(Mseq.mob, GHOST, 1))
        self.play(FadeIn(s2), FadeIn(s2l), run_time=0.5)
        self.wait(2.0)
        # pupil-aware joint optimization: the view follows the pupil across the eyebox
        span = SPAN / EB * ps
        rp = R_P / EB * ps
        mc = Mseq.mob.get_center()
        pup = Circle(radius=rp, color=BLUE, stroke_width=4).move_to(mc)
        self.add(pup)
        self.caption("Both SLMs optimized jointly, for every pupil position")
        self.play(pup.animate.move_to(mc + LEFT * span), rec.k.animate.set_value(0), run_time=1.1, rate_func=smooth)
        self.play(pup.animate.move_to(mc + RIGHT * span), rec.k.animate.set_value(24), run_time=2.2, rate_func=linear)
        self.play(pup.animate.move_to(mc), rec.k.animate.set_value(12), run_time=1.1, rate_func=smooth)
        self.clear_all()


# ================================================================== T3 Payoff: captures, numbers, end card
class T3_Payoff(Base):
    def construct(self):
        self.header("Experimental captures (paper Fig. 1)")
        rows = [("Conventional", "teaser_spec_single.png", "teaser_single_center.png", "teaser_single_right.png", 1.3),
                ("Ours", "teaser_spec_multi.png", "teaser_multi_center.png", "teaser_multi_right.png", -1.4)]
        hs = 2.4
        pups = []
        for (name, sp, vc, vr, y) in rows:
            s = img(sp, h=hs).move_to([-3.35, y, 0])
            a = img(vc, h=hs).move_to([0.55, y, 0])
            b = img(vr, h=hs).move_to([4.65, y, 0])
            lab = tx(name, 24, GRN if name == "Ours" else WHITE).rotate(PI / 2).move_to([-6.2, y, 0])
            W = s.width
            pc = Circle(radius=0.188 * W, color=RED, stroke_width=3).move_to(s.get_center() + RIGHT * 0.006 * W)
            pb = Circle(radius=0.188 * W, color=BLUE, stroke_width=3).move_to(s.get_center() + RIGHT * 0.006 * W)
            self.add(s, a, b, lab, pc, pb, framed(a, RED, 2), framed(b, BLUE, 2))
            pups.append((pb, s.get_center() + RIGHT * 0.248 * W))
        top = 1.3 + hs / 2 + 0.22
        self.add(tx("Fourier plane", 22, DIM).move_to([-3.35, top, 0]),
                 tx("center view (red pupil)", 22, DIM).move_to([0.55, top, 0]),
                 tx("right view (blue pupil)", 22, DIM).move_to([4.65, top, 0]))
        self.caption("Real prototype: move the pupil, ours stays bright")
        self.wait(0.6)
        self.play(*[p.animate.move_to(t) for (p, t) in pups], run_time=2.2)
        self.wait(1.5)
        self.clear_all()

        # --- headline numbers
        big = tx("9×", 120, GRN, bold=True).move_to([-4.6, 0.55, 0])
        bl1 = tx("étendue", 32).next_to(big, DOWN, buff=0.2)
        bl2 = tx("3 × 3 lasers (paper Sec. 4)", 20, DIM).next_to(bl1, DOWN, buff=0.15)
        ttl = tx("9 × 9 light field, PSNR (simulation, paper Table 1)", 20, DIM).move_to([2.35, 2.1, 0])
        data = [("single source", 16.60, DIM), ("fixed random mask", 18.63, DIM),
                ("steered + filter, 9 frames", 23.95, DIM), ("ours, 6 frames", 24.65, GRN)]
        x0, sc = 1.3, 0.17
        bars = VGroup()
        for i, (nm, v, c) in enumerate(data):
            y = 1.25 - i * 0.75
            lab = tx(nm, 22, WHITE if c == GRN else "#CFCFCF").next_to([x0 - 0.15, y, 0], LEFT, buff=0)
            r = Rectangle(width=v * sc, height=0.45, stroke_width=0, fill_color=c, fill_opacity=0.85)
            r.move_to([x0 + v * sc / 2, y, 0])
            val = tx(f"{v:.2f} dB", 22, GRN if c == GRN else WHITE).next_to(r, RIGHT, buff=0.15)
            bars.add(VGroup(lab, r, val))
        self.add(big, bl1, bl2, ttl, *[b[0] for b in bars])
        self.caption("Best light-field quality, with fewer frames (6 vs. 9)")
        self.play(LaggedStart(*[GrowFromEdge(b[1], LEFT) for b in bars], lag_ratio=0.15), run_time=1.2)
        self.add(*[b[2] for b in bars])
        self.clear_all()

        # --- end card
        e1 = tx("Large Étendue 3D Holographic Display", 46, bold=True).move_to([0, 2.75, 0])
        e2 = tx("with Content-adaptive Dynamic Fourier Modulation", 32).move_to([0, 2.1, 0])
        s = gimg("ms_9.png", h=2.0).move_to([0, 0.35, 0])
        tag = tx("more sources for étendue · a dynamic Fourier mask for image quality", 24, DIM).move_to([0, -1.1, 0])
        a1 = tx("Brian Chao, Manu Gopakumar, Suyeon Choi, Jonghyun Kim, Liang Shi, Gordon Wetzstein", 22).move_to([0, -1.95, 0])
        a2 = tx("Stanford University · NVIDIA · MIT", 20, DIM).move_to([0, -2.45, 0])
        a3 = tx("SIGGRAPH Asia 2024 · arXiv 2409.03143 · bchao1.github.io/holo_dfm", 24, GRN).move_to([0, -3.1, 0])
        self.add(e1, e2, s, framed(s, GHOST), tag, a1, a2, a3)
        self.wait(5.5)
