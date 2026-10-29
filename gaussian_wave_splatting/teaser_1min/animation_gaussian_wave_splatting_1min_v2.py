"""Gaussian Wave Splatting — 1-minute teaser, v2.

Condensed from the 5-min teaser (../animation_gws_v4.py): helpers are copied, not imported.
Source: Choi*, Chao*, Yang, Gopakumar, Wetzstein, "Gaussian Wave Splatting for Computer-Generated
Holography", ACM TOG 44(4) (SIGGRAPH 2025), arXiv 2505.06582.
Assets are read (never written) from ../data (paper figure crops) and ../data/generated (wave-optics caches
built by the 5-min script). Render with ./render.sh; scenes in order: T1_Problem, T2_Method, T3_Payoff.
"""
from manim import *
import numpy as np
import matplotlib
import os
import re

DATA = "/home/brianchc/project_animations/gaussian_wave_splatting/data"
GEN = os.path.join(DATA, "generated")

config.background_color = BLACK

# ------------------------------------------------------------------ style (from the 5-min version)
FONT = "CMU Serif"
MATHFONT = "Latin Modern Math"
DIM = "#8A8A8A"
RED = "#FF5252"
GREEN = "#69F0AE"
BLUE = "#448AFF"
AMBER = "#FFCA28"
CYAN = "#4DD0E1"
PURPLE = "#E040FB"
CAPTION_Y = -3.5
MATHCH = set("×⊙→≈∼−∈ℝ≪≤↓↔∑∏∬∫ℱ𝒫𝒢𝒩ℋ∗≠∞∂∘⊤●⇒")


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


def I(v):
    return f"<i>{v}</i>"


def B(v):
    return f"<b>{v}</b>"


KB = B("k")
UHAT = "<i>Û</i>"
SQ = sup("2")


def gen(name):
    p = os.path.join(GEN, name)
    assert os.path.exists(p), f"missing cached asset {p} (run ../render.sh once to build it)"
    return p


def domain_rgb(u, gamma=0.6, amax=None, sat=0.85):
    a = np.abs(u)
    amax = amax or (np.percentile(a, 99.7) + 1e-12)
    v = np.clip(a / amax, 0, 1) ** gamma
    h = (np.angle(u) + np.pi) / (2 * np.pi)
    hsv = np.stack([h, np.full_like(h, sat), v], -1)
    return (matplotlib.colors.hsv_to_rgb(hsv) * 255).astype(np.uint8)


def grey(a, gamma=1.0):
    return (np.clip(a, 0, 1) ** gamma * 255).astype(np.uint8)


def wave_lut_g(g):
    """Diverging map for real wave fields: blue (−) · black (0) · amber (+)."""
    t = np.linspace(-1, 1, 256)[:, None]
    neg = np.array([0x29, 0x79, 0xFF]) / 255.0
    pos = np.array([0xFF, 0xCA, 0x28]) / 255.0
    rgb = np.where(t < 0, (-t) ** g * neg, t ** g * pos)
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


def wave_lut():
    t = np.linspace(-1, 1, 256)[:, None]
    neg = np.array([0x29, 0x79, 0xFF]) / 255.0
    pos = np.array([0xFF, 0xCA, 0x28]) / 255.0
    rgb = np.where(t < 0, -t * neg, t * pos) ** 0.8
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


WLUT = wave_lut()
WLUT2 = wave_lut_g(1.5)


def wave_rgb(re, scale):
    idx = np.clip((re / scale * 0.5 + 0.5) * 255, 0, 255).astype(np.uint8)
    return WLUT[idx]


def rgba(a):
    if a.ndim == 2:
        a = np.stack([a] * 3, -1)
    if a.shape[2] == 3:
        a = np.concatenate([a, np.full(a.shape[:2] + (1,), 255, np.uint8)], -1)
    return np.ascontiguousarray(a.astype(np.uint8))


def img(path, h=None, w=None):
    p = path if os.path.isabs(path) else os.path.join(DATA, path)
    m = ImageMobject(p)
    if h is not None:
        m.set(height=h)
    if w is not None:
        m.set(width=w)
    return m


def arr_img(a, h=None, w=None):
    m = ImageMobject(rgba(a))
    if h is not None:
        m.set(height=h)
    if w is not None:
        m.set(width=w)
    return m


def framed(m, color=WHITE, sw=1.5):
    return SurroundingRectangle(m, buff=0, color=color, stroke_width=sw)


def live(m, fn):
    m.add_updater(lambda mob: setattr(mob, "pixel_array", rgba(fn())))
    return m


def arrow(a, b, buff=0.1, color=WHITE, sw=3):
    return Arrow(a, b, buff=buff, color=color, stroke_width=sw,
                 max_tip_length_to_length_ratio=0.18, tip_length=0.16)


def label(m, s, size=22, color=WHITE, d=UP, buff=0.15):
    """Label above/below m, on a common baseline so labels in a row line up."""
    t = tx(s, size, color)
    plain = re.sub(r"<[^>]+>", "", s)
    desc = any(c in "gjpqy(),;[]{}|µφμβγηρχψξζ" for c in plain) or 'rise="-' in s
    dd, asc = 0.0027 * size, 0.0098 * size
    base = m.get_top()[1] + buff + dd if d is UP else m.get_bottom()[1] - buff - asc
    t.shift(UP * (base - (t.get_bottom()[1] + (dd if desc else 0))))
    t.set_x(m.get_center()[0])
    return t


# ------------------------------------------------------------------ scene base
class Base(Scene):
    def setup(self):
        self.cap = None
        self.clock = ValueTracker(0)
        self.clock.add_updater(lambda m, dt: m.increment_value(dt))
        self.add(self.clock)

    def _hold_caption(self):
        if self.cap is not None:
            left = self.cap_need - (self.renderer.time - self.cap_t0)
            if left > 0.05:
                self.wait(left)

    def caption(self, s, wait=None, size=28):
        self._hold_caption()
        if self.cap is not None:
            self.remove(self.cap)
        c = tx(s, size)
        if c.width > 13.2:
            c.scale_to_fit_width(13.2)
        c.move_to([0, CAPTION_Y, 0])
        self.cap = c
        plain = re.sub(r"<[^>]+>", "", s)
        self.cap_t0 = self.renderer.time
        self.cap_need = len(plain) / 17.0 + 0.8
        self.add(c)
        if wait:
            self.wait(wait)
        return c

    def uncaption(self):
        self._hold_caption()
        if self.cap is not None:
            self.remove(self.cap)
            self.cap = None

    def clear_all(self, run_time=0.45):
        self.uncaption()
        mobs = [m for m in self.mobjects if m is not self.clock]
        for m in mobs:
            m.clear_updaters()
        if mobs:
            self.play(*[FadeOut(m) for m in mobs], run_time=run_time)

    def run(self, tracker, target, t, rate=smooth):
        self.play(tracker.animate.set_value(target), run_time=t, rate_func=rate)


# ================================================================== T1 Problem
class T1_Problem(Base):
    def construct(self):
        clk = self.clock
        # 1. hook: title over a live coherent wavefield
        title = tx("Gaussian Wave Splatting", 66, bold=True).move_to([0, 2.95, 0])
        subt = tx("for Computer-Generated Holography", 36).move_to([0, 2.2, 0])
        auth = tx("Choi*, Chao*, Yang, Gopakumar, Wetzstein  ·  SIGGRAPH 2025", 22, DIM).move_to([0, -2.55, 0])
        U = np.load(gen("three_U.npy"))
        field = arr_img(wave_rgb(np.real(U), 1.6), h=3.5).move_to([0, -0.1, 0])
        live(field, lambda: wave_rgb(np.real(U * np.exp(-1j * 2 * np.pi * 0.9 * clk.get_value())), 1.6))
        box = framed(field, DIM, 1.2)
        self.add(title, subt, auth)
        self.play(FadeIn(field), Create(box), run_time=0.8)
        self.caption("A hologram recreates the light wave of a 3D scene.")
        self.clear_all()

        # 2. a hologram focuses at many depths
        hp = img(gen("holo_phase.png"), 3.8).move_to([-4.3, 0.4, 0])
        lhp = label(hp, "hologram (phase pattern)", 24)
        frames = np.load(gen("focus_frames.npy"))
        z = ValueTracker(0.0)

        def fr():
            v = np.clip(z.get_value(), 0, 45) / 0.5
            i0 = int(np.floor(v))
            i1 = min(i0 + 1, len(frames) - 1)
            w = v - i0
            return ((1 - w) * frames[i0] + w * frames[i1]).astype(np.uint8)
        rec = arr_img(frames[0], 3.8).move_to([4.3, 0.4, 0])
        live(rec, fr)
        rb = framed(rec, DIM, 1.2)
        lrec = label(rec, "light after a distance " + I("z"), 24)
        ar = arrow([-2.2, 0.4, 0], [2.2, 0.4, 0])
        la = tx("propagate", 22, DIM).next_to(ar, UP, buff=0.15)
        zval = always_redraw(lambda: tx(f"z = {z.get_value():.0f} mm", 26).next_to(rec, DOWN, buff=0.2))
        foc = tx("toy simulation: star at 12 mm, “3D” at 36 mm", 18, DIM).next_to(ar, DOWN, buff=0.2)
        self.play(FadeIn(hp), GrowArrow(ar), FadeIn(rec), Create(rb), run_time=0.6)
        self.add(lhp, lrec, la, zval, foc)
        self.caption("Holograms need the full wavefield of a 3D scene.")
        self.run(z, 12, 1.6)
        self.wait(0.6)
        self.run(z, 36, 1.8)
        self.wait(1.0)
        self.clear_all()

        # 3. what 3D input? points / meshes vs Gaussian splats
        h = 2.75
        xs = [-4.5, 0, 4.5]
        names = ["point cloud", "mesh", "Gaussian splats"]
        files = ["fig5_lego_target_point.png", "fig5_lego_target_polygon.png", "fig5_lego_target_exact.png"]
        notes = [("holes, poor occlusion", RED), ("overly smooth", RED), ("photoreal, from a few photos", GREEN)]
        ims = [img(f, h).move_to([x, 0.55, 0]) for f, x in zip(files, xs)]
        labs = [(label(m, n, 26), label(m, nt, 22, c, DOWN)) for m, n, (nt, c) in zip(ims, names, notes)]
        src = tx("target renderings of each 3D representation (paper Fig. 5)", 18, DIM).move_to([0, -1.75, 0])
        self.play(FadeIn(ims[0]), FadeIn(ims[1]), run_time=0.4)
        self.add(*labs[0], *labs[1], src)
        self.caption("Classic CGH uses points or meshes: holes, blur.")
        self.play(FadeIn(ims[2]), run_time=0.4)
        self.add(*labs[2])
        self.caption("Neural rendering gives photoreal Gaussians. Can holograms use them?")
        self.clear_all()


# ================================================================== T2 Method
class T2_Method(Base):
    def construct(self):
        clk = self.clock
        # 4. key fact: Gaussian <-> Gaussian (live 2D FFT)
        N = 256
        th = ValueTracker(0.0)
        asp = ValueTracker(1.0)
        mx = ValueTracker(0.0)
        yy, xx = np.mgrid[0:N, 0:N].astype(float)
        xx, yy = xx - N / 2, N / 2 - yy
        s0 = 9.0

        def space_full():
            c, s = np.cos(th.get_value()), np.sin(th.get_value())
            X, Y = xx - mx.get_value(), yy - 0.5 * mx.get_value()
            u, v = c * X + s * Y, -s * X + c * Y
            return np.exp(-0.5 * ((u / (s0 * asp.get_value())) ** 2 + (v / s0) ** 2))

        def space():
            return grey(space_full()[80:176, 80:176])

        def spec():
            F = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(space_full())))[104:152, 104:152]
            return domain_rgb(F, 0.7, amax=np.abs(F).max())
        sp = arr_img(space(), 3.2).move_to([-3.4, 0.6, 0])
        fq = arr_img(spec(), 3.2).move_to([3.4, 0.6, 0])
        live(sp, space)
        live(fq, spec)
        sb, fb = framed(sp, DIM, 1.2), framed(fq, DIM, 1.2)
        ls = label(sp, "a Gaussian", 26)
        lf = label(fq, "its Fourier transform (hue = phase)", 26)
        ar = arrow([-1.6, 0.6, 0], [1.6, 0.6, 0])
        arl = tx("ℱ", 32).next_to(ar, UP, buff=0.1)
        self.play(FadeIn(sp), FadeIn(fq), Create(sb), Create(fb), GrowArrow(ar), run_time=0.6)
        self.add(ls, lf, arl)
        self.caption("Key fact: a Gaussian's Fourier transform is a Gaussian.")
        self.run(asp, 2.2, 1.3)
        self.run(th, PI / 3, 1.3)
        self.run(mx, 20, 1.3)
        self.clear_all()

        # 5. a tilted splat's wavefield, travelling to the SLM (computed Huygens sum)
        U = np.load(gen("tilted_U.npy"))
        disk = np.load(gen("tilted_disk.npy"))

        def side():
            re_ = np.real(U * np.exp(-1j * 2 * PI * 0.8 * clk.get_value()))
            a = WLUT2[np.clip((re_ / 1.05 * 0.5 + 0.5) * 255, 0, 255).astype(np.uint8)].astype(np.float32)
            a = a * (1 - disk[..., None]) + 255 * disk[..., None]
            return a.astype(np.uint8)
        sv = arr_img(side(), w=9.6).move_to([-0.4, 0.45, 0])
        live(sv, side)
        slm = Rectangle(width=0.16, height=sv.height, fill_color="#546E7A", fill_opacity=1, stroke_width=0
                        ).next_to(sv, RIGHT, buff=0.05)
        lslm = tx("SLM", 22, DIM).next_to(slm, UP, buff=0.12)
        lpw = tx("plane wave", 22).next_to(sv, UP, buff=0.12).align_to(sv, LEFT)
        lg = tx("tilted Gaussian splat", 22).next_to(sv, DOWN, buff=0.12).align_to(sv, LEFT).shift(RIGHT * 0.6)
        lr = tx("its wavefield", 22, AMBER).next_to(sv, DOWN, buff=0.12).align_to(slm, RIGHT)
        self.play(FadeIn(sv), FadeIn(slm), run_time=0.5)
        self.add(lslm, lpw, lg, lr)
        self.caption("So each splat's wavefield has a closed form.")
        self.wait(0.8)
        self.clear_all()

        # 6. the Gaussian Wave Splatting equation, term by term, + one splat's field
        parts = [
            (f"{UHAT}{sub(I('i'))}({KB}) = 2π ", WHITE, None),
            ("det(" + B("J") + ")", AMBER, "tilt"),
            (" det(" + I("S") + sub(I("i")) + ")", GREEN, "scale"),
            (f" exp(−2π{SQ} {KB}{sup('T')} Σ{sub(I('i'))} {KB})", CYAN, "Gaussian shape"),
            (f" exp({I('j')} {KB}·<b>μ</b>{sub(I('i'))})", PURPLE, "position and depth"),
        ]
        EY = 2.35
        full = tx("".join(col(p, c) for p, c, _ in parts), 40).move_to([0, EY, 0])
        x0 = full.get_left()[0]
        shown, prev_w = None, 0.0
        for k in range(1, len(parts) + 1):
            m = tx("".join(col(p, c) for p, c, _ in parts[:k]), 40)
            m.move_to([x0, EY, 0], aligned_edge=LEFT).set_y(EY)
            if shown is not None:
                self.remove(shown)
            self.add(m)
            shown = m
            _, c, name = parts[k - 1]
            if name:
                self.add(tx(name, 20, c).move_to([x0 + (prev_w + m.width) / 2, EY - 0.62, 0]))
                self.wait(0.3)
            prev_w = m.width
        self.caption("One formula per splat, no per-primitive FFT.")
        gs = img(gen("gws_slm.png"), 2.6).move_to([-3.0, -0.85, 0])
        gr = img(gen("gws_rec.png"), 2.6).move_to([3.0, -0.85, 0])
        gsf, grf = framed(gs, DIM, 1.2), framed(gr, DIM, 1.2)
        lgs = label(gs, "its field at the hologram plane", 20, DIM)
        lgr = label(gr, "focused back: the splat", 20, DIM)
        ar2 = arrow(gs.get_right() + RIGHT * 0.25, gr.get_left() + LEFT * 0.25)
        self.play(FadeIn(gs), Create(gsf), GrowArrow(ar2), FadeIn(gr), Create(grf), run_time=0.6)
        self.add(lgs, lgr)
        self.wait(1.2)
        self.clear_all()

        # 7. occlusion: alpha wave blending (toy)
        h = 3.1
        xs = [-4.4, 0, 4.4]
        keys = ["naive", "alpha", "target"]
        names = ["no occlusion", "alpha wave blending", "ray-traced target"]
        colors = [RED, GREEN, WHITE]
        ims = [img(gen(f"occ_{k}.png"), h).move_to([x, 0.5, 0]) for k, x in zip(keys, xs)]
        self.play(*[FadeIn(m) for m in ims], run_time=0.4)
        self.add(*[label(m, n, 26, c) for m, n, c in zip(ims, names, colors)])
        self.add(tx("toy simulation: red splat in front of a blue one", 18, DIM).move_to([0, -1.5, 0]))
        self.caption("Alpha wave blending: near splats occlude far ones.")
        self.wait(0.8)
        self.clear_all()

        # 8. fast: every frequency in parallel
        n, size = 10, 2.6

        def grid(cx):
            cs = size / n
            return VGroup(*[Square(cs * 0.86).set_stroke(DIM, 1).set_fill(BLUE, 0.0).move_to(
                [cx + (j + 0.5) * cs - size / 2, 0.75 + size / 2 - (i + 0.5) * cs, 0])
                for i in range(n) for j in range(n)])
        ge, gf = grid(-3.6), grid(3.6)
        le = tx("exact: one Gaussian at a time", 24).next_to(ge, UP, buff=0.2)
        lf = tx("fast: all frequencies at once", 24, AMBER).next_to(gf, UP, buff=0.2)
        self.play(FadeIn(ge), FadeIn(gf), run_time=0.4)
        self.add(le, lf)
        self.caption("In the Fourier domain, all frequencies run in parallel.")
        self.play(*[s.animate.set_fill(AMBER, 0.85) for s in gf], run_time=0.5)
        cnt = ValueTracker(0)
        ctext = always_redraw(lambda: tx(f"Gaussian {int(cnt.get_value())} of many", 20, DIM)
                              .next_to(ge, DOWN, buff=0.15))
        self.add(ctext)
        tf = tx("6.84 s", 30, AMBER).next_to(gf, DOWN, buff=0.15)
        self.play(cnt.animate.set_value(6), *[s.animate.set_fill(BLUE, 0.35) for s in ge], run_time=1.8,
                  rate_func=linear)
        ctext.clear_updaters()
        self.remove(ctext)
        te = tx("180 s", 30).next_to(ge, DOWN, buff=0.15)
        self.add(te, tf)
        big = tx("nearly 30× faster", 40, AMBER, bold=True).move_to([0, -2.05, 0])
        src = tx("1024 × 1280 hologram, 15k Gaussians (paper Table 2)", 18, DIM).next_to(big, DOWN, buff=0.15)
        self.add(big, src)
        self.wait(1.6)
        self.clear_all()


# ================================================================== T3 Payoff
class T3_Payoff(Base):
    def construct(self):
        # 9. captured comparison + headline number
        w = 4.1
        xs = [-4.4, 0, 4.4]
        keys = ["point", "polygon", "exact"]
        names = ["point-based CGH", "polygon-based CGH", "Gaussian Wave Splatting"]
        colors = [WHITE, WHITE, AMBER]
        ims = [img(f"fig5_lego_captured_{k}.png", w=w).move_to([x, 1.2, 0]) for k, x in zip(keys, xs)]
        self.play(*[FadeIn(m) for m in ims], run_time=0.5)
        self.add(*[label(m, n, 24, c) for m, n, c in zip(ims, names, colors)])
        src = tx("experimentally captured on a holographic display prototype (paper Fig. 5)", 18, DIM
                 ).next_to(ims[1], DOWN, buff=0.15)
        self.add(src)
        self.caption("Photoreal 3D holograms on a real display.")
        big = tx("5–11 dB higher PSNR", 44, AMBER, bold=True).move_to([0, -1.55, 0])
        bsub = tx("than point / polygon CGH (simulation, paper Table 1)", 22, DIM).next_to(big, DOWN, buff=0.15)
        self.add(big, bsub)
        self.wait(4.5)
        self.clear_all()

        # 10. captured focal stack of one hologram
        big = img("fig6_v0_rear.png", 5.0).move_to([0, 0.35, 0])
        src = tx("experimentally captured, Mip-NeRF 360 “garden” (paper Fig. 6)", 18, DIM).next_to(big, UP, buff=0.15)
        flab = tx("rear focus", 26, AMBER).next_to(big, DOWN, buff=0.18)
        self.play(FadeIn(big), run_time=0.5)
        self.add(src, flab)
        self.caption("One captured hologram, refocused through the scene.")
        self.wait(0.6)
        for s, nm in [("mid", "middle"), ("front", "front")]:
            nxt = img(f"fig6_v0_{s}.png", 5.0).move_to(big)
            self.play(FadeIn(nxt), run_time=0.7)
            self.remove(big)
            big = nxt
            self.remove(flab)
            flab = tx(f"{nm} focus", 26, AMBER).next_to(big, DOWN, buff=0.18)
            self.add(flab)
            self.wait(1.1)
        self.clear_all()

        # 11. end card
        t1 = tx("Gaussian Wave Splatting", 56, bold=True).move_to([0, 2.1, 0])
        t1b = tx("for Computer-Generated Holography", 34).move_to([0, 1.35, 0])
        t4 = tx("Suyeon Choi*, Brian Chao*, Jacqueline Yang, Manu Gopakumar, Gordon Wetzstein", 24).move_to([0, 0.35, 0])
        t5 = tx("Stanford University  ·  ACM TOG 44(4), SIGGRAPH 2025", 22, DIM).move_to([0, -0.15, 0])
        t2 = tx("arXiv 2505.06582  ·  bchao1.github.io/gaussian-wave-splatting", 28, AMBER).move_to([0, -1.2, 0])
        t3 = tx("code: github.com/computational-imaging/hsplat", 24, DIM).move_to([0, -1.75, 0])
        t6 = tx("* equal contribution", 18, DIM).move_to([0, -3.4, 0])
        self.add(t1, t1b, t4, t5, t2, t3, t6)
        self.wait(5.0)
