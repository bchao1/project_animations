"""Random-phase Wave Splatting — 1-minute teaser, v1.

Condensed from the 5-min teaser (../animation_rpws_v3.py): helpers are copied, not imported.
Source: Chao*, Yang*, Choi, Gopakumar, Koiso, Wetzstein, "Random-phase Wave Splatting of Translucent Primitives
for Computer-generated Holography", arXiv 2508.17480 (2025).
Assets are read (never written) from ../data (paper figure crops) and ../data/generated (toy wave-optics caches
built by the 5-min script). Render with ./render.sh; scenes in order: T1_Hook, T2_Method, T3_Payoff.
"""
from manim import *
import numpy as np
import matplotlib
import os
import re
from scipy.ndimage import gaussian_filter

DATA = "/home/brianchc/project_animations/random_phase_wave_splatting/data"
GEN = os.path.join(DATA, "generated")

config.background_color = BLACK

# ------------------------------------------------------------------ style (from the 5-min version)
FONT = "CMU Serif"
MATHFONT = "Latin Modern Math"          # CMU Serif lacks × − → ≈ ∑ ∏ ℱ 𝒫 𝔼 ∠ …
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
MATHCH = set("×⊙→≈∼−∈ℝ≪≤↓↔∑∏∬∫ℱ𝒫𝒢𝒩ℋ𝒪𝒰𝔼∗≠∞∂∘⊤●⇒∠✓")
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


def supsub(base, s_):
    """Superscript 'base' followed by a subscript that stays inside the superscript (Pango rise is absolute)."""
    return (f'<span rise="4500"><small><small>{base}</small></small></span>'
            f'<span rise="2600"><small><small><small>{s_}</small></small></small></span>')


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


XB = B("x")
KB = B("k")
SQ = sup("2")
W1, W2 = I("w") + sub("1"), I("w") + sub("2")
PH1, PH2 = "φ" + sub("1"), "φ" + sub("2")
A1, A2 = "α" + sub("1"), "α" + sub("2")
C1, C2 = I("c") + sub("1"), I("c") + sub("2")


EQ_CROSS = (f"|{W1} {I('e')}{supsub(I('i') + 'φ', '1')} + {W2} {I('e')}{supsub(I('i') + 'φ', '2')}|{SQ} = {W1}{SQ} + {W2}{SQ} + "
            f"{col('2' + W1 + W2 + ' cos(' + PH1 + ' − ' + PH2 + ')', AMBER)}")


# toy-simulation constants (must match the 5-min script that built the caches)
TFR = 24
R_SUB = 0.22
S_MAX = 0.72
NV = 41


def gen(name):
    return os.path.join(GEN, name)


def cmap_rgb(a, cmap="inferno"):
    return (matplotlib.colormaps[cmap](np.clip(a, 0, 1))[..., :3] * 255).astype(np.uint8)


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
    """Attach an updater that rewrites the image pixels every frame."""
    m.add_updater(lambda mob: setattr(mob, "pixel_array", rgba(fn())))
    return m


def interp(frames, v):
    """Linear interpolation into a stack of uint8 frames at fractional index v."""
    v = float(np.clip(v, 0, len(frames) - 1))
    i0 = int(np.floor(v))
    i1 = min(i0 + 1, len(frames) - 1)
    w = v - i0
    return ((1 - w) * frames[i0] + w * frames[i1]).astype(np.uint8)


def arrow(a, b, buff=0.1, color=WHITE, sw=3):
    return Arrow(a, b, buff=buff, color=color, stroke_width=sw,
                 max_tip_length_to_length_ratio=0.18, tip_length=0.16)


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

    # Captions stay up long enough to read: ~17 characters per second + 0.8 s.
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
        s = re.sub(r"(\d) × (\d)", "\\1\u00a0×\u00a0\\2", s)
        s = s.replace("√(1 − α)", "√(1\u00a0−\u00a0α)")
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
    dd, asc = 0.0027 * size, 0.0098 * size        # CMU Serif descender / ascender depth per font-size unit
    base = m.get_top()[1] + buff + dd if d is UP else m.get_bottom()[1] - buff - asc
    t.shift(UP * (base - (t.get_bottom()[1] + (dd if desc else 0))))
    t.set_x(m.get_center()[0])
    return t


def toy_note(s="toy simulation: λ = 520 nm, 8 µm pixels"):
    return tx(s, 16, DIM)


# ================================================================== T1 Hook: title + eyebox
class T1_Hook(Base):
    def construct(self):
        title = tx("Random-phase Wave Splatting", 58, bold=True).move_to([0, 2.95, 0])
        subt = tx("of Translucent Primitives for Computer-generated Holography", 30).move_to([0, 2.25, 0])
        auth = tx("Brian Chao*, Jacqueline Yang*, Suyeon Choi, Manu Gopakumar, Ryota Koiso, Gordon Wetzstein",
                  20, DIM).move_to([0, -2.2, 0])
        aff = tx("Stanford University  ·  KDDI Research  ·  arXiv 2508.17480 (2025)", 20, DIM).move_to([0, -2.6, 0])
        cum = np.load(gen("tm_cum.npy"))
        frames = np.load(gen("tm_frames.npy"))
        t = ValueTracker(0.0)
        a = arr_img(frames[0], 2.7).move_to([-1.7, -0.05, 0])
        b = arr_img(cum[0], 2.7).move_to([1.7, -0.05, 0])
        live(a, lambda: frames[int(np.clip(t.get_value(), 0, TFR - 1))])
        live(b, lambda: cum[int(np.clip(t.get_value(), 0, TFR - 1))])
        la = label(a, "one random-phase frame", 20, DIM)
        lb = always_redraw(lambda: label(b, f"average of {int(np.clip(t.get_value(), 0, TFR - 1)) + 1} frames", 20, DIM))
        self.add(title, subt, auth, aff)
        self.play(FadeIn(a), FadeIn(b), run_time=0.5)
        self.add(la, lb, framed(a, DIM, 1.2), framed(b, DIM, 1.2))
        self.caption("Holograms from splats, done with random phase and many frames.")
        self.run(t, TFR - 0.01, 3.8, rate=lambda x: x ** 1.6)
        self.clear_all(run_time=0.4)

        # eyebox: smooth (GWS) vs random (RPWS), pupil sliding across the spectrum
        h = 2.8
        lf_s = np.load(gen("lf_smooth.npy"))
        lf_r = np.load(gen("lf_rpws.npy"))
        s = ValueTracker(0.0)
        idx = lambda: (s.get_value() + S_MAX) / (2 * S_MAX) * (NV - 1)
        xs = [-5.2, -2.2, 2.2, 5.2]
        spec_s = img(gen("spec7_smooth.png"), h).move_to([xs[0], 0.55, 0])
        spec_r = img(gen("spec7_rand.png"), h).move_to([xs[2], 0.55, 0])
        vs = arr_img(lf_s[NV // 2], h).move_to([xs[1], 0.55, 0])
        vr = arr_img(lf_r[NV // 2], h).move_to([xs[3], 0.55, 0])
        live(vs, lambda: interp(lf_s, idx()))
        live(vr, lambda: interp(lf_r, idx()))
        g1 = tx("smooth phase (GWS)", 28, RED).move_to([(xs[0] + xs[1]) / 2, 2.75, 0])
        g2 = tx("random phase (RPWS)", 28, GREEN).move_to([(xs[2] + xs[3]) / 2, 2.75, 0])
        labs = [label(spec_s, "eyebox", 22, DIM), label(vs, "what the eye sees", 22, DIM),
                label(spec_r, "eyebox", 22, DIM), label(vr, "what the eye sees", 22, DIM)]
        rad = R_SUB * h / 2

        def circ(spec):
            return always_redraw(lambda: Circle(radius=rad, color=CYAN, stroke_width=5).move_to(
                spec.get_center() + RIGHT * s.get_value() * h / 2))
        sep = Line([0, 2.9, 0], [0, -1.4, 0], color=DIM, stroke_width=1.5)
        self.play(FadeIn(spec_s), FadeIn(spec_r), FadeIn(vs), FadeIn(vr), run_time=0.5)
        self.add(g1, g2, sep, *labs, circ(spec_s), circ(spec_r), framed(vs, DIM, 1.2), framed(vr, DIM, 1.2))
        self.add(tx("toy simulation: star 6 mm and checkerboard 18 mm from the SLM, λ = 520 nm, 8 µm pixels, 24 frames",
                    16, DIM).move_to([0, -1.75, 0]))
        self.caption("Smooth-phase holograms send light straight ahead, into a tiny eyebox.")
        self.run(s, -S_MAX, 2.2, smooth)
        self.caption("Move your eye, and the image vanishes.")
        self.run(s, S_MAX, 3.2, smooth)
        self.caption("Random phase fills the whole eyebox: bright everywhere, with parallax.")
        self.run(s, -S_MAX, 3.2, smooth)
        self.run(s, 0.0, 1.4, smooth)
        self.clear_all(run_time=0.4)


# ================================================================== T2 Method: the cross term + toy
class T2_Method(Base):
    def construct(self):
        eq = tx(EQ_CROSS, 38).move_to([0, 2.3, 0])
        lab = tx("cross term", 22, AMBER).next_to(eq, DOWN, buff=0.12).align_to(eq, RIGHT).shift(LEFT * 1.0)
        self.add(eq, lab)
        self.caption("But random phase breaks alpha blending: overlapping waves pick up a cross term.")
        l1 = tx(f"smooth phase:  cos = 1  ⇒  amplitudes add (GWS)", 28, RED).move_to([0, 0.75, 0])
        l2 = tx(f"random phase:  𝔼[cos] = 0  ⇒  intensities add, on average", 28, GREEN).move_to([0, 0.0, 0])
        self.hold()
        self.add(l1, l2)
        self.caption("Key insight: blend in intensity (√ weights), and let frame averaging cancel the cross term.")
        self.clear_all(run_time=0.4)

        gt = np.load(gen("toy_gt.npy"))
        nrp = np.load(gen("toy_nrp.npy")).astype(np.float32)
        rp = np.load(gen("toy_rpws.npy")).astype(np.float32)
        t = ValueTracker(0.0)
        ti = lambda: int(np.clip(t.get_value(), 0, TFR - 1))
        disp = lambda a: (np.clip(a, 0, 1) ** (1 / 1.8) * 255).astype(np.uint8)
        gtb = gaussian_filter(gt, (6, 6, 0))
        errs = {}

        def err(which, i):
            if (which, i) not in errs:
                a = (nrp if which == "n" else rp)[i]
                errs[(which, i)] = cmap_rgb(np.abs(gaussian_filter(a, (6, 6, 0)) - gtb).sum(-1) / 0.9, "inferno")
            return errs[(which, i)]
        h = 2.8
        xs = [-4.4, 0, 4.4]
        mg = arr_img(disp(gt), h).move_to([xs[0], 1.0, 0])
        mn = arr_img(disp(nrp[0]), h).move_to([xs[1], 1.0, 0])
        mr = arr_img(disp(rp[0]), h).move_to([xs[2], 1.0, 0])
        live(mn, lambda: disp(nrp[ti()]))
        live(mr, lambda: disp(rp[ti()]))
        en = arr_img(err("n", 0), 1.3).next_to(mn, DOWN, buff=0.35)
        er = arr_img(err("r", 0), 1.3).next_to(mr, DOWN, buff=0.35)
        live(en, lambda: err("n", ti()))
        live(er, lambda: err("r", ti()))
        self.play(FadeIn(mg), FadeIn(mn), FadeIn(mr), run_time=0.5)
        self.add(label(mg, "target", 26), label(mn, "naïve random phase", 26, RED), label(mr, "RPWS", 26, GREEN))
        self.add(en, er, framed(en, DIM, 1), framed(er, DIM, 1),
                 tx("error", 20, DIM).next_to(en, LEFT, buff=0.2), tx("error", 20, DIM).next_to(er, LEFT, buff=0.2))
        cnt = always_redraw(lambda: tx(f"frames averaged: {ti() + 1}", 26).move_to([xs[0], -1.05, 0]))
        self.add(cnt, tx("toy simulation: two translucent Gaussians, orange in front of blue", 16, DIM).move_to([0, -2.55, 0]))
        self.caption("Average frames: naïve random phase stays wrong; RPWS converges to the target.")
        self.run(t, TFR - 0.01, 4.5, rate=lambda x: x ** 1.5)
        self.caption("Time multiplexing, composited back to front, is what makes the blending exact.")
        for m in (mn, mr, en, er):
            m.clear_updaters()
        self.clear_all(run_time=0.4)


# ================================================================== T3 Payoff + end card
class T3_Payoff(Base):
    def construct(self):
        w = 6.3
        a = img("teaser_gws_front.png", w=w).move_to([-3.3, 0.85, 0])
        b = img("teaser_rpws_back.png", w=w).move_to([3.3, 0.85, 0])
        self.play(FadeIn(a), FadeIn(b), run_time=0.5)
        self.add(tx("simulated holograms, paper Fig. 1", 18, DIM).next_to(a, DOWN, buff=0.12).set_x(0))
        head = tx("+10.5 dB light-field PSNR over GWS", 36, AMBER).move_to([0, -1.75, 0])
        sub_ = tx("24.87 vs 14.35 dB, Blender scenes, 24 frames (paper Table 1)", 20, DIM).move_to([0, -2.35, 0])
        self.add(head, sub_)
        self.caption("Natural defocus blur and a full eyebox, from Gaussians or triangles.")
        self.hold(1.0)
        self.clear_all(run_time=0.4)
        t1 = tx("Random-phase Wave Splatting", 48, bold=True).move_to([0, 1.8, 0])
        t1b = tx("of Translucent Primitives for Computer-generated Holography", 30).move_to([0, 1.1, 0])
        t2 = tx("arXiv 2508.17480", 30).move_to([0, 0.15, 0])
        t4 = tx("Brian Chao*, Jacqueline Yang*, Suyeon Choi, Manu Gopakumar, Ryota Koiso, Gordon Wetzstein",
                22, DIM).move_to([0, -0.9, 0])
        t5 = tx("Stanford University  ·  KDDI Research  ·  * equal contribution", 20, DIM).move_to([0, -1.35, 0])
        self.add(t1, t1b, t2, t4, t5)
        self.wait(4.5)
