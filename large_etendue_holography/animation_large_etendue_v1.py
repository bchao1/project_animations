"""Large Étendue 3D Holographic Display with Content-adaptive Dynamic Fourier Modulation
— 3Blue1Brown-style teaser, v1.

Source: Chao, Gopakumar, Choi, Kim, Shi, Wetzstein, SIGGRAPH Asia 2024, arXiv 2409.03143.
Paper figures in data/ are extracted from the arXiv source (large-etendue-holography/pdfs/src).
data/generated/toy_cgh.npz comes from precompute_toy_cgh.py (toy re-implementation, labelled
"toy simulation" on screen; every number shown is from the paper).

Render (see render.sh):
    manim -ql --media_dir media animation_large_etendue_v1.py <Scene>
Scenes, in order: S00_Title ... S09_Outlook.
"""
from manim import *
import numpy as np
from PIL import Image
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
GEN = os.path.join(DATA, "generated")

config.background_color = BLACK

# ------------------------------------------------------------------ style
FONT = "CMU Serif"
MATHFONT = "Latin Modern Math"          # CMU Serif lacks × → ≈ − ∈ ℱ ℋ 𝒫 𝒰 ...
GRN = "#39E27D"                          # laser green (light, waves, spectra)
RED = "#FF5252"
BLUE = "#448AFF"
AMBER = "#FFCA28"                        # amplitude SLM / eyebox
CYAN = "#4DD0E1"                         # lenses
PURPLE = "#E040FB"
DIM = "#8A8A8A"
GHOST = "#3A3A3A"
SLMC = "#90A4AE"
SRC3 = ["#FFB74D", GRN, "#4FC3F7"]       # tints marking the 3 sources of one row

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


def col(x, c):
    return f'<span foreground="{c}">{x}</span>'


def tx(s, size=30, color=WHITE, bold=False, italic=False):
    if bold:
        s = f"<b>{s}</b>"
    if italic:
        s = f"<i>{s}</i>"
    return MarkupText(mathify(s), font=FONT, font_size=size, color=color)


PHI = "<i>φ</i>"
LAM = "<i>λ</i>"
TH = "<i>θ</i>"
PF = "𝒫"


# ------------------------------------------------------------------ colour maps
def gmap(v):
    """[0,1] → laser-green RGBA uint8 (crests white-green)."""
    v = np.clip(v, 0, 1)
    r = 0.10 * v + 0.75 * v ** 3
    g = v ** 0.85
    b = 0.25 * v + 0.60 * v ** 3
    a = np.stack([r, g, b, np.ones_like(v)], -1)
    return (a * 255).astype(np.uint8)


def graymap(v, tint=(1.0, 1.0, 1.0)):
    v = np.clip(v, 0, 1)
    a = np.stack([v * tint[0], v * tint[1], v * tint[2], np.ones_like(v)], -1)
    return (a * 255).astype(np.uint8)


def logspec(s, ref=None, decades=3.2):
    ref = s.max() if ref is None else ref
    return np.clip((np.log10(s / ref + 1e-12) + decades) / decades, 0, 1)


def gen(name):
    return os.path.join(GEN, name)


# ------------------------------------------------------------------ assets
TOY = np.load(gen("toy_cgh.npz"))
N_, S_, M_, CELL, EB, R_P, SPAN, NF = [int(v) for v in TOY["meta"]]
C_ = M_ // 2
SRC_ORDER = [(0, 0), (0, 1), (0, -1), (-1, 0), (1, 0), (-1, 1), (1, -1), (-1, -1), (1, 1)]


def build_assets():
    os.makedirs(GEN, exist_ok=True)
    if os.path.exists(gen("done_v1")):
        return
    # --- multi-source spectrum build-up (S00, S05): shifted copies of the single-source spectrum
    base = TOY["I1_spec_raw"].astype(np.float64)
    ref = base.max()
    acc = np.zeros_like(base)
    for k, (dy, dx) in enumerate(SRC_ORDER):
        acc = acc + np.roll(base, (dy * CELL, dx * CELL), axis=(0, 1))
        Image.fromarray(gmap(logspec(acc, ref))).save(gen(f"ms_{k + 1}.png"))
    Image.fromarray(gmap(logspec(base, ref))).save(gen("spec_single.png"))
    # single source behind a Fourier filter (only the central order cell passes)
    filt = np.zeros_like(base)
    sl = slice(C_ - CELL // 2, C_ + CELL // 2)
    filt[sl, sl] = base[sl, sl]
    Image.fromarray(gmap(logspec(filt, ref))).save(gen("spec_single_filt.png"))
    # --- Fourier modulation (S06): spectrum without mask, masks P^(t), masked spectra
    e = slice(C_ - EB // 2, C_ + EB // 2)
    nom = TOY["V6_spec_nomask"][e, e].astype(np.float64)
    r2 = nom.max()
    Image.fromarray(gmap(logspec(nom, r2))).save(gen("v6_nomask.png"))
    frames = TOY["V6_spec_frames"]
    for t in range(frames.shape[0]):
        Image.fromarray(gmap(logspec(frames[t][e, e] + 1e-9, r2))).save(gen(f"v6_masked_{t}.png"))
        P = TOY["V6_P"][t]
        Image.fromarray(graymap(P, (1.0, 0.85, 0.35))).resize((400, 400), Image.NEAREST).save(gen(f"v6_P_{t}.png"))
    # fixed random binary mask of configuration IV (same generator as precompute_toy_cgh.py)
    import torch
    g = torch.Generator().manual_seed(3)
    fixed = (torch.rand(1, NF, NF, generator=g) > 0.5).float().numpy()[0]
    Image.fromarray(graymap(fixed, (1.0, 0.85, 0.35))).resize((400, 400), Image.NEAREST).save(gen("iv_P.png"))
    ivs = TOY["IV1_spec"][e, e].astype(np.float64)
    Image.fromarray(gmap(logspec(ivs + 1e-9, r2))).save(gen("iv_masked.png"))
    # --- toy views (center of the pupil path = index 12)
    for key in ("I1", "III1", "IV1", "V1", "V6", "target"):
        v = TOY[f"{key}_path"].astype(np.float32)
        Image.fromarray(gmap(v[12])).save(gen(f"view_{key}_c.png"))
    # --- crops of paper Fig. 6 (simulation) and Fig. 7 (experiment)
    sim = Image.open(os.path.join(DATA, "sim_full.png")).convert("RGB")
    for r, nm in enumerate(["I", "II", "III", "IV", "VI", "VII", "V1", "V6"]):
        y0 = r * 510
        sim.crop((0, y0, 800, y0 + 500)).save(gen(f"sim_{nm}_center.png"))
        sim.crop((2640, y0, 3440, y0 + 500)).save(gen(f"sim_{nm}_right.png"))
    ex = Image.open(os.path.join(DATA, "exp_scene1.png")).convert("RGB")
    for c, nm in enumerate(["I", "III", "IV", "V1", "V6"]):
        x0 = c * 810
        ex.crop((x0, 510, x0 + 800, 1010)).save(gen(f"exp_{nm}_front.png"))
        ex.crop((x0, 1020, x0 + 800, 1415)).save(gen(f"exp_{nm}_right.png"))
    open(gen("done_v1"), "w").write("ok")


build_assets()


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


# ------------------------------------------------------------------ optics widgets
def lens(x, y, h, color=CYAN):
    e = Ellipse(width=0.32, height=h, stroke_color=color, stroke_width=2.5,
                fill_color=color, fill_opacity=0.12).move_to([x, y, 0])
    return e


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


def eye(x, y, r=0.42, facing=LEFT):
    ball = Circle(radius=r, stroke_color=WHITE, stroke_width=2).move_to([x, y, 0])
    pupil = Ellipse(width=0.12, height=0.36, fill_color=WHITE, fill_opacity=1,
                    stroke_width=0).move_to([x - r * 0.93 if facing is LEFT else x + r * 0.93, y, 0])
    return VGroup(ball, pupil)


def arrow(a, b, buff=0.1, color=WHITE, sw=3):
    return Arrow(a, b, buff=buff, color=color, stroke_width=sw,
                 max_tip_length_to_length_ratio=0.18, tip_length=0.16)


# ------------------------------------------------------------------ live wave field
class WaveField:
    """Computed 2-D (transverse × propagation) monochromatic field of a 1-D phase SLM.

    Left of the SLM: a plane wave. Right: angular-spectrum propagation of
    aperture · exp(i φ(x)). Units are array pixels; λ = lam px (exaggerated vs. real SLMs).
    """

    def __init__(self, center, height=4.2, H=300, W=620, zs=150, lam=6.0):
        self.H, self.W, self.zs, self.lam = H, W, zs, lam
        self.p = ValueTracker(10.0)
        self.mode = "lens"
        self.focus = 260.0
        self.t = 0.0
        self.speed = 0.9
        self.x = np.arange(H) - H / 2 + 0.5
        half = 0.31 * H
        self.half = half
        self.ap = np.exp(-(np.abs(self.x) / half) ** 12)
        fx = np.fft.fftfreq(H)
        kz2 = (1 / lam) ** 2 - fx ** 2
        self.prop = kz2 > 0
        self.kz = 2 * np.pi * np.sqrt(np.maximum(kz2, 0))
        self.z = np.arange(W - zs)
        self.zin = np.arange(zs) - zs
        self.key = None
        self.U = None
        self.mob = ImageMobject(np.zeros((H, W, 4), np.uint8)).set(height=height).move_to(center)
        self.mob.set_resampling_algorithm(RESAMPLING_ALGORITHMS["linear"])
        self.s = height / H
        self.mob.add_updater(self._update)

    # geometry helpers (screen coordinates)
    def slm_x(self):
        return self.mob.get_left()[0] + self.zs * self.s

    def cy(self):
        return self.mob.get_center()[1]

    def phase(self):
        p = self.p.get_value()
        pix = np.floor((self.x + 1000 * p) / p)
        if self.mode == "grating":
            return np.pi * (pix % 2)
        # lens: quantized per pixel (value at pixel centre)
        xc = (pix + 0.5) * p - 1000 * p
        return (-np.pi * xc ** 2 / (self.lam * self.focus)) % (2 * np.pi)

    def _field(self):
        key = (self.mode, round(self.p.get_value(), 4), self.focus)
        if key != self.key:
            u0 = self.ap * np.exp(1j * self.phase())
            A = np.fft.fft(u0)
            A = A * self.prop
            self.U = np.fft.ifft(A[:, None] * np.exp(1j * self.kz[:, None] * self.z[None, :]), axis=0)
            self.key = key
        return self.U

    def _update(self, mob, dt):
        self.t += dt * self.speed
        U = self._field()
        ph = np.exp(-2j * np.pi * self.t)
        right = np.real(U * ph)
        k = 2 * np.pi / self.lam
        left = self.ap[:, None] * np.cos(k * self.zin[None, :] - 2 * np.pi * self.t)
        a = np.concatenate([left, right], 1)
        amp = np.concatenate([np.repeat(self.ap[:, None], self.zs, 1), np.abs(U)], 1)
        v = np.clip(a, 0, None) * 0.85 + 0.25 * np.clip(amp, 0, 1.4) ** 2
        mob.pixel_array = gmap(v)


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
        s = re.sub(r"(\d) × (\d)", "\\1\u00a0×\u00a0\\2", s)   # never wrap inside “3 × 3”
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

    def title_card(self, num, title, hold=2.2):
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


def build_seq(names, h):
    return FrameSeq([load_rgba(gen(n)) for n in names], h)


# ================================================================== S00 Title
class S00_Title(Base):
    def construct(self):
        t1 = tx("Large Étendue 3D Holographic Display", 54, bold=True).move_to([0, 2.85, 0])
        t2 = tx("with Content-adaptive Dynamic Fourier Modulation", 38).move_to([0, 2.1, 0])
        seq = build_seq([f"ms_{k}.png" for k in range(1, 10)], 3.0)
        seq.mob.move_to([0, -0.1, 0])
        box = framed(seq.mob, GHOST, 1.2)
        a1 = tx("Brian Chao, Manu Gopakumar, Suyeon Choi, Jonghyun Kim, Liang Shi, Gordon Wetzstein",
                24).move_to([0, -2.35, 0])
        a2 = tx("Stanford University · NVIDIA · MIT", 22, DIM).move_to([0, -2.85, 0])
        a3 = tx("SIGGRAPH Asia 2024", 22, DIM).move_to([0, -3.3, 0])
        self.add(t1, t2, a1, a2, a3)
        self.play(FadeIn(seq.mob), Create(box), run_time=1.0)
        self.add(seq.k)
        self.wait(0.8)
        self.play(seq.k.animate.set_value(8), run_time=4.0, rate_func=linear)
        self.wait(3.5)


# ================================================================== S01 Diffraction
class S01_Diffraction(Base):
    def construct(self):
        self.title_card("Part 1", "A hologram steers light by diffraction")
        self.header("1 · Diffraction and pixel pitch")
        wf = WaveField(center=[-1.9, 0.25, 0], height=4.4)
        xs = wf.slm_x()
        cy = wf.cy()
        H_ap = 2 * wf.half * wf.s
        frame = framed(wf.mob, GHOST, 1)

        def slm_pixels():
            p = wf.p.get_value()
            ph = wf.phase()
            g = VGroup()
            # draw one rectangle per SLM pixel inside the aperture
            x = wf.x
            idx = np.floor((x + 1000 * p) / p).astype(int)
            inside = np.abs(x) < wf.half
            for k in np.unique(idx[inside]):
                rows = np.where((idx == k) & inside)[0]
                y_top = cy + (wf.H / 2 - rows[0]) * wf.s
                y_bot = cy + (wf.H / 2 - rows[-1] - 1) * wf.s
                v = ph[rows[0]] / (2 * np.pi)
                g.add(Rectangle(width=0.16, height=max(y_top - y_bot, 0.01), stroke_width=0.5,
                                stroke_color=SLMC, fill_color=interpolate_color(BLACK, WHITE, 0.15 + 0.8 * v),
                                fill_opacity=1).move_to([xs, (y_top + y_bot) / 2, 0]))
            return g

        pix = always_redraw(slm_pixels)
        lab_laser = tx("laser (plane wave)", 22, DIM).next_to(wf.mob, UP, buff=0.12).align_to(wf.mob, LEFT)
        lab_slm = tx("phase SLM", 22, WHITE).move_to([xs, wf.mob.get_top()[1] + 0.25, 0])
        self.add(wf.mob, frame, pix, lab_laser, lab_slm)
        self.caption("A holographic display shapes laser light with a phase-only spatial light modulator (SLM)")
        self.wait(2.0)
        focus_x = xs + wf.focus * wf.s
        dot = Dot([focus_x, cy, 0], radius=0.07, color=WHITE)
        flab = tx("a 3D point", 22).next_to(dot, DOWN, buff=0.9)
        self.add(dot, flab)
        self.caption("Each pixel delays the wave; together they can focus light to a point floating in space")
        self.wait(2.5)
        self.remove(dot, flab)
        # --- the finest pattern: alternating pixels = grating of period 2p
        wf.mode = "grating"
        self.caption("The finest pattern an SLM can show: alternate neighbouring pixels (period 2<i>p</i>)")
        eqx = 4.35
        eq = tx(f"sin {TH} = {LAM} / 2<i>p</i>", 46).move_to([eqx, 1.9, 0])
        eqn = tx("maximum diffraction angle", 22, DIM).next_to(eq, DOWN, buff=0.18)
        self.add(eq, eqn)

        def theta():
            return np.arcsin(min(wf.lam / (2 * wf.p.get_value()), 0.999))

        def guides():
            th = theta()
            L = 3.7
            o = np.array([xs, cy, 0])
            up = DashedLine(o, o + L * np.array([np.cos(th), np.sin(th), 0]), color=AMBER, stroke_width=2.5,
                            dash_length=0.08)
            dn = DashedLine(o, o + L * np.array([np.cos(th), -np.sin(th), 0]), color=AMBER, stroke_width=2.5,
                            dash_length=0.08)
            arc = Arc(radius=1.1, start_angle=0, angle=th, arc_center=o, color=AMBER, stroke_width=2.5)
            lab = tx(TH, 26, AMBER).move_to(o + 1.4 * np.array([np.cos(th / 2), np.sin(th / 2), 0]))
            return VGroup(up, dn, arc, lab)

        g = always_redraw(guides)
        readout = always_redraw(lambda: tx(
            f"<i>p</i> = {wf.p.get_value():.1f} px   {TH} = {np.degrees(theta()):.1f}°", 26).move_to([eqx, 0.55, 0]))
        simnote = tx(f"(simulation units, {LAM} = 6 px)", 20, DIM).move_to([eqx, 0.05, 0])
        self.wait(1.5)
        self.add(g, readout, simnote)
        self.caption("The two diffracted beams leave at ±θ, exactly where the grating equation predicts")
        self.play(wf.p.animate.set_value(6.5), run_time=3.5)
        self.caption("Smaller pixels → wider angles. Larger pixels → light barely bends")
        self.play(wf.p.animate.set_value(16.0), run_time=3.5)
        self.play(wf.p.animate.set_value(10.0), run_time=2.0)
        real = VGroup(
            tx(f"real SLM: <i>p</i> = 10.8 µm, {LAM} = 632.8 nm", 24),
            tx(f"→ {TH} ≈ ±1.7°", 30, AMBER),
        ).arrange(DOWN, buff=0.2).move_to([eqx, -1.35, 0])
        self.add(real)
        self.caption("Real SLM pixels are microns wide, so the light spreads over only a few degrees")
        self.caption("(holographic film has features of tens of nanometres; no SLM comes close)")
        self.uncaption()
        self.wait(0.5)


# ================================================================== S02 Etendue
NLAM = 1000 * 632.8e-6        # N·λ in mm (paper Fig. 3: 1000 px, 632.8 nm, p = 10.8 µm)


def fov_deg(w, alpha=1):
    return np.degrees(2 * np.arctan(alpha * NLAM / (2 * w)))


class S02_Etendue(Base):
    def construct(self):
        self.title_card("Part 2", "Étendue: field of view × eyebox")
        self.header("2 · Étendue")
        w = ValueTracker(2.0)       # eyebox (mm)
        x0 = -6.4
        cy = -0.55
        Ls = 1.8

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
            lb2 = tx("FoV", 22, GRN).move_to(o + np.array([1.05, 0, 0]))
            return VGroup(lit, slm, ln, eb, f1, f2, arc, lb1, lb2)

        dg = always_redraw(diagram)
        top = tx("SLM", 22).move_to([x0, cy + Ls / 2 + 0.3, 0])
        eyelab = always_redraw(lambda: tx("eyepiece <i>g</i>", 22, CYAN).move_to([x0 + Gs(), cy + Ls / 2 + 0.55, 0]))
        self.add(dg, top, eyelab)
        # equations
        e1 = tx(f"<i>G</i> = 4<i>A</i> sin{sup('2')}{TH}", 36).move_to([3.7, 2.65, 0])
        self.add(e1)
        self.caption("Étendue: emitting area × solid angle of the light. Lenses, mirrors, free space all conserve it")
        e2 = tx(f"<i>G</i>{sub('SLM')} = 4<i>L</i>{sub('<i>x</i>')}<i>L</i>{sub('<i>y</i>')} sin{sup('2')}{TH} = "
                f"{LAM}{sup('2')}<i>N</i>{sub('<i>x</i>')}<i>N</i>{sub('<i>y</i>')}", 32).move_to([3.7, 1.95, 0])
        self.add(e2)
        self.caption("With sin θ = λ/2p, an SLM's étendue is fixed by its pixel count")
        e3 = tx(f"FoV = 2 tan{sup('−1')}(<i>Np</i> / 2<i>g</i>)     <i>w</i> = <i>g</i>{LAM} / <i>p</i>", 28,
                ).move_to([3.7, 1.3, 0])
        self.add(e3)
        # plot
        ax = Axes(x_range=[0, 10, 2], y_range=[0, 40, 10], x_length=5.0, y_length=2.75,
                  axis_config={"color": DIM, "stroke_width": 2, "include_ticks": True, "tick_size": 0.05},
                  tips=False).move_to([3.85, -1.15, 0])
        xl = tx("eyebox (mm)", 20, DIM).next_to(ax.x_axis, DOWN, buff=0.1).align_to(ax, RIGHT)
        yl = tx("FoV (°)", 20, DIM).next_to(ax.y_axis, UP, buff=0.08).align_to(ax.y_axis, LEFT)
        nums = VGroup(*[tx(str(v), 16, DIM).next_to(ax.c2p(v, 0), DOWN, buff=0.08) for v in (2, 4, 6, 8, 10)],
                      *[tx(str(v), 16, DIM).next_to(ax.c2p(0, v), LEFT, buff=0.08) for v in (10, 20, 30, 40)])
        curve = ax.plot(lambda x: fov_deg(x), x_range=[0.85, 10, 0.02], color=WHITE, stroke_width=2.5)
        clab = tx("α = 1", 20).move_to(ax.c2p(8.6, 7.5))

        def rect():
            ww = w.get_value()
            f = fov_deg(ww)
            r = Polygon(ax.c2p(0, 0), ax.c2p(ww, 0), ax.c2p(ww, f), ax.c2p(0, f), stroke_color=AMBER,
                        stroke_width=2, fill_color=AMBER, fill_opacity=0.25)
            d = Dot(ax.c2p(ww, f), radius=0.06, color=AMBER)
            return VGroup(r, d)

        rr = always_redraw(rect)
        self.add(ax, xl, yl, nums, curve, clab, rr)
        self.caption("Swap the eyepiece: FoV and eyebox trade off, their product stays put")
        self.play(w.animate.set_value(7.5), run_time=3.5)
        self.play(w.animate.set_value(1.1), run_time=3.5)
        self.play(w.animate.set_value(3.0), run_time=2.0)
        note = tx("curve: 1000 × 1000 SLM, 10.8 µm pixels, 632.8 nm (paper Fig. 3)", 18, DIM).next_to(xl, DOWN, buff=0.1).align_to(ax, RIGHT)
        self.add(note)
        self.caption("A 1080p SLM: an 80° FoV comes with an eyebox under 1 mm, or vice versa")
        self.caption("A pupil moves a few millimetres. We need far more étendue than the pixels give")
        self.uncaption()
        self.wait(0.4)


# ================================================================== S03 Prior work
class S03_Prior(Base):
    def construct(self):
        self.header("3 · Stretching étendue: prior approaches")
        fig = img("baseline.png", w=13.4).move_to([0, 0.55, 0])
        self.add(fig)
        src = tx("paper Fig. 4 (baseline configurations)", 18, DIM).next_to(fig, DOWN, buff=0.08).align_to(fig, RIGHT)
        self.add(src)
        self.caption("Everything else follows the same recipe: add degrees of freedom before or after the SLM")
        W = fig.width
        left = fig.get_left()[0]
        top = fig.get_top()[1]
        bot = fig.get_bottom()[1]

        def hl(a, b, color):
            return Rectangle(width=(b - a) * W, height=fig.height + 0.06, stroke_color=color,
                             stroke_width=3).move_to([left + (a + b) / 2 * W, (top + bot) / 2, 0])

        vy = bot - 0.5
        b1 = hl(0.185, 0.36, RED)
        v1 = tx("static mask → low contrast, speckle", 22, RED).move_to([left + 0.27 * W, vy, 0])
        self.add(b1, v1)
        self.caption("Static scattering or phase masks widen the angles, but lack the freedom for clean images")
        b2 = hl(0.36, 0.68, AMBER)
        v2 = tx("all sources at once → copies, few degrees of freedom", 22, AMBER).move_to([left + 0.52 * W, vy - 0.42, 0])
        self.add(b2, v2)
        self.caption("Many sources at once: every copy shows the same hologram. A fixed random mask only partly helps")
        b3 = hl(0.68, 1.0, PURPLE)
        v3 = tx("steered → needs very fast SLMs", 22, PURPLE).move_to([left + 0.84 * W, vy, 0])
        self.add(b3, v3)
        self.caption("Steering one source at a time needs very fast SLMs, and high diffraction orders remain")
        self.caption("(pupil replication copies one wavefront, so it cannot show 3D parallax across the eyebox)")
        self.uncaption()
        self.wait(0.3)


# ================================================================== S04 Fourier lens
class S04_FourierLens(Base):
    def construct(self):
        self.title_card("Part 4", "A lens performs a Fourier transform")
        self.header("4 · The Fourier plane is the eyebox")
        xs, xl, xf, cy = -6.2, -3.9, -1.6, 0.2
        fL = xl - xs
        slm = slm_bar(xs, cy, 3.2, n=16, seed=2)
        ln = lens(xl, cy, 3.6)
        fp = DashedLine([xf, cy - 2.0, 0], [xf, cy + 2.0, 0], color=DIM, stroke_width=2)
        l1 = tx("SLM", 22).move_to([xs, cy + 2.0, 0])
        l2 = tx("lens <i>f</i>", 22, CYAN).move_to([xl, cy + 2.15, 0])
        l3 = tx("Fourier plane", 22, DIM).move_to([xf, cy + 2.3, 0])
        self.add(slm, ln, fp, l1, l2, l3)
        self.caption("Put a lens one focal length after the SLM and look at its back focal plane")
        bundles = VGroup()
        for ang, c in ((0.0, GRN), (0.20, AMBER), (-0.32, "#4FC3F7")):
            yF = cy + fL * np.tan(ang) * 1.0
            rays = VGroup()
            for y0 in np.linspace(cy - 1.2, cy + 1.2, 5):
                yl = y0 + fL * np.tan(ang)
                rays.add(VGroup(Line([xs + 0.1, y0, 0], [xl, yl, 0], color=c, stroke_width=2),
                                Line([xl, yl, 0], [xf, yF, 0], color=c, stroke_width=2)))
            bundles.add(rays)
        for b in bundles:
            self.play(LaggedStart(*[Create(r) for r in b], lag_ratio=0.1), run_time=1.1)
        eq = tx(f"direction {TH}  →  point <i>u</i> = <i>f</i> tan {TH} ≈ {LAM}<i>f</i> · <i>f</i>{sub('<i>x</i>')}", 26
                ).move_to([-3.9, -2.45, 0])
        self.add(eq)
        self.caption("Each propagation direction lands on one point: the lens computes the Fourier transform")
        # computed spectrum of the pixelated SLM
        sp = gimg("spec_single.png", h=3.9).move_to([3.7, 0.25, 0])
        fr = framed(sp, GHOST, 1)
        grid = VGroup()
        s0 = sp.get_corner(UL)
        cw = sp.height / 4
        for i in range(1, 4):
            grid.add(DashedLine(s0 + RIGHT * i * cw, s0 + RIGHT * i * cw + DOWN * sp.height, color=GHOST,
                                stroke_width=1.2, dash_length=0.06))
            grid.add(DashedLine(s0 + DOWN * i * cw, s0 + DOWN * i * cw + RIGHT * sp.height, color=GHOST,
                                stroke_width=1.2, dash_length=0.06))
        spl = tx(f"computed |ℱ{{pixelated SLM field}}|{sup('2')}  (log)", 20, DIM).next_to(sp, UP, buff=0.12)
        self.play(FadeIn(sp), run_time=0.8)
        self.add(fr, grid, spl)
        cen = Square(side_length=cw, color=WHITE, stroke_width=2.5).move_to(sp.get_center())
        hdo = tx("higher diffraction orders (pixel grid)", 20, AMBER).next_to(sp, DOWN, buff=0.15)
        self.add(cen, hdo)
        self.caption("Pixels repeat the spectrum: a central order plus higher diffraction orders (HDOs)")
        cwl = tx(f"cell = {LAM}<i>f</i> / <i>p</i>", 22).next_to(cen, RIGHT, buff=0.08).shift(UP * 0.3)
        self.add(cwl)
        self.caption("One order is λf/p wide. That width is the eyebox, the same w = gλ/p as before")
        filt = gimg("spec_single_filt.png", h=3.9).move_to(sp)
        self.play(FadeIn(filt), run_time=0.8)
        self.caption("Conventional displays block every HDO with a Fourier filter, keeping one small eyebox")
        self.clear_all(keep=[])
        # captured single-source spectrum + pupil
        spec = img("teaser_spec_single.png", w=4.8).move_to([-3.2, 0.35, 0])
        view_c = img("teaser_single_center.png", w=4.6).move_to([3.3, 0.35, 0])
        view_r = img("teaser_single_right.png", w=4.6).move_to([3.3, 0.35, 0])
        cap1 = tx("captured Fourier spectrum (single source)", 22, DIM).next_to(spec, UP, buff=0.15)
        cap2 = tx("view through the pupil", 22, DIM).next_to(view_c, UP, buff=0.15)
        W = spec.width
        c0 = spec.get_center() + RIGHT * (0.506 - 0.5) * W
        c1 = spec.get_center() + RIGHT * (0.748 - 0.5) * W
        pup = Circle(radius=0.188 * W, color=RED, stroke_width=4).move_to(c0)
        self.add(spec, view_c, cap1, cap2, pup)
        src = tx("experimental captures, paper Fig. 1 (b–d)", 18, DIM).next_to(view_c, DOWN, buff=0.12).align_to(view_c, RIGHT)
        self.add(src)
        self.caption("Our pupil samples the Fourier plane. Centred on the spectrum, the image looks fine")
        self.play(pup.animate.move_to(c1).set_color(BLUE), FadeIn(view_r), run_time=2.5)
        self.caption("Shift the pupil a few millimetres and the image fades to black: the eyebox is tiny")
        self.uncaption()
        self.wait(0.3)


# ================================================================== S05 Multi-source
class S05_MultiSource(Base):
    def construct(self):
        self.title_card("Part 5", "Many sources, shifted spectra")
        self.header("5 · Multi-source illumination")
        xsrc, xcol, xslm, cy = -6.4, -5.0, -3.3, 0.9
        fcol = xcol - xsrc
        ys = [0.75, 0.0, -0.75]
        srcs = VGroup(*[Dot([xsrc, cy + y, 0], radius=0.08, color=SRC3[i]) for i, y in enumerate(ys)])
        cl = lens(xcol, cy, 2.6)
        slm = slm_bar(xslm, cy, 2.4, n=12, seed=5)
        beams = VGroup()
        for i, y in enumerate(ys):
            ang = -np.arctan(y / fcol)
            h = 1.1
            a1 = [xcol, cy + h, 0]
            a2 = [xcol, cy - h, 0]
            d = (xslm - xcol) * np.tan(ang)
            beams.add(beam(a1, a2, [xslm, cy + h + d, 0], [xslm, cy - h + d, 0], SRC3[i], 0.18))
        cones = VGroup(*[Polygon([xsrc, cy + y, 0], [xcol, cy + 1.1, 0], [xcol, cy - 1.1, 0], stroke_width=0,
                                 fill_color=SRC3[i], fill_opacity=0.12) for i, y in enumerate(ys)])
        l1 = tx("sources", 20).move_to([xsrc, cy + 1.6, 0])
        l2 = tx("collimator", 20, CYAN).move_to([xcol, cy + 1.6, 0])
        l3 = tx("phase SLM", 20).move_to([xslm, cy + 1.6, 0])
        self.add(cl, slm, l1, l2, l3)
        self.play(FadeIn(srcs[1]), FadeIn(cones[1]), FadeIn(beams[1]), run_time=0.6)
        self.play(FadeIn(srcs[0]), FadeIn(srcs[2]), FadeIn(cones[0]), FadeIn(cones[2]),
                  FadeIn(beams[0]), FadeIn(beams[2]), run_time=0.8)
        self.caption("Now light the SLM with several lasers at once, each arriving from a different angle")
        eq = tx(f"𝒰{sub('slm')}{sup('(<i>j</i>)')} = ℱ{{ <i>e</i>{sup(f'<i>i</i>{PHI}')} <i>u</i>{sub('src')}{sup('(<i>j</i>)')} "
                f"<i>e</i>{sup('<i>i</i> <b>k</b>' + sup('(<i>j</i>)') + '·<b>x</b>')} }}", 30).move_to([-3.7, -1.0, 0])
        self.add(eq)
        eqn = tx("a tilt in space = a shift in frequency", 22, DIM).next_to(eq, DOWN, buff=0.2)
        self.add(eqn)
        # spectrum build-up
        seq = build_seq([f"ms_{k}.png" for k in range(1, 10)], 4.3)
        seq.mob.move_to([3.6, 0.2, 0])
        fr = framed(seq.mob, GHOST, 1)
        cw = seq.mob.height / 4
        ebox = Square(side_length=3 * cw, color=AMBER, stroke_width=2.5).move_to(seq.mob)
        c1 = Square(side_length=cw, color=WHITE, stroke_width=2).move_to(seq.mob)
        spl = tx("computed Fourier plane (toy simulation)", 20, DIM).next_to(seq.mob, UP, buff=0.12)
        self.add(seq.mob, seq.k, fr, c1, spl)
        # 3x3 array icon
        icon = VGroup()
        for (dy, dx) in SRC_ORDER:
            icon.add(Dot([-5.9 + 0.3 * dx, -2.35 - 0.3 * dy, 0], radius=0.06, color=GHOST))
        icl = tx("3 × 3 source array", 20, DIM).next_to(icon, RIGHT, buff=0.25)
        self.add(icon, icl)
        icon[0].set_color(GRN)
        self.caption("Mutually incoherent sources: each adds its own shifted copy of the spectrum")
        for k in range(1, 9):
            icon[k].set_color(GRN)
            self.play(seq.k.animate.set_value(k), run_time=0.45, rate_func=linear)
        self.add(ebox)
        ebl = tx("expanded eyebox", 20, AMBER).next_to(ebox, DOWN, buff=0.12)
        self.add(ebl)
        self.caption("Tilts matched to the ±1st diffraction orders tile the copies side by side")
        self.caption("The eyebox grows 3× per axis while the FoV stays: étendue grows α² = 9×")
        self.clear_all()
        # the catch: copies are correlated
        a = gimg("view_target_c.png", h=3.3).move_to([-3.4, 0.2, 0])
        b = gimg("view_III1_c.png", h=3.3).move_to([3.4, 0.2, 0])
        la = tx("target view", 22, DIM).next_to(a, UP, buff=0.15)
        lb = tx("multi-source, no Fourier filter", 22, DIM).next_to(b, UP, buff=0.15)
        nt = tx("toy simulation (our re-implementation, 64 × 64 SLM)", 18, DIM).next_to(b, DOWN, buff=0.12)
        self.add(a, b, la, lb, nt, framed(a, GHOST), framed(b, GHOST))
        self.caption("The catch: every copy carries the same hologram, overlapping other sources' HDOs")
        self.caption("One phase pattern cannot serve all copies at once: ghosts and low contrast")
        self.uncaption()
        self.wait(0.3)


# ================================================================== S06 Fourier modulation
class S06_FourierMod(Base):
    def construct(self):
        self.title_card("Part 6", "Content-adaptive dynamic Fourier modulation")
        self.header("6 · A second SLM in the Fourier plane")
        cy = 2.0
        xs = [-6.3, -5.2, -3.9, -2.4, -0.9, 0.3]
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
        labs = VGroup(
            tx("sources", 18).move_to([xs[0], cy + 0.8, 0]),
            tx("phase SLM", 18).move_to([xs[2], cy + 0.8, 0]),
            tx("Fourier lens", 18, CYAN).move_to([xs[3], cy + 0.85, 0]),
            tx("amplitude SLM", 18, AMBER).move_to([xs[4], cy + 0.8, 0]),
        )
        self.add(glow, srcs, l1, s1, l2, s2, ey, labs)
        right = VGroup(
            tx("Fourier plane = eyebox plane", 22, AMBER),
            tx("20 × 20 Fourier display in simulation", 20, DIM),
            tx("1080p LCoS in the prototype", 20, DIM),
        ).arrange(DOWN, buff=0.14, aligned_edge=LEFT).move_to([4.2, 2.05, 0])
        self.add(right)
        self.caption("The fix: an amplitude SLM right in the Fourier plane, reshaping the spectrum")
        # panels
        py = -0.65
        ps = 2.35
        xp = [-5.0, -1.75, 1.5, 4.9]
        sp = gimg("v6_nomask.png", h=ps).move_to([xp[0], py, 0])
        Pseq = build_seq([f"v6_P_{t}.png" for t in range(6)], ps)
        Pseq.mob.move_to([xp[1], py, 0])
        Pseq.mob.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
        Mseq = build_seq([f"v6_masked_{t}.png" for t in range(6)], ps)
        Mseq.mob.move_to([xp[2], py, 0])
        view = gimg("view_V6_c.png", h=ps).move_to([xp[3], py, 0])
        ops = VGroup(tx("×", 40).move_to([(xp[0] + xp[1]) / 2, py, 0]),
                     tx("=", 40).move_to([(xp[1] + xp[2]) / 2, py, 0]),
                     tx("→", 40).move_to([(xp[2] + xp[3]) / 2 - 0.08, py, 0]))
        ly = py + ps / 2 + 0.25
        pl = VGroup(tx("spectrum (all 9 sources)", 20, DIM).move_to([xp[0], ly, 0]),
                    always_redraw(lambda: tx(f"mask {PF}{sup('(<i>t</i>)')},  <i>t</i> = {int(round(Pseq.k.get_value())) + 1}", 20, AMBER).move_to([xp[1], ly, 0])),
                    tx("modulated spectrum", 20, DIM).move_to([xp[2], ly, 0]),
                    tx(f"eye sees Σ{sub('<i>t</i>')} over 6 frames", 20, DIM).move_to([xp[3], ly, 0]))
        frs = VGroup(*[framed(m, GHOST, 1) for m in (sp, Pseq.mob, Mseq.mob, view)])
        nt = tx("toy simulation", 18, DIM).next_to(view, DOWN, buff=0.1).align_to(view, RIGHT)
        self.add(sp, Pseq.mob, Mseq.mob, Pseq.k, Mseq.k, ops, pl, frs, nt)
        self.caption("Each frame, the mask passes only the parts of the spectrum this content needs")
        for rep in range(2):
            for t in range(6):
                Pseq.k.set_value(t)
                Mseq.k.set_value(t)
                self.wait(0.55)
        self.add(view)
        eq = tx(f"<i>f</i>{sup('(<i>j</i>)')}({PHI}, {PF}, <i>z</i>) = ℱ{sup('−1')}{{ 𝒰{sub('slm')}{sup('(<i>j</i>)')} · ℋ(<i>z</i>) · {PF} }}",
                30).move_to([-1.2, -2.55, 0])
        self.add(eq)
        self.caption("Phase and amplitude patterns switch in sync; the eye averages T frames")
        for t in range(6):
            Pseq.k.set_value(t)
            Mseq.k.set_value(t)
            self.wait(0.45)
        # contrast: fixed random mask
        self.remove(Pseq.mob, Mseq.mob, view, pl[1])
        rp = gimg("iv_P.png", h=ps, nearest=True).move_to([xp[1], py, 0])
        rm = gimg("iv_masked.png", h=ps).move_to([xp[2], py, 0])
        rv = gimg("view_IV1_c.png", h=ps).move_to([xp[3], py, 0])
        rl = tx("fixed random mask (Jo et al. 2022)", 20, RED).move_to([xp[1], ly, 0])
        self.add(rp, rm, rv, rl)
        self.caption("Compare a fixed random mask: static, not content adaptive, one frame")
        self.wait(1.0)
        self.remove(rp, rm, rv, rl)
        self.add(Pseq.mob, Mseq.mob, view, pl[1])
        self.caption("Ours: optimized for the content, and different in every frame")
        for t in range(6):
            Pseq.k.set_value(t)
            Mseq.k.set_value(t)
            self.wait(0.45)
        self.uncaption()


# ================================================================== S07 Optimization
class S07_Optimization(Base):
    def construct(self):
        self.title_card("Part 7", "Pupil-aware light field optimization")
        self.header("7 · Computer-generated holography")
        eq = tx(f"minimize over {{{PHI}{sup('(<i>t</i>)')}, {PF}{sup('(<i>t</i>)')}}}   ‖ <i>s</i> √( (1/<i>T</i>) Σ{sub('<i>t</i>')} Σ{sub('<i>j</i>')} "
                f"| H2LF( <i>f</i>{sup('(<i>j</i>)')}({PHI}{sup('(<i>t</i>)')}, {PF}{sup('(<i>t</i>)')}) ) |{sup('2')} ) − lf{sub('target')} ‖",
                28).move_to([0, 2.75, 0])
        if eq.width > 13.4:
            eq.scale_to_fit_width(13.4)
        self.add(eq)
        self.caption("Supervise with a large-baseline target light field: what each pupil position should see")
        # eyebox with 9x9 pupils
        E = 3.7
        ec = np.array([-4.2, -0.2, 0])
        bg = gimg("v6_nomask.png", h=E).move_to(ec).set_opacity(0.35)
        ebox = Square(side_length=E, color=AMBER, stroke_width=2.5).move_to(ec)
        span = SPAN / EB * E
        offs = np.linspace(-span, span, 9)
        dots = VGroup(*[Dot(ec + np.array([u, -v, 0]), radius=0.035, color=WHITE) for v in offs for u in offs])
        rp = R_P / EB * E
        pup = Circle(radius=rp, color=BLUE, stroke_width=4).move_to(dots[40])
        el = tx("eyebox: 9 × 9 pupils, 4 mm each", 20, AMBER).next_to(ebox, DOWN, buff=0.15)
        self.add(bg, ebox, dots, el, pup)
        tgt = FrameSeq([gmap(TOY["target_grid"][k].astype(np.float32)) for k in range(81)], 2.6)
        rec = FrameSeq([gmap(TOY["V6_grid"][k].astype(np.float32)) for k in range(81)], 2.6)
        tgt.mob.move_to([0.5, -0.15, 0])
        rec.mob.move_to([3.95, -0.15, 0])
        for s in (tgt, rec):
            s.mob.set_resampling_algorithm(RESAMPLING_ALGORITHMS["linear"])
        tl = tx("target view <i>p</i>", 22, DIM).next_to(tgt.mob, UP, buff=0.15)
        rl = tx("hologram seen through pupil <i>p</i>", 22, DIM).next_to(rec.mob, UP, buff=0.15)
        nt = tx("toy simulation, ours with 6 frames", 18, DIM).next_to(rec.mob, DOWN, buff=0.1).align_to(rec.mob, RIGHT)
        vs = tx("vs.", 24, DIM).move_to([2.22, -0.15, 0])
        tgt.k.set_value(40)
        rec.k.set_value(40)
        self.add(tgt.mob, rec.mob, tgt.k, rec.k, tl, rl, nt, vs, framed(tgt.mob, GHOST), framed(rec.mob, GHOST))
        self.caption("Each iteration picks one random view: a binary pupil mask in the Fourier plane")
        rng = np.random.default_rng(7)
        for _ in range(7):
            k = int(rng.integers(0, 81))
            self.play(pup.animate.move_to(dots[k]), run_time=0.35)
            tgt.k.set_value(k)
            rec.k.set_value(k)
            self.wait(0.35)
        self.caption("Compare, then take a gradient step on all phase and amplitude patterns")
        self.caption("One view per step keeps memory manageable despite 9 sources, 6 frames and HDOs")
        # sweep middle row
        seqt = [gmap(TOY["target_path"][i].astype(np.float32)) for i in range(25)]
        seqr = [gmap(TOY["V6_path"][i].astype(np.float32)) for i in range(25)]
        self.remove(tgt.mob, rec.mob)
        tgt2 = FrameSeq(seqt, 2.6)
        rec2 = FrameSeq(seqr, 2.6)
        tgt2.mob.move_to(tgt.mob)
        rec2.mob.move_to(rec.mob)
        self.add(tgt2.mob, rec2.mob, tgt2.k, rec2.k)
        row_l = ec + np.array([-span, 0, 0])
        row_r = ec + np.array([span, 0, 0])
        self.play(pup.animate.move_to(row_l), run_time=0.6)
        self.caption("Slide the pupil across the expanded eyebox: the near shapes shift against the bars (parallax)")
        self.play(pup.animate.move_to(row_r), tgt2.k.animate.set_value(24), rec2.k.animate.set_value(24),
                  run_time=3.0, rate_func=linear)
        self.play(pup.animate.move_to(row_l), tgt2.k.animate.set_value(0), rec2.k.animate.set_value(0),
                  run_time=3.0, rate_func=linear)
        self.uncaption()
        self.wait(0.3)


# ================================================================== S08 Results
class S08_Results(Base):
    def construct(self):
        self.title_card("Part 8", "Results")
        self.header("8 · Prototype")
        photo = img("hw_setup.png", w=8.6).move_to([-2.35, 0.35, 0])
        self.add(photo, framed(photo, GHOST))
        specs = VGroup(
            tx("3 × 3 fiber-tip sources, 8.17 mm apart", 21),
            tx("200 mm collimator → 2.34° per source", 21),
            tx("TI DLP phase SLM (4-bit)", 21),
            tx("1080p LCoS amplitude SLM", 21),
            tx("75 mm Fourier transform lens", 21),
            tx("diagonal FoV 7.78°", 24, GRN),
            tx("eyebox 8.53 mm × 8.53 mm", 24, AMBER),
        ).arrange(DOWN, buff=0.2, aligned_edge=LEFT).move_to([4.5, 0.35, 0])
        if specs.width > 4.3:
            specs.scale_to_fit_width(4.3)
            specs.move_to([4.55, 0.35, 0])
        self.add(specs)
        self.caption("The benchtop prototype: nine fiber tips, a phase SLM, and an amplitude SLM at the Fourier plane")
        self.caption("A pupil iris on a translation stage plays the role of the moving eye")
        self.clear_all()
        # captured: conventional vs ours with moving pupil
        self.header("8 · Captured: single source vs. ours")
        rows = [("Conventional", "teaser_spec_single.png", "teaser_single_center.png", "teaser_single_right.png", 1.35),
                ("Ours", "teaser_spec_multi.png", "teaser_multi_center.png", "teaser_multi_right.png", -1.45)]
        hs = 2.45
        pups = []
        for (name, sp, vc, vr, y) in rows:
            s = img(sp, h=hs).move_to([-3.35, y, 0])
            a = img(vc, h=hs).move_to([0.55, y, 0])
            b = img(vr, h=hs).move_to([4.65, y, 0])
            lab = tx(name, 24).rotate(PI / 2).move_to([-6.2, y, 0])
            W = s.width
            pc = Circle(radius=0.188 * W, color=RED, stroke_width=3).move_to(s.get_center() + RIGHT * 0.006 * W)
            pb = Circle(radius=0.188 * W, color=BLUE, stroke_width=3).move_to(s.get_center() + RIGHT * 0.006 * W)
            self.add(s, a, b, lab, pc, pb, framed(a, RED, 2), framed(b, BLUE, 2))
            pups.append((pb, s.get_center() + RIGHT * 0.248 * W))
        top = 1.35 + hs / 2 + 0.2
        self.add(tx("Fourier spectrum", 22, DIM).move_to([-3.35, top, 0]),
                 tx("center view (red pupil)", 22, DIM).move_to([0.55, top, 0]),
                 tx("right view (blue pupil)", 22, DIM).move_to([4.65, top, 0]))
        self.caption("Move the pupil right: the single source fades to black, ours stays bright")
        self.play(*[p.animate.move_to(t) for (p, t) in pups], run_time=2.5)
        self.caption("Nine sources fill the Fourier plane: experimental captures, paper Fig. 1")
        self.clear_all()
        # simulation table
        self.header("8 · Simulation: 9 × 9 light field, PSNR (paper Table 1)")
        data = [("I. single source + filter", 16.60, 0.23, DIM), ("II. phase mask", 14.61, 0.15, DIM),
                ("III. multiple sources", 13.35, 0.24, DIM), ("IV. + fixed random mask", 18.63, 0.26, DIM),
                ("VI. steered (9 frames)", 20.02, 0.43, DIM), ("VII. steered + filter (9 frames)", 23.95, 0.51, DIM),
                ("ours (1 frame)", 21.05, 0.39, GRN), ("ours (6 frames)", 24.65, 0.66, GRN)]
        x0 = -1.4
        sc = 0.22
        bars = VGroup()
        for i, (nm, ps, ss, c) in enumerate(data):
            y = 2.35 - i * 0.62
            lab = tx(nm, 22, WHITE if c == GRN else "#CFCFCF").next_to([x0 - 0.15, y, 0], LEFT, buff=0)
            r = Rectangle(width=ps * sc, height=0.4, stroke_width=0, fill_color=c, fill_opacity=0.85)
            r.move_to([x0 + ps * sc / 2, y, 0])
            val = tx(f"{ps:.2f} dB / {ss:.2f}", 20).next_to(r, RIGHT, buff=0.15)
            bars.add(VGroup(lab, r, val))
        for b in bars:
            self.add(b[0])
        self.play(LaggedStart(*[GrowFromEdge(b[1], LEFT) for b in bars], lag_ratio=0.12), run_time=2.0)
        for b in bars:
            self.add(b[2])
        leg = tx("PSNR / SSIM, averaged over 6 scenes · 3 × 3 sources · 20 × 20 Fourier display", 18, DIM
                 ).move_to([0.6, -2.65, 0])
        self.add(leg)
        self.caption("Best light field quality, beating steered illumination with fewer frames (6 vs. 9)")
        self.caption("(a single source still reaches 46.23 dB on a plain 2D image, but not over a large eyebox)")
        self.clear_all()
        # Fig. 6 crops
        self.header("8 · Simulated views (paper Fig. 6)")
        cols = [("I", "I. single", "16.13 dB"), ("III", "III. multi", "11.28 dB"), ("IV", "IV. random mask", "18.27 dB"),
                ("VII", "VII. steered + filter", "25.07 dB"), ("V6", "ours, 6 frames", "26.01 dB")]
        xw = 2.55
        for i, (k, nm, ps) in enumerate(cols):
            x = -5.2 + i * 2.6
            a = gimg(f"sim_{k}_center.png", w=xw).move_to([x, 1.05, 0])
            b = gimg(f"sim_{k}_right.png", w=xw).move_to([x, -1.2, 0])
            c = GRN if k == "V6" else WHITE
            self.add(a, b, tx(nm, 20, c).next_to(a, UP, buff=0.12), tx(ps, 20, c).next_to(b, DOWN, buff=0.12))
        self.add(tx("center", 18, DIM).rotate(PI / 2).move_to([-6.75, 1.05, 0]),
                 tx("right view", 18, DIM).rotate(PI / 2).move_to([-6.75, -1.2, 0]))
        self.caption("Ours removes the copies and HDO ghosts, with clean parallax in the side views")
        self.clear_all()
        # experiment
        self.header("8 · Experimental captures (paper Fig. 7, Table 2)")
        cols = [("I", "I. single source", ""), ("III", "III. multi", "12.72 / 0.19"), ("IV", "IV. random mask", "12.46 / 0.17"),
                ("V1", "ours, 1 frame", "13.83 / 0.23"), ("V6", "ours, 6 frames", "14.43 / 0.40")]
        for i, (k, nm, ps) in enumerate(cols):
            x = -5.2 + i * 2.6
            a = gimg(f"exp_{k}_front.png", w=xw).move_to([x, 1.05, 0])
            b = gimg(f"exp_{k}_right.png", w=xw).move_to([x, -1.05, 0])
            c = GRN if k.startswith("V") else WHITE
            self.add(a, b, tx(nm, 20, c).next_to(a, UP, buff=0.12))
            if ps:
                self.add(tx(ps, 20, c).next_to(b, DOWN, buff=0.12))
        self.add(tx("front focus", 18, DIM).rotate(PI / 2).move_to([-6.75, 1.05, 0]),
                 tx("right view", 18, DIM).rotate(PI / 2).move_to([-6.75, -1.05, 0]))
        self.add(tx("PSNR / SSIM over captured light field views", 18, DIM).move_to([2.6, -2.55, 0]))
        self.caption("On the real display: the single source goes dark off-center; ours with 6 frames is best")
        self.uncaption()
        self.wait(0.3)


# ================================================================== S09 Outlook
class S09_Outlook(Base):
    def construct(self):
        self.header("9 · Limitations and outlook")
        items = VGroup(
            tx("• Benchtop: fiber splitters → nanophotonic phased arrays, waveguides", 26),
            tx("• 240 Hz amplitude display → ~13.33 Hz in color-sequential, 6-frame mode", 26),
            tx("• Faster LCoS (> 720 Hz) would raise the frame rate", 26),
            tx("• No real-time synthesis yet: neural CGH for 4D light fields", 26),
        ).arrange(DOWN, buff=0.35, aligned_edge=LEFT).move_to([0, 0.9, 0])
        self.add(items)
        self.caption("A benchtop proof of concept: the paper lists what it takes to make it wearable")
        self.caption("Summary: more sources for étendue, a dynamic Fourier mask for image quality")
        self.clear_all(keep=[])
        self.remove(self.hdr)
        t1 = tx("Large Étendue 3D Holographic Display", 46, bold=True).move_to([0, 2.2, 0])
        t2 = tx("with Content-adaptive Dynamic Fourier Modulation", 32).move_to([0, 1.55, 0])
        s = gimg("ms_9.png", h=2.1).move_to([0, -0.3, 0])
        a1 = tx("Brian Chao, Manu Gopakumar, Suyeon Choi, Jonghyun Kim, Liang Shi, Gordon Wetzstein", 22).move_to([0, -2.0, 0])
        a2 = tx("SIGGRAPH Asia 2024 · arXiv 2409.03143 · bchao1.github.io/holo_dfm", 24, GRN).move_to([0, -2.6, 0])
        self.add(t1, t2, s, framed(s, GHOST), a1, a2)
        self.wait(5.0)
