"""Random-phase Wave Splatting — three 1-minute explainers (3Blue1Brown style), v1.

Source: Chao*, Yang*, Choi, Gopakumar, Koiso, Wetzstein, "Random-phase Wave Splatting of Translucent Primitives for
Computer-generated Holography", arXiv 2508.17480, and its supplementary material
(../data/Random_phase_Wave_Splatting (2).pdf). Look and helpers follow the multisource explainer
(../../large_etendue_holography/animation_large_etendue_v3.py).

All wave-optics visuals are toy simulations from sims.py (cached in data/generated/); this file only turns them into
display assets (build_assets, flag data/generated/done_v1) and animates them.

Render (see render.sh):
    manim -ql --media_dir media animation_rpws_explainer_v1.py <Scene>
Scenes: S00_Title (3-min cut only), P1_Difference, P2_AlphaBlending, P3_BackToFront.
"""
from manim import *
import numpy as np
from PIL import Image
import matplotlib
import fcntl
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
GEN = os.path.join(DATA, "generated")
FIGS = os.path.join(HERE, "..", "random-phase-wave-splatting", "pdfs", "figs")
ASSET_FLAG = "done_v1"

config.background_color = BLACK

# ------------------------------------------------------------------ style (= multisource explainer)
FONT = "CMU Serif"
MATHFONT = "Latin Modern Math"
SYMFONT = "DejaVu Sans"
GRN = "#39E27D"                          # laser green (light)
RED = "#FF5252"
BLUE = "#448AFF"
SKY = "#4FC3F7"
AMBER = "#FFCA28"                        # cross term, formula guides
CYAN = "#4DD0E1"                         # back-to-front
PURPLE = "#E040FB"                       # square roots
ORANGE = "#FF9E40"
DIM = "#8A8A8A"
GHOST = "#3A3A3A"
SLMC = "#90A4AE"

HEADER_Y = 3.62
CAPTION_Y = -3.5
MATHCH = set("×⊙→≈∼−∈ℝ≪≤≥↓↔∑∏∬∫ℱ𝒫𝒢𝒩ℋ𝒪𝒰𝔼∗≠∞∂∘⊤●⇒∠✓√′⟶")
SYMCH = set("✗■")


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


SQ = sup("2")
W1, W2 = I("w") + sub("1"), I("w") + sub("2")
A1, A2 = "α" + sub("1"), "α" + sub("2")
C1, C2 = I("c") + sub("1"), I("c") + sub("2")
PH1, PH2 = "φ" + sub("1"), "φ" + sub("2")
Ii = I("i")


def e_(x):
    return I("e") + sup(x)


# ------------------------------------------------------------------ colour maps
def gmap(v):
    """[0,1] → laser-green RGBA uint8 (crests white-green), as in the multisource explainer."""
    v = np.clip(v, 0, 1)
    a = np.stack([0.10 * v + 0.75 * v ** 3, v ** 0.85, 0.25 * v + 0.60 * v ** 3, np.ones_like(v)], -1)
    return (a * 255).astype(np.uint8)


def cmap_rgb(a, cmap="inferno"):
    return (matplotlib.colormaps[cmap](np.clip(a, 0, 1))[..., :3] * 255).astype(np.uint8)


def disp(a):
    """Linear intensity → display (gamma 2.2)."""
    return (np.clip(a, 0, 1) ** (1 / 2.2) * 255).astype(np.uint8)


def rgba(a):
    a = np.asarray(a)
    if a.ndim == 2:
        a = np.stack([a] * 3, -1)
    if a.shape[-1] == 3:
        a = np.concatenate([a, np.full(a.shape[:-1] + (1,), 255, np.uint8)], -1)
    return np.ascontiguousarray(a.astype(np.uint8))


def gen(name):
    return os.path.join(GEN, name)


# ------------------------------------------------------------------ display assets
FCOL = {"R": np.array([0.88, 0.25, 0.98]), "Q": np.array([0.22, 0.89, 0.49]),
        "P": np.array([0.31, 0.76, 0.97]), "occ": np.array([1.0, 0.79, 0.16])}


def build_assets():
    os.makedirs(GEN, exist_ok=True)
    if os.path.exists(gen(ASSET_FLAG)):
        return
    with open(gen("lock"), "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        if os.path.exists(gen(ASSET_FLAG)):
            return
        if not os.path.exists(gen("done_sims")):
            import sims
            for k, sd in (("A", 1), ("C", 2), ("D", 3), ("E", 4), ("F", 5)):
                getattr(sims, "sim_" + k)(np.random.default_rng(sd))
            open(gen("done_sims"), "w").write("ok")
        # ---- C: Fig. 2 replica
        c = np.load(gen("ab_toy.npz"))
        emax = float(c["err_nrp"][-1].max())
        np.save(gen("c_gt.npy"), rgba(disp(c["gt"])))
        np.save(gen("c_gws.npy"), rgba(disp(c["gws"])))
        np.save(gen("c_nrp.npy"), np.stack([rgba(disp(x.astype(float))) for x in c["nrp"]]))
        np.save(gen("c_rpws.npy"), np.stack([rgba(disp(x.astype(float))) for x in c["rpws"]]))
        np.save(gen("c_enrp.npy"), np.stack([rgba(cmap_rgb(x / emax)) for x in c["err_nrp"]]))
        np.save(gen("c_erpws.npy"), np.stack([rgba(cmap_rgb(x / emax)) for x in c["err_rpws"]]))
        deficit = (c["gt"] - c["nrp"][-1].astype(float)).mean(-1)
        dmax = float(max(deficit.max(), c["cross"].max()))
        np.save(gen("c_deficit.npy"), rgba(cmap_rgb(np.clip(deficit, 0, None) / dmax, "magma")))
        np.save(gen("c_cross.npy"), rgba(cmap_rgb(c["cross"] / dmax, "magma")))
        # ---- E: Fig. 3 replica, error maps for Part 3
        e = np.load(gen("e_views.npz"))
        lin = lambda a: (a.astype(float) / 255) ** 2.2
        for m in ("gwssqrt", "rpws"):
            err = np.abs(lin(e[f"{m}_lf"]) - lin(e["gt_lf"])).mean(-1)
            np.save(gen(f"e_err_{m}.npy"), np.stack([rgba(cmap_rgb(x / 0.35)) for x in err]))
        # ---- F: side views (time-averaged intensity, one colour per source; occluder light dimmed)
        f = np.load(gen("xzF.npz"))
        for m in ("g", "r", "one"):
            out = 0
            for k, cc in FCOL.items():
                v = np.clip(f[f"{m}_{k}"] / 0.55, 0, 1) ** 0.5
                out = out + v[..., None] * cc * (0.32 if k == "occ" else 1.0)
            np.save(gen(f"f_{m}.npy"), rgba((np.clip(out, 0, 1) * 255).astype(np.uint8)))
        # ---- paper Fig. 3 crops (columns: GT, GWS-NRP, GWS + sqrt, RPWS w/o sqrt, RPWS full)
        im = Image.open(os.path.join(FIGS, "alpha_blending_ablation_hi.png")).convert("RGB")
        s = im.size[0] / 2000
        for name, x0 in (("gt", 18), ("gwssqrt", 812), ("rpws", 1604)):
            for row, (y0, y1) in (("fs", (62, 322)), ("lf", (328, 588))):
                im.crop((int(x0 * s), int(y0 * s), int((x0 + 394) * s), int(y1 * s))).save(
                    gen(f"fig3_{name}_{row}.png"))
        open(gen(ASSET_FLAG), "w").write("ok")


build_assets()


# ------------------------------------------------------------------ mobject helpers
def img(path, h=None, w=None):
    m = ImageMobject(path)
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


class FrameSeq:
    """ImageMobject whose pixels follow a ValueTracker index into a stack of equal-size RGBA frames."""

    def __init__(self, frames, h=None, w=None, tracker=None):
        self.frames = frames
        self.k = tracker if tracker is not None else ValueTracker(0)
        self.mob = arr_img(frames[0], h, w)
        self.mob.add_updater(lambda m: setattr(
            m, "pixel_array", self.frames[int(np.clip(round(self.k.get_value()), 0, len(self.frames) - 1))]))


def arrow(a, b, buff=0.1, color=WHITE, sw=3):
    return Arrow(a, b, buff=buff, color=color, stroke_width=sw, max_tip_length_to_length_ratio=0.18, tip_length=0.16)


def label(m, s, size=22, color=WHITE, d=UP, buff=0.15):
    """Label above/below m on a common baseline so labels in a row line up."""
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


def cell(ok, w=2.3, h=0.36):
    r = Rectangle(width=w, height=h, stroke_width=0, fill_color=GRN if ok else RED, fill_opacity=0.28)
    return VGroup(r, tx("✓" if ok else "✗", 22, GRN if ok else RED).move_to(r))


# ------------------------------------------------------------------ scene base
class Base(Scene):
    def setup(self):
        self.cap = None
        self.hdr = None

    def wait(self, duration=1.0, **kw):
        # real frames, so tracker-driven images keep updating during waits
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
        s = s.replace("₁", sub("1")).replace("₂", sub("2")).replace("²", sup("2"))
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
        self.cap_need = len(plain) / 16.0 + 0.9
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

    def part_card(self, k, title, hold=1.6):
        a = tx(f"Random-phase Wave Splatting  ·  Part {k} of 3", 28, DIM)
        b = tx(title, 48, bold=True)
        if b.width > 12.8:
            b.scale_to_fit_width(12.8)
        g = VGroup(a, b).arrange(DOWN, buff=0.32)
        self.add(g)
        self.wait(hold)
        self.remove(g)

    def clear_all(self, keep=(), run_time=0.6):
        self.uncaption()
        keep = set(keep) | {self.hdr}
        mobs = [m for m in self.mobjects if m not in keep]
        for m in mobs:
            m.clear_updaters()
        if mobs:
            self.play(*[FadeOut(m) for m in mobs], run_time=run_time)

    def run(self, tracker, target, t, rate=linear):
        self.play(tracker.animate.set_value(target), run_time=t, rate_func=rate)


# ================================================================== equations (paper notation)
def _ui(j="i"):
    return f"|{I('u')}{sub(I(j))}|"


def eq_gws(nrp=False, hl=False):
    """Eq. 1 (GWS) or Eq. 3 (GWS-NRP)."""
    ph = (I("ikz") + sub(I("i"))) if not nrp else (I("ikz") + sub(I("i")) + " + " + col(I("i") + "φ" + sub(I("i")), AMBER))
    expo = f"{I('e')}{supsub(I('ikz'), I('i'))}" if not nrp else \
        f"exp({I('ikz')}{sub(I('i'))} + {col(I('i') + 'φ' + sub(I('i')), AMBER)})"
    amp = f"{I('c')}{sub(I('i'))} {I('o')}{sub(I('i'))} {_ui()}"
    if hl:
        amp = col(amp, PURPLE)
    return (f"{I('u')}{sub('SLM')} = ∑{sub(I('i'))} 𝒫( {amp} {I('T')}{sub(I('i'))} {expo} ; −{I('z')}{sub(I('i'))} )")


def eq_gws_T(hl=False):
    s = f"{I('T')}{sub(I('i'))} = ∏{sub(I('j') + '&lt;' + I('i'))} (1 − {I('o')}{sub(I('j'))} {_ui('j')})"
    return col(s, CYAN) if hl else s


def eq_rpws(hl=()):
    gi = f"{I('g')}{sub(I('i'))}"
    rec_l = f"{I('g')}{sub(I('i') + '−1')}{sup('(' + I('t') + ')')} = 𝒫( "
    mg = f"{I('M')}{sub(I('i'))} {gi}"
    sq = f"√{I('c')}{sub(I('i'))} √({I('o')}{sub(I('i'))}{_ui()})"
    ph = f"{I('i')}φ{sub(I('i'))}{sup('(' + I('t') + ')')}"
    tail = " ; Δ" + I("z") + " )"
    if "order" in hl:
        rec_l, mg, tail = col(rec_l, CYAN), col(mg, CYAN), col(tail, CYAN)
    if "sqrt" in hl:
        sq = col(sq, PURPLE)
    if "phase" in hl:
        ph = col(ph, AMBER)
    return f"{rec_l}{mg} + {sq} exp({I('i')}∠{I('u')}{sub(I('i'))} + {ph}){tail}"


def eq_mask(hl=()):
    s = f"{I('M')}{sub(I('i'))} = √(1 − {I('o')}{sub(I('i'))}{_ui()})"
    if "order" in hl or "sqrt" in hl:
        s = col(s, PURPLE if "sqrt" in hl and "order" not in hl else CYAN)
    return s


def eq_tm(hl=()):
    s = f"{I('I')} = (1/{I('T')}) ∑{sub(I('t'))} |𝒪( {I('u')}{sub('SLM')}{sup('(' + I('t') + ')')} )|{SQ}"
    return col(s, GRN) if "frames" in hl else s


EQ11 = (f"|{I('u')}|{SQ} = {W1}{SQ} + {W2}{SQ} + "
        + col(f"2{W1}{W2} cos({PH1} − {PH2})", AMBER))


# ================================================================== S00 Title (3-min cut only)
class S00_Title(Base):
    def construct(self):
        t1 = tx("Random-phase Wave Splatting", 58, bold=True).move_to([0, 2.5, 0])
        t2 = tx("of Translucent Primitives for Computer-generated Holography", 32).move_to([0, 1.78, 0])
        parts = VGroup(
            tx("1 · How RPWS differs from GWS", 30),
            tx("2 · Why random-phase alpha blending works", 30),
            tx("3 · Why back-to-front compositing works", 30),
        ).arrange(DOWN, buff=0.28, aligned_edge=LEFT).move_to([0, -0.2, 0])
        a1 = tx("Brian Chao*, Jacqueline Yang*, Suyeon Choi, Manu Gopakumar, Ryota Koiso, Gordon Wetzstein",
                22, DIM).move_to([0, -2.45, 0])
        a2 = tx("Stanford University  ·  KDDI Research  ·  arXiv 2508.17480", 22, DIM).move_to([0, -2.9, 0])
        self.add(t1, t2, a1, a2)
        self.wait(0.6)
        for p in parts:
            self.add(p)
            self.wait(0.7)
        self.wait(2.2)


# ================================================================== Part 1
class XZLive:
    """Computed x–z side view (rows = x, columns = z) of a primitive's wave, animated in time like the multisource
    WaveField. frames: (T, H, W) complex; k selects the time-multiplexed frame, avg blends to the T-frame mean |u|²."""

    def __init__(self, frames, h, center):
        self.F = frames
        self.Iavg = np.mean(np.abs(frames) ** 2, 0)
        self.k = ValueTracker(0)
        self.avg = ValueTracker(0)
        self.t = 0.0
        H, W = frames.shape[1:]
        self.H, self.W = H, W
        self.mob = ImageMobject(np.zeros((H, W, 4), np.uint8)).set(height=h).move_to(center)
        self.mob.set_resampling_algorithm(RESAMPLING_ALGORITHMS["linear"])
        self.s = h / H
        self.mob.add_updater(self._update)

    def _update(self, mob, dt):
        self.t += dt * 0.9
        a = self.avg.get_value()
        if a < 0.999:
            U = self.F[int(np.clip(self.k.get_value(), 0, len(self.F) - 1))]
            re = np.real(U * np.exp(-2j * np.pi * self.t))
            v = np.clip(re, 0, None) * 0.55 + 0.42 * np.clip(np.abs(U), 0, 1.3) ** 2
        else:
            v = 0
        if a > 0.001:
            v = (1 - a) * v + a * np.clip(self.Iavg, 0, 1) ** 0.5
        mob.pixel_array = gmap(v)

    def pt(self, z, x):
        """Screen point of (z px, transverse x px; x = 0 on the axis, +x downward like the array rows)."""
        L = self.mob.get_left()[0]
        cy = self.mob.get_center()[1]
        return np.array([L + (z + 0.5) * self.s, cy - x * self.s, 0])


def spectrum_plot(f, S, lam, color, w=3.0, h=1.05):
    ax = Axes(x_range=[-1, 1, 0.5], y_range=[0, 1.15, 1], x_length=w, y_length=h, tips=False,
              axis_config=dict(stroke_color=DIM, stroke_width=1.5, include_ticks=False))
    x = f * lam
    keep = np.abs(x) <= 1
    x, y = x[keep], np.clip(S[keep], 0, 1.15)
    pts = [ax.c2p(a, b) for a, b in zip(x, y)]
    curve = VMobject(stroke_color=color, stroke_width=2).set_points_as_corners(pts)
    fill = Polygon(ax.c2p(x[0], 0), *pts, ax.c2p(x[-1], 0), stroke_width=0, fill_color=color, fill_opacity=0.25)
    return VGroup(ax, fill, curve), ax


class P1_Difference(Base):
    def construct(self):
        self.part_card(1, "How RPWS differs from GWS")
        self.header("1 · How RPWS differs from GWS: the math")
        # ---------------------------------------------------------- physics: smooth vs random phase
        Us = np.load(gen("xzA_smooth.npy"))[None]
        Ur = np.load(gen("xzA_rand.npy"))
        meta = np.load(gen("xzA_meta.npz"))
        h = 2.3
        top = XZLive(Us, h, [-4.0, 1.42, 0])
        bot = XZLive(Ur, h, [-4.0, -1.18, 0])
        lt = label(top.mob, f"smooth phase  {col('(GWS)', DIM)}", 22)
        lb = label(bot.mob, f"random phase  {col('(RPWS)', DIM)}", 22)
        prim = [Line(p.pt(0, -60), p.pt(0, 60), color=WHITE, stroke_width=4) for p in (top, bot)]
        ptxt = tx("primitive", 18, DIM).next_to(prim[0], LEFT, buff=0.1)
        note = toy_note("toy simulation · λ = 6 px, SLM pitch p = 10 px").next_to(bot.mob, DOWN, buff=0.1)
        self.add(top.mob, bot.mob, framed(top.mob, GHOST, 1), framed(bot.mob, GHOST, 1), lt, lb, *prim, ptxt, note)
        self.caption("GWS gives each primitive a smooth phase, so its light leaves as a narrow beam")
        self.wait(1.0)
        # random frames flicker
        self.add(bot.k)
        self.caption("RPWS multiplies each primitive by a random phase φ ∼ 𝒰(−π, π): light fills the whole cone")
        self.play(bot.k.animate.set_value(10), run_time=4.0, rate_func=linear)
        # spectra (S3.1)
        sp_s, ax_s = spectrum_plot(meta["f"], meta["Ss"], 6.0, GRN)
        sp_r, ax_r = spectrum_plot(meta["f"], meta["Sr"] / np.percentile(meta["Sr"], 95) * 0.8, 6.0, GRN)
        sp_s.move_to([0.25, 1.35, 0])
        sp_r.move_to([0.25, -1.25, 0])
        band = VGroup()
        for ax in (ax_s, ax_r):
            for xb in (-0.3, 0.3):
                band.add(DashedLine(ax.c2p(xb, 0), ax.c2p(xb, 1.1), color=AMBER, stroke_width=2, dash_length=0.06))
        bl = tx(f"SLM band  ±{I('λ')}/2{I('p')}", 18, AMBER).next_to(ax_r, DOWN, buff=0.12)
        tl = label(sp_s, f"|{I('û')}({I('k')})|{SQ}  vs  sin {I('θ')}", 20, DIM)
        self.add(sp_s, sp_r, band, bl, tl)
        self.caption("In frequency: a smooth phase is a spike, a random phase is flat across the SLM band (Wiener–Khinchin)")
        self.play(bot.k.animate.set_value(20), run_time=3.0, rate_func=linear)
        # time average + S3.2 spread formula
        fx = tx(f"σ{SQ}({I('z')}) ≈ σ{SQ}(0) + ({I('z')}/{I('k')}){SQ} σ{SQ}{sub(I('k'))}", 30, AMBER).move_to([4.8, 1.4, 0])
        fxn = tx("spread of a propagating wave (supp. S3.2)", 18, DIM).next_to(fx, DOWN, buff=0.14)
        self.add(fx, fxn)
        zz = np.arange(0, 480, 4)

        def guides(p, sig):
            g = VGroup()
            for sgn in (1, -1):
                pts = [p.pt(z, sgn * sig[z]) for z in zz]
                g.add(DashedVMobject(VMobject(stroke_color=AMBER, stroke_width=3).set_points_smoothly(pts),
                                     num_dashes=40))
            return g
        g_s = guides(top, meta["sig_pred_s"])
        g_r = guides(bot, meta["sig_pred_r"])
        self.caption(f"Average 24 frames: the glow spreads exactly as σ²({I('z')}) ≈ σ²(0) + ({I('z')}/{I('k')})²σ²{sub(I('k'))} predicts (dashed)")
        self.play(top.avg.animate.set_value(1), bot.avg.animate.set_value(1), run_time=1.2)
        self.play(Create(g_s), Create(g_r), run_time=1.5)
        out = VGroup(
            tx(f"{col('random phase', GRN)}: large eyebox,", 24),
            tx("parallax, natural defocus", 24),
            tx(f"{col('smooth phase', DIM)}: a sliver of", 24),
            tx("the SLM bandwidth", 24),
        ).arrange(DOWN, buff=0.12, aligned_edge=LEFT).move_to([4.75, -1.15, 0])
        self.add(out)
        self.caption("Wide angles are what give holograms a large eyebox, parallax and natural defocus")
        self.hold(0.6)
        self.clear_all()
        # ---------------------------------------------------------- the equations
        lg = tx(f"GWS  {col('(Choi et al. 2025)', DIM)}", 24).move_to([-4.95, 2.85, 0])
        e1 = tx(eq_gws(), 30).move_to([0.9, 2.85, 0])
        e2 = tx(eq_gws_T(), 26).move_to([0.9, 2.3, 0])
        lr = tx(f"RPWS  {col('(ours)', DIM)}", 24, GRN).move_to([-4.95, 1.45, 0])
        e4 = tx(eq_rpws(), 30).move_to([0.9, 1.45, 0])
        e5 = tx(eq_mask(), 26).move_to([-1.3, 0.88, 0])
        e6 = tx(eq_tm(), 26).move_to([3.1, 0.88, 0])
        for e in (e1, e4):
            if e.width > 10.8:
                e.scale_to_fit_width(10.8)
        tags = [tx("Eqs. 1–2", 16, DIM).next_to(e2, RIGHT, buff=0.3), tx("Eqs. 4–6", 16, DIM).next_to(e6, RIGHT, buff=0.3)]
        self.add(lg, e1, e2, lr, e4, e5, e6, *tags)
        self.caption("GWS sums primitive waves front to back. RPWS composites them back to front — over T frames")
        self.hold(0.3)
        # table rows revealed with matching highlight colours
        X0, X1, X2 = -6.6, -2.1, 3.3
        hdr = VGroup(tx("GWS", 24, DIM).move_to([X1, 0.2, 0]), tx("RPWS", 24, GRN).move_to([X2, 0.2, 0]))
        rule = Line([-6.4, -0.05, 0], [6.4, -0.05, 0], color=GHOST, stroke_width=1.5)
        self.add(hdr, rule)
        rows = [
            ("phase", AMBER, "phase", f"smooth: ∠{I('u')}{sub(I('i'))} = {I('kz')}{sub(I('i'))}",
             f"random φ{sub(I('i'))}{sup('(' + I('t') + ')')} ∼ 𝒰(−π, π)"),
            ("sqrt", PURPLE, "blending", f"amplitude  {I('c')} {I('o')}|{I('u')}|", f"intensity  √{I('c')} √({I('o')}|{I('u')}|)  → Part 2"),
            ("order", CYAN, "order", "front → back, one global map T", "back → front, mask at each plane  → Part 3"),
            ("frames", GRN, "frames", "1", f"{I('T')}, averaged  → Part 2"),
        ]
        hl = []
        caps = {
            "phase": "① Random phase, redrawn in every frame t, instead of a smooth phase",
            "sqrt": "② Square roots: RPWS blends intensities, GWS blends amplitudes",
            "order": "③ Back to front: light is masked at each primitive's own plane",
            "frames": "④ The hologram is an average over T frames — and that average is the alpha blend",
        }
        for i, (key, c, name, g, r) in enumerate(rows):
            y = -0.5 - 0.55 * i
            hl.append(key)
            self.remove(e4, e5, e6)
            e4 = tx(eq_rpws(hl), 30).move_to([0.9, 1.45, 0])
            if e4.width > 10.8:
                e4.scale_to_fit_width(10.8)
            e5 = tx(eq_mask(hl), 26).move_to([-1.3, 0.88, 0])
            e6 = tx(eq_tm(hl), 26).move_to([3.1, 0.88, 0])
            row = VGroup(tx(name, 22, c).move_to([0, y, 0]).align_to(np.array([X0, 0, 0]), LEFT),
                         tx(g, 22, DIM).move_to([X1, y, 0]), tx(r, 22, c).move_to([X2, y, 0]))
            self.add(e4, e5, e6, row)
            if key == "order":
                self.remove(e2)
                e2 = tx(eq_gws_T(True), 26).move_to([0.9, 2.3, 0])
                self.add(e2)
            if key == "sqrt":
                self.remove(e1)
                e1 = tx(eq_gws(hl=True), 30).move_to([0.9, 2.85, 0])
                if e1.width > 10.8:
                    e1.scale_to_fit_width(10.8)
                self.add(e1)
            self.caption(caps[key])
        self.hold(0.4)
        self.clear_all()
        # ---------------------------------------------------------- GWS-NRP and the 2 × 2 ablation (paper Fig. 3)
        e3 = tx(eq_gws(nrp=True), 30).move_to([0, 2.85, 0])
        e3n = tx("GWS-NRP (Eq. 3): GWS with a random phase, as suggested in the GWS supplement", 20, DIM).next_to(e3, DOWN, buff=0.14)
        self.add(e3, e3n)
        self.caption("Why not simply add a random phase to GWS? The paper shows that this provably fails")
        E = np.load(gen("e_views.npz"))
        names = [("gt", "ground truth"), ("nrp", "GWS-NRP"), ("gwssqrt", "GWS + √"), ("nosqrt", "RPWS w/o √"),
                 ("rpws", "RPWS")]
        kv = ValueTracker(12)
        panels = []
        W = 2.45
        for j, (m, nm) in enumerate(names):
            fs = FrameSeq(np.stack([rgba(a) for a in E[f"{m}_lf"]]), w=W, tracker=kv)
            fs.mob.move_to([-5.2 + 2.6 * j, 0.25, 0])
            panels.append(fs.mob)
            self.add(fs.mob, framed(fs.mob, GRN if m == "rpws" else GHOST, 1.5),
                     label(fs.mob, nm, 22, GRN if m == "rpws" else WHITE))
        r1 = tx("back-to-front", 20, DIM).move_to([-5.2, -1.38, 0])
        r2 = tx("√ blending", 20, DIM).move_to([-5.2, -1.8, 0])
        cells = VGroup()
        for j, (b2f, sq) in enumerate([(False, False), (False, True), (True, False), (True, True)]):
            x = -2.6 + 2.6 * j
            cells.add(cell(b2f, 2.45).move_to([x, -1.38, 0]), cell(sq, 2.45).move_to([x, -1.8, 0]))
        tn = toy_note("toy simulation of paper Fig. 3 · λ = 520 nm, 8 µm pixels, 24 frames · light-field view").move_to([0, -2.3, 0])
        self.add(r1, r2, cells, tn, kv)
        self.caption("GWS-NRP gets the colours wrong: the translucent primitives come out dark")
        self.run(kv, 24, 2.2, smooth)
        self.caption("Fixing only the blending (GWS + √) still breaks at occlusion borders once you look off-axis")
        self.run(kv, 0, 3.5, smooth)
        self.run(kv, 24, 3.0, smooth)
        self.caption("Only both changes reproduce the scene. Part 2: why √. Part 3: why back to front")
        self.run(kv, 12, 2.0, smooth)
        self.hold(0.5)


# ================================================================== Part 2
class P2_AlphaBlending(Base):
    def construct(self):
        self.part_card(2, "Why random-phase alpha blending works")
        self.header("2 · Random-phase alpha wave blending")
        # ---------------------------------------------------------- setup: Eqs. 7, 8, 11
        gtimg = arr_img(np.load(gen("c_gt.npy")), 2.6).move_to([-4.6, 0.4, 0])
        self.add(gtimg, framed(gtimg, GHOST, 1))
        l1 = tx(f"{I('u')}{sub('1')} = {A1} {e_(I('i') + PH1)}  (front)", 22, ORANGE).next_to(gtimg, DOWN, buff=0.2)
        l2 = tx(f"{I('u')}{sub('2')} = {A2} {e_(I('i') + PH2)}  (behind)", 22, SKY).next_to(l1, DOWN, buff=0.12)
        self.add(l1, l2)
        r7 = tx(f"ray splatting:   {I('c')} = {A1}{C1} + (1 − {A1}){A2}{C2}", 30).move_to([1.9, 1.7, 0])
        r8 = tx(f"wave splatting:   {I('u')} = {W1} {e_(I('i') + PH1)} + {W2} {e_(I('i') + PH2)}", 30).move_to([1.9, 0.85, 0])
        r11 = tx(EQ11, 32).move_to([1.9, -0.25, 0])
        tags = [tx(f"Eq. {n}", 16, DIM).next_to(m, RIGHT, buff=0.25) for n, m in ((7, r7), (8, r8), (11, r11))]
        self.add(r7, tags[0])
        self.caption("Two translucent primitives over one pixel. Ray splatting blends their colours like this")
        self.add(r8, tags[1])
        self.caption("With waves, each primitive also carries a phase. Which weights w make |u| equal the colour c?")
        self.add(r11, tags[2])
        self.caption("Square it: the intensity has an extra cross term that depends on the phase difference")
        self.hold(0.3)
        self.clear_all(keep=[r11, tags[2]])
        self.play(VGroup(r11, tags[2]).animate.move_to([0, 2.8, 0]), run_time=0.6)
        # ---------------------------------------------------------- phasor diagram
        O = np.array([-3.9, -0.5, 0])
        U = 1.45
        grid = VGroup(Line(O + LEFT * 2.4, O + RIGHT * 2.4, color=GHOST, stroke_width=1.5),
                      Line(O + DOWN * 2.3, O + UP * 2.3, color=GHOST, stroke_width=1.5))
        unit = DashedVMobject(Circle(radius=U, color=DIM, stroke_width=1.5).move_to(O), num_dashes=48)
        unit_l = tx(f"|{I('u')}| = {I('c')} = 1", 18, DIM).move_to(O + np.array([1.35, 1.35, 0]))
        rng = np.random.default_rng(11)
        NF = 60
        ph1 = rng.uniform(-np.pi, np.pi, NF)
        ph2 = rng.uniform(-np.pi, np.pi, NF)
        w1 = ValueTracker(0.5)
        w2 = ValueTracker(0.5)
        nf = ValueTracker(0)            # frames drawn so far
        rand = ValueTracker(0)          # 0: smooth (aligned), 1: random
        PH0 = 0.55
        samples = {"key": None, "vals": []}

        def phases():
            if rand.get_value() < 0.5:
                return PH0, PH0
            j = int(np.clip(nf.get_value(), 0, NF - 1))
            return ph1[j], ph2[j]

        def phasors():
            a, b = phases()
            p1 = O + U * w1.get_value() * np.array([np.cos(a), np.sin(a), 0])
            p2 = p1 + U * w2.get_value() * np.array([np.cos(b), np.sin(b), 0])
            g = VGroup(
                Arrow(O, p1, buff=0, color=ORANGE, stroke_width=5, max_tip_length_to_length_ratio=0.15),
                Arrow(p1, p2, buff=0, color=SKY, stroke_width=5, max_tip_length_to_length_ratio=0.15),
                Arrow(O, p2, buff=0, color=WHITE, stroke_width=3, max_tip_length_to_length_ratio=0.12),
            )
            return g
        ph_mob = always_redraw(phasors)
        lab_w = always_redraw(lambda: tx(
            f"{col(W1, ORANGE)} = {w1.get_value():.2f}    {col(W2, SKY)} = {w2.get_value():.2f}", 24).move_to(O + np.array([0, 2.5, 0])))
        # intensity meter
        BX, BY, BS = 0.6, -2.35, 1.55

        def mean_I():
            if rand.get_value() < 0.5:
                return (w1.get_value() + w2.get_value()) ** 2, 1
            n = int(np.clip(nf.get_value(), 0, NF - 1)) + 1
            v = np.abs(w1.get_value() * np.exp(1j * ph1[:n]) + w2.get_value() * np.exp(1j * ph2[:n])) ** 2
            return v.mean(), n

        def meter():
            a, b = phases()
            cur = abs(w1.get_value() * np.exp(1j * a) + w2.get_value() * np.exp(1j * b)) ** 2
            m, n = mean_I()
            g = VGroup()
            for x, v, c in ((BX, cur, WHITE), (BX + 1.3, m, AMBER)):
                g.add(Rectangle(width=0.7, height=max(v * BS, 0.01), stroke_width=0, fill_color=c, fill_opacity=0.85)
                      .move_to([x, BY + v * BS / 2, 0]))
            t1 = w1.get_value() ** 2 + w2.get_value() ** 2
            g.add(DashedLine([BX - 0.6, BY + BS, 0], [BX + 2.0, BY + BS, 0], color=WHITE, stroke_width=2))
            g.add(DashedLine([BX - 0.6, BY + t1 * BS, 0], [BX + 2.0, BY + t1 * BS, 0], color=SKY, stroke_width=2))
            g.add(tx(f"{I('c')}{SQ} = 1", 20).next_to([BX + 2.0, BY + BS, 0], RIGHT, buff=0.1))
            g.add(tx(f"{W1}{SQ} + {W2}{SQ} = {t1:.2f}", 20, SKY).next_to([BX + 2.0, BY + t1 * BS, 0], RIGHT, buff=0.1)
                  .shift(DOWN * (0.22 if abs(t1 - 1) < 0.12 else 0)))
            g.add(tx(f"|{I('u')}|{SQ}", 18).move_to([BX, BY - 0.25, 0]),
                  tx(f"mean of {n}" if rand.get_value() > 0.5 else "mean", 18, AMBER).move_to([BX + 1.3, BY - 0.25, 0]))
            return g
        mt = always_redraw(meter)
        self.add(grid, unit, unit_l, ph_mob, lab_w, mt)
        ex = tx(f"example: {A1} = 0.5, {A2} = 1, {C1} = {C2} = 1  →  {I('c')} = 1", 20, DIM).move_to([4.5, 1.9, 0])
        self.add(ex)
        s12 = tx(f"smooth phase ({PH1} = {PH2}):", 24).move_to([4.6, 1.0, 0])
        s12b = tx(f"|{I('u')}| = {W1} + {W2}  → amplitudes add", 24).move_to([4.6, 0.55, 0])
        s12c = tx("Eq. 12 = GWS alpha wave blending", 18, DIM).move_to([4.6, 0.15, 0])
        self.add(s12, s12b, s12c)
        self.caption(f"Smooth phase: the phasors line up and amplitudes add — GWS weights {W1} = {A1}{C1}, {W2} = (1 − {A1}){A2}{C2}")
        self.hold(0.4)
        # random phase with GWS weights (GWS-NRP)
        self.remove(s12, s12b, s12c)
        r1 = tx(f"random phase, GWS weights:", 24).move_to([4.6, 1.0, 0])
        r1b = tx(f"𝔼[cos({PH1} − {PH2})] = 0", 24, AMBER).move_to([4.6, 0.55, 0])
        r1c = tx(f"𝔼|{I('u')}|{SQ} = {W1}{SQ} + {W2}{SQ}   (Eqs. 13–14)", 22).move_to([4.6, 0.1, 0])
        self.add(r1, r1b, r1c)
        rand.set_value(1)
        self.caption("Random phase: every frame the phasors point somewhere new, and the cross term averages to zero")
        self.run(nf, 30, 6.0)
        r1d = tx(f"too dark: GWS-NRP loses 2{W1}{W2}", 22, RED).move_to([4.6, -0.4, 0])
        self.add(r1d)
        self.caption(f"So with GWS weights the average lands on {W1}² + {W2}² = 0.5, half the brightness it should have")
        self.hold(0.3)
        # RPWS square-root weights
        self.remove(r1, r1c, r1d)
        q1 = tx(f"RPWS: fit primitives to {I('c')}{SQ} (Eq. 15)", 24, PURPLE).move_to([4.6, 1.0, 0])
        q2 = tx(f"{W1} = √({A1}′{C1}′),  {W2} = √((1 − {A1}′){A2}′{C2}′)", 22, PURPLE).move_to([4.6, 0.1, 0])
        q3 = tx(f"𝔼|{I('u')}|{SQ} = {A1}′{C1}′ + (1 − {A1}′){A2}′{C2}′ = {I('c')}{SQ}", 22).move_to([4.6, -0.4, 0])
        self.add(q1, q2)
        nf.set_value(0)
        self.play(w1.animate.set_value(np.sqrt(0.5)), w2.animate.set_value(np.sqrt(0.5)), run_time=1.2)
        self.add(q3)
        self.caption("RPWS fits its primitives to intensity and takes square roots: on average |u|² = c² exactly (Eq. 17)")
        self.run(nf, 40, 6.0)
        self.hold(0.3)
        self.clear_all()
        # ---------------------------------------------------------- pixel toy, paper Fig. 2
        C = np.load(gen("ab_toy.npz"))
        tl = list(C["t"])
        kt = ValueTracker(0)
        h = 2.25
        xs = [-5.0, -1.75, 1.5, 4.75]
        gt = arr_img(np.load(gen("c_gt.npy")), h).move_to([xs[0], 1.25, 0])
        gw = arr_img(np.load(gen("c_gws.npy")), h).move_to([xs[1], 1.25, 0])
        nr = FrameSeq(np.load(gen("c_nrp.npy")), h, tracker=kt)
        rp = FrameSeq(np.load(gen("c_rpws.npy")), h, tracker=kt)
        nr.mob.move_to([xs[2], 1.25, 0])
        rp.mob.move_to([xs[3], 1.25, 0])
        en = FrameSeq(np.load(gen("c_enrp.npy")), 1.75, tracker=kt)
        er = FrameSeq(np.load(gen("c_erpws.npy")), 1.75, tracker=kt)
        en.mob.move_to([xs[2], -1.35, 0])
        er.mob.move_to([xs[3], -1.35, 0])
        for m, nm, c in ((gt, "ground truth", WHITE), (gw, "GWS (smooth phase)", WHITE), (nr.mob, "GWS-NRP", RED),
                         (rp.mob, "RPWS", GRN)):
            self.add(m, framed(m, GHOST, 1), label(m, nm, 22, c))
        self.add(en.mob, er.mob, framed(en.mob, GHOST, 1), framed(er.mob, GHOST, 1),
                 label(en.mob, "squared error", 18, DIM), label(er.mob, "squared error", 18, DIM))
        cnt = always_redraw(lambda: tx(
            f"{I('T')} = {tl[int(np.clip(round(kt.get_value()), 0, len(tl) - 1))]} frames", 30, AMBER).move_to([-3.4, -1.35, 0]))
        tn = toy_note("toy simulation of paper Fig. 2 · per-pixel random phase").move_to([-3.4, -1.95, 0])
        self.add(cnt, tn, kt)
        self.caption("The same rule per pixel, as in the paper's Fig. 2: orange in front of blue")
        self.run(kt, 23, 5.0)
        self.caption("RPWS converges to the ground truth as frames pile up. GWS-NRP never does")
        self.run(kt, len(tl) - 1, 3.0)
        self.hold(0.3)
        self.clear_all()
        dm = arr_img(np.load(gen("c_deficit.npy")), 2.9).move_to([-2.2, 0.35, 0])
        cm = arr_img(np.load(gen("c_cross.npy")), 2.9).move_to([2.2, 0.35, 0])
        eqs = tx("≈", 60).move_to([0, 0.35, 0])
        self.add(dm, cm, framed(dm, GHOST, 1), framed(cm, GHOST, 1), eqs,
                 label(dm, "measured: ground truth − GWS-NRP", 22),
                 label(cm, f"predicted: cross term 2{W1}{W2}", 22, AMBER))
        self.caption(f"The missing light is a ring where the two overlap — exactly the cross term 2{W1}{W2} of Eq. 11")
        self.hold(0.4)
        self.clear_all()
        # ---------------------------------------------------------- convergence, supp. Fig. S3
        D = np.load(gen("conv.npz"))
        N = D["N"]
        keep = N >= 10
        ax = Axes(x_range=[1, 4, 1], y_range=[-0.3, 0.3, 0.1], x_length=7.2, y_length=4.2, tips=False,
                  axis_config=dict(stroke_color=DIM, stroke_width=1.5, include_ticks=True, tick_size=0.05))
        ax.move_to([-2.4, -0.15, 0])
        xl = VGroup(*[tx(f"10{sup(str(k))}", 18, DIM).next_to(ax.c2p(k, -0.3), DOWN, buff=0.12) for k in (1, 2, 3, 4)])
        yl = VGroup(*[tx(t_, 16, DIM).next_to(ax.c2p(1, v), LEFT, buff=0.1)
                      for t_, v in (("−0.2", -0.2), ("0", 0.0), ("0.2", 0.2))])
        xt = tx(f"frames {I('N')}", 20, DIM).next_to(ax, DOWN, buff=0.5)
        yt = tx(f"cross term averaged over {I('N')} frames:  (1/{I('N')}) ∑ cos({PH1} − {PH2})", 20, DIM).next_to(ax, UP, buff=0.15)

        def env(key, c, op):
            x = np.log10(N[keep])
            s = D[key + "_std"][keep]
            pts = [ax.c2p(a, b) for a, b in zip(x, s)] + [ax.c2p(a, -b) for a, b in zip(x[::-1], s[::-1])]
            return Polygon(*pts, stroke_width=0, fill_color=c, fill_opacity=op)

        def curve(y, c, dashed=True):
            x = np.log10(N[keep])
            m = VMobject(stroke_color=c, stroke_width=2.5).set_points_smoothly([ax.c2p(a, b) for a, b in zip(x, y)])
            return DashedVMobject(m, num_dashes=50) if dashed else m
        e2 = env("two", PURPLE, 0.28)
        eu = env("uniform", GRN, 0.45)
        th = VGroup(curve(1 / np.sqrt(2 * N[keep]), WHITE), curve(-1 / np.sqrt(2 * N[keep]), WHITE))
        trials = VGroup(*[curve(D["uniform_trials"][i][keep], GRN, False).set_stroke(width=1, opacity=0.6)
                          for i in range(10)])
        self.add(ax, xl, yl, xt, yt)
        leg = VGroup(
            tx(f"{col('■', GRN)} uniform φ ∼ 𝒰(−π, π)", 22),
            tx(f"{col('■', PURPLE)} two values φ ∈ {{0, π}}", 22),
            tx(f"{col('- -', WHITE)} ±1/√(2{I('N')})  (supp. S4)", 22),
        ).arrange(DOWN, buff=0.18, aligned_edge=LEFT).move_to([4.35, 1.5, 0])
        self.add(leg)
        self.play(Create(trials), FadeIn(e2), FadeIn(eu), run_time=2.0)
        self.add(th)
        self.caption("How fast does the cross term vanish? The supplement proves 1/√(2N), the fastest possible rate")
        cond = VGroup(
            tx("optimal  ⇔  𝔼[e" + sup(I("i") + "φ") + "] = 𝔼[e" + sup("2" + I("i") + "φ") + "] = 0", 24, AMBER),
            tx(f"uniform 2π: ✓, and it uses the full SLM band", 22, GRN),
            tx(f"{{0, π}}: 𝔼[e{sup('2' + I('i') + 'φ')}] = 1 → slower, 1/√{I('N')}", 22, PURPLE),
        ).arrange(DOWN, buff=0.2, aligned_edge=LEFT).move_to([4.3, -0.6, 0])
        self.add(cond)
        self.caption("Uniform 2π phase hits that rate — a two-level phase does not")
        self.caption("Time multiplexing is not a speckle trick: it is the alpha blend itself")
        self.hold(0.6)


# ================================================================== Part 3
FZ_O, FZ_W, F_HALF = 200, 440, 55
F_SRC = {"R": -150.0, "Q": -28.0, "P": 92.0}
TH = np.radians(-13.0)


class P3_BackToFront(Base):
    def construct(self):
        self.part_card(3, "Why back-to-front compositing works")
        self.header("3 · Back-to-front wave compositing")
        h = 4.3
        cx, cy = -3.95, 0.05
        s = h / 400
        W = FZ_W * s
        L = cx - W / 2

        def P(z, x):
            return np.array([L + z * s, cy - x * s, 0])
        rear = Line(P(0, -200), P(0, 200), color=DIM, stroke_width=2)
        occ = Rectangle(width=0.09, height=2 * F_HALF * s, stroke_color=WHITE, stroke_width=1.5,
                        fill_color=GREY_B, fill_opacity=0.9).move_to(P(FZ_O, 0))
        slm = Rectangle(width=0.14, height=h, stroke_color=SLMC, stroke_width=1.2, fill_color=SLMC,
                        fill_opacity=0.35).move_to(P(FZ_W, 0) + RIGHT * 0.07)
        dots = VGroup(*[Dot(P(0, x), radius=0.075, color=ManimColor.from_rgb(tuple(FCOL[k]))) for k, x in F_SRC.items()])
        dl = VGroup(*[tx(k, 22, ManimColor.from_rgb(tuple(FCOL[k]))).next_to(P(0, x), LEFT, buff=0.15) for k, x in F_SRC.items()])
        lab = VGroup(label(rear, "rear primitives", 18, DIM).shift(RIGHT * 0.3),
                     tx("occluder", 18).next_to(occ, UP, buff=0.12),
                     label(slm, "SLM", 18, SLMC))
        self.add(rear, occ, slm, dots, dl, lab)
        # GWS: one transmittance map, the occluder projected straight back along z
        sil = VGroup(DashedLine(P(FZ_O, -F_HALF), P(0, -F_HALF), color=RED, stroke_width=2),
                     DashedLine(P(FZ_O, F_HALF), P(0, F_HALF), color=RED, stroke_width=2))
        hole = Rectangle(width=0.16, height=2 * F_HALF * s, stroke_width=0, fill_color=RED, fill_opacity=0.5).move_to(P(0, 0))
        tT = tx(f"GWS: {I('T')} = ∏(1 − {I('o')}{sub(I('j'))}|{I('u')}{sub(I('j'))}|)", 26).move_to([3.3, 2.4, 0])
        tT2 = tx("one map, applied to each primitive at its own plane", 20, DIM).next_to(tT, DOWN, buff=0.12)
        self.add(tT, tT2)
        self.caption("GWS computes one transmittance map T and multiplies each primitive by it before propagating")
        self.play(Create(sil), FadeIn(hole), run_time=1.2)
        self.caption("That is the occluder's shadow cast straight back along z — as if light only travelled on-axis")
        self.hold(0.2)
        # rays at −13°: Q's clears the occluder, P's runs into it
        tt = np.tan(TH)

        def ray(x0, z1, c, dashed=False, sw=3):
            a, b = P(0, x0), P(z1, x0 + z1 * tt)
            return DashedLine(a, b, color=c, stroke_width=sw, dash_length=0.08) if dashed else Line(a, b, color=c, stroke_width=sw)
        eye_x = F_SRC["Q"] + (FZ_W + 60) * tt
        eyeg = VGroup(Circle(radius=0.22, color=WHITE, stroke_width=2),
                      Dot(radius=0.06, color=WHITE).shift(LEFT * 0.2)).move_to(P(FZ_W + 75, eye_x))
        eyel = tx("eye at −13°", 18).next_to(eyeg, UP, buff=0.1)
        qr = ray(F_SRC["Q"], FZ_W + 55, GRN, True)
        qx = tx("✗", 34, RED).move_to(P(0, F_SRC["Q"]) + RIGHT * 0.25)
        self.add(eyeg, eyel)
        self.play(Create(qr), run_time=0.8)
        self.add(qx)
        b1 = tx(f"{col('Q', GRN)} sits in the shadow → GWS deletes it", 22).move_to([3.3, 1.35, 0])
        b2 = tx("but its slanted light clears the occluder: black hole", 22, RED).move_to([3.3, 0.95, 0])
        self.add(b1, b2)
        self.caption("Q is deleted — yet light leaving Q at a slant passes the occluder: the eye should see it")
        pr = VGroup(ray(F_SRC["P"], FZ_O, SKY), ray(F_SRC["P"], FZ_W + 55, RED, True))
        hit = P(FZ_O, F_SRC["P"] + FZ_O * tt)
        px = tx("✗", 30, WHITE).move_to(hit + RIGHT * 0.22)
        self.play(Create(pr), run_time=0.8)
        self.add(px)
        b3 = tx(f"{col('P', SKY)} is outside the shadow → GWS keeps all of it", 22).move_to([3.3, 0.25, 0])
        b4 = tx("including light that goes through the occluder: leak", 22, RED).move_to([3.3, -0.15, 0])
        self.add(b3, b4)
        self.caption("P keeps its whole cone — even the light that should hit the occluder: light leakage")
        b5 = tx(f"fine for smooth phase (on-axis light), wrong once light leaves at up to ±{I('θ')}{sub('max')}", 20, DIM).move_to([3.3, -0.9, 0])
        if b5.width > 6.4:
            b5.scale_to_fit_width(6.4)
        self.add(b5)
        self.hold(0.4)
        # ---------------------------------------------------------- waves: GWS + √ vs RPWS
        fg = arr_img(np.load(gen("f_g.npy")), h).move_to([cx, cy, 0])
        fr = arr_img(np.load(gen("f_r.npy")), h).move_to([cx, cy, 0])
        self.remove(b1, b2, b3, b4, b5, tT, tT2)
        self.add(fg)
        self.bring_to_front(rear, occ, slm, dots, dl, sil, hole, qr, qx, pr, px)
        wt = tx("waves: GWS + √ (supp. Alg. 5)", 26).move_to([3.3, 2.4, 0])
        wn = toy_note("toy simulation · λ = 6 px, p = 10 px, 24-frame average").move_to([3.3, 2.0, 0])
        self.add(wt, wn)
        g1 = tx(f"no light from {col('Q', GRN)} at all", 22).move_to([3.3, 1.2, 0])
        g2 = tx(f"{col('P', SKY)}'s cone streams through the occluder", 22).move_to([3.3, 0.8, 0])
        self.add(g1, g2)
        self.caption("The wave simulation agrees: Q is gone, and P's light pours straight through the occluder")
        self.hold(0.4)
        # RPWS recursion, revealed left → right
        self.remove(fg, g1, g2, wt, sil, hole, qx, px)
        self.remove(qr, pr)
        self.add(fr)
        self.bring_to_front(rear, occ, slm, dots, dl)
        zr = ValueTracker(0)
        cover = always_redraw(lambda: Rectangle(width=max(W - zr.get_value() * s, 0.001), height=h + 0.02,
                                                stroke_width=0, fill_color=BLACK, fill_opacity=1)
                              .align_to(P(zr.get_value(), 0), LEFT).set_y(cy))
        self.add(cover)
        self.bring_to_front(rear, occ, slm, dots, dl)
        wt = tx("RPWS (Eqs. 4–5, supp. Alg. 3)", 26, GRN).move_to([3.3, 2.4, 0])
        rec = tx(eq_rpws(("order",)), 22).move_to([3.3, 1.75, 0])
        if rec.width > 6.6:
            rec.scale_to_fit_width(6.6)
        self.add(wt, rec)
        st = [tx("① propagate the rear light to the occluder's plane", 21),
              tx(f"② multiply by {I('M')} = √(1 − α) there", 21, CYAN),
              tx("③ add the occluder's own wave, propagate on", 21)]
        for i, m in enumerate(st):
            m.move_to([3.3, 0.95 - 0.45 * i, 0])
        self.add(st[0])
        self.caption("RPWS works back to front: carry the light to each primitive's plane first")
        self.run(zr, FZ_O, 2.5)
        self.add(st[1])
        self.play(occ.animate.set_fill(CYAN, 0.9), run_time=0.4)
        self.play(occ.animate.set_fill(GREY_B, 0.9), run_time=0.4)
        self.caption("Then occlude it where the occluder actually is, for every direction of light")
        self.add(st[2])
        self.run(zr, FZ_W, 2.5)
        self.remove(cover)
        r1 = tx(f"{col('Q', GRN)} peeks around the edge", 22).move_to([3.3, -0.65, 0])
        r2 = tx(f"{col('P', SKY)} is cut off where it hits the occluder", 22).move_to([3.3, -1.05, 0])
        self.add(r1, r2)
        self.caption("Now Q's slanted light gets around the occluder and P's blocked light stays blocked")
        self.hold(0.4)
        self.clear_all()
        # ---------------------------------------------------------- 2-D toy: views across the eyebox
        E = np.load(gen("e_views.npz"))
        kv = ValueTracker(12)
        hh = 2.45
        xs = [-4.1, 0.0, 4.1]
        mobs = []
        for x, (m, nm, c) in zip(xs, (("gt", "ground truth", WHITE), ("gwssqrt", "GWS + √  (front → back)", RED),
                                      ("rpws", "RPWS  (back → front)", GRN))):
            fs = FrameSeq(np.stack([rgba(a) for a in E[f"{m}_lf"]]), hh, tracker=kv)
            fs.mob.move_to([x, 1.25, 0])
            mobs.append(fs.mob)
            self.add(fs.mob, framed(fs.mob, GHOST, 1), label(fs.mob, nm, 22, c))
        for x, m in zip(xs[1:], ("gwssqrt", "rpws")):
            fe = FrameSeq(np.load(gen(f"e_err_{m}.npy")), 1.7, tracker=kv)
            fe.mob.move_to([x, -1.45, 0])
            self.add(fe.mob, framed(fe.mob, GHOST, 1), label(fe.mob, "error vs ground truth", 18, DIM))
        bar = Line([-5.6, -1.2, 0], [-2.6, -1.2, 0], color=DIM, stroke_width=2)
        knob = always_redraw(lambda: Dot([-5.6 + 3.0 * kv.get_value() / 24, -1.2, 0], radius=0.09, color=AMBER))
        bl = tx("viewpoint across the eyebox", 18, DIM).next_to(bar, DOWN, buff=0.15)
        tn = toy_note("toy simulation · λ = 520 nm, 8 µm, 24 frames").next_to(bl, DOWN, buff=0.12)
        self.add(bar, knob, bl, tn, kv)
        self.caption("Look through the hologram from different viewpoints. Head-on, both methods match")
        self.run(kv, 24, 3.0, smooth)
        self.caption("Off-axis, GWS + √ shows dark gaps and leaks at the occlusion border. RPWS stays correct")
        self.run(kv, 0, 3.5, smooth)
        self.run(kv, 12, 1.8, smooth)
        self.hold(0.2)
        self.clear_all()
        # ---------------------------------------------------------- close: paper Fig. 3
        hh = 2.05
        ims = []
        for j, (m, nm, c) in enumerate((("gt", "ground truth", WHITE), ("gwssqrt", "GWS with √", RED),
                                         ("rpws", "RPWS (full)", GRN))):
            a = img(gen(f"fig3_{m}_fs.png"), hh).move_to([-4.3 + 4.3 * j, 1.05, 0])
            b = img(gen(f"fig3_{m}_lf.png"), hh).move_to([-4.3 + 4.3 * j, -1.15, 0])
            self.add(a, b, label(a, nm, 22, c))
        src = tx("paper Fig. 3 (simulated focal slice and light-field view)", 18, DIM).move_to([0, -2.45, 0])
        self.add(src)
        self.caption("RPWS is the painter's algorithm for waves: back to front, with propagation and square roots")
        self.hold(1.0)
