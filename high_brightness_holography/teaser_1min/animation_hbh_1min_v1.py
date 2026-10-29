"""High-Brightness Holographic Projection — 1-minute teaser, v1.

Condensed from the 5-min teaser (../animation_hbh_v3.py, ../storyboard.md). Source: Chao, Gopakumar, Choi, Wetzstein,
"High-brightness holographic projection", Optics Letters 48(15), 4041 (2023) + Supplement 1 + Visualization 1.
Reads ../data (supplement crops, video frames) and ../data/generated (toy caches built by ../precompute_toy.py and the
5-min script's build_assets(), flag done_v1) read-only. Helpers are copied from the 5-min script, not imported.

Render (see render.sh):  manim -ql --media_dir media animation_hbh_1min_v1.py <Scene>
Scenes, in order: T1_Hook, T2_Method, T3_Payoff.
"""
from manim import *
import numpy as np
from PIL import Image
import matplotlib
import glob
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
GEN = os.path.join(DATA, "generated")
VID = os.path.join(DATA, "video")
if not os.path.exists(os.path.join(GEN, "done_v1")):
    raise RuntimeError("build the 5-min assets first: cd .. && ./render.sh l (or import ../animation_hbh_v3.py)")

config.background_color = BLACK

# ------------------------------------------------------------------ style (same as random_phase_wave_splatting)
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
MATHCH = set("×⊙→≈∼−∈ℝ≪≤≥↓↔∑∏∬∫ℱ𝒫𝒢𝒩ℋ𝒪𝒰𝔼∗≠∞∂∘⊤●⇒∠✓ℒ")
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


SQ = sup("2")
LREC = "ℒ" + sub("recon")
LREG = "ℒ" + sub("reg")
USLM = I("u") + sub("SLM")
USRC = I("u") + sub("src")
ATGT = I("a") + sub("target")
EQ1 = (f"{I('u')}{sub(I('z'))} = {I('f')}( {USLM}, {I('z')} ),     {USLM} = {I('e')}{sup(I('i') + 'φ')} {USRC}")
EQ2 = (f"φ{sup('(' + I('k') + ')')} = φ{sup('(' + I('k') + '−1)')} − α (∂ℒ/∂φ){sup('T')} "
       f"ℒ( {I('s')} · |{I('f')}({USLM}, {I('z')})|, {ATGT} )")
EQ3 = (f"{LREG} = {I('L')}{sub('2')}( |{I('f')}({USLM}, {I('z')})|,  "
       f"√( ∑|{USRC}|{SQ} / ∑{ATGT}{SQ} ) · {ATGT} )")
EQ4 = f"ℒ = {LREC} + γ {LREG}"
BRIGHT = f"brightness = 1 / ( {I('s')}{sub('R')}{SQ} + {I('s')}{sub('G')}{SQ} + {I('s')}{sub('B')}{SQ} )"

# ------------------------------------------------------------------ toy geometry (must match precompute_toy.py)
H_SLM, W_SLM = 192, 320
GH, GW = 576, 704
RH, RW = 180, 300
CR0, CR1, CC0, CC1 = 64, 512, 64, 640      # displayed window of the image plane (rows, cols of the grid)
N1, G1, RN1 = 400, 1280, 372               # 1D toy: SLM pixels, grid, ROI pixels
K1 = G1 // 2 - 420                         # first grid row kept in the 1D x–z maps (840 rows)


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


def disp_rgb(I, norm):
    return (np.clip(I / norm, 0, 1) ** (1 / 2.2) * 255).astype(np.uint8)


def disp_log(I, top, decades=3.5):
    return cmap_rgb(np.clip((np.log10(np.maximum(I, 1e-12) / top) + decades) / decades, 0, 1), "inferno")


def crop2(a):
    return a[CR0:CR1, CC0:CC1]


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


def interp(frames, v):
    v = float(np.clip(v, 0, len(frames) - 1))
    i0 = int(np.floor(v))
    i1 = min(i0 + 1, len(frames) - 1)
    w = v - i0
    return ((1 - w) * frames[i0] + w * frames[i1]).astype(np.uint8)


def arrow(a, b, buff=0.1, color=WHITE, sw=3):
    return Arrow(a, b, buff=buff, color=color, stroke_width=sw,
                 max_tip_length_to_length_ratio=0.18, tip_length=0.16)


def phase_strip(phi, h, w=0.22):
    """1D SLM phase as a vertical twilight strip (rows = SLM pixels)."""
    c = matplotlib.colormaps["twilight"](np.mod(phi, 2 * np.pi) / (2 * np.pi))[:, :3]
    a = (np.repeat(c[:, None, :], 6, 1) * 255).astype(np.uint8)
    m = arr_img(a)
    m.stretch_to_fit_height(h)
    m.stretch_to_fit_width(w)
    return m


class Clip:
    """Video frames (data/video/<name>/%04d.jpg, 30 fps) as an ImageMobject that advances with scene time."""

    def __init__(self, name, h=None, w=None, start=0.0, loop=True, fps=30):
        self.paths = sorted(glob.glob(os.path.join(VID, name, "*.jpg")))
        self.fps, self.loop, self.t, self.i = fps, loop, start, -1
        self.mob = ImageMobject(self._frame(int(start * fps)))
        if h is not None:
            self.mob.set(height=h)
        if w is not None:
            self.mob.set(width=w)
        self.mob.add_updater(self._tick)

    def _frame(self, i):
        return rgba(np.asarray(Image.open(self.paths[i]).convert("RGB")))

    def _tick(self, m, dt):
        self.t += dt
        i = int(self.t * self.fps)
        i = i % len(self.paths) if self.loop else min(i, len(self.paths) - 1)
        if i != self.i:
            m.pixel_array = self._frame(i)
            self.i = i


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
    t = tx(s, size, color)
    plain = re.sub(r"<[^>]+>", "", s)
    desc = any(c in "gjpqy(),;[]{}|µφμβγηρχψξζ" for c in plain) or 'rise="-' in s
    dd, asc = 0.0027 * size, 0.0098 * size
    base = m.get_top()[1] + buff + dd if d is UP else m.get_bottom()[1] - buff - asc
    t.shift(UP * (base - (t.get_bottom()[1] + (dd if desc else 0))))
    t.set_x(m.get_center()[0])
    return t


def toy_note(s="toy simulation: phase-only SLM, 10.8 µm pixels, λ = 517.7 nm"):
    return tx(s, 16, DIM)


def plane_box(m, r0, r1, c0, c1, color=CYAN, dashed=True, sw=2.0):
    """Rectangle over an image-plane crop (grid rows r0:r1, cols c0:c1)."""
    w, h = m.width, m.height
    L, T = m.get_left()[0], m.get_top()[1]
    x0 = L + (c0 - CC0) / (CC1 - CC0) * w
    x1 = L + (c1 - CC0) / (CC1 - CC0) * w
    y0 = T - (r0 - CR0) / (CR1 - CR0) * h
    y1 = T - (r1 - CR0) / (CR1 - CR0) * h
    r = Rectangle(width=x1 - x0, height=y0 - y1, color=color, stroke_width=sw).move_to([(x0 + x1) / 2, (y0 + y1) / 2, 0])
    return DashedVMobject(r, num_dashes=60) if dashed else r


def roi_box(m, **kw):
    return plane_box(m, (GH - RH) // 2, (GH + RH) // 2, (GW - RW) // 2, (GW + RW) // 2, **kw)


def meter(x, y, h, frac_fn, color, label_s, w=0.55):
    """Vertical fill bar (0..1) with a live percentage."""
    frame = Rectangle(width=w, height=h, color=DIM, stroke_width=1.5).move_to([x, y, 0])
    fill = always_redraw(lambda: Rectangle(width=w - 0.08, height=max(1e-3, (h - 0.08) * frac_fn()), stroke_width=0,
                                           fill_color=color, fill_opacity=0.9
                                           ).align_to(frame.get_bottom() + UP * 0.04, DOWN).set_x(x))
    pct = always_redraw(lambda: tx(f"{frac_fn() * 100:.0f}%", 24, color).next_to(frame, UP, buff=0.12))
    lab = VGroup(*[tx(t_, 18, DIM) for t_ in label_s.split("|")]).arrange(DOWN, buff=0.06).next_to(frame, DOWN, buff=0.15)
    return VGroup(frame, fill, pct, lab)



# ================================================================== T1 Hook
class T1_Hook(Base):
    def construct(self):
        title = tx("High-Brightness Holographic Projection", 52, bold=True)
        title.scale_to_fit_width(min(title.width, 12.6)).move_to([0, 2.95, 0])
        clip = Clip("OL2023", h=3.3)
        clip.mob.move_to([0, -0.05, 0])
        fr = framed(clip.mob, DIM, 1.2)
        n2 = VGroup(tx("captured; the brighter half", 18, DIM), tx("uses the light-efficiency loss", 18, GREEN)
                    ).arrange(DOWN, buff=0.08).move_to([4.0, -0.05, 0])
        auth = tx("Brian Chao, Manu Gopakumar, Suyeon Choi, Gordon Wetzstein", 22, DIM).move_to([0, 2.3, 0])
        aff = tx("Stanford University  ·  Optics Letters 48(15), 4041 (2023)", 20, DIM).move_to([0, -2.45, 0])
        self.add(title, auth, clip.mob, fr, n2, aff)
        self.caption("A projector that puts all of its light where the picture is.")
        self.hold(0.6)
        clip.mob.clear_updaters()
        self.remove(title, auth, clip.mob, fr, n2, aff)
        self.uncaption()
        # conventional vs holographic (1D toy side views)
        d = np.load(os.path.join(GEN, "toy_1d.npz"))
        prof = d["prof"]
        amp = np.load(os.path.join(GEN, "xz_amp.npy"))
        xzh = np.load(os.path.join(GEN, "xz_noloss.npy"))
        hh, ww = 3.4, 4.8
        A = arr_img(amp)
        A.stretch_to_fit_height(hh).stretch_to_fit_width(ww).move_to([-3.4, 0.55, 0])
        Hm = arr_img(xzh)
        Hm.stretch_to_fit_height(hh).stretch_to_fit_width(ww).move_to([3.4, 0.55, 0])

        def bar(m, amp_panel):
            top = m.get_top()[1] - 220 / 840 * hh
            bot = m.get_top()[1] - 620 / 840 * hh
            if amp_panel:
                v = np.zeros(N1)
                r0 = (N1 - RN1) // 2
                v[r0:r0 + RN1] = prof
                b = arr_img((np.repeat(v[:, None], 6, 1) * 255).astype(np.uint8))
                b.stretch_to_fit_height(top - bot).stretch_to_fit_width(0.22)
            else:
                b = phase_strip(d["phi_noloss"], top - bot)
            return b.move_to([m.get_left()[0] - 0.16, (top + bot) / 2, 0])
        self.add(A, bar(A, True), framed(A, DIM, 1), label(A, "conventional: blocks light", 26, RED))
        self.add(tx(f"light that gets through: {prof.sum() / N1 * 100:.0f}%", 24, RED).next_to(A, DOWN, buff=0.3))
        self.caption("Conventional projectors form an image by blocking light. Dark pixels waste it.")
        self.add(Hm, bar(Hm, False), framed(Hm, DIM, 1), label(Hm, "hologram: steers light", 26, GREEN))
        self.add(tx(f"light that lands in the image: {float(d['eff_noloss']) * 100:.0f}%", 24, GREEN).next_to(Hm, DOWN, buff=0.3),
                 toy_note("toy simulation (1D side view): phase-only SLM, 10.8 µm pixels, λ = 517.7 nm, same laser power"
                          ).move_to([0, -2.5, 0]))
        self.caption("A phase-only hologram blocks nothing: it redirects light into the bright parts.")
        self.caption("But even a hologram lets some light stray outside the picture.")
        self.clear_all(run_time=0.5)


# ================================================================== T2 Method
class T2_Method(Base):
    def construct(self):
        eq = tx(EQ3, 30).move_to([0, 2.85, 0])
        self.add(eq)
        m = np.load(os.path.join(GEN, "toy_main.npz"))
        its = m["snap_iters"]
        S0 = np.load(os.path.join(GEN, "snap_noloss_log.npy"))
        S1 = np.load(os.path.join(GEN, "snap_loss_log.npy"))
        e0, e1 = m["cat_g_noloss_snapeff"], m["cat_g_loss_snapeff"]
        k = ValueTracker(0.0)
        h = 3.3
        a = arr_img(S0[0], h).move_to([-3.35, 0.3, 0])
        b = arr_img(S1[0], h).move_to([3.35, 0.3, 0])
        live(a, lambda: interp(S0, k.get_value()))
        live(b, lambda: interp(S1, k.get_value()))
        self.add(a, b, roi_box(a), roi_box(b), label(a, "usual image loss only", 24, RED),
                 label(b, "+ light-efficiency loss", 24, GREEN))
        fe = lambda arr: float(np.interp(k.get_value(), np.arange(len(arr)), arr))
        self.add(always_redraw(lambda: tx(f"light in the image: {fe(e0) * 100:.0f}%", 24, RED).next_to(a, DOWN, buff=0.2)),
                 always_redraw(lambda: tx(f"light in the image: {fe(e1) * 100:.0f}%", 24, GREEN).next_to(b, DOWN, buff=0.2)),
                 toy_note("toy simulation: SGD on a 192 × 320 phase SLM, green channel, log scale; cyan = image region"
                          ).move_to([0, -2.45, 0]))
        self.caption("Key idea: rescale the target so it carries all of the laser's power, and match it.")
        self.run(k, len(its) - 1, 5.5, lambda x: x ** 0.7)
        self.caption("The border goes dark: the light moves into the picture (toy: 64% to 94%).")
        for mm in (a, b):
            mm.clear_updaters()
        self.clear_all(keep=[eq], run_time=0.5)
        # camera-calibrated model on real hardware (paper Fig. S9, castle, with the loss)
        w = 4.0
        xs = [-4.2, 0, 4.2]
        for x, k_, n, cl in zip(xs, ["asm", "citl", "model"], ["ideal model", "camera in the loop", "learned model"],
                                [RED, AMBER, GREEN]):
            im = img(f"s9_{k_}_castle_loss.png", w=w).move_to([x, 0.35, 0])
            self.add(im, label(im, n, 24, cl))
        self.add(tx("captured, all with the loss (paper Fig. S9); red: brightness gain, blue: PSNR / LPIPS", 18, DIM
                    ).move_to([0, -1.55, 0]))
        self.caption("On real hardware, it needs a camera-calibrated model of the projector.")
        self.caption("With a learned model: +24% brighter. With the ideal model: +0%.")
        self.clear_all(run_time=0.5)


# ================================================================== T3 Payoff
class T3_Payoff(Base):
    def construct(self):
        clip = Clip("visualization1", w=7.6, loop=False)
        clip.mob.move_to([0, 0.95, 0])
        self.add(clip.mob, framed(clip.mob, DIM, 1),
                 tx("experimentally captured (paper Visualization 1)", 18, DIM).next_to(clip.mob, DOWN, buff=0.12))
        head = tx("around 30% brighter, image quality retained", 36, AMBER).move_to([0, -2.05, 0])
        sub_ = tx("test-set average: +33% captured, +66% in simulation (supplement)", 20, DIM).move_to([0, -2.65, 0])
        self.add(head, sub_)
        self.caption("Same laser, same projector: the image is brighter and the border is dark.")
        self.hold(0.8)
        clip.mob.clear_updaters()
        self.clear_all(run_time=0.5)
        t1 = tx("High-Brightness Holographic Projection", 48, bold=True).move_to([0, 1.7, 0])
        t2 = tx("Optics Letters 48(15), 4041 (2023)", 28).move_to([0, 0.85, 0])
        t4 = tx("Brian Chao, Manu Gopakumar, Suyeon Choi, Gordon Wetzstein", 24, DIM).move_to([0, -0.1, 0])
        t5 = tx("Stanford University  ·  bchao1.github.io", 22, DIM).move_to([0, -0.6, 0])
        self.add(t1, t2, t4, t5)
        self.wait(4.0)
