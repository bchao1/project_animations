"""High-Brightness Holographic Projection — teaser animation, v4.

Source: Chao, Gopakumar, Choi, Wetzstein, "High-brightness holographic projection", Optics Letters 48(15), 4041 (2023),
plus its Supplement 1 and Visualization 1. Paper-finder records: high-brightness-holography/.
Paper figure crops in data/ come from the supplement (scratch/crop_figs.py); video frames in data/video/<name>/.
Toy wave-optics results are computed by precompute_toy.py (torch autograd SGD, angular spectrum method) into
data/generated/toy_*.npz; build_assets() turns them into display arrays (delete data/generated/done_v1 to rebuild).

Render (see render.sh):
    manim -ql --media_dir media animation_hbh_v4.py <Scene>
Scenes, in order: S00_Title ... S09_Outlook.
"""
from manim import *
import numpy as np
from PIL import Image
import matplotlib
import fcntl
import glob
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
GEN = os.path.join(DATA, "generated")
VID = os.path.join(DATA, "video")
ASSET_FLAG = "done_v1"

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


def build_assets():
    os.makedirs(GEN, exist_ok=True)
    if os.path.exists(gen(ASSET_FLAG)):
        return
    need = ["toy_main.npz", "toy_1d.npz", "toy_sparsity.npz", "toy_citl.npz"]
    missing = [f for f in need if not os.path.exists(gen(f))]
    if missing:
        raise RuntimeError(f"run precompute_toy.py first (missing {missing})")
    with open(gen("lock"), "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        if os.path.exists(gen(ASSET_FLAG)):
            return
        m = np.load(gen("toy_main.npz"))
        # --- RGB results at the same laser power (common normalisation = with-loss ROI level)
        for name in ("cat", "coffee"):
            I0, I1 = m[f"{name}_noloss_I"], m[f"{name}_loss_I"]
            roi = I1[(GH - RH) // 2:(GH + RH) // 2, (GW - RW) // 2:(GW + RW) // 2]
            nrm = np.percentile(roi, 99.0, axis=(0, 1))
            np.save(gen(f"{name}_noloss_rgb.npy"), crop2(disp_rgb(I0, nrm)))
            np.save(gen(f"{name}_loss_rgb.npy"), crop2(disp_rgb(I1, nrm)))
            tgt = np.zeros((GH, GW, 3))
            tgt[(GH - RH) // 2:(GH + RH) // 2, (GW - RW) // 2:(GW + RW) // 2] = m[f"{name}_target"]
            np.save(gen(f"{name}_target_rgb.npy"), crop2(disp_rgb(tgt, 1.0)))
        # --- green-channel SGD snapshots on a log scale (top = with-loss ROI level)
        top = np.percentile(m["cat_loss_I"][..., 1], 99.9)
        for tag in ("noloss", "loss"):
            S = m[f"cat_g_{tag}_snapI"].astype(np.float32)
            np.save(gen(f"snap_{tag}_log.npy"), np.stack([crop2(disp_log(s, top)) for s in S]))
            np.save(gen(f"snap_{tag}_lin.npy"), np.stack([crop2(cmap_rgb((s / top) ** 0.5, "inferno")) for s in S]))
        # --- 1D x–z maps (rows = x, cols = z), common normalisation
        d = np.load(gen("toy_1d.npz"))
        nrm1 = np.percentile(d["xz_noloss"], 99.7)
        for tag in ("noloss", "loss", "short"):
            np.save(gen(f"xz_{tag}.npy"), cmap_rgb((d[f"xz_{tag}"] / nrm1) ** 0.5))
        np.save(gen("xz_snap_noloss.npy"), np.stack([cmap_rgb((x.astype(np.float32) / nrm1) ** 0.5)
                                                    for x in d["snap_xz_noloss"]]))
        # amplitude display: light passes the mask straight (intensity = transmission), same colour scale
        prof = d["prof"]
        amp = np.zeros((840, 360))
        r0 = (G1 - RN1) // 2 - K1
        amp[r0:r0 + RN1, :] = prof[:, None]
        np.save(gen("xz_amp.npy"), cmap_rgb((amp / nrm1) ** 0.5))
        cone = d["cone"].astype(np.float64)
        cone = cone / (cone.max(0, keepdims=True) + 1e-12)
        np.save(gen("xz_cone.npy"), cmap_rgb(cone ** 0.6))
        # --- CITL toy
        c = np.load(gen("toy_citl.npz"))
        big = np.zeros((GH, GW))
        big[(GH - RH) // 2:(GH + RH) // 2, (GW - RW) // 2:(GW + RW) // 2] = c["target"]
        ref = np.percentile(c["model_noloss"].astype(np.float32)[(GH - RH) // 2:(GH + RH) // 2,
                                                                 (GW - RW) // 2:(GW + RW) // 2], 99.0)
        for k in ("asm_noloss", "citl_noloss", "model_noloss"):
            I_ = c[k].astype(np.float32)
            np.save(gen(f"citl_{k}.npy"), crop2(disp_rgb(np.stack([I_] * 3, -1), ref)))
        ideal = m["cat_noloss_I"][..., 1]
        np.save(gen("citl_ideal.npy"), crop2(disp_rgb(np.stack([ideal] * 3, -1),
                                                      np.percentile(ideal[(GH - RH) // 2:(GH + RH) // 2,
                                                                          (GW - RW) // 2:(GW + RW) // 2], 99.0))))
        np.save(gen("citl_target.npy"), crop2(disp_rgb(np.stack([big] * 3, -1), 1.0)))
        ill = c["ill_true"]
        np.save(gen("citl_ill.npy"), (np.clip(ill, 0, 1) * 255).astype(np.uint8))
        open(gen(ASSET_FLAG), "w").write("ok")


build_assets()


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


# ================================================================== S00 Title
class S00_Title(Base):
    def construct(self):
        title = tx("High-Brightness Holographic Projection", 56, bold=True)
        title.scale_to_fit_width(min(title.width, 12.6)).move_to([0, 2.95, 0])
        subt = tx("a light-efficiency loss, AI-driven CGH and camera-in-the-loop calibration", 26, DIM
                  ).move_to([0, 2.3, 0])
        clip = Clip("OL2023", h=3.5)
        clip.mob.move_to([0, -0.1, 0])
        fr = framed(clip.mob, DIM, 1.2)
        n1 = VGroup(tx("captured on the", 18, DIM), tx("prototype", 18, DIM)).arrange(DOWN, buff=0.08
                                                                                   ).move_to([-3.2, 0.4, 0])
        n2 = VGroup(tx("the brighter half:", 18, DIM), tx("with the light-", 18, GREEN),
                    tx("efficiency loss", 18, GREEN)).arrange(DOWN, buff=0.08).move_to([3.3, 0.4, 0])
        auth = tx("Brian Chao, Manu Gopakumar, Suyeon Choi, Gordon Wetzstein", 22, DIM).move_to([0, -2.45, 0])
        aff = tx("Stanford University  ·  Optics Letters 48(15), 4041 (2023)", 22, DIM).move_to([0, -2.9, 0])
        self.add(title, subt, clip.mob, fr, n1, n2, auth, aff)
        self.wait(7.4)


# ================================================================== S01 Waste
class S01_Waste(Base):
    def construct(self):
        self.title_card("Part 1", "Projectors throw light away")
        self.header("1 · Projectors throw light away")
        d = np.load(gen("toy_1d.npz"))
        prof = d["prof"]
        amp = np.load(gen("xz_amp.npy"))
        xzh = np.load(gen("xz_noloss.npy"))
        hh, ww = 3.3, 4.6
        A = arr_img(amp)
        A.stretch_to_fit_height(hh).stretch_to_fit_width(ww).move_to([-3.35, 0.55, 0])
        Hm = arr_img(xzh)
        Hm.stretch_to_fit_height(hh).stretch_to_fit_width(ww).move_to([3.35, 0.55, 0])
        # SLM / panel positions on the x–z maps: rows 220..620 of 840
        def slm_bar(m, amp_panel):
            top = m.get_top()[1] - 220 / 840 * hh
            bot = m.get_top()[1] - 620 / 840 * hh
            x = m.get_left()[0] - 0.16
            if amp_panel:
                v = np.zeros(N1)
                r0 = (N1 - RN1) // 2
                v[r0:r0 + RN1] = prof
                a = (np.repeat(v[:, None], 6, 1) * 255).astype(np.uint8)
                bar = arr_img(a)
                bar.stretch_to_fit_height(top - bot).stretch_to_fit_width(0.22)
            else:
                bar = phase_strip(d["phi_noloss"], top - bot)
            bar.move_to([x, (top + bot) / 2, 0])
            return bar
        b1, b2 = slm_bar(A, True), slm_bar(Hm, False)
        self.play(FadeIn(A), FadeIn(b1), run_time=0.6)
        self.add(label(A, "conventional: amplitude panel", 24, RED), framed(A, DIM, 1))
        self.add(tx("panel", 18, DIM).next_to(b1, LEFT, buff=0.1),
                 tx(f"propagation {I('z')} →", 18, DIM).next_to(A, DOWN, buff=0.12))
        conv = float(prof.sum() / N1)
        self.caption("A conventional projector forms an image by blocking light: dark pixels absorb it.")
        r1 = tx(f"light that gets through: {conv * 100:.0f}%", 22, RED).next_to(A, DOWN, buff=0.55)
        self.add(r1)
        self.hold()
        self.play(FadeIn(Hm), FadeIn(b2), run_time=0.6)
        self.add(label(Hm, "holographic: phase-only SLM", 24, GREEN), framed(Hm, DIM, 1))
        self.add(tx("SLM", 18, DIM).next_to(b2, LEFT, buff=0.1),
                 tx(f"propagation {I('z')} →", 18, DIM).next_to(Hm, DOWN, buff=0.12))
        r2 = tx(f"light that lands in the image: {float(d['eff_noloss']) * 100:.0f}%", 22, GREEN
                ).next_to(Hm, DOWN, buff=0.55)
        self.add(r2)
        self.add(toy_note("toy simulation (1D side view): three bright bars, same laser power and colour scale in both panels"
                          ).move_to([0, -2.55, 0]))
        self.caption("A phase-only hologram blocks nothing. It bends light, so dark regions donate theirs to bright ones.")
        self.clear_all()

        # ---- sparsity sweep (toy)
        sp = np.load(gen("toy_sparsity.npz"))
        rows = sp["rows"]
        lv = rows[:, 0]
        conv_b = rows[:, 8] / rows[0, 8]
        holo_b = rows[:, 2] / rows[0, 8]
        tg = sp["targets"]
        ax = Axes(x_range=[0, 1, 0.25], y_range=[0, 24, 4], x_length=6.0, y_length=4.0, tips=False,
                  axis_config={"color": DIM, "stroke_width": 2, "include_numbers": False}).move_to([1.7, 0.45, 0])
        xt = VGroup(*[tx(f"{v:.2g}", 16, DIM).next_to(ax.c2p(v, 0), DOWN, buff=0.1) for v in (0, 0.25, 0.5, 0.75, 1.0)])
        yt = VGroup(*[tx(f"{v}×", 16, DIM).next_to(ax.c2p(0, v), LEFT, buff=0.1) for v in (1, 8, 16, 24)])
        xl = tx("sparsity (fraction of black pixels)", 20, DIM).next_to(ax, DOWN, buff=0.45)
        yl = tx("brightness at equal laser power", 20, DIM).next_to(ax, UP, buff=0.15)
        s = ValueTracker(0.0)
        fx = lambda: float(np.interp(s.get_value(), np.arange(len(lv)), lv))
        th = arr_img((np.clip(tg[0], 0, 1) ** (1 / 2.2) * 255).astype(np.uint8), h=2.2).move_to([-4.3, 0.9, 0])
        live(th, lambda: (np.clip(tg[int(round(s.get_value()))], 0, 1) ** (1 / 2.2) * 255).astype(np.uint8))
        thl = always_redraw(lambda: tx(f"target, {fx() * 100:.0f}% black", 22).next_to(th, UP, buff=0.15))
        k_of = lambda: int(round(s.get_value()))

        def curve(vals, c):
            return always_redraw(lambda: VGroup(
                *[Dot(ax.c2p(lv[k], vals[k]), radius=0.06, color=c) for k in range(k_of() + 1)],
                *[Line(ax.c2p(lv[k], vals[k]), ax.c2p(lv[k + 1], vals[k + 1]), color=c, stroke_width=3)
                  for k in range(k_of())]))
        cc, hc = curve(conv_b, RED), curve(holo_b, GREEN)
        self.play(Create(ax), FadeIn(th), run_time=0.7)
        self.add(xt, yt, xl, yl, thl, cc, hc)
        lc = tx("conventional", 20, RED).next_to(ax.c2p(lv[-1], conv_b[-1]), RIGHT, buff=0.15)
        lh = tx("hologram", 20, GREEN).next_to(ax.c2p(lv[-1], holo_b[-1]), RIGHT, buff=0.15)
        note = toy_note("toy simulation: 192 × 320 SLM, λ = 517.7 nm, SGD without any light-efficiency term"
                        ).move_to([0, -2.75, 0])
        self.add(note)
        self.caption("Now black out more and more of the image, at the same laser power.")
        self.run(s, len(lv) - 1, 5.0, smooth)
        self.add(lc, lh)
        self.caption("Conventional: equally dim. Hologram: ever brighter. (Paper, Fig. 3: over 70% vs nearly 0% efficiency.)")
        th.clear_updaters()
        self.clear_all()

        # ---- paper Fig. S6
        w = 3.1
        xs = [-4.8, -1.6, 1.6, 4.8]
        names = ["conventional", f"hologram, 0.17 {I('d')}*", f"hologram, 0.33 {I('d')}*", f"hologram, 0.67 {I('d')}*"]
        for x, k, n in zip(xs, ["conv", "h017", "h033", "h067"], names):
            a = img(f"s6_{k}_monarch.png", w=w).move_to([x, 1.0, 0])
            b = img(f"s6_{k}_macaw.png", w=w).move_to([x, -0.85, 0])
            self.add(a, b, label(a, n, 22, RED if k == "conv" else GREEN))
        self.add(tx("simulated, same power budget; red labels = brightness gain over conventional (paper Fig. S6)", 18,
                    DIM).move_to([0, -2.25, 0]))
        self.caption("Paper simulation, same power budget: the holographic images come out +1042% to +1225% brighter.")
        self.clear_all()


# ================================================================== S02 Holography
class S02_Holography(Base):
    def construct(self):
        self.title_card("Part 2", "Steering light with phase")
        self.header("2 · Steering light with phase")
        e1 = tx(EQ1, 30).move_to([0, 2.75, 0])
        self.add(e1)
        d = np.load(gen("toy_1d.npz"))
        snaps = np.load(gen("xz_snap_noloss.npy"))
        sphi = d["snap_phi_noloss"]
        its = d["snap_iters"]
        prof = d["prof"]
        hh, ww = 3.6, 7.6
        k = ValueTracker(0.0)
        xzm = arr_img(snaps[0])
        xzm.stretch_to_fit_height(hh).stretch_to_fit_width(ww).move_to([-0.6, 0.35, 0])
        live(xzm, lambda: interp(snaps, k.get_value()))
        top = xzm.get_top()[1]
        y_slm0, y_slm1 = top - 220 / 840 * hh, top - 620 / 840 * hh
        strip = phase_strip(sphi[0], y_slm0 - y_slm1).move_to([xzm.get_left()[0] - 0.2, (y_slm0 + y_slm1) / 2, 0])
        strip.add_updater(lambda m: setattr(m, "pixel_array", rgba((np.repeat(
            matplotlib.colormaps["twilight"](np.mod(sphi[int(round(k.get_value()))], 2 * np.pi) / (2 * np.pi))[:, None, :3],
            6, 1) * 255).astype(np.uint8))))
        # target / achieved profile at the image plane (right)
        r0 = (G1 - RN1) // 2 - K1
        gx = xzm.get_right()[0] + 0.25
        ys = lambda r: top - r / 840 * hh

        def prof_curve(vals, color, width=0.9):
            pts = [[gx + width * float(v), ys(r0 + i), 0] for i, v in enumerate(vals)]
            return VMobject(color=color, stroke_width=2.5).set_points_as_corners(pts)
        tgt_c = prof_curve(prof, DIM)
        I_snap = lambda: interp(snaps, k.get_value())
        xzraw = d["snap_xz_noloss"]

        def ach():
            i = int(round(k.get_value()))
            col_ = xzraw[i][:, -1].astype(np.float32)[r0:r0 + RN1]
            return prof_curve(col_ / (np.percentile(col_, 99) + 1e-9) * 0.95, AMBER)
        ach_c = always_redraw(ach)
        self.play(FadeIn(xzm), FadeIn(strip), run_time=0.6)
        self.add(tx("phase SLM", 20, DIM).next_to(strip, LEFT, buff=0.12).rotate(PI / 2),
                 tx(f"propagation {I('z')} →", 20, DIM).next_to(xzm, DOWN, buff=0.12),
                 tgt_c, ach_c, tx("target", 18, DIM).move_to([gx + 0.45, top + 0.2, 0]),
                 tx("image plane", 18, AMBER).move_to([gx + 0.45, xzm.get_bottom()[1] - 0.2, 0]))
        cnt = always_redraw(lambda: tx(f"SGD iteration {int(its[int(round(k.get_value()))])}", 24).move_to([-4.9, -1.85, 0]))
        self.add(cnt, toy_note("toy simulation (1D side view): 400 SLM pixels, 10.8 µm, λ = 517.7 nm, image 0.64 d* away"
                               ).move_to([0, -2.5, 0]))
        self.caption("A phase SLM only delays the light. Where the delayed waves meet, they interfere.")
        self.caption("Gradient descent tunes every delay until the interference pattern is the target.")
        self.run(k, len(its) - 1, 6.5, lambda x: x ** 0.8)
        self.caption("Light that would have hit dark regions is redirected into the bright bars.")
        for m in (xzm, strip):
            m.clear_updaters()
        self.clear_all(keep=[e1])
        e2 = tx(EQ2, 28).move_to([0, 1.6, 0])
        self.add(e2)
        self.caption("The paper's solver: stochastic gradient descent on the phase φ (2000 iterations, learning rate 0.02).")
        sline = tx(f"{I('s')}: the scale that best matches |{I('f')}| to the target (closed form, every step)", 26
                   ).move_to([0, 0.35, 0])
        bl = tx(BRIGHT, 30, AMBER).move_to([0, -0.6, 0])
        self.add(sline, bl, tx("(supplement, Sec. 4)", 18, DIM).next_to(bl, DOWN, buff=0.15))
        self.caption(f"{I('s')} is how much laser amplitude the image needs. A smaller {I('s')} means a brighter image.")
        self.clear_all()


# ================================================================== S03 Stray light
class S03_Stray(Base):
    def construct(self):
        self.title_card("Part 3", "Where does the rest of the light go?")
        self.header("3 · Where does the rest of the light go?")
        m = np.load(gen("toy_main.npz"))
        snaps = np.load(gen("snap_noloss_log.npy"))
        effs = m["cat_g_noloss_snapeff"]
        its = m["snap_iters"]
        k = ValueTracker(0.0)
        h = 4.3
        v = arr_img(snaps[0], h).move_to([-2.2, 0.35, 0])
        live(v, lambda: interp(snaps, k.get_value()))
        box = roi_box(v)
        slm = plane_box(v, (GH - H_SLM) // 2, (GH + H_SLM) // 2, (GW - W_SLM) // 2, (GW + W_SLM) // 2, color=DIM, sw=1.2)
        self.play(FadeIn(v), run_time=0.6)
        self.add(box, slm, label(v, "image plane, log scale", 22, DIM))
        self.add(tx("image region", 18, CYAN).next_to(box, UP, buff=0.06).align_to(box, LEFT),
                 tx("SLM footprint", 16, DIM).next_to(slm, DOWN, buff=0.06).align_to(slm, RIGHT))
        f = lambda: float(np.interp(k.get_value(), np.arange(len(effs)), effs))
        mt = meter(3.5, 0.45, 3.0, f, GREEN, "light inside|the image")
        mo = meter(5.4, 0.45, 3.0, lambda: 1 - f(), RED, "light outside|(wasted)")
        self.add(mt, mo)
        cnt = always_redraw(lambda: tx(f"SGD iteration {int(its[int(round(k.get_value()))])}", 22
                                       ).move_to([4.45, 2.65, 0]))
        self.add(cnt, toy_note("toy simulation: 192 × 320 phase SLM, 10.8 µm pixels, λ = 517.7 nm (green), "
                               "image at 0.64 d*, 3.5 decades shown").move_to([0, -2.35, 0]))
        self.caption(f"Optimize a 2D hologram with the usual image loss {LREC} only.")
        self.run(k, len(its) - 1, 6.5, lambda x: x ** 0.7)
        self.caption(f"The image looks right. But {LREC} only looks inside the image region...")
        self.caption("...so a third of the laser light is scattered into the border and wasted.")
        self.caption("Wasted light is lost brightness. The fix is one extra loss term.")
        v.clear_updaters()
        self.clear_all()


# ================================================================== S04 Loss
class S04_Loss(Base):
    def construct(self):
        self.title_card("Part 4", "The light-efficiency loss")
        self.header("4 · The light-efficiency loss")
        e3 = tx(EQ3, 30).move_to([0, 2.6, 0])
        e4 = tx(EQ4, 30).move_to([0, 1.8, 0])
        self.add(e3)
        self.caption("Idea: rescale the target in advance so that it carries all of the laser's power.")
        # power bars
        y = 0.3
        lp = Rectangle(width=6.0, height=0.36, stroke_width=0, fill_color=AMBER, fill_opacity=0.9).move_to([0.6, y + 0.4, 0])
        tp = Rectangle(width=6.0, height=0.36, stroke_width=0, fill_color=GREEN, fill_opacity=0.9).move_to([0.6, y - 0.4, 0])
        self.play(GrowFromEdge(lp, LEFT), run_time=0.6)
        self.add(tx(f"laser power  ∑|{USRC}|{SQ}", 22, AMBER).next_to(lp, LEFT, buff=0.3))
        self.play(GrowFromEdge(tp, LEFT), run_time=0.6)
        self.add(tx("power of the scaled target", 22, GREEN).next_to(tp, LEFT, buff=0.3))
        self.add(tx("(both equal, by construction)", 18, DIM).next_to(tp, DOWN, buff=0.2))
        self.add(e4)
        self.caption("Matching it pixel by pixel leaves no light for anywhere else. Its weight: γ.")
        self.clear_all()

        # ---- side-by-side optimisation (green channel, log scale)
        m = np.load(gen("toy_main.npz"))
        its = m["snap_iters"]
        S0, S1 = np.load(gen("snap_noloss_log.npy")), np.load(gen("snap_loss_log.npy"))
        e0, e1 = m["cat_g_noloss_snapeff"], m["cat_g_loss_snapeff"]
        k = ValueTracker(0.0)
        h = 3.5
        a = arr_img(S0[0], h).move_to([-3.35, 0.45, 0])
        b = arr_img(S1[0], h).move_to([3.35, 0.45, 0])
        live(a, lambda: interp(S0, k.get_value()))
        live(b, lambda: interp(S1, k.get_value()))
        self.play(FadeIn(a), FadeIn(b), run_time=0.6)
        self.add(roi_box(a), roi_box(b), label(a, f"{LREC} only", 26, RED), label(b, f"{LREC} + γ {LREG}", 26, GREEN))
        fe = lambda arr: (lambda: float(np.interp(k.get_value(), np.arange(len(arr)), arr)))
        ra = always_redraw(lambda: tx(f"light in the image: {fe(e0)() * 100:.0f}%", 24, RED).next_to(a, DOWN, buff=0.2))
        rb = always_redraw(lambda: tx(f"light in the image: {fe(e1)() * 100:.0f}%", 24, GREEN).next_to(b, DOWN, buff=0.2))
        cnt = always_redraw(lambda: tx(f"iteration {int(its[int(round(k.get_value()))])}", 22, DIM).move_to([0, 0.45, 0]))
        self.add(ra, rb, cnt, toy_note(f"toy simulation: same target, same SGD, green channel, log scale; γ = 0.1").move_to(
            [0, -2.45, 0]))
        self.caption("Toy: the same optimization, without and with the new term.")
        self.run(k, len(its) - 1, 6.5, lambda x: x ** 0.7)
        self.caption("With the loss, the border goes dark: the light moved into the picture.")
        for mm in (a, b):
            mm.clear_updaters()
        self.clear_all()

        # ---- RGB result at equal laser power
        h = 3.5
        a = arr_img(np.load(gen("cat_noloss_rgb.npy")), h).move_to([-3.35, 0.5, 0])
        b = arr_img(np.load(gen("cat_loss_rgb.npy")), h).move_to([3.35, 0.5, 0])
        self.play(FadeIn(a), FadeIn(b), run_time=0.6)
        self.add(label(a, "without the loss", 26, RED), label(b, "with the loss", 26, GREEN))
        gain = float(m["cat_gain"])
        ps0, ps1 = m["cat_noloss_psnr"].mean(), m["cat_loss_psnr"].mean()
        self.add(tx(f"toy: +{gain * 100:.0f}% brightness (1 / ∑{I('s')}{SQ})", 30, AMBER).move_to([0, -1.6, 0]),
                 toy_note(f"toy simulation, RGB (636.4 / 517.7 / 440.8 nm), same laser power and display scale; "
                          f"PSNR {ps0:.0f} → {ps1:.0f} dB").move_to([0, -2.3, 0]))
        self.caption("Same laser, same picture: the image optimized with the loss is clearly brighter.")
        self.caption("The paper's simulations: +66% on average over the test set.")
        self.clear_all()


# ================================================================== S05 Distance
class S05_Distance(Base):
    def construct(self):
        self.title_card("Part 5", "Why the image must be far from the SLM")
        self.header("5 · Why the image must be far from the SLM")
        d = np.load(gen("toy_1d.npz"))
        cone = np.load(gen("xz_cone.npy"))
        hh, ww = 3.4, 6.2
        cm = arr_img(cone)
        cm.stretch_to_fit_height(hh).stretch_to_fit_width(ww).move_to([-3.2, 0.55, 0])
        self.play(FadeIn(cm), run_time=0.6)
        self.add(framed(cm, DIM, 1), tx(f"distance from the SLM →  (0 to 1.1 {I('d')}*)", 18, DIM).next_to(cm, DOWN, buff=0.12),
                 label(cm, "light from a single SLM pixel", 22))
        # geometry overlay: apex at left-centre
        apex = np.array([cm.get_left()[0], cm.get_center()[1], 0])
        th = np.arcsin(517.7e-9 / (2 * 10.8e-6))
        # cone half-angle in display units: x spans 1.1 d*, y spans 840 px of the grid
        ds = float(d["dstar"])
        L1 = ds * np.tan(th) / (10.8e-6 * 840) * hh              # half-width at d* in display units
        xd = apex[0] + ww / 1.1
        up = Line(apex, [xd, apex[1] + L1, 0], color=CYAN, stroke_width=2.5)
        dn = Line(apex, [xd, apex[1] - L1, 0], color=CYAN, stroke_width=2.5)
        ax_ = DashedLine(apex, [xd, apex[1], 0], color=DIM, stroke_width=1.5)
        vL = Line([xd, apex[1], 0], [xd, apex[1] + L1, 0], color=AMBER, stroke_width=3)
        self.play(Create(up), Create(dn), Create(ax_), run_time=0.7)
        arc = Arc(radius=1.1, start_angle=0, angle=np.arctan(L1 / (xd - apex[0])), arc_center=apex, color=CYAN)
        self.add(arc, tx("θ", 22, CYAN).move_to(apex + np.array([1.35, 0.14, 0])), vL,
                 tx(f"{I('L')} = {I('d')} tan θ", 20, AMBER).move_to([cm.get_right()[0] + 0.75, apex[1] + L1 / 2, 0]))
        eqs = VGroup(tx(f"θ = sin{sup('−1')}(λ / 2{I('p')})", 28),
                     tx(f"{I('d')}* = {I('R')} / tan θ", 28),
                     tx(f"{I('R')}: half-diagonal of the SLM", 18, DIM)).arrange(DOWN, buff=0.22).move_to([3.9, 1.35, 0])
        self.add(eqs)
        self.caption(f"Each pixel's light spreads into a cone of half-angle θ = sin{sup('−1')}(λ/2{I('p')}): it only reaches so far sideways.")
        proto = VGroup(tx(f"prototype: {I('p')} = 10.8 µm, λ = 517 nm", 22), tx(f"{I('R')} = 7.45 mm  ⇒  {I('d')}* = 31.11 cm", 22),
                       tx(f"image at {I('z')} ≈ 20 cm ≈ 0.64 {I('d')}*", 22, AMBER)).arrange(DOWN, buff=0.16, aligned_edge=LEFT
                                                                                           ).move_to([3.9, -0.6, 0])
        self.add(proto)
        self.caption(f"At {I('d')}*, one pixel's cone covers the whole image: it can send light anywhere in it.")
        self.add(toy_note("toy simulation: one lit pixel, 1D, λ = 517.7 nm, 10.8 µm pixels (brightness normalized per distance)"
                          ).move_to([0, -2.45, 0]))
        self.clear_all()

        # ---- short vs long (1D toy)
        hh, ww = 1.9, 6.4
        s_ = arr_img(np.load(gen("xz_short.npy")))
        s_.stretch_to_fit_height(hh).stretch_to_fit_width(ww * 0.12 / 0.64 * 2.2).move_to([-0.9, 1.25, 0])
        l_ = arr_img(np.load(gen("xz_noloss.npy")))
        l_.stretch_to_fit_height(hh).stretch_to_fit_width(ww).move_to([-0.9, -0.95, 0])
        s_.align_to(l_, LEFT)
        self.play(FadeIn(s_), FadeIn(l_), run_time=0.6)
        prof = d["prof"]
        r0g = (G1 - RN1) // 2
        for mm, key in ((s_, "I_short"), (l_, "I_noloss")):
            Ig = d[key]
            ampg = np.sqrt(Ig[r0g:r0g + RN1])
            sc = (ampg * np.sqrt(prof)).sum() / (ampg ** 2).sum()
            rec = (sc * ampg) ** 2
            gx = l_.get_right()[0] + 0.2
            yy = lambda r, mm=mm: mm.get_top()[1] - (r0g - K1 + r) / 840 * hh
            tc = VMobject(color=DIM, stroke_width=2).set_points_as_corners([[gx + 0.8 * prof[i], yy(i), 0] for i in range(RN1)])
            ac = VMobject(color=AMBER, stroke_width=2).set_points_as_corners(
                [[gx + 0.8 * min(rec[i], 1.25), yy(i), 0] for i in range(RN1)])
            self.add(tc, ac)
        self.add(tx("target", 16, DIM).move_to([l_.get_right()[0] + 0.6, s_.get_top()[1] + 0.15, 0]),
                 tx("image", 16, AMBER).move_to([l_.get_right()[0] + 0.6, l_.get_bottom()[1] - 0.15, 0]))
        self.add(tx(f"{I('z')} = 0.12 {I('d')}*", 24, RED).next_to(s_, LEFT, buff=0.35),
                 tx(f"{I('z')} = 0.64 {I('d')}*", 24, GREEN).next_to(l_, LEFT, buff=0.35),
                 tx(f"toy PSNR {float(d['psnr_short']):.1f} dB", 22, RED).move_to([l_.get_right()[0] + 2.5, s_.get_y(), 0]),
                 tx(f"toy PSNR {float(d['psnr_noloss']):.1f} dB", 22, GREEN).move_to([l_.get_right()[0] + 2.5, l_.get_y(), 0]),
                 toy_note("toy simulation (1D side view), same three-bar target; the short view is stretched 2.2× in z "
                          "to be visible").move_to([0, -2.45, 0]))
        self.caption("Too close, and light cannot travel sideways to where it is needed: the bars come out wrong.")
        self.clear_all()

        # ---- paper Fig. S4 (digitised)
        xs = [0.032, 0.161, 0.322, 0.643, 0.965]
        wo = [45, 60, 66, 67, 63]
        wi = [66, 80, 89, 93, 91]
        ax = Axes(x_range=[0, 1, 0.2], y_range=[40, 100, 10], x_length=7.2, y_length=4.0, tips=False,
                  axis_config={"color": DIM, "stroke_width": 2, "include_numbers": False}).move_to([-0.2, 0.3, 0])
        xt = VGroup(*[tx(f"{v:.1f}", 16, DIM).next_to(ax.c2p(v, 40), DOWN, buff=0.1) for v in (0, 0.2, 0.4, 0.6, 0.8, 1.0)])
        yt = VGroup(*[tx(f"{v}%", 16, DIM).next_to(ax.c2p(0, v), LEFT, buff=0.1) for v in (40, 60, 80, 100)])
        xl = tx(f"propagation distance ({I('d')}*)", 20, DIM).next_to(ax, DOWN, buff=0.45)
        yl = tx("light efficiency", 20, DIM).rotate(PI / 2).next_to(ax, LEFT, buff=0.6)
        c0 = VMobject(color=RED, stroke_width=4).set_points_as_corners([ax.c2p(x, y) for x, y in zip(xs, wo)])
        c1 = VMobject(color=GREEN, stroke_width=4).set_points_as_corners([ax.c2p(x, y) for x, y in zip(xs, wi)])
        d0 = VGroup(*[Dot(ax.c2p(x, y), radius=0.06, color=RED) for x, y in zip(xs, wo)])
        d1 = VGroup(*[Dot(ax.c2p(x, y), radius=0.06, color=GREEN) for x, y in zip(xs, wi)])
        self.play(Create(ax), run_time=0.6)
        self.add(xt, yt, xl, yl, tx("paper Fig. S4 (simulation, whole test set)", 18, DIM).move_to([0, -2.8, 0]))
        self.play(Create(c0), FadeIn(d0), run_time=1.0)
        self.add(tx("without the loss", 20, RED).next_to(ax.c2p(0.8, 63), DOWN, buff=0.25))
        self.play(Create(c1), FadeIn(d1), run_time=1.0)
        self.add(tx("with the loss", 20, GREEN).next_to(ax.c2p(0.8, 92), UP, buff=0.2))
        pk = DashedLine(ax.c2p(0.643, 40), ax.c2p(0.643, 100), color=AMBER, stroke_width=2)
        self.add(pk, tx(f"0.64 {I('d')}* ≈ 20 cm", 20, AMBER).next_to(ax.c2p(0.643, 44), RIGHT, buff=0.12))
        self.caption("Paper: efficiency peaks near 0.64 d*, 67% without the loss and 93% with it.")
        self.clear_all()


# ================================================================== S06 Reality gap
class S06_Reality(Base):
    def construct(self):
        self.title_card("Part 6", "Long distances break ideal models")
        self.header("6 · Long distances break ideal models")
        w = 6.0
        a = img("s7_long.png", w=w).move_to([-3.2, 0.55, 0])
        b = img("s7_short.png", w=w).move_to([3.2, 0.55, 0])
        self.play(FadeIn(a), FadeIn(b), run_time=0.6)
        self.add(label(a, f"long distance, 0.64 {I('d')}*", 24, RED), label(b, f"short distance, 0.31 {I('d')}*", 24),
                 tx("experimentally captured, camera-in-the-loop SGD (paper Fig. S7); insets: zoom-ins", 18, DIM
                    ).move_to([0, -1.75, 0]))
        self.caption("On real hardware, the long throw brings fringes and colour blur: real optics differ from the model.")
        self.clear_all()
        # toy: ideal simulation vs toy real display
        h = 2.3
        xs = [-4.8, -1.6, 1.6, 4.8]
        tg = arr_img(np.load(gen("citl_target.npy")), h).move_to([xs[0], 0.7, 0])
        il = arr_img(np.load(gen("citl_ill.npy")), 1.7).move_to([xs[1], 0.7, 0])
        idl = arr_img(np.load(gen("citl_ideal.npy")), h).move_to([xs[2], 0.7, 0])
        cap = arr_img(np.load(gen("citl_asm_noloss.npy")), h).move_to([xs[3], 0.7, 0])
        c = np.load(gen("toy_citl.npz"))
        self.play(FadeIn(tg), run_time=0.4)
        self.add(label(tg, "target", 22))
        self.add(il, framed(il, DIM, 1), label(il, "toy laser beam", 22),
                 tx("+ phase errors", 18, DIM).next_to(il, DOWN, buff=0.15),
                 tx("+ 0.5% distance error", 18, DIM).next_to(il, DOWN, buff=0.48))
        self.caption("Toy: a 'real' projector with uneven illumination, phase errors and a slightly wrong distance.")
        self.add(idl, label(idl, "ideal-model prediction", 22, GREEN))
        self.caption("Optimized with the ideal model (ASM), the simulation looks perfect...")
        self.add(cap, label(cap, "toy 'camera' capture", 22, RED),
                 tx(f"toy PSNR {c['asm_noloss_m'][2]:.1f} dB", 22, RED).next_to(cap, DOWN, buff=0.2))
        self.add(toy_note("toy simulation, green channel; 'camera' = the toy real projector").move_to([0, -2.45, 0]))
        self.caption("...but the projector shows fringes and a dim border: the model was wrong.")
        self.clear_all()


# ================================================================== S07 Camera in the loop
class S07_Camera(Base):
    def construct(self):
        self.title_card("Part 7", "Learning the real projector")
        self.header("7 · Learning the real projector")
        # loop diagram
        pos = {"slm": [-4.6, 1.2, 0], "proj": [0, 1.2, 0], "cam": [4.6, 1.2, 0], "cmp": [4.6, -1.0, 0], "upd": [-4.6, -1.0, 0]}
        txt = {"slm": "SLM phase φ", "proj": "projector + screen", "cam": "camera", "cmp": "error vs target",
               "upd": "update φ"}
        boxes = {}
        for k_, p in pos.items():
            t = tx(txt[k_], 24)
            r = RoundedRectangle(width=t.width + 0.5, height=0.8, corner_radius=0.15, color=CYAN, stroke_width=2).move_to(p)
            t.move_to(p)
            boxes[k_] = VGroup(r, t)
        order = ["slm", "proj", "cam", "cmp", "upd", "slm"]
        arrs = VGroup()
        for a_, b_ in zip(order[:-1], order[1:]):
            A, Bx = boxes[a_][0], boxes[b_][0]
            if pos[a_][1] == pos[b_][1]:
                s_ = A.get_right() if pos[b_][0] > pos[a_][0] else A.get_left()
                e_ = Bx.get_left() if pos[b_][0] > pos[a_][0] else Bx.get_right()
            else:
                s_ = A.get_bottom() if pos[b_][1] < pos[a_][1] else A.get_top()
                e_ = Bx.get_top() if pos[b_][1] < pos[a_][1] else Bx.get_bottom()
            arrs.add(arrow(s_, e_, buff=0.08, color=AMBER))
        self.play(*[FadeIn(b) for b in boxes.values()], run_time=0.6)
        self.play(LaggedStart(*[GrowArrow(a) for a in arrs], lag_ratio=0.3), run_time=1.2)
        path = VMobject().set_points_as_corners([pos["slm"], pos["proj"], pos["cam"], pos["cmp"], pos["upd"], pos["slm"]])
        dot = Dot(pos["slm"], radius=0.1, color=AMBER)
        self.add(tx("(gradients flow through a model of the optics)", 20, DIM).move_to([0, -1.55, 0]))
        self.caption("Camera-in-the-loop (CITL): display the hologram, photograph it, correct φ with the real error.")
        self.play(MoveAlongPath(dot, path), run_time=3.5, rate_func=linear)
        self.remove(dot)
        self.clear_all()
        # learned model
        mm = img("s5_model.png", w=11.0).move_to([0, 1.3, 0])
        self.add(mm, tx("learned wave propagation model (paper Fig. S5; architecture of Choi et al. 2022)", 18, DIM
                        ).next_to(mm, DOWN, buff=0.15))
        facts = VGroup(tx("4800 SLM phase patterns per colour, captured on the screen", 24),
                       tx("train / validation / test = 8 : 1 : 1,  batch size 4,  learning rate 4e−4", 24),
                       tx("about 24 hours of training on one NVIDIA A6000", 24)).arrange(DOWN, buff=0.2).move_to([0, -1.3, 0])
        self.add(facts)
        self.caption("Better: learn the projector. Show thousands of patterns, photograph the screen, fit a model.")
        self.caption("Then optimize every hologram through that calibrated model: SGD-model.")
        self.clear_all()
        # toy three-way
        h = 2.9
        xs = [-4.4, 0, 4.4]
        c = np.load(gen("toy_citl.npz"))
        for x, k_, n, cl in zip(xs, ["asm", "citl", "model"], ["SGD, ideal model", "SGD, camera in the loop",
                                                               "SGD, calibrated model"], [RED, AMBER, GREEN]):
            m_ = arr_img(np.load(gen(f"citl_{k_}_noloss.npy")), h).move_to([x, 0.6, 0])
            self.add(m_, label(m_, n, 22, cl), tx(f"toy PSNR {c[k_ + '_noloss_m'][2]:.1f} dB", 22, cl).next_to(m_, DOWN, buff=0.2))
        self.add(toy_note("toy simulation: all three captured on the same toy 'real' projector; model = learned illumination, "
                          "phase response, aberration, distance").move_to([0, -2.4, 0]))
        self.caption("Toy: CITL fixes part of the error; a calibrated model fixes nearly all of it.")
        self.clear_all()
        # paper Fig. S9 castle
        w = 3.9
        xs = [-4.1, 0, 4.1]
        for x, k_, n, cl in zip(xs, ["asm", "citl", "model"], ["ASM", "CITL", "learned model"], [RED, AMBER, GREEN]):
            a = img(f"s9_{k_}_castle_noloss.png", w=w).move_to([x, 1.15, 0])
            b = img(f"s9_{k_}_castle_loss.png", w=w).move_to([x, -1.05, 0])
            self.add(a, b, label(a, n, 24, cl))
        self.add(tx("without", 20, DIM).move_to([-6.55, 1.15, 0]).rotate(PI / 2),
                 tx("with loss", 20, DIM).move_to([-6.55, -1.05, 0]).rotate(PI / 2),
                 tx("captured (paper Fig. S9); blue: PSNR / LPIPS, red: brightness gain from the loss", 18, DIM
                    ).move_to([0, -2.55, 0]))
        self.caption("Captured: the loss adds +0% with ASM, +14% with CITL and +24% with the learned model.")
        self.caption("The loss only pays off when the model knows where the light really goes.")
        self.clear_all()


# ================================================================== S08 Results
class S08_Results(Base):
    def construct(self):
        self.title_card("Part 8", "Results on the prototype")
        self.header("8 · Results on the prototype")
        sc = img("s1_schematic.png", h=3.2).move_to([-4.88, 0.45, 0])
        ph = img("s1_photo.png", h=3.2).move_to([-0.72, 0.45, 0])
        self.play(FadeIn(sc), FadeIn(ph), run_time=0.6)
        self.add(tx("paper Fig. S1", 16, DIM).next_to(ph, DOWN, buff=0.1).align_to(ph, RIGHT))
        specs = VGroup(tx("phase SLM: TI DLP6750Q1EVM", 20), tx("1280 × 800 pixels, 10.8 µm, 4 bits", 20, DIM),
                       tx("laser: FISBA RGBeam, 50 mW", 20), tx("636.4 / 517.7 / 440.8 nm", 20, DIM),
                       tx(f"hologram at {I('z')} ≈ 20 cm", 20), tx("19.7 / 19.9 / 20.1 cm (R / G / B)", 20, DIM),
                       tx("Nikon 50 mm f/1.4 projection lens", 20), tx("screen 87 cm away: 12 cm × 21 cm", 20, DIM),
                       tx("8 frames time-multiplexed", 20)).arrange(DOWN, buff=0.1, aligned_edge=LEFT)
        specs.scale_to_fit_width(min(specs.width, 4.5)).move_to([4.5, 0.45, 0])
        self.add(specs)
        self.caption("The prototype: a 4-bit phase SLM, an RGB laser, and a camera lens as the projection lens.")
        self.caption("The hologram forms 20 cm from the SLM and is magnified 16.4× onto the screen.")
        self.clear_all()
        # Visualization 1
        clip = Clip("visualization1", w=8.8, loop=False)
        clip.mob.move_to([0, 0.35, 0])
        self.add(clip.mob, framed(clip.mob, DIM, 1),
                 tx("experimentally captured, paper Visualization 1 (left: with loss, right: without)", 18, DIM
                    ).next_to(clip.mob, DOWN, buff=0.15))
        self.caption("Captured on the prototype: the wipe swaps between the image with and without the loss.")
        self.caption("Same laser power. The border darkens and the picture brightens.")
        self.wait(1.0)
        self.clear_all()
        # Fig. S10 selection
        w = 3.9
        xs = [-4.1, 0, 4.1]
        for x, k_ in zip(xs, ["bunny1", "cat", "swallowtail"]):
            a = img(f"s10_{k_}_noloss.png", w=w).move_to([x, 1.25, 0])
            b = img(f"s10_{k_}_loss.png", w=w).move_to([x, -0.95, 0])
            self.add(a, b)
        self.add(tx("without", 20, RED).move_to([-6.55, 1.25, 0]).rotate(PI / 2),
                 tx("with loss", 20, GREEN).move_to([-6.55, -0.95, 0]).rotate(PI / 2),
                 tx("captured, learned model (paper Fig. S10); blue: PSNR / LPIPS, red: brightness gain", 18, DIM
                    ).move_to([0, 2.72, 0]))
        head = tx("around 30% brighter, image quality retained", 30, AMBER).move_to([0, -2.45, 0])
        self.caption("Captured: +36%, +33% and +32% brighter, with similar PSNR and LPIPS.")
        self.add(head)
        self.caption("Over the whole test set: 33% brighter captured, 66% in simulation.")
        self.clear_all()
        # gamma sweep
        w = 3.1
        xs = [-4.8, -1.6, 1.6, 4.8]
        for x, k_, n in zip(xs, ["w0", "w1", "w100", "winf"], ["γ = 0", "γ = 1", "γ = 100", "γ = ∞"]):
            a = img(f"s8_{k_}_monarch.png", w=w).move_to([x, 0.9, 0])
            b = img(f"s8_{k_}_eagle.png", w=w).move_to([x, -1.2, 0])
            self.add(a, b, label(a, n, 26, GREEN if k_ != "w0" else RED))
        self.add(tx("captured, loss weight sweep (paper Fig. S8)", 18, DIM).move_to([0, -2.55, 0]))
        self.caption("The weight γ is a dial: more light in the image for a little fidelity, e.g. to project under sunlight.")
        self.clear_all()
        # sparse + small ROI
        w = 3.6
        a = img("s10_glasses_noloss.png", w=w).move_to([-4.2, 1.1, 0])
        b = img("s10_glasses_loss.png", w=w).move_to([-4.2, -1.05, 0])
        self.add(a, b, label(a, "sparse content, captured", 22), tx("without", 18, RED).next_to(a, LEFT, buff=0.15).rotate(PI / 2),
                 tx("with", 18, GREEN).next_to(b, LEFT, buff=0.15).rotate(PI / 2))
        c_ = img("s12_small_noloss.png", w=w).move_to([1.0, 1.1, 0])
        d_ = img("s12_small_loss.png", w=w).move_to([1.0, -1.05, 0])
        self.add(c_, d_, label(c_, "small image region, simulated", 22),
                 tx("paper Figs. S10, S12", 18, DIM).move_to([-1.6, -2.6, 0]))
        big = VGroup(tx("10-fold", 44, AMBER, bold=True), tx("brighter with the loss", 22, AMBER),
                     tx("for a 100 × 160 pixel region", 20, DIM), tx("(simulated sinusoid)", 18, DIM)
                     ).arrange(DOWN, buff=0.12).move_to([4.95, 0.05, 0])
        self.caption("Sparse content already uses light well, so the loss adds less there (+9%).")
        self.add(big)
        self.caption("Small projection regions gain the most: 10-fold brighter in simulation.")
        self.clear_all()


# ================================================================== S09 Outlook
class S09_Outlook(Base):
    def construct(self):
        self.title_card("Part 9", "Limitations and outlook")
        self.header("9 · Limitations and outlook")
        items = [
            ("About 20 minutes to optimize one phase pattern (one RTX A6000)",
             "→ train a network that maps images directly to phases"),
            ("Undiffracted zeroth-order light lowers contrast",
             "→ higher-efficiency SLMs, or an off-axis setup with a phase ramp"),
            ("Video: each frame needs a different scale s",
             "→ adjust the laser power frame by frame"),
        ]
        g = VGroup()
        for a, b in items:
            g.add(VGroup(tx(a, 26), tx(b, 24, AMBER)).arrange(DOWN, aligned_edge=LEFT, buff=0.12))
        g.arrange(DOWN, aligned_edge=LEFT, buff=0.4).move_to([0, 0.9, 0])
        for it in g:
            self.add(it)
            self.wait(1.0)
        self.caption("Natural targets: head-up displays in cars, where content is sparse text and pictograms.")
        self.clear_all()
        summ = [
            "1   Holography redirects light instead of blocking it",
            "2   A light-efficiency loss keeps every photon inside the image",
            "3   A camera-calibrated model makes the long throw work on real hardware",
        ]
        sg = VGroup(*[tx(s, 28) for s in summ]).arrange(DOWN, aligned_edge=LEFT, buff=0.32).move_to([0, 1.2, 0])
        res = tx("around 30% brighter captured images at the same laser power, with image quality retained", 24, AMBER
                 ).move_to([0, -0.6, 0])
        self.add(sg)
        self.wait(1.2)
        self.add(res)
        self.caption("Brighter projection from the same laser, by putting the light where the picture is.")
        self.clear_all()
        self.remove(self.hdr)
        t1 = tx("High-Brightness Holographic Projection", 48, bold=True).move_to([0, 2.0, 0])
        t2 = tx("Optics Letters 48(15), 4041 (2023)  ·  doi 10.1364/OL.489617", 26).move_to([0, 1.2, 0])
        t4 = tx("Brian Chao, Manu Gopakumar, Suyeon Choi, Gordon Wetzstein", 24, DIM).move_to([0, 0.2, 0])
        t5 = tx("Stanford University", 22, DIM).move_to([0, -0.3, 0])
        t6 = tx("bchao1.github.io", 20, DIM).move_to([0, -0.85, 0])
        self.add(t1, t2, t4, t5, t6)
        self.wait(4.0)
