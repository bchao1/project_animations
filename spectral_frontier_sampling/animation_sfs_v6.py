"""Spectral Frontier Sampling (SFS) — 3b1b-style 5-min teaser with slide sections, v6.

v6 (2026-10-29): slide fixes from the deck screenshots: in slide mode (CAPTIONS=0) S00 has no title text (the slide
has its own title) and a larger spectrogram; S06 "free" label moved off the koi label; S06 note spacing.

v5 (2026-10-29): S03 drops "too late: drops signal" (no image's SNR = delta frontier rises above SPEED-3's grid);
it shows the share of SPEED's compute spent on noise per image instead. Label fixes in S02, S04, S06.

v4 (2026-10-29): fixes from the v3 480p sheets: L-shaped plot axes (manim Axes put the x axis through y = 0);
S02 labels in one right-hand column; S03 marks each image's first growth; S04 thumbnail triangle moved clear of
the panel; S05/S06 demo at step 19 (the estimate exists from step 16 for the climber); S06 test plot zoomed and
its per-sample beat uses the extremes, koi pond (grows at step 11) vs astronaut (38) vs SPEED (27); S06 extrapolate
is a labelled sketch (dots are coarse-grid powers, the rule estimates full-res power); S08 label spacing;
SPEED v8 caption timing (len/15 + 1 s).

v3 (2026-10-29): rebuilt from the 707e9d7 draft and the release rule (cont_mode). One video, one manim section per
beat (render with --save_sections for the deck). Running example: climber_ridge (cinematic 14, paper Fig. 1); the
per-sample pair: mountain lake (50) and Kyoto torii (12), paper Fig. 2. The frontier leads the story; adaptivity
(each image has its own frontier) is the second point. Style and helpers copied from SPEED v8
(../spectral_progressive_diffusion/animation_spectral_progressive_v8.py).

Data (see storyboard.md): data/capture_v3/<idx>/ (scratch/capture_v3.py, GPU: x0hat frames, spectra, rule trace),
data/sfs_v3.npz + data/sfs_v3.json + data/img/ (scratch/prep_v3.py, CPU). Every number on screen comes from these
files or from ~/slides/assets/papers/sfs/CONTEXT.md (marked in the code).

    CAPTIONS=0 ./render.sh s animation_sfs_v3.py      (slide sections)      ./render.sh l | h   (video)
"""
from manim import *
import numpy as np
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
CAP = os.path.join(DATA, "capture_v3")
IMGD = os.path.join(DATA, "img")

config.background_color = BLACK
config.disable_caching = True      # the play-call hash cannot serialize the data closures
# parallel scenes race on the shared text cache and on manim's mkdir calls (network disk): one text dir per process,
# media dirs created once at import (render.sh imports the script before the parallel renders)
config.text_dir = os.path.join(HERE, "media", "texts", str(os.getpid()))
for _q in ("480p15", "1080p60"):
    os.makedirs(os.path.join(HERE, "media", "videos", "animation_sfs_v6", _q, "sections"), exist_ok=True)
    os.makedirs(os.path.join(HERE, "media", "videos", "animation_sfs_v6", _q, "partial_movie_files"), exist_ok=True)
os.makedirs(os.path.join(HERE, "media", "images", "animation_sfs_v6"), exist_ok=True)

# ------------------------------------------------------------------ style (SPEED v8)
FONT = "CMU Serif"
MATHFONT = "Latin Modern Math"
SIG = "#58C4DD"        # signal (3b1b blue)
NOI = "#FC6255"        # noise (3b1b red)
FRONT = "#FFCA28"      # ours
OURS = "#FFCA28"
DIM = "#8A8A8A"
GHOST = "#3A3A3A"
BASE_C = "#B0BEC5"     # SPEED stages
FULL_C = "#8A8A8A"     # full res
WASTE = "#FF5252"      # compute spent on noise
SPD_C = "#E040FB"      # speedups (purple)
AMB = "#FFCA28"        # counts

CAPTIONS = os.environ.get("CAPTIONS", "1") != "0"
HEADERS = CAPTIONS
PART_CARDS = CAPTIONS
SECTION_HOLD = 0.3
HEADER_Y = 3.62
CAPTION_Y = -3.5
MATHCH = set("×→≈∼−∈≤≥·≲⋯")


def mathify(s):
    return "".join(f'<span font_family="{MATHFONT}">{c}</span>' if c in MATHCH else c for c in s)


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


XH = "<i>x̂</i>" + sub("0")
OM = "<i>ω</i>"


def img(path, h=None, w=None):
    p = path if os.path.isabs(path) else os.path.join(DATA, path)
    m = ImageMobject(p)
    if h is not None:
        m.height = h
    if w is not None:
        m.width = w
    return m


def rgba(a):
    a = np.clip(a, 0.0, 1.0)
    out = np.empty(a.shape[:2] + (4,), np.uint8)
    out[..., :3] = (a * 255 + 0.5).astype(np.uint8)
    out[..., 3] = 255
    return out


def hexrgb(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)])


# ------------------------------------------------------------------ data
Z = dict(np.load(os.path.join(DATA, "sfs_v3.npz")))
META = json.load(open(os.path.join(DATA, "sfs_v3.json")))
N, NF, DELTA = 50, 256, META["delta"]
AX_F, AX_Y = Z["axis_f"], Z["axis_y"]
LD = np.log10(DELTA)
PM = {int(k): v for k, v in META["prompts"].items()}
RUN = 14                       # climber_ridge, the running example (paper Fig. 1)
LO, HI = 50, 12                # mountain lake (lower detail) and Kyoto torii (higher detail), paper Fig. 2
GALLERY = [35, 99, 54, 14, 12, 58]   # smooth -> detailed
SHORT = {14: "climber", 50: "mountain lake", 12: "Kyoto torii", 35: "astronaut, dunes", 58: "koi pond",
         99: "beach umbrella", 54: "jellyfish"}

# log10 SNR colours: noise side (SNR < δ) dark red, signal side blue ramp, as in SPEED's noise slide
_NOISE = [hexrgb("#120505"), hexrgb("#6E2420")]
_SIGNAL = [hexrgb("#10304A"), hexrgb(SIG), hexrgb("#E3F6FC")]


def snr_rgb(v):
    v = np.asarray(v, float)
    out = np.zeros(v.shape + (3,))
    a = np.clip((v + 4) / (LD + 4), 0, 1)[..., None]
    noise = _NOISE[0] * (1 - a) + _NOISE[1] * a
    b = np.clip((v - LD) / (2 - LD), 0, 1)[..., None]
    lo = np.clip(b * 2, 0, 1)
    hi = np.clip(b * 2 - 1, 0, 1)
    sig = np.where(b < 0.5, _SIGNAL[0] * (1 - lo) + _SIGNAL[1] * lo, _SIGNAL[1] * (1 - hi) + _SIGNAL[2] * hi)
    out[:] = np.where((v < LD)[..., None], noise, sig)
    return out


def yfrac(f, c):
    """vertical position (0..1) of frequency bin f: linear (c = 0) or compute-scaled (c = 1, area = compute)."""
    f = np.asarray(f, float)
    return (1 - c) * f / (NF // 2) + c * np.interp(f, AX_F, AX_Y)


def f_of_y(y, c):
    fl = y * (NF // 2)
    if c <= 0:
        return fl
    fc = np.interp(y, AX_Y, AX_F)
    return (1 - c) * fl + c * fc


def spec_pixels(idx, c=0.0, upto=N - 1, H=360, colw=8):
    ls = Z[f"lsnr_{idx}"]
    rows = np.linspace(1, 0, H)                     # top row first
    f = np.clip(np.round(f_of_y(rows, c)), 1, NF // 2).astype(int)
    a = snr_rgb(ls[:, f].T)                         # [H, steps, 3]
    a[:, upto + 1:] = 0.0
    return rgba(np.repeat(a, colw, axis=1))


class Spec:
    """A spectrogram panel: steps 0..49 left to right, frequency bins 1..128 bottom to top (linear or compute)."""

    def __init__(self, idx, center, w, h, c=0.0, upto=N - 1):
        self.idx, self.c, self.w, self.h = idx, c, w, h
        self.center = np.array(center, float)
        self.im = ImageMobject(spec_pixels(idx, c, upto))
        self.im.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
        self.im.stretch_to_fit_width(w)
        self.im.stretch_to_fit_height(h)
        self.im.move_to(self.center)

    @property
    def left(self):
        return self.center[0] - self.w / 2

    @property
    def bottom(self):
        return self.center[1] - self.h / 2

    def P(self, step, f, c=None):
        c = self.c if c is None else c
        return np.array([self.left + step / N * self.w, self.bottom + yfrac(f, c) * self.h, 0])

    def set_pixels(self, c=None, upto=N - 1, idx=None):
        self.c = self.c if c is None else c
        self.idx = self.idx if idx is None else idx
        self.im.pixel_array = spec_pixels(self.idx, self.c, upto)

    def stairs(self, ceil, color, sw=4, upto=N, c=None):
        """step line of per-step ceilings (bins); ceil[i] holds on [i, i + 1)."""
        pts = []
        for i in range(min(upto, N)):
            a, b = self.P(i, ceil[i], c), self.P(i + 1, ceil[i], c)
            if pts:
                pts.append(a)
            pts += [a, b]
        m = VMobject(stroke_color=color, stroke_width=sw)
        m.set_points_as_corners(pts)
        return m

    def fill(self, lo, hi, color, op=0.55, c=None):
        """rectangles between per-step bins lo < hi (compute spent between two ceilings)."""
        g = VGroup()
        for i in range(N):
            if hi[i] > lo[i] + 0.01:
                a, b = self.P(i, lo[i], c), self.P(i + 1, hi[i], c)
                g.add(Rectangle(width=b[0] - a[0] + 0.004, height=b[1] - a[1], stroke_width=0, fill_color=color,
                                fill_opacity=op).move_to((a + b) / 2))
        return g

    def axes(self, size=20, c=None, xlab=True, ylab=True):
        c = self.c if c is None else c
        g = VGroup()
        for s in (0, 25, 49):
            p = self.P(s + 0.5, 0, 0)
            p[1] = self.bottom
            g.add(tx(str(s), size, GREY_B).next_to(p, DOWN, buff=0.1))
        if xlab:
            g.add(tx("denoising step", size + 2, GREY_B).next_to(self.im, DOWN, buff=0.45))
        if ylab:
            for f in (32, 64, 128):
                p = self.P(0, f, c)
                g.add(tx(str(f), size, GREY_B).next_to(p, LEFT, buff=0.1))
            g.add(tx("frequency", size + 2, GREY_B).rotate(PI / 2).next_to(self.im, LEFT, buff=0.55))
        return g


def ceil_of(sides):
    return np.asarray(sides, float) / 2


FULL_CEIL = np.full(N, NF / 2)
S3_CEIL = ceil_of(Z["speed3"])
S2_CEIL = ceil_of(Z["speed2"])


def ours_ceil(idx):
    return ceil_of(Z[f"ours_{idx}"])


def front(idx):
    return Z[f"front_{idx}"].astype(float)


def xhat(idx, i):
    """path of the x0hat preview at step i (grid in its name)."""
    d = os.path.join(CAP, str(idx))
    for f in os.listdir(d):
        if f.startswith(f"x0_{i:02d}_n") and f.endswith(".png"):
            return os.path.join(d, f)
    raise FileNotFoundError((idx, i))


def grid_of(idx, i):
    return int(re.search(r"_n(\d+)\.png", xhat(idx, i)).group(1))


def fmt_x(v):
    return f"{v:.1f}×"


def grow_bar(bar, edge, run_time=0.5):
    """bar grows from one edge at full thickness (GrowFromEdge also scales the thickness)."""
    start = bar.copy()
    if edge is LEFT:
        start.stretch(0.001, 0, about_edge=LEFT)
    else:
        start.stretch(0.001, 1, about_edge=DOWN)
    return Transform(start, bar, run_time=run_time), start


class LAx(VGroup):
    """Plain L-shaped axes: x along the bottom, y along the left (manim Axes put the x axis through y = 0)."""

    def __init__(self, center, w, h, xr, yr, **kw):
        super().__init__(**kw)
        self.xr, self.yr, self.w, self.h = xr, yr, w, h
        self.o = np.array(center, float) - np.array([w / 2, h / 2, 0])
        self.add(Line(self.o, self.o + RIGHT * w, stroke_width=2, color=GREY_B),
                 Line(self.o, self.o + UP * h, stroke_width=2, color=GREY_B))

    def c2p(self, x, y):
        return self.o + np.array([(x - self.xr[0]) / (self.xr[1] - self.xr[0]) * self.w,
                                  (y - self.yr[0]) / (self.yr[1] - self.yr[0]) * self.h, 0])

    def xlabels(self, vals, size=20, fmt=str):
        return VGroup(*[tx(fmt(v), size, GREY_B).next_to(self.c2p(v, self.yr[0]), DOWN, buff=0.1) for v in vals])


# ------------------------------------------------------------------ scene base (SPEED v8)
class Base(Scene):
    def setup(self):
        self.cap = None
        self.hdr = None

    def step(self, name, hold=SECTION_HOLD):
        if self.mobjects and hold:
            self.wait(hold)
        self.next_section(name)

    def header(self, s):
        if not HEADERS:
            return
        if self.hdr is not None:
            self.remove(self.hdr)
        self.hdr = tx(s, 24, DIM).to_corner(UL, buff=0.35).set_y(HEADER_Y)
        self.add(self.hdr)

    def _hold_caption(self):
        if self.cap is not None:
            left = self.cap_need - (self.renderer.time - self.cap_t0)
            if left > 0.05:
                self.wait(left)

    def caption(self, s, wait=None, size=28):
        if not CAPTIONS:
            return None
        self._hold_caption()
        if self.cap is not None:
            self.remove(self.cap)
        c = tx(s, size)
        if c.width > 13.2:
            c.scale_to_fit_width(13.2)
        c.move_to([0, CAPTION_Y, 0])
        self.cap = c
        self.cap_t0 = self.renderer.time
        self.cap_need = len(re.sub(r"<[^>]+>", "", s)) / 15.0 + 1.0
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
        g = VGroup(tx(num, 30, DIM), tx(title, 50, bold=True)).arrange(DOWN, buff=0.3)
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


def fill_columns(scene, sp, upto_tracker, idx, c):
    """updater: reveal spectrogram columns up to the tracker's value."""
    def upd(m, dt):
        sp.set_pixels(c=c, upto=int(upto_tracker.get_value()), idx=idx)
    return upd


def frames_player(idx, h, center, tracker, nearest=False):
    """x0hat preview that follows a step tracker (frames are drawn at a fixed display size)."""
    cache = {}

    def arr(i):
        if i not in cache:
            from PIL import Image
            im = Image.open(xhat(idx, i)).convert("RGB").resize((360, 360), Image.LANCZOS)
            cache[i] = rgba(np.asarray(im) / 255.0)
        return cache[i]
    m = ImageMobject(arr(0))
    m.height = h
    m.move_to(center)
    m.add_updater(lambda mob, dt: setattr(mob, "pixel_array", arr(int(tracker.get_value()))))
    return m


# ================================================================== S00 Title
class S00_Title(Base):
    def construct(self):
        self.step("title")
        t = tx("Spectral Frontier Sampling", 60, bold=True).move_to([0, 2.8, 0])
        u = tx("Per-sample continuous expansion for fast diffusion sampling", 30, GREY_B).move_to([0, 2.05, 0])
        if CAPTIONS:
            self.add(t, u)
            sp = Spec(RUN, [0, -0.75, 0], 7.0, 3.6, c=1.0, upto=-1)
        else:
            sp = Spec(RUN, [0, 0.2, 0], 9.6, 5.4, c=1.0, upto=-1)
        self.add(sp.im)
        k = ValueTracker(-1)
        sp.im.add_updater(fill_columns(self, sp, k, RUN, 1.0))
        self.play(k.animate.set_value(N - 1), run_time=3.0, rate_func=linear)
        sp.im.clear_updaters()
        fr = sp.stairs(front(RUN), WHITE, 2.5)
        st = sp.stairs(ours_ceil(RUN), OURS, 5)
        self.play(Create(fr), run_time=0.7)
        self.play(Create(st), run_time=1.5)
        self.wait(1.5)


# ================================================================== S01 Frontier
class S01_Frontier(Base):
    def construct(self):
        self.title_card("Part 1", "The problem")
        self.header("1 · The spectral frontier")
        self.step("xhat")
        k = ValueTracker(0)
        left = frames_player(RUN, 4.4, [-4.2, 0.35, 0], k)
        lab_l = tx("model's clean guess " + XH, 26).next_to(left, UP, buff=0.22)
        sl = always_redraw(lambda: tx(f"step {int(k.get_value())} of 50", 24, AMB).next_to(left, DOWN, buff=0.22))
        self.add(left, lab_l, sl)
        self.caption("Diffusion builds an image from coarse to fine.")
        self.play(k.animate.set_value(N - 1), run_time=4.0, rate_func=linear)
        self.step("column")
        sp = Spec(RUN, [2.5, 0.35, 0], 6.4, 4.4, c=0.0, upto=-1)
        ax = sp.axes()
        lab_r = tx("signal-to-noise ratio, per frequency and step", 26).next_to(sp.im, UP, buff=0.22)
        self.add(sp.im, ax, lab_r)
        key = VGroup(Square(0.22, fill_color=hexrgb("#6E2420"), fill_opacity=1, stroke_width=0),
                     tx("still noise", 20, GREY_B), Square(0.22, fill_color=SIG, fill_opacity=1, stroke_width=0),
                     tx("signal", 20, GREY_B)).arrange(RIGHT, buff=0.15).next_to(sp.im, DOWN, buff=0.95)
        self.add(key)
        k.set_value(0)
        kk = ValueTracker(-1)
        sp.im.add_updater(fill_columns(self, sp, kk, RUN, 0.0))
        k.add_updater(lambda m, dt: m.set_value(max(0, min(N - 1, kk.get_value()))))
        self.caption("Each column shows which frequencies have left the noise at that step.")
        self.play(kk.animate.set_value(N - 1), run_time=5.0, rate_func=linear)
        sp.im.clear_updaters()
        k.clear_updaters()
        self.step("frontier")
        fr = sp.stairs(front(RUN), WHITE, 3)
        fl = tx("spectral frontier", 24, WHITE).move_to(sp.P(24, front(RUN)[24] + 18))
        self.play(Create(fr), run_time=1.2)
        self.add(fl)
        self.caption("The edge between signal and noise moves up over time: the spectral frontier.")
        self.wait(2.0)


# ================================================================== S02 Waste
class S02_Waste(Base):
    def construct(self):
        self.header("2 · Where compute goes")
        self.step("area")
        sp = Spec(RUN, [-1.0, 0.2, 0], 7.4, 4.8, c=0.0)
        ax = sp.axes()
        fr = sp.stairs(front(RUN), WHITE, 3)
        self.add(sp.im, ax, fr)
        th = ValueTracker(0.0)

        def upd(m, dt):
            c = th.get_value()
            sp.set_pixels(c=c)
            m.become(sp.stairs(front(RUN), WHITE, 3, c=c))
        fr.add_updater(upd)
        axg = always_redraw(lambda: sp.axes(c=th.get_value()))
        self.remove(ax)
        self.add(axg)
        note = tx("height ∝ compute per step: area = cost", 22, GREY_B).next_to(sp.im, UP, buff=0.18).align_to(sp.im, LEFT)
        self.caption("Stretch the height to match compute: area becomes cost.")
        self.play(th.animate.set_value(1.0), run_time=2.0)
        fr.clear_updaters()
        axg.clear_updaters()
        sp.c = 1.0
        self.add(note)
        self.step("fullres")
        full = sp.stairs(FULL_CEIL, FULL_C, 5)
        wf = sp.fill(front(RUN), FULL_CEIL, WASTE, 0.45)
        self.play(Create(full), run_time=0.6)
        self.play(FadeIn(wf), run_time=0.7)
        lf = tx("full res · " + col("70.5 s", AMB), 24, FULL_C).next_to(sp.P(N, NF / 2), RIGHT, buff=0.2)
        lw = tx("still noise", 24, WASTE).move_to(sp.P(14, 100))
        self.add(lf, lw)
        self.caption("Full resolution pays for every frequency, even those that are still noise.")
        self.step("speed")
        self.play(FadeOut(wf), run_time=0.4)
        s3 = sp.stairs(S3_CEIL, BASE_C, 5)
        ws = sp.fill(front(RUN), S3_CEIL, WASTE, 0.45)
        self.play(Create(s3), run_time=1.0)
        self.play(FadeIn(ws), run_time=0.6)
        cs = PM[RUN]["compute"]["speed3"]
        ls = VGroup(tx("SPEED-3", 24, BASE_C), tx(f"{cs * 100:.0f}% of full-res compute", 20, AMB)).arrange(
            DOWN, aligned_edge=LEFT, buff=0.08).next_to(sp.P(N, 64), RIGHT, buff=0.2)
        self.add(ls)
        self.remove(lw)
        self.caption("SPEED grows in 2× stages. Each stage adds a whole octave that is still noise.")
        self.step("more_stages")
        ghosts = VGroup()
        for S in (4, 6, 8):
            ks = np.linspace(20, 44, S - 1).round().astype(int)
            sides = np.full(N, 64.0)
            for j, kk in enumerate(ks):
                sides[kk:] = 64 * 4 ** ((j + 1) / (S - 1))
            ghosts.add(sp.stairs(np.minimum(sides, NF) / 2, BASE_C, 2).set_stroke(opacity=0.45))
        self.play(LaggedStart(*[Create(g) for g in ghosts], lag_ratio=0.5), run_time=1.6)
        self.add(tx("more stages: less waste,\nbut more hand-set choices", 22, GREY_B).next_to(sp.im, RIGHT, buff=0.2)
                 .shift(UP * 0.7))
        self.caption("More stages waste less, but each stage is another hand-set choice.")
        self.wait(1.5)


# ================================================================== S03 Per-sample
class S03_PerSample(Base):
    def construct(self):
        self.header("3 · Every image has its own frontier")
        self.step("morph")
        order = [35, LO, HI, 58]
        sp = Spec(order[0], [1.6, 0.3, 0], 7.6, 4.6, c=1.0)
        ax = sp.axes()
        fr = sp.stairs(front(order[0]), WHITE, 3)
        th = img(os.path.join(IMGD, f"ours_{order[0]}.jpg"), 3.2).move_to([-4.7, 0.6, 0])
        nm = tx(SHORT[order[0]], 26).next_to(th, DOWN, buff=0.2)

        def mark(idx):
            k0 = PM[idx]["first_grow"]
            d = Dot(sp.P(k0, 32), radius=0.09, color=OURS)
            t = tx(f"leaves the first grid at step {k0}", 22, OURS).next_to(nm, DOWN, buff=0.15)
            return VGroup(d, t)
        mk = mark(order[0])
        self.add(sp.im, ax, fr, th, nm, mk)
        self.caption("Each image has its own frontier. Smooth images reach fine detail late.")
        self.wait(1.2)
        for idx in order[1:]:
            th2 = img(os.path.join(IMGD, f"ours_{idx}.jpg"), 3.2).move_to(th.get_center())
            nm2 = tx(SHORT[idx], 26).next_to(th2, DOWN, buff=0.2)
            sp.set_pixels(idx=idx)
            self.play(Transform(fr, sp.stairs(front(idx), WHITE, 3)), FadeOut(th), FadeIn(th2),
                      FadeOut(nm), FadeIn(nm2), FadeOut(mk), run_time=0.8)
            th, nm = th2, nm2
            mk = mark(idx)
            self.add(mk)
            self.wait(1.5)
        self.step("pair")
        self.clear_all()
        P = {}
        for j, idx in enumerate((LO, HI)):
            s = Spec(idx, [-3.4 + 6.8 * j, -0.1, 0], 5.4, 3.6, c=1.0)
            t = img(os.path.join(IMGD, f"ours_{idx}.jpg"), 1.5).move_to(s.P(0, 0) + np.array([0.8, 2.75 + 0.0, 0]))
            t.move_to([s.left + 0.8, s.bottom + s.h - 0.8, 0])
            lab = tx(SHORT[idx] + (" · lower detail" if idx == LO else " · higher detail"), 26).next_to(s.im, UP, buff=0.2)
            frl = s.stairs(front(idx), WHITE, 3)
            P[idx] = s
            self.add(s.im, s.axes(size=18, ylab=(j == 0)), frl, t, lab)
        self.caption("A smooth lake and a detailed temple: very different frontiers.")
        self.step("same_speed")
        for idx, s in P.items():
            st = s.stairs(S3_CEIL, BASE_C, 4)
            self.play(Create(st), run_time=0.7)
            self.add(s.fill(front(idx), S3_CEIL, WASTE, 0.45))
            yc, yf = yfrac(S3_CEIL, 1.0), yfrac(np.minimum(front(idx), S3_CEIL), 1.0)
            share = float((yc - yf).sum() / yc.sum())          # share of SPEED-3's compute above this image's frontier
            self.add(tx(f"{share * 100:.0f}% of SPEED's compute is still noise", 22, WASTE).next_to(s.im, DOWN, buff=0.75))
        self.add(tx("SPEED-3: the same steps for every image", 22, BASE_C).move_to([0, 2.95, 0]))
        self.caption("SPEED uses one schedule for every image. The smooth one wastes more of it.")
        self.step("question")
        if CAPTIONS:
            self.clear_all()
            self.add(tx("Grow with each image's own frontier?", 40, bold=True))
            self.wait(2.0)


# ================================================================== S04 Tracking
class S04_Tracking(Base):
    def construct(self):
        self.title_card("Part 2", "The method")
        self.header("4 · Frontier tracking")
        self.step("ladder")
        sp = Spec(RUN, [2.5, 0.4, 0], 6.6, 4.6, c=1.0)
        self.add(sp.im, sp.axes(), sp.stairs(front(RUN), WHITE, 3))
        ticks = VGroup(*[Line(sp.P(N, f) + RIGHT * 0.05, sp.P(N, f) + RIGHT * 0.22, stroke_width=1.5, color=GREY_B)
                         for f in range(32, 129, 1)])
        self.play(LaggedStart(*[Create(t) for t in ticks], lag_ratio=0.01), run_time=1.0)
        cand = tx("candidate grid sides 64, 66, …, 256", 22, GREY_B).next_to(sp.im, UP, buff=0.15).align_to(sp.im, RIGHT)
        self.add(cand)
        self.caption("Any grid size is allowed, not only 2× steps.")
        self.step("grow")
        oc = ours_ceil(RUN)
        st = sp.stairs(oc, OURS, 5)
        shown = {}
        picks = [0, 35, 37, 41, 44, 49]        # x0hat at 64, ..., 256 (Fig. 1 grids)
        k = ValueTracker(0)
        st_live = always_redraw(lambda: sp.stairs(oc, OURS, 5, upto=max(1, int(k.get_value()) + 1)))
        self.add(st_live)
        thumbs = Group()
        for j, i in enumerate(picks):
            g = grid_of(RUN, i)
            h = 0.4 + 1.0 * (g - 64) / (256 - 64)
            m = img(xhat(RUN, i), h)
            thumbs.add(m)
        x = -6.9
        for m in thumbs:
            m.move_to([x + m.width / 2, -2.25 + m.height / 2, 0])
            x += m.width + 0.08
        labs = VGroup(*[tx(f"{grid_of(RUN, i)}²", 18, AMB).next_to(m, DOWN, buff=0.06) for m, i in zip(thumbs, picks)])
        self.caption("When the grid's edge fills with signal, grow to the next size that is still noise.")
        prev = 0
        for j, i in enumerate(picks):
            self.play(k.animate.set_value(i), run_time=max(0.3, 0.06 * (i - prev)), rate_func=linear)
            dot = Dot(sp.P(i + 0.5, oc[i]), radius=0.07, color=OURS)
            self.add(dot)
            self.play(FadeIn(thumbs[j]), run_time=0.3)
            self.add(labs[j])
            prev = i
        st_live.clear_updaters()
        self.step("saved")
        self.remove(cand)
        sv = sp.fill(oc, FULL_CEIL, WASTE, 0.4)
        self.play(FadeIn(sv), run_time=0.6)
        self.add(sp.stairs(FULL_CEIL, FULL_C, 4), sp.stairs(S3_CEIL, BASE_C, 2).set_stroke(opacity=0.5))
        m = PM[RUN]
        self.add(tx("skipped: frequencies that are still noise", 22, WHITE).move_to(sp.P(20, 108)))
        self.add(tx(f"{m['t_ours']:.0f} s vs {m['t_full']:.0f} s full res", 24, AMB, bold=True).next_to(sp.im, UP, buff=0.15)
                 .align_to(sp.im, RIGHT),
                 tx(fmt_x(m["speedup"]) + " faster", 24, SPD_C, bold=True).next_to(sp.im, UP, buff=0.15).align_to(sp.im, LEFT))
        self.caption("We skip almost all of the noise-only compute.")
        self.step("cap")
        box = RoundedRectangle(width=3.6, height=1.5, corner_radius=0.12, stroke_color=GREY_B, stroke_width=2,
                               fill_color=BLACK, fill_opacity=0.9).move_to([-4.3, 2.2, 0])
        a = tx("64² → 256²", 26).move_to(box.get_center() + UP * 0.25)
        x_ = Cross(a, stroke_color=WASTE, stroke_width=4)
        b = tx("at most 2× per growth", 22, GREY_B).move_to(box.get_center() + DOWN * 0.4)
        self.add(box, a, b)
        self.play(Create(x_), run_time=0.5)
        self.caption("One growth is at most 2×, so one bad guess cannot jump to full size.")
        self.wait(1.0)


# ================================================================== S05 Measure
class S05_Measure(Base):
    def construct(self):
        self.header("5 · Reading the frontier while sampling")
        self.step("unknown")
        sp = Spec(RUN, [0.3, 0.3, 0], 8.0, 4.8, c=1.0)
        fr = sp.stairs(front(RUN), WHITE, 3)
        self.add(sp.im, sp.axes(), fr)
        self.caption("The true frontier needs the final image, which we do not have yet.")
        q = tx("?", 120, WHITE, bold=True).move_to(sp.center)
        self.play(sp.im.animate.set_opacity(0.15), fr.animate.set_stroke(opacity=0.15), run_time=0.8)
        self.add(q)
        self.step("xhat_now")
        self.clear_all()
        i0 = 19
        x = img(xhat(RUN, i0), 4.2).move_to([-3.6, 0.2, 0])
        g = grid_of(RUN, i0)
        lab = tx(f"clean guess {XH} at step {i0}, grid {g}²", 26).next_to(x, UP, buff=0.2)
        self.add(x, lab)
        grid = VGroup(*[Line(x.get_corner(DL) + RIGHT * x.width * j / 16, x.get_corner(UL) + RIGHT * x.width * j / 16,
                             stroke_width=1, color=WHITE, stroke_opacity=0.35) for j in range(17)],
                      *[Line(x.get_corner(DL) + UP * x.height * j / 16, x.get_corner(DR) + UP * x.height * j / 16,
                             stroke_width=1, color=WHITE, stroke_opacity=0.35) for j in range(17)])
        self.play(FadeIn(grid), run_time=0.5)
        self.caption("At every step the model already guesses the clean image.")
        self.wait(1.0)
        self.play(FadeOut(grid), run_time=0.4)
        self.step("rings")
        r = np.load(os.path.join(CAP, str(RUN), "rings.npz"))
        S = r[f"stats_{i0:02d}"]
        sig = float(r["sig"][list(r["i"]).index(i0)])
        C = S[0] / (1 - sig)
        ax = LAx([3.4, 0.0, 0], 5.2, 4.0, (np.log10(1.6), np.log10(40)), (-0.6, 3.2))
        lx = VGroup(*[tx(str(v), 20, GREY_B).next_to(ax.c2p(np.log10(v), -0.6), DOWN, buff=0.1) for v in (2, 4, 8, 16, 32)])
        self.add(ax, lx, tx("ring frequency " + OM, 22, GREY_B).next_to(ax, DOWN, buff=0.5),
                 tx("power (log)", 22, GREY_B).rotate(PI / 2).next_to(ax, LEFT, buff=0.25))
        dots = VGroup()
        for w in range(2, 33):
            if C[w] <= 0:
                continue
            ok = w <= round(0.375 * g / 2)
            d = Dot(ax.c2p(np.log10(w), np.log10(C[w])), radius=0.06, color=SIG if ok else GHOST)
            dots.add(d)
        self.play(LaggedStart(*[FadeIn(d) for d in dots], lag_ratio=0.04), run_time=1.2)
        self.add(tx("measured: rings 2–12", 22, SIG).move_to(ax.c2p(np.log10(9), 2.9)))
        self.add(tx(f"grid edge {OM} = {g // 2}", 20, GREY_B).next_to(ax.c2p(np.log10(g // 2), 0.4), UP, buff=0.1))
        self.caption("Its low frequencies are already clear: measure their power.")
        self.wait(1.0)


# ================================================================== S06 Estimate
class S06_Estimate(Base):
    def test_series(self, idx):
        tr = Z[f"tr_{idx}"]
        lps = Z[f"lps_{idx}"]
        m = tr[:, 1] == 64
        i, e, sd, u = tr[m, 0].astype(int), tr[m, 2], tr[m, 3], tr[m, 4]
        with np.errstate(divide="ignore"):
            v = e + np.log(np.maximum(u, 1e-9)) - sd ** 2      # the release test: e + ln u − sd² ≥ ln P*(σ_{i+1})
        return i, v / np.log(10), sd / np.log(10), lps / np.log(10)

    def construct(self):
        self.header("6 · From low rings to the edge")
        self.step("extrapolate")
        i0 = 19
        r = np.load(os.path.join(CAP, str(RUN), "rings.npz"))
        S = r[f"stats_{i0:02d}"]
        sig = float(r["sig"][list(r["i"]).index(i0)])
        C = S[0] / (1 - sig)
        # sketch: the measured low rings (real, step 19, grid 64) and their trend carried to the grid edge with a
        # widening band. The rule's real estimate (GLS with the prior and the VAE map) is in the next beat, in its units.
        ax = LAx([-2.6, 0.1, 0], 6.0, 4.4, (np.log10(1.6), np.log10(40)), (-1.8, 3.2))
        self.add(ax, VGroup(*[tx(str(v), 20, GREY_B).next_to(ax.c2p(np.log10(v), -1.8), DOWN, buff=0.1)
                              for v in (2, 4, 8, 16, 32)]),
                 tx("ring frequency " + OM, 22, GREY_B).next_to(ax, DOWN, buff=0.5),
                 tx("power (log)", 22, GREY_B).rotate(PI / 2).next_to(ax, LEFT, buff=0.25))
        ws = [w for w in range(2, 13) if C[w] > 0]
        for w in ws:
            self.add(Dot(ax.c2p(np.log10(w), np.log10(C[w])), radius=0.06, color=SIG))
        a1, a0 = np.polyfit(np.log10(ws), np.log10(C[ws]), 1)
        xs = np.linspace(np.log10(12), np.log10(32), 30)
        up = [ax.c2p(x, a0 + a1 * x + 0.9 * (x - xs[0]) / (xs[-1] - xs[0]) + 0.05) for x in xs]
        dn = [ax.c2p(x, max(-1.75, a0 + a1 * x - 0.9 * (x - xs[0]) / (xs[-1] - xs[0]) - 0.05)) for x in xs]
        cone = Polygon(*up, *dn[::-1], stroke_width=0, fill_color=OURS, fill_opacity=0.3)
        mid = DashedLine(ax.c2p(xs[0], a0 + a1 * xs[0]), ax.c2p(xs[-1], a0 + a1 * xs[-1]), color=OURS, stroke_width=3)
        self.play(FadeIn(cone), Create(mid), run_time=0.9)
        x32 = np.log10(32)
        self.add(Dot(ax.c2p(x32, a0 + a1 * x32), radius=0.08, color=OURS),
                 tx("estimate at the grid edge", 22, OURS).next_to(ax.c2p(x32, a0 + a1 * x32 + 0.95), UP, buff=0.05),
                 tx("sketch", 18, DIM).move_to(ax.c2p(np.log10(30), 3.0)))
        self.add(tx("power at nearby frequencies\nmoves together", 24, GREY_B).move_to([3.9, 1.6, 0]),
                 tx("no extra network calls · no training", 22, GREY_B).move_to([3.9, 0.3, 0]))
        self.caption("Power at nearby frequencies moves together, so the low rings predict the edge.")
        self.step("test")
        self.clear_all()
        i, v, s, lps = self.test_series(RUN)
        ok = np.isfinite(v)
        x0, x1 = 12, 24
        yl = min(v[ok].min(), lps[x1 + 1]) - 0.35
        yh = max(v[ok].max(), lps[x0 + 1]) + 0.35
        ax2 = LAx([0.2, 0.2, 0], 8.6, 4.4, (x0, x1), (yl, yh))
        self.add(ax2, ax2.xlabels(range(x0, x1 + 1, 4)), tx("denoising step", 22, GREY_B).next_to(ax2, DOWN, buff=0.5),
                 tx("power at the grid edge (log)", 22, GREY_B).rotate(PI / 2).next_to(ax2, LEFT, buff=0.25))
        thr = VMobject(stroke_color=WHITE, stroke_width=3).set_points_as_corners(
            [ax2.c2p(t, lps[t + 1]) for t in range(x0, x1 + 1)])
        self.play(Create(DashedVMobject(thr, num_dashes=36)), run_time=0.8)
        self.add(tx("needed to leave the noise", 22, WHITE).next_to(ax2.c2p(x0 + 0.3, lps[x0 + 1]), UR, buff=0.08))
        self.add(tx(f"steps 0–{int(i[ok][0]) - 1}: no estimate yet,\nlow rings still noisy", 18, DIM)
                 .move_to(ax2.c2p(14.0, yl + 0.5)))
        pts = [ax2.c2p(a, b) for a, b in zip(i, v) if np.isfinite(b) and x0 <= a <= x1]
        band = Polygon(*[ax2.c2p(a, b + sd) for a, b, sd in zip(i, v, s) if np.isfinite(b) and x0 <= a <= x1],
                       *[ax2.c2p(a, b - sd) for a, b, sd in zip(i, v, s) if np.isfinite(b) and x0 <= a <= x1][::-1],
                       stroke_width=0, fill_color=OURS, fill_opacity=0.25)
        line = VMobject(stroke_color=OURS, stroke_width=4).set_points_as_corners(pts)
        self.play(FadeIn(band), Create(line), run_time=1.4)
        self.add(tx("estimated edge power", 22, OURS).move_to(ax2.c2p(19.5, yl + 0.25)))
        fire = [a for a, b in zip(i, v) if np.isfinite(b) and b >= lps[int(a) + 1]]
        if fire:
            f0 = int(fire[0])
            p0 = ax2.c2p(f0, v[list(i).index(f0)])
            self.add(Dot(p0, radius=0.1, color=OURS), tx(f"grow at step {f0 + 1}", 26, OURS, bold=True).next_to(p0, DR, buff=0.12))
        self.caption("When the edge's estimated power passes the noise threshold, grow.")
        self.step("two_samples")
        self.clear_all()
        KOI, AST = 58, 35
        x0, x1 = 5, 42
        ser = {idx: self.test_series(idx) for idx in (KOI, AST)}
        lps = ser[AST][3]
        vals = np.concatenate([sv[1][np.isfinite(sv[1])] for sv in ser.values()] + [lps[x0 + 1:x1 + 2]])
        yl, yh = vals.min() - 0.3, vals.max() + 0.3
        ax3 = LAx([0.0, 0.25, 0], 9.6, 4.4, (x0, x1), (yl, yh))
        self.add(ax3, ax3.xlabels([10, 20, 30, 40]), tx("denoising step", 22, GREY_B).next_to(ax3, DOWN, buff=0.5),
                 tx("power at the grid edge (log)", 22, GREY_B).rotate(PI / 2).next_to(ax3, LEFT, buff=0.25))
        thr = VMobject(stroke_color=WHITE, stroke_width=3).set_points_as_corners([ax3.c2p(t, lps[t + 1]) for t in range(x0, x1 + 1)])
        self.add(DashedVMobject(thr, num_dashes=50))
        pbar = lps[27]                                  # the power whose threshold step is SPEED-3's first growth (27)
        sl = Line(ax3.c2p(x0, pbar), ax3.c2p(x1, pbar), color=BASE_C, stroke_width=3)
        self.play(Create(sl), run_time=0.6)
        self.add(Dot(ax3.c2p(26, pbar), radius=0.09, color=BASE_C),
                 tx("SPEED: step 27 for every image", 22, BASE_C)
                 .next_to(ax3.c2p(x1, pbar), UP, buff=0.12).align_to(ax3.c2p(x1, 0), RIGHT))
        for idx, c, d in ((KOI, "#FF9100", UR), (AST, OURS, DL)):
            i, v, s, _ = ser[idx]
            pts = [ax3.c2p(a, b) for a, b in zip(i, v) if np.isfinite(b) and x0 <= a <= x1]
            if len(pts) > 1:
                self.play(Create(VMobject(stroke_color=c, stroke_width=4).set_points_as_corners(pts)), run_time=1.0)
            k0 = PM[idx]["first_grow"]
            pf = ax3.c2p(k0 - 1, v[list(i).index(k0 - 1)])
            th_ = img(os.path.join(IMGD, f"ours_{idx}.jpg"), 0.9)
            self.add(Dot(pf, radius=0.1, color=c))
            lab = tx(f"{SHORT[idx]}: grows at step {k0}", 24, c, bold=True)
            if d is UR:
                lab.next_to(pf, UR, buff=0.15)
            else:
                lab.move_to(ax3.c2p(33, yl + 0.35))
            th_.next_to(lab, RIGHT if d is UR else LEFT, buff=0.15)
            self.add(lab, th_)
        self.caption("Each image gets its own moment. SPEED uses one average for all.")
        self.step("free")
        self.add(tx("cost: a few FFTs per step", 24, GREY_B).move_to([4.6, 3.05, 0]))
        self.caption("It costs only a few FFTs per step, and no training.")
        self.wait(1.0)


# ================================================================== S07 Loop
class S07_Loop(Base):
    def construct(self):
        self.header("7 · Close the loop")
        self.step("loop")
        names = ["measure", "estimate", "test", "grow"]
        nodes = VGroup()
        for j, nm in enumerate(names):
            a = PI / 2 - j * PI / 2
            b = RoundedRectangle(width=1.9, height=0.7, corner_radius=0.12, stroke_color="#90A4AE", stroke_width=2,
                                 fill_color="#1E2A33", fill_opacity=1).move_to([-4.6 + 1.5 * np.cos(a), 0.4 + 1.3 * np.sin(a), 0])
            nodes.add(VGroup(b, tx(nm, 22).move_to(b)))
        arrows = VGroup(*[CurvedArrow(nodes[j].get_center() + 0.0, nodes[(j + 1) % 4].get_center(), angle=-PI / 3,
                                      stroke_width=2, color=GREY_B, tip_length=0.15).scale(0.55) for j in range(4)])
        self.add(nodes, arrows)
        sp = Spec(RUN, [2.6, 0.3, 0], 6.4, 4.4, c=1.0)
        self.add(sp.im, sp.axes(size=18), sp.stairs(front(RUN), WHITE, 3))
        k = ValueTracker(1)
        oc = ours_ceil(RUN)
        live = always_redraw(lambda: sp.stairs(oc, OURS, 5, upto=int(k.get_value())))
        self.add(live)
        self.caption("Repeat every step: the grid follows the frontier.")
        self.play(k.animate.set_value(N), run_time=4.0, rate_func=linear)
        live.clear_updaters()
        self.step("pair_loop")
        self.clear_all()
        for j, idx in enumerate((LO, HI)):
            s = Spec(idx, [-3.4 + 6.8 * j, -0.1, 0], 5.4, 3.6, c=1.0)
            t = img(os.path.join(IMGD, f"ours_{idx}.jpg"), 1.4).move_to([s.left + 0.75, s.bottom + s.h - 0.75, 0])
            self.add(s.im, s.axes(size=18, ylab=(j == 0)), s.stairs(front(idx), WHITE, 3), t,
                     s.stairs(S3_CEIL, BASE_C, 2).set_stroke(opacity=0.6),
                     tx(SHORT[idx], 26).next_to(s.im, UP, buff=0.2))
            st = s.stairs(ours_ceil(idx), OURS, 5)
            self.play(Create(st), run_time=1.4)
            self.add(s.fill(ours_ceil(idx), FULL_CEIL, WASTE, 0.35))
            m = PM[idx]
            self.add(tx(f"{m['t_ours']:.1f} s · " + fmt_x(m["speedup"]) + " vs full res", 24, SPD_C, bold=True)
                     .next_to(s.im, DOWN, buff=0.75))
        self.caption("Smooth images stay small longer. Detailed ones grow early.")
        self.step("gallery")
        self.clear_all()
        xs = np.linspace(-5.6, 5.6, len(GALLERY))
        for x, idx in zip(xs, GALLERY):
            s = Spec(idx, [x, 1.55, 0], 1.75, 1.35, c=1.0)
            t = img(os.path.join(IMGD, f"ours_{idx}.jpg"), 1.75).move_to([x, -0.45, 0])
            self.add(s.im, s.stairs(front(idx), WHITE, 1.5), s.stairs(ours_ceil(idx), OURS, 3), t,
                     tx(fmt_x(PM[idx]["speedup"]), 30, SPD_C, bold=True).move_to([x, -1.7, 0]),
                     tx(SHORT[idx], 18, GREY_B).move_to([x, -2.15, 0]))
        self.add(tx("smooth", 22, GREY_B).move_to([-5.6, 2.75, 0]), tx("detailed", 22, GREY_B).move_to([5.6, 2.75, 0]),
                 Arrow([-4.6, 2.75, 0], [4.6, 2.75, 0], color=GREY_B, stroke_width=2, buff=0))
        self.add(tx("wall time vs full resolution, same prompt and seed, 2048², B200", 20, DIM).move_to([0, -2.75, 0]))
        self.caption("Same model, same settings: each image gets its own schedule and speed.")
        self.wait(1.5)


# ================================================================== S08 Results
class S08_Results(Base):
    def construct(self):
        self.title_card("Part 3", "Results")
        self.header("8 · Results")
        self.step("spread")
        rows = META["sets"]["cinematic"]["rows"]
        sp = sorted(rows, key=lambda r: r["speedup"])
        lo_x, hi_x = 2.5, 8.0
        X = lambda v: -6.0 + 12.0 * (v - lo_x) / (hi_x - lo_x)            # noqa: E731
        base_y, th = -1.9, 0.3
        bins = {}
        g = Group()
        for r in sp:
            b = int((r["speedup"] - lo_x) // 0.5)
            k = bins.get(b, 0)
            bins[b] = k + 1
            x = X(lo_x + 0.5 * b + 0.25) + (k % 3 - 1) * (th + 0.02)
            y = base_y + th / 2 + (k // 3) * (th + 0.02)
            g.add(img(os.path.join(IMGD, f"thumb_{r['idx']}.jpg"), th).move_to([x, y, 0]))
        axl = Line([X(lo_x), base_y - 0.05, 0], [X(hi_x), base_y - 0.05, 0], color=GREY_B, stroke_width=2)
        tk = VGroup(*[tx(f"{v}×", 22, SPD_C).next_to([X(v), base_y - 0.05, 0], DOWN, buff=0.12) for v in (3, 4, 5, 6, 7, 8)])
        self.add(axl, tk, tx("speedup vs full resolution (wall time, 2048², B200)", 22, GREY_B).move_to([0, base_y - 0.95, 0]))
        self.play(LaggedStart(*[FadeIn(m) for m in g], lag_ratio=0.01), run_time=2.0)
        spd = [r["speedup"] for r in rows]
        med = float(np.median(spd))
        ml = DashedLine([X(med), base_y, 0], [X(med), 2.0, 0], color=WHITE, stroke_width=2)
        self.add(ml, tx(f"median {med:.1f}×", 24, WHITE).next_to(ml, UP, buff=0.1))
        self.add(tx(f"100 cinematic prompts: {min(spd):.1f}× to {max(spd):.1f}×", 30, bold=True).move_to([0, 2.95, 0]))
        self.caption(f"100 cinematic prompts: from {min(spd):.1f}× to {max(spd):.1f}× faster, median {med:.0f}×.")
        self.step("styles")
        self.clear_all()
        sets = [("lineart", 17, "line art"), ("anime", 0, "anime"), ("pexels", 0, "product photo")]
        sty = {s: {r["idx"]: r for r in META["sets"][s]["rows"]} for s, _, _ in sets}
        for j, (s, i, nm) in enumerate(sets):
            x = -4.4 + 4.4 * j
            a = img(os.path.join(IMGD, f"{s}_full_{i}.jpg"), 2.0).move_to([x - 1.05, 0.5, 0])
            b = img(os.path.join(IMGD, f"{s}_ours_{i}.jpg"), 2.0).move_to([x + 1.05, 0.5, 0])
            r = sty[s][i]
            self.add(a, b, tx(nm, 26, bold=True).move_to([x, 2.05, 0]),
                     tx("full res", 20, GREY_B).next_to(a, DOWN, buff=0.12), tx("ours", 20, OURS).next_to(b, DOWN, buff=0.12),
                     tx(fmt_x(r["speedup"]) + " faster", 30, SPD_C, bold=True).move_to([x, -1.35, 0]))
        allsp = [r["speedup"] for s, _, _ in sets for r in META["sets"][s]["rows"]]
        self.add(tx(f"60 prompts in three smooth styles: {min(allsp):.1f}× to {max(allsp):.1f}× (paper Fig. 4 method)", 22, DIM)
                 .move_to([0, -2.3, 0]))
        self.caption(f"Smooth styles gain the most: up to {max(allsp):.0f}× faster.")
        self.step("power")
        self.clear_all()
        dt = Z["dec_t"]
        ax = LAx([0.4, 0.35, 0], 8.0, 3.8, (0, 10), (0, 20))
        bars = VGroup(*[Rectangle(width=0.62, height=ax.c2p(0, t)[1] - ax.c2p(0, 0)[1], stroke_width=0, fill_color=OURS,
                                  fill_opacity=0.85).move_to(ax.c2p(k + 0.5, t / 2)) for k, t in enumerate(dt)])
        self.add(ax, tx("detail of the image (power at the first grid's edge), deciles", 22, GREY_B).next_to(ax, DOWN, buff=0.6),
                 tx("time per image (s)", 22, GREY_B).next_to(ax, UP, buff=0.15).align_to(ax, LEFT),
                 VGroup(*[tx(str(v), 20, GREY_B).next_to(ax.c2p(0, v), LEFT, buff=0.1) for v in (0, 10, 20)]))
        gs = [grow_bar(b, DOWN, 0.5) for b in bars]
        self.play(LaggedStart(*[g[0] for g in gs], lag_ratio=0.08), run_time=1.4)
        self.remove(*[g[1] for g in gs])
        self.add(bars)
        self.add(tx("6.45×", 30, SPD_C, bold=True).next_to(bars[0], UP, buff=0.15),
                 tx("3.95×", 30, SPD_C, bold=True).next_to(bars[-1], UP, buff=0.15),
                 tx("smoothest 10%", 20, GREY_B).next_to(bars[0], DOWN, buff=0.1),
                 tx("most detailed 10%", 20, GREY_B).next_to(bars[-1], DOWN, buff=0.1))
        self.add(tx("20,000 prompts · 2048² · B200 · speedup vs full res 70.8 s", 20, DIM).move_to([0, -2.75, 0]))
        self.caption("On 20,000 prompts, time follows each image's detail.")
        self.step("bars")
        self.clear_all()
        data = [("full res, 50 steps", 70.5, FULL_C), ("SPEED-2", 28.98, BASE_C), ("SPEED-3", 24.9, BASE_C),
                ("ours", 15.1, OURS)]       # CONTEXT 4.2: 2K, 1,000 random 4KLSDB prompts, B200 (SPEED-2 on 200)
        x0, sc = -3.2, 8.4 / 70.5
        for j, (nm, t, c) in enumerate(data):
            y = 1.7 - 1.05 * j
            self.add(tx(nm, 26, c, bold=(nm == "ours")).move_to([x0 - 0.25, y, 0]).align_to([x0 - 0.25, 0, 0], RIGHT))
            bar = Rectangle(width=t * sc, height=0.55, stroke_width=0, fill_color=c, fill_opacity=0.9)
            bar.move_to([x0 + t * sc / 2, y, 0])
            an, st_ = grow_bar(bar, LEFT, 0.4)
            self.play(an)
            self.remove(st_)
            self.add(bar)
            self.add(tx(f"{t:.1f} s", 24, AMB).next_to(bar, RIGHT, buff=0.15))
        self.add(tx("4.68× vs full res", 32, SPD_C, bold=True).move_to([2.6, -2.0, 0]),
                 tx("1.65× vs SPEED-3", 32, SPD_C, bold=True).move_to([-2.6, -2.0, 0]),
                 tx("HPSv3 −0.18 vs SPEED-3 · 1,000 random prompts · 2048² · B200 · not yet in the paper", 20, DIM)
                 .move_to([0, -2.65, 0]))
        self.caption("On average: 4.7× faster than full resolution, 1.65× faster than SPEED-3.")
        self.step("images")
        self.clear_all()
        i = 40
        t = META["fig3"][str(i)]
        cols = [("fullres", "full res"), ("speed2", "SPEED-2"), ("cont_mode", "ours")]
        for j, (arm, nm) in enumerate(cols):
            x = -4.3 + 4.3 * j
            m = img(os.path.join(IMGD, f"fig3_{arm}_{i}.jpg"), 3.9).move_to([x, 0.15, 0])
            sp_ = t["fullres"] / t[arm]
            self.add(m, tx(nm, 26, OURS if arm == "cont_mode" else WHITE, bold=arm == "cont_mode").next_to(m, UP, buff=0.15),
                     tx(f"{t[arm]:.0f} s · " + (fmt_x(sp_) if arm != "fullres" else "1×"), 26,
                        SPD_C if arm != "fullres" else GREY_B, bold=True).next_to(m, DOWN, buff=0.15))
        self.add(tx("same prompt and noise · 2048² · paper Fig. 3", 20, DIM).move_to([0, -2.75, 0]))
        self.caption("Same prompt and noise, a fraction of the time.")
        self.wait(1.5)


# ================================================================== S09 Outlook
class S09_Outlook(Base):
    def construct(self):
        self.header("9 · Takeaways")
        self.step("three")
        self.add(tx("Spectral Frontier Sampling in three ideas", 38, bold=True).move_to([0, 2.4, 0]))
        pts = ["1.  Trace the frontier: grow the grid just above the frequencies that left the noise",
               "2.  Every image has its own frontier: read it online from the model's clean guess",
               "3.  Training-free: no extra network calls, a few FFTs per step"]
        for k, p in enumerate(pts):
            self.add(tx(p, 25).move_to([0, 1.25 - 0.8 * k, 0]).align_to([-6.2, 0, 0], LEFT))
            self.wait(1.9)
        self.add(tx("2.9–7.7× faster cinematic images · up to 10× on smooth styles · 1.65× vs SPEED-3", 26, FRONT)
                 .move_to([0, -1.45, 0]))
        self.step("end")
        self.clear_all()
        if self.hdr is not None:
            self.remove(self.hdr)
        self.add(tx("Spectral Frontier Sampling", 56, bold=True).move_to([0, 1.0, 0]),
                 tx("Pay only for frequencies that have left the noise", 30, FRONT).move_to([0, 0.1, 0]),
                 tx("CVPR 2027 submission", 24, DIM).move_to([0, -0.7, 0]))
        self.wait(4.0)
